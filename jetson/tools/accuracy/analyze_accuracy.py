#!/usr/bin/env python3
"""Accuracy study analysis (runs from jetson/scripts/run_accuracy.sh). Python env: /home/nvidia/data/envs/vlm-eval.

    python analyze_accuracy.py <sintel|msvd> [out.json]

Inputs, under /home/nvidia/data/accuracy:
  runs/<ds>/<config>/<clip>.json      pipeline result (selected frames, timing, generated text)
  runs/<ds>/<config>/<clip>/          --dump-dir: manifest.jsonl, img_NNN.f32 (what the LLM received), logits0.f32
  dense/<ds>/<clip>/{native,aligned}/ every frame encoded whole (jetson-embed-frames): the clip's full visual content

Metrics (one row per clip and config; means and paired tests in the summary):
  ViT level -- the visual tokens handed to the LLM
    spec_H     spectral (von Neumann) entropy of the token set, bits: Shannon entropy of the normalised squared
               singular values of the L2-normalised tokens; 2**spec_H = effective rank (independent directions)
    cb_H       Shannon entropy (bits) of the tokens' histogram over a k-means codebook of the clip's dense tokens
    cb_cover   share of the clip's dense token mass whose codebook cells the run's tokens hit
    cb_jsd     Jensen-Shannon divergence (bits, 0..1) between the run's and the whole clip's codebook histograms
    t_cover    mean over all frames of the clip of the best cosine (mean-pooled frame embedding) to a frame the
               run selected: how well every moment of the clip is represented
  Window cache -- per image vs the same frame encoded whole (dense aligned)
    cos_reused / cos_crop / cos_full   mean token cosine for windows reused from the cache, re-encoded in a crop,
    cos_pw                             ... re-encoded by the per-window encode (--wc-per-window),
                                       or encoded with the whole frame; frac_reused share of tokens reused
  LLM input / output vs the baseline config of the same clip
    n_img_tok, n_txt_tok   sequence composition
    chamfer    symmetric mean best-match cosine between the run's and the baseline's image tokens
    kl0, jsd0  KL(base || run) and JSD of the first-token distributions; top1 agreement, top5 overlap
    cap_*      generated text vs the baseline's: BLEU-4, ROUGE-L, sentence-embedding cosine
  MSVD -- generated caption vs the ~40 ground-truth captions per clip
    BLEU-4, METEOR, ROUGE-L, CIDEr (pycocoevalcap, corpus level), sbert_max / sbert_mean (all-mpnet-base-v2 cosine),
    content-word precision (share of the caption's content words found in any reference: hallucination proxy)
"""
import collections, glob, json, math, os, re, sys
import numpy as np

ACC = "/home/nvidia/data/accuracy"
CONFIGS = ["base", "sel", "align", "wc", "arch", "ours", "ours_nocache", "ours_pw", "uni_arch"]
K = 64
STOP = set("a an the is are was were be been being of in on at to for with and or but by from as it its this that "
           "these those there their they them he she his her him we you i our your into onto over under up down "
           "out off while who which what when where how some any all each other another one two three video "
           "image images frame frames scene shows showing seen appears appear can also very".split())


# ---------------------------------------------------------------- loading
def load_run(ds, cfg, clip):
    j = f"{ACC}/runs/{ds}/{cfg}/{clip}.json"
    d = f"{ACC}/runs/{ds}/{cfg}/{clip}"
    if not os.path.exists(j) or not os.path.exists(f"{d}/manifest.jsonl"):
        return None
    try:   # a run in progress or one that failed
        res = json.load(open(j))
        segs = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
        if "inference" not in res:
            return None
    except (ValueError, OSError):
        return None
    imgs = []
    for s in segs:
        if s["type"] == "image":
            x = np.fromfile(f"{d}/{s['file']}", dtype=np.float32).reshape(s["nx"] * s["ny"], s["n_embd"])
            imgs.append(dict(s, x=x))
    gen = next((s["tokens"] for s in segs if s["type"] == "generated"), [])
    txt = sum(len(s["tokens"]) for s in segs if s["type"] == "text")
    lg = np.fromfile(f"{d}/logits0.f32", dtype=np.float32) if os.path.exists(f"{d}/logits0.f32") else None
    frames = [f["index"] for f in res.get("selected_frames", [])]
    return dict(res=res, imgs=imgs, gen=gen, n_txt=txt, logits=lg, frames=frames,
                text=res["inference"]["generated_text"].strip())


_dense_cache = {}
def load_dense(ds, clip, mode):
    key = (ds, clip, mode)
    if key not in _dense_cache:
        d = f"{ACC}/dense/{ds}/{clip}/{mode}"
        if not os.path.exists(f"{d}/meta.json"):
            return None
        meta = json.load(open(f"{d}/meta.json"))
        fr = {}
        for m in meta["frames"]:
            idx = int(m["frame"].split("_")[-1])
            fr[idx] = dict(nx=m["nx"], ny=m["ny"],
                           x=np.fromfile(f"{d}/{m['frame']}.f32", dtype=np.float32).reshape(m["nx"] * m["ny"], -1))
        if len(_dense_cache) > 6:
            _dense_cache.clear()
        _dense_cache[key] = fr
    return _dense_cache[key]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- ViT-level metrics
def spectral_entropy(x):
    u = unit(x.astype(np.float64))
    g = u.T @ u if u.shape[0] > u.shape[1] else u @ u.T
    ev = np.clip(np.linalg.eigvalsh(g), 0, None)
    p = ev / ev.sum()
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def kmeans(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    u = unit(x)
    c = u[rng.choice(len(u), size=min(k, len(u)), replace=False)]
    for _ in range(iters):
        a = np.argmax(u @ c.T, axis=1)
        for j in range(len(c)):
            m = u[a == j]
            if len(m):
                c[j] = unit(m.mean(0))
    return c


def hist(x, cb):
    a = np.argmax(unit(x) @ cb.T, axis=1)
    h = np.bincount(a, minlength=len(cb)).astype(np.float64)
    return h / h.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def jsd(p, q):
    m = (p + q) / 2
    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log2(a[s] / b[s])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------- text metrics
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bleu4(hyp, refs):
    from collections import Counter
    h = tok(hyp)
    if not h:
        return 0.0
    ps = []
    for n in range(1, 5):
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        mx = Counter()
        for r in refs:
            rt = tok(r)
            for g, c in Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1)).items():
                mx[g] = max(mx[g], c)
        num = sum(min(c, mx[g]) for g, c in hc.items())
        ps.append((num + 1) / (max(sum(hc.values()), 0) + 1))   # add-1 smoothing (sentence level)
    rl = min((len(tok(r)) for r in refs), key=lambda L: (abs(L - len(h)), L))
    bp = 1.0 if len(h) > rl else math.exp(1 - rl / max(len(h), 1))
    return bp * math.exp(sum(math.log(p) for p in ps) / 4)


def rouge_l(hyp, refs):
    def lcs(a, b):
        dp = [0] * (len(b) + 1)
        for x in a:
            prev = 0
            for j, y in enumerate(b):
                cur = dp[j + 1]
                dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
                prev = cur
        return dp[-1]
    h = tok(hyp)
    best = 0.0
    for r in refs:
        rt = tok(r)
        l = lcs(h, rt)
        if l:
            p, rc = l / len(h), l / len(rt)
            best = max(best, (1 + 1.2 ** 2) * p * rc / (rc + 1.2 ** 2 * p))
    return best


def content_precision(hyp, refs):
    vocab = set(w for r in refs for w in tok(r))
    words = [w for w in tok(hyp) if w not in STOP and not w.isdigit()]
    if not words:
        return float("nan")
    stem = lambda w: w[:-1] if w.endswith("s") and len(w) > 3 else w   # crude plural folding
    vs = set(stem(w) for w in vocab)
    return sum(stem(w) in vs for w in words) / len(words)


_sbert = None
def sbert(texts):
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    return _sbert.encode(texts, normalize_embeddings=True, batch_size=64)


def softmax(z):
    z = z.astype(np.float64) - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------- per clip
def analyze_clip(ds, clip, cfgs):
    runs = {c: load_run(ds, c, clip) for c in cfgs}
    runs = {c: r for c, r in runs.items() if r and r["imgs"]}
    if "base" not in runs:
        return []
    dn, da = load_dense(ds, clip, "native"), load_dense(ds, clip, "aligned")
    if not dn or not da:
        return []
    allx = np.concatenate([f["x"] for f in dn.values()] + [f["x"] for f in da.values()])
    cb = kmeans(allx[np.random.default_rng(0).choice(len(allx), size=min(len(allx), 20000), replace=False)], K)
    p_dense = (hist(np.concatenate([f["x"] for f in dn.values()]), cb) +
               hist(np.concatenate([f["x"] for f in da.values()]), cb)) / 2
    base = runs["base"]
    bx = np.concatenate([i["x"] for i in base["imgs"]])
    bpr = softmax(base["logits"]) if base["logits"] is not None else None
    rows = []
    for c, r in runs.items():
        x = np.concatenate([i["x"] for i in r["imgs"]])
        aligned = r["imgs"][0]["nx"] == da[1]["nx"] and r["imgs"][0]["ny"] == da[1]["ny"]
        dref = da if aligned else dn
        h = hist(x, cb)
        row = dict(clip=clip, cfg=c, n_frames=len(r["imgs"]), n_img_tok=len(x), n_txt_tok=r["n_txt"],
                   ttft=r["res"]["timing"]["ttft_ms"], spec_H=spectral_entropy(x), cb_H=entropy(h),
                   cb_cover=float(p_dense[h > 0].sum()), cb_jsd=jsd(h, p_dense), text=r["text"])
        # temporal coverage: every dense frame vs the frames the run selected (same preprocessing)
        sel = [f for f in r["frames"] if f in dref]
        if sel:
            m = {f: unit(dref[f]["x"].mean(0)) for f in dref}
            S = np.stack([m[f] for f in sel])
            row["t_cover"] = float(np.mean([np.max(S @ m[f]) for f in dref]))
        # window cache: each image vs the same frame encoded whole (aligned)
        if aligned and len(r["frames"]) == len(r["imgs"]):
            acc = collections.defaultdict(list)
            n_re, n_tot = 0, 0
            for img, f in zip(r["imgs"], r["frames"]):
                if f not in da:
                    continue
                cs = np.sum(unit(img["x"]) * unit(da[f]["x"]), axis=1)
                nwx, nwy = img["nx"] // 4, img["ny"] // 4
                rx, ry, rw, rh = img["rect"]
                mask = np.zeros((img["ny"], img["nx"]), bool)   # True = re-encoded this frame
                if img["action"] in ("key", "full"):
                    mask[:] = True
                    acc["cos_full"].extend(cs.tolist())
                elif img["action"] == "pw":   # per-window encode: the listed windows re-encoded
                    for w in img["windows"]:
                        wy, wx = divmod(w, nwx)
                        mask[wy * 4:(wy + 1) * 4, wx * 4:(wx + 1) * 4] = True
                    acc["cos_pw"].extend(cs[mask.ravel()].tolist())
                    acc["cos_reused"].extend(cs[~mask.ravel()].tolist())
                elif img["action"] == "crop":
                    mask[ry * 4:(ry + rh) * 4, rx * 4:(rx + rw) * 4] = True
                    acc["cos_crop"].extend(cs[mask.ravel()].tolist())
                    acc["cos_reused"].extend(cs[~mask.ravel()].tolist())
                else:   # skip: the whole frame is the cache
                    acc["cos_reused"].extend(cs.tolist())
                n_re += int(mask.sum()); n_tot += mask.size
            for k, v in acc.items():
                row[k] = float(np.mean(v))
            row["frac_reused"] = 1 - n_re / max(n_tot, 1)
        # vs the baseline
        if c != "base":
            ux, ub = unit(x), unit(bx)
            sim = ux @ ub.T
            row["chamfer"] = float((sim.max(1).mean() + sim.max(0).mean()) / 2)
            row["cb_jsd_base"] = jsd(h, hist(bx, cb))
            if bpr is not None and r["logits"] is not None:
                q = softmax(r["logits"])
                s = bpr > 1e-12
                row["kl0"] = float((bpr[s] * np.log(bpr[s] / np.maximum(q[s], 1e-30))).sum())
                row["jsd0"] = jsd(bpr, q)
                row["top1"] = int(np.argmax(bpr) == np.argmax(q))
                row["top5"] = len(set(np.argsort(-bpr)[:5]) & set(np.argsort(-q)[:5])) / 5
            row["cap_bleu_base"] = bleu4(r["text"], [base["text"]])
            row["cap_rouge_base"] = rouge_l(r["text"], [base["text"]])
            row["cap_exact_base"] = int(r["text"] == base["text"])
        rows.append(row)
    return rows


def coco_scores(preds, refs):
    """corpus-level BLEU-4, METEOR, ROUGE-L, CIDEr with pycocoevalcap (refs: clip -> list of captions)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    norm = lambda s: " ".join(tok(s))
    gts = {k: [norm(c) for c in refs[k]] for k in preds}
    res = {k: [norm(preds[k]) or "empty"] for k in preds}
    out = {}
    out["BLEU-4"] = Bleu(4).compute_score(gts, res, verbose=0)[0][3]
    out["METEOR"] = Meteor().compute_score(gts, res)[0]
    out["ROUGE-L"] = Rouge().compute_score(gts, res)[0]
    cider, per = Cider().compute_score(gts, res)
    out["CIDEr"] = cider
    return out, dict(zip(preds.keys(), per))


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        d = np.array(a) - np.array(b)
        return float(w(d[d != 0]).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        return float("nan")


def main():
    ds = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"{ACC}/analysis_{ds}.json"
    cfgs = [c for c in CONFIGS if os.path.isdir(f"{ACC}/runs/{ds}/{c}")]
    cfgs += sorted(os.path.basename(p) for p in glob.glob(f"{ACC}/runs/{ds}/*") if os.path.basename(p) not in cfgs)
    clips = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ACC}/runs/{ds}/base/*.json"))
    rows = []
    for i, clip in enumerate(clips):
        rows += analyze_clip(ds, clip, cfgs)
        print(f"[{i + 1}/{len(clips)}] {clip}", file=sys.stderr, flush=True)
    # captions: sentence-embedding similarity to the baseline caption (and to the references for MSVD)
    texts = sorted(set(r["text"] for r in rows))
    emb = dict(zip(texts, sbert(texts)))
    btext = {r["clip"]: r["text"] for r in rows if r["cfg"] == "base"}
    for r in rows:
        if r["cfg"] != "base":
            r["cap_sbert_base"] = float(emb[r["text"]] @ emb[btext[r["clip"]]])
    summary = {}
    if ds == "msvd":
        refs = json.load(open(f"{ACC}/msvd/captions.json"))
        ref_emb = {c: sbert(refs[c]) for c in btext}
        for r in rows:
            s = ref_emb[r["clip"]] @ emb[r["text"]]
            r["sbert_max"], r["sbert_mean"] = float(s.max()), float(s.mean())
            r["content_prec"] = content_precision(r["text"], refs[r["clip"]])
            r["bleu4_s"] = bleu4(r["text"], refs[r["clip"]])
        for c in cfgs:
            preds = {r["clip"]: r["text"] for r in rows if r["cfg"] == c}
            common = [k for k in preds if k in btext]
            if len(common) < 2:
                continue
            sc, per = coco_scores({k: preds[k] for k in common}, refs)
            for r in rows:
                if r["cfg"] == c and r["clip"] in per:
                    r["cider_s"] = float(per[r["clip"]])
            summary[c] = dict(sc, n=len(common))
    # means per config + paired tests vs base
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("clip",)]
    keys = sorted(set(k for r in rows for k, v in r.items() if isinstance(v, (int, float))))
    for c in cfgs:
        rs = [r for r in rows if r["cfg"] == c]
        if not rs:
            continue
        s = summary.setdefault(c, {})
        s["n"] = len(rs)
        for k in keys:
            v = [r[k] for r in rs if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
            if v:
                s[k] = float(np.mean(v))
        if c != "base":
            for k in ("sbert_max", "cider_s", "spec_H", "cb_cover", "t_cover"):
                a = {r["clip"]: r[k] for r in rs if k in r}
                b = {r["clip"]: r[k] for r in rows if r["cfg"] == "base" and k in r}
                common = [x for x in a if x in b]
                if len(common) > 5:
                    s[f"p_{k}"] = wilcoxon([a[x] for x in common], [b[x] for x in common])
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    show = ["n", "n_frames", "n_img_tok", "ttft", "spec_H", "cb_H", "cb_cover", "cb_jsd", "t_cover", "frac_reused",
            "cos_full", "cos_crop", "cos_pw", "cos_reused", "chamfer", "kl0", "top1", "top5", "cap_sbert_base", "cap_bleu_base",
            "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "sbert_max", "sbert_mean", "content_prec", "p_sbert_max", "p_cider_s"]
    print("| metric | " + " | ".join(c for c in cfgs if c in summary) + " |")
    print("|---|" + "---|" * len([c for c in cfgs if c in summary]))
    for k in show:
        if any(k in summary[c] for c in summary):
            print(f"| {k} | " + " | ".join(
                (f"{summary[c][k]:.4g}" if k in summary[c] else "") for c in cfgs if c in summary) + " |")


if __name__ == "__main__":
    main()
