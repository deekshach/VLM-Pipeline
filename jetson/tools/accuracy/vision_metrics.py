#!/usr/bin/env python3
"""Vision-side metrics: what the visual tokens handed to the LLM carry, without the LLM's answer.
Python env: /home/nvidia/data/envs/vlm-eval.

    python vision_metrics.py [datasets...]        CLIP=0: skip the grounding metrics (they load CLIP ViT-L/14)

Configs (runs_eval, else the earlier runs): base (8 uniform frames), qwen2fps (Qwen2.5-VL's 2 fps video input,
scripts/run_eval.sh e8), ours_nocache (motion-selected frames), pw24 (+ per-window cache), pwd24 (+ drop-clean).
The runs need their --dump-dir; the reference is the clip's dense ViT encode (every pool frame encoded whole:
run_accuracy.sh for Sintel / MSVD, scripts/embed_dense_eval.sh for the others), native for the uniform runs and
Qwen 2 fps, window-aligned for the motion-selected ones (the matching geometry, as analyze_accuracy.py).

Per clip and config:
  diversity   tokens      visual tokens the LLM receives (drop-clean: the key frame + the re-encoded windows)
              spec_H      spectral entropy (bits) of those tokens; 2**spec_H = effective rank. Grows with the token
                          count by construction (at most log2(min(tokens, 3584))), so read it next to `tokens`
              redund      mean cosine between the frames' mean tokens (Qwen 2 fps: per frame pair); 1 = duplicates
  coverage    t_cover     mean over the clip's frames of the best cosine to a frame the run sent (mean tokens)
              tok_recall  mean over the clip's tokens of the best cosine to a token the run sent
              tok_prec    mean over the run's tokens of the best cosine to a token of the clip (faithfulness)
              sub_energy  share of the clip's token energy inside the run's top-64 principal directions: the span
                          the run covers, at the same dimension for every config
              cb_cover    share of the clip's token mass in k-means cells (64, on the clip) the run hits
              cb_jsd      Jensen-Shannon divergence (bits) between the run's and the clip's cell histograms
  cache       cos_key / cos_reenc / cos_reused   token cosine to the same frame encoded whole (aligned): frames
                          encoded whole, windows re-encoded, windows reused from the cache (not for drop-clean,
                          whose reused windows never reach the LLM); frac_sent = tokens sent / tokens of the frames
  grounding   CLIP ViT-L/14 (openai) on the frames the run sent, as images (drop-clean sends the same frames)
              clip_ground captions: mean over the reference captions of the best frame-caption cosine;
                          multiple choice: best frame cosine to "<question> <correct option>"
              clip_mc     multiple choice: CLIP picks the option whose text best matches a frame (1 = correct)
              clip_r1     captions: the clip's mean frame embedding retrieves its own captions first among the
                          dataset's clips (per-dataset share in the summary)
Writes <ACC>/vision_metrics.csv (per clip) and vision_metrics.md (means per dataset and config over the clips
every config has). Token sets are subsampled to MAX_TOK for the nearest-neighbour and codebook metrics.
"""
import csv, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_accuracy as aa

ACC = aa.ACC
SOURCES = [f"{ACC}/runs_eval", f"{ACC}/runs"]
EVAL = "/mnt/external_ssd/VLM-datasets/eval"
QRES = "/mnt/external_ssd/VLM-datasets/qres"   # prepare_qres.py: every frame at Qwen's per-frame resolution
QFRAMES = f"{QRES}/qwen2fps"
DATASETS = ["sintel", "msvd", "charades", "tacos", "nextqa", "videomme", "perception"]
CFGS = ["base", "qwen2fps", "ours_nocache", "pw24", "pwd24"]
WIN = 4          # tokens per window side (the ViT's 112 px windows)
K_SUB = 64       # principal directions for sub_energy
K_CB = 64        # codebook cells
MAX_TOK = 8192
RNG = np.random.default_rng(0)
unit = aa.unit


def clips(ds):
    if ds == "msvd":
        return open(f"{ACC}/msvd_clips_45.txt").read().strip().split(",")
    return sorted(os.listdir(f"{pool_root(ds)}/training/clean"))


def pool_root(ds):
    return f"{QRES}/pool/{ds}"


def sub(x, n=MAX_TOK):
    return x if len(x) <= n else x[np.sort(RNG.choice(len(x), n, replace=False))]


# ---------------------------------------------------------------- loading
def reencoded(s):
    """True where the frame's tokens were encoded this frame (vs reused from the window cache)"""
    nx, ny = s["nx"], s["ny"]
    m = np.zeros((ny, nx), bool)
    a = s.get("action", "full")
    if a in ("key", "full"):
        m[:] = True
    elif a == "pw":
        for w in s.get("windows", []):
            wy, wx = divmod(w, nx // WIN)
            m[wy * WIN:(wy + 1) * WIN, wx * WIN:(wx + 1) * WIN] = True
    elif a == "crop":
        rx, ry, rw, rh = s["rect"]
        m[ry * WIN:(ry + rh) * WIN, rx * WIN:(rx + rw) * WIN] = True
    return m.ravel()


def load(ds, cfg, clip):
    """text, frame paths and, with a dump, one group per image (Qwen 2 fps: per frame pair): x = the image's tokens,
    sent = what the LLM received of them, frame = pool index, action / mask (window cache). None if not run."""
    for src in SOURCES:
        j, d = f"{src}/{ds}/{cfg}/{clip}.json", f"{src}/{ds}/{cfg}/{clip}"
        if not os.path.exists(j):
            continue
        res = json.load(open(j))
        if "requests" in res:   # llama-temporal-pair-vlm
            if not res["requests"]:
                return None
            r = res["requests"][0]
            n = r["grid"][0] * r["grid"][1]
            e = f"{d}/{clip}.video.embd.bin"
            x = np.fromfile(e, np.float32).reshape(-1, 3584) if os.path.exists(e) else None
            groups = None if x is None else [dict(x=x[i:i + n], sent=x[i:i + n], frame=None, action="full",
                                                  nx=r["grid"][0], ny=r["grid"][1]) for i in range(0, len(x), n)]
            return dict(text=r["text"].strip(), groups=groups,
                        paths=sorted(glob.glob(f"{QFRAMES}/{ds}/{clip}/frame_*.png")))
        drop = res.get("switches", {}).get("wc_drop_clean", False)
        frames = [f["index"] for f in res.get("selected_frames", [])]
        groups = None
        if os.path.exists(f"{d}/manifest.jsonl"):
            imgs = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
            imgs = [s for s in imgs if s["type"] == "image"]
            groups = []
            for k, s in enumerate(imgs):
                x = np.fromfile(f"{d}/{s['file']}", np.float32).reshape(s["nx"] * s["ny"], s["n_embd"])
                m = reencoded(s)
                groups.append(dict(x=x, sent=x[m] if drop and s.get("action") in ("pw", "skip") else x,
                                   frame=frames[k] if k < len(frames) else None, action=s.get("action", "full"),
                                   mask=m, nx=s["nx"], ny=s["ny"]))
        return dict(text=res.get("inference", {}).get("generated_text", "").strip(), groups=groups, drop=drop,
                    paths=[f"{pool_root(ds)}/training/clean/{clip}/frame_{i:04d}.png" for i in frames])
    return None


# ---------------------------------------------------------------- token metrics
def eig_unit(x):
    """eigen-decomposition of the unit tokens' Gram: (eigenvalues ascending, top-K_SUB principal directions as
    columns, 3584 x k)"""
    u = unit(x)
    if len(u) > u.shape[1]:
        w, v = np.linalg.eigh((u.T @ u).astype(np.float64))
        return w, v[:, -K_SUB:]
    w, v = np.linalg.eigh((u @ u.T).astype(np.float64))
    k = min(K_SUB, len(w))
    top = u.T.astype(np.float64) @ v[:, -k:]
    return w, top / (np.linalg.norm(top, axis=0, keepdims=True) + 1e-12)


def spec_entropy(w):
    p = np.clip(w, 0, None)
    p = p / p.sum()
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def best_cos(a, b, chunk=2048):
    """for each row of a, the best cosine to a row of b (both unit)"""
    return np.concatenate([(a[i:i + chunk] @ b.T).max(1) for i in range(0, len(a), chunk)])


def token_metrics(run, ref, cb, p_ref):
    x = np.concatenate([g["sent"] for g in run["groups"]])
    w, top = eig_unit(x)
    out = dict(tokens=len(x), spec_H=spec_entropy(w))
    means = unit(np.stack([g["x"].mean(0) for g in run["groups"]]))   # per frame (pair), whole image
    if len(means) > 1:
        s = means @ means.T
        out["redund"] = float(s[~np.eye(len(s), dtype=bool)].mean())
    rm = unit(np.stack([f["x"].mean(0) for f in ref.values()]))
    out["t_cover"] = float((rm @ means.T).max(1).mean())
    R = unit(sub(np.concatenate([f["x"] for f in ref.values()])))
    X = unit(sub(x))
    out["tok_recall"] = float(best_cos(R, X).mean())
    out["tok_prec"] = float(best_cos(X, R).mean())
    out["sub_energy"] = float(((R.astype(np.float64) @ top) ** 2).sum() / len(R))
    h = aa.hist(x, cb)
    out["cb_cover"], out["cb_jsd"] = float(p_ref[h > 0].sum()), aa.jsd(h, p_ref)
    return out


def cache_metrics(run, da):
    acc = {"cos_key": [], "cos_reenc": [], "cos_reused": []}
    n_sent = n_all = 0
    for g in run["groups"]:
        f = da.get(g["frame"])
        if f is None or (f["nx"], f["ny"]) != (g["nx"], g["ny"]):
            return {}
        cs = np.sum(unit(g["x"]) * unit(f["x"]), axis=1)
        if g["action"] in ("key", "full"):
            acc["cos_key"] += cs.tolist()
        else:
            acc["cos_reenc"] += cs[g["mask"]].tolist()
            if not run["drop"]:
                acc["cos_reused"] += cs[~g["mask"]].tolist()
        n_sent += len(g["sent"]); n_all += len(g["x"])
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["frac_sent"] = n_sent / n_all
    return out


# ---------------------------------------------------------------- CLIP grounding
_clip = None
_img_cache = {}


def clip_model():
    global _clip
    if _clip is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-large-patch14"
        dev = os.environ.get("CLIP_DEVICE", "cpu")   # the GPU is usually busy with the pipeline
        _clip = (CLIPModel.from_pretrained(name).eval().to(dev), CLIPProcessor.from_pretrained(name), dev, torch)
    return _clip


def clip_images(paths):
    from PIL import Image
    m, proc, dev, torch = clip_model()
    todo = [p for p in paths if p not in _img_cache]
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        with torch.no_grad():
            e = m.get_image_features(**proc(images=[Image.open(p).convert("RGB") for p in batch],
                                             return_tensors="pt").to(dev))
        for p, v in zip(batch, unit(e.float().cpu().numpy())):
            _img_cache[p] = v
    return np.stack([_img_cache[p] for p in paths])


def clip_texts(texts):
    m, proc, dev, torch = clip_model()
    with torch.no_grad():
        e = m.get_text_features(**proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(dev))
    return unit(e.float().cpu().numpy())


def mc_options(prompt):
    """(question, {letter: option text}) from an items.json multiple-choice prompt"""
    q = prompt.split("\nOptions:")[0].strip()
    return q, dict(re.findall(r"^([A-E])\. (.+)$", prompt, re.M))


# ---------------------------------------------------------------- main
def main():
    dss = sys.argv[1:] or DATASETS
    use_clip = os.environ.get("CLIP", "1") != "0"
    msvd_refs = json.load(open(f"{ACC}/msvd/captions.json"))
    rows = []
    for ds in dss:
        items = json.load(open(f"{EVAL}/{ds}/items.json")) if os.path.exists(f"{EVAL}/{ds}/items.json") else {}
        mc = bool(items) and "answer" in next(iter(items.values()))
        vids = {}   # (cfg, clip) -> mean frame embedding, for retrieval
        caps = {}
        for v in clips(ds):
            dn, da = aa.load_dense(ds, v, "native"), aa.load_dense(ds, v, "aligned")
            cb = p_ref = None
            if dn and da:
                allx = np.concatenate([f["x"] for f in dn.values()] + [f["x"] for f in da.values()])
                cb = aa.kmeans(sub(allx, 20000), K_CB)
            for c in CFGS:
                run = load(ds, c, v)
                if run is None:
                    continue
                row = dict(dataset=ds, clip=v, cfg=c, frames=len(run["paths"]))
                if run["groups"] and cb is not None:
                    f0 = next(iter(da.values()))
                    aligned = (run["groups"][0]["nx"], run["groups"][0]["ny"]) == (f0["nx"], f0["ny"])
                    ref = da if aligned else dn
                    p_ref = aa.hist(np.concatenate([f["x"] for f in ref.values()]), cb)
                    row.update(token_metrics(run, ref, cb, p_ref))
                    if c in ("pw24", "pwd24") and aligned:
                        row.update(cache_metrics(run, da))
                if use_clip and run["paths"] and ds != "sintel":
                    F = clip_images(run["paths"])
                    if mc:
                        q, opts = mc_options(items[v]["prompt"])
                        L = sorted(opts)
                        T = clip_texts([f"{q} {opts[k]}" for k in L])
                        s = (F @ T.T).max(0)
                        row["clip_mc"] = float(L[int(np.argmax(s))] == items[v]["answer"])
                        row["clip_ground"] = float(s[L.index(items[v]["answer"])])
                    else:
                        refs = msvd_refs[v] if ds == "msvd" else items[v]["refs"]
                        if v not in caps:
                            caps[v] = clip_texts(refs)
                        row["clip_ground"] = float((F @ caps[v].T).max(0).mean())
                        vids[c, v] = unit(F.mean(0))
                rows.append(row)
                print(ds, v, c, {k: round(x, 3) for k, x in row.items() if isinstance(x, float)}, flush=True)
        # retrieval: each clip's mean frame embedding against every clip's mean caption embedding
        if caps:
            cv = sorted(caps)
            C = unit(np.stack([caps[v].mean(0) for v in cv]))
            for row in rows:
                if row["dataset"] == ds and (row["cfg"], row["clip"]) in vids:
                    s = C @ vids[row["cfg"], row["clip"]]
                    row["clip_r1"] = float(cv[int(np.argmax(s))] == row["clip"])

    keys = ["frames", "tokens", "spec_H", "redund", "t_cover", "tok_recall", "tok_prec", "sub_energy", "cb_cover",
            "cb_jsd", "cos_key", "cos_reenc", "cos_reused", "frac_sent", "clip_ground", "clip_mc", "clip_r1"]
    with open(f"{ACC}/vision_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "clip", "cfg"] + keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = []
    for ds in dss:
        rs = [r for r in rows if r["dataset"] == ds]
        have = [c for c in CFGS if any(r["cfg"] == c and "tokens" in r for r in rs)]
        common = {r["clip"] for r in rs} if not have else \
            set.intersection(*({r["clip"] for r in rs if r["cfg"] == c and "tokens" in r} for c in have))
        have = have or [c for c in CFGS if any(r["cfg"] == c for r in rs)]
        if not common:
            continue
        md += [f"### {ds} ({len(common)} clips)", "", "| metric | " + " | ".join(have) + " |",
               "|---|" + "---|" * len(have)]
        for k in keys:
            vals = {c: [r[k] for r in rs if r["cfg"] == c and r["clip"] in common and k in r] for c in have}
            if any(vals.values()):
                md.append(f"| {k} | " + " | ".join(f"{np.mean(vals[c]):.4g}" if vals[c] else "" for c in have) + " |")
        md.append("")
    open(f"{ACC}/vision_metrics.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
