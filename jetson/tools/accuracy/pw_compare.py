"""Window cache variants on the same clips: no cache (reference), rectangle cache, per-window cache.

    ACC_RUNS=/home/nvidia/data/accuracy/runs_pwsmall python pw_compare.py <sintel|msvd> [ref=ours_nocache] [cfgs...]

Per config, over all clips present for every config:
  ViT       share of ViT tokens re-encoded on non-key frames, ViT ms per request, TTFT
  fidelity  token cosine vs the same frame encoded whole (dense aligned reference), split into tokens re-encoded
            this frame and tokens reused from the cache, over non-key frames
  LLM       vs the reference config (same frames, no cache): first-token KL and top-1 agreement, caption exact
            match and SBERT similarity; MSVD: CIDEr / METEOR / SBERT vs the ground-truth captions
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_accuracy as aa


def reencoded_mask(img):
    """True for tokens re-encoded this frame (key / full: all)."""
    ny, nx = img["ny"], img["nx"]
    m = np.zeros((ny, nx), bool)
    a = img["action"]
    if a in ("key", "full"):
        m[:] = True
    elif a == "crop":
        rx, ry, rw, rh = img["rect"]
        m[ry * 4:(ry + rh) * 4, rx * 4:(rx + rw) * 4] = True
    elif a == "pw":
        for w in img["windows"]:
            wy, wx = divmod(w, nx // 4)
            m[wy * 4:wy * 4 + 4, wx * 4:wx * 4 + 4] = True
    return m.ravel()


def main():
    ds = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "ours_nocache"
    cfgs = sys.argv[3:] or ["ours_nocache", "ours", "ours_pw"]
    clips = None
    for c in cfgs:
        cl = {f[:-5] for f in os.listdir(f"{aa.RUNS}/{ds}/{c}") if f.endswith(".json")}
        clips = cl if clips is None else clips & cl
    clips = sorted(clips)
    runs = {c: {v: aa.load_run(ds, c, v) for v in clips} for c in cfgs}
    clips = [v for v in clips if all(runs[c][v] is not None for c in cfgs)]
    print(f"{ds}: {len(clips)} clips ({', '.join(clips)})")

    rows = {}
    texts = {c: [runs[c][v]["text"] for v in clips] for c in cfgs}
    emb = {c: aa.sbert(texts[c]) for c in cfgs}
    for c in cfgs:
        r = dict(enc_frac=[], vit_ms=[], ttft=[], cos_enc=[], cos_reu=[], cos_all=[], kl=[], top1=[], exact=[], sbert=[])
        for i, v in enumerate(clips):
            run, rr = runs[c][v], runs[ref][v]
            t = run["res"]["timing"]
            r["ttft"].append(t.get("ttft_ms", np.nan))
            wc = t.get("breakdown", {}).get("window_cache", {}).get("frames", [])
            r["vit_ms"].append(sum(f["encode_ms"] for f in wc) if wc else np.nan)
            dense = aa.load_dense(ds, v, "aligned")
            n_enc = n_tok = 0
            for img, f in list(zip(run["imgs"], run["frames"]))[1:]:   # non-key frames
                mk = reencoded_mask(img)
                n_enc += int(mk.sum()); n_tok += mk.size
                if dense and f in dense and dense[f]["x"].shape == img["x"].shape:
                    cs = np.sum(aa.unit(img["x"]) * aa.unit(dense[f]["x"]), axis=1)
                    r["cos_all"].extend(cs.tolist())
                    r["cos_enc"].extend(cs[mk].tolist())
                    r["cos_reu"].extend(cs[~mk].tolist())
            r["enc_frac"].append(n_enc / n_tok if n_tok else np.nan)
            if run["logits"] is not None and rr["logits"] is not None:
                p, q = aa.softmax(rr["logits"]), aa.softmax(run["logits"])
                r["kl"].append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
                r["top1"].append(float(p.argmax() == q.argmax()))
            r["exact"].append(float(run["text"] == rr["text"]))
            r["sbert"].append(float(emb[c][i] @ emb[ref][i]))
        rows[c] = {k: (float(np.nanmean(x)) if len(x) else float("nan")) for k, x in r.items()}
        rows[c]["n_reused_tok"] = len(r["cos_reu"])

    if ds == "msvd":
        refs = json.load(open(f"{aa.ACC}/msvd/captions.json"))
        for c in cfgs:
            preds = {v: runs[c][v]["text"] for v in clips}
            sc, _ = aa.coco_scores(preds, refs)
            rows[c]["CIDEr"], rows[c]["METEOR"] = sc["CIDEr"], sc["METEOR"]
            rows[c]["sbert_gt"] = float(np.mean([np.max(aa.sbert(refs[v]) @ aa.sbert([preds[v]])[0]) for v in clips]))

    keys = ["enc_frac", "vit_ms", "ttft", "cos_all", "cos_enc", "cos_reu", "n_reused_tok", "kl", "top1", "exact", "sbert"]
    keys += [k for k in ("CIDEr", "METEOR", "sbert_gt") if k in rows[cfgs[0]]]
    print("| metric | " + " | ".join(cfgs) + " |")
    print("|---|" + "---|" * len(cfgs))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{rows[c][k]:.4g}" for c in cfgs) + " |")
    json.dump(rows, open(f"{aa.RUNS}/pw_compare_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
