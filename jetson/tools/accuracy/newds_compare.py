"""The techniques on the small NExT-QA / Video-MME / Charades sets (prepare_newds.py, run_accuracy.sh).

    python newds_compare.py <nextqa|videomme|charades> [cfgs...]

Per config, over the clips present for every config:
  cost      frames, image tokens, ViT ms, TTFT, share of ViT tokens re-encoded (per-window cache)
  accuracy  multiple choice: option letter correct; Charades: CIDEr / METEOR / SBERT vs the Charades-STA sentences
  vs base   same first token, same answer (multiple choice) or caption SBERT
  vs ours_nocache (same frames, no cache)  first-token KL and agreement, same answer or caption SBERT
paused: runs whose wall time overlapped a board guard pause. The window is generous (it includes model load and
generation), so this over-counts; check the TTFT spread for +30 s outliers before dropping any run.
"""
import datetime as dt, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_accuracy as aa

ROOT = "/mnt/external_ssd/VLM-datasets/eval"
GUARD = f"{aa.ACC}/board_guard.log"


def guard_pauses():
    """Pause intervals from the board guard log (times only: a day boundary is a time going backwards)."""
    if not os.path.exists(GUARD):
        return []
    first = dt.datetime.fromtimestamp(os.path.getctime(GUARD)).date()
    day, prev, cur, out = dt.datetime.combine(first, dt.time()), None, None, []
    for l in open(GUARD):
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (\w+)", l)
        if not m:
            continue
        t = dt.timedelta(hours=int(m[1]), minutes=int(m[2]), seconds=int(m[3]))
        if prev is not None and t < prev:
            day += dt.timedelta(days=1)
        prev = t
        if m[4] == "paused":
            cur = day + t
        elif m[4] == "resumed" and cur:
            out.append((cur, day + t))
            cur = None
    return out


def paused(path, res, pauses):
    end = dt.datetime.fromtimestamp(os.path.getmtime(path))
    start = end - dt.timedelta(seconds=res["timing"]["total_ms"] / 1000 + 20)   # + model load
    return any(a < end and b > start for a, b in pauses)


def letter(text):
    m = re.match(r"\s*\(?([A-E])\b", text)
    return m[1] if m else None


def main():
    ds = sys.argv[1]
    cfgs = sys.argv[2:] or ["base", "uni_arch", "sel", "ours_nocache", "pw12", "pw24", "pw36"]
    items = json.load(open(f"{ROOT}/{ds}/items.json"))
    mc = ds != "charades"
    clips = sorted(v for v in items if all(os.path.exists(f"{aa.RUNS}/{ds}/{c}/{v}.json") for c in cfgs))
    runs = {c: {v: aa.load_run(ds, c, v) for v in clips} for c in cfgs}
    clips = [v for v in clips if all(runs[c][v] is not None for c in cfgs)]
    pauses = guard_pauses()
    print(f"{ds}: {len(clips)} clips")

    emb = {} if mc else {c: aa.sbert([runs[c][v]["text"] for v in clips]) for c in cfgs}
    rows = {}
    for c in cfgs:
        r = {k: [] for k in ("frames", "img_tok", "vit_ms", "ttft", "enc_frac", "acc",
                             "top1_base", "same_base", "kl_nc", "top1_nc", "same_nc")}
        n_paused = 0
        for i, v in enumerate(clips):
            run = runs[c][v]
            res = run["res"]
            r["frames"].append(len(run["imgs"]))
            r["img_tok"].append(sum(im["nx"] * im["ny"] for im in run["imgs"]))
            n_paused += paused(f"{aa.RUNS}/{ds}/{c}/{v}.json", res, pauses)
            b = res["timing"]["breakdown"]
            r["ttft"].append(res["timing"]["ttft_ms"])
            r["vit_ms"].append(sum(b[k]["vit_ms"] for k in ("rest", "anchor") if k in b))
            wc = res["timing"]["breakdown"].get("window_cache", {}).get("frames", [])
            if wc:
                r["enc_frac"].append(sum(f["tokens_encoded"] for f in wc[1:]) / (wc[0]["tokens_encoded"] * (len(wc) - 1))
                                     if len(wc) > 1 else 1.0)
            if mc:
                r["acc"].append(float(letter(run["text"]) == items[v]["answer"]))
            for ref, kt, ks, kk in (("base", "top1_base", "same_base", None), ("ours_nocache", "top1_nc", "same_nc", "kl_nc")):
                if ref not in runs:
                    continue
                rr = runs[ref][v]
                if run["logits"] is not None and rr["logits"] is not None:
                    p, q = aa.softmax(rr["logits"]), aa.softmax(run["logits"])
                    r[kt].append(float(p.argmax() == q.argmax()))
                    if kk:
                        r[kk].append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
                r[ks].append(float(letter(run["text"]) == letter(rr["text"])) if mc else float(emb[c][i] @ emb[ref][i]))
        rows[c] = {k: (float(np.mean(x)) if len(x) else float("nan")) for k, x in r.items()}
        rows[c]["paused"] = n_paused
        if not mc:
            preds = {v: runs[c][v]["text"] for v in clips}
            refs = {v: items[v]["refs"] for v in clips}
            sc, _ = aa.coco_scores(preds, refs)
            rows[c]["CIDEr"], rows[c]["METEOR"] = sc["CIDEr"], sc["METEOR"]
            rows[c]["sbert_gt"] = float(np.mean([np.max(aa.sbert(refs[v]) @ aa.sbert([preds[v]])[0]) for v in clips]))

    keys = ["frames", "img_tok", "vit_ms", "ttft", "paused", "enc_frac"]
    keys += ["acc"] if mc else ["CIDEr", "METEOR", "sbert_gt"]
    keys += ["top1_base", "same_base", "kl_nc", "top1_nc", "same_nc"]
    print("| metric | " + " | ".join(cfgs) + " |")
    print("|---|" + "---|" * len(cfgs))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{rows[c][k]:.4g}" for c in cfgs) + " |")
    if mc:
        for v in clips:
            print(v, items[v]["answer"], items[v].get("qtype", ""), " ".join(f"{c}={letter(runs[c][v]['text'])}" for c in cfgs))
    json.dump(rows, open(f"{aa.RUNS}/newds_compare_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
