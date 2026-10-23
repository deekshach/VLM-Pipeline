#!/usr/bin/env python3
"""Per-clip scores of the main evaluation: the task score next to the visual-token metrics, one row per clip and
config, with the per-dataset mean (the aggregate tables of qwen2fps_compare.py and vision_metrics.py) under each.
Python env: /home/nvidia/data/envs/vlm-eval.

    python per_clip_scores.py [out prefix=/home/nvidia/data/accuracy/per_clip_scores]   -> <prefix>.csv, <prefix>.md

Task score, over the clips every config with runs has (as qwen2fps_compare.py):
  multiple choice   correct (1/0) and the predicted letter
  captions          SBERT best cosine to a reference; CIDEr per clip (pycocoevalcap, document frequencies from the
                    dataset's clips, so the mean is the corpus CIDEr); ROUGE-L per clip (its mean is the corpus value)
  Sintel            SBERT cosine to the base caption
Visual-token metrics per clip are read from vision_metrics.csv (run vision_metrics.py first).
"""
import csv, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qwen2fps_compare as q
import analyze_accuracy as aa

ARMS = ["base", "uni4", "qwen2fps", "ours_nocache", "pw24", "pwd24"]
VM = ["tokens", "spec_Hm", "tok_recall", "redund", "clip_ground", "clip_mc", "clip_r1"]
NAMES = {"sintel": "Sintel", "msvd": "MSVD", "charades": "Charades-STA", "tacos": "TACoS", "nextqa": "NExT-QA",
         "videomme": "Video-MME", "perception": "Perception Test"}


def rouge_l(preds, refs):
    from pycocoevalcap.rouge.rouge import Rouge
    norm = lambda s: " ".join(aa.tok(s))
    gts = {k: [norm(c) for c in refs[k]] for k in preds}
    res = {k: [norm(preds[k]) or "empty"] for k in preds}
    return dict(zip(preds, Rouge().compute_score(gts, res)[1]))


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else f"{q.ACC}/per_clip_scores"
    vm = {(r["dataset"], r["clip"], r["cfg"]): r for r in csv.DictReader(open(f"{q.ACC}/vision_metrics.csv"))}
    msvd_refs = json.load(open(f"{q.ACC}/msvd/captions.json"))
    rows = []
    for ds in q.DATASETS:
        runs = {c: {v: q.run(ds, c, v) for v in q.clips(ds)} for c in ARMS}
        have = [c for c in ARMS if any(runs[c].values())]
        common = [v for v in q.clips(ds) if all(runs[c][v] for c in have)]
        p = f"{q.EVAL}/{ds}/items.json"
        items = json.load(open(p)) if os.path.exists(p) else {}
        score = {c: {v: {} for v in common} for c in have}
        if items and "answer" in next(iter(items.values())):
            for c in have:
                for v in common:
                    l = q.letter(runs[c][v][0])
                    score[c][v].update(pred=l or "", gold=items[v]["answer"], correct=int(l == items[v]["answer"]))
        elif ds == "sintel":
            b = aa.sbert([runs["base"][v][0] for v in common])
            for c in have:
                e = aa.sbert([runs[c][v][0] for v in common])
                for v, s in zip(common, (e * b).sum(1)):
                    score[c][v]["sbert"] = float(s)
        else:
            refs = {v: (msvd_refs[v] if ds == "msvd" else items[v]["refs"]) for v in common}
            for c in have:
                preds = {v: runs[c][v][0] for v in common}
                _, cider = aa.coco_scores(preds, refs)
                rl = rouge_l(preds, refs)
                for v in common:
                    score[c][v].update(sbert=float((aa.sbert(refs[v]) @ aa.sbert([preds[v]])[0]).max()),
                                       cider=float(cider[v]), rouge_l=float(rl[v]))
        for v in common:
            for c in have:
                m = vm.get((ds, v, c), {})
                rows.append(dict(dataset=ds, clip=v, cfg=c, frames=runs[c][v][1], **score[c][v],
                                 **{k: m.get(k, "") for k in VM}))
    fields = ["dataset", "clip", "cfg", "frames", "correct", "pred", "gold", "sbert", "cider", "rouge_l"] + VM
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # markdown: per dataset and metric, clips x arms, the mean last
    def fmt(x, k):
        if x in ("", None):
            return "—"
        if k == "correct":
            return "✓" if int(x) else "✗"
        if k in ("tokens", "frames"):
            return f"{float(x):.0f}" if float(x) == int(float(x)) else f"{float(x):.1f}"
        return f"{float(x):.2f}" if k in ("cider", "clip_mc", "clip_r1") else f"{float(x):.3f}"
    out = []
    for ds in q.DATASETS:
        R = [r for r in rows if r["dataset"] == ds]
        if not R:
            continue
        cl = list(dict.fromkeys(r["clip"] for r in R))
        cfgs = [c for c in ARMS if any(r["cfg"] == c for r in R)]
        get = {(r["clip"], r["cfg"]): r for r in R}
        ks = [k for k in ["correct", "sbert", "cider", "rouge_l", "frames"] + VM if any(r.get(k) not in ("", None) for r in R)]
        out.append(f"## {NAMES[ds]} ({len(cl)} clips)\n")
        for k in ks:
            out += [f"### {k}\n", "| clip | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
            for v in cl:
                out.append(f"| {v} | " + " | ".join(fmt(get[(v, c)].get(k), k) for c in cfgs) + " |")
            mean = []
            for c in cfgs:
                xs = [float(get[(v, c)][k]) for v in cl if get[(v, c)].get(k) not in ("", None)]
                mean.append((f"{100 * st.mean(xs):.1f}%" if k == "correct" else fmt(st.mean(xs), k))
                            if xs else "—")
            out += ["| **mean** | " + " | ".join(mean) + " |", ""]
    open(f"{prefix}.md", "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} rows -> {prefix}.csv, {prefix}.md")


if __name__ == "__main__":
    main()
