#!/usr/bin/env python3.10
"""Summarize run_pair_reuse_eval.sh: how much is reusable and what it does to the LLM output.

    python3.10 pair_reuse_report.py sintel|msvd spread|dense
"""
import json
import os
import re
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_reuse_results")


def norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def main():
    ds, setting = sys.argv[1], sys.argv[2]
    d = json.load(open(os.path.join(R, f"{ds}_{setting}.json")))
    vids = d["videos"]
    names = [r["name"] for r in vids[0]["runs"]]
    runs = {n: [next(r for r in v["runs"] if r["name"] == n) for v in vids] for n in names}

    cc = [v["consecutive_pair_token_cos"] for v in vids if v["consecutive_pair_token_cos"]]
    print(f"\n== {ds} {setting}: {len(vids)} videos, {np.mean([v['n_pairs'] for v in vids]):.1f} pairs each, "
          f"{np.mean([r['n_visual'] for r in runs['baseline']]):.0f} visual tokens, ViT {np.mean([v['vit_ms'] for v in vids]):.0f} ms, "
          f"baseline prefill {np.mean([r['prefill_ms'] for r in runs['baseline']]):.0f} ms")
    print("true tokens, one pair to the next: median cosine %.2f; share with cosine >= 0.95 / 0.90 / 0.80 / 0.70: %s" % (
        np.mean([c["median"] for c in cc]),
        " / ".join(f"{100 * np.mean([c[k] for c in cc]):.0f}%" for k in ("ge_0.95", "ge_0.90", "ge_0.80", "ge_0.70"))))

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    base_text = [r["text"] for r in runs["baseline"]]

    cap = None
    if ds == "msvd":
        from pycocoevalcap.cider.cider import Cider
        gt = {}
        for sf in ["msvd_train.json", "msvd_test.json"]:
            for it in json.load(open(os.path.join("/data/spring26/CPP/datasets/MSVD", sf))):
                gt[it["video_id"]] = [" ".join(norm(c)) for c in it["caption"]]
        ids = [v["video"] for v in vids]
        keep = [i for i, v in enumerate(ids) if v in gt]
        rng = np.random.default_rng(0)
        boots = [rng.choice(keep, len(keep)) for _ in range(300)]

        def cider(sample, texts):
            gts = {k: gt[ids[i]] for k, i in enumerate(sample)}
            res = {k: [" ".join(norm(texts[i]))] for k, i in enumerate(sample)}
            return Cider().compute_score(gts, res)[0]
        base_boot = np.array([cider(b, base_text) for b in boots])
        cap = (keep, boots, cider, base_boot)

    print(f"\n{'variant':16s} {'reused':>7s} {'pairs':>6s} {'rect':>6s} {'cos in':>7s} {'LLM tok':>8s} {'prefill':>8s} "
          f"{'KL':>7s} {'top-1':>6s} {'same':>5s} {'R-L':>5s}" + (f" {'CIDEr':>6s} {'dCIDEr [95%]':>22s}" if cap else ""))
    print(f"{'':16s} {'tokens':>7s} {'skipd':>6s} {'encode':>6s} {'':>7s} {'vs base':>8s} {'vs base':>8s} "
          f"{'':>7s} {'agree':>6s} {'text':>5s} {'vs b.':>5s}")
    base_prefill = np.mean([r["prefill_ms"] for r in runs["baseline"]])
    for n in names:
        rs = runs[n]
        m = lambda k, default=0.0: float(np.mean([r.get(k, default) for r in rs]))
        rl = np.mean([scorer.score(b, r["text"])["rougeL"].fmeasure for b, r in zip(base_text, rs)])
        line = (f"{n:16s} {100 * m('frac_reused'):6.0f}% {100 * m('pairs_all_reused'):5.0f}% "
                f"{100 * m('frac_rect', 1.0):5.0f}% {m('cos_input', 1.0):7.3f} {100 * m('frac_visual'):7.0f}% "
                f"{100 * m('prefill_ms') / base_prefill:7.0f}% {m('kl_mean'):7.4f} {100 * m('top1_agree'):5.0f}% "
                f"{100 * m('same_text'):4.0f}% {rl:5.2f}")
        if cap:
            keep, boots, cider, base_boot = cap
            texts = [r["text"] for r in rs]
            diff = np.array([cider(b, texts) for b in boots]) - base_boot
            line += f" {cider(keep, texts):6.3f}"
            if n != "baseline":
                line += f" {np.mean(diff):+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]"
        print(line)


if __name__ == "__main__":
    main()
