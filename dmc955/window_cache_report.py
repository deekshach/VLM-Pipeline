#!/usr/bin/env python3.10
"""Summarize run_window_cache_eval.sh: stage latency, embedding fidelity, caption accuracy.

    python3.10 window_cache_report.py sintel|msvd
"""
import json
import os
import re
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_cache_results")
CONFIGS = ["full_native", "full_native2", "full_aligned", "legacy",
           "window_t6", "window_t24", "window_t48", "window_t96", "window_t48_nograph"]
REF = "full_native"


def load(ds, name, kind):
    path = os.path.join(R, f"{ds}_{name}_{kind}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def latency(ds):
    print(f"\n== {ds}: latency per request, ms (mean over measured requests; fixed 128-token generation)")
    print(f"{'config':20s} {'n':>4s} {'analysis':>8s} {'ViT':>7s} {'prefill':>8s} {'decode':>8s} {'stages':>8s} "
          f"{'vs full':>8s} {'ViT/frame':>9s} {'tok enc':>8s}  actions")
    base = None
    for name in CONFIGS:
        d = load(ds, name, "lat")
        if not d or not d.get("requests"):
            continue
        rq = d["requests"]
        m = lambda k: float(np.mean([r[k] for r in rq]))
        stages = m("analysis_ms") + m("vit_ms") + m("prefill_ms") + m("decode_ms")
        if name == REF:
            base = stages
        frames = [f for r in rq for f in r["frames"]]
        acts = {}
        for f in frames:
            acts[f["action"]] = acts.get(f["action"], 0) + 1
        grid_tok = np.mean([r["grid"][0] * r["grid"][1] for r in rq])
        enc = np.mean([f["tokens_encoded"] for f in frames]) / grid_tok
        vit_frame = m("vit_ms") / np.mean([r["n_frames"] for r in rq])
        delta = f"{100 * (stages / base - 1):+.1f}%" if base else ""
        acts_s = " ".join(f"{k}:{100 * v / len(frames):.0f}%" for k, v in sorted(acts.items()))
        print(f"{name:20s} {len(rq):4d} {m('analysis_ms'):8.1f} {m('vit_ms'):7.1f} {m('prefill_ms'):8.1f} "
              f"{m('decode_ms'):8.1f} {stages:8.1f} {delta:>8s} {vit_frame:9.1f} {100 * enc:7.0f}%  {acts_s}")


def fidelity(ds):
    print(f"\n== {ds}: embeddings handed to the LLM vs a full encode of the same frame (non-key frames)")
    print(f"{'config':20s} {'cos mean':>9s} {'cos p05':>8s} {'<0.90':>7s} {'reused':>8s} {'fresh':>8s} {'tok reused':>10s}")
    for name in CONFIGS:
        d = load(ds, name, "acc")
        if not d or not d.get("requests"):
            continue
        fr = [f for r in d["requests"] for f in r["frames"][1:] if "cos_mean" in f]
        if not fr:
            continue
        grid = {id(f): r["grid"][0] * r["grid"][1] for r in d["requests"] for f in r["frames"]}
        reused = [f["cos_reused"] for f in fr if f["cos_reused"] >= 0]
        fresh = [f["cos_fresh"] for f in fr if f["cos_fresh"] >= 0]
        frac_reused = np.mean([1 - f["tokens_updated"] / grid[id(f)] for f in fr])
        print(f"{name:20s} {np.mean([f['cos_mean'] for f in fr]):9.4f} {np.mean([f['cos_p05'] for f in fr]):8.4f} "
              f"{100 * np.mean([f['frac_below_090'] for f in fr]):6.1f}% "
              f"{(np.mean(reused) if reused else float('nan')):8.4f} {(np.mean(fresh) if fresh else float('nan')):8.4f} "
              f"{100 * frac_reused:9.1f}%")


def text_vs_reference(ds):
    from rouge_score import rouge_scorer
    ref = load(ds, REF, "acc")
    if not ref:
        return
    ref_text = {r["video"]: r["text"] for r in ref["requests"]}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    print(f"\n== {ds}: generated text vs the {REF} text for the same video "
          f"(full_native2 is the same config rerun: the run-to-run floor)")
    print(f"{'config':20s} {'ROUGE-1':>8s} {'ROUGE-L':>8s} {'identical':>9s} {'words':>6s}")
    for name in CONFIGS:
        d = load(ds, name, "acc")
        if not d or name == REF or not d.get("requests"):
            continue
        r1, rl, same, words = [], [], [], []
        for r in d["requests"]:
            if r["video"] not in ref_text:
                continue
            s = scorer.score(ref_text[r["video"]], r["text"])
            r1.append(s["rouge1"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
            same.append(r["text"].strip() == ref_text[r["video"]].strip())
            words.append(len(r["text"].split()))
        print(f"{name:20s} {np.mean(r1):8.3f} {np.mean(rl):8.3f} {100 * np.mean(same):8.0f}% {np.mean(words):6.1f}")


def captions(ds):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    gt = {}
    for sf in ["msvd_train.json", "msvd_test.json"]:
        for it in json.load(open(os.path.join("/data/spring26/CPP/datasets/MSVD", sf))):
            gt[it["video_id"]] = [" ".join(norm(c)) for c in it["caption"]]

    def score(vids, pred):
        gts = {v: gt[v] for v in vids}
        res = {v: [pred[v]] for v in vids}
        return (Cider().compute_score(gts, res)[0], Bleu(4).compute_score(gts, res, verbose=0)[0][3],
                Rouge().compute_score(gts, res)[0])

    preds = {}
    for name in CONFIGS:
        d = load(ds, name, "acc")
        if d and d.get("requests"):
            preds[name] = {r["video"]: " ".join(norm(r["text"])) for r in d["requests"] if r["video"] in gt}
    if REF not in preds:
        return
    vids = sorted(set.intersection(*[set(p) for p in preds.values()]))
    print(f"\n== {ds}: captions vs human references, {len(vids)} videos "
          f"(dCIDEr = paired bootstrap vs {REF}, 95% interval)")
    print(f"{'config':20s} {'CIDEr':>7s} {'BLEU-4':>7s} {'ROUGE-L':>8s} {'dCIDEr [lo, hi]':>24s}")
    rng = np.random.default_rng(0)
    boots = [list(rng.choice(vids, len(vids))) for _ in range(300)]

    def boot_score(sample, pred):
        # resampled videos may repeat; give each draw its own key
        gts = {i: gt[v] for i, v in enumerate(sample)}
        res = {i: [pred[v]] for i, v in enumerate(sample)}
        return Cider().compute_score(gts, res)[0]

    ref_boot = [boot_score(b, preds[REF]) for b in boots]
    for name, pred in preds.items():
        c, b4, rl = score(vids, pred)
        if name == REF:
            ci = ""
        else:
            diff = np.array([boot_score(b, pred) for b in boots]) - np.array(ref_boot)
            ci = f"{np.mean(diff):+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]"
        print(f"{name:20s} {c:7.3f} {b4:7.3f} {rl:8.3f} {ci:>24s}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "sintel"
    latency(ds)
    fidelity(ds)
    text_vs_reference(ds)
    if ds == "msvd":
        captions(ds)
