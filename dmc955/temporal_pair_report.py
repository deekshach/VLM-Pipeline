#!/usr/bin/env python3.10
"""Summarize run_temporal_pair_eval.sh: stage latency and caption accuracy per mode.

    python3.10 temporal_pair_report.py sintel|msvd [_faoff]
"""
import json
import os
import re
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temporal_pair_results")
MODES = ["images", "pairs", "video"]
TAG = sys.argv[2] if len(sys.argv) > 2 else ""   # "_faoff": accuracy runs made with -fa off
NON_ASCII = re.compile(r"[^\x00-\x7f]")


def load(ds, mode, kind):
    path = os.path.join(R, f"{ds}_{mode}_{kind}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def latency(ds):
    print(f"\n== {ds}: latency per request, ms (mean over measured requests; fixed 128-token generation)")
    print(f"{'mode':8s} {'n':>4s} {'vis tok':>8s} {'ViT':>7s} {'prefill':>8s} {'decode':>8s} {'stages':>8s} {'vs images':>9s} "
          f"{'ViT':>6s} {'prefill':>7s} {'decode':>6s}")
    base = None
    for mode in MODES:
        d = load(ds, mode, "lat")
        if not d or not d.get("requests"):
            continue
        rq = d["requests"]
        m = {k: float(np.mean([r[k] for r in rq])) for k in ("vit_ms", "prefill_ms", "decode_ms", "n_visual_tokens")}
        m["stages"] = m["vit_ms"] + m["prefill_ms"] + m["decode_ms"]
        if mode == "images":
            base = m
        rel = lambda k: f"{100 * (m[k] / base[k] - 1):+.0f}%" if base else ""
        print(f"{mode:8s} {len(rq):4d} {m['n_visual_tokens']:8.0f} {m['vit_ms']:7.1f} {m['prefill_ms']:8.1f} "
              f"{m['decode_ms']:8.1f} {m['stages']:8.1f} {rel('stages'):>9s} {rel('vit_ms'):>6s} {rel('prefill_ms'):>7s} "
              f"{rel('decode_ms'):>6s}")


def texts(ds):
    print(f"\n== {ds}: generated text (natural stopping), two runs per mode")
    print(f"{'mode':8s} {'run':>3s} {'n':>4s} {'words':>6s} {'hit 128-token cap':>18s} {'non-ASCII text':>15s} "
          f"{'same as run 1':>14s} {'ViT ms':>7s} {'prefill ms':>10s}")
    first = {}
    for mode in MODES:
        for k in (1, 2):
            d = load(ds, mode, f"acc{k}{TAG}")
            if not d or not d.get("requests"):
                continue
            rq = d["requests"]
            print(f"{mode:8s} {k:3d} {len(rq):4d} {np.mean([len(r['text'].split()) for r in rq]):6.1f} "
                  f"{100 * np.mean([r['n_tokens'] >= d['n_predict'] for r in rq]):17.0f}% "
                  f"{100 * np.mean([bool(NON_ASCII.search(r['text'])) for r in rq]):14.0f}% "
                  f"{100 * np.mean([r['text'] == first.get(mode, {}).get(r['video']) for r in rq]):13.0f}% "
                  f"{np.mean([r['vit_ms'] for r in rq]):7.1f} {np.mean([r['prefill_ms'] for r in rq]):10.1f}")
            first.setdefault(mode, {r["video"]: r["text"] for r in rq})


def captions(ds):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    gt = {}
    for sf in ["msvd_train.json", "msvd_test.json"]:
        for it in json.load(open(os.path.join("/data/spring26/CPP/datasets/MSVD", sf))):
            gt[it["video_id"]] = [" ".join(norm(c)) for c in it["caption"]]

    preds = {}
    for mode in MODES:
        for k in (1, 2):
            d = load(ds, mode, f"acc{k}{TAG}")
            if d and d.get("requests"):
                preds[(mode, k)] = {r["video"]: " ".join(norm(r["text"])) for r in d["requests"] if r["video"] in gt}
    if not preds:
        return
    vids = sorted(set.intersection(*[set(p) for p in preds.values()]))

    def cider(sample, pred):
        gts = {i: gt[v] for i, v in enumerate(sample)}
        res = {i: [pred[v]] for i, v in enumerate(sample)}
        return Cider().compute_score(gts, res)[0]

    print(f"\n== {ds}: captions vs human references, {len(vids)} videos")
    print(f"{'mode':8s} {'run':>3s} {'CIDEr':>7s} {'BLEU-4':>7s} {'ROUGE-L':>8s}")
    for (mode, k), pred in preds.items():
        gts = {v: gt[v] for v in vids}
        res = {v: [pred[v]] for v in vids}
        print(f"{mode:8s} {k:3d} {Cider().compute_score(gts, res)[0]:7.3f} "
              f"{Bleu(4).compute_score(gts, res, verbose=0)[0][3]:7.3f} {Rouge().compute_score(gts, res)[0]:8.3f}")

    # mean of the two runs per mode, bootstrap over videos, difference to images
    rng = np.random.default_rng(0)
    boots = [list(rng.choice(vids, len(vids))) for _ in range(300)]
    mean2 = lambda mode, sample: np.mean([cider(sample, preds[(mode, k)]) for k in (1, 2) if (mode, k) in preds])
    base = np.array([mean2("images", b) for b in boots])
    print(f"\n{'mode':8s} {'CIDEr (mean of runs)':>21s} {'difference to images [95% interval]':>38s}")
    for mode in MODES:
        if (mode, 1) not in preds:
            continue
        cur = np.array([mean2(mode, b) for b in boots])
        diff = cur - base
        ci = "" if mode == "images" else f"{np.mean(diff):+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]"
        print(f"{mode:8s} {mean2(mode, vids):21.3f} {ci:>38s}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "sintel"
    if not TAG:
        latency(ds)
    texts(ds)
    if ds == "msvd":
        captions(ds)
