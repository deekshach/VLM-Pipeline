#!/usr/bin/env python3.10
"""How many 112 px windows (and whole frames) stay unchanged, per change threshold.

Pixels only, no model. Same score as window_cache_vlm.cpp: frames resized to a
multiple of 112, window score = max over its 4x4 token cells (28 px) of the mean
|dRGB| (0-255). A window is "same" when score <= threshold.

  prev  compare each frame with the previous sampled frame
  ref   compare with the pixels the window was last encoded from (what a cache
        can actually use: a reused window keeps its old reference)

    python3.10 window_reuse_opportunity.py sintel|msvd
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

DATASETS = {
    "sintel": "/data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel/training/clean",
    "msvd": "/data/spring26/CPP/datasets/MSVD/MSVD_prepared/training/final",
}
THRESHOLDS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
# 0 = 8 frames spread evenly over the clip (what the eval used); N = every Nth frame
SAMPLINGS = [1, 2, 4, 8, 0]
WIN, TOK = 112, 28


def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    aw, ah = max(WIN, round(w / WIN) * WIN), max(WIN, round(h / WIN) * WIN)
    if (aw, ah) != (w, h):
        img = cv2.resize(img, (aw, ah), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.int16)


def cell_diff(a, b):
    """mean |dRGB| per 28 px token cell -> (rows, cols)"""
    d = np.abs(a - b).mean(axis=2)
    h, w = d.shape
    return d.reshape(h // TOK, TOK, w // TOK, TOK).mean(axis=(1, 3))


def window_score(cells):
    r, c = cells.shape
    return cells.reshape(r // 4, 4, c // 4, 4).max(axis=(1, 3))


def pick(n, sampling):
    if sampling == 0:
        return sorted(set(int(round(i * (n - 1) / 7)) for i in range(8))) if n >= 8 else list(range(n))
    return list(range(0, n, sampling))


def video_stats(args):
    root, vid = args
    files = sorted(f for f in os.listdir(os.path.join(root, vid)) if f.endswith(".png"))
    frames = [load(os.path.join(root, vid, f)) for f in files]
    out = {}
    for s in SAMPLINGS:
        idx = pick(len(frames), s)
        if len(idx) < 2:
            continue
        sel = [frames[i] for i in idx]
        # prev: one score map per frame pair
        prev = np.stack([window_score(cell_diff(sel[i], sel[i - 1])) for i in range(1, len(sel))])
        res = {"n_pairs": len(sel) - 1, "n_win": int(prev[0].size), "prev": {}, "ref": {}}
        for t in THRESHOLDS:
            same = prev <= t
            frac = same.reshape(len(same), -1).mean(axis=1)
            res["prev"][t] = [float(same.mean()), float((frac == 1).mean()),
                              float((frac >= 0.75).mean()), float((frac >= 0.5).mean())]
            # ref: simulate the cache; dirty windows refresh their reference pixels
            ref = sel[0].copy()
            fr = []
            for cur in sel[1:]:
                dirty = window_score(cell_diff(cur, ref)) > t
                fr.append(1.0 - dirty.mean())
                for wy, wx in zip(*np.nonzero(dirty)):
                    ref[wy * WIN:(wy + 1) * WIN, wx * WIN:(wx + 1) * WIN] = \
                        cur[wy * WIN:(wy + 1) * WIN, wx * WIN:(wx + 1) * WIN]
            fr = np.array(fr)
            res["ref"][t] = [float(fr.mean()), float((fr == 1).mean()),
                             float((fr >= 0.75).mean()), float((fr >= 0.5).mean())]
        out[s] = res
    return vid, out


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "sintel"
    root = DATASETS[ds]
    vids = sorted(v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v)))
    with ProcessPoolExecutor(16) as ex:
        results = dict(ex.map(video_stats, [(root, v) for v in vids]))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_cache_results")
    json.dump(results, open(os.path.join(out_dir, f"reuse_opportunity_{ds}.json"), "w"))

    for s in SAMPLINGS:
        per = [r[s] for r in results.values() if s in r]
        if not per:
            continue
        w = np.array([p["n_pairs"] for p in per], dtype=float)
        name = "8 frames spread evenly" if s == 0 else f"every {s} frame(s)"
        print(f"\n== {ds}: {name}  ({len(per)} videos, {int(w.sum())} frame pairs; weighted by pairs)")
        print(f"{'thr':>4s} | {'win same':>8s} {'frame same':>10s} {'>=75% same':>10s} {'>=50% same':>10s} | "
              f"{'win same':>8s} {'frame same':>10s} {'>=75% same':>10s} {'>=50% same':>10s}")
        print(f"{'':4s} | {'-- vs previous frame --':^41s} | {'-- vs last-encoded pixels (cache) --':^41s}")
        for t in THRESHOLDS:
            row = []
            for mode in ("prev", "ref"):
                m = np.array([p[mode][t] for p in per])
                row += list(100 * (m * w[:, None]).sum(axis=0) / w.sum())
            print(f"{t:4d} | {row[0]:7.1f}% {row[1]:9.1f}% {row[2]:9.1f}% {row[3]:9.1f}% | "
                  f"{row[4]:7.1f}% {row[5]:9.1f}% {row[6]:9.1f}% {row[7]:9.1f}%")


if __name__ == "__main__":
    main()
