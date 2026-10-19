#!/usr/bin/env python3
"""Temporal and spatial redundancy of the evaluation clips: why a fixed frame count and whole-frame re-encoding waste work.

    python video_stats.py [out_dir=experiments/video_stats] [workers=12]     (conda env VLM: OpenCV)

Every clip is read as the pipeline sees it: the extracted frames at 4 fps (Sintel, 24 fps native, every 6th frame),
scaled so the long side is 672 px and both sides are multiples of the 112 px ViT attention window. Per pair of
consecutive frames:
  motion     mean Farneback flow magnitude, % of frame width per second
  moving     share of pixels whose flow is >= 1 px (at 672 px, 4 fps)
  dirty      share of 112 px windows whose mean |dRGB| exceeds 24: the window cache's (A6) re-encode test
  conc10     share of the flow magnitude in the 10% of windows that move most (spread evenly: 10%)
Per clip, besides the means:
  burst25    share of the clip's motion in its 25% most active pairs (constant motion: 25%)
  uni8_*     the baseline's 8 uniformly spaced frames: share of windows dirty between consecutive samples;
             share of the 7 gaps that are redundant (< 10% dirty) or under-sampled (> 50% dirty)
Writes per_clip.csv, per_pair.csv (the motion time series) and summary.md.
"""
import csv, glob, os, sys
from multiprocessing import Pool
import cv2
import numpy as np

SETS = {   # name -> (frames root, frame stride to 4 fps)
    "msvd": ("/home/nvidia/data/accuracy/msvd/training/clean", 1),
    "sintel": ("/home/nvidia/data/MPI-Sintel/training/clean", 6),
    "charades": ("/mnt/external_ssd/VLM-datasets/eval/charades/training/clean", 1),
    "nextqa": ("/mnt/external_ssd/VLM-datasets/eval/nextqa/training/clean", 1),
    "videomme": ("/mnt/external_ssd/VLM-datasets/eval/videomme/training/clean", 1),
    "perception": ("/mnt/external_ssd/VLM-datasets/eval/perception/training/clean", 1),
    "tacos": ("/mnt/external_ssd/VLM-datasets/eval/tacos/training/clean", 1),
}
FPS, WIN, LONG, WC_T = 4.0, 112, 672, 24.0


def load(path):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    s = LONG / max(h, w)
    W, H = (max(WIN, int(round(x * s / WIN)) * WIN) for x in (w, h))
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)


def dirty(a, b):
    """Share of 112 px windows whose mean |dRGB| exceeds the window-cache threshold."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    H, W = d.shape
    m = d.reshape(H // WIN, WIN, W // WIN, WIN).mean(axis=(1, 3))
    return float((m > WC_T).mean())


def clip_stats(job):
    ds, clip, frames = job
    ims = [load(f) for f in frames]
    grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in ims]
    H, W = grays[0].shape
    pairs = []
    for t in range(len(ims) - 1):
        fl = cv2.calcOpticalFlowFarneback(grays[t], grays[t + 1], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        mag = np.hypot(fl[..., 0], fl[..., 1])
        win = mag.reshape(H // WIN, WIN, W // WIN, WIN).mean(axis=(1, 3)).ravel()
        k = max(1, int(round(0.1 * win.size)))
        pairs.append(dict(motion=100 * mag.mean() / W * FPS, moving=float((mag >= 1).mean()), dirty=dirty(ims[t], ims[t + 1]),
                          conc10=float(np.sort(win)[-k:].sum() / max(win.sum(), 1e-9))))
    mo = np.array([p["motion"] for p in pairs])
    k = max(1, int(round(0.25 * len(mo))))
    u = [int(round(i * (len(ims) - 1) / 7)) for i in range(8)]
    ud = [dirty(ims[a], ims[b]) for a, b in zip(u, u[1:])]
    row = dict(dataset=ds, clip=clip, frames=len(ims), windows=(H // WIN) * (W // WIN),
               motion=float(mo.mean()), motion_p90=float(np.percentile(mo, 90)),
               burst25=float(np.sort(mo)[-k:].sum() / max(mo.sum(), 1e-9)),
               moving=float(np.mean([p["moving"] for p in pairs])), dirty=float(np.mean([p["dirty"] for p in pairs])),
               conc10=float(np.mean([p["conc10"] for p in pairs])),
               uni8_dirty=float(np.mean(ud)), uni8_redundant=float(np.mean([d < 0.1 for d in ud])),
               uni8_under=float(np.mean([d > 0.5 for d in ud])))
    return row, [dict(dataset=ds, clip=clip, t=t, **p) for t, p in enumerate(pairs)]


def jobs():
    for ds, (root, stride) in SETS.items():
        seen = set()
        for clip in sorted(os.listdir(root)):
            real = os.path.realpath(f"{root}/{clip}")
            if real in seen:   # Perception Test: several questions share one video's frames
                continue
            seen.add(real)
            fs = sorted(glob.glob(f"{real}/*.png"))[::stride]
            if len(fs) >= 8:
                yield ds, clip, fs


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/video_stats")
    os.makedirs(out, exist_ok=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 12) as pool:
        res = pool.map(clip_stats, list(jobs()), chunksize=1)
    rows = [r for r, _ in res]
    for name, data in (("per_clip.csv", rows), ("per_pair.csv", [p for _, ps in res for p in ps])):
        with open(f"{out}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in d.items()} for d in data)
    keys = ["motion", "burst25", "moving", "conc10", "dirty", "uni8_dirty", "uni8_redundant", "uni8_under"]
    lines = ["| dataset | clips | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for ds in list(SETS) + ["all"]:
        R = [r for r in rows if ds in ("all", r["dataset"])]
        lines.append(f"| {ds} | {len(R)} | " + " | ".join(f"{np.median([r[k] for r in R]):.3g}" for k in keys) + " |")
    lines.append("\nMedians over clips. motion: % of frame width per second; the rest are shares (0-1).")
    open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
