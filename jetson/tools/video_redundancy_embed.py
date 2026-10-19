#!/usr/bin/env python3
"""Temporal redundancy in the model's own view: how alike Qwen2.5-VL's ViT finds consecutive 2 fps frames.

    python video_redundancy_embed.py [out_dir=experiments/video_redundancy]

Frames: the Qwen 2 fps samples (video_redundancy.py, FRAMES_DIR) embedded by scripts/embed_qwen2fps.sh (a block of
<= 40 consecutive samples per clip); MSVD and Sintel from the accuracy study's dense references (native), taken at a
0.5 s step (MSVD: 4 fps extraction, longer clips spread over 50 frames; Sintel: 24 fps).
Each frame = the mean of its merged visual tokens (what the LLM receives), centred on the mean over all frames of
all clips (the embeddings share a large common direction, which would make any two frames look alike) and
normalised. Measured per clip:
  next     cosine between consecutive samples (0.5 s apart)
  far      cosine between the clip's first and last sample (same scene, the most time apart)
and, as the floor, random pairs of frames from different clips. Writes embed_per_clip.csv and embed_pairs.csv
(every consecutive pair; novelty = 1 - cosine) and embed_summary.md.
"""
import csv, json, os, sys
import numpy as np

QWEN = "/mnt/external_ssd/VLM-datasets/qwen2fps/embed"
DENSE = "/home/nvidia/data/accuracy/dense"
DATASETS = ["msvd", "sintel", "charades", "nextqa", "videomme", "perception", "tacos"]


def pooled(d, step=1):
    m = json.load(open(f"{d}/meta.json"))
    fr = m["frames"][::step]
    return np.stack([np.fromfile(f"{d}/{f['frame']}.f32", dtype=np.float32).reshape(f["nx"] * f["ny"], -1).mean(0) for f in fr])


def clips():
    for ds in DATASETS:
        if ds in ("msvd", "sintel"):
            for c in sorted(os.listdir(f"{DENSE}/{ds}")):
                if ds == "sintel":
                    step = 12
                else:
                    n = len(json.load(open(f"{DENSE}/{ds}/{c}/native/meta.json"))["frames"])
                    a, b = c.rsplit("_", 2)[1:]
                    step = max(1, round(0.5 * n / max(1, int(b) - int(a))))   # samples 0.5 s apart
                P = pooled(f"{DENSE}/{ds}/{c}/native", step)
                if len(P) >= 2:
                    yield ds, c, P
        elif os.path.isdir(f"{QWEN}/{ds}"):
            for c in sorted(os.listdir(f"{QWEN}/{ds}")):
                try:
                    P = pooled(f"{QWEN}/{ds}/{c}")
                except (OSError, ValueError):   # not embedded (yet)
                    continue
                yield ds, c, P


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/video_redundancy")
    os.makedirs(out, exist_ok=True)
    data = list(clips())
    mu = np.concatenate([P for _, _, P in data]).mean(0)
    unit = lambda X: (X - mu) / np.linalg.norm(X - mu, axis=-1, keepdims=True)
    rows, pairs = [], []
    for ds, c, P in data:
        X = unit(P)
        nxt = (X[1:] * X[:-1]).sum(1)
        rows.append(dict(dataset=ds, clip=c, frames=len(X), next=float(np.median(nxt)), next_min=float(nxt.min()), far=float(X[0] @ X[-1])))
        pairs += [dict(dataset=ds, clip=c, t=t + 1, cos=float(v), novelty=float(1 - v)) for t, v in enumerate(nxt)]
    rng = np.random.default_rng(0)
    cross = []
    while len(cross) < 5000:
        i, j = rng.choice(len(data), 2, replace=False)
        A, B = data[i][2], data[j][2]
        cross.append(float(unit(A[rng.integers(len(A))]) @ unit(B[rng.integers(len(B))])))
    for name, d in (("embed_per_clip.csv", rows), ("embed_pairs.csv", pairs)):
        with open(f"{out}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(d[0]))
            w.writeheader()
            w.writerows({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in r.items()} for r in d)
    q = lambda v: " / ".join(f"{x:.3f}" for x in np.percentile(v, [25, 50, 75]))
    lines = ["| dataset | clips | pairs | next (pairs, p25 / p50 / p75) | far (clips) |", "|---|---|---|---|---|"]
    for ds in DATASETS + ["all"]:
        R = [r for r in rows if ds in ("all", r["dataset"])]
        Pp = [p["cos"] for p in pairs if ds in ("all", p["dataset"])]
        if R:
            lines.append(f"| {ds} | {len(R)} | {len(Pp)} | {q(Pp)} | {q([r['far'] for r in R])} |")
    lines.append(f"| different clips | | {len(cross)} | {q(cross)} | |")
    lines.append("\nCentred cosine of mean-pooled merged visual tokens; next = consecutive 2 fps samples, far = first vs last sample.")
    open(f"{out}/embed_summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
