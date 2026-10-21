#!/usr/bin/env python3
"""Frames per clip handed to the LLM: uniform baseline vs Qwen2.5-VL's own 2 fps sampling vs the design.

    python3 frame_counts.py [out_dir=jetson/experiments/video_redundancy]

baseline  --baseline --base-frames 8: 8 frames spread evenly over the clip's frame pool (runs/<ds>/base)
qwen_2fps qwen-vl-utils smart_nframes on the source video (2 fps, even, >= 4): 'sampled' in per_clip.csv of
          video_redundancy.py (Perception Test: per video; its questions share the video)
ours      frames the motion-based selection keeps (--live-flow --anchor-first; the window cache does not change
          the selection, so any config with it gives the same count): runs/<ds>/pw24, else ours_nocache
pool      frames the pipeline selects from (the prepared clip, at most 50)
Covers the 123 clips of the main evaluation (MSVD: the 45-clip set). Writes frame_counts.csv and frame_counts.md.
"""
import csv, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../experiments/video_redundancy")
ACC = "/home/nvidia/data/accuracy"
RUNS = f"{ACC}/runs"
DATASETS = ["sintel", "msvd", "charades", "nextqa", "videomme", "perception", "tacos"]

qwen = {}
for r in csv.DictReader(open(os.path.join(OUT, "per_clip.csv"))):
    clip = r["clip"].rsplit("_q", 1)[0] if r["dataset"] == "perception" else r["clip"]
    qwen[r["dataset"], clip] = int(r["sampled"])


def frames(ds, cfg, clip):
    f = f"{RUNS}/{ds}/{cfg}/{clip}.json"
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    return j["n_frames_selected"], j["n_frames_total"]


rows = []
for ds in DATASETS:
    if ds == "msvd":
        clips = open(f"{ACC}/msvd_clips_45.txt").read().strip().split(",")
    else:
        root = "/home/nvidia/data/MPI-Sintel" if ds == "sintel" else f"/mnt/external_ssd/VLM-datasets/eval/{ds}"
        clips = sorted(os.listdir(f"{root}/training/clean"))
    for v in clips:
        ours, pool = next((x for x in (frames(ds, c, v) for c in ("pw24", "ours_nocache")) if x[0]), (None, None))
        q = qwen.get((ds, v.rsplit("_q", 1)[0] if ds == "perception" else v))
        rows.append(dict(dataset=ds, clip=v, pool=pool, baseline=8, qwen_2fps=q, ours=ours))

with open(os.path.join(OUT, "frame_counts.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

md = ["| dataset | clips | baseline | Qwen 2 fps: median (range) | ours: median (range) | ours / baseline | ours / Qwen 2 fps |",
      "|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    r = [x for x in rows if x["dataset"] == ds]
    q, o = [x["qwen_2fps"] for x in r], [x["ours"] for x in r]
    md.append(f"| {ds} | {len(r)} | 8 | {st.median(q):g} ({min(q)}-{max(q)}) | {st.median(o):g} ({min(o)}-{max(o)}) | "
              f"{sum(o) / (8 * len(r)):.0%} | {sum(o) / sum(q):.0%} |")
q, o = [x["qwen_2fps"] for x in rows], [x["ours"] for x in rows]
md.append(f"| all | {len(rows)} | 8 | {st.median(q):g} ({min(q)}-{max(q)}) | {st.median(o):g} ({min(o)}-{max(o)}) | "
          f"{sum(o) / (8 * len(rows)):.0%} | {sum(o) / sum(q):.0%} |")
md += ["", "| dataset | clip | pool | baseline | Qwen 2 fps | ours | ours / baseline | ours / Qwen 2 fps |",
       "|---|---|---|---|---|---|---|---|"]
md += [f"| {x['dataset']} | {x['clip']} | {x['pool']} | 8 | {x['qwen_2fps']} | {x['ours']} | {x['ours'] / 8:.0%} | "
       f"{x['ours'] / x['qwen_2fps']:.0%} |" for x in rows]
open(os.path.join(OUT, "frame_counts.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
