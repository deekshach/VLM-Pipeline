#!/usr/bin/env python3
"""Compute Sintel optical flow with the Orin's OFA engine (VPI) and write it as .flo files.

Output mirrors the dataset layout the pipeline reads (<out>/training/flow/<video>/frame_XXXX.flo,
flow from frame XXXX to XXXX+1), at the OFA grid resolution (1/4 per side with --grid 4), in
full-resolution pixel units. Also reports accuracy against Sintel's ground truth, with the
ground truth averaged over the same 4x4 blocks.

    /usr/bin/python3 ofa_flow_sintel.py <sintel_dir> <out_dir> [--levels 5] [--grid 4] [--quality high]
"""
import argparse, glob, json, os, statistics as st, struct, time
import numpy as np
import cv2
import vpi

ap = argparse.ArgumentParser()
ap.add_argument("sintel"); ap.add_argument("out")
ap.add_argument("--levels", type=int, default=5)
ap.add_argument("--grid", type=int, default=4)
ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
ap.add_argument("--videos", default="")
ap.add_argument("--no-write", action="store_true")
a = ap.parse_args()
Q = {"low": vpi.OptFlowQuality.LOW, "medium": vpi.OptFlowQuality.MEDIUM, "high": vpi.OptFlowQuality.HIGH}[a.quality]


def read_flo(p):
    with open(p, "rb") as f:
        assert struct.unpack("f", f.read(4))[0] == 202021.25
        w, h = struct.unpack("ii", f.read(8))
        return np.fromfile(f, np.float32, w * h * 2).reshape(h, w, 2)


def write_flo(p, fl):
    with open(p, "wb") as f:
        f.write(struct.pack("f", 202021.25)); f.write(struct.pack("ii", fl.shape[1], fl.shape[0]))
        fl.astype(np.float32).tofile(f)


def prep(bgr):
    return (vpi.asimage(bgr, vpi.Format.BGR8)
            .convert(vpi.Format.Y8_ER, backend=vpi.Backend.CUDA)
            .gaussian_pyramid(a.levels, backend=vpi.Backend.CUDA)
            .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC))


def block_mean(gt, g, h, w):
    H, W = h * g, w * g
    pad = np.zeros((H, W, 2), np.float32); pad[:gt.shape[0], :gt.shape[1]] = gt
    cnt = np.zeros((H, W, 1), np.float32); cnt[:gt.shape[0], :gt.shape[1]] = 1
    s = pad.reshape(h, g, w, g, 2).sum((1, 3)); c = cnt.reshape(h, g, w, g, 1).sum((1, 3))
    return s / np.maximum(c, 1)


videos = a.videos.split(",") if a.videos else sorted(os.listdir(f"{a.sintel}/training/clean"))
summary = {"levels": a.levels, "grid": a.grid, "quality": a.quality, "videos": {}}
all_t, all_epe = [], []
for v in videos:
    frames = sorted(glob.glob(f"{a.sintel}/training/clean/{v}/frame_*.png"))
    od = f"{a.out}/training/flow/{v}"
    os.makedirs(od, exist_ok=True)
    epe, mag_o, mag_g, ts = [], [], [], []
    prev = prep(cv2.imread(frames[0]))
    for i in range(1, len(frames)):
        t0 = time.perf_counter()
        cur = prep(cv2.imread(frames[i]))
        with vpi.Backend.OFA:
            mv = vpi.optflow_dense(prev, cur, quality=Q, gridsize=a.grid)
        with mv.rlock_cpu() as d:
            fl = np.float32(d) / 32.0          # S10.5 fixed point -> pixels
        ts.append((time.perf_counter() - t0) * 1e3)
        prev = cur
        name = f"frame_{i:04d}.flo"            # flow from frame i to i+1, as in Sintel
        if not a.no_write:
            write_flo(f"{od}/{name}", fl)
        gtp = f"{a.sintel}/training/flow/{v}/{name}"
        if os.path.exists(gtp):
            gt = block_mean(read_flo(gtp), a.grid, fl.shape[0], fl.shape[1])
            epe.append(float(np.linalg.norm(fl - gt, axis=2).mean()))
            mag_o.append(float(np.linalg.norm(fl, axis=2).mean())); mag_g.append(float(np.linalg.norm(gt, axis=2).mean()))
    r = float(np.corrcoef(mag_o, mag_g)[0, 1]) if len(mag_o) > 2 and np.std(mag_o) > 0 and np.std(mag_g) > 0 else float("nan")
    summary["videos"][v] = dict(pairs=len(ts), ms_per_pair=st.median(ts), epe=st.mean(epe), mean_mag_ofa=st.mean(mag_o),
                                mean_mag_gt=st.mean(mag_g), per_frame_mag_corr=r)
    all_t += ts; all_epe += epe
    print(f"{v:12s} {len(ts):3d} pairs  {st.median(ts):5.1f} ms/pair  EPE {st.mean(epe):6.2f} px  "
          f"mean |flow| OFA {st.mean(mag_o):6.2f} vs GT {st.mean(mag_g):6.2f}  per-frame corr {r:5.2f}", flush=True)
summary["overall"] = dict(pairs=len(all_t), ms_per_pair=st.median(all_t), epe=st.mean(all_epe))
print(f"overall: {len(all_t)} pairs, median {st.median(all_t):.1f} ms/pair (incl. PNG decode), mean EPE {st.mean(all_epe):.2f} px")
os.makedirs(a.out, exist_ok=True)
json.dump(summary, open(f"{a.out}/ofa_flow_summary_L{a.levels}.json", "w"), indent=1)
