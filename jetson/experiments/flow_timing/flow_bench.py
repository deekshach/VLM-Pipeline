#!/usr/bin/env python3
"""Time optical flow between consecutive Sintel frames on the Orin.

CPU (OpenCV): Farneback with the parameters of generation-only/msvd_prepare.py, and DIS.
OFA (VPI):    the Orin's optical-flow hardware engine, with the CUDA/VIC pre-processing VPI needs.

    /usr/bin/python3 flow_bench.py <frame_dir> [pairs]
"""
import glob, statistics as st, sys, time
import cv2
import numpy as np

frames = sorted(glob.glob(sys.argv[1] + "/frame_*.png"))
pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
bgr = [cv2.imread(f) for f in frames[:pairs + 1]]
gray = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in bgr]
print(f"{len(gray) - 1} pairs of {gray[0].shape[1]}x{gray[0].shape[0]} px")


def bench(name, fn, warm=1):
    for i in range(warm):
        fn(0)
    t = []
    for i in range(len(gray) - 1):
        t0 = time.perf_counter(); fn(i); t.append((time.perf_counter() - t0) * 1e3)
    print(f"{name:52s} median {st.median(t):8.1f} ms/pair   min {min(t):8.1f}")


def farneback(i):
    cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None, pyr_scale=0.5, levels=3, winsize=15,
                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


for threads in (cv2.getNumberOfCPUs(), 1):
    cv2.setNumThreads(threads)
    bench(f"CPU Farneback (msvd_prepare params), {threads} threads", farneback)
    for preset, pname in ((cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "ultrafast"), (cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, "medium")):
        dis = cv2.DISOpticalFlow_create(preset)
        bench(f"CPU DIS {pname}, {threads} threads", lambda i: dis.calc(gray[i], gray[i + 1], None))
cv2.setNumThreads(cv2.getNumberOfCPUs())

import vpi

LEVELS = 3   # pyramid levels, as in VPI's sample


def prep(b):
    return (vpi.asimage(b, vpi.Format.BGR8)
            .convert(vpi.Format.Y8_ER, backend=vpi.Backend.CUDA)
            .gaussian_pyramid(LEVELS, backend=vpi.Backend.CUDA)
            .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC))


for quality, qname in ((vpi.OptFlowQuality.LOW, "low"), (vpi.OptFlowQuality.MEDIUM, "medium"), (vpi.OptFlowQuality.HIGH, "high")):
    for grid in (4, 1):
        def ofa(i, quality=quality, grid=grid, include_prep=True):
            a, b = prep(bgr[i]), prep(bgr[i + 1])
            with vpi.Backend.OFA:
                mv = vpi.optflow_dense(a, b, quality=quality, gridsize=grid)
            with mv.rlock_cpu() as d:
                np.float32(d) / 32.0
        bench(f"OFA (VPI) quality={qname} grid={grid} incl. prep+readback", ofa, warm=2)

pre = [prep(b) for b in bgr]
vpi.Stream.default.sync()
for quality, qname in ((vpi.OptFlowQuality.LOW, "low"), (vpi.OptFlowQuality.HIGH, "high")):
    def ofa_only(i, quality=quality):
        with vpi.Backend.OFA:
            mv = vpi.optflow_dense(pre[i], pre[i + 1], quality=quality, gridsize=4)
        vpi.Stream.default.sync()
    bench(f"OFA (VPI) quality={qname} grid=4, flow only", ofa_only, warm=2)
