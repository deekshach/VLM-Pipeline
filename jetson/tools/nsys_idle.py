#!/usr/bin/env python3
"""GPU busy vs idle time per pipeline stage of a llama-jetson-vlm nsys trace (scripts/profile_nsys.sh).

    python3 nsys_idle.py trace.sqlite

Per stage (all NVTX ranges of that name): wall time, GPU busy time (union of the kernel intervals inside the
ranges), idle = wall - busy, kernel count, mean kernel duration, and the gaps between consecutive kernels (median,
p90, and the share of idle time in gaps over 1 ms -- host-side work rather than launch latency).
"""
import collections, sqlite3, sys
import numpy as np

STAGES = ["vit", "prefill_img", "prefill_txt", "decode"]
db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("select id, value from StringIds"))
ranges = collections.defaultdict(list)
for text, tid, s, e in db.execute("select text, textId, start, end from NVTX_EVENTS where end is not null"):
    name = text if text else S.get(tid)
    if name in STAGES:
        ranges[name].append((s, e))
ks = sorted(db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
K = np.array(ks, dtype=np.int64)
print("| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |")
print("|---|---|---|---|---|---|---|---|---|")
for st in STAGES:
    wall = busy = 0.0
    n = 0
    gaps = []
    for s, e in ranges[st]:
        wall += (e - s) / 1e6
        sel = K[(K[:, 0] >= s) & (K[:, 1] <= e)]
        if not len(sel):
            continue
        n += len(sel)
        cur_s, cur_e = sel[0]
        for a, b in sel[1:]:   # union of intervals; gaps between them
            if a > cur_e:
                busy += (cur_e - cur_s) / 1e6
                gaps.append((a - cur_e) / 1e3)
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        busy += (cur_e - cur_s) / 1e6
    if not n:
        continue
    g = np.array(gaps) if gaps else np.zeros(1)
    idle = wall - busy
    big = g[g > 1000].sum() / 1e3
    print(f"| {st} | {wall:.0f} | {busy:.0f} | {idle / wall * 100:.1f}% | {n} | {busy * 1e3 / n:.0f} | {np.median(g):.1f} | "
          f"{np.percentile(g, 90):.1f} | {big / max(idle, 1e-9) * 100:.0f}% |")
