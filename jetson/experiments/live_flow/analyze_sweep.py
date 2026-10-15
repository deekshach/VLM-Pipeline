#!/usr/bin/env python3
"""Summarise run_sweep.sh: TTFT and its breakdown per configuration, frame selection vs ground truth,
and a timeline figure of one low-motion and one high-motion clip.

    python3 analyze_sweep.py [results_dir]
"""
import glob, json, os, statistics as st, sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONFIGS = ["gt_files", "live_serial", "live_pipelined", "live_anchor"]
LABEL = {"gt_files": "GT flow files", "live_serial": "live OFA, serial", "live_pipelined": "live OFA, pipelined",
         "live_anchor": "live OFA, pipelined + anchor-first"}

data = {}
for p in glob.glob(f"{R}/*.json"):
    base = os.path.basename(p)[:-5]
    for c in CONFIGS:
        if base.endswith("_" + c):
            data.setdefault(base[: -len(c) - 1], {})[c] = json.load(open(p))
videos = sorted(v for v in data if all(c in data[v] for c in CONFIGS))


def frames(j):
    return [f["index"] for f in j["selected_frames"]]


def bd(j):
    return j["timing"]["breakdown"]


def gpu_ms(j):
    b = bd(j)
    g = b["rest"]["vit_ms"] + b["rest"]["prefill_img_ms"] + b["rest"]["prefill_txt_ms"]
    if "anchor" in b:
        a = b["anchor"]
        g += a["vit_ms"] + a["prefill_img_ms"] + a["prefill_txt_ms"]
    return g


print(f"{len(videos)} clips with all {len(CONFIGS)} configurations\n")
print("| configuration | mean frames | mean TTFT ms | flow / analysis ms (critical path) | ViT ms | image prefill ms | text prefill ms | other ms |")
print("|---|---|---|---|---|---|---|---|")
for c in CONFIGS:
    J = [data[v][c] for v in videos]
    ttft = st.mean(j["timing"]["ttft_ms"] for j in J)
    nfr = st.mean(len(frames(j)) for j in J)
    vit = st.mean(bd(j)["rest"]["vit_ms"] + bd(j).get("anchor", {}).get("vit_ms", 0) for j in J)
    pimg = st.mean(bd(j)["rest"]["prefill_img_ms"] + bd(j).get("anchor", {}).get("prefill_img_ms", 0) for j in J)
    ptxt = st.mean(bd(j)["rest"]["prefill_txt_ms"] + bd(j).get("anchor", {}).get("prefill_txt_ms", 0) for j in J)
    if c == "live_anchor":   # analysis overlaps the anchor; what is exposed is the wait for the later of the two
        crit = st.mean(max(bd(j)["analysis_ms"], bd(j)["anchor"]["done_at_ms"]) - bd(j)["anchor"]["total_ms"] for j in J)
        crit_s = f"{st.mean(bd(j)['analysis_ms'] for j in J):.0f} (exposed {crit:.0f})"
    else:
        crit = st.mean(bd(j)["analysis_ms"] for j in J)
        crit_s = f"{crit:.0f}"
    other = ttft - crit - vit - pimg - ptxt
    print(f"| {LABEL[c]} | {nfr:.2f} | {ttft:.0f} | {crit_s} | {vit:.0f} | {pimg:.0f} | {ptxt:.0f} | {other:.0f} |")

# live flow stage costs (serial) and the pipelined wall time
S = [bd(data[v]["live_serial"])["live_flow"] for v in videos]
Pp = [bd(data[v]["live_pipelined"])["live_flow"] for v in videos]
A = [bd(data[v]["live_anchor"])["live_flow"] for v in videos]
print("\nLive flow per clip (mean over clips; serial = each stage timed alone):")
for k in ["decode_sum_ms", "prep_ms", "ofa_ms", "readback_ms", "stats_ms", "wall_ms"]:
    print(f"  serial {k:14s} {st.mean(s[k] for s in S):7.0f} ms")
print(f"  pipelined wall          {st.mean(p['wall_ms'] for p in Pp):7.0f} ms  (decode wall {st.mean(p['decode_wall_ms'] for p in Pp):.0f} ms)")
print(f"  pipelined + anchor wall {st.mean(p['wall_ms'] for p in A):7.0f} ms")
print(f"  pairs per clip          {st.mean(s['n_pairs'] for s in S):7.1f}")

# frame selection agreement with ground truth (live flow without anchor; anchor changes the picks by design)
same_n = sum(len(frames(data[v]["gt_files"])) == len(frames(data[v]["live_pipelined"])) for v in videos)
ident = sum(len(set(frames(data[v]["gt_files"])) & set(frames(data[v]["live_pipelined"]))) for v in videos)
tot = sum(len(frames(data[v]["live_pipelined"])) for v in videos)
consistent = sum(frames(data[v]["live_serial"]) == frames(data[v]["live_pipelined"]) for v in videos)
print(f"\nSelection, live OFA vs GT: same count {same_n}/{len(videos)}, identical frames {ident}/{tot}; "
      f"serial and pipelined pick the same frames on {consistent}/{len(videos)} clips")

print("\nPer clip TTFT ms (frames):")
print("| clip | " + " | ".join(LABEL[c] for c in CONFIGS) + " |")
print("|---|" + "---|" * len(CONFIGS))
for v in videos:
    print(f"| {v} | " + " | ".join(f"{data[v][c]['timing']['ttft_ms']:.0f} ({len(frames(data[v][c]))})" for c in CONFIGS) + " |")

# timeline figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing: no figure")
    sys.exit(0)

low = min(videos, key=lambda v: len(frames(data[v]["live_pipelined"])))
high = max(videos, key=lambda v: len(frames(data[v]["live_pipelined"])))
COL = {"flow": "#8c8c8c", "select": "#c7c7c7", "vit": "#1f77b4", "pimg": "#ff7f0e", "ptxt": "#ffbb78",
       "load": "#d9d9d9", "wait": "#ffffff"}
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
for ax, v in zip(axes, [low, high]):
    rows = []
    for c in CONFIGS:
        j = data[v][c]; b = bd(j)
        segs_cpu, segs_gpu = [], []
        segs_cpu.append((0, b["flow_ms"], "flow"))
        segs_cpu.append((b["flow_ms"], b["analysis_ms"] - b["flow_ms"], "select"))
        t = 0.0
        if "anchor" in b:
            a = b["anchor"]
            for k, key in (("vit", "vit_ms"), ("pimg", "prefill_img_ms"), ("ptxt", "prefill_txt_ms")):
                segs_gpu.append((t, a[key], k)); t += a[key]
            t = max(a["done_at_ms"], b["analysis_ms"])
        else:
            t = b["analysis_ms"]
        t += b["rest_image_load_ms"] + b["rest_tokenize_ms"]
        r = b["rest"]
        n = max(1, r["n_images"])
        for _ in range(n):   # per-image average, interleaved as the pipeline runs them
            segs_gpu.append((t, r["vit_ms"] / n, "vit")); t += r["vit_ms"] / n
            segs_gpu.append((t, r["prefill_img_ms"] / n, "pimg")); t += r["prefill_img_ms"] / n
        segs_gpu.append((t, r["prefill_txt_ms"], "ptxt"))
        rows.append((LABEL[c], segs_cpu, segs_gpu, j["timing"]["ttft_ms"]))
    for i, (lab, cpu, gpu, ttft) in enumerate(reversed(rows)):
        y = i * 1.0
        for x0, w, k in cpu:
            ax.barh(y + 0.18, w, left=x0, height=0.3, color=COL[k], edgecolor="none")
        for x0, w, k in gpu:
            ax.barh(y - 0.18, w, left=x0, height=0.3, color=COL[k], edgecolor="none")
        ax.plot([ttft, ttft], [y - 0.4, y + 0.4], color="k", lw=1.2)
        ax.text(ttft + 60, y, f"{ttft:.0f} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8)
    ax.set_title(f"{v}: {len(frames(data[v]['live_pipelined']))} frames selected (live OFA)", fontsize=10)
    ax.set_xlabel("ms from request start (upper bar: CPU / OFA analysis, lower bar: GPU encode)", fontsize=8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL[k], label=l) for k, l in [("flow", "flow + motion stats"), ("select", "selection + spatial"),
           ("vit", "ViT"), ("pimg", "image prefill"), ("ptxt", "text prefill")]], loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.suptitle("TTFT timeline, Jetson AGX Orin, ViT flash attention off (black line = first token ready)", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
out = os.path.join(R, "fig_ttft_timeline.png")
fig.savefig(out, dpi=130)
print(f"\nfigure: {out}")
