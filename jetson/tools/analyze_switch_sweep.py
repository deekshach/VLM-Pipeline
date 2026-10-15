#!/usr/bin/env python3
"""Summarise jetson/scripts/run_switch_sweep.sh: mean TTFT and its parts per switch configuration,
the change against the plain adaptive pipeline, and TTFT per clip.

    python3 analyze_switch_sweep.py [sweep_dir]
"""
import collections, glob, json, os, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "switch_sweep")
ORDER = ["uniform", "uniform_all", "adaptive", "adaptive_align", "adaptive_wcache", "adaptive_winbatch",
         "adaptive_f32out", "adaptive_f32out_bias", "adaptive_gemmtable", "adaptive_all", "adaptive_all_anchor"]
LABEL = {
    "uniform": "uniform 8 frames, no optimizations",
    "uniform_all": "uniform 8 frames + window cache, window batch, GEMM switches",
    "adaptive": "adaptive + live OFA flow (reference)",
    "adaptive_align": "+ window align only",
    "adaptive_wcache": "+ window cache (threshold 24)",
    "adaptive_winbatch": "+ window align + ViT window batch",
    "adaptive_f32out": "+ GEMM F32 output",
    "adaptive_f32out_bias": "+ GEMM F32 output + bias fusion",
    "adaptive_gemmtable": "+ tuned GEMM table",
    "adaptive_all": "+ all of the above",
    "adaptive_all_anchor": "+ all of the above + anchor-first",
}

runs = collections.defaultdict(dict)
for p in glob.glob(f"{D}/*.json"):
    clip, cfg = os.path.basename(p)[:-5].split("__")
    runs[cfg][clip] = json.load(open(p))
clips = sorted(set.intersection(*(set(runs[c]) for c in ORDER if c in runs)))


def parts(j):
    b = j["timing"]["breakdown"]
    r, a = b["rest"], b.get("anchor", {})
    g = lambda k: r.get(k, 0) + a.get(k, 0)
    return dict(ttft=j["timing"]["ttft_ms"], analysis=b["analysis_ms"] if j["mode"] == "adaptive" else 0.0,
                vit=g("vit_ms"), pimg=g("prefill_img_ms"), ptxt=g("prefill_txt_ms"),
                frames=len(j["selected_frames"]), tok=g("tokens_encoded"),
                crop=g("n_crop"), skip=g("n_skip"), wc=g("wc_diff_ms") + g("wc_prep_ms"))


print(f"{len(clips)} clips: {', '.join(clips)}\n")
print("| configuration | mean TTFT ms | vs reference | analysis | ViT | image prefill | text prefill | frames | ViT tokens | crops / skips |")
print("|---|---|---|---|---|---|---|---|---|---|")
ref = st.mean(parts(runs["adaptive"][c])["ttft"] for c in clips) if "adaptive" in runs else None
for cfg in ORDER:
    if cfg not in runs:
        continue
    P = [parts(runs[cfg][c]) for c in clips]
    m = {k: st.mean(p[k] for p in P) for k in P[0]}
    vs = f"{(m['ttft'] / ref - 1) * 100:+.0f}%" if ref else ""
    print(f"| {LABEL[cfg]} | {m['ttft']:.0f} | {vs} | {m['analysis']:.0f} | {m['vit']:.0f} | {m['pimg']:.0f} | {m['ptxt']:.0f} | "
          f"{m['frames']:.1f} | {m['tok']:.0f} | {sum(p['crop'] for p in P)} / {sum(p['skip'] for p in P)} |")

print("\nTTFT per clip (ms):")
print("| configuration | " + " | ".join(clips) + " |")
print("|---|" + "---|" * len(clips))
for cfg in ORDER:
    if cfg in runs:
        print(f"| {cfg} | " + " | ".join(f"{runs[cfg][c]['timing']['ttft_ms']:.0f}" for c in clips) + " |")
