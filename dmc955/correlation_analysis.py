#!/usr/bin/env python3
"""
Attention Correlation Analysis
Analyzes whether pre-LLM token scores predict LLM attention.
Reads JSON results from correlation analysis runs.

Primary metric: Prefill Attention Retained % (prompt tokens attending to visual tokens)
Secondary metric: Decode-time correlation (autoregressive attention)
"""

import json
import sys
import os
import glob

def load_results(result_dir):
    """Load all correlation JSON results from a directory."""
    results = {}
    for path in sorted(glob.glob(os.path.join(result_dir, "correlation_*.json"))):
        with open(path) as f:
            data = json.load(f)
        video = data.get("video_name", os.path.basename(path).replace("correlation_", "").replace(".json", ""))
        results[video] = data
    return results

def load_eviction_results(result_dir):
    """Load post-prefill eviction benchmark results (baseline + eviction variants)."""
    groups = {}  # video -> {variant: data}
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        fname = os.path.basename(path)
        with open(path) as f:
            data = json.load(f)
        video = data.get("video_name", "")
        if not video:
            continue
        # Determine variant from filename prefix
        if fname.startswith("baseline_"):
            variant = "baseline"
        elif fname.startswith("evict50_"):
            variant = "evict50"
        elif fname.startswith("evict30_"):
            variant = "evict30"
        else:
            continue
        if video not in groups:
            groups[video] = {}
        groups[video][variant] = data
    return groups

def print_eviction_summary(groups):
    """Print post-prefill eviction benchmark summary."""
    if not groups:
        return

    print("=" * 100)
    print("POST-PREFILL KV EVICTION RESULTS")
    print("  Evict lowest-attention visual tokens from KV cache after prefill, before decode")
    print("=" * 100)
    print()

    # Table: Eviction stats
    print("Table: Post-Prefill Eviction — Decode Speed & Attention Retention")
    print("-" * 100)
    print(f"{'Video':<15} {'Variant':<12} {'Tokens':>8} {'Evicted':>8} {'Attn Ret%':>10} "
          f"{'Gen t/s':>8} {'Gen ms':>10} {'Evict ms':>10} {'Total ms':>10}")
    print("-" * 100)

    summary = {"baseline": [], "evict50": [], "evict30": []}

    for video in sorted(groups.keys()):
        variants = groups[video]
        for vname in ["baseline", "evict50", "evict30"]:
            data = variants.get(vname)
            if not data:
                continue

            eviction = data.get("postprefill_eviction", {})
            timing = data.get("timing", {})
            n_gen = data.get("n_tokens_generated", 0)
            gen_ms = timing.get("generation_ms", 0)
            gen_ts = (n_gen / (gen_ms / 1000.0)) if gen_ms > 0 else 0
            total_ms = timing.get("total_ms", 0)

            n_before = eviction.get("n_tokens_before", 0)
            n_evicted = eviction.get("n_tokens_evicted", 0)
            attn_ret = eviction.get("attn_retained_pct", 0)
            evict_ms = eviction.get("eviction_overhead_ms", 0)

            if vname == "baseline":
                n_before = 0
                n_evicted = 0
                attn_ret = 100.0
                evict_ms = 0

            summary[vname].append({
                "gen_ts": gen_ts, "gen_ms": gen_ms, "attn_ret": attn_ret,
                "evict_ms": evict_ms, "total_ms": total_ms,
                "n_before": n_before, "n_evicted": n_evicted,
            })

            print(f"{video:<15} {vname:<12} {n_before:>8} {n_evicted:>8} {attn_ret:>9.1f}% "
                  f"{gen_ts:>8.1f} {gen_ms:>10.0f} {evict_ms:>10.1f} {total_ms:>10.0f}")

    print("-" * 100)

    # Means
    for vname in ["baseline", "evict50", "evict30"]:
        vals = summary[vname]
        if not vals:
            continue
        n = len(vals)
        mean_ts = sum(v["gen_ts"] for v in vals) / n
        mean_gen = sum(v["gen_ms"] for v in vals) / n
        mean_ret = sum(v["attn_ret"] for v in vals) / n
        mean_evict = sum(v["evict_ms"] for v in vals) / n
        mean_total = sum(v["total_ms"] for v in vals) / n
        print(f"{'Mean':<15} {vname:<12} {'':>8} {'':>8} {mean_ret:>9.1f}% "
              f"{mean_ts:>8.1f} {mean_gen:>10.0f} {mean_evict:>10.1f} {mean_total:>10.0f}")

    print()

    # Speedup summary
    baseline_vals = summary.get("baseline", [])
    if baseline_vals:
        mean_baseline_ts = sum(v["gen_ts"] for v in baseline_vals) / len(baseline_vals)
        for vname in ["evict50", "evict30"]:
            vals = summary[vname]
            if vals:
                mean_ts = sum(v["gen_ts"] for v in vals) / len(vals)
                mean_ret = sum(v["attn_ret"] for v in vals) / len(vals)
                speedup = mean_ts / mean_baseline_ts if mean_baseline_ts > 0 else 0
                print(f"  {vname}: {speedup:.2f}x decode speedup, {mean_ret:.1f}% attention retained")
    print()

def print_summary(results):
    """Print correlation analysis summary."""
    print("=" * 90)
    print("ATTENTION CORRELATION ANALYSIS")
    print("Pre-LLM Score vs LLM Attention (Layer 2)")
    print("=" * 90)
    print()

    # ================================================================
    # PRIMARY METRIC: Prefill Attention Retained %
    # ================================================================
    print("=" * 90)
    print("PRIMARY METRIC: Prefill Attention Retained %")
    print("  'If we keep X% of visual tokens (by pre-LLM score),")
    print("   what % of the prompt's total visual attention is retained?'")
    print("=" * 90)
    print()

    print("Table 1: Prefill Attention Retention per Video")
    print("-" * 90)
    print(f"{'Video':<15} {'Keep %':>8} {'Retained %':>12} {'n_tokens':>10} {'Total Attn':>12} {'Kept Attn':>12}")
    print("-" * 90)

    retained_vals = []
    keep_ratio_vals = []

    for video, data in sorted(results.items()):
        corr = data.get("attention_correlation", {})
        prefill = corr.get("prefill", {})
        if not prefill:
            print(f"{video:<15} {'N/A':>8}")
            continue

        retained = prefill.get("attn_retained_pct", 0)
        keep_r = prefill.get("keep_ratio", 0) * 100
        n = prefill.get("n_visual_tokens", 0)
        total = prefill.get("attn_total", 0)
        kept = prefill.get("attn_on_kept", 0)

        retained_vals.append(retained)
        keep_ratio_vals.append(keep_r)

        print(f"{video:<15} {keep_r:>7.1f}% {retained:>11.1f}% {n:>10} {total:>12.4f} {kept:>12.4f}")

    print("-" * 90)
    if retained_vals:
        mean_retained = sum(retained_vals) / len(retained_vals)
        mean_keep = sum(keep_ratio_vals) / len(keep_ratio_vals)
        print(f"{'Mean':<15} {mean_keep:>7.1f}% {mean_retained:>11.1f}%")
    print()

    # Per-frame prefill retention
    print("Table 2: Per-Frame Prefill Attention Retained %")
    print("-" * 90)
    print(f"{'Video':<15} {'Frame 1':>8} {'Frame 2':>8} {'Frame 3':>8} {'Frame 4':>8} "
          f"{'Frame 5':>8} {'Frame 6':>8} {'Frame 7':>8} {'Frame 8':>8} {'Frame 9':>8} {'Frame 10':>8}")
    print("-" * 90)

    all_per_frame_prefill = []
    for video, data in sorted(results.items()):
        corr = data.get("attention_correlation", {})
        prefill = corr.get("prefill", {})
        per_frame = prefill.get("per_frame", [])
        if not per_frame:
            continue

        vals = [pf.get("prefill_attn_retained_pct", 0) for pf in per_frame]
        all_per_frame_prefill.append(vals)
        row = f"{video:<15}"
        for v in vals[:10]:
            row += f" {v:>7.1f}%"
        print(row)

    print("-" * 90)
    if all_per_frame_prefill:
        max_len = max(len(v) for v in all_per_frame_prefill)
        means = []
        for i in range(max_len):
            frame_vals = [v[i] for v in all_per_frame_prefill if i < len(v)]
            means.append(sum(frame_vals) / len(frame_vals))
        row = f"{'Mean':<15}"
        for m in means[:10]:
            row += f" {m:>7.1f}%"
        print(row)
    print()

    # ================================================================
    # SECONDARY METRIC: Decode-time Correlation
    # ================================================================
    print("=" * 90)
    print("SECONDARY METRIC: Decode-time Correlation")
    print("  (Autoregressive generation attention - diluted by text tokens)")
    print("=" * 90)
    print()

    # Table: Global correlation per video
    print("Table 3: Decode-time Global Correlation per Video")
    print("-" * 90)
    print(f"{'Video':<15} {'Pearson r':>10} {'Spearman p':>12} {'Kept Attn':>12} {'Pruned Attn':>12} {'Ratio':>8} {'n_tokens':>10}")
    print("-" * 90)

    pearson_vals = []
    spearman_vals = []
    ratio_vals = []

    for video, data in sorted(results.items()):
        corr = data.get("attention_correlation", {})
        if not corr:
            print(f"{video:<15} {'N/A':>10}")
            continue

        pr = corr.get("pearson_r", 0)
        sr = corr.get("spearman_rho", 0)
        ak = corr.get("avg_attn_kept", 0)
        ap = corr.get("avg_attn_pruned", 0)
        ratio = corr.get("attn_ratio", 0)
        n = corr.get("n_tokens_analyzed", 0)

        pearson_vals.append(pr)
        spearman_vals.append(sr)
        if ratio > 0:
            ratio_vals.append(ratio)

        print(f"{video:<15} {pr:>10.4f} {sr:>12.4f} {ak:>12.6f} {ap:>12.6f} {ratio:>7.2f}x {n:>10}")

    print("-" * 90)
    if pearson_vals:
        print(f"{'Mean':<15} {sum(pearson_vals)/len(pearson_vals):>10.4f} "
              f"{sum(spearman_vals)/len(spearman_vals):>12.4f} "
              f"{'':>12} {'':>12} "
              f"{sum(ratio_vals)/len(ratio_vals):>7.2f}x")
    print()

    # Per-frame Pearson r
    print("Table 4: Per-Frame Pearson r (all videos)")
    print("-" * 90)
    print(f"{'Video':<15} {'Frame 1':>8} {'Frame 2':>8} {'Frame 3':>8} {'Frame 4':>8} "
          f"{'Frame 5':>8} {'Frame 6':>8} {'Frame 7':>8} {'Frame 8':>8} {'Frame 9':>8} {'Frame 10':>8}")
    print("-" * 90)

    all_per_frame = []
    for video, data in sorted(results.items()):
        corr = data.get("attention_correlation", {})
        per_frame = corr.get("per_frame", [])
        if not per_frame:
            continue

        vals = [pf.get("pearson_r", 0) for pf in per_frame]
        all_per_frame.append(vals)
        row = f"{video:<15}"
        for v in vals[:10]:
            row += f" {v:>8.4f}"
        print(row)

    print("-" * 90)
    if all_per_frame:
        max_len = max(len(v) for v in all_per_frame)
        means = []
        for i in range(max_len):
            frame_vals = [v[i] for v in all_per_frame if i < len(v)]
            means.append(sum(frame_vals) / len(frame_vals))
        row = f"{'Mean':<15}"
        for m in means[:10]:
            row += f" {m:>8.4f}"
        print(row)
    print()

    # Token counts
    print("Table 5: Token Counts")
    print("-" * 60)
    print(f"{'Video':<15} {'Total':>8} {'Kept':>8} {'Pruned':>8} {'Keep %':>8}")
    print("-" * 60)
    for video, data in sorted(results.items()):
        corr = data.get("attention_correlation", {})
        n = corr.get("n_tokens_analyzed", 0)
        nk = corr.get("n_kept", 0)
        np_ = corr.get("n_pruned", 0)
        pct = (nk / n * 100) if n > 0 else 0
        print(f"{video:<15} {n:>8} {nk:>8} {np_:>8} {pct:>7.1f}%")
    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Prefill summary (primary)
    if retained_vals:
        mean_retained = sum(retained_vals) / len(retained_vals)
        mean_keep = sum(keep_ratio_vals) / len(keep_ratio_vals)
        print()
        print(f"  PRIMARY: Keeping {mean_keep:.0f}% of tokens retains {mean_retained:.1f}% of prompt visual attention")
        print(f"           ({len(retained_vals)} videos analyzed)")
        print()

        if mean_retained > 90:
            print("  Verdict: EXCELLENT — pruning removes low-attention tokens while")
            print("           preserving >90% of the prompt's visual attention budget.")
        elif mean_retained > 75:
            print("  Verdict: GOOD — pruning preserves most of the prompt's visual")
            print("           attention, with modest loss on pruned tokens.")
        elif mean_retained > 60:
            print("  Verdict: MODERATE — noticeable attention loss from pruning,")
            print("           but majority of visual attention is still retained.")
        else:
            print("  Verdict: WEAK — significant attention loss from pruning.")
            print("           The scorer may not be well-aligned with LLM attention.")
        print()

    # Decode summary (secondary)
    if pearson_vals:
        mean_p = sum(pearson_vals) / len(pearson_vals)
        mean_s = sum(spearman_vals) / len(spearman_vals)
        mean_r = sum(ratio_vals) / len(ratio_vals) if ratio_vals else 0

        pos_p = sum(1 for v in pearson_vals if v > 0)
        pos_s = sum(1 for v in spearman_vals if v > 0)
        pos_r = sum(1 for v in ratio_vals if v > 1.0)

        print(f"  SECONDARY (decode-time):")
        print(f"    Mean Pearson r:      {mean_p:.4f} ({pos_p}/{len(pearson_vals)} videos positive)")
        print(f"    Mean Spearman p:     {mean_s:.4f} ({pos_s}/{len(spearman_vals)} videos positive)")
        print(f"    Mean Attention Ratio: {mean_r:.3f}x ({pos_r}/{len(ratio_vals)} videos > 1.0)")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 correlation_analysis.py <result_dir>")
        print("  result_dir: directory containing correlation_*.json or eviction benchmark files")
        sys.exit(1)

    result_dir = sys.argv[1]

    # Try correlation results
    results = load_results(result_dir)
    if results:
        print(f"Loaded {len(results)} correlation results from {result_dir}")
        print()
        print_summary(results)

    # Try eviction benchmark results
    eviction_groups = load_eviction_results(result_dir)
    if eviction_groups:
        print(f"Loaded eviction results for {len(eviction_groups)} videos from {result_dir}")
        print()
        print_eviction_summary(eviction_groups)

    if not results and not eviction_groups:
        print(f"No results found in {result_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()
