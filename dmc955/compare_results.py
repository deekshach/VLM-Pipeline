#!/usr/bin/env python3
"""
Compare adaptive vs baseline results across all MPI-Sintel videos.
Reads JSON result files and produces a comprehensive comparison table.
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def load_result(path):
    """Load a JSON result file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None

def truncate_text(text, max_len=80):
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Find all video names from result files
    videos = set()
    for f in RESULTS_DIR.glob("*_baseline.json"):
        videos.add(f.stem.replace("_baseline", ""))
    for f in RESULTS_DIR.glob("*_adaptive.json"):
        videos.add(f.stem.replace("_adaptive", ""))
    videos = sorted(videos)

    if not videos:
        print("No result files found.")
        sys.exit(1)

    # Collect data
    rows = []
    for video in videos:
        bl = load_result(RESULTS_DIR / f"{video}_baseline.json")
        ad = load_result(RESULTS_DIR / f"{video}_adaptive.json")
        if bl and ad:
            rows.append((video, bl, ad))

    if not rows:
        print("No matched baseline/adaptive pairs found.")
        sys.exit(1)

    # ===== TABLE 1: Frame Selection & Latency Comparison =====
    print("=" * 140)
    print("TABLE 1: Frame Selection & Latency Breakdown")
    print("=" * 140)
    header = f"{'Video':<16} {'Total':>5} | {'BL Frm':>6} {'AD Frm':>6} {'Reduc':>6} | {'BL Encode':>10} {'AD Encode':>10} {'Speedup':>8} | {'BL GenTm':>9} {'AD GenTm':>9} | {'BL Total':>9} {'AD Total':>9} {'Speedup':>8}"
    print(header)
    print("-" * 140)

    sum_bl_frames = 0
    sum_ad_frames = 0
    sum_bl_encode = 0
    sum_ad_encode = 0
    sum_bl_gen = 0
    sum_ad_gen = 0
    sum_bl_total = 0
    sum_ad_total = 0

    for video, bl, ad in rows:
        n_total = bl.get("n_frames_total", 0)
        bl_frames = bl.get("n_frames_selected", 0)
        ad_frames = ad.get("n_frames_selected", 0)
        frame_reduction = (1.0 - ad_frames / bl_frames) * 100 if bl_frames > 0 else 0

        bl_encode = bl["timing"]["encode_ms"]
        ad_encode = ad["timing"]["encode_ms"]
        encode_speedup = bl_encode / ad_encode if ad_encode > 0 else 0

        bl_gen = bl["timing"]["generation_ms"]
        ad_gen = ad["timing"]["generation_ms"]

        bl_total = bl["timing"]["total_ms"]
        ad_total = ad["timing"]["total_ms"]
        total_speedup = bl_total / ad_total if ad_total > 0 else 0

        sum_bl_frames += bl_frames
        sum_ad_frames += ad_frames
        sum_bl_encode += bl_encode
        sum_ad_encode += ad_encode
        sum_bl_gen += bl_gen
        sum_ad_gen += ad_gen
        sum_bl_total += bl_total
        sum_ad_total += ad_total

        print(f"{video:<16} {n_total:>5} | {bl_frames:>6} {ad_frames:>6} {frame_reduction:>5.1f}% | "
              f"{bl_encode:>9.0f}ms {ad_encode:>9.0f}ms {encode_speedup:>7.2f}x | "
              f"{bl_gen:>8.0f}ms {ad_gen:>8.0f}ms | "
              f"{bl_total:>8.0f}ms {ad_total:>8.0f}ms {total_speedup:>7.2f}x")

    n = len(rows)
    print("-" * 140)
    avg_reduction = (1.0 - sum_ad_frames / sum_bl_frames) * 100 if sum_bl_frames > 0 else 0
    avg_encode_speedup = sum_bl_encode / sum_ad_encode if sum_ad_encode > 0 else 0
    avg_total_speedup = sum_bl_total / sum_ad_total if sum_ad_total > 0 else 0
    print(f"{'AVERAGE':<16} {'':>5} | {sum_bl_frames/n:>6.1f} {sum_ad_frames/n:>6.1f} {avg_reduction:>5.1f}% | "
          f"{sum_bl_encode/n:>9.0f}ms {sum_ad_encode/n:>9.0f}ms {avg_encode_speedup:>7.2f}x | "
          f"{sum_bl_gen/n:>8.0f}ms {sum_ad_gen/n:>8.0f}ms | "
          f"{sum_bl_total/n:>8.0f}ms {sum_ad_total/n:>8.0f}ms {avg_total_speedup:>7.2f}x")

    # ===== TABLE 2: Quality Comparison =====
    print()
    print("=" * 120)
    print("TABLE 2: Quality Metrics Comparison (Shannon Entropy & Motion Coverage)")
    print("=" * 120)
    header2 = f"{'Video':<16} | {'BL Entropy':>11} {'AD Entropy':>11} {'Diff':>8} | {'BL MotCov':>10} {'AD MotCov':>10} | {'BL Tokens':>9} {'AD Tokens':>9} | {'Complexity':>10}"
    print(header2)
    print("-" * 120)

    sum_bl_ent = 0
    sum_ad_ent = 0
    sum_bl_cov = 0
    sum_ad_cov = 0
    sum_bl_tok = 0
    sum_ad_tok = 0

    for video, bl, ad in rows:
        bl_ent = bl["metrics"]["shannon_entropy"]
        ad_ent = ad["metrics"]["shannon_entropy"]
        ent_diff = ad_ent - bl_ent

        bl_cov = bl["metrics"]["motion_coverage"]
        ad_cov = ad["metrics"]["motion_coverage"]

        bl_tok = bl["inference"]["n_tokens"]
        ad_tok = ad["inference"]["n_tokens"]

        complexity = ad.get("global_complexity", 0)

        sum_bl_ent += bl_ent
        sum_ad_ent += ad_ent
        sum_bl_cov += bl_cov
        sum_ad_cov += ad_cov
        sum_bl_tok += bl_tok
        sum_ad_tok += ad_tok

        print(f"{video:<16} | {bl_ent:>10.4f}b {ad_ent:>10.4f}b {ent_diff:>+7.4f} | "
              f"{bl_cov:>9.1%} {ad_cov:>9.1%} | "
              f"{bl_tok:>9} {ad_tok:>9} | {complexity:>10.3f}")

    print("-" * 120)
    avg_ent_diff = (sum_ad_ent - sum_bl_ent) / n
    print(f"{'AVERAGE':<16} | {sum_bl_ent/n:>10.4f}b {sum_ad_ent/n:>10.4f}b {avg_ent_diff:>+7.4f} | "
          f"{sum_bl_cov/n:>9.1%} {sum_ad_cov/n:>9.1%} | "
          f"{sum_bl_tok/n:>9.1f} {sum_ad_tok/n:>9.1f} | {'':>10}")

    # ===== TABLE 3: Generated Text Comparison =====
    print()
    print("=" * 160)
    print("TABLE 3: Generated Output Comparison (first 100 chars)")
    print("=" * 160)
    print(f"{'Video':<16} | {'Baseline Output':<70} | {'Adaptive Output':<70}")
    print("-" * 160)

    for video, bl, ad in rows:
        bl_text = truncate_text(bl["inference"]["generated_text"], 68)
        ad_text = truncate_text(ad["inference"]["generated_text"], 68)
        print(f"{video:<16} | {bl_text:<70} | {ad_text:<70}")

    # ===== TABLE 4: Adaptive-Only Metrics (Spatial Analysis) =====
    print()
    print("=" * 100)
    print("TABLE 4: Adaptive-Only Metrics (Motion Analysis & Spatial Savings)")
    print("=" * 100)
    print(f"{'Video':<16} | {'Complexity':>10} | {'FlowTime':>10} | {'AvgSpatial':>11} | {'TempReduc':>10} | {'Frames Selected'}")
    print("-" * 100)

    for video, bl, ad in rows:
        complexity = ad.get("global_complexity", 0)
        flow_ms = ad["timing"]["flow_analysis_ms"]
        spatial = ad["metrics"].get("avg_spatial_savings", 0)
        temp_red = ad["metrics"]["temporal_reduction"]
        frames = [sf["index"] for sf in ad.get("selected_frames", [])]
        frame_str = str(frames) if len(str(frames)) <= 40 else str(frames[:5]) + "..."

        print(f"{video:<16} | {complexity:>10.3f} | {flow_ms:>9.0f}ms | {spatial:>10.1%} | {temp_red:>9.1%} | {frame_str}")

    # ===== SUMMARY =====
    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Videos tested:          {n}")
    print(f"  Avg frames (baseline):  {sum_bl_frames/n:.1f}")
    print(f"  Avg frames (adaptive):  {sum_ad_frames/n:.1f}")
    print(f"  Avg frame reduction:    {avg_reduction:.1f}%")
    print(f"  Avg encode speedup:     {avg_encode_speedup:.2f}x")
    print(f"  Avg total speedup:      {avg_total_speedup:.2f}x")
    print(f"  Avg entropy (baseline): {sum_bl_ent/n:.4f} bits")
    print(f"  Avg entropy (adaptive): {sum_ad_ent/n:.4f} bits")
    print(f"  Avg entropy diff:       {avg_ent_diff:+.4f} bits")
    print(f"  Avg motion cov (BL):    {sum_bl_cov/n:.1%}")
    print(f"  Avg motion cov (AD):    {sum_ad_cov/n:.1%}")
    print()

    # Write summary JSON
    summary = {
        "n_videos": n,
        "avg_frames_baseline": sum_bl_frames / n,
        "avg_frames_adaptive": sum_ad_frames / n,
        "avg_frame_reduction_pct": avg_reduction,
        "avg_encode_speedup": avg_encode_speedup,
        "avg_total_speedup": avg_total_speedup,
        "avg_entropy_baseline": sum_bl_ent / n,
        "avg_entropy_adaptive": sum_ad_ent / n,
        "avg_entropy_diff": avg_ent_diff,
        "avg_motion_coverage_baseline": sum_bl_cov / n,
        "avg_motion_coverage_adaptive": sum_ad_cov / n,
        "per_video": []
    }
    for video, bl, ad in rows:
        summary["per_video"].append({
            "video": video,
            "frames_baseline": bl.get("n_frames_selected", 0),
            "frames_adaptive": ad.get("n_frames_selected", 0),
            "encode_ms_baseline": bl["timing"]["encode_ms"],
            "encode_ms_adaptive": ad["timing"]["encode_ms"],
            "total_ms_baseline": bl["timing"]["total_ms"],
            "total_ms_adaptive": ad["timing"]["total_ms"],
            "entropy_baseline": bl["metrics"]["shannon_entropy"],
            "entropy_adaptive": ad["metrics"]["shannon_entropy"],
            "motion_coverage_baseline": bl["metrics"]["motion_coverage"],
            "motion_coverage_adaptive": ad["metrics"]["motion_coverage"],
            "complexity": ad.get("global_complexity", 0),
            "text_baseline": bl["inference"]["generated_text"][:200],
            "text_adaptive": ad["inference"]["generated_text"][:200],
        })

    summary_path = RESULTS_DIR / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON written to: {summary_path}")


if __name__ == "__main__":
    main()
