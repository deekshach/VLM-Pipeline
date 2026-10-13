#!/usr/bin/env python3
"""
Compare adaptive (temporal-only) vs adaptive+spatial-crop results
across all MPI-Sintel videos.

Reads _adaptive.json vs _spatial_crop.json pairs and produces
comparison tables covering encode times, token counts, crop stats,
and quality metrics.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_result(path):
    """Load a JSON result file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def truncate_text(text, max_len=60):
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Find all video names from result files
    videos = set()
    for f in RESULTS_DIR.glob("*_adaptive.json"):
        videos.add(f.stem.replace("_adaptive", ""))
    for f in RESULTS_DIR.glob("*_spatial_crop.json"):
        videos.add(f.stem.replace("_spatial_crop", ""))
    videos = sorted(videos)

    if not videos:
        print("No result files found.")
        sys.exit(1)

    # Collect matched pairs
    rows = []
    for video in videos:
        ad = load_result(RESULTS_DIR / f"{video}_adaptive.json")
        sc = load_result(RESULTS_DIR / f"{video}_spatial_crop.json")
        if ad and sc:
            rows.append((video, ad, sc))

    if not rows:
        print("No matched adaptive/spatial_crop pairs found.")
        # Show what we have
        ad_count = len(list(RESULTS_DIR.glob("*_adaptive.json")))
        sc_count = len(list(RESULTS_DIR.glob("*_spatial_crop.json")))
        print(f"  Found {ad_count} adaptive results, {sc_count} spatial_crop results")
        sys.exit(1)

    # ===== TABLE 1: Encode Time & Token Comparison =====
    print("=" * 150)
    print("TABLE 1: Encode Time & Vision Token Comparison (Adaptive vs Spatial Crop)")
    print("=" * 150)
    header = (
        f"{'Video':<16} {'Frm':>3} | "
        f"{'AD Encode':>10} {'SC Encode':>10} {'Speedup':>8} | "
        f"{'AD TokEst':>9} {'SC TokEst':>9} {'Savings':>8} | "
        f"{'Cropped':>7} | "
        f"{'AD Total':>9} {'SC Total':>9} {'Speedup':>8}"
    )
    print(header)
    print("-" * 150)

    sum_ad_encode = 0
    sum_sc_encode = 0
    sum_ad_tokens_est = 0
    sum_sc_tokens_est = 0
    sum_ad_total = 0
    sum_sc_total = 0
    sum_cropped = 0
    sum_total_frames = 0

    for video, ad, sc in rows:
        n_frames = sc.get("n_frames_selected", 0)

        ad_encode = ad["timing"]["encode_ms"]
        sc_encode = sc["timing"]["encode_ms"]
        encode_speedup = ad_encode / sc_encode if sc_encode > 0 else 0

        # Token estimates from spatial crop metrics
        sc_tok_full = sc["metrics"].get("total_tokens_full", 0)
        sc_tok_crop = sc["metrics"].get("total_tokens_cropped", 0)
        # For adaptive, estimate tokens as full (no cropping)
        ad_tok_est = sc_tok_full  # same frames, all full-size
        tok_savings_pct = sc["metrics"].get("total_token_savings_pct", 0)

        n_cropped = sc["metrics"].get("n_frames_cropped", 0)
        n_selected = sc["metrics"].get("n_frames_total_selected", n_frames)

        ad_total = ad["timing"]["total_ms"]
        sc_total = sc["timing"]["total_ms"]
        total_speedup = ad_total / sc_total if sc_total > 0 else 0

        sum_ad_encode += ad_encode
        sum_sc_encode += sc_encode
        sum_ad_tokens_est += ad_tok_est
        sum_sc_tokens_est += sc_tok_crop
        sum_ad_total += ad_total
        sum_sc_total += sc_total
        sum_cropped += n_cropped
        sum_total_frames += n_selected

        print(
            f"{video:<16} {n_frames:>3} | "
            f"{ad_encode:>9.0f}ms {sc_encode:>9.0f}ms {encode_speedup:>7.2f}x | "
            f"{ad_tok_est:>9} {sc_tok_crop:>9} {tok_savings_pct:>6.1f}% | "
            f"{n_cropped:>3}/{n_selected:<3} | "
            f"{ad_total:>8.0f}ms {sc_total:>8.0f}ms {total_speedup:>7.2f}x"
        )

    n = len(rows)
    print("-" * 150)
    avg_encode_speedup = sum_ad_encode / sum_sc_encode if sum_sc_encode > 0 else 0
    avg_tok_savings = (
        (1.0 - sum_sc_tokens_est / sum_ad_tokens_est) * 100
        if sum_ad_tokens_est > 0 else 0
    )
    avg_total_speedup = sum_ad_total / sum_sc_total if sum_sc_total > 0 else 0
    print(
        f"{'AVERAGE':<16} {'':>3} | "
        f"{sum_ad_encode/n:>9.0f}ms {sum_sc_encode/n:>9.0f}ms {avg_encode_speedup:>7.2f}x | "
        f"{sum_ad_tokens_est/n:>9.0f} {sum_sc_tokens_est/n:>9.0f} {avg_tok_savings:>6.1f}% | "
        f"{sum_cropped:>3}/{sum_total_frames:<3} | "
        f"{sum_ad_total/n:>8.0f}ms {sum_sc_total/n:>8.0f}ms {avg_total_speedup:>7.2f}x"
    )

    # ===== TABLE 2: Quality Comparison =====
    print()
    print("=" * 120)
    print("TABLE 2: Quality Metrics (Adaptive vs Spatial Crop)")
    print("=" * 120)
    header2 = (
        f"{'Video':<16} | "
        f"{'AD Entropy':>11} {'SC Entropy':>11} {'Diff':>8} | "
        f"{'AD MotCov':>10} {'SC MotCov':>10} | "
        f"{'AD Tokens':>9} {'SC Tokens':>9} | "
        f"{'Complexity':>10}"
    )
    print(header2)
    print("-" * 120)

    sum_ad_ent = 0
    sum_sc_ent = 0
    sum_ad_cov = 0
    sum_sc_cov = 0
    sum_ad_tok = 0
    sum_sc_tok = 0

    for video, ad, sc in rows:
        ad_ent = ad["metrics"]["shannon_entropy"]
        sc_ent = sc["metrics"]["shannon_entropy"]
        ent_diff = sc_ent - ad_ent

        ad_cov = ad["metrics"]["motion_coverage"]
        sc_cov = sc["metrics"]["motion_coverage"]

        ad_tok = ad["inference"]["n_tokens"]
        sc_tok = sc["inference"]["n_tokens"]

        complexity = sc.get("global_complexity", 0)

        sum_ad_ent += ad_ent
        sum_sc_ent += sc_ent
        sum_ad_cov += ad_cov
        sum_sc_cov += sc_cov
        sum_ad_tok += ad_tok
        sum_sc_tok += sc_tok

        print(
            f"{video:<16} | "
            f"{ad_ent:>10.4f}b {sc_ent:>10.4f}b {ent_diff:>+7.4f} | "
            f"{ad_cov:>9.1%} {sc_cov:>9.1%} | "
            f"{ad_tok:>9} {sc_tok:>9} | "
            f"{complexity:>10.3f}"
        )

    print("-" * 120)
    avg_ent_diff = (sum_sc_ent - sum_ad_ent) / n
    print(
        f"{'AVERAGE':<16} | "
        f"{sum_ad_ent/n:>10.4f}b {sum_sc_ent/n:>10.4f}b {avg_ent_diff:>+7.4f} | "
        f"{sum_ad_cov/n:>9.1%} {sum_sc_cov/n:>9.1%} | "
        f"{sum_ad_tok/n:>9.1f} {sum_sc_tok/n:>9.1f} | "
        f"{'':>10}"
    )

    # ===== TABLE 3: Per-Video Crop Statistics =====
    print()
    print("=" * 130)
    print("TABLE 3: Spatial Crop Statistics (per video)")
    print("=" * 130)
    header3 = (
        f"{'Video':<16} | "
        f"{'Frames':>6} {'Cropped':>7} {'CropPct':>8} | "
        f"{'AvgCropW':>8} {'AvgCropH':>8} | "
        f"{'OrigW':>5} {'OrigH':>5} | "
        f"{'AreaReduc':>9} | "
        f"{'TokFull':>7} {'TokCrop':>7} {'TokSave':>8}"
    )
    print(header3)
    print("-" * 130)

    for video, ad, sc in rows:
        crops = sc.get("per_frame_crops", [])
        n_total = len(crops)
        n_cropped = sum(1 for c in crops if c.get("was_cropped", False))
        crop_pct = (n_cropped / n_total * 100) if n_total > 0 else 0

        cropped_entries = [c for c in crops if c.get("was_cropped", False)]
        avg_cw = (
            sum(c["crop_w"] for c in cropped_entries) / len(cropped_entries)
            if cropped_entries else 0
        )
        avg_ch = (
            sum(c["crop_h"] for c in cropped_entries) / len(cropped_entries)
            if cropped_entries else 0
        )

        orig_w = crops[0]["orig_w"] if crops else 0
        orig_h = crops[0]["orig_h"] if crops else 0

        # Average area reduction across cropped frames
        area_reductions = []
        for c in cropped_entries:
            orig_area = c["orig_w"] * c["orig_h"]
            crop_area = c["crop_w"] * c["crop_h"]
            area_reductions.append(1.0 - crop_area / orig_area if orig_area > 0 else 0)
        avg_area_reduc = (
            sum(area_reductions) / len(area_reductions)
            if area_reductions else 0
        )

        tok_full = sc["metrics"].get("total_tokens_full", 0)
        tok_crop = sc["metrics"].get("total_tokens_cropped", 0)
        tok_save_pct = sc["metrics"].get("total_token_savings_pct", 0)

        print(
            f"{video:<16} | "
            f"{n_total:>6} {n_cropped:>7} {crop_pct:>7.1f}% | "
            f"{avg_cw:>8.0f} {avg_ch:>8.0f} | "
            f"{orig_w:>5} {orig_h:>5} | "
            f"{avg_area_reduc:>8.1%} | "
            f"{tok_full:>7} {tok_crop:>7} {tok_save_pct:>7.1f}%"
        )

    # ===== TABLE 4: Generated Text Comparison =====
    print()
    print("=" * 150)
    print("TABLE 4: Generated Output Comparison (first 60 chars)")
    print("=" * 150)
    print(f"{'Video':<16} | {'Adaptive Output':<64} | {'Spatial Crop Output':<64}")
    print("-" * 150)

    for video, ad, sc in rows:
        ad_text = truncate_text(ad["inference"]["generated_text"], 62)
        sc_text = truncate_text(sc["inference"]["generated_text"], 62)
        print(f"{video:<16} | {ad_text:<64} | {sc_text:<64}")

    # ===== SUMMARY =====
    print()
    print("=" * 80)
    print("OVERALL SUMMARY: Adaptive vs Spatial Crop")
    print("=" * 80)
    print(f"  Videos compared:         {n}")
    print(f"  Avg encode time (AD):    {sum_ad_encode/n:.0f} ms")
    print(f"  Avg encode time (SC):    {sum_sc_encode/n:.0f} ms")
    print(f"  Avg encode speedup:      {avg_encode_speedup:.2f}x")
    print(f"  Avg total time (AD):     {sum_ad_total/n:.0f} ms")
    print(f"  Avg total time (SC):     {sum_sc_total/n:.0f} ms")
    print(f"  Avg total speedup:       {avg_total_speedup:.2f}x")
    print(f"  Avg est. token savings:  {avg_tok_savings:.1f}%")
    print(f"  Total frames cropped:    {sum_cropped} / {sum_total_frames}")
    print(f"  Avg entropy (AD):        {sum_ad_ent/n:.4f} bits")
    print(f"  Avg entropy (SC):        {sum_sc_ent/n:.4f} bits")
    print(f"  Avg entropy diff:        {avg_ent_diff:+.4f} bits")
    print(f"  Avg motion cov (AD):     {sum_ad_cov/n:.1%}")
    print(f"  Avg motion cov (SC):     {sum_sc_cov/n:.1%}")
    print()

    # Write summary JSON
    summary = {
        "comparison": "adaptive_vs_spatial_crop",
        "n_videos": n,
        "avg_encode_ms_adaptive": sum_ad_encode / n,
        "avg_encode_ms_spatial_crop": sum_sc_encode / n,
        "avg_encode_speedup": avg_encode_speedup,
        "avg_total_ms_adaptive": sum_ad_total / n,
        "avg_total_ms_spatial_crop": sum_sc_total / n,
        "avg_total_speedup": avg_total_speedup,
        "avg_token_savings_pct": avg_tok_savings,
        "total_frames_cropped": sum_cropped,
        "total_frames_selected": sum_total_frames,
        "avg_entropy_adaptive": sum_ad_ent / n,
        "avg_entropy_spatial_crop": sum_sc_ent / n,
        "avg_entropy_diff": avg_ent_diff,
        "avg_motion_coverage_adaptive": sum_ad_cov / n,
        "avg_motion_coverage_spatial_crop": sum_sc_cov / n,
        "per_video": [],
    }
    for video, ad, sc in rows:
        sc_crops = sc.get("per_frame_crops", [])
        n_cropped_v = sum(1 for c in sc_crops if c.get("was_cropped", False))
        summary["per_video"].append({
            "video": video,
            "encode_ms_adaptive": ad["timing"]["encode_ms"],
            "encode_ms_spatial_crop": sc["timing"]["encode_ms"],
            "total_ms_adaptive": ad["timing"]["total_ms"],
            "total_ms_spatial_crop": sc["timing"]["total_ms"],
            "n_frames_selected": sc.get("n_frames_selected", 0),
            "n_frames_cropped": n_cropped_v,
            "token_savings_pct": sc["metrics"].get("total_token_savings_pct", 0),
            "entropy_adaptive": ad["metrics"]["shannon_entropy"],
            "entropy_spatial_crop": sc["metrics"]["shannon_entropy"],
            "motion_coverage_adaptive": ad["metrics"]["motion_coverage"],
            "motion_coverage_spatial_crop": sc["metrics"]["motion_coverage"],
            "complexity": sc.get("global_complexity", 0),
            "text_adaptive": ad["inference"]["generated_text"][:200],
            "text_spatial_crop": sc["inference"]["generated_text"][:200],
        })

    summary_path = RESULTS_DIR / "spatial_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON written to: {summary_path}")


if __name__ == "__main__":
    main()
