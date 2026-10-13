#!/usr/bin/env python3
"""
Three-way comparison: Baseline (uniform) vs Adaptive (temporal) vs Spatial Crop
across all MPI-Sintel videos.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Find all video names
    videos = set()
    for suffix in ["_baseline.json", "_adaptive.json", "_spatial_crop.json"]:
        for f in RESULTS_DIR.glob(f"*{suffix}"):
            videos.add(f.stem.replace(suffix.replace(".json", ""), ""))
    videos = sorted(videos)

    # Collect matched triples
    rows = []
    for video in videos:
        bl = load_result(RESULTS_DIR / f"{video}_baseline.json")
        ad = load_result(RESULTS_DIR / f"{video}_adaptive.json")
        sc = load_result(RESULTS_DIR / f"{video}_spatial_crop.json")
        if bl and ad and sc:
            rows.append((video, bl, ad, sc))

    if not rows:
        print("No matched baseline/adaptive/spatial_crop triples found.")
        sys.exit(1)

    n = len(rows)

    # ===== TABLE 1: Frame Count & Encode Time =====
    print("=" * 170)
    print("TABLE 1: Frame Count & Encode Time — Baseline vs Adaptive vs Spatial Crop")
    print("=" * 170)
    print(f"{'Video':<16} {'Tot':>3} | "
          f"{'BL Frm':>6} {'AD Frm':>6} {'SC Frm':>6} | "
          f"{'BL Encode':>10} {'AD Encode':>10} {'SC Encode':>10} | "
          f"{'AD/BL':>6} {'SC/BL':>6} {'SC/AD':>6} | "
          f"{'BL Total':>9} {'AD Total':>9} {'SC Total':>9} | "
          f"{'AD/BL':>6} {'SC/BL':>6}")
    print("-" * 170)

    s = {k: 0 for k in [
        "bl_f", "ad_f", "sc_f",
        "bl_enc", "ad_enc", "sc_enc",
        "bl_tot", "ad_tot", "sc_tot",
    ]}

    for video, bl, ad, sc in rows:
        n_total = bl.get("n_frames_total", 0)
        bl_f = bl.get("n_frames_selected", 0)
        ad_f = ad.get("n_frames_selected", 0)
        sc_f = sc.get("n_frames_selected", 0)

        bl_enc = bl["timing"]["encode_ms"]
        ad_enc = ad["timing"]["encode_ms"]
        sc_enc = sc["timing"]["encode_ms"]

        ad_bl_enc = bl_enc / ad_enc if ad_enc > 0 else 0
        sc_bl_enc = bl_enc / sc_enc if sc_enc > 0 else 0
        sc_ad_enc = ad_enc / sc_enc if sc_enc > 0 else 0

        bl_tot = bl["timing"]["total_ms"]
        ad_tot = ad["timing"]["total_ms"]
        sc_tot = sc["timing"]["total_ms"]

        ad_bl_tot = bl_tot / ad_tot if ad_tot > 0 else 0
        sc_bl_tot = bl_tot / sc_tot if sc_tot > 0 else 0

        for k, v in [("bl_f", bl_f), ("ad_f", ad_f), ("sc_f", sc_f),
                      ("bl_enc", bl_enc), ("ad_enc", ad_enc), ("sc_enc", sc_enc),
                      ("bl_tot", bl_tot), ("ad_tot", ad_tot), ("sc_tot", sc_tot)]:
            s[k] += v

        print(f"{video:<16} {n_total:>3} | "
              f"{bl_f:>6} {ad_f:>6} {sc_f:>6} | "
              f"{bl_enc:>9.0f}ms {ad_enc:>9.0f}ms {sc_enc:>9.0f}ms | "
              f"{ad_bl_enc:>5.2f}x {sc_bl_enc:>5.2f}x {sc_ad_enc:>5.2f}x | "
              f"{bl_tot:>8.0f}ms {ad_tot:>8.0f}ms {sc_tot:>8.0f}ms | "
              f"{ad_bl_tot:>5.2f}x {sc_bl_tot:>5.2f}x")

    print("-" * 170)
    print(f"{'AVERAGE':<16} {'':>3} | "
          f"{s['bl_f']/n:>6.1f} {s['ad_f']/n:>6.1f} {s['sc_f']/n:>6.1f} | "
          f"{s['bl_enc']/n:>9.0f}ms {s['ad_enc']/n:>9.0f}ms {s['sc_enc']/n:>9.0f}ms | "
          f"{s['bl_enc']/s['ad_enc']:>5.2f}x {s['bl_enc']/s['sc_enc']:>5.2f}x {s['ad_enc']/s['sc_enc']:>5.2f}x | "
          f"{s['bl_tot']/n:>8.0f}ms {s['ad_tot']/n:>8.0f}ms {s['sc_tot']/n:>8.0f}ms | "
          f"{s['bl_tot']/s['ad_tot']:>5.2f}x {s['bl_tot']/s['sc_tot']:>5.2f}x")

    # ===== TABLE 2: Quality Metrics =====
    print()
    print("=" * 150)
    print("TABLE 2: Quality Metrics — Baseline vs Adaptive vs Spatial Crop")
    print("=" * 150)
    print(f"{'Video':<16} | "
          f"{'BL Entropy':>11} {'AD Entropy':>11} {'SC Entropy':>11} | "
          f"{'BL MotCov':>9} {'AD MotCov':>9} {'SC MotCov':>9} | "
          f"{'BL Tok':>6} {'AD Tok':>6} {'SC Tok':>6} | "
          f"{'Cmplx':>6}")
    print("-" * 150)

    se = {k: 0 for k in ["bl_ent", "ad_ent", "sc_ent", "bl_cov", "ad_cov", "sc_cov",
                           "bl_tok", "ad_tok", "sc_tok"]}

    for video, bl, ad, sc in rows:
        bl_ent = bl["metrics"]["shannon_entropy"]
        ad_ent = ad["metrics"]["shannon_entropy"]
        sc_ent = sc["metrics"]["shannon_entropy"]

        bl_cov = bl["metrics"]["motion_coverage"]
        ad_cov = ad["metrics"]["motion_coverage"]
        sc_cov = sc["metrics"]["motion_coverage"]

        bl_tok = bl["inference"]["n_tokens"]
        ad_tok = ad["inference"]["n_tokens"]
        sc_tok = sc["inference"]["n_tokens"]

        cmplx = sc.get("global_complexity", ad.get("global_complexity", 0))

        for k, v in [("bl_ent", bl_ent), ("ad_ent", ad_ent), ("sc_ent", sc_ent),
                      ("bl_cov", bl_cov), ("ad_cov", ad_cov), ("sc_cov", sc_cov),
                      ("bl_tok", bl_tok), ("ad_tok", ad_tok), ("sc_tok", sc_tok)]:
            se[k] += v

        print(f"{video:<16} | "
              f"{bl_ent:>10.4f}b {ad_ent:>10.4f}b {sc_ent:>10.4f}b | "
              f"{bl_cov:>8.1%} {ad_cov:>8.1%} {sc_cov:>8.1%} | "
              f"{bl_tok:>6} {ad_tok:>6} {sc_tok:>6} | "
              f"{cmplx:>6.3f}")

    print("-" * 150)
    print(f"{'AVERAGE':<16} | "
          f"{se['bl_ent']/n:>10.4f}b {se['ad_ent']/n:>10.4f}b {se['sc_ent']/n:>10.4f}b | "
          f"{se['bl_cov']/n:>8.1%} {se['ad_cov']/n:>8.1%} {se['sc_cov']/n:>8.1%} | "
          f"{se['bl_tok']/n:>6.1f} {se['ad_tok']/n:>6.1f} {se['sc_tok']/n:>6.1f} | "
          f"{'':>6}")

    # ===== TABLE 3: Spatial Crop Effectiveness =====
    print()
    print("=" * 130)
    print("TABLE 3: Spatial Crop Effectiveness (why crops did/didn't fire)")
    print("=" * 130)
    print(f"{'Video':<16} | "
          f"{'Frames':>6} {'Cropped':>7} | "
          f"{'AvgActPatch':>11} | "
          f"{'TokFull':>7} {'TokCrop':>7} {'TokSave%':>8} | "
          f"{'AvgSpatSave':>11} | "
          f"{'Reason no crop'}")
    print("-" * 130)

    total_frames = 0
    total_cropped = 0

    for video, bl, ad, sc in rows:
        crops = sc.get("per_frame_crops", [])
        n_sel = len(crops)
        n_crop = sum(1 for c in crops if c.get("was_cropped", False))
        total_frames += n_sel
        total_cropped += n_crop

        # Average active patches from spatial analysis
        sa_list = sc.get("spatial_analysis", [])
        if sa_list:
            avg_active = sum(s["n_selected_patches"] / s["n_total_patches"] for s in sa_list) / len(sa_list)
        else:
            avg_active = 0

        tok_full = sc["metrics"].get("total_tokens_full", 0)
        tok_crop = sc["metrics"].get("total_tokens_cropped", 0)
        tok_save = sc["metrics"].get("total_token_savings_pct", 0)

        avg_spat = sc["metrics"].get("avg_spatial_savings", 0)

        # Diagnose reason
        if n_crop > 0:
            reason = f"CROPPED {n_crop} frames"
        elif avg_active >= 0.9:
            reason = "all patches active (global motion)"
        elif avg_active >= 0.7:
            reason = "bbox covers >90% area (spread patches)"
        else:
            reason = "bbox covers >90% (wide patch spread)"

        print(f"{video:<16} | "
              f"{n_sel:>6} {n_crop:>7} | "
              f"{avg_active:>10.0%} | "
              f"{tok_full:>7} {tok_crop:>7} {tok_save:>7.1f}% | "
              f"{avg_spat:>10.1%} | "
              f"{reason}")

    print("-" * 130)
    print(f"{'TOTAL':<16} | {total_frames:>6} {total_cropped:>7} |")

    # ===== TABLE 4: Generated Text Side-by-Side =====
    print()
    print("=" * 200)
    print("TABLE 4: Generated Descriptions (first 55 chars)")
    print("=" * 200)
    print(f"{'Video':<16} | {'Baseline':<58} | {'Adaptive':<58} | {'Spatial Crop':<58}")
    print("-" * 200)

    for video, bl, ad, sc in rows:
        def trunc(t, m=56):
            return t[:m-3] + "..." if len(t) > m else t
        print(f"{video:<16} | "
              f"{trunc(bl['inference']['generated_text']):<58} | "
              f"{trunc(ad['inference']['generated_text']):<58} | "
              f"{trunc(sc['inference']['generated_text']):<58}")

    # ===== OVERALL SUMMARY =====
    print()
    print("=" * 80)
    print("OVERALL SUMMARY: Three-Way Comparison")
    print("=" * 80)
    print(f"  Videos compared:             {n}")
    print()
    print(f"  --- Frame Selection ---")
    print(f"  Avg frames (BL):             {s['bl_f']/n:.1f}")
    print(f"  Avg frames (AD):             {s['ad_f']/n:.1f}  ({(1-s['ad_f']/s['bl_f'])*100:.1f}% reduction vs BL)")
    print(f"  Avg frames (SC):             {s['sc_f']/n:.1f}  ({(1-s['sc_f']/s['bl_f'])*100:.1f}% reduction vs BL)")
    print()
    print(f"  --- Encode Time ---")
    print(f"  Avg encode (BL):             {s['bl_enc']/n:.0f} ms")
    print(f"  Avg encode (AD):             {s['ad_enc']/n:.0f} ms  ({s['bl_enc']/s['ad_enc']:.2f}x vs BL)")
    print(f"  Avg encode (SC):             {s['sc_enc']/n:.0f} ms  ({s['bl_enc']/s['sc_enc']:.2f}x vs BL, {s['ad_enc']/s['sc_enc']:.2f}x vs AD)")
    print()
    print(f"  --- Total Time ---")
    print(f"  Avg total (BL):              {s['bl_tot']/n:.0f} ms")
    print(f"  Avg total (AD):              {s['ad_tot']/n:.0f} ms  ({s['bl_tot']/s['ad_tot']:.2f}x vs BL)")
    print(f"  Avg total (SC):              {s['sc_tot']/n:.0f} ms  ({s['bl_tot']/s['sc_tot']:.2f}x vs BL, {s['ad_tot']/s['sc_tot']:.2f}x vs AD)")
    print()
    print(f"  --- Quality ---")
    print(f"  Avg entropy (BL):            {se['bl_ent']/n:.4f} bits")
    print(f"  Avg entropy (AD):            {se['ad_ent']/n:.4f} bits  ({(se['ad_ent']-se['bl_ent'])/n:+.4f} vs BL)")
    print(f"  Avg entropy (SC):            {se['sc_ent']/n:.4f} bits  ({(se['sc_ent']-se['bl_ent'])/n:+.4f} vs BL)")
    print()
    print(f"  Avg motion coverage (BL):    {se['bl_cov']/n:.1%}")
    print(f"  Avg motion coverage (AD):    {se['ad_cov']/n:.1%}")
    print(f"  Avg motion coverage (SC):    {se['sc_cov']/n:.1%}")
    print()
    print(f"  --- Spatial Cropping ---")
    print(f"  Frames cropped:              {total_cropped} / {total_frames} ({total_cropped/total_frames*100:.1f}%)")
    print(f"  Avg est. vision token save:  {(1-s.get('sc_tok_est', se['sc_tok'])/max(1, s.get('ad_tok_est', se['ad_tok'])))*100:.1f}%")
    print()

    # Write summary JSON
    summary = {
        "comparison": "three_way_bl_ad_sc",
        "n_videos": n,
        "frame_selection": {
            "avg_frames_baseline": s["bl_f"] / n,
            "avg_frames_adaptive": s["ad_f"] / n,
            "avg_frames_spatial_crop": s["sc_f"] / n,
        },
        "encode_time": {
            "avg_ms_baseline": s["bl_enc"] / n,
            "avg_ms_adaptive": s["ad_enc"] / n,
            "avg_ms_spatial_crop": s["sc_enc"] / n,
            "speedup_ad_vs_bl": s["bl_enc"] / s["ad_enc"],
            "speedup_sc_vs_bl": s["bl_enc"] / s["sc_enc"],
            "speedup_sc_vs_ad": s["ad_enc"] / s["sc_enc"],
        },
        "total_time": {
            "avg_ms_baseline": s["bl_tot"] / n,
            "avg_ms_adaptive": s["ad_tot"] / n,
            "avg_ms_spatial_crop": s["sc_tot"] / n,
            "speedup_ad_vs_bl": s["bl_tot"] / s["ad_tot"],
            "speedup_sc_vs_bl": s["bl_tot"] / s["sc_tot"],
            "speedup_sc_vs_ad": s["ad_tot"] / s["sc_tot"],
        },
        "quality": {
            "avg_entropy_baseline": se["bl_ent"] / n,
            "avg_entropy_adaptive": se["ad_ent"] / n,
            "avg_entropy_spatial_crop": se["sc_ent"] / n,
            "avg_motion_cov_baseline": se["bl_cov"] / n,
            "avg_motion_cov_adaptive": se["ad_cov"] / n,
            "avg_motion_cov_spatial_crop": se["sc_cov"] / n,
        },
        "spatial_cropping": {
            "total_frames_selected": total_frames,
            "total_frames_cropped": total_cropped,
            "crop_rate_pct": total_cropped / total_frames * 100,
        },
        "per_video": [],
    }
    for video, bl, ad, sc in rows:
        sc_crops = sc.get("per_frame_crops", [])
        summary["per_video"].append({
            "video": video,
            "frames_bl": bl.get("n_frames_selected", 0),
            "frames_ad": ad.get("n_frames_selected", 0),
            "frames_sc": sc.get("n_frames_selected", 0),
            "n_cropped": sum(1 for c in sc_crops if c.get("was_cropped", False)),
            "encode_bl": bl["timing"]["encode_ms"],
            "encode_ad": ad["timing"]["encode_ms"],
            "encode_sc": sc["timing"]["encode_ms"],
            "total_bl": bl["timing"]["total_ms"],
            "total_ad": ad["timing"]["total_ms"],
            "total_sc": sc["timing"]["total_ms"],
            "entropy_bl": bl["metrics"]["shannon_entropy"],
            "entropy_ad": ad["metrics"]["shannon_entropy"],
            "entropy_sc": sc["metrics"]["shannon_entropy"],
            "complexity": sc.get("global_complexity", 0),
        })

    out_path = RESULTS_DIR / "three_way_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON written to: {out_path}")


if __name__ == "__main__":
    main()
