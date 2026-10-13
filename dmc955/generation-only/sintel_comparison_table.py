#!/usr/bin/env python3
import json, os, sys

base_dir = "/data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_comparison"

videos = sorted(os.listdir(os.path.join(base_dir, "baseline")))
videos = [v.replace(".json", "") for v in videos if v.endswith(".json")]

rows = []
for vid in videos:
    bp = os.path.join(base_dir, "baseline", f"{vid}.json")
    pp = os.path.join(base_dir, "pyramid", f"{vid}.json")
    if not os.path.exists(bp) or not os.path.exists(pp):
        continue
    with open(bp) as f:
        b = json.load(f)
    with open(pp) as f:
        p = json.load(f)

    rows.append({
        "video": vid,
        "b_total_ms": b.get("total_time_ms", 0),
        "b_frames": b.get("n_frames", 0),
        "b_vis_tok": b.get("n_visual_tokens", 0),
        "b_kept": b.get("n_visual_tokens_kept", 0),
        "b_prefill": b.get("prefill_time_ms", 0),
        "b_gen": b.get("generation_time_ms", 0),
        "b_encode": b.get("encode_time_ms", 0),
        "b_tps": b.get("tokens_per_sec", 0),
        "b_n_gen": b.get("n_tokens_generated", 0),
        "p_total_ms": p.get("total_time_ms", 0),
        "p_frames": p.get("n_frames", 0),
        "p_frames_total": p.get("n_frames_total", 0),
        "p_vis_tok": p.get("n_visual_tokens", 0),
        "p_kept": p.get("n_visual_tokens_kept", 0),
        "p_prefill": p.get("prefill_time_ms", 0),
        "p_gen": p.get("generation_time_ms", 0),
        "p_encode": p.get("encode_time_ms", 0),
        "p_evict": p.get("eviction_time_ms", 0),
        "p_select": p.get("selection_time_ms", 0),
        "p_tps": p.get("tokens_per_sec", 0),
        "p_n_gen": p.get("n_tokens_generated", 0),
        "p_vit_skip": p.get("n_vit_skipped", 0),
        "p_vit_crop": p.get("n_vit_cropped", 0),
        "p_vit_full": p.get("n_vit_full", 0),
        "p_complexity": p.get("global_complexity", 0),
    })

if not rows:
    print("No results found!")
    sys.exit(1)

# Print main table
print()
hdr = (f"{'Video':<16} │ {'E2E (ms)':>8} {'Frames':>6} │ "
       f"{'E2E (ms)':>8} {'Frames':>6} {'Selected':>8} │ {'Speedup':>7} {'ΔFrames':>7}")
sep = "─" * 16 + "─┼─" + "─" * 15 + "─┼─" + "─" * 25 + "─┼─" + "─" * 15
print(f"{'':16} │ {'BASELINE':^15} │ {'PYRAMID (all features)':^25} │ {'COMPARISON':^15}")
print(sep)
print(hdr)
print(sep)

b_total_sum = p_total_sum = 0
b_frames_sum = p_frames_sum = 0

for r in rows:
    speedup = r["b_total_ms"] / r["p_total_ms"] if r["p_total_ms"] > 0 else 0
    df = r["p_frames"] - r["b_frames"]
    b_total_sum += r["b_total_ms"]
    p_total_sum += r["p_total_ms"]
    b_frames_sum += r["b_frames"]
    p_frames_sum += r["p_frames"]

    print(f"{r['video']:<16} │ {r['b_total_ms']:7.0f}  {r['b_frames']:6d} │ "
          f"{r['p_total_ms']:7.0f}  {r['p_frames']:6d} {r['p_frames']:>5}/{r['p_frames_total']:<3} │ "
          f"{speedup:6.2f}x {df:+6d}")

print(sep)
n = len(rows)
speedup_avg = b_total_sum / p_total_sum if p_total_sum > 0 else 0
print(f"{'TOTAL':<16} │ {b_total_sum:7.0f}  {b_frames_sum:6d} │ "
      f"{p_total_sum:7.0f}  {p_frames_sum:6d} {'':>9} │ "
      f"{speedup_avg:6.2f}x {p_frames_sum - b_frames_sum:+6d}")
print(f"{'AVERAGE':<16} │ {b_total_sum/n:7.0f}  {b_frames_sum/n:6.1f} │ "
      f"{p_total_sum/n:7.0f}  {p_frames_sum/n:6.1f} {'':>9} │ "
      f"{speedup_avg:6.2f}x")

# Detailed breakdown table
print(f"\n\n{'DETAILED BREAKDOWN':^100}")
sep2 = "─" * 100
print(sep2)
print(f"{'Video':<16} │ {'B_VisTok':>8} {'B_Prefill':>9} {'B_Gen':>7} {'B_t/s':>6} │ "
      f"{'P_VisTok':>8} {'P_Kept':>6} {'P_Prefill':>9} {'P_Evict':>7} {'P_Gen':>7} {'P_t/s':>6} │ {'TokSave':>7}")
print(sep2)

b_vis_sum = p_vis_sum = p_kept_sum = 0
for r in rows:
    tok_save = 1.0 - r["p_kept"] / r["b_vis_tok"] if r["b_vis_tok"] > 0 else 0
    b_vis_sum += r["b_vis_tok"]
    p_vis_sum += r["p_vis_tok"]
    p_kept_sum += r["p_kept"]
    print(f"{r['video']:<16} │ {r['b_vis_tok']:8d} {r['b_prefill']:8.0f}  {r['b_gen']:6.0f}  {r['b_tps']:5.1f} │ "
          f"{r['p_vis_tok']:8d} {r['p_kept']:6d} {r['p_prefill']:8.0f}  {r['p_evict']:6.0f}  {r['p_gen']:6.0f}  {r['p_tps']:5.1f} │ "
          f"{tok_save:6.0%}")

print(sep2)
tok_save_avg = 1.0 - p_kept_sum / b_vis_sum if b_vis_sum > 0 else 0
print(f"{'AVERAGE':<16} │ {b_vis_sum/n:8.0f} {sum(r['b_prefill'] for r in rows)/n:8.0f}  "
      f"{sum(r['b_gen'] for r in rows)/n:6.0f}  {sum(r['b_tps'] for r in rows)/n:5.1f} │ "
      f"{p_vis_sum/n:8.0f} {p_kept_sum/n:6.0f} {sum(r['p_prefill'] for r in rows)/n:8.0f}  "
      f"{sum(r['p_evict'] for r in rows)/n:6.0f}  {sum(r['p_gen'] for r in rows)/n:6.0f}  "
      f"{sum(r['p_tps'] for r in rows)/n:5.1f} │ {tok_save_avg:6.0%}")

# ViT savings breakdown
print(f"\n\n{'VIT ENCODER SAVINGS':^80}")
print("─" * 80)
print(f"{'Video':<16} │ {'Complexity':>10} {'Full':>5} {'Skip':>5} {'Crop':>5} │ {'ViT Savings':>11}")
print("─" * 80)
for r in rows:
    total_enc = r["p_vit_full"] + r["p_vit_skip"] + r["p_vit_crop"]
    vit_save = (r["p_vit_skip"] + r["p_vit_crop"] * 0.5) / total_enc * 100 if total_enc > 0 else 0
    print(f"{r['video']:<16} │ {r['p_complexity']:10.3f} {r['p_vit_full']:5d} {r['p_vit_skip']:5d} {r['p_vit_crop']:5d} │ {vit_save:10.0f}%")

# Save JSON
with open(os.path.join(base_dir, "comparison_results.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nJSON saved to {base_dir}/comparison_results.json")
