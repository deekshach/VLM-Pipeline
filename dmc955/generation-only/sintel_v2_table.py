#!/usr/bin/env python3
import json, os

base = "/data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_v2"
rows = []

for vid in sorted(os.listdir(os.path.join(base, "baseline"))):
    vid = vid.replace(".json", "")
    bp = os.path.join(base, "baseline", f"{vid}.json")
    pp = os.path.join(base, "pyramid", f"{vid}.json")
    if not os.path.exists(pp):
        continue
    with open(bp) as f: b = json.load(f)
    with open(pp) as f: p = json.load(f)
    rows.append({"video": vid, "b": b, "p": p})

if not rows:
    print("No results!")
    exit(1)

n = len(rows)

# Main table
print(f"\n{'='*170}")
print(f"  SINTEL FULL COMPARISON: Baseline (8 uniform frames) vs Pyramid (adaptive ≤10, spatial cache, MOTION+KF, eviction, defrag)")
print(f"  Both generate exactly 128 tokens (--ignore-eos)")
print(f"{'='*170}")
print()

hdr = (f"{'Video':<14} │ {'ViT':>5} {'Prefill':>7} {'Decode':>6} {'E2E':>6} {'Fr':>3} {'VisTok':>6} {'Entropy':>7} {'CosDist':>7} │ "
       f"{'ViT':>5} {'Prefill':>7} {'Decode':>6} {'E2E':>6} {'Fr':>3} {'VisTok':>6} {'Kept':>5} {'Entropy':>7} {'CosDist':>7} │ "
       f"{'Spdup':>5}")
print(f"{'':14} │ {'─── BASELINE ───':^50} │ {'─── PYRAMID ───':^63} │")
print(f"{'─'*14}─┼─{'─'*50}─┼─{'─'*63}─┼─{'─'*5}")
print(hdr)
print(f"{'─'*14}─┼─{'─'*50}─┼─{'─'*63}─┼─{'─'*5}")

sums = {k: 0 for k in ['b_vit','b_pre','b_dec','b_e2e','b_fr','b_vis','b_ent','b_cos',
                         'p_vit','p_pre','p_dec','p_e2e','p_fr','p_vis','p_kept','p_ent','p_cos']}

for r in rows:
    b, p = r['b'], r['p']
    bv = b.get('encode_time_ms',0); bp_ = b.get('prefill_time_ms',0); bd = b.get('generation_time_ms',0)
    be = b.get('total_time_ms',0); bf = b.get('n_frames',0); bvt = b.get('n_visual_tokens',0)
    bent = b.get('embedding_entropy',0); bcos = b.get('avg_cosine_dist',0)

    pv = p.get('encode_time_ms',0); pp_ = p.get('prefill_time_ms',0); pd = p.get('generation_time_ms',0)
    pe = p.get('total_time_ms',0); pf = p.get('n_frames',0); pvt = p.get('n_visual_tokens',0)
    pk = p.get('n_visual_tokens_kept',0)
    pent = p.get('embedding_entropy',0); pcos = p.get('avg_cosine_dist',0)

    speedup = be / pe if pe > 0 else 0

    sums['b_vit']+=bv; sums['b_pre']+=bp_; sums['b_dec']+=bd; sums['b_e2e']+=be
    sums['b_fr']+=bf; sums['b_vis']+=bvt; sums['b_ent']+=bent; sums['b_cos']+=bcos
    sums['p_vit']+=pv; sums['p_pre']+=pp_; sums['p_dec']+=pd; sums['p_e2e']+=pe
    sums['p_fr']+=pf; sums['p_vis']+=pvt; sums['p_kept']+=pk; sums['p_ent']+=pent; sums['p_cos']+=pcos

    print(f"{r['video']:<14} │ {bv:5.0f} {bp_:7.0f} {bd:6.0f} {be:6.0f} {bf:3d} {bvt:6d} {bent:7.3f} {bcos:7.4f} │ "
          f"{pv:5.0f} {pp_:7.0f} {pd:6.0f} {pe:6.0f} {pf:3d} {pvt:6d} {pk:5d} {pent:7.3f} {pcos:7.4f} │ "
          f"{speedup:5.2f}x")

print(f"{'─'*14}─┼─{'─'*50}─┼─{'─'*63}─┼─{'─'*5}")

# Averages
speedup_avg = sums['b_e2e'] / sums['p_e2e'] if sums['p_e2e'] > 0 else 0
print(f"{'AVERAGE':<14} │ {sums['b_vit']/n:5.0f} {sums['b_pre']/n:7.0f} {sums['b_dec']/n:6.0f} {sums['b_e2e']/n:6.0f} "
      f"{sums['b_fr']/n:3.0f} {sums['b_vis']/n:6.0f} {sums['b_ent']/n:7.3f} {sums['b_cos']/n:7.4f} │ "
      f"{sums['p_vit']/n:5.0f} {sums['p_pre']/n:7.0f} {sums['p_dec']/n:6.0f} {sums['p_e2e']/n:6.0f} "
      f"{sums['p_fr']/n:3.0f} {sums['p_vis']/n:6.0f} {sums['p_kept']/n:5.0f} {sums['p_ent']/n:7.3f} {sums['p_cos']/n:7.4f} │ "
      f"{speedup_avg:5.2f}x")

# Summary section
print(f"\n\n{'SUMMARY':^80}")
print(f"{'─'*80}")
print(f"{'Metric':<30} {'Baseline':>15} {'Pyramid':>15} {'Change':>15}")
print(f"{'─'*80}")

items = [
    ('ViT Encode (ms)',     sums['b_vit']/n,  sums['p_vit']/n),
    ('Prefill (ms)',        sums['b_pre']/n,  sums['p_pre']/n),
    ('Decode (ms)',         sums['b_dec']/n,  sums['p_dec']/n),
    ('End-to-End (ms)',     sums['b_e2e']/n,  sums['p_e2e']/n),
    ('Frames',              sums['b_fr']/n,   sums['p_fr']/n),
    ('Visual Tokens (pre-evict)', sums['b_vis']/n, sums['p_vis']/n),
    ('Visual Tokens (kept)',sums['b_vis']/n,  sums['p_kept']/n),
    ('Embedding Entropy',   sums['b_ent']/n,  sums['p_ent']/n),
    ('Avg Cosine Distance', sums['b_cos']/n,  sums['p_cos']/n),
]

for name, bval, pval in items:
    if bval > 0:
        pct = (pval - bval) / bval * 100
        print(f"{name:<30} {bval:15.1f} {pval:15.1f} {pct:+14.1f}%")
    else:
        print(f"{name:<30} {bval:15.1f} {pval:15.1f} {'N/A':>15}")

# Save JSON
with open(os.path.join(base, "comparison_v2.json"), "w") as f:
    json.dump([{"video": r["video"],
                "baseline": {k: r["b"].get(k,0) for k in
                    ["encode_time_ms","prefill_time_ms","generation_time_ms","total_time_ms",
                     "n_frames","n_visual_tokens","n_visual_tokens_kept",
                     "embedding_entropy","avg_cosine_dist","tokens_per_sec","n_tokens_generated"]},
                "pyramid": {k: r["p"].get(k,0) for k in
                    ["encode_time_ms","prefill_time_ms","generation_time_ms","total_time_ms",
                     "n_frames","n_frames_total","n_visual_tokens","n_visual_tokens_kept",
                     "embedding_entropy","avg_cosine_dist","tokens_per_sec","n_tokens_generated",
                     "n_vit_skipped","n_vit_cropped","n_vit_full","global_complexity","selection_time_ms","eviction_time_ms"]},
               } for r in rows], f, indent=2)
print(f"\nJSON saved to {base}/comparison_v2.json")
