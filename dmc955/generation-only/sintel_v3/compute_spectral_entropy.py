#!/usr/bin/env python3
"""
Compute exact spectral entropy from saved embedding files using numpy SVD.

Spectral entropy = Shannon entropy of normalized singular values.
    H = -Σ p_i * log(p_i)  where  p_i = σ_i / Σσ_i

Uses numpy.linalg.svd (LAPACK dgesdd) for exact computation.
"""

import numpy as np
import json
import os
import csv
import struct
import sys

def load_embeddings(path):
    """Load binary embedding file: [int32 n_tok, int32 n_embd, float32 data...]"""
    with open(path, 'rb') as f:
        n_tok, n_embd = struct.unpack('ii', f.read(8))
        data = np.frombuffer(f.read(n_tok * n_embd * 4), dtype=np.float32)
        return data.reshape(n_tok, n_embd), n_tok, n_embd

def spectral_entropy(X):
    """
    Compute spectral entropy of matrix X [n_tok × n_embd].

    1. Center X (subtract column means)
    2. Full SVD → singular values σ
    3. Normalize: p_i = σ_i / Σσ_i
    4. Shannon entropy: H = -Σ p_i log(p_i)

    Returns: entropy, singular values, effective rank (e^H)
    """
    # Center
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Full SVD (economy mode: only min(n,m) singular values)
    sigma = np.linalg.svd(X_centered, compute_uv=False)

    # Remove near-zero singular values
    sigma = sigma[sigma > 1e-10]

    if len(sigma) == 0:
        return 0.0, np.array([]), 1.0

    # Normalize to probability distribution
    p = sigma / sigma.sum()

    # Shannon entropy
    H = -np.sum(p * np.log(p))

    # Effective rank = e^H
    eff_rank = np.exp(H)

    return float(H), sigma, float(eff_rank)

def avg_cosine_distance(X, n_sample=500, seed=42):
    """Average pairwise cosine distance on a subsample."""
    rng = np.random.RandomState(seed)
    n = min(len(X), n_sample)
    idx = rng.choice(len(X), n, replace=False)
    Xs = X[idx]

    # Normalize rows
    norms = np.linalg.norm(Xs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    Xn = Xs / norms

    # Cosine similarity matrix
    cos_sim = Xn @ Xn.T

    # Upper triangle average (excluding diagonal)
    mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
    avg_cos_dist = 1.0 - cos_sim[mask].mean()

    return float(avg_cos_dist)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.join(base_dir, "baseline")
    pyramid_dir = os.path.join(base_dir, "pyramid")

    videos = sorted([f.replace('.embd', '') for f in os.listdir(baseline_dir) if f.endswith('.embd')])

    print(f"Computing exact spectral entropy for {len(videos)} videos...")
    print(f"{'Video':<14} │ {'B_Tok':>6} {'B_Entropy':>9} {'B_EffRank':>9} {'B_CosDist':>9} │ "
          f"{'P_Tok':>6} {'P_Entropy':>9} {'P_EffRank':>9} {'P_CosDist':>9} │ {'ΔEntropy':>8}")
    print(f"{'─'*14}─┼─{'─'*37}─┼─{'─'*39}─┼─{'─'*8}")

    rows = []

    for vid in videos:
        b_embd_path = os.path.join(baseline_dir, f"{vid}.embd")
        p_embd_path = os.path.join(pyramid_dir, f"{vid}.embd")
        b_json_path = os.path.join(baseline_dir, f"{vid}.json")
        p_json_path = os.path.join(pyramid_dir, f"{vid}.json")

        if not all(os.path.exists(p) for p in [b_embd_path, p_embd_path, b_json_path, p_json_path]):
            continue

        # Load embeddings
        B, bn, bd = load_embeddings(b_embd_path)
        P, pn, pd = load_embeddings(p_embd_path)

        # Load JSON for timing data
        with open(b_json_path) as f: bj = json.load(f)
        with open(p_json_path) as f: pj = json.load(f)

        # Compute exact spectral entropy
        b_ent, b_sigma, b_rank = spectral_entropy(B)
        p_ent, p_sigma, p_rank = spectral_entropy(P)

        # Compute cosine distance
        b_cos = avg_cosine_distance(B)
        p_cos = avg_cosine_distance(P)

        delta = (p_ent / b_ent - 1) * 100 if b_ent > 0 else 0

        print(f"{vid:<14} │ {bn:6d} {b_ent:9.4f} {b_rank:9.1f} {b_cos:9.4f} │ "
              f"{pn:6d} {p_ent:9.4f} {p_rank:9.1f} {p_cos:9.4f} │ {delta:+7.1f}%")

        rows.append({
            'video': vid,
            'b_n_tok': bn, 'b_n_embd': bd,
            'b_spectral_entropy': b_ent, 'b_effective_rank': b_rank,
            'b_avg_cosine_dist': b_cos, 'b_n_singular_values': len(b_sigma),
            'b_top10_sv_share': float(b_sigma[:10].sum() / b_sigma.sum()) if len(b_sigma) > 0 else 0,
            'p_n_tok': pn, 'p_n_embd': pd,
            'p_spectral_entropy': p_ent, 'p_effective_rank': p_rank,
            'p_avg_cosine_dist': p_cos, 'p_n_singular_values': len(p_sigma),
            'p_top10_sv_share': float(p_sigma[:10].sum() / p_sigma.sum()) if len(p_sigma) > 0 else 0,
            'entropy_change_pct': delta,
            # Timing from JSON
            'b_e2e_ms': bj.get('total_time_ms', 0),
            'b_vit_ms': bj.get('encode_time_ms', 0),
            'b_prefill_ms': bj.get('prefill_time_ms', 0),
            'b_decode_ms': bj.get('generation_time_ms', 0),
            'b_frames': bj.get('n_frames', 0),
            'b_vis_tok_kept': bj.get('n_visual_tokens_kept', 0),
            'p_e2e_ms': pj.get('total_time_ms', 0),
            'p_vit_ms': pj.get('encode_time_ms', 0),
            'p_prefill_ms': pj.get('prefill_time_ms', 0),
            'p_decode_ms': pj.get('generation_time_ms', 0),
            'p_frames': pj.get('n_frames', 0),
            'p_vis_tok_kept': pj.get('n_visual_tokens_kept', 0),
            'p_eviction_ms': pj.get('eviction_time_ms', 0),
            'p_selection_ms': pj.get('selection_time_ms', 0),
            'p_complexity': pj.get('global_complexity', 0),
            'speedup': bj.get('total_time_ms', 1) / max(1, pj.get('total_time_ms', 1)),
            'tok_savings_pct': (1 - pj.get('n_visual_tokens_kept', 0) / max(1, bj.get('n_visual_tokens', 1))) * 100,
        })

    # Averages
    n = len(rows)
    def avg(key): return sum(r[key] for r in rows) / n

    print(f"{'─'*14}─┼─{'─'*37}─┼─{'─'*39}─┼─{'─'*8}")
    de = (avg('p_spectral_entropy') / avg('b_spectral_entropy') - 1) * 100
    print(f"{'AVERAGE':<14} │ {avg('b_n_tok'):6.0f} {avg('b_spectral_entropy'):9.4f} {avg('b_effective_rank'):9.1f} {avg('b_avg_cosine_dist'):9.4f} │ "
          f"{avg('p_n_tok'):6.0f} {avg('p_spectral_entropy'):9.4f} {avg('p_effective_rank'):9.1f} {avg('p_avg_cosine_dist'):9.4f} │ {de:+7.1f}%")

    print(f"\n  Max possible entropy = log(min(n_tok, n_embd)) = log(3584) = {np.log(3584):.4f}")
    print(f"  Baseline effective rank: {avg('b_effective_rank'):.1f} independent directions out of 3584 dims")
    print(f"  Pyramid  effective rank: {avg('p_effective_rank'):.1f} independent directions out of 3584 dims")
    print(f"  Top-10 SV share: baseline {avg('b_top10_sv_share')*100:.1f}%, pyramid {avg('p_top10_sv_share')*100:.1f}%")

    # Save CSV
    csv_path = os.path.join(base_dir, "sintel_v3_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'Video',
            'B_ViT_ms', 'B_Prefill_ms', 'B_Decode_ms', 'B_E2E_ms',
            'B_Frames', 'B_VisTok_Kept', 'B_NumEmbeddings',
            'B_SpectralEntropy', 'B_EffectiveRank', 'B_AvgCosDist', 'B_Top10SV_Share',
            'P_ViT_ms', 'P_Prefill_ms', 'P_Decode_ms', 'P_E2E_ms',
            'P_Frames', 'P_VisTok_Kept', 'P_NumEmbeddings',
            'P_SpectralEntropy', 'P_EffectiveRank', 'P_AvgCosDist', 'P_Top10SV_Share',
            'P_Selection_ms', 'P_Eviction_ms', 'P_Complexity',
            'Speedup', 'VisTok_Savings_Pct', 'Entropy_Change_Pct'
        ])

        for r in rows:
            w.writerow([
                r['video'],
                f"{r['b_vit_ms']:.0f}", f"{r['b_prefill_ms']:.0f}", f"{r['b_decode_ms']:.0f}", f"{r['b_e2e_ms']:.0f}",
                r['b_frames'], r['b_vis_tok_kept'], r['b_n_tok'],
                f"{r['b_spectral_entropy']:.4f}", f"{r['b_effective_rank']:.1f}",
                f"{r['b_avg_cosine_dist']:.4f}", f"{r['b_top10_sv_share']:.4f}",
                f"{r['p_vit_ms']:.0f}", f"{r['p_prefill_ms']:.0f}", f"{r['p_decode_ms']:.0f}", f"{r['p_e2e_ms']:.0f}",
                r['p_frames'], r['p_vis_tok_kept'], r['p_n_tok'],
                f"{r['p_spectral_entropy']:.4f}", f"{r['p_effective_rank']:.1f}",
                f"{r['p_avg_cosine_dist']:.4f}", f"{r['p_top10_sv_share']:.4f}",
                f"{r['p_selection_ms']:.0f}", f"{r['p_eviction_ms']:.0f}",
                f"{r['p_complexity']:.3f}",
                f"{r['speedup']:.2f}", f"{r['tok_savings_pct']:.1f}", f"{r['entropy_change_pct']:.1f}"
            ])

        # Average row
        w.writerow([
            'AVERAGE',
            f"{avg('b_vit_ms'):.0f}", f"{avg('b_prefill_ms'):.0f}", f"{avg('b_decode_ms'):.0f}", f"{avg('b_e2e_ms'):.0f}",
            f"{avg('b_frames'):.0f}", f"{avg('b_vis_tok_kept'):.0f}", f"{avg('b_n_tok'):.0f}",
            f"{avg('b_spectral_entropy'):.4f}", f"{avg('b_effective_rank'):.1f}",
            f"{avg('b_avg_cosine_dist'):.4f}", f"{avg('b_top10_sv_share'):.4f}",
            f"{avg('p_vit_ms'):.0f}", f"{avg('p_prefill_ms'):.0f}", f"{avg('p_decode_ms'):.0f}", f"{avg('p_e2e_ms'):.0f}",
            f"{avg('p_frames'):.0f}", f"{avg('p_vis_tok_kept'):.0f}", f"{avg('p_n_tok'):.0f}",
            f"{avg('p_spectral_entropy'):.4f}", f"{avg('p_effective_rank'):.1f}",
            f"{avg('p_avg_cosine_dist'):.4f}", f"{avg('p_top10_sv_share'):.4f}",
            f"{avg('p_selection_ms'):.0f}", f"{avg('p_eviction_ms'):.0f}",
            f"{avg('p_complexity'):.3f}",
            f"{avg('speedup'):.2f}", f"{avg('tok_savings_pct'):.1f}", f"{de:.1f}"
        ])

    print(f"\nCSV saved: {csv_path}")

if __name__ == '__main__':
    main()
