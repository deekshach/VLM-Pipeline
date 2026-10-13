#!/usr/bin/env python3
"""
MSVD Evaluation: compare Baseline vs Emma on accuracy + latency metrics.
Uses pycocoevalcap (BLEU, METEOR, ROUGE-L, CIDEr) and BERTScore.
"""

import os
import sys
import json
import struct
import numpy as np

# Metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

MANIFEST = "/data/spring26/CPP/datasets/MSVD/MSVD_prepared/manifest.json"
RESULTS_DIR = "/data/spring26/CPP/llama.cpp/dmc955/generation-only/msvd_results"
REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.json")


def load_results(mode, video_ids):
    """Load JSON results for a mode (baseline/emma)."""
    results = {}
    for vid in video_ids:
        path = os.path.join(RESULTS_DIR, mode, f"{vid}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[vid] = json.load(f)
    return results


def compute_coco_metrics(predictions, references):
    """
    Compute BLEU-1..4, METEOR, ROUGE-L, CIDEr using pycocoevalcap.
    predictions: {vid: "generated text"}
    references: {vid: ["ref1", "ref2", ...]}
    """
    # pycocoevalcap expects {id: [str]} format
    gts = {}
    res = {}
    for vid in predictions:
        if vid not in references:
            continue
        gts[vid] = references[vid]
        res[vid] = [predictions[vid]]

    if not gts:
        return {}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    metrics = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                metrics[m] = s
        else:
            metrics[method] = score

    return metrics


def compute_bertscore(predictions, references):
    """Compute BERTScore (F1) using bert_score package."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None

    # Use first reference per video for BERTScore
    vids = [v for v in predictions if v in references]
    if not vids:
        return None

    cands = [predictions[v] for v in vids]
    refs = [references[v][0] for v in vids]  # first reference

    P, R, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)
    return float(F1.mean())


def read_embd(path):
    """Read embedding binary file."""
    with open(path, 'rb') as f:
        n_tok, n_embd = struct.unpack('ii', f.read(8))
        data = np.frombuffer(f.read(n_tok * n_embd * 4), dtype=np.float32)
        return data.reshape(n_tok, n_embd)


def compute_spectral_entropy(embd, max_sample=200):
    """Compute spectral entropy of embeddings."""
    n_tok = embd.shape[0]
    if n_tok < 2:
        return 0.0

    idx = np.random.RandomState(42).choice(n_tok, min(n_tok, max_sample), replace=False)
    X = embd[idx].astype(np.float64)
    X -= X.mean(axis=0)

    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S = S[S > 1e-10]
    p = S / S.sum()
    return float(-np.sum(p * np.log(p)))


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    video_ids = [v['video_id'] for v in manifest]
    references = {v['video_id']: v['captions'] for v in manifest}

    print(f"Evaluating {len(video_ids)} videos\n")

    # Load results
    baseline = load_results("baseline", video_ids)
    emma = load_results("emma", video_ids)

    print(f"Baseline results: {len(baseline)}/{len(video_ids)}")
    print(f"Emma results: {len(emma)}/{len(video_ids)}")

    # Common videos (both succeeded)
    common = sorted(set(baseline.keys()) & set(emma.keys()))
    print(f"Common (both succeeded): {len(common)}\n")

    if not common:
        print("No common results to evaluate!")
        return

    # =========================================================
    # 1. ACCURACY METRICS
    # =========================================================
    print("=" * 60)
    print("1. ACCURACY METRICS")
    print("=" * 60)

    base_preds = {v: baseline[v]['generated_text'] for v in common}
    emma_preds = {v: emma[v]['generated_text'] for v in common}
    refs = {v: references[v] for v in common if v in references}

    print("\nComputing BLEU, METEOR, ROUGE-L, CIDEr...")
    base_metrics = compute_coco_metrics(base_preds, refs)
    emma_metrics = compute_coco_metrics(emma_preds, refs)

    print(f"\n{'Metric':<12} {'Baseline':>10} {'Emma':>10} {'Delta':>10}")
    print("-" * 45)
    for m in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
        bv = base_metrics.get(m, 0)
        ev = emma_metrics.get(m, 0)
        delta = ev - bv
        sign = "+" if delta >= 0 else ""
        print(f"{m:<12} {bv:>10.4f} {ev:>10.4f} {sign}{delta:>9.4f}")

    print("\nComputing BERTScore...")
    try:
        base_bert = compute_bertscore(base_preds, refs)
        emma_bert = compute_bertscore(emma_preds, refs)
    except Exception as e:
        print(f"  BERTScore failed: {e}")
        base_bert = None
        emma_bert = None
    if base_bert is not None and emma_bert is not None:
        delta = emma_bert - base_bert
        sign = "+" if delta >= 0 else ""
        print(f"{'BERTScore':<12} {base_bert:>10.4f} {emma_bert:>10.4f} {sign}{delta:>9.4f}")

    # =========================================================
    # 2. LATENCY METRICS
    # =========================================================
    print("\n" + "=" * 60)
    print("2. LATENCY METRICS (averaged over all videos)")
    print("=" * 60)

    latency_fields = [
        ('encode_time_ms', 'ViT Encode'),
        ('prefill_time_ms', 'Total Prefill'),
        ('eviction_time_ms', 'KV Eviction'),
        ('generation_time_ms', 'Generation'),
        ('total_time_ms', 'Total'),
    ]

    print(f"\n{'Stage':<20} {'Baseline':>12} {'Emma':>12} {'Speedup':>10}")
    print("-" * 58)

    report_latency = {}
    for field, label in latency_fields:
        bvals = [baseline[v].get(field, 0) for v in common]
        evals = [emma[v].get(field, 0) for v in common]
        bavg = np.mean(bvals)
        eavg = np.mean(evals)
        speedup = bavg / eavg if eavg > 0 else float('inf')
        print(f"{label:<20} {bavg:>10.1f}ms {eavg:>10.1f}ms {speedup:>9.2f}x")
        report_latency[field] = {'baseline': float(bavg), 'emma': float(eavg), 'speedup': float(speedup)}

    # LLM-only prefill
    bpre = [baseline[v]['prefill_time_ms'] - baseline[v].get('encode_time_ms', 0) for v in common]
    epre = [emma[v]['prefill_time_ms'] - emma[v].get('encode_time_ms', 0) for v in common]
    bavg = np.mean(bpre)
    eavg = np.mean(epre)
    print(f"{'LLM Prefill':<20} {bavg:>10.1f}ms {eavg:>10.1f}ms {bavg/eavg if eavg > 0 else 0:>9.2f}x")

    # tok/s
    btps = [baseline[v].get('tokens_per_sec', 0) for v in common]
    etps = [emma[v].get('tokens_per_sec', 0) for v in common]
    print(f"\n{'Decode tok/s':<20} {np.mean(btps):>10.1f}    {np.mean(etps):>10.1f}    {np.mean(etps)/np.mean(btps) if np.mean(btps) > 0 else 0:>8.2f}x")

    # =========================================================
    # 3. TOKEN REDUCTION
    # =========================================================
    print("\n" + "=" * 60)
    print("3. VISUAL TOKEN REDUCTION")
    print("=" * 60)

    bvt = [baseline[v].get('n_visual_tokens', 0) for v in common]
    evt = [emma[v].get('n_visual_tokens', 0) for v in common]
    bvk = [baseline[v].get('n_visual_tokens_kept', 0) for v in common]
    evk = [emma[v].get('n_visual_tokens_kept', 0) for v in common]

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Emma':>12}")
    print("-" * 58)
    print(f"{'Avg Visual Tokens (prefill)':<30} {np.mean(bvt):>12.0f} {np.mean(evt):>12.0f}")
    print(f"{'Avg Tokens Kept (post-evict)':<30} {np.mean(bvk):>12.0f} {np.mean(evk):>12.0f}")
    reduction = (1 - np.mean(evk) / np.mean(bvt)) * 100 if np.mean(bvt) > 0 else 0
    print(f"{'KV Reduction vs Baseline':<30} {'—':>12} {reduction:>11.1f}%")

    # =========================================================
    # 4. SPECTRAL ENTROPY (Information Retained)
    # =========================================================
    print("\n" + "=" * 60)
    print("4. INFORMATION RETENTION (Spectral Entropy)")
    print("=" * 60)

    base_ent = []
    emma_ent = []
    for v in common:
        for mode, ent_list in [("baseline", base_ent), ("emma", emma_ent)]:
            embd_path = os.path.join(RESULTS_DIR, mode, f"{v}.embd")
            if os.path.exists(embd_path):
                embd = read_embd(embd_path)
                ent_list.append(compute_spectral_entropy(embd))

    if base_ent and emma_ent:
        print(f"\n{'Metric':<30} {'Baseline':>12} {'Emma':>12}")
        print("-" * 58)
        print(f"{'Avg Spectral Entropy':<30} {np.mean(base_ent):>12.4f} {np.mean(emma_ent):>12.4f}")
        print(f"{'Std Spectral Entropy':<30} {np.std(base_ent):>12.4f} {np.std(emma_ent):>12.4f}")
        ratio = np.mean(emma_ent) / np.mean(base_ent) * 100 if np.mean(base_ent) > 0 else 0
        print(f"{'Entropy Retained':<30} {'100.0%':>12} {ratio:>11.1f}%")
    else:
        print("  No embedding files found for entropy computation")

    # =========================================================
    # 5. PER-VIDEO DETAIL TABLE
    # =========================================================
    print("\n" + "=" * 60)
    print("5. PER-VIDEO COMPARISON")
    print("=" * 60)

    print(f"\n{'Video':<30} {'B_Total':>8} {'E_Total':>8} {'Speed':>6} {'B_Tok':>6} {'E_Tok':>6} {'B_Kept':>7} {'E_Kept':>7}")
    print("-" * 85)
    for v in common:
        bd = baseline[v]
        ed = emma[v]
        speedup = bd['total_time_ms'] / ed['total_time_ms'] if ed['total_time_ms'] > 0 else 0
        print(f"{v[:29]:<30} {bd['total_time_ms']:>7.0f}ms {ed['total_time_ms']:>7.0f}ms {speedup:>5.2f}x {bd['n_tokens_generated']:>6} {ed['n_tokens_generated']:>6} {bd['n_visual_tokens_kept']:>7} {ed['n_visual_tokens_kept']:>7}")

    # =========================================================
    # 6. GENERATED TEXT SAMPLES
    # =========================================================
    print("\n" + "=" * 60)
    print("6. GENERATED TEXT SAMPLES (first 5 videos)")
    print("=" * 60)

    for v in common[:5]:
        print(f"\n--- {v} ---")
        print(f"  Reference: {references.get(v, ['N/A'])[0]}")
        print(f"  Baseline:  {baseline[v]['generated_text'][:200]}")
        print(f"  Emma:      {emma[v]['generated_text'][:200]}")

    # =========================================================
    # Save full report as JSON
    # =========================================================
    report = {
        'n_videos': len(common),
        'accuracy': {
            'baseline': base_metrics,
            'emma': emma_metrics,
        },
        'bertscore': {
            'baseline': base_bert,
            'emma': emma_bert,
        },
        'latency': report_latency,
        'token_reduction': {
            'baseline_avg_visual_tokens': float(np.mean(bvt)),
            'emma_avg_visual_tokens': float(np.mean(evt)),
            'baseline_avg_kept': float(np.mean(bvk)),
            'emma_avg_kept': float(np.mean(evk)),
            'kv_reduction_pct': float(reduction),
        },
        'spectral_entropy': {
            'baseline_mean': float(np.mean(base_ent)) if base_ent else None,
            'emma_mean': float(np.mean(emma_ent)) if emma_ent else None,
        },
    }

    if base_bert is not None:
        report['accuracy']['baseline']['BERTScore'] = base_bert
    if emma_bert is not None:
        report['accuracy']['emma']['BERTScore'] = emma_bert

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
