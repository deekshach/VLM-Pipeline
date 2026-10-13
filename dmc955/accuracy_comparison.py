#!/usr/bin/env python3
"""
Accuracy comparison: Adaptive vs Baseline VLM outputs on MPI-Sintel.

Metrics computed:
  - BLEU-1, BLEU-2, BLEU-4 (n-gram precision)
  - ROUGE-1, ROUGE-2, ROUGE-L (recall-oriented overlap)
  - Jaccard similarity (word-level set overlap)
  - Cosine similarity (TF-based word vectors)

Uses baseline output as the reference and adaptive output as the hypothesis.
"""

import json
import os
import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

RESULTS_DIR = "/data/spring26/CPP/llama.cpp/dmc955/results"


def load_texts():
    """Load baseline and adaptive generated texts for all videos."""
    videos = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith("_baseline.json"):
            video_name = f.replace("_baseline.json", "")
            baseline_path = os.path.join(RESULTS_DIR, f)
            adaptive_path = os.path.join(RESULTS_DIR, f"{video_name}_adaptive.json")
            if not os.path.exists(adaptive_path):
                continue
            with open(baseline_path) as bf:
                baseline = json.load(bf)
            with open(adaptive_path) as af:
                adaptive = json.load(af)
            videos.append({
                "video": video_name,
                "text_baseline": baseline["inference"]["generated_text"],
                "text_adaptive": adaptive["inference"]["generated_text"],
                "frames_baseline": baseline["n_frames_selected"],
                "frames_adaptive": adaptive["n_frames_selected"],
                "complexity": adaptive.get("global_complexity", 0),
            })
    return videos


def compute_bleu(reference, hypothesis):
    """Compute BLEU-1, BLEU-2, BLEU-4 scores."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return bleu1, bleu2, bleu4


def compute_rouge(reference, hypothesis):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return (scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure)


def compute_jaccard(text_a, text_b):
    """Word-level Jaccard similarity."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def compute_cosine(text_a, text_b):
    """TF-based cosine similarity (no external deps)."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    counter_a = Counter(words_a)
    counter_b = Counter(words_b)
    all_words = set(counter_a.keys()) | set(counter_b.keys())
    dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v ** 2 for v in counter_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in counter_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def main():
    videos = load_texts()
    if not videos:
        print("No results found!")
        return

    print("=" * 110)
    print("  Accuracy Comparison: Adaptive vs Baseline (baseline = reference)")
    print("=" * 110)

    # Per-video results
    results = []
    for v in videos:
        bleu1, bleu2, bleu4 = compute_bleu(v["text_baseline"], v["text_adaptive"])
        rouge1, rouge2, rougeL = compute_rouge(v["text_baseline"], v["text_adaptive"])
        jaccard = compute_jaccard(v["text_baseline"], v["text_adaptive"])
        cosine = compute_cosine(v["text_baseline"], v["text_adaptive"])
        results.append({
            "video": v["video"],
            "frames_b": v["frames_baseline"],
            "frames_a": v["frames_adaptive"],
            "complexity": v["complexity"],
            "bleu1": bleu1, "bleu2": bleu2, "bleu4": bleu4,
            "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL,
            "jaccard": jaccard, "cosine": cosine,
        })

    # Table 1: BLEU scores
    print()
    print("-" * 95)
    print(f"  {'Video':<16} {'Frames':>8} {'Cmplx':>7} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-4':>8} {'Jaccard':>9} {'Cosine':>8}")
    print("-" * 95)
    for r in results:
        frames_str = f"{r['frames_a']}/{r['frames_b']}"
        print(f"  {r['video']:<16} {frames_str:>8} {r['complexity']:>7.3f} "
              f"{r['bleu1']:>8.3f} {r['bleu2']:>8.3f} {r['bleu4']:>8.3f} "
              f"{r['jaccard']:>9.3f} {r['cosine']:>8.3f}")
    print("-" * 95)

    # Averages
    n = len(results)
    avg = {k: sum(r[k] for r in results) / n for k in ["bleu1", "bleu2", "bleu4", "jaccard", "cosine"]}
    print(f"  {'AVERAGE':<16} {'':>8} {'':>7} "
          f"{avg['bleu1']:>8.3f} {avg['bleu2']:>8.3f} {avg['bleu4']:>8.3f} "
          f"{avg['jaccard']:>9.3f} {avg['cosine']:>8.3f}")

    # Table 2: ROUGE scores
    print()
    print("-" * 80)
    print(f"  {'Video':<16} {'Frames':>8} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
    print("-" * 80)
    for r in results:
        frames_str = f"{r['frames_a']}/{r['frames_b']}"
        print(f"  {r['video']:<16} {frames_str:>8} "
              f"{r['rouge1']:>10.3f} {r['rouge2']:>10.3f} {r['rougeL']:>10.3f}")
    print("-" * 80)
    avg_rouge = {k: sum(r[k] for r in results) / n for k in ["rouge1", "rouge2", "rougeL"]}
    print(f"  {'AVERAGE':<16} {'':>8} "
          f"{avg_rouge['rouge1']:>10.3f} {avg_rouge['rouge2']:>10.3f} {avg_rouge['rougeL']:>10.3f}")

    # Summary
    print()
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"  Videos analyzed:        {n}")
    print(f"  Avg BLEU-1:             {avg['bleu1']:.3f}")
    print(f"  Avg BLEU-2:             {avg['bleu2']:.3f}")
    print(f"  Avg BLEU-4:             {avg['bleu4']:.3f}")
    print(f"  Avg ROUGE-1 (F1):       {avg_rouge['rouge1']:.3f}")
    print(f"  Avg ROUGE-2 (F1):       {avg_rouge['rouge2']:.3f}")
    print(f"  Avg ROUGE-L (F1):       {avg_rouge['rougeL']:.3f}")
    print(f"  Avg Jaccard similarity:  {avg['jaccard']:.3f}")
    print(f"  Avg Cosine similarity:   {avg['cosine']:.3f}")

    # Correlation: complexity vs accuracy
    print()
    print("-" * 80)
    print("  Accuracy by Complexity Tier")
    print("-" * 80)
    tiers = [
        ("Low  (< 0.2)", [r for r in results if r["complexity"] < 0.2]),
        ("Med  (0.2-0.6)", [r for r in results if 0.2 <= r["complexity"] < 0.6]),
        ("High (>= 0.6)", [r for r in results if r["complexity"] >= 0.6]),
    ]
    print(f"  {'Tier':<18} {'Count':>6} {'BLEU-1':>8} {'ROUGE-1':>9} {'ROUGE-L':>9} {'Cosine':>8}")
    print(f"  {'-'*16:<18} {'-'*5:>6} {'-'*7:>8} {'-'*8:>9} {'-'*8:>9} {'-'*7:>8}")
    for label, tier_results in tiers:
        if not tier_results:
            continue
        tn = len(tier_results)
        tb1 = sum(r["bleu1"] for r in tier_results) / tn
        tr1 = sum(r["rouge1"] for r in tier_results) / tn
        trl = sum(r["rougeL"] for r in tier_results) / tn
        tc = sum(r["cosine"] for r in tier_results) / tn
        print(f"  {label:<18} {tn:>6} {tb1:>8.3f} {tr1:>9.3f} {trl:>9.3f} {tc:>8.3f}")

    # Save results
    output = {
        "n_videos": n,
        "avg_bleu1": avg["bleu1"],
        "avg_bleu2": avg["bleu2"],
        "avg_bleu4": avg["bleu4"],
        "avg_rouge1": avg_rouge["rouge1"],
        "avg_rouge2": avg_rouge["rouge2"],
        "avg_rougeL": avg_rouge["rougeL"],
        "avg_jaccard": avg["jaccard"],
        "avg_cosine": avg["cosine"],
        "per_video": results,
    }
    out_path = os.path.join(RESULTS_DIR, "accuracy_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"  Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
