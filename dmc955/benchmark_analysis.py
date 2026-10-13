#!/usr/bin/env python3
"""
Streaming VLM Benchmark Analysis

Compares baseline vs optimized streaming VLM outputs using:
- BLEU-1/2/4 (nltk)
- ROUGE-1/2/L (rouge_score)
- Jaccard similarity
- Cosine similarity (TF-IDF)
- Latency breakdown and encoder savings
- Shannon entropy comparison
- Per-video detailed comparison tables

Usage:
    python3 benchmark_analysis.py ./streaming_results
    python3 benchmark_analysis.py ./streaming_results --output summary.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Optional imports with graceful fallback
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("WARNING: nltk not installed. BLEU scores will be unavailable.", file=sys.stderr)
    print("  Install with: pip install nltk", file=sys.stderr)

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("WARNING: rouge_score not installed. ROUGE scores will be unavailable.", file=sys.stderr)
    print("  Install with: pip install rouge-score", file=sys.stderr)


def tokenize_simple(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


def compute_bleu(reference, hypothesis):
    """Compute BLEU-1/2/4 scores."""
    if not HAS_NLTK:
        return {"bleu_1": None, "bleu_2": None, "bleu_4": None}

    ref_tokens = tokenize_simple(reference)
    hyp_tokens = tokenize_simple(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    smooth = SmoothingFunction().method1

    bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_4": bleu_4}


def compute_rouge(reference, hypothesis):
    """Compute ROUGE-1/2/L scores."""
    if not HAS_ROUGE:
        return {"rouge_1": None, "rouge_2": None, "rouge_l": None}

    if not reference.strip() or not hypothesis.strip():
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge_1": scores['rouge1'].fmeasure,
        "rouge_2": scores['rouge2'].fmeasure,
        "rouge_l": scores['rougeL'].fmeasure,
    }


def compute_jaccard(text_a, text_b):
    """Compute Jaccard similarity between two texts."""
    tokens_a = set(tokenize_simple(text_a))
    tokens_b = set(tokenize_simple(text_b))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_cosine(text_a, text_b):
    """Compute cosine similarity using TF vectors."""
    tokens_a = tokenize_simple(text_a)
    tokens_b = tokenize_simple(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    all_words = set(counter_a.keys()) | set(counter_b.keys())

    dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
    norm_a = sum(v ** 2 for v in counter_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in counter_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def load_result(path):
    """Load a JSON result file."""
    with open(path, 'r') as f:
        return json.load(f)


def find_paired_results(results_dir):
    """Find paired baseline/optimized result files."""
    baseline_files = sorted(glob.glob(os.path.join(results_dir, "streaming_baseline_*.json")))
    pairs = []

    for bf in baseline_files:
        video_name = os.path.basename(bf).replace("streaming_baseline_", "").replace(".json", "")
        # Look for optimized counterparts
        opt_path = os.path.join(results_dir, f"streaming_optimized_{video_name}.json")
        opt_pruning_path = os.path.join(results_dir, f"streaming_optimized_pruning_{video_name}.json")
        opt_prellm_path = os.path.join(results_dir, f"streaming_optimized_prellm_{video_name}.json")
        opt_prellm_pruning_path = os.path.join(results_dir, f"streaming_optimized_prellm_pruning_{video_name}.json")

        opt_files = []
        if os.path.exists(opt_path):
            opt_files.append(("optimized", opt_path))
        if os.path.exists(opt_pruning_path):
            opt_files.append(("optimized+pruning", opt_pruning_path))
        if os.path.exists(opt_prellm_path):
            opt_files.append(("optimized+prellm", opt_prellm_path))
        if os.path.exists(opt_prellm_pruning_path):
            opt_files.append(("optimized+prellm+pruning", opt_prellm_pruning_path))

        if opt_files:
            pairs.append({
                "video": video_name,
                "baseline": bf,
                "optimized_variants": opt_files,
            })

    return pairs


def analyze_pair(baseline_data, optimized_data, variant_name):
    """Analyze a single baseline vs optimized pair."""
    result = {
        "variant": variant_name,
        "video": baseline_data.get("video_name", "unknown"),
    }

    # Extract generated text
    baseline_text = baseline_data.get("inference", {}).get("generated_text", "")
    optimized_text = optimized_data.get("inference", {}).get("generated_text", "")

    # Warn about empty text (usually means context window overflow)
    result["warnings"] = []
    if not baseline_text.strip():
        msg = f"WARNING: {result['video']} baseline has empty generated text (context window too small?)"
        print(msg, file=sys.stderr)
        result["warnings"].append("baseline_empty_text")
    if not optimized_text.strip():
        msg = f"WARNING: {result['video']} {variant_name} has empty generated text (context window too small?)"
        print(msg, file=sys.stderr)
        result["warnings"].append("optimized_empty_text")

    # Quality metrics (meaningful only if both texts are non-empty)
    result["quality"] = {}
    if baseline_text.strip() and optimized_text.strip():
        result["quality"].update(compute_bleu(baseline_text, optimized_text))
        result["quality"].update(compute_rouge(baseline_text, optimized_text))
        result["quality"]["jaccard"] = compute_jaccard(baseline_text, optimized_text)
        result["quality"]["cosine"] = compute_cosine(baseline_text, optimized_text)
    else:
        result["quality"] = {
            "bleu_1": None, "bleu_2": None, "bleu_4": None,
            "rouge_1": None, "rouge_2": None, "rouge_l": None,
            "jaccard": None, "cosine": None,
        }

    # Latency breakdown
    bt = baseline_data.get("timing", {})
    ot = optimized_data.get("timing", {})

    result["latency"] = {
        "baseline": {
            "selection_ms": bt.get("selection_ms", 0),
            "encode_ms": bt.get("encode_ms", 0),
            "generation_ms": bt.get("generation_ms", 0),
            "total_ms": bt.get("total_ms", 0),
        },
        "optimized": {
            "selection_ms": ot.get("selection_ms", 0),
            "encode_ms": ot.get("encode_ms", 0),
            "generation_ms": ot.get("generation_ms", 0),
            "pruning_overhead_ms": ot.get("pruning_overhead_ms", 0),
            "total_ms": ot.get("total_ms", 0),
        },
    }

    b_total = bt.get("total_ms", 1)
    o_total = ot.get("total_ms", 1)
    result["latency"]["speedup"] = b_total / o_total if o_total > 0 else 0

    b_encode = bt.get("encode_ms", 1)
    o_encode = ot.get("encode_ms", 1)
    result["latency"]["encode_speedup"] = b_encode / o_encode if o_encode > 0 else 0

    # Encoder savings
    bm = baseline_data.get("metrics", {})
    om = optimized_data.get("metrics", {})

    result["encoder"] = {
        "baseline_tokens_encoded": bm.get("total_tokens_encoded", 0),
        "baseline_tokens_cached": bm.get("total_tokens_cached", 0),
        "optimized_tokens_encoded": om.get("total_tokens_encoded", 0),
        "optimized_tokens_cached": om.get("total_tokens_cached", 0),
        "optimized_encoder_savings_pct": om.get("encoder_savings_pct", 0),
        "optimized_encode_speedup": om.get("encode_speedup", 1.0),
    }

    # Frame selection
    result["selection"] = {
        "baseline_frames": baseline_data.get("n_frames_selected", 0),
        "optimized_frames": optimized_data.get("n_frames_selected", 0),
        "total_frames": baseline_data.get("n_frames_total", 0),
        "baseline_method": baseline_data.get("streaming", {}).get("online_selection", {}).get("method", "unknown"),
        "optimized_method": optimized_data.get("streaming", {}).get("online_selection", {}).get("method", "unknown"),
    }

    # Shannon entropy
    result["entropy"] = {
        "baseline": bm.get("shannon_entropy", None),
        "optimized": om.get("shannon_entropy", None),
    }

    # Pruning summary
    pruning = optimized_data.get("pruning", {})
    if pruning.get("enabled", False):
        result["pruning"] = {
            "initial_tokens": pruning.get("n_visual_tokens_initial", 0),
            "after_round1": pruning.get("n_visual_tokens_after_round1", 0),
            "after_round2": pruning.get("n_visual_tokens_after_round2", 0),
            "total_reduction_pct": pruning.get("total_reduction_pct", 0),
            "overhead_ms": pruning.get("pruning_overhead_ms", 0),
        }

    prefill = optimized_data.get("prefill_pruning", {})
    if prefill.get("enabled", False):
        result["prefill_pruning"] = {
            "tokens_pruned": prefill.get("n_tokens_pruned", 0),
            "overhead_ms": prefill.get("overhead_ms", 0),
        }

    pre_llm = optimized_data.get("pre_llm_pruning", {})
    if pre_llm.get("enabled", False):
        result["pre_llm_pruning"] = {
            "tokens_before": pre_llm.get("total_tokens_before", 0),
            "tokens_injected": pre_llm.get("total_tokens_injected", 0),
            "tokens_pruned": pre_llm.get("total_tokens_pruned", 0),
            "savings_pct": pre_llm.get("savings_pct", 0),
            "keep_ratio": pre_llm.get("keep_ratio", 0),
        }

    # Tokens generated
    result["generation"] = {
        "baseline_tokens": baseline_data.get("inference", {}).get("n_tokens", 0),
        "optimized_tokens": optimized_data.get("inference", {}).get("n_tokens", 0),
    }

    return result


def print_table(headers, rows, title=None):
    """Print a formatted table."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def format_float(v, decimals=2):
    """Format a float or return 'N/A'."""
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Streaming VLM Benchmark Analysis")
    parser.add_argument("results_dir", help="Directory with streaming result JSON files")
    parser.add_argument("--output", "-o", help="Output summary JSON path",
                        default="streaming_benchmark_summary.json")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pairs = find_paired_results(args.results_dir)
    if not pairs:
        print(f"No paired baseline/optimized results found in {args.results_dir}", file=sys.stderr)
        print("Expected files: streaming_baseline_<video>.json + streaming_optimized_<video>.json")
        sys.exit(1)

    print(f"Found {len(pairs)} video pairs to analyze")

    all_analyses = []

    for pair in pairs:
        baseline_data = load_result(pair["baseline"])
        for variant_name, opt_path in pair["optimized_variants"]:
            optimized_data = load_result(opt_path)
            analysis = analyze_pair(baseline_data, optimized_data, variant_name)
            all_analyses.append(analysis)

    # =====================================================================
    # Table 1: Comprehensive Side-by-Side Comparison
    # =====================================================================
    comp_headers = [
        "Video", "Variant",
        "B.Frames", "O.Frames",
        "B.Encode(ms)", "O.Encode(ms)", "Enc.Spdup",
        "B.Gen(ms)", "O.Gen(ms)", "Gen.Spdup",
        "B.Total(ms)", "O.Total(ms)", "Tot.Spdup",
        "B.Tokens", "O.Tokens",
        "B.Entropy", "O.Entropy",
    ]
    comp_rows = []
    for a in all_analyses:
        bl = a["latency"]["baseline"]
        ol = a["latency"]["optimized"]
        gen = a["generation"]
        ent = a["entropy"]
        sel = a["selection"]

        b_gen = bl["generation_ms"]
        o_gen = ol["generation_ms"]
        gen_speedup = b_gen / o_gen if o_gen > 0 else 0

        comp_rows.append([
            a["video"], a["variant"],
            sel["baseline_frames"], sel["optimized_frames"],
            f"{bl['encode_ms']:.0f}", f"{ol['encode_ms']:.0f}",
            f"{a['latency']['encode_speedup']:.2f}x",
            f"{b_gen:.0f}", f"{o_gen:.0f}",
            f"{gen_speedup:.2f}x",
            f"{bl['total_ms']:.0f}", f"{ol['total_ms']:.0f}",
            f"{a['latency']['speedup']:.2f}x",
            gen["baseline_tokens"], gen["optimized_tokens"],
            format_float(ent.get("baseline"), 3),
            format_float(ent.get("optimized"), 3),
        ])
    print_table(comp_headers, comp_rows, "Comprehensive Baseline vs Optimized Comparison")

    # =====================================================================
    # Table 2: Quality Metrics
    # =====================================================================
    quality_headers = ["Video", "Variant", "BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Jaccard", "Cosine"]
    quality_rows = []
    for a in all_analyses:
        q = a["quality"]
        quality_rows.append([
            a["video"], a["variant"],
            format_float(q.get("bleu_1")), format_float(q.get("bleu_2")), format_float(q.get("bleu_4")),
            format_float(q.get("rouge_1")), format_float(q.get("rouge_2")), format_float(q.get("rouge_l")),
            format_float(q.get("jaccard")), format_float(q.get("cosine")),
        ])
    print_table(quality_headers, quality_rows, "Quality Metrics (Optimized vs Baseline)")

    # =====================================================================
    # Table 3: Encoder Savings
    # =====================================================================
    encoder_headers = ["Video", "Variant", "B.Encoded", "B.Cached", "O.Encoded", "O.Cached",
                       "Savings%", "Enc.Speedup"]
    encoder_rows = []
    for a in all_analyses:
        e = a["encoder"]
        encoder_rows.append([
            a["video"], a["variant"],
            e["baseline_tokens_encoded"], e["baseline_tokens_cached"],
            e["optimized_tokens_encoded"], e["optimized_tokens_cached"],
            f"{e['optimized_encoder_savings_pct']:.1f}%",
            f"{e['optimized_encode_speedup']:.2f}x",
        ])
    print_table(encoder_headers, encoder_rows, "Encoder Savings")

    # =====================================================================
    # Table 4: Pruning Summary (if any)
    # =====================================================================
    pruning_rows = []
    for a in all_analyses:
        if "pruning" in a:
            p = a["pruning"]
            pf = a.get("prefill_pruning", {})
            pruning_rows.append([
                a["video"], a["variant"],
                p["initial_tokens"], p["after_round1"], p["after_round2"],
                f"{p['total_reduction_pct']:.1f}%", f"{p['overhead_ms']:.1f}ms",
                pf.get("tokens_pruned", "N/A"), f"{pf.get('overhead_ms', 0):.0f}ms",
            ])
    if pruning_rows:
        pruning_headers = ["Video", "Variant", "Initial", "After R1", "After R2",
                           "Reduction", "Prn.OH", "Prefill.Pruned", "Prefill.OH"]
        print_table(pruning_headers, pruning_rows, "Pruning Summary")

    # =====================================================================
    # Table 5: Pre-LLM Pruning Summary (if any)
    # =====================================================================
    prellm_rows = []
    for a in all_analyses:
        if "pre_llm_pruning" in a:
            p = a["pre_llm_pruning"]
            prellm_rows.append([
                a["video"], a["variant"],
                p["tokens_before"], p["tokens_injected"], p["tokens_pruned"],
                f"{p['savings_pct']:.1f}%", f"{p['keep_ratio']:.2f}",
            ])
    if prellm_rows:
        prellm_headers = ["Video", "Variant", "Full Tokens", "Injected", "Pruned",
                          "Savings%", "KeepRatio"]
        print_table(prellm_headers, prellm_rows, "Pre-LLM Pruning Summary")

    # =====================================================================
    # Aggregate Statistics
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  Aggregate Statistics")
    print(f"{'=' * 70}")

    n = len(all_analyses)
    if n > 0:
        def safe_avg(key_path):
            values = []
            for a in all_analyses:
                obj = a
                for k in key_path:
                    obj = obj.get(k, {}) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                if obj is not None and isinstance(obj, (int, float)):
                    values.append(obj)
            return sum(values) / len(values) if values else None

        avg_bleu1 = safe_avg(["quality", "bleu_1"])
        avg_rouge_l = safe_avg(["quality", "rouge_l"])
        avg_jaccard = safe_avg(["quality", "jaccard"])
        avg_cosine = safe_avg(["quality", "cosine"])
        avg_speedup = safe_avg(["latency", "speedup"])
        avg_enc_savings = safe_avg(["encoder", "optimized_encoder_savings_pct"])
        avg_enc_speedup = safe_avg(["encoder", "optimized_encode_speedup"])
        avg_b_entropy = safe_avg(["entropy", "baseline"])
        avg_o_entropy = safe_avg(["entropy", "optimized"])
        avg_b_tokens = safe_avg(["generation", "baseline_tokens"])
        avg_o_tokens = safe_avg(["generation", "optimized_tokens"])

        print(f"  Videos analyzed:        {n}")
        if avg_bleu1 is not None:
            print(f"  Avg BLEU-1:             {avg_bleu1:.3f}")
        if avg_rouge_l is not None:
            print(f"  Avg ROUGE-L:            {avg_rouge_l:.3f}")
        if avg_jaccard is not None:
            print(f"  Avg Jaccard:            {avg_jaccard:.3f}")
        if avg_cosine is not None:
            print(f"  Avg Cosine:             {avg_cosine:.3f}")
        if avg_speedup is not None:
            print(f"  Avg Total Speedup:      {avg_speedup:.2f}x")
        if avg_enc_savings is not None:
            print(f"  Avg Encoder Savings:    {avg_enc_savings:.1f}%")
        if avg_enc_speedup is not None:
            print(f"  Avg Encode Speedup:     {avg_enc_speedup:.2f}x")
        if avg_b_entropy is not None:
            print(f"  Avg Baseline Entropy:   {avg_b_entropy:.3f}")
        if avg_o_entropy is not None:
            print(f"  Avg Optimized Entropy:  {avg_o_entropy:.3f}")
        if avg_b_tokens is not None:
            print(f"  Avg Baseline Tokens:    {avg_b_tokens:.0f}")
        if avg_o_tokens is not None:
            print(f"  Avg Optimized Tokens:   {avg_o_tokens:.0f}")

    # Write summary JSON
    summary = {
        "n_videos": n,
        "analyses": all_analyses,
        "aggregate": {
            "avg_bleu_1": safe_avg(["quality", "bleu_1"]),
            "avg_bleu_2": safe_avg(["quality", "bleu_2"]),
            "avg_bleu_4": safe_avg(["quality", "bleu_4"]),
            "avg_rouge_1": safe_avg(["quality", "rouge_1"]),
            "avg_rouge_2": safe_avg(["quality", "rouge_2"]),
            "avg_rouge_l": safe_avg(["quality", "rouge_l"]),
            "avg_jaccard": safe_avg(["quality", "jaccard"]),
            "avg_cosine": safe_avg(["quality", "cosine"]),
            "avg_total_speedup": safe_avg(["latency", "speedup"]),
            "avg_encode_speedup": safe_avg(["latency", "encode_speedup"]),
            "avg_encoder_savings_pct": safe_avg(["encoder", "optimized_encoder_savings_pct"]),
            "avg_baseline_entropy": safe_avg(["entropy", "baseline"]),
            "avg_optimized_entropy": safe_avg(["entropy", "optimized"]),
            "avg_baseline_tokens": safe_avg(["generation", "baseline_tokens"]),
            "avg_optimized_tokens": safe_avg(["generation", "optimized_tokens"]),
        },
    }

    output_path = os.path.join(args.results_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
