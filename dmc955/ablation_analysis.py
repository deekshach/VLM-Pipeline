#!/usr/bin/env python3
"""
Pre-LLM Pruning Ablation Study Analysis

Compares quality metrics across pruning strategies to evaluate
whether scored pruning outperforms random and quantify individual
signal contributions.

Strategies compared:
  1. combined     - tri-signal (motion 0.3 + norm 0.3 + attn 0.4)
  2. random       - random token selection
  3. motion-only  - only motion signal
  4. norm-only    - only embedding L2 norm signal
  5. attn-only    - only ViT attention received signal
  6. motion-norm  - motion + norm (no ViT attn overhead)

All are compared against baseline (no pre-LLM pruning) results.

Usage:
    python3 ablation_analysis.py ./streaming_results
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def tokenize_simple(text):
    return re.findall(r'\w+', text.lower())


def compute_bleu(reference, hypothesis):
    if not HAS_NLTK:
        return {"bleu_1": None, "bleu_2": None, "bleu_4": None}
    ref_tokens = tokenize_simple(reference)
    hyp_tokens = tokenize_simple(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
    smooth = SmoothingFunction().method1
    return {
        "bleu_1": sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu_2": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "bleu_4": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }


def compute_rouge(reference, hypothesis):
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
    tokens_a = set(tokenize_simple(text_a))
    tokens_b = set(tokenize_simple(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_cosine(text_a, text_b):
    tokens_a = tokenize_simple(text_a)
    tokens_b = tokenize_simple(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    ca = Counter(tokens_a)
    cb = Counter(tokens_b)
    words = set(ca.keys()) | set(cb.keys())
    dot = sum(ca.get(w, 0) * cb.get(w, 0) for w in words)
    na = sum(v ** 2 for v in ca.values()) ** 0.5
    nb = sum(v ** 2 for v in cb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def format_float(v, d=3):
    return f"{v:.{d}f}" if v is not None else "N/A"


def print_table(headers, rows, title=None):
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def main():
    parser = argparse.ArgumentParser(description="Pre-LLM Pruning Ablation Analysis")
    parser.add_argument("results_dir", help="Directory with streaming result JSON files")
    parser.add_argument("--ablation-dir", default=None,
                        help="Subdirectory for ablation results (default: <results_dir>/ablation)")
    parser.add_argument("--output", "-o", default="ablation_summary.json",
                        help="Output JSON path (relative to results_dir)")
    args = parser.parse_args()

    results_dir = args.results_dir
    ablation_dir = args.ablation_dir or os.path.join(results_dir, "ablation")

    if not os.path.isdir(ablation_dir):
        print(f"Error: ablation directory not found: {ablation_dir}", file=sys.stderr)
        sys.exit(1)

    # Load baseline results (reference)
    baselines = {}
    for bf in glob.glob(os.path.join(results_dir, "streaming_baseline_*.json")):
        video = os.path.basename(bf).replace("streaming_baseline_", "").replace(".json", "")
        with open(bf) as f:
            baselines[video] = json.load(f)

    # Also load optimized (no pruning) as reference
    optimized_refs = {}
    for of in glob.glob(os.path.join(results_dir, "streaming_optimized_*.json")):
        bn = os.path.basename(of)
        if "pruning" in bn or "prellm" in bn:
            continue
        video = bn.replace("streaming_optimized_", "").replace(".json", "")
        with open(of) as f:
            optimized_refs[video] = json.load(f)

    # Load ablation results
    strategies = defaultdict(dict)  # strategy -> video -> data
    for af in sorted(glob.glob(os.path.join(ablation_dir, "ablation_*_*.json"))):
        bn = os.path.basename(af).replace("ablation_", "").replace(".json", "")
        # Parse strategy_video format - strategy may contain hyphens
        # Known strategies: combined, random, motion-only, norm-only, attn-only, motion-norm
        known_strategies = ["combined", "random", "motion-only", "norm-only", "attn-only", "motion-norm"]
        matched = False
        for strat in sorted(known_strategies, key=len, reverse=True):
            if bn.startswith(strat + "_"):
                video = bn[len(strat) + 1:]
                with open(af) as f:
                    strategies[strat][video] = json.load(f)
                matched = True
                break
        if not matched:
            print(f"  Warning: Could not parse strategy from {bn}", file=sys.stderr)

    if not strategies:
        print("No ablation results found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(baselines)} baseline videos")
    print(f"Loaded {len(optimized_refs)} optimized (no pruning) videos")
    print(f"Loaded {len(strategies)} strategies: {list(strategies.keys())}")

    # Get common videos across all strategies
    all_videos = set()
    for strat_data in strategies.values():
        all_videos.update(strat_data.keys())
    common_videos = sorted(all_videos)

    # =====================================================================
    # Table 1: Quality Metrics per Strategy (averaged across videos)
    # Compare each strategy against BASELINE text
    # =====================================================================
    strategy_order = ["combined", "motion-norm", "motion-only", "norm-only", "attn-only", "random"]
    strategy_avgs = {}

    for strat in strategy_order:
        if strat not in strategies:
            continue
        metrics = defaultdict(list)
        for video in common_videos:
            if video not in strategies[strat] or video not in baselines:
                continue
            baseline_text = baselines[video].get("inference", {}).get("generated_text", "")
            strat_text = strategies[strat][video].get("inference", {}).get("generated_text", "")
            if not baseline_text.strip() or not strat_text.strip():
                continue

            bleu = compute_bleu(baseline_text, strat_text)
            rouge = compute_rouge(baseline_text, strat_text)
            jacc = compute_jaccard(baseline_text, strat_text)
            cos = compute_cosine(baseline_text, strat_text)

            for k, v in bleu.items():
                if v is not None:
                    metrics[k].append(v)
            for k, v in rouge.items():
                if v is not None:
                    metrics[k].append(v)
            metrics["jaccard"].append(jacc)
            metrics["cosine"].append(cos)

        avgs = {k: (sum(v) / len(v) if v else None) for k, v in metrics.items()}
        avgs["n_videos"] = len(metrics.get("jaccard", []))
        strategy_avgs[strat] = avgs

    # Also add optimized (no pruning) as reference
    if optimized_refs:
        metrics = defaultdict(list)
        for video in common_videos:
            if video not in optimized_refs or video not in baselines:
                continue
            baseline_text = baselines[video].get("inference", {}).get("generated_text", "")
            opt_text = optimized_refs[video].get("inference", {}).get("generated_text", "")
            if not baseline_text.strip() or not opt_text.strip():
                continue
            bleu = compute_bleu(baseline_text, opt_text)
            rouge = compute_rouge(baseline_text, opt_text)
            jacc = compute_jaccard(baseline_text, opt_text)
            cos = compute_cosine(baseline_text, opt_text)
            for k, v in bleu.items():
                if v is not None:
                    metrics[k].append(v)
            for k, v in rouge.items():
                if v is not None:
                    metrics[k].append(v)
            metrics["jaccard"].append(jacc)
            metrics["cosine"].append(cos)
        avgs = {k: (sum(v) / len(v) if v else None) for k, v in metrics.items()}
        avgs["n_videos"] = len(metrics.get("jaccard", []))
        strategy_avgs["optimized(nopruning)"] = avgs

    quality_headers = ["Strategy", "N", "BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Jaccard", "Cosine"]
    quality_rows = []
    display_order = ["optimized(nopruning)"] + strategy_order
    for strat in display_order:
        if strat not in strategy_avgs:
            continue
        a = strategy_avgs[strat]
        quality_rows.append([
            strat, a["n_videos"],
            format_float(a.get("bleu_1")), format_float(a.get("bleu_2")), format_float(a.get("bleu_4")),
            format_float(a.get("rouge_1")), format_float(a.get("rouge_2")), format_float(a.get("rouge_l")),
            format_float(a.get("jaccard")), format_float(a.get("cosine")),
        ])
    print_table(quality_headers, quality_rows,
                "Table 1: Average Quality vs Baseline (higher = more similar to baseline)")

    # =====================================================================
    # Table 2: Quality Delta vs Random (how much better is scored vs random)
    # =====================================================================
    if "random" in strategy_avgs:
        random_avgs = strategy_avgs["random"]
        delta_headers = ["Strategy", "dBLEU-1", "dBLEU-4", "dROUGE-L", "dJaccard", "dCosine"]
        delta_rows = []
        for strat in strategy_order:
            if strat not in strategy_avgs or strat == "random":
                continue
            a = strategy_avgs[strat]
            def delta(key):
                v = a.get(key)
                r = random_avgs.get(key)
                if v is not None and r is not None:
                    d = v - r
                    return f"{d:+.4f}"
                return "N/A"
            delta_rows.append([strat, delta("bleu_1"), delta("bleu_4"), delta("rouge_l"), delta("jaccard"), delta("cosine")])
        print_table(delta_headers, delta_rows,
                    "Table 2: Quality Delta vs Random (positive = better than random)")

    # =====================================================================
    # Table 3: Per-Video Quality (Jaccard + Cosine) for each strategy
    # =====================================================================
    pv_headers = ["Video"] + [s for s in strategy_order if s in strategies] + ["opt(noprune)"]
    pv_rows_jaccard = []
    pv_rows_cosine = []
    for video in common_videos:
        if video not in baselines:
            continue
        baseline_text = baselines[video].get("inference", {}).get("generated_text", "")
        if not baseline_text.strip():
            continue
        j_row = [video]
        c_row = [video]
        for strat in strategy_order:
            if strat not in strategies:
                continue
            if video in strategies[strat]:
                strat_text = strategies[strat][video].get("inference", {}).get("generated_text", "")
                if strat_text.strip():
                    j_row.append(format_float(compute_jaccard(baseline_text, strat_text)))
                    c_row.append(format_float(compute_cosine(baseline_text, strat_text)))
                else:
                    j_row.append("N/A")
                    c_row.append("N/A")
            else:
                j_row.append("-")
                c_row.append("-")
        # Add optimized no-pruning reference
        if video in optimized_refs:
            opt_text = optimized_refs[video].get("inference", {}).get("generated_text", "")
            if opt_text.strip():
                j_row.append(format_float(compute_jaccard(baseline_text, opt_text)))
                c_row.append(format_float(compute_cosine(baseline_text, opt_text)))
            else:
                j_row.append("N/A")
                c_row.append("N/A")
        else:
            j_row.append("-")
            c_row.append("-")
        pv_rows_jaccard.append(j_row)
        pv_rows_cosine.append(c_row)

    print_table(pv_headers, pv_rows_jaccard,
                "Table 3a: Per-Video Jaccard Similarity vs Baseline")
    print_table(pv_headers, pv_rows_cosine,
                "Table 3b: Per-Video Cosine Similarity vs Baseline")

    # =====================================================================
    # Table 4: Latency Comparison (encode time)
    # =====================================================================
    latency_headers = ["Strategy", "Avg Encode(ms)", "Avg Gen(ms)", "Avg Total(ms)", "vs Baseline"]
    latency_rows = []

    # Baseline average
    b_encode_avg = []
    b_gen_avg = []
    b_total_avg = []
    for video in common_videos:
        if video in baselines:
            bt = baselines[video].get("timing", {})
            b_encode_avg.append(bt.get("encode_ms", 0))
            b_gen_avg.append(bt.get("generation_ms", 0))
            b_total_avg.append(bt.get("total_ms", 0))
    if b_encode_avg:
        be = sum(b_encode_avg) / len(b_encode_avg)
        bg = sum(b_gen_avg) / len(b_gen_avg)
        bt = sum(b_total_avg) / len(b_total_avg)
        latency_rows.append(["baseline", f"{be:.0f}", f"{bg:.0f}", f"{bt:.0f}", "1.00x"])
        baseline_total = bt
    else:
        baseline_total = 1

    for strat in display_order:
        if strat == "optimized(nopruning)":
            source = optimized_refs
        elif strat in strategies:
            source = strategies[strat]
        else:
            continue
        enc_vals, gen_vals, tot_vals = [], [], []
        for video in common_videos:
            if video in source:
                t = source[video].get("timing", {})
                enc_vals.append(t.get("encode_ms", 0))
                gen_vals.append(t.get("generation_ms", 0))
                tot_vals.append(t.get("total_ms", 0))
        if enc_vals:
            ae = sum(enc_vals) / len(enc_vals)
            ag = sum(gen_vals) / len(gen_vals)
            at = sum(tot_vals) / len(tot_vals)
            speedup = baseline_total / at if at > 0 else 0
            latency_rows.append([strat, f"{ae:.0f}", f"{ag:.0f}", f"{at:.0f}", f"{speedup:.2f}x"])

    print_table(latency_headers, latency_rows, "Table 4: Average Latency per Strategy")

    # =====================================================================
    # Table 5: Pre-LLM Pruning Token Stats
    # =====================================================================
    token_headers = ["Strategy", "Avg Full Tokens", "Avg Injected", "Avg Pruned", "Savings%"]
    token_rows = []
    for strat in strategy_order:
        if strat not in strategies:
            continue
        before_vals, inj_vals, pruned_vals = [], [], []
        for video in common_videos:
            if video in strategies[strat]:
                p = strategies[strat][video].get("pre_llm_pruning", {})
                if p.get("enabled", False):
                    before_vals.append(p.get("total_tokens_before", 0))
                    inj_vals.append(p.get("total_tokens_injected", 0))
                    pruned_vals.append(p.get("total_tokens_pruned", 0))
        if before_vals:
            ab = sum(before_vals) / len(before_vals)
            ai = sum(inj_vals) / len(inj_vals)
            ap = sum(pruned_vals) / len(pruned_vals)
            sav = (ap / ab * 100) if ab > 0 else 0
            token_rows.append([strat, f"{ab:.0f}", f"{ai:.0f}", f"{ap:.0f}", f"{sav:.1f}%"])
    if token_rows:
        print_table(token_headers, token_rows, "Table 5: Pre-LLM Token Pruning Statistics")

    # =====================================================================
    # Summary: Statistical Significance Indication
    # =====================================================================
    print(f"\n{'=' * 80}")
    print("  Summary & Key Findings")
    print(f"{'=' * 80}")

    if "combined" in strategy_avgs and "random" in strategy_avgs:
        c = strategy_avgs["combined"]
        r = strategy_avgs["random"]
        print(f"\n  Combined vs Random:")
        for metric in ["bleu_1", "bleu_4", "rouge_l", "jaccard", "cosine"]:
            cv = c.get(metric)
            rv = r.get(metric)
            if cv is not None and rv is not None:
                diff = cv - rv
                pct = (diff / rv * 100) if rv != 0 else 0
                better = "BETTER" if diff > 0 else ("WORSE" if diff < 0 else "EQUAL")
                print(f"    {metric:10s}: combined={cv:.4f}, random={rv:.4f}, delta={diff:+.4f} ({pct:+.1f}%) [{better}]")

    # Per-signal contribution analysis
    print(f"\n  Individual Signal Analysis (vs random):")
    for strat in ["motion-only", "norm-only", "attn-only", "motion-norm"]:
        if strat in strategy_avgs and "random" in strategy_avgs:
            s = strategy_avgs[strat]
            r = strategy_avgs["random"]
            j_delta = (s.get("jaccard", 0) or 0) - (r.get("jaccard", 0) or 0)
            c_delta = (s.get("cosine", 0) or 0) - (r.get("cosine", 0) or 0)
            print(f"    {strat:15s}: Jaccard delta={j_delta:+.4f}, Cosine delta={c_delta:+.4f}")

    # Recommendation
    if strategy_avgs:
        best_strat = max(
            [(s, a.get("cosine", 0) or 0) for s, a in strategy_avgs.items() if s != "optimized(nopruning)"],
            key=lambda x: x[1]
        )
        print(f"\n  Best strategy by Cosine: {best_strat[0]} ({best_strat[1]:.4f})")

    # Save summary
    summary = {
        "strategy_averages": {s: {k: v for k, v in a.items()} for s, a in strategy_avgs.items()},
        "n_videos": len(common_videos),
        "videos": common_videos,
    }
    output_path = os.path.join(results_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {output_path}")


if __name__ == "__main__":
    main()
