#!/bin/bash
# Post-prefill KV eviction benchmark
# Runs 3 variants per video: baseline, evict 50%, evict 30%

set -e

BINARY="./build/bin/llama-streaming-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
OUTDIR="./streaming_results/postprefill_eviction"

VIDEOS=(
    "alley_1" "alley_2" "ambush_2" "bamboo_1" "bamboo_2"
    "bandage_1" "cave_2" "market_5" "mountain_1" "temple_2"
)

COMMON_ARGS="-n 500 -ngl 99 -c 16384"

mkdir -p "$OUTDIR"

echo "=== Post-Prefill KV Eviction Benchmark ==="
echo "Variants: baseline, evict-50%, evict-30%"
echo "Videos: ${#VIDEOS[@]}"
echo ""

for video in "${VIDEOS[@]}"; do
    echo "============================================"
    echo "Video: $video"
    echo "============================================"

    # Variant 1: Baseline (no eviction)
    echo "--- Baseline ---"
    $BINARY \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$video" \
        --cumulative-threshold 15.0 \
        --max-selected-frames 10 \
        --output-json "$OUTDIR/baseline_${video}.json" \
        $COMMON_ARGS \
        2>&1 | grep -E "Generated|Total time|Summary"
    echo ""

    # Variant 2: Post-prefill eviction, keep 50%
    echo "--- Eviction 50% ---"
    $BINARY \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$video" \
        --cumulative-threshold 15.0 \
        --max-selected-frames 10 \
        --enable-postprefill-eviction \
        --postprefill-keep-ratio 0.5 \
        --output-json "$OUTDIR/evict50_${video}.json" \
        $COMMON_ARGS \
        2>&1 | grep -E "Generated|Post-prefill|Total time|Summary"
    echo ""

    # Variant 3: Post-prefill eviction, keep 30%
    echo "--- Eviction 30% ---"
    $BINARY \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$video" \
        --cumulative-threshold 15.0 \
        --max-selected-frames 10 \
        --enable-postprefill-eviction \
        --postprefill-keep-ratio 0.3 \
        --output-json "$OUTDIR/evict30_${video}.json" \
        $COMMON_ARGS \
        2>&1 | grep -E "Generated|Post-prefill|Total time|Summary"
    echo ""
done

echo "=== All post-prefill eviction benchmarks complete ==="
echo "Results in: $OUTDIR/"
echo ""
echo "Run: python3 dmc955/correlation_analysis.py $OUTDIR"
