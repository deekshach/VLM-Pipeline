#!/bin/bash
# Run attention correlation analysis across all 10 Sintel videos
# This validates whether pre-LLM scoring predicts LLM attention

set -e

BINARY="./build/bin/llama-streaming-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
OUTDIR="./streaming_results/correlation"

VIDEOS=(
    "alley_1" "alley_2" "ambush_2" "bamboo_1" "bamboo_2"
    "bandage_1" "cave_2" "market_5" "mountain_1" "temple_2"
)

mkdir -p "$OUTDIR"

echo "=== Attention Correlation Analysis ==="
echo "Analyzing correlation between pre-LLM scores and LLM attention"
echo "Videos: ${#VIDEOS[@]}"
echo ""

for video in "${VIDEOS[@]}"; do
    echo "--- Running: $video ---"
    $BINARY \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$video" \
        --cumulative-threshold 15.0 \
        --max-selected-frames 10 \
        --analyze-attention-correlation \
        --output-json "$OUTDIR/correlation_${video}.json" \
        -n 500 -ngl 99 -c 16384 \
        2>&1 | grep -E "Pearson|Spearman|ratio|Generated"
    echo ""
done

echo "=== All correlation analyses complete ==="
echo "Results in: $OUTDIR/"
echo ""
echo "Run: python3 dmc955/correlation_analysis.py $OUTDIR"
