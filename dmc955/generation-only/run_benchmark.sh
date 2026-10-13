#!/bin/bash
# Pyramid KV Prefill Benchmark — RTX 4090
# Compares baseline vs pyramid across multiple Sintel videos

set -e

source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
BIN="$(dirname "$0")/../../build/bin/llama-pyramid-gen"
OUTDIR="$(dirname "$0")/results"
N_FRAMES=8
N_PREDICT=128

mkdir -p "$OUTDIR"

VIDEOS="alley_1 bamboo_1 cave_2 market_5 temple_2"

echo "============================================"
echo "Pyramid KV Prefill Benchmark"
echo "Model: $MODEL"
echo "Frames: $N_FRAMES, Max predict: $N_PREDICT"
echo "============================================"
echo ""

for VIDEO in $VIDEOS; do
    echo "--- $VIDEO ---"

    echo "  Running baseline..."
    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" \
        --video "$VIDEO" --n-frames $N_FRAMES -ngl 99 -c 8192 \
        --n-predict $N_PREDICT --baseline \
        --output-json "$OUTDIR/baseline_${VIDEO}.json" 2>&1 | \
        grep -E "(Prefill:|Generation:|Total:)" | sed 's/^/    /'

    echo "  Running pyramid (pvc_min=0.20)..."
    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" \
        --video "$VIDEO" --n-frames $N_FRAMES -ngl 99 -c 8192 \
        --n-predict $N_PREDICT --pyramid-pvc-min 0.20 \
        --output-json "$OUTDIR/pyramid_020_${VIDEO}.json" 2>&1 | \
        grep -E "(Prefill:|Generation:|Total:|Evict|tokens)" | sed 's/^/    /'

    echo "  Running pyramid (pvc_min=0.40)..."
    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" \
        --video "$VIDEO" --n-frames $N_FRAMES -ngl 99 -c 8192 \
        --n-predict $N_PREDICT --pyramid-pvc-min 0.40 \
        --output-json "$OUTDIR/pyramid_040_${VIDEO}.json" 2>&1 | \
        grep -E "(Prefill:|Generation:|Total:|Evict|tokens)" | sed 's/^/    /'

    echo ""
done

echo "Results saved to $OUTDIR/"
echo "Done."
