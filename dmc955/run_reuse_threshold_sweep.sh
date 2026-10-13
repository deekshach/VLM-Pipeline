#!/bin/bash
# Sweep --patch-reuse-motion thresholds to find optimal cache aggressiveness
set -e

BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-spatial-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
RESULTS_DIR="/data/spring26/CPP/llama.cpp/dmc955/results_reuse_sweep"
PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
N_PREDICT=150
N_GPU=99
BASE_FRAMES=8
PATCH_GRID=8

export LD_LIBRARY_PATH="/data/spring26/CPP/llama.cpp/build/bin:$LD_LIBRARY_PATH"
mkdir -p "$RESULTS_DIR"

VIDEOS="alley_1 sleeping_1 bamboo_1 cave_2 ambush_5 market_5"
# Thresholds: 0 (original binary), 3, 5, 8, 12
THRESHOLDS="0 3 5 8 12"

echo "=============================================="
echo "  Patch Reuse Threshold Sweep"
echo "  Grid: ${PATCH_GRID}x${PATCH_GRID}"
echo "  Thresholds: $THRESHOLDS"
echo "=============================================="

for THRESH in $THRESHOLDS; do
    echo ""
    echo "===== THRESHOLD: $THRESH ====="
    for VIDEO in $VIDEOS; do
        OUT="$RESULTS_DIR/${VIDEO}_thresh${THRESH}.json"
        echo "  [RUN] $VIDEO (thresh=$THRESH)..."
        "$BINARY" \
            -m "$MODEL" --mmproj "$MMPROJ" --video "$VIDEO" \
            --base-frames $BASE_FRAMES --patch-grid $PATCH_GRID \
            --patch-reuse-motion $THRESH \
            -p "$PROMPT" -n $N_PREDICT -ngl $N_GPU \
            --output-json "$OUT" \
            --log-disable 2>/dev/null || echo "  [WARN] failed"
        echo "  [DONE] $VIDEO"
    done
done

echo ""
echo "All done. Results in $RESULTS_DIR"
