#!/bin/bash
# Run embedding cache experiments with finer patch grid (8x8)
set -e

BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-spatial-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
RESULTS_DIR="/data/spring26/CPP/llama.cpp/dmc955/results_emb_cache_v2"
PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
N_PREDICT=150
N_GPU=99
BASE_FRAMES=8
PATCH_GRID=8

export LD_LIBRARY_PATH="/data/spring26/CPP/llama.cpp/build/bin:$LD_LIBRARY_PATH"

mkdir -p "$RESULTS_DIR"

VIDEOS="alley_1 sleeping_1 bamboo_1 cave_2 ambush_5 market_5"

echo "=============================================="
echo "  Embedding Cache Experiment v2 (grid=${PATCH_GRID})"
echo "  Videos: $VIDEOS"
echo "=============================================="

for VIDEO in $VIDEOS; do
    echo ""
    echo "=== $VIDEO ==="

    # BASELINE (uniform, no cache)
    BASELINE_JSON="$RESULTS_DIR/${VIDEO}_baseline.json"
    echo "  [RUN] Baseline..."
    "$BINARY" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --base-frames $BASE_FRAMES \
        --patch-grid $PATCH_GRID \
        -p "$PROMPT" \
        -n $N_PREDICT \
        -ngl $N_GPU \
        --baseline \
        --output-json "$BASELINE_JSON" \
        --log-disable \
        2>/dev/null || echo "  [WARN] Baseline failed for $VIDEO"
    echo "  [DONE] Baseline"

    # EMBEDDING CACHE
    CACHE_JSON="$RESULTS_DIR/${VIDEO}_embedding_cache.json"
    echo "  [RUN] Embedding cache (grid=$PATCH_GRID)..."
    "$BINARY" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --base-frames $BASE_FRAMES \
        --patch-grid $PATCH_GRID \
        -p "$PROMPT" \
        -n $N_PREDICT \
        -ngl $N_GPU \
        --output-json "$CACHE_JSON" \
        --log-disable \
        2>/dev/null || echo "  [WARN] Embedding cache failed for $VIDEO"
    echo "  [DONE] Embedding cache"
done

echo ""
echo "All done. Results in $RESULTS_DIR"
