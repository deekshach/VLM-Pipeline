#!/bin/bash
# Run embedding cache experiments on representative MPI-Sintel videos
# Compares baseline (uniform, full encode) vs embedding cache (adaptive + cache)
set -e

BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-spatial-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
RESULTS_DIR="/data/spring26/CPP/llama.cpp/dmc955/results_emb_cache"
PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
N_PREDICT=150
N_GPU=99
BASE_FRAMES=8

export LD_LIBRARY_PATH="/data/spring26/CPP/llama.cpp/build/bin:$LD_LIBRARY_PATH"

mkdir -p "$RESULTS_DIR"

# Representative subset: low, medium, high motion
VIDEOS="alley_1 sleeping_1 bamboo_1 cave_2 ambush_5 market_5"

echo "=============================================="
echo "  Embedding Cache Experiment"
echo "  Videos: $VIDEOS"
echo "  Model: $(basename $MODEL)"
echo "  Base frames: $BASE_FRAMES"
echo "=============================================="

for VIDEO in $VIDEOS; do
    echo ""
    echo "=============================================="
    echo "  Video: $VIDEO"
    echo "=============================================="

    # BASELINE
    BASELINE_JSON="$RESULTS_DIR/${VIDEO}_baseline.json"
    echo "  [RUN] Baseline..."
    "$BINARY" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --base-frames $BASE_FRAMES \
        -p "$PROMPT" \
        -n $N_PREDICT \
        -ngl $N_GPU \
        --baseline \
        --output-json "$BASELINE_JSON" \
        --log-disable \
        2>/dev/null || echo "  [WARN] Baseline failed for $VIDEO"
    echo "  [DONE] Baseline -> $BASELINE_JSON"

    # EMBEDDING CACHE (adaptive mode)
    CACHE_JSON="$RESULTS_DIR/${VIDEO}_embedding_cache.json"
    echo "  [RUN] Embedding cache..."
    "$BINARY" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --base-frames $BASE_FRAMES \
        -p "$PROMPT" \
        -n $N_PREDICT \
        -ngl $N_GPU \
        --output-json "$CACHE_JSON" \
        --log-disable \
        2>/dev/null || echo "  [WARN] Embedding cache failed for $VIDEO"
    echo "  [DONE] Embedding cache -> $CACHE_JSON"
done

echo ""
echo "=============================================="
echo "  All experiments completed!"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
