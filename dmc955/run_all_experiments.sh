#!/bin/bash
# Run adaptive and baseline experiments on all MPI-Sintel videos
# Usage: ./run_all_experiments.sh

set -e

BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
RESULTS_DIR="/data/spring26/CPP/llama.cpp/dmc955/results"
PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
N_PREDICT=150
N_GPU=99
BASE_FRAMES=8

export LD_LIBRARY_PATH="/data/spring26/CPP/llama.cpp/build/bin:$LD_LIBRARY_PATH"

mkdir -p "$RESULTS_DIR"

# Get all video names
VIDEOS=$(ls "$SINTEL/training/clean/" | sort)
TOTAL=$(echo "$VIDEOS" | wc -w)

echo "=============================================="
echo "  MPI-Sintel VLM Experiment Suite (v2)"
echo "  Videos: $TOTAL"
echo "  Model: $(basename $MODEL)"
echo "  Base frames: $BASE_FRAMES"
echo "  Max tokens: $N_PREDICT"
echo "  Algorithm: Python-matched (absolute norm, step function)"
echo "=============================================="

IDX=0
for VIDEO in $VIDEOS; do
    IDX=$((IDX + 1))
    echo ""
    echo "=============================================="
    echo "  [$IDX/$TOTAL] Video: $VIDEO"
    echo "=============================================="

    # Run BASELINE (uniform sampling)
    BASELINE_JSON="$RESULTS_DIR/${VIDEO}_baseline.json"
    if [ -f "$BASELINE_JSON" ]; then
        echo "  [SKIP] Baseline already exists: $BASELINE_JSON"
    else
        echo "  [RUN]  Baseline (uniform $BASE_FRAMES frames)..."
        "$BINARY" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --video "$VIDEO" \
            --baseline \
            --base-frames $BASE_FRAMES \
            -p "$PROMPT" \
            -n $N_PREDICT \
            -ngl $N_GPU \
            --output-json "$BASELINE_JSON" \
            --log-disable \
            2>/dev/null || echo "  [WARN] Baseline failed for $VIDEO"
        echo "  [DONE] Baseline -> $BASELINE_JSON"
    fi

    # Run ADAPTIVE (motion-guided) — uses Python-matched defaults
    # (base=8, min=6, max=24, selection_percentile=70, spatial_threshold=0.05)
    ADAPTIVE_JSON="$RESULTS_DIR/${VIDEO}_adaptive.json"
    if [ -f "$ADAPTIVE_JSON" ]; then
        echo "  [SKIP] Adaptive already exists: $ADAPTIVE_JSON"
    else
        echo "  [RUN]  Adaptive (motion-guided)..."
        "$BINARY" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --video "$VIDEO" \
            --base-frames $BASE_FRAMES \
            -p "$PROMPT" \
            -n $N_PREDICT \
            -ngl $N_GPU \
            --output-json "$ADAPTIVE_JSON" \
            --log-disable \
            2>/dev/null || echo "  [WARN] Adaptive failed for $VIDEO"
        echo "  [DONE] Adaptive -> $ADAPTIVE_JSON"
    fi
done

echo ""
echo "=============================================="
echo "  All experiments completed!"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
echo ""
echo "Run the comparison script:"
echo "  python3 /data/spring26/CPP/llama.cpp/dmc955/compare_results.py"
