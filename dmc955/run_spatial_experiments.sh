#!/bin/bash
# Run spatial-crop adaptive experiments on all MPI-Sintel videos
# Also ensures existing _adaptive.json results exist for comparison
# Usage: ./run_spatial_experiments.sh

set -e

SPATIAL_BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-spatial-vlm"
ADAPTIVE_BINARY="/data/spring26/CPP/llama.cpp/build/bin/llama-adaptive-vlm"
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
echo "  MPI-Sintel Spatial Crop Experiment Suite"
echo "  Videos: $TOTAL"
echo "  Model: $(basename $MODEL)"
echo "  Base frames: $BASE_FRAMES"
echo "  Max tokens: $N_PREDICT"
echo "  Mode: Adaptive + Spatial Cropping"
echo "=============================================="

IDX=0
for VIDEO in $VIDEOS; do
    IDX=$((IDX + 1))
    echo ""
    echo "=============================================="
    echo "  [$IDX/$TOTAL] Video: $VIDEO"
    echo "=============================================="

    # Ensure ADAPTIVE (temporal-only) result exists for comparison
    ADAPTIVE_JSON="$RESULTS_DIR/${VIDEO}_adaptive.json"
    if [ -f "$ADAPTIVE_JSON" ]; then
        echo "  [OK]   Adaptive result exists: $ADAPTIVE_JSON"
    else
        echo "  [RUN]  Adaptive (temporal-only, for comparison)..."
        "$ADAPTIVE_BINARY" \
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

    # Run SPATIAL CROP
    SPATIAL_JSON="$RESULTS_DIR/${VIDEO}_spatial_crop.json"
    if [ -f "$SPATIAL_JSON" ]; then
        echo "  [SKIP] Spatial crop already exists: $SPATIAL_JSON"
    else
        echo "  [RUN]  Spatial crop (adaptive + cropping)..."
        "$SPATIAL_BINARY" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --video "$VIDEO" \
            --base-frames $BASE_FRAMES \
            -p "$PROMPT" \
            -n $N_PREDICT \
            -ngl $N_GPU \
            --output-json "$SPATIAL_JSON" \
            --log-disable \
            2>/dev/null || echo "  [WARN] Spatial crop failed for $VIDEO"
        echo "  [DONE] Spatial crop -> $SPATIAL_JSON"
    fi
done

echo ""
echo "=============================================="
echo "  All spatial crop experiments completed!"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
echo ""
echo "Run the comparison script:"
echo "  python3 /data/spring26/CPP/llama.cpp/dmc955/compare_spatial_results.py"
