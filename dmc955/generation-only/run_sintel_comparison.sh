#!/bin/bash
# Full Sintel comparison: Baseline vs Pyramid (all features enabled)
# Baseline: uniform 8 frames, no pruning
# Pyramid: adaptive sampling + spatial cache + pre-LLM MOTION+KF + post-prefill eviction + defrag

source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
SINTEL="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
BIN="/data/spring26/CPP/llama.cpp/build/bin/llama-pyramid-gen"
OUT="/data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_comparison"

mkdir -p "$OUT/baseline" "$OUT/pyramid"

VIDEOS=$(ls "$SINTEL/training/final/")

for vid in $VIDEOS; do
    echo ">>> $vid"

    # Baseline: uniform 8 frames, no pyramid
    if [ ! -f "$OUT/baseline/${vid}.json" ]; then
        $BIN -m "$MODEL" --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL" --video "$vid" \
            --n-frames 8 --n-predict 256 --baseline \
            --output-json "$OUT/baseline/${vid}.json" \
            -ngl 99 2>/dev/null
    fi

    # Pyramid: adaptive + spatial cache + pre-LLM + post-prefill eviction + defrag
    if [ ! -f "$OUT/pyramid/${vid}.json" ]; then
        $BIN -m "$MODEL" --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL" --video "$vid" \
            --n-predict 256 --adaptive --spatial-cache \
            --output-json "$OUT/pyramid/${vid}.json" \
            -ngl 99 2>/dev/null
    fi

    echo "  done"
done

echo ""
echo "=== Generating comparison table ==="
python3 /data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_comparison_table.py
