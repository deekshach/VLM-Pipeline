#!/bin/bash
# Sintel comparison v2: controlled generation length, diversity metrics
# Baseline: uniform 8 frames, no pruning, 128 forced tokens
# Pyramid: adaptive (max 10 frames) + spatial cache + MOTION+KF + eviction + defrag, 128 forced tokens

source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
SINTEL="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
BIN="/data/spring26/CPP/llama.cpp/build/bin/llama-pyramid-gen"
OUT="/data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_v2"
N_PREDICT=128

mkdir -p "$OUT/baseline" "$OUT/pyramid"

VIDEOS=$(ls "$SINTEL/training/final/")

for vid in $VIDEOS; do
    echo -n "[$vid] "

    # Baseline: uniform 8 frames, no pyramid, forced 128 tokens
    if [ ! -f "$OUT/baseline/${vid}.json" ]; then
        $BIN -m "$MODEL" --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL" --video "$vid" \
            --n-frames 8 --n-predict $N_PREDICT --baseline --ignore-eos \
            --output-json "$OUT/baseline/${vid}.json" \
            -ngl 99 2>/dev/null
        echo -n "B "
    else
        echo -n "b "
    fi

    # Pyramid: adaptive (max 10) + all features, forced 128 tokens
    if [ ! -f "$OUT/pyramid/${vid}.json" ]; then
        $BIN -m "$MODEL" --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL" --video "$vid" \
            --n-predict $N_PREDICT --adaptive --spatial-cache \
            --max-selected-frames 10 --ignore-eos \
            --output-json "$OUT/pyramid/${vid}.json" \
            -ngl 99 2>/dev/null
        echo -n "P "
    else
        echo -n "p "
    fi

    echo "done"
done

echo ""
echo "=== Results ==="
python3 /data/spring26/CPP/llama.cpp/dmc955/generation-only/sintel_v2_table.py
