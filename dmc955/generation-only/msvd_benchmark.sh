#!/bin/bash
# MSVD Benchmark: Baseline vs Emma on 20 test videos
# Collects per-video JSON results, then runs evaluation script

set -e

BIN="/data/spring26/CPP/llama.cpp/build/bin/llama-pyramid-gen"
source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
SINTEL_DIR="/data/spring26/CPP/datasets/MSVD/MSVD_prepared"
MANIFEST="/data/spring26/CPP/datasets/MSVD/MSVD_prepared/manifest.json"
OUTDIR="/data/spring26/CPP/llama.cpp/dmc955/generation-only/msvd_results"
N_PREDICT=128
N_FRAMES=8

mkdir -p "$OUTDIR/baseline" "$OUTDIR/emma"

# Read video IDs from manifest
VIDEOS=$(python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
for v in m:
    print(v['video_id'])
")

TOTAL=$(echo "$VIDEOS" | wc -l)
echo "=== MSVD Benchmark: $TOTAL videos ==="
echo "N_FRAMES=$N_FRAMES, N_PREDICT=$N_PREDICT"
echo ""

COUNT=0
for VID in $VIDEOS; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] $VID"

    # --- Baseline ---
    OUTJSON="$OUTDIR/baseline/${VID}.json"
    if [ ! -f "$OUTJSON" ]; then
        echo "  Baseline..."
        "$BIN" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL_DIR" \
            --video "$VID" \
            --n-frames "$N_FRAMES" \
            --n-predict "$N_PREDICT" \
            --baseline \
            --output-json "$OUTJSON" \
            --prompt "Describe what is happening in this video in one sentence." \
            -ngl 99 \
            > "$OUTDIR/baseline/${VID}.log" 2>&1 || echo "  BASELINE FAILED"
    else
        echo "  Baseline: cached"
    fi

    # --- Emma (with adaptive frame selection) ---
    OUTJSON="$OUTDIR/emma/${VID}.json"
    if [ ! -f "$OUTJSON" ]; then
        echo "  Emma..."
        "$BIN" \
            -m "$MODEL" \
            --mmproj "$MMPROJ" \
            --sintel-dir "$SINTEL_DIR" \
            --video "$VID" \
            --adaptive \
            --n-predict "$N_PREDICT" \
            --output-json "$OUTJSON" \
            --prompt "Describe what is happening in this video in one sentence." \
            -ngl 99 \
            > "$OUTDIR/emma/${VID}.log" 2>&1 || echo "  EMMA FAILED"
    else
        echo "  Emma: cached"
    fi

    echo ""
done

echo "=== All runs complete ==="
echo "Running evaluation..."

python3 /data/spring26/CPP/llama.cpp/dmc955/generation-only/msvd_evaluate.py
