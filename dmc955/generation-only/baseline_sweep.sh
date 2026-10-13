#!/bin/bash
# Baseline latency sweep: 1, 2, 4, 8, 16 frames
# Measures: ViT encode, prefill, generation latency

set -e

BIN="/data/spring26/CPP/llama.cpp/build/bin/llama-pyramid-gen"
source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
VIDEO="alley_1"
OUTDIR="/data/spring26/CPP/llama.cpp/dmc955/generation-only/baseline_sweep_results"
N_PREDICT=256

mkdir -p "$OUTDIR"

FRAME_COUNTS=(1 2 4 8 16)

echo "=== Baseline Latency Sweep ==="
echo "Video: $VIDEO"
echo "Frame counts: ${FRAME_COUNTS[*]}"
echo "Output dir: $OUTDIR"
echo ""

for N in "${FRAME_COUNTS[@]}"; do
    echo "---------- Running baseline with $N frames ----------"
    OUTJSON="$OUTDIR/baseline_${N}frames.json"
    LOGFILE="$OUTDIR/baseline_${N}frames.log"

    "$BIN" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --n-frames "$N" \
        --n-predict "$N_PREDICT" \
        --baseline \
        --output-json "$OUTJSON" \
        -ngl 99 \
        2>&1 | tee "$LOGFILE"

    echo ""
    echo "Saved: $OUTJSON"
    echo ""
done

echo ""
echo "=== Sweep Complete ==="
echo ""

# Summary table from JSON files
echo "Frames | Encode (ms) | Prefill (ms) | Gen (ms) | Tokens | tok/s"
echo "-------|-------------|--------------|----------|--------|------"
for N in "${FRAME_COUNTS[@]}"; do
    OUTJSON="$OUTDIR/baseline_${N}frames.json"
    if [ -f "$OUTJSON" ]; then
        python3 -c "
import json, sys
with open('$OUTJSON') as f:
    d = json.load(f)
enc = d.get('encode_time_ms', 0)
pre = d.get('prefill_time_ms', 0)
gen = d.get('generation_time_ms', 0)
tok = d.get('n_tokens_generated', 0)
tps = d.get('tokens_per_sec', 0)
print(f'  {$N:>4}  | {enc:>11.1f} | {pre:>12.1f} | {gen:>8.1f} | {tok:>6} | {tps:.1f}')
"
    fi
done
