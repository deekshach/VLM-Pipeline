#!/bin/bash
# Emma latency sweep: fixed frame counts (1,2,4,8,16) + adaptive
# Emma = adaptive selection + full ViT + pre-LLM pruning + post-prefill eviction

set -e

BIN="/data/spring26/CPP/llama.cpp/build/bin/llama-pyramid-gen"
source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"   # single source of truth for weights
VIDEO="alley_1"
OUTDIR="/data/spring26/CPP/llama.cpp/dmc955/generation-only/emma_sweep_results"
N_PREDICT=256

mkdir -p "$OUTDIR"

FRAME_COUNTS=(1 2 4 8 16)

echo "=== Emma Latency Sweep (Fixed Frame Counts) ==="

for N in "${FRAME_COUNTS[@]}"; do
    echo "---------- Emma with $N frames ----------"
    OUTJSON="$OUTDIR/emma_${N}frames.json"
    LOGFILE="$OUTDIR/emma_${N}frames.log"

    "$BIN" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --video "$VIDEO" \
        --n-frames "$N" \
        --n-predict "$N_PREDICT" \
        --output-json "$OUTJSON" \
        -ngl 99 \
        2>&1 | tee "$LOGFILE"

    echo ""
done

echo "---------- Emma ADAPTIVE ----------"
OUTJSON="$OUTDIR/emma_adaptive.json"
LOGFILE="$OUTDIR/emma_adaptive.log"

"$BIN" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --video "$VIDEO" \
    --adaptive \
    --n-predict "$N_PREDICT" \
    --output-json "$OUTJSON" \
    -ngl 99 \
    2>&1 | tee "$LOGFILE"

echo ""
echo "=== Sweep Complete ==="
echo ""

# Summary
echo "Mode    | Frames | Encode(ms) | Prefill(ms) | Evict(ms) | Gen(ms) | Tok | tok/s | Vis_tok | Kept | Kept%"
echo "--------|--------|------------|-------------|-----------|---------|-----|-------|---------|------|------"
for N in "${FRAME_COUNTS[@]}"; do
    F="$OUTDIR/emma_${N}frames.json"
    if [ -f "$F" ]; then
        python3 -c "
import json
with open('$F') as f:
    d = json.load(f)
enc = d.get('encode_time_ms', 0)
pre = d.get('prefill_time_ms', 0)
evi = d.get('eviction_time_ms', 0)
gen = d.get('generation_time_ms', 0)
tok = d.get('n_tokens_generated', 0)
tps = d.get('tokens_per_sec', 0)
vt  = d.get('n_visual_tokens', 0)
vk  = d.get('n_visual_tokens_kept', 0)
pct = 100.0 * vk / vt if vt > 0 else 0
print(f'Emma-{$N:>2} | {$N:>6} | {enc:>10.1f} | {pre:>11.1f} | {evi:>9.1f} | {gen:>7.1f} | {tok:>3} | {tps:>5.1f} | {vt:>7} | {vk:>4} | {pct:>5.1f}%')
"
    fi
done

F="$OUTDIR/emma_adaptive.json"
if [ -f "$F" ]; then
    python3 -c "
import json
with open('$F') as f:
    d = json.load(f)
nf  = d.get('n_frames', 0)
enc = d.get('encode_time_ms', 0)
pre = d.get('prefill_time_ms', 0)
evi = d.get('eviction_time_ms', 0)
gen = d.get('generation_time_ms', 0)
tok = d.get('n_tokens_generated', 0)
tps = d.get('tokens_per_sec', 0)
vt  = d.get('n_visual_tokens', 0)
vk  = d.get('n_visual_tokens_kept', 0)
pct = 100.0 * vk / vt if vt > 0 else 0
print(f'Adapt   | {nf:>6} | {enc:>10.1f} | {pre:>11.1f} | {evi:>9.1f} | {gen:>7.1f} | {tok:>3} | {tps:>5.1f} | {vt:>7} | {vk:>4} | {pct:>5.1f}%')
"
fi
