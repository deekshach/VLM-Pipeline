#!/bin/bash
# TTFT of the adaptive pipeline on all Sintel clips: ground-truth flow files vs live OFA flow
# (serial, pipelined, pipelined + anchor-first). ViT flash attention off; 8 generated tokens
# (decode is not part of TTFT).
#
#   bash run_sweep.sh [out_dir]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
# originally run with dmc955 llama-adaptive-vlm; llama-jetson-vlm has the same flags and defaults
BIN="$REPO/build/bin/llama-jetson-vlm"
OUT="${1:-$HERE/results}"
mkdir -p "$OUT"
P="Describe the scene, motion patterns, and key visual elements in these video frames."
CONFIGS=(
    "gt_files:"
    "live_serial:--live-flow-serial"
    "live_pipelined:--live-flow"
    "live_anchor:--live-flow --anchor-first"
)
for V in $(ls /home/nvidia/data/MPI-Sintel/training/clean | sort); do
    for C in "${CONFIGS[@]}"; do
        NAME="${C%%:*}"; FLAGS="${C#*:}"
        J="$OUT/${V}_${NAME}.json"
        [ -f "$J" ] && continue
        # shellcheck disable=SC2086
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --video "$V" --base-frames 8 -p "$P" -n 8 -ngl 99 \
            --vit-flash-attn off $FLAGS --output-json "$J" > "$OUT/${V}_${NAME}.log" 2>&1 || echo "FAIL $V $NAME"
    done
done
ls "$OUT"/*.json | wc -l
