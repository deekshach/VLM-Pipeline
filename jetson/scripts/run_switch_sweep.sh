#!/bin/bash
# TTFT of llama-jetson-vlm with each optimization switched on alone and all together.
# ViT flash attention off throughout (the study's setting); 8 generated tokens (decode is not TTFT).
#
#   bash jetson/scripts/run_switch_sweep.sh [out_dir] [clips, comma-separated | all]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
OUT="${1:-$REPO/jetson/experiments/switch_sweep}"
CLIPS="${2:-alley_1,sleeping_1,bandage_1,temple_2,ambush_5,cave_2}"
[ "$CLIPS" = all ] && CLIPS="$(ls /home/nvidia/data/MPI-Sintel/training/clean | paste -sd,)"
TABLE="$REPO/jetson/gemm/table_orin.txt"
WC="--window-cache --wc-threshold 24"
ALL="$WC --vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE"
mkdir -p "$OUT"
P="Describe the scene, motion patterns, and key visual elements in these video frames."
CONFIGS=(
    "uniform:--baseline"
    "uniform_all:--baseline $ALL"
    "adaptive:--live-flow"
    "adaptive_align:--live-flow --window-align"
    "adaptive_wcache:--live-flow $WC"
    "adaptive_winbatch:--live-flow --window-align --vit-window-batch"
    "adaptive_f32out:--live-flow --gemm-f32-out"
    "adaptive_f32out_bias:--live-flow --gemm-f32-out --gemm-bias-fuse"
    "adaptive_gemmtable:--live-flow --gemm-table $TABLE"
    "adaptive_all:--live-flow $ALL"
    "adaptive_all_anchor:--live-flow $ALL --anchor-first"
)
for V in ${CLIPS//,/ }; do
    for C in "${CONFIGS[@]}"; do
        NAME="${C%%:*}"; FLAGS="${C#*:}"
        J="$OUT/${V}__${NAME}.json"
        [ -f "$J" ] && continue
        # shellcheck disable=SC2086
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --video "$V" --base-frames 8 -p "$P" -n 8 -ngl 99 \
            --vit-flash-attn off $FLAGS --output-json "$J" > "$OUT/${V}__${NAME}.log" 2>&1 || echo "FAIL $V $NAME"
    done
done
ls "$OUT"/*.json | wc -l
