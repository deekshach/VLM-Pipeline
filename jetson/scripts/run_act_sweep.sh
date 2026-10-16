#!/bin/bash
# TTFT of llama-jetson-vlm with the GEMM-input switches (--act-dedup, --act-fuse) on top of every other switch,
# same clips, prompt and settings as run_switch_sweep.sh (ViT flash attention off; LLM flash attention auto = on).
#
#   bash jetson/scripts/run_act_sweep.sh [out_dir] [clips, comma-separated | all]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
OUT="${1:-$REPO/jetson/experiments/act_fusion}"
CLIPS="${2:-alley_1,sleeping_1,bandage_1,temple_2,ambush_5,cave_2}"
[ "$CLIPS" = all ] && CLIPS="$(ls /home/nvidia/data/MPI-Sintel/training/clean | paste -sd,)"
TABLE="$REPO/jetson/gemm/table_orin.txt"
ALL="--window-cache --wc-threshold 24 --vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE"
mkdir -p "$OUT"
P="Describe the scene, motion patterns, and key visual elements in these video frames."
CONFIGS=(
    "all_anchor:--live-flow $ALL --anchor-first"
    "all_anchor_dedup:--live-flow $ALL --anchor-first --act-dedup"
    "all_anchor_fuse:--live-flow $ALL --anchor-first --act-fuse"
    "uniform_all:--baseline $ALL"
    "uniform_all_fuse:--baseline $ALL --act-fuse"
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
