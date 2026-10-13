#!/bin/bash
# Latency and accuracy of per-frame images vs reference frame pairs (temporal_pair_vlm.cpp).
#
#   bash run_temporal_pair_eval.sh sintel|msvd [on|off]
#
# The second argument is the LLM flash-attention setting (default on). With
# "off" only the accuracy runs are made and files carry a _faoff suffix.
#
# Every mode runs on the same videos and frames:
#   lat   fixed 128-token generation (--ignore-eos), warm, for stage timings
#   acc1/acc2  natural stopping, twice: the model is not bit-reproducible, so
#         two runs show how much of a caption-score difference is noise
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/model_config.sh"
BIN="$HERE/../build/bin/llama-temporal-pair-vlm"
export LD_LIBRARY_PATH="$HERE/../build/bin:${LD_LIBRARY_PATH:-}"
R="$HERE/temporal_pair_results"
V="$HERE/window_cache_results"
DS="${1:-sintel}"
FA="${2:-on}"
TAG=""; [[ "$FA" == "off" ]] && TAG="_faoff"
mkdir -p "$R"

if [[ "$DS" == "sintel" ]]; then
    DATA=(--sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel --frames-subdir training/clean)
    PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
    ROUNDS=2
else
    DATA=(--sintel-dir /data/spring26/CPP/datasets/MSVD/MSVD_prepared --frames-subdir training/final)
    PROMPT="Describe what is happening in this video in one sentence."
    ROUNDS=1
fi
VIDS="$(cat "$V/${DS}_videos.txt")"
COMMON=(-m "$MODEL" --mmproj "$MMPROJ" -ngl 99 -c 16384 -fa "$FA" --temp 0 --seed 42 -n 128
        "${DATA[@]}" --videos "$VIDS" --base-frames 8 -p "$PROMPT")

for MODE in images pairs video; do
    if [[ "$FA" == "on" ]]; then
        echo "[$(date +%T)] $DS $MODE lat"
        "$BIN" "${COMMON[@]}" --mode "$MODE" --ignore-eos --rounds "$ROUNDS" --warmup-requests 1 \
            --output-json "$R/${DS}_${MODE}_lat.json" > "$R/${DS}_${MODE}_lat.log" 2>&1 || echo "  FAILED"
    fi
    for K in 1 2; do
        echo "[$(date +%T)] $DS $MODE acc$K$TAG"
        "$BIN" "${COMMON[@]}" --mode "$MODE" --rounds 1 --warmup-requests 1 \
            --output-json "$R/${DS}_${MODE}_acc$K$TAG.json" > "$R/${DS}_${MODE}_acc$K$TAG.log" 2>&1 || echo "  FAILED"
    done
done
echo "[$(date +%T)] $DS done"
