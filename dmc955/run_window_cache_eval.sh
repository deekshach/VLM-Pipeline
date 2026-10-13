#!/bin/bash
# Latency and accuracy of the per-frame ViT policies in window_cache_vlm.cpp.
#
#   bash run_window_cache_eval.sh sintel|msvd
#
# Every config runs twice on the same videos and frames:
#   lat  fixed 128-token generation (--ignore-eos), warm, for stage timings
#   acc  natural stopping plus --verify, for captions and embedding fidelity
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/model_config.sh"
BIN="$HERE/../build/bin/llama-window-cache-vlm"
export LD_LIBRARY_PATH="$HERE/../build/bin:${LD_LIBRARY_PATH:-}"
R="$HERE/window_cache_results"
DS="${1:-sintel}"

if [[ "$DS" == "sintel" ]]; then
    DATA=(--sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel --frames-subdir training/clean)
    PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
    ROUNDS=2
else
    DATA=(--sintel-dir /data/spring26/CPP/datasets/MSVD/MSVD_prepared --frames-subdir training/final)
    PROMPT="Describe what is happening in this video in one sentence."
    ROUNDS=1
fi
VIDS="$(cat "$R/${DS}_videos.txt")"
COMMON=(-m "$MODEL" --mmproj "$MMPROJ" -ngl 99 -c 16384 -fa on --temp 0 --seed 42 -n 128
        "${DATA[@]}" --videos "$VIDS" --base-frames 8 -p "$PROMPT")

run() {   # name, then policy flags
    local name="$1"; shift
    echo "[$(date +%T)] $DS $name lat"
    "$BIN" "${COMMON[@]}" "$@" --ignore-eos --rounds "$ROUNDS" --warmup-requests 1 \
        --output-json "$R/${DS}_${name}_lat.json" > "$R/${DS}_${name}_lat.log" 2>&1 || echo "  FAILED"
    echo "[$(date +%T)] $DS $name acc"
    "$BIN" "${COMMON[@]}" "$@" --verify --rounds 1 --warmup-requests 0 \
        --output-json "$R/${DS}_${name}_acc.json" > "$R/${DS}_${name}_acc.log" 2>&1 || echo "  FAILED"
}

run full_native   --policy full
run full_native2  --policy full
run full_aligned  --policy full --align 112
run legacy        --policy legacy
for T in 6 24 48 96; do
    run "window_t$T" --policy window --align 112 --diff-threshold "$T"
done
GGML_CUDA_DISABLE_GRAPHS=1 run window_t48_nograph --policy window --align 112 --diff-threshold 48
echo "[$(date +%T)] $DS done"
