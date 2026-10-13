#!/bin/bash
# Reuse potential on top of the paired-frame baseline (pair_reuse_vlm.cpp), -fa off.
#
#   bash run_pair_reuse_eval.sh sintel|msvd spread|dense
#
#   spread  8 frames spread evenly over the clip (4 pairs), as in the earlier evals
#   dense   the first 16 consecutive stored frames (8 pairs): little change between pairs
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/model_config.sh"
BIN="$HERE/../build/bin/llama-pair-reuse-vlm"
export LD_LIBRARY_PATH="$HERE/../build/bin:${LD_LIBRARY_PATH:-}"
R="$HERE/pair_reuse_results"
DS="${1:-msvd}"
SET="${2:-spread}"
mkdir -p "$R"

if [[ "$DS" == "sintel" ]]; then
    DATA=(--sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel --frames-subdir training/clean)
    PROMPT="Describe the scene, motion patterns, and key visual elements in these video frames."
    N=64
else
    DATA=(--sintel-dir /data/spring26/CPP/datasets/MSVD/MSVD_prepared --frames-subdir training/final)
    PROMPT="Describe what is happening in this video in one sentence."
    N=128
fi
if [[ "$SET" == "dense" ]]; then FRAMES=(--base-frames 16 --frame-step 1); else FRAMES=(--base-frames 8); fi

echo "[$(date +%T)] $DS $SET"
"$BIN" -m "$MODEL" --mmproj "$MMPROJ" -ngl 99 -c 16384 -fa off --temp 0 --seed 42 -n "$N" \
    "${DATA[@]}" --videos "$(cat "$HERE/window_cache_results/${DS}_videos.txt")" "${FRAMES[@]}" -p "$PROMPT" \
    --pixel-thresholds 6,24,48,96 --cos-thresholds 0.95,0.9,0.8,0.7,0.5 \
    --output-json "$R/${DS}_${SET}.json" > "$R/${DS}_${SET}.log" 2>&1 || echo "  FAILED"
echo "[$(date +%T)] $DS $SET done"
