#!/bin/bash
# Nsight Compute counters (occupancy, cache hit rates, memory throughput, warp stalls) for one stage
# of a llama-jetson-vlm request, with a given set of switches. Needs sudo for GPU counters.
#
#   sudo bash jetson/scripts/profile_ncu.sh <name> <vit|prefill_img|prefill_txt> <video> [skip] [count] [switch flags...]
#   e.g. sudo bash jetson/scripts/profile_ncu.sh wb vit alley_1 100 120 --live-flow --window-align --vit-window-batch
#
# The model (15 GB) and the GPU share the Orin's RAM, so ncu's default kernel replay runs out of memory
# backing up device memory; application replay reruns the program once per counter group instead
# (minutes per run). Kernels are selected inside the main thread's "request/<stage>/" NVTX range;
# skip/count pick a window of launches (default: skip 100, count 120, about three ViT layers).
# For the ViT alone, tools/vit_shape_probe (jetson-vit-probe) loads only the vision tower, which
# allows the faster kernel replay.
set -u
[ $# -ge 3 ] || { sed -n 2,14p "$0"; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
NAME="$1"; STAGE="$2"; VIDEO="$3"; SKIP="${4:-100}"; COUNT="${5:-120}"
shift $(( $# >= 5 ? 5 : $# ))
OUT="$REPO/jetson/experiments/profiles/ncu"
mkdir -p "$OUT"
NCU="${NCU:-/usr/local/cuda-12.6/bin/ncu}"
P="Describe the scene, motion patterns, and key visual elements in these video frames."
"$NCU" --replay-mode application --nvtx --nvtx-include "request/$STAGE/" --launch-skip "$SKIP" --launch-count "$COUNT" \
    --section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
    --section WarpStateStats -o "$OUT/${NAME}_$STAGE" -f \
    "$REPO/build/bin/llama-jetson-vlm" -m "$MODEL" --mmproj "$MMPROJ" --video "$VIDEO" --base-frames 8 -p "$P" -n 4 -ngl 99 \
    --vit-flash-attn off "$@" --output-json "$OUT/${NAME}_$STAGE.json" > "$OUT/${NAME}_$STAGE.log" 2>&1
grep -E "==ERROR==|No kernels were profiled" "$OUT/${NAME}_$STAGE.log" && echo "!! see $OUT/${NAME}_$STAGE.log"
"$NCU" --import "$OUT/${NAME}_$STAGE.ncu-rep" --page raw --csv > "$OUT/${NAME}_${STAGE}_raw.csv" 2>/dev/null
python3 "$REPO/jetson/tools/ncu_kernel_table.py" "$OUT/${NAME}_${STAGE}_raw.csv" | tee "$OUT/${NAME}_${STAGE}_summary.md"
chown -R "${SUDO_USER:-$(id -un)}:" "$OUT"
