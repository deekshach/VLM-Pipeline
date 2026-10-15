#!/bin/bash
# Nsight Systems trace of one llama-jetson-vlm request with a given set of switches (no sudo).
# NVTX ranges (request, flow, select, anchor, vit, prefill_img, prefill_txt, wc_diff, decode) let
# tools/nsys_stage_kernels.py split GPU time by stage and kernel class, and compare two traces.
#
#   bash jetson/scripts/profile_nsys.sh <name> <video> [switch flags...]
#   e.g. bash jetson/scripts/profile_nsys.sh winbatch alley_1 --live-flow --window-align --vit-window-batch
set -u
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
NAME="$1"; VIDEO="$2"; shift 2
OUT="$REPO/jetson/experiments/profiles/nsys"
mkdir -p "$OUT"
NSYS="${NSYS:-$(command -v nsys || echo /usr/local/bin/nsys)}"
P="Describe the scene, motion patterns, and key visual elements in these video frames."
"$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node -s none --cpuctxsw=none -o "$OUT/$NAME" --force-overwrite true \
    "$REPO/build/bin/llama-jetson-vlm" -m "$MODEL" --mmproj "$MMPROJ" --video "$VIDEO" --base-frames 8 -p "$P" -n 8 -ngl 99 \
    --vit-flash-attn off "$@" --output-json "$OUT/$NAME.json" > "$OUT/$NAME.log" 2>&1
"$NSYS" export --type sqlite --force-overwrite true -o "$OUT/$NAME.sqlite" "$OUT/$NAME.nsys-rep" > /dev/null 2>&1
python3 "$REPO/jetson/tools/nsys_stage_kernels.py" "$OUT/$NAME.sqlite"
echo "trace: $OUT/$NAME.nsys-rep   (compare: python3 jetson/tools/nsys_stage_kernels.py A.sqlite B.sqlite)"
