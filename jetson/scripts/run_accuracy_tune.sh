#!/bin/bash
# Accuracy study, tuning variants of the full system (run_accuracy.sh "ours") aimed at keeping accuracy:
#   ours_skip     window cache reuses a frame only when nothing changed; any dirty window -> full encode (no crops:
#                 a crop is encoded without the rest of the frame, so its tokens miss the global-attention context)
#   ours_nocache  no window cache
#   ours_min5     at least 5 frames (adaptive selection picks the minimum, 3, on most short clips)
#   ours_min5_skip  both
#
#   bash jetson/scripts/run_accuracy_tune.sh <sintel|msvd> [clips | all] [configs]
set -u
DS="${1:?sintel or msvd}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
ACC=/home/nvidia/data/accuracy
TABLE="$REPO/jetson/gemm/table_orin.txt"
if [ "$DS" = sintel ]; then
    ROOT=/home/nvidia/data/MPI-Sintel; OFA="--ofa-levels 4"; N=96
    P="Describe the scene, motion patterns, and key visual elements in these video frames. Answer in English."
else
    ROOT=$ACC/msvd; OFA="--ofa-levels 3"; N=48
    P="Describe what is happening in this video in one sentence. Answer in English."
fi
CLIPS="${2:-all}"
[ "$CLIPS" = all ] && CLIPS="$(ls $ROOT/training/clean | paste -sd,)"
CONFIGS="${3:-ours_skip,ours_nocache,ours_min5,ours_min5_skip}"
ARCH="--vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE --act-fuse --anchor-first"
WC="--window-cache --wc-threshold 24"
flags() {
    case "$1" in
        ours_skip)      echo "--live-flow $OFA --window-align $WC --wc-rekey-frac 0 $ARCH" ;;
        ours_nocache)   echo "--live-flow $OFA --window-align $ARCH" ;;
        ours_min5)      echo "--live-flow $OFA --window-align $WC $ARCH --min-frames 5" ;;
        ours_min5_skip) echo "--live-flow $OFA --window-align $WC --wc-rekey-frac 0 $ARCH --min-frames 5" ;;
    esac
}
for V in ${CLIPS//,/ }; do
    for C in ${CONFIGS//,/ }; do
        OUT=$ACC/runs/$DS/$C; mkdir -p $OUT
        [ -f $OUT/$V.json ] && continue
        # shellcheck disable=SC2046
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" -c 16384 --no-mmap --sintel-dir $ROOT --video $V --base-frames 8 \
            -p "$P" -n $N --temp 0 -ngl 99 --vit-flash-attn off $(flags $C) --dump-dir $OUT/$V \
            --output-json $OUT/$V.json > $OUT/$V.log 2>&1 || echo "FAIL $V $C"
    done
    echo "[$(date +%T)] tune $DS $V done"
done
