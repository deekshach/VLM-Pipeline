#!/bin/bash
# Accuracy study: what each optimization changes in what the LLM sees and answers.
#
#   bash jetson/scripts/run_accuracy.sh <sintel|msvd|nextqa|videomme|charades> [clips, comma-separated | all] [configs]
#
# nextqa / videomme / charades: small sets from jetson/tools/accuracy/prepare_newds.py, a prompt per clip from
# <root>/items.json (multiple choice: answer with the option letter). DENSE=0 skips the dense ViT reference.
#
# Per clip and config: llama-jetson-vlm with greedy decoding (--temp 0) and --dump-dir (the embeddings of every
# image as handed to the LLM, the text tokens, the first-token logits, the generated tokens), plus the dense ViT
# reference (every frame encoded whole: native resolution as the baseline sees it, and window-aligned).
# Configs are a ladder, each adding one group of optimizations to the previous, so a change in accuracy can be
# attributed:
#   base      uniform 8 frames, native resolution, stock kernels               (the reference system)
#   sel       adaptive frame selection from live OFA flow
#   align     + frames resized to the 112 px window grid
#   wc        + window cache (threshold WC_T, default 24)
#   arch      + ViT window batch, GEMM F32 out, bias fusion, GEMM table, act-fuse (the kernel switches)
#   ours      + anchor-first (the full system)
#   uni_arch  uniform 8 frames + window align + the kernel switches, no window cache (numerics alone)
#   ours_nocache  ours without the window cache (what the cache adds)
#   pw<T>     ours with the per-window cache (--wc-per-window) at threshold T instead of the rectangle cache
#   pwd<T>    pw<T> + --wc-drop-clean (later frames prefill only their re-encoded windows' tokens)
# Same context size for all (-c 16384): greedy output depends on the attention kernel's KV padding.
# Results: /home/nvidia/data/accuracy/runs/<dataset>/<config>/<clip>.json and .../<clip>/ (dump); RUNS=<dir> elsewhere.
set -u
DS="${1:?sintel or msvd}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN_DIR="${BIN_DIR:-$REPO/build/bin}"   # BIN_DIR=<frozen copy of build/bin>: keep a long run on one build
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # searched before the binaries' RUNPATH
BIN="$BIN_DIR/llama-jetson-vlm"
ACC=/home/nvidia/data/accuracy
TABLE="$REPO/jetson/gemm/table_orin.txt"
WC_T="${WC_T:-24}"
if [ "$DS" = sintel ]; then
    ROOT=/home/nvidia/data/MPI-Sintel; OFA="--ofa-levels 4"; N=96
    P="Describe the scene, motion patterns, and key visual elements in these video frames. Answer in English."
elif [ "$DS" = msvd ]; then
    ROOT=$ACC/msvd; OFA="--ofa-levels 3"; N=48   # 320-640 px frames: the OFA allows 3 pyramid levels
    P="Describe what is happening in this video in one sentence. Answer in English."
else
    ROOT=/mnt/external_ssd/VLM-datasets/eval/$DS; OFA="--ofa-levels 3"; N=48
    grep -q '"refs"' $ROOT/items.json || N=8   # multiple choice: the option letter
    P=""
fi
CLIPS="${2:-all}"
[ "$CLIPS" = all ] && CLIPS="$(ls $ROOT/training/clean | paste -sd,)"
CONFIGS="${3:-base,sel,align,wc,arch,ours,uni_arch}"
ARCH="--vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE --act-fuse"
flags() {
    case "$1" in
        base)     echo "--baseline" ;;
        sel)      echo "--live-flow $OFA" ;;
        align)    echo "--live-flow $OFA --window-align" ;;
        wc)       echo "--live-flow $OFA --window-align --window-cache --wc-threshold $WC_T" ;;
        arch)     echo "--live-flow $OFA --window-align --window-cache --wc-threshold $WC_T $ARCH" ;;
        ours)     echo "--live-flow $OFA --window-align --window-cache --wc-threshold $WC_T $ARCH --anchor-first" ;;
        uni_arch) echo "--baseline --window-align $ARCH" ;;
        ours_nocache) echo "--live-flow $OFA --window-align $ARCH --anchor-first" ;;
        pwd*)     echo "--live-flow $OFA --window-align --wc-per-window --wc-drop-clean --wc-threshold ${1#pwd} $ARCH --anchor-first" ;;
        pw*)      echo "--live-flow $OFA --window-align --wc-per-window --wc-threshold ${1#pw} $ARCH --anchor-first" ;;
        wc*)      echo "--live-flow $OFA --window-align --window-cache --wc-threshold ${1#wc} $ARCH --anchor-first" ;;
    esac
}
for V in ${CLIPS//,/ }; do
    [ -f $ROOT/items.json ] && P="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["prompt"])' $ROOT/items.json $V)"
    # dense ViT reference (no LLM on the GPU; ~1 s per Sintel frame)
    for MODE in native aligned; do
        D=$ACC/dense/$DS/$V/$MODE
        [ "${DENSE:-1}" = 0 ] || [ -f $D/meta.json ] || "$BIN_DIR/jetson-embed-frames" "$MODEL" "$MMPROJ" $ROOT/training/clean/$V $D $MODE \
            > /dev/null 2>&1 || echo "FAIL dense $V $MODE"
    done
    for C in ${CONFIGS//,/ }; do
        OUT=${RUNS:-$ACC/runs}/$DS/$C; mkdir -p $OUT
        [ -f $OUT/$V.json ] && continue
        # shellcheck disable=SC2046
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" -c 16384 --no-mmap --sintel-dir $ROOT --video $V --base-frames 8 \
            -p "$P" -n $N --temp 0 -ngl 99 --vit-flash-attn off $(flags $C) --dump-dir $OUT/$V \
            --decode-threads 3 --output-json $OUT/$V.json > $OUT/$V.log 2>&1 || echo "FAIL $V $C"
        sleep "${COOL:-10}"   # board-safe: let the power rails and temperature settle between runs
    done
    echo "[$(date +%T)] $DS $V done"
done
