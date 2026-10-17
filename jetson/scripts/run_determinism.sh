#!/bin/bash
# Is the pipeline's LLM output deterministic? Runs one clip and config REPS times per variant (greedy, one generated
# token) and prints a hash of the first-token logits of each run: identical hashes = deterministic.
#
#   bash jetson/scripts/run_determinism.sh [clip=alley_1] [reps=3] [variants]
#
# Variants (each toggles one suspect; all use the baseline config, uniform 8 frames, stock kernels):
#   ref        as in the accuracy study
#   nographs   GGML_CUDA_DISABLE_GRAPHS=1 (CUDA graph capture/replay off, LLM and ViT)
#   nofa       LLM flash attention off (-fa off)
#   t1         one CPU thread (-t 1 -tb 1)
#   ub2048     one ubatch per image chunk (-ub 2048: no 512 + 64 split)
#   smallctx   -c 8192 (different KV padding)
#
# Result (alley_1): logits differed every run with FA on at -c 16384, not with -fa off or -c <= 12288. Cause: ggml-cuda's
# fused mul_mat_vec + SwiGLU wrote its output over its own input (ggml-alloc reuse), a race; fixed in ggml-cuda.cu
# (ggml_cuda_fused_dst_overlaps). The same race is reproducible without the ViT: PROBE_CTX=16384 jetson-prefill-probe.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
V="${1:-alley_1}"; REPS="${2:-3}"; VARIANTS="${3:-ref,nographs,nofa,t1,ub2048,smallctx}"
OUT=/home/nvidia/data/accuracy/determinism/$V; mkdir -p $OUT
P="Describe the scene, motion patterns, and key visual elements in these video frames. Answer in English."
for VAR in ${VARIANTS//,/ }; do
    ENV=(); F=(-c 16384)
    case $VAR in
        nographs) ENV=(GGML_CUDA_DISABLE_GRAPHS=1) ;;
        nofa)     F+=(-fa off) ;;
        t1)       F+=(-t 1 -tb 1) ;;
        ub2048)   F+=(-ub 2048) ;;
        smallctx) F=(-c 8192) ;;
    esac
    for R in $(seq 1 $REPS); do
        D=$OUT/$VAR/r$R; rm -rf $D; mkdir -p $OUT/$VAR
        env "${ENV[@]}" "$REPO/build/bin/llama-jetson-vlm" -m "$MODEL" --mmproj "$MMPROJ" --no-mmap "${F[@]}" --video $V \
            --base-frames 8 -p "$P" -n 1 --temp 0 -ngl 99 --vit-flash-attn off --baseline --dump-dir $D \
            --output-json $D.json > $D.log 2>&1 || { echo "$VAR r$R FAIL"; continue; }
        sleep "${COOL:-10}"   # board-safe: let the power rails and temperature settle between runs
        echo "$VAR r$R logits $(md5sum < $D/logits0.f32 | cut -c1-12)  images $(cat $D/img_*.f32 | md5sum | cut -c1-12)"
    done
done
