#!/bin/bash
# Probes for the evaluation's opportunity analysis, under the current power mode (no sudo):
#   B. LLM prefill time against token count (jetson-prefill-probe), N = 576..2560 in steps of 64 and 3072..6144 in
#      steps of 512, stock kernels and every LLM switch: the 512-token ubatch staircase that decides what dropping
#      clean tokens can save                                                   -> <out>/prefill_{stock,design}.csv
#   C. ViT time against the number of re-encoded windows (jetson-vit-probe w:), Sintel alley_1 at 9 x 4 windows,
#      batched window attention only and every ViT switch: the window cache's fixed cost and its per-window cost
#                                                                               -> <out>/vitw_{winbatch,design}.csv
#   T. the GEMM table retuned for this mode (jetson-gemm-tune over gemm/shapes.txt)
#                                                       -> jetson/gemm/table_orin_<tag>.txt, <out>/tune_report.txt
#
#   bash jetson/scripts/run_probes.sh <tag, e.g. 15w> [B C T]
#
# Out: jetson/experiments/probes/<tag>/. The junction must be under 50 C before each part; the board guard is
# stopped for the run and restarted on exit (it pauses requests on over-current events).
set -u
TAG="${1:?mode tag, e.g. 15w}"; shift
PARTS="${*:-B C T}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
M=/home/nvidia/data/models/Qwen2.5VL
FRAME=/home/nvidia/data/MPI-Sintel/training/clean/alley_1/frame_0001.png
TABLE="$REPO/jetson/gemm/table_orin.txt"
OUT="$REPO/jetson/experiments/probes/$TAG"
GUARD_LOG=/home/nvidia/data/accuracy/board_guard.log
TZ=$(grep -l tj-thermal /sys/devices/virtual/thermal/thermal_zone*/type | head -1 | xargs dirname)/temp
tj() { echo $(( $(cat "$TZ") / 1000 )); }
log() { echo "[$(date +%T)] $*"; }
cool() { while [ "$(tj)" -ge 50 ]; do sleep 10; done; }
mkdir -p "$OUT"
{ nvpmodel -q 2>/dev/null | head -1; echo "gpu $(cat /sys/class/devfreq/17000000.gpu/cur_freq) Hz"; } > "$OUT/mode.txt"

GPID=$(pgrep -x bash -a | awk '/board_guard.sh \/home/ {print $1}')
[ -n "$GPID" ] && kill "$GPID" && echo "[$(date +%T)] guard stopped by run_probes.sh (restarted on exit)" >> "$GUARD_LOG"
restart_guard() {
    pgrep -x bash -a | grep -q "board_guard.sh /home" && return
    (cd "$REPO" && setsid nohup bash jetson/scripts/board_guard.sh "$GUARD_LOG" > /dev/null 2>&1 < /dev/null &)
    log "guard restarted"
}
trap restart_guard EXIT

LLM_ENV="GGML_CUDA_JETSON_GEMM_F32OUT=1 GGML_CUDA_JETSON_GEMM_BIAS=1 GGML_CUDA_JETSON_GEMM_TABLE=$TABLE GGML_CUDA_JETSON_ACT_FUSE=1 GGML_CUDA_JETSON_GEMM_RESIDUAL=1"
VIT_ENV="MTMD_VIT_WINDOW_BATCH=1 $LLM_ENV GGML_CUDA_JETSON_ROPE_F16=1"

if [[ " $PARTS " == *" B "* ]]; then
    NS=$(seq -s, 576 64 2560),$(seq -s, 3072 512 6144)
    for CFG in stock design; do
        E=""; [ $CFG = design ] && E="$LLM_ENV"
        cool; log "B prefill $CFG"
        # shellcheck disable=SC2086
        env $E "$REPO/build/bin/jetson-prefill-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" "$NS" 1 2 \
            2> "$OUT/prefill_$CFG.log" | grep -E '^(N,|[0-9]+,)' > "$OUT/prefill_$CFG.csv"
        log "   $(($(wc -l < "$OUT/prefill_$CFG.csv") - 1)) token counts"
    done
fi

if [[ " $PARTS " == *" C "* ]]; then
    KS="w:1,2,3,4,6,8,9,12,15,18,21,24,27,30,33,36"
    for CFG in winbatch design; do
        E="MTMD_VIT_WINDOW_BATCH=1"; [ $CFG = design ] && E="$VIT_ENV"
        cool; log "C vit per-window $CFG"
        # shellcheck disable=SC2086
        env $E "$REPO/build/bin/jetson-vit-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" "$M/mmproj-BF16.gguf" "$FRAME" \
            "$KS" 2 5 on 2> "$OUT/vitw_$CFG.log" | grep -E '^(key|windows,|[0-9]+,)' > "$OUT/vitw_$CFG.csv"
        log "   $(head -1 "$OUT/vitw_$CFG.csv")"
    done
fi

if [[ " $PARTS " == *" T "* ]]; then
    cool; log "T gemm table for $TAG"
    "$REPO/build/bin/jetson-gemm-tune" "$REPO/jetson/gemm/shapes.txt" "$REPO/jetson/gemm/table_orin_$TAG.txt" \
        > "$OUT/tune_report.txt" 2> "$OUT/tune.log"
    log "   $(grep -c . "$REPO/jetson/gemm/table_orin_$TAG.txt") shapes -> gemm/table_orin_$TAG.txt"
fi
log "done"
