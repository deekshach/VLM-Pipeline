#!/bin/bash
# Microarchitecture of the kernel switches against stock llama.cpp kernels, ViT flash attention on (the main setting),
# one switch added per step in the E3 ladder's order. Needs sudo (GPU counters for ncu):
#
#   sudo bash jetson/scripts/run_microarch_fa.sh [wait-for-pid]
#
# With a pid it first waits for that process (the evaluation campaign) to end. Then, at MAXN with clocks pinned,
# the board guard stopped (restarted on exit) and the junction under 50 C before each part:
#   A. Nsight Systems, the 8-frame request (Sintel alley_1): stock, every kernel switch, and every switch but the
#      residual-in-GEMM one (it may slow the LLM prefill) -> experiments/profiles/nsys/{stock,design,design_nores}_fa
#   B. Nsight Compute, one full-frame ViT encode (9 x 4 windows, every kernel of it), the ladder
#      base, winbatch, f32bias, table, actfuse, ropef16, design -> experiments/profiles/microarch_ladder/vit_<cfg>
#   C. Nsight Compute, the last 512-token ubatch of an 8-frame image prefill (4,608 tokens), the ladder steps that
#      touch the LLM: base, f32bias, table, actfuse, design -> experiments/profiles/microarch_ladder/llm_<cfg>
# Afterwards (no sudo): python3 jetson/tools/microarch_ladder.py jetson/experiments/profiles/microarch_ladder
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCU="${NCU:-/usr/local/cuda-12.6/bin/ncu}"
M=/home/nvidia/data/models/Qwen2.5VL
FRAME=/home/nvidia/data/MPI-Sintel/training/clean/alley_1/frame_0001.png
TABLE="$REPO/jetson/gemm/table_orin.txt"
OUT="$REPO/jetson/experiments/profiles/microarch_ladder"
GUARD_LOG=/home/nvidia/data/accuracy/board_guard.log
USR="${SUDO_USER:-$(id -un)}"
TZ=$(grep -l tj-thermal /sys/devices/virtual/thermal/thermal_zone*/type | head -1 | xargs dirname)/temp
tj() { echo $(( $(cat "$TZ") / 1000 )); }
log() { echo "[$(date +%T)] $*"; }
cool() { while [ "$(tj)" -ge 50 ]; do sleep 10; done; }

if [ -n "${1:-}" ]; then
    log "waiting for pid $1"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    while pgrep -f "llama-jetson-vlm|llama-temporal-pair-vlm" > /dev/null; do sleep 30; done
    log "pid $1 ended"
fi
F=/sys/class/devfreq/17000000.gpu
MODE=$(nvpmodel -q 2>/dev/null | head -1 | sed 's/.*: //')
if [ "$MODE" != MAXN ] || [ "$(cat $F/min_freq)" != "$(cat $F/max_freq)" ]; then
    log "board is '$MODE', clocks $(cat $F/min_freq)-$(cat $F/max_freq): want MAXN pinned (sudo jetson_clocks)"; exit 1
fi
GPID=$(pgrep -x bash -a | awk '/board_guard.sh \/home/ {print $1}')
[ -n "$GPID" ] && kill "$GPID" && echo "[$(date +%T)] guard stopped by run_microarch_fa.sh (restarted on exit)" >> "$GUARD_LOG"
restart_guard() {
    chown -R "$USR:" "$OUT" "$REPO/jetson/experiments/profiles/nsys" 2>/dev/null
    pgrep -x bash -a | grep -q "board_guard.sh /home" && return
    (cd "$REPO" && sudo -u "$USR" setsid nohup bash jetson/scripts/board_guard.sh "$GUARD_LOG" > /dev/null 2>&1 < /dev/null &)
    log "guard restarted"
}
trap restart_guard EXIT
mkdir -p "$OUT"

# kernel switches as environment (the engine's own names), cumulative in the ladder's order
declare -A ENVS
ENVS[base]=""
ENVS[winbatch]="MTMD_VIT_WINDOW_BATCH=1"
ENVS[f32bias]="${ENVS[winbatch]} GGML_CUDA_JETSON_GEMM_F32OUT=1 GGML_CUDA_JETSON_GEMM_BIAS=1"
ENVS[table]="${ENVS[f32bias]} GGML_CUDA_JETSON_GEMM_TABLE=$TABLE"
ENVS[actfuse]="${ENVS[table]} GGML_CUDA_JETSON_ACT_FUSE=1"
ENVS[ropef16]="${ENVS[actfuse]} GGML_CUDA_JETSON_ROPE_F16=1"
ENVS[design]="${ENVS[ropef16]} GGML_CUDA_JETSON_GEMM_RESIDUAL=1"
SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis
          --section WarpStateStats --section ComputeWorkloadAnalysis
          --metrics lts__t_sectors.sum,lts__d_sectors_fill_sysmem.sum,sm__inst_executed.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active)
export_csv() {
    "$NCU" --import "$OUT/$1.ncu-rep" --page raw --csv --print-units base > "$OUT/$1_raw.csv" 2>/dev/null
    log "   $(($(wc -l < "$OUT/$1_raw.csv") - 2)) kernels -> $1_raw.csv"
}

# A. nsys, whole request
ARCH="--vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE --act-fuse --vit-rope-f16"
cool; log "A stock_fa"
sudo -u "$USR" bash "$HERE/profile_nsys.sh" stock_fa alley_1 --baseline --vit-flash-attn on | tail -3
cool; log "A design_fa"
sudo -u "$USR" bash "$HERE/profile_nsys.sh" design_fa alley_1 --baseline --window-align $ARCH --gemm-residual-fuse --vit-flash-attn on | tail -3
cool; log "A design_nores_fa"
sudo -u "$USR" bash "$HERE/profile_nsys.sh" design_nores_fa alley_1 --baseline --window-align $ARCH --vit-flash-attn on | tail -3

# B. ncu, one full-frame ViT encode per ladder step
for CFG in base winbatch f32bias table actfuse ropef16 design; do
    N="vit_$CFG"; cool; log "B $N"
    # shellcheck disable=SC2086
    env ${ENVS[$CFG]} "$NCU" --nvtx --nvtx-include "vit_9x4/" "${SECTIONS[@]}" -o "$OUT/$N" -f \
        "$REPO/build/bin/jetson-vit-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" "$M/mmproj-BF16.gguf" "$FRAME" 9x4 1 1 on \
        > "$OUT/$N.log" 2>&1
    grep -E "==ERROR==|No kernels were profiled" "$OUT/$N.log" && log "   !! see $OUT/$N.log"
    export_csv "$N"
done

# C. ncu, the last ubatch of an 8-frame image prefill per ladder step that touches the LLM
for CFG in base f32bias table actfuse design; do
    N="llm_$CFG"; cool; log "C $N"
    # shellcheck disable=SC2086
    env ${ENVS[$CFG]} "$NCU" --nvtx --nvtx-include "prefill_last_4608/" "${SECTIONS[@]}" -o "$OUT/$N" -f \
        "$REPO/build/bin/jetson-prefill-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" 4608 1 1 > "$OUT/$N.log" 2>&1
    grep -E "==ERROR==|No kernels were profiled" "$OUT/$N.log" && log "   !! see $OUT/$N.log"
    export_csv "$N"
done
log "done"
