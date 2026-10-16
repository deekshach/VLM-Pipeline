#!/bin/bash
# Ablation of the engine switches: every valid combination, so each microarchitectural change can be
# attributed to the switch (or pair of switches) that causes it.
#
#   ViT (jetson-vit-probe, flash attention off): W = MTMD_VIT_WINDOW_BATCH, F = GGML_CUDA_JETSON_GEMM_F32OUT,
#       B = GGML_CUDA_JETSON_GEMM_BIAS (needs F), T = GGML_CUDA_JETSON_GEMM_TABLE; 2 x 6 = 12 configs, each on a
#       full frame (9x4 windows) and the window-cache crops 5x2 and 1x2.
#   LLM prefill (jetson-prefill-probe): F, B, T (6 configs) x N = 1152, 2304, 4608 image tokens
#       (2, 4, 8 frames: the adaptive-sampling axis, i.e. KV-cache length).
#
#   bash jetson/scripts/run_ablation.sh nsys     # no sudo: real-clock kernel time of every config (mean of
#                                                # 5 ViT / 3 prefill repetitions), whole
#                                                # frame / whole prefill; also fixes the ncu launch windows
#   sudo bash jetson/scripts/run_ablation.sh ncu # hardware counters, two layers per run (incl. global layer 7)
#   sudo bash jetson/scripts/run_ablation.sh warm # LLM DRAM read with caches NOT flushed between kernels (one
#                                                # result per ubatch range) vs the flushed per-kernel reads
#   python3 jetson/tools/ablation_analyze.py     # afterwards, no sudo
set -u
MODE="${1:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NSYS="${NSYS:-$(command -v nsys || echo /usr/local/bin/nsys)}"
NCU="${NCU:-/usr/local/cuda-12.6/bin/ncu}"
M=/home/nvidia/data/models/Qwen2.5VL
LLM="$M/Qwen2.5-VL-7B-Instruct-BF16.gguf"; MMPROJ="$M/mmproj-BF16.gguf"
FRAME=/home/nvidia/data/MPI-Sintel/training/clean/alley_1/frame_0001.png
TABLE="$REPO/jetson/gemm/table_orin.txt"
OUT="$REPO/jetson/experiments/profiles/ablation"
VIT_CFGS="base T F FT FB FBT W WT WF WFT WFB WFBT"
LLM_CFGS="base T F FT FB FBT"
SHAPES="9x4 5x2 1x2"
NS="1152 2304 4608"
mkdir -p "$OUT/nsys" "$OUT/ncu" "$OUT/warm"

envs_for() {   # config name -> environment assignments
    local c="$1" e=()
    [[ "$c" == *W* ]] && e+=(MTMD_VIT_WINDOW_BATCH=1)
    [[ "$c" == *F* ]] && e+=(GGML_CUDA_JETSON_GEMM_F32OUT=1)
    [[ "$c" == *B* ]] && e+=(GGML_CUDA_JETSON_GEMM_BIAS=1)
    [[ "$c" == *T* ]] && e+=(GGML_CUDA_JETSON_GEMM_TABLE="$TABLE")
    echo "${e[@]}"
}

case "$MODE" in
nsys)
    for C in $VIT_CFGS; do
        echo "[$(date +%T)] nsys vit_$C"
        env PROBE_MARK_ALL=1 $(envs_for "$C") "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node -s none --cpuctxsw=none \
            -o "$OUT/nsys/vit_$C" --force-overwrite true "$REPO/build/bin/jetson-vit-probe" "$LLM" "$MMPROJ" "$FRAME" \
            9x4,5x2,1x2 2 5 off > "$OUT/nsys/vit_$C.log" 2>&1
        "$NSYS" export --type sqlite --force-overwrite true -o "$OUT/nsys/vit_$C.sqlite" "$OUT/nsys/vit_$C.nsys-rep" > /dev/null 2>&1
    done
    for C in $LLM_CFGS; do
        echo "[$(date +%T)] nsys llm_$C"
        env PROBE_MARK_ALL=1 $(envs_for "$C") "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node -s none --cpuctxsw=none \
            -o "$OUT/nsys/llm_$C" --force-overwrite true "$REPO/build/bin/jetson-prefill-probe" "$LLM" \
            1152,2304,4608 1 3 > "$OUT/nsys/llm_$C.log" 2>&1
        "$NSYS" export --type sqlite --force-overwrite true -o "$OUT/nsys/llm_$C.sqlite" "$OUT/nsys/llm_$C.nsys-rep" > /dev/null 2>&1
    done
    # launch windows for ncu: the kernel indices (inside each NVTX range) of two whole layers
    python3 "$REPO/jetson/tools/ablation_analyze.py" --windows "$OUT/nsys" > "$OUT/windows.txt"
    echo "[$(date +%T)] done; launch windows in $OUT/windows.txt"
    ;;
ncu|warm)
    [ -f "$OUT/windows.txt" ] || { echo "run '$0 nsys' first"; exit 1; }
    if [ "$MODE" = ncu ]; then
        DEST="$OUT/ncu"
        SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis
                  --section WarpStateStats --metrics lts__t_sectors.sum,lts__d_sectors_fill_sysmem.sum)
        REPLAY=(--replay-mode kernel --cache-control all)
        FILTER=cat
    else
        # DRAM bytes with caches kept between kernels: one result per whole LLM ubatch range (app-range replay).
        # Per-kernel profiling cannot measure this: kernel replay saves/restores memory around each kernel, and
        # per-kernel application replay disturbs L2 between kernels (no change measured even where the real-run
        # time shows reuse), hangs at the LLM flash-attention kernel and in ViT configs without F. app-range
        # crashes the ViT probe, so the ViT warm effect is measured as time instead (report section 3).
        DEST="$OUT/warm"
        SECTIONS=(--metrics gpu__time_duration.sum,lts__t_sectors.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sector_hit_rate.pct)
        FILTER='grep -E ^llm_base_'
    fi
    # windows.txt: <run name> <nvtx range> <skip> <count>, e.g. "vit_WF_9x4 vit_9x4 237 83"
    grep -v '^#' "$OUT/windows.txt" | $FILTER | while read -r NAME RANGE SKIP COUNT; do
        CFG="${NAME#*_}"; CFG="${CFG%_*}"
        [ -f "$DEST/${NAME}_raw.csv" ] && { echo "   $NAME exists, skipped"; continue; }
        echo "[$(date +%T)] $MODE $NAME"
        if [[ "$NAME" == vit_* ]]; then
            APP=("$REPO/build/bin/jetson-vit-probe" "$LLM" "$MMPROJ" "$FRAME" "${RANGE#vit_}" 2 1 off)
        else
            APP=("$REPO/build/bin/jetson-prefill-probe" "$LLM" "${RANGE#prefill_last_}" 1 1)
        fi
        WINDOW=(--launch-skip "$SKIP" --launch-count "$COUNT")
        [ "$MODE" = warm ] && { REPLAY=(--replay-mode app-range --cache-control none); WINDOW=(); }   # whole range
        run() {
            env $(envs_for "$CFG") timeout -k 30 "${RUN_TIMEOUT:-2400}" "$NCU" --nvtx --nvtx-include "$RANGE/" "${WINDOW[@]}" \
                "$@" "${SECTIONS[@]}" -o "$DEST/$NAME" -f "${APP[@]}" > "$DEST/$NAME.log" 2>&1
        }
        run "${REPLAY[@]}"
        if grep -qE "==ERROR==|No kernels were profiled" "$DEST/$NAME.log" && [ "$MODE" = ncu ]; then
            echo "   kernel replay failed, retrying with application replay"
            run --replay-mode application --cache-control all
        fi
        grep -E "==ERROR==|No kernels were profiled" "$DEST/$NAME.log" && echo "   !! see $DEST/$NAME.log"
        [ -s "$DEST/$NAME.ncu-rep" ] || { echo "   !! no report (timed out after ${RUN_TIMEOUT:-2400} s?), see $DEST/$NAME.log"; continue; }
        "$NCU" --import "$DEST/$NAME.ncu-rep" --page raw --csv --print-units base > "$DEST/${NAME}_raw.csv" 2>/dev/null
        echo "   $(($(wc -l < "$DEST/${NAME}_raw.csv") - 2)) kernels"
    done
    chown -R "${SUDO_USER:-$(id -un)}:" "$OUT"
    echo "[$(date +%T)] done; analyse with: python3 jetson/tools/ablation_analyze.py"
    ;;
*)
    sed -n 2,19p "$0"; exit 1 ;;
esac
