#!/bin/bash
# Hardware counters for each optimization switch, off vs on (sudo for GPU counters):
#   A. ViT kernels (vision tower only, jetson-vit-probe, fast kernel replay), ViT flash attention off:
#      configurations base / winbatch (MTMD_VIT_WINDOW_BATCH) / f32bias (GGML_CUDA_JETSON_GEMM_F32OUT+BIAS)
#      x shapes 9x4 (full frame, 2304 patches), 5x2 (640) and 1x2 (128) windows -- the window-cache crops.
#   B. The LLM prefill down projection (3584 x 18944, 512 tokens), the cuBLASLt heuristic's split-K
#      algorithm vs the tuned one (jetson-gemm-tune --profile).
#
#   sudo bash jetson/scripts/run_microarch_ncu.sh
#   python3 jetson/tools/ncu_compare.py jetson/experiments/profiles/microarch     # afterwards, no sudo
#
# Besides the usual sections, two L2 counters: lts__t_sectors.sum (all L2 sector requests) and
# lts__d_sectors_fill_sysmem.sum (sectors filled from DRAM, x 32 B = main-memory read traffic).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCU="${NCU:-/usr/local/cuda-12.6/bin/ncu}"
M=/home/nvidia/data/models/Qwen2.5VL
FRAME=/home/nvidia/data/MPI-Sintel/training/clean/alley_1/frame_0001.png
OUT="$REPO/jetson/experiments/profiles/microarch"
mkdir -p "$OUT"
SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis
          --section WarpStateStats --metrics lts__t_sectors.sum,lts__d_sectors_fill_sysmem.sum)

export_csv() {   # $1 = report basename
    "$NCU" --import "$OUT/$1.ncu-rep" --page raw --csv --print-units base > "$OUT/$1_raw.csv" 2>/dev/null
    echo "   $(($(wc -l < "$OUT/$1_raw.csv") - 2)) kernels -> $1_raw.csv"
}

# A. ViT
for CFG in base winbatch f32bias; do
    ENVS=()
    [ "$CFG" = winbatch ] && ENVS=(MTMD_VIT_WINDOW_BATCH=1)
    [ "$CFG" = f32bias ]  && ENVS=(GGML_CUDA_JETSON_GEMM_F32OUT=1 GGML_CUDA_JETSON_GEMM_BIAS=1)
    for S in 9x4 5x2 1x2; do
        N="vit_${CFG}_${S}"
        echo "[$(date +%T)] $N"
        env "${ENVS[@]}" "$NCU" --nvtx --nvtx-include "vit_$S/" --launch-skip 80 --launch-count 120 "${SECTIONS[@]}" \
            -o "$OUT/$N" -f "$REPO/build/bin/jetson-vit-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" "$M/mmproj-BF16.gguf" \
            "$FRAME" "$S" 2 1 off > "$OUT/$N.log" 2>&1
        grep -E "==ERROR==|No kernels were profiled" "$OUT/$N.log" && echo "   !! see $OUT/$N.log"
        export_csv "$N"
    done
done

# B. prefill down projection: heuristic (split-K) vs tuned algorithm
for RANGE in heuristic tuned; do
    N="gemm_down_${RANGE}"
    echo "[$(date +%T)] $N"
    "$NCU" --nvtx --nvtx-include "$RANGE/" "${SECTIONS[@]}" -o "$OUT/$N" -f \
        "$REPO/build/bin/jetson-gemm-tune" --profile 3584 512 18944 0 0 "$REPO/jetson/gemm/table_orin.txt" > "$OUT/$N.log" 2>&1
    grep -E "==ERROR==|No kernels were profiled" "$OUT/$N.log" && echo "   !! see $OUT/$N.log"
    export_csv "$N"
done

chown -R "${SUDO_USER:-$(id -un)}:" "$OUT"
echo "[$(date +%T)] done; compare with: python3 jetson/tools/ncu_compare.py $OUT"
