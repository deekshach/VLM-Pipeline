#!/usr/bin/env bash
# Nsight Compute on ~3 ViT layers of one encode per shape: full frame (9x4 windows)
# vs window crops (5x2, 1x2), with flash attention on and off. Run as: sudo bash run_ncu.sh
# Kernel replay is fine here: only the vision tower is on the GPU (~1.5-2 GB to back up).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU=/usr/local/cuda-12.6/bin/ncu
M=/home/nvidia/data/models/Qwen2.5VL
FRAME=/home/nvidia/data/MPI-Sintel/training/clean/alley_1/frame_0001.png
for FA in off on; do
    for S in 9x4 5x2 1x2; do
        T="${S}_fa${FA}"
        echo "[$(date +%T)] ncu vit_$S, flash attention $FA"
        "$NCU" --nvtx --nvtx-include "vit_$S/" --launch-skip 80 --launch-count 120 \
            --section SpeedOfLight --section LaunchStats --section Occupancy \
            --section MemoryWorkloadAnalysis --section WarpStateStats \
            -o "$HERE/ncu_$T" -f \
            "$HERE/../../../build/bin/jetson-vit-probe" "$M/Qwen2.5-VL-7B-Instruct-BF16.gguf" "$M/mmproj-BF16.gguf" "$FRAME" "$S" 2 1 "$FA" \
            > "$HERE/ncu_$T.log" 2>&1
        grep -E "==ERROR==|No kernels were profiled" "$HERE/ncu_$T.log" && echo "   !! see ncu_$T.log"
        "$NCU" --import "$HERE/ncu_$T.ncu-rep" --page raw --csv > "$HERE/ncu_${T}_raw.csv" 2>/dev/null
        echo "   $(($(wc -l < "$HERE/ncu_${T}_raw.csv") - 2)) kernels profiled"
    done
done
chown -R "${SUDO_USER:-nvidia}:" "$HERE"
echo "[$(date +%T)] done"
