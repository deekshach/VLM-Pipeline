#!/bin/bash
# GPU Defrag vs CPU Defrag comparison experiment
# Runs baseline + 3 pruning configs (aggressive, moderate, conservative) on 6 videos
# Uses the newly built binary with GPU-native defrag

set -e

BUILD_DIR="${BUILD_DIR:-../build}"
BINARY="${BUILD_DIR}/bin/llama-adaptive-patch-cache-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL_DIR="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
OUTPUT_DIR="./pruning_results_gpu_defrag"
N_PREDICT=150
NGL=99
CTX=8192

VIDEOS=("alley_1" "bamboo_1" "sleeping_1" "mountain_1" "shaman_2" "bandage_1")

COMMON_ARGS="-m ${MODEL} --mmproj ${MMPROJ} --sintel-dir ${SINTEL_DIR} -n ${N_PREDICT} -ngl ${NGL} -c ${CTX}"

mkdir -p "${OUTPUT_DIR}"

run_config() {
    local tag="$1"
    local extra_args="$2"
    local desc="$3"
    echo "=== ${desc} ==="
    for video in "${VIDEOS[@]}"; do
        outfile="${OUTPUT_DIR}/${tag}_${video}.json"
        if [ -f "$outfile" ]; then
            echo "  [SKIP] ${video}"
            continue
        fi
        echo -n "  ${video}: "
        ${BINARY} ${COMMON_ARGS} --video "${video}" ${extra_args} --output-json "${outfile}" 2>&1 | \
            grep -E "Generated|Pruning summary|Shannon entropy" | tr '\n' ' '
        echo ""
    done
    echo ""
}

echo "=== GPU Defrag Experiment ==="
echo "Videos: ${VIDEOS[*]}"
echo ""

# Baseline: no pruning (same as before, for fresh timing comparison)
run_config "baseline" "" "Phase 1: Baseline (no pruning)"

# Aggressive: 85% reduction
run_config "aggressive" "--enable-pruning --patch-keep-ratio 0.3 --token-keep-ratio 0.5 --motion-weight 0.3 --prune-layer 2" \
    "Phase 2: Aggressive pruning (pkr=0.3, tkr=0.5)"

# Moderate: ~65% reduction
run_config "moderate" "--enable-pruning --patch-keep-ratio 0.5 --token-keep-ratio 0.7 --motion-weight 0.3 --prune-layer 2" \
    "Phase 3: Moderate pruning (pkr=0.5, tkr=0.7)"

# Conservative: ~37% reduction
run_config "conservative" "--enable-pruning --patch-keep-ratio 0.7 --token-keep-ratio 0.9 --motion-weight 0.3 --prune-layer 2" \
    "Phase 4: Conservative pruning (pkr=0.7, tkr=0.9)"

echo "=== All experiments complete ==="
echo "Results in: ${OUTPUT_DIR}"
