#!/bin/bash
# Prefill-Stage Motion Pruning Experiment
# Compares: baseline vs prefill-only vs combined (prefill + decode) pruning
# Reports TTFT (encode_ms), total latency (total_ms), and quality (shannon_entropy)

set -e

BUILD_DIR="${BUILD_DIR:-../build}"
BINARY="${BUILD_DIR}/bin/llama-adaptive-patch-cache-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL_DIR="/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel"
OUTPUT_DIR="./prefill_pruning_results"
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
            grep -E "Generated|Pruning|Prefill|Shannon entropy" | tr '\n' ' '
        echo ""
    done
    echo ""
}

echo "=== Prefill Pruning Experiment ==="
echo "Videos: ${VIDEOS[*]}"
echo ""

# 1. Baseline: no pruning at all
run_config "baseline" "" "Config 1: Baseline (no pruning)"

# 2. Prefill-only: pruning between frame decodes, keep_ratio=0.6
run_config "prefill_0.6" "--enable-prefill-pruning --prefill-keep-ratio 0.6" \
    "Config 2: Prefill-only pruning (keep=0.6)"

# 3. Prefill-only with higher keep ratio
run_config "prefill_0.8" "--enable-prefill-pruning --prefill-keep-ratio 0.8" \
    "Config 3: Prefill-only pruning (keep=0.8)"

# 4. Combined: prefill + decode pruning (moderate decode settings)
run_config "combined_0.6" "--enable-prefill-pruning --prefill-keep-ratio 0.6 --enable-pruning --patch-keep-ratio 0.5 --token-keep-ratio 0.7 --motion-weight 0.3 --prune-layer 2" \
    "Config 4: Combined prefill(0.6) + decode pruning"

# 5. Combined with conservative prefill
run_config "combined_0.8" "--enable-prefill-pruning --prefill-keep-ratio 0.8 --enable-pruning --patch-keep-ratio 0.5 --token-keep-ratio 0.7 --motion-weight 0.3 --prune-layer 2" \
    "Config 5: Combined prefill(0.8) + decode pruning"

echo "=== All experiments complete ==="
echo "Results in: ${OUTPUT_DIR}"
