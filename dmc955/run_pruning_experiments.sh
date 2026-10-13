#!/bin/bash
# Progressive Visual Token Pruning — Parameter Sweep Experiments
#
# Sweeps key pruning parameters across representative Sintel videos
# and compares against baseline (no pruning).

set -e

# ---- Configuration ----
BUILD_DIR="${BUILD_DIR:-../build}"
BINARY="${BUILD_DIR}/bin/llama-adaptive-patch-cache-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL_DIR="${SINTEL_DIR:-/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel}"
OUTPUT_DIR="${OUTPUT_DIR:-./pruning_results}"
N_PREDICT="${N_PREDICT:-256}"
NGL="${NGL:-99}"
CTX="${CTX:-8192}"

# Representative videos (mix of low/medium/high motion)
VIDEOS=("alley_1" "ambush_2" "bamboo_1" "cave_2" "market_5" "temple_2")

# ---- Parameter grids ----
PATCH_KEEP_RATIOS=(0.3 0.5 0.7 1.0)
TOKEN_KEEP_RATIOS=(0.5 0.7 0.9 1.0)
PRUNE_LAYERS=(2 8 14 20)
MOTION_WEIGHTS=(0.0 0.3 1.0)

mkdir -p "${OUTPUT_DIR}"

# Common args
COMMON_ARGS="-m ${MODEL} --mmproj ${MMPROJ} --sintel-dir ${SINTEL_DIR} -n ${N_PREDICT} -ngl ${NGL} -c ${CTX}"

echo "=== Progressive Visual Token Pruning Experiments ==="
echo "Build dir: ${BUILD_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Videos: ${VIDEOS[*]}"
echo ""

# ---- 1. Baseline runs (no pruning) ----
echo "=== Phase 1: Baseline runs (no pruning) ==="
for video in "${VIDEOS[@]}"; do
    outfile="${OUTPUT_DIR}/baseline_${video}.json"
    if [ -f "$outfile" ]; then
        echo "  [SKIP] Baseline for ${video} already exists"
        continue
    fi
    echo "  Running baseline for ${video}..."
    ${BINARY} ${COMMON_ARGS} --video "${video}" --output-json "${outfile}" 2>&1 | tail -20
    echo ""
done

# ---- 2. Patch keep ratio sweep ----
echo "=== Phase 2: Patch keep ratio sweep (motion_weight=0.3, token_keep=0.7) ==="
for video in "${VIDEOS[@]}"; do
    for pkr in "${PATCH_KEEP_RATIOS[@]}"; do
        outfile="${OUTPUT_DIR}/pkr_${pkr}_${video}.json"
        if [ -f "$outfile" ]; then
            echo "  [SKIP] pkr=${pkr} for ${video} already exists"
            continue
        fi
        echo "  Running pkr=${pkr} for ${video}..."
        ${BINARY} ${COMMON_ARGS} --video "${video}" \
            --enable-pruning \
            --patch-keep-ratio "${pkr}" \
            --token-keep-ratio 0.7 \
            --motion-weight 0.3 \
            --prune-layer 2 \
            --output-json "${outfile}" 2>&1 | tail -5
        echo ""
    done
done

# ---- 3. Token keep ratio sweep ----
echo "=== Phase 3: Token keep ratio sweep (patch_keep=0.5, motion_weight=0.3) ==="
for video in "${VIDEOS[@]}"; do
    for tkr in "${TOKEN_KEEP_RATIOS[@]}"; do
        outfile="${OUTPUT_DIR}/tkr_${tkr}_${video}.json"
        if [ -f "$outfile" ]; then
            echo "  [SKIP] tkr=${tkr} for ${video} already exists"
            continue
        fi
        echo "  Running tkr=${tkr} for ${video}..."
        ${BINARY} ${COMMON_ARGS} --video "${video}" \
            --enable-pruning \
            --patch-keep-ratio 0.5 \
            --token-keep-ratio "${tkr}" \
            --motion-weight 0.3 \
            --prune-layer 2 \
            --output-json "${outfile}" 2>&1 | tail -5
        echo ""
    done
done

# ---- 4. Prune layer sweep ----
echo "=== Phase 4: Prune layer sweep (pkr=0.5, tkr=0.7, motion_weight=0.3) ==="
for video in "${VIDEOS[@]}"; do
    for layer in "${PRUNE_LAYERS[@]}"; do
        outfile="${OUTPUT_DIR}/layer_${layer}_${video}.json"
        if [ -f "$outfile" ]; then
            echo "  [SKIP] layer=${layer} for ${video} already exists"
            continue
        fi
        echo "  Running layer=${layer} for ${video}..."
        ${BINARY} ${COMMON_ARGS} --video "${video}" \
            --enable-pruning \
            --patch-keep-ratio 0.5 \
            --token-keep-ratio 0.7 \
            --motion-weight 0.3 \
            --prune-layer "${layer}" \
            --output-json "${outfile}" 2>&1 | tail -5
        echo ""
    done
done

# ---- 5. Motion weight sweep ----
echo "=== Phase 5: Motion weight sweep (pkr=0.5, tkr=0.7) ==="
for video in "${VIDEOS[@]}"; do
    for mw in "${MOTION_WEIGHTS[@]}"; do
        outfile="${OUTPUT_DIR}/mw_${mw}_${video}.json"
        if [ -f "$outfile" ]; then
            echo "  [SKIP] mw=${mw} for ${video} already exists"
            continue
        fi
        echo "  Running motion_weight=${mw} for ${video}..."
        ${BINARY} ${COMMON_ARGS} --video "${video}" \
            --enable-pruning \
            --patch-keep-ratio 0.5 \
            --token-keep-ratio 0.7 \
            --motion-weight "${mw}" \
            --prune-layer 2 \
            --output-json "${outfile}" 2>&1 | tail -5
        echo ""
    done
done

echo "=== All experiments complete ==="
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "To analyze results, parse the JSON files. Key metrics:"
echo "  - pruning.total_reduction_pct: % of visual tokens removed"
echo "  - pruning.pruning_overhead_ms: time spent on pruning"
echo "  - inference.generation_time_ms: total generation time"
echo "  - metrics.shannon_entropy: output quality proxy"
