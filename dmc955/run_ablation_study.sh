#!/bin/bash
# Pre-LLM Pruning Ablation Study
#
# Runs all 6 pruning strategies across all videos to compare:
#   1. combined   - tri-signal (motion 0.3 + norm 0.3 + attn 0.4)
#   2. random     - random token selection (deterministic seed)
#   3. motion-only - only motion signal
#   4. norm-only  - only embedding L2 norm signal
#   5. attn-only  - only ViT attention received signal
#   6. motion-norm - motion + norm (no ViT attn, avoids flash attn disable)
#
# Usage:
#   ./run_ablation_study.sh
#   ./run_ablation_study.sh --videos "alley_1 bamboo_1"  # subset

set -euo pipefail

BINARY="./build/bin/llama-streaming-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
RESULTS_DIR="./streaming_results/ablation"
COMMON_ARGS="-n 150 -ngl 99 -c 16384 --fps 4.0"

VIDEOS=(alley_1 bamboo_1 sleeping_1 mountain_1 shaman_2 bandage_1 cave_2 temple_2 ambush_2 market_5)
STRATEGIES=(combined random motion-only norm-only attn-only motion-norm)

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --videos)
            shift
            IFS=' ' read -r -a VIDEOS <<< "$1"
            shift
            ;;
    esac
done

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Pre-LLM Pruning Ablation Study ==="
echo "Videos: ${VIDEOS[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo "Results: $RESULTS_DIR"
echo ""

for strategy in "${STRATEGIES[@]}"; do
    echo "=== Strategy: $strategy ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/ablation_${strategy}_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $strategy/$video (exists)"
            continue
        fi
        echo "  [RUN] $strategy/$video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --cumulative-threshold 15.0 \
            --max-selected-frames 10 \
            --enable-pre-llm-pruning \
            --pre-llm-keep-ratio 0.5 \
            --pre-llm-strategy "$strategy" \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_${strategy}_${video}.txt"
        echo ""
    done
    echo ""
done

echo "=== Ablation Study Complete ==="
echo "Results in: $RESULTS_DIR"
