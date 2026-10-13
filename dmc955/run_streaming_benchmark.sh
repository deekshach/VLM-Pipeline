#!/bin/bash
# Streaming VLM Benchmark Orchestration Script
#
# Runs baseline and optimized modes across multiple Sintel videos,
# then compares outputs with the benchmark analysis script.
#
# Usage:
#   ./run_streaming_benchmark.sh
#   ./run_streaming_benchmark.sh --skip-baseline         # skip phase 1
#   ./run_streaming_benchmark.sh --skip-optimized        # skip phase 2
#   ./run_streaming_benchmark.sh --skip-pruning          # skip phase 3
#   ./run_streaming_benchmark.sh --skip-pre-llm          # skip phase 4
#   ./run_streaming_benchmark.sh --skip-pre-llm-pruning  # skip phase 5

set -euo pipefail

# === Configuration ===
BINARY="./build/bin/llama-streaming-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
RESULTS_DIR="./streaming_results"

VIDEOS=(alley_1 bamboo_1 sleeping_1 mountain_1 shaman_2 bandage_1 cave_2 temple_2 ambush_2 market_5)

# Common args
# Context must be large enough for 16 baseline frames: 16 * 592 tokens + text ≈ 10000
COMMON_ARGS="-n 150 -ngl 99 -c 16384 --fps 4.0"

# Parse flags
SKIP_BASELINE=false
SKIP_OPTIMIZED=false
SKIP_PRUNING=false
SKIP_PRE_LLM=false
SKIP_PRE_LLM_PRUNING=false

for arg in "$@"; do
    case "$arg" in
        --skip-baseline)         SKIP_BASELINE=true ;;
        --skip-optimized)        SKIP_OPTIMIZED=true ;;
        --skip-pruning)          SKIP_PRUNING=true ;;
        --skip-pre-llm)          SKIP_PRE_LLM=true ;;
        --skip-pre-llm-pruning)  SKIP_PRE_LLM_PRUNING=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build with: cmake --build build --target llama-streaming-vlm"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Streaming VLM Benchmark ==="
echo "Videos: ${VIDEOS[*]}"
echo "Results: $RESULTS_DIR"
echo ""

# === Phase 1: Baseline ===
if [ "$SKIP_BASELINE" = false ]; then
    echo "=== Phase 1: Baseline (uniform selection, no cache, no pruning) ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/streaming_baseline_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $video (exists: $OUTPUT)"
            continue
        fi
        echo "  [RUN] $video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --baseline \
            --baseline-uniform-count 16 \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_baseline_${video}.txt"
        echo ""
    done
    echo ""
fi

# === Phase 2: Optimized (embedding cache, no pruning) ===
if [ "$SKIP_OPTIMIZED" = false ]; then
    echo "=== Phase 2: Optimized (online selection + embedding cache) ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/streaming_optimized_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $video (exists: $OUTPUT)"
            continue
        fi
        echo "  [RUN] $video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --cumulative-threshold 15.0 \
            --max-selected-frames 10 \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_optimized_${video}.txt"
        echo ""
    done
    echo ""
fi

# === Phase 3: Optimized + Pruning ===
if [ "$SKIP_PRUNING" = false ]; then
    echo "=== Phase 3: Optimized + Pruning (online selection + cache + LLM pruning) ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/streaming_optimized_pruning_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $video (exists: $OUTPUT)"
            continue
        fi
        echo "  [RUN] $video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --cumulative-threshold 15.0 \
            --max-selected-frames 10 \
            --enable-pruning \
            --enable-prefill-pruning \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_optimized_pruning_${video}.txt"
        echo ""
    done
    echo ""
fi

# === Phase 4: Optimized + Pre-LLM Pruning (no decode-stage pruning) ===
if [ "$SKIP_PRE_LLM" = false ]; then
    echo "=== Phase 4: Optimized + Pre-LLM Pruning ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/streaming_optimized_prellm_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $video (exists: $OUTPUT)"
            continue
        fi
        echo "  [RUN] $video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --cumulative-threshold 15.0 \
            --max-selected-frames 10 \
            --enable-pre-llm-pruning \
            --pre-llm-keep-ratio 0.5 \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_optimized_prellm_${video}.txt"
        echo ""
    done
    echo ""
fi

# === Phase 5: Optimized + Pre-LLM Pruning + Decode-Stage Pruning ===
if [ "$SKIP_PRE_LLM_PRUNING" = false ]; then
    echo "=== Phase 5: Optimized + Pre-LLM + Decode Pruning ==="
    for video in "${VIDEOS[@]}"; do
        OUTPUT="$RESULTS_DIR/streaming_optimized_prellm_pruning_${video}.json"
        if [ -f "$OUTPUT" ]; then
            echo "  [SKIP] $video (exists: $OUTPUT)"
            continue
        fi
        echo "  [RUN] $video -> $OUTPUT"
        $BINARY -m "$MODEL" --mmproj "$MMPROJ" \
            --video "$video" \
            --cumulative-threshold 15.0 \
            --max-selected-frames 10 \
            --enable-pre-llm-pruning \
            --pre-llm-keep-ratio 0.5 \
            --enable-pruning \
            --output-json "$OUTPUT" \
            $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/log_optimized_prellm_pruning_${video}.txt"
        echo ""
    done
    echo ""
fi

# === Phase 6: Analysis ===
echo "=== Phase 6: Benchmark Analysis ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/benchmark_analysis.py" "$RESULTS_DIR" --output streaming_benchmark_summary.json

echo ""
echo "=== Benchmark Complete ==="
echo "Results: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/streaming_benchmark_summary.json"
