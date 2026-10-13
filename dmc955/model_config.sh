# model_config.sh — THE single definition of which weights every harness uses.
#
# Source this; do not hardcode model paths anywhere else.
#     source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"
#
# WHY THIS FILE EXISTS
#   The scripts here had drifted into two camps: some on 7B BF16, some on 3B q8_0.
#   That is not a cosmetic inconsistency. Decode is bandwidth-bound — every token
#   drags the full weight set across the bus — so quantization moves the dominant
#   phase by roughly 2x. Any number compared across the two camps is meaningless.
#
#   Pinned to 7B BF16: unquantized weights keep the baseline faithful to the
#   published model, and quantization is then a variable you can study rather than
#   an uncontrolled difference baked into the harness.
#
# Override for a one-off run without editing this file:
#     MODEL=/path/to/other.gguf bash bench_serving.sh

MODELS_DIR="${MODELS_DIR:-/data/spring26/CPP/modelsRepo/Qwen2.5VL}"

MODEL="${MODEL:-$MODELS_DIR/Qwen2.5-VL-7B-Instruct-BF16.gguf}"
MMPROJ="${MMPROJ:-$MODELS_DIR/mmproj-BF16.gguf}"

# Short tag for output filenames, so results from different weights never collide.
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL" .gguf | sed 's/Qwen2.5-VL-//; s/-Instruct//')}"

[[ -f "$MODEL"  ]] || { echo "!! model not found: $MODEL"   >&2; exit 1; }
[[ -f "$MMPROJ" ]] || { echo "!! mmproj not found: $MMPROJ" >&2; exit 1; }
