#!/usr/bin/env bash
# bench_serving.sh — edge-serving benchmark for the vanilla VLM.
#
# WHY THIS EXISTS, alongside bench_vanilla.sh:
#
#   bench_vanilla.sh forks a fresh process per configuration. That is the right
#   shape for COMPARING configurations — process isolation stops allocator state,
#   KV residue and CUDA context drift leaking from one config into the next.
#   But it makes every request a cold start, and no deployed system behaves that
#   way.
#
#   bench_serving.sh models the deployment instead: ONE loaded model handling a
#   queue of requests back-to-back, exactly as an edge box serving a video feed
#   does. Clocks stay ramped, CUDA graphs stay captured, the allocator settles,
#   the page cache is hot.
#
#   Neither replaces the other. Use bench_vanilla.sh to answer "is variant A
#   faster than variant B"; use this to answer "what latency does this system
#   actually deliver in service, and does it hold up".
#
# EVERY request is recorded, including the first. The cold-to-steady ramp is a
# result, not noise: you want to know how many requests it takes to reach steady
# state, and whether latency then DRIFTS — thermal throttling, allocator growth,
# fragmentation — over a sustained run. That drift is invisible to a per-process
# benchmark and is exactly what bites an edge deployment.
#
# Usage:
#   bash bench_serving.sh                        # default queue x 4 rounds
#   bash bench_serving.sh --rounds 10            # longer soak, for thermal drift
#   bash bench_serving.sh --videos "alley_1,cave_2" --frames 8
#   bash bench_serving.sh --fa off               # the pruning variants' regime
set -uo pipefail

ROOT="/data/spring26/CPP"
BIN="$ROOT/llama.cpp/build/bin/llama-baseline-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL="$ROOT/modelsRepo/MPI_bkp/MPI-Sintel"
OUTDIR="$ROOT/llama.cpp/dmc955/serving_results"

VIDEOS="alley_1,bamboo_1,cave_2,sleeping_1,market_5"
ROUNDS=4
FRAMES=8            # fixed frame budget; ignored when SAMPLE_FPS > 0
# Constant-rate sampling (Qwen's policy). SAMPLE_FPS=0 keeps the fixed budget.
# SOURCE_FPS is the assumed capture rate of the frames on disk; lowering it below
# Sintel's native 24 reinterprets each clip as longer, which is how a short
# dataset can be made to exercise the duration-dependent path.
SAMPLE_FPS=0
SOURCE_FPS=24
NPREDICT=128
NGL=99
NCTX=16384
SEED=42
FA="on"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --videos)  VIDEOS="$2"; shift 2 ;;
        --rounds)  ROUNDS="$2"; shift 2 ;;
        --frames)  FRAMES="$2"; shift 2 ;;
        --fps)         SAMPLE_FPS="$2"; shift 2 ;;
        --source-fps)  SOURCE_FPS="$2"; shift 2 ;;
        --fa)      FA="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -x "$BIN" ]] || { echo "!! Missing $BIN — bash profile.sh build"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "!! nvidia-smi broken — bash fix_gpu.sh"; exit 1; }
mkdir -p "$OUTDIR"

# Encode the weights and the sampling policy in the name so runs from different
# configurations can never overwrite or be mistaken for each other.
if [[ "$SAMPLE_FPS" != "0" ]]; then
    SAMPLE_DESC="fps${SAMPLE_FPS}@src${SOURCE_FPS}"
    FPS_ARGS=(--fps "$SAMPLE_FPS" --source-fps "$SOURCE_FPS")
else
    SAMPLE_DESC="f${FRAMES}"
    FPS_ARGS=()
fi
# Fixed-length generation. --ignore-eos biases every EOG token to -INFINITY inside
# common_init_from_params, so EOS can never be sampled and every request generates
# exactly n_predict tokens.
#
# This is REQUIRED for a comparable decode measurement on BF16 weights. 7B BF16 is
# numerically non-deterministic (CUDA reduction order flips near-tie argmax under an
# 8-bit mantissa), so generation LENGTH varied 9-128 tokens for identical input,
# making decode_ms and request_ms meaningless. With EOS suppressed, gen_ms sd falls
# to ~0.1%. The generated TEXT past the natural stopping point is degenerate — this
# is a latency measurement, not a quality one.
IGNORE_EOS="${IGNORE_EOS:-1}"
[[ "$IGNORE_EOS" == "1" ]] && EOS_ARGS=(--ignore-eos) || EOS_ARGS=()

STEM="$OUTDIR/serve_${MODEL_TAG}_fa${FA}_${SAMPLE_DESC}_r${ROUNDS}"

{
    echo "date:      $(date -Is)"
    echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "driver:    $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "git:       $(git -C "$ROOT/llama.cpp" rev-parse --short HEAD 2>/dev/null)"
    echo "videos:    $VIDEOS"
    echo "model:     $MODEL"
    echo "mmproj:    $MMPROJ"
    echo "rounds:    $ROUNDS   n_predict: $NPREDICT   fa: $FA"
    if [[ "$SAMPLE_FPS" != "0" ]]; then
        echo "sampling:  constant-rate ${SAMPLE_FPS} fps, source assumed ${SOURCE_FPS} fps"
    else
        echo "sampling:  fixed budget of $FRAMES frames"
    fi
} > "$STEM.env.txt"
cat "$STEM.env.txt"
echo

# Sample the GPU while the queue runs. Thermal throttling and clock decay are
# the failure modes a sustained edge workload actually hits, and neither shows
# up in a per-request timer.
#
# ONE streaming nvidia-smi, not a bash loop respawning it 4x/second. The loop
# version silently froze the SM-clock column at its first (idle) reading while
# every other column tracked correctly — 23/23 samples reporting 210 MHz during
# a run drawing 350 W. A measurement tool that lies quietly is worse than no
# tool, so the whole per-iteration subshell/process-substitution/printf chain is
# gone. --loop-ms also keeps NVML initialised between samples and costs far less.
nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits --loop-ms=250 > "$STEM.gpu.raw" 2>/dev/null &
POLLER=$!

echo "== Serving $(( $(tr -cd ',' <<<"$VIDEOS" | wc -c) + 1 )) videos x $ROUNDS rounds from one loaded model ..."
"$BIN" -m "$MODEL" --mmproj "$MMPROJ" --sintel-dir "$SINTEL" \
    --serve-videos "$VIDEOS" --serve-rounds "$ROUNDS" --base-frames "$FRAMES" \
    "${FPS_ARGS[@]}" "${EOS_ARGS[@]}" \
    --output-json "$STEM.json" \
    -ngl "$NGL" -c "$NCTX" -n "$NPREDICT" --temp 0 --seed "$SEED" -fa "$FA" \
    --no-flow-analysis > "$STEM.log" 2>&1
RC=$?

kill "$POLLER" 2>/dev/null; wait "$POLLER" 2>/dev/null

# Add the header and a synthesised time column (rows are one --loop-ms apart).
awk -v dt=0.25 'BEGIN { print "t_s,sm_mhz,temp_c,power_w,util_pct,mem_mib" }
     NF { gsub(/, /, ","); printf "%.2f,%s\n", (NR-1)*dt, $0 }' \
    "$STEM.gpu.raw" > "$STEM.gpu.csv"
rm -f "$STEM.gpu.raw"

if [[ $RC -ne 0 ]]; then
    echo "!! FAILED (rc=$RC) — see $STEM.log"; tail -5 "$STEM.log" | sed 's/^/   /'; exit $RC
fi

# Pass the stem: the summariser needs the .log too, to split ViT from prefill.
python3 "$ROOT/llama.cpp/dmc955/bench_serving_summarize.py" "$STEM" "$STEM.gpu.csv"
