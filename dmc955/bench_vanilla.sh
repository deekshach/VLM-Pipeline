#!/usr/bin/env bash
# bench_vanilla.sh — performance benchmark for the VANILLA VLM (baseline_vlm.cpp).
#
# This is the *macro* counterpart to profile.sh:
#   bench_vanilla.sh : how fast is it, and how does that scale with visual tokens
#   profile.sh       : why is it that fast (nsys timeline / perf counters / ncu roofline)
#
# What it controls for (the things that make naive VLM timings useless):
#   - model load time  : excluded (we read the per-phase timers, not total_ms)
#   - flow analysis    : NOT RUN. --no-flow-analysis skips build_motion_profile,
#                        which is reporting-only (frames are picked by
#                        select_frames_uniform before it ever runs) and costs ~800 ms
#                        plus the largest single CPU hotspot in the binary.
#   - sampling noise   : greedy decode (--temp 0) + fixed seed => deterministic output,
#                        so token counts are identical run-to-run and t/s is comparable
#   - cold start       : --warmup-iters runs inference an extra time INSIDE the
#                        measured process (resetting KV/sampler/history so the work
#                        is identical), leaving GPU clocks ramped, the CUDA context
#                        hot and the graphs captured. This replaces the old separate
#                        warmup process, which barely helped — the GPU falls back to
#                        idle clocks while the next process loads 4.5 GB of GGUF.
#   - run-to-run drift : REPS repeats, summarizer reports median + spread
#   - CUDA graphs      : left ENABLED here (production-accurate). profile.sh disables
#                        them for kernel attribution — do not compare the two directly.
#   - flash attention  : swept as an explicit axis, see below.
#
# WHY FLASH ATTENTION IS AN AXIS AND NOT A SETTING
#   The vanilla baseline resolves -fa auto to ENABLED. But streaming_vlm.cpp:3930
#   force-disables flash attention whenever --enable-pruning (or correlation
#   analysis) is on, because flash attn fuses the softmax that the cb_eval capture
#   needs. Measured on alley_1 @ 8 frames: LLM prefill 238 ms -> 430 ms (+81%) with
#   flash off; ViT is unaffected. So comparing a flash-ON vanilla against a
#   flash-OFF pruning variant hands the optimization a ~190 ms handicap that has
#   nothing to do with pruning. Always compare like against like:
#       vs. --enable-pruning / correlation runs        -> use the fa=off rows
#       vs. postprefill-eviction-only or stock decode  -> use the fa=on rows
#
# Phase split comes free from mtmd-helper's own logs:
#   "image slice encoded in N ms"        -> ViT encode
#   "image decoded (batch i/n) in N ms"  -> LLM prefill of the visual tokens
# baseline_vlm.cpp's encode_ms lumps both together; the summarizer separates them.
#
# Usage:
#   bash bench_vanilla.sh                # full sweep: VIDEOS x FRAMES x FA x REPS
#   bash bench_vanilla.sh --quick        # 1 video, 8 frames, both fa modes, 1 rep
#   bash bench_vanilla.sh --reps 5
#   bash bench_vanilla.sh --videos "alley_1 cave_2" --frames "8 16"
#   bash bench_vanilla.sh --fa off       # only the flash-off regime
set -uo pipefail

ROOT="/data/spring26/CPP"
BIN="$ROOT/llama.cpp/build/bin/llama-baseline-vlm"
source "$(dirname "${BASH_SOURCE[0]}")/model_config.sh"   # single source of truth for weights
SINTEL="$ROOT/modelsRepo/MPI_bkp/MPI-Sintel"
OUTDIR="$ROOT/llama.cpp/dmc955/bench_results"

# --- sweep configuration ----------------------------------------------------
# FRAMES is the axis that matters for your research: visual token count scales
# ~592 tokens/frame, so this is the knob every optimization you have targets.
VIDEOS="alley_1 bamboo_1 cave_2 sleeping_1 market_5"
FRAMES="4 8 16"
FA_MODES="on off"   # LLM flash attention; see the note above before dropping one
REPS=3
WARMUP=1            # in-process warmup iterations before the measured one
NPREDICT=128
NGL=99
NCTX=16384          # 16 frames * ~592 tok + text  ~= 9.5k; 16k is safe headroom
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)  VIDEOS="alley_1"; FRAMES="8"; REPS=1; shift ;;
        --reps)   REPS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --videos) VIDEOS="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --fa)     FA_MODES="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

for fa in $FA_MODES; do
    [[ "$fa" == "on" || "$fa" == "off" || "$fa" == "auto" ]] || {
        echo "!! --fa takes on|off|auto (got '$fa')"; exit 1; }
done

# --- preflight --------------------------------------------------------------
[[ -x "$BIN"    ]] || { echo "!! Missing $BIN — run: bash profile.sh build"; exit 1; }
[[ -f "$MODEL"  ]] || { echo "!! Missing model: $MODEL";  exit 1; }
[[ -f "$MMPROJ" ]] || { echo "!! Missing mmproj: $MMPROJ"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "!! nvidia-smi broken — see fix_gpu.sh"; exit 1; }

mkdir -p "$OUTDIR"

# Record the environment alongside the numbers; a benchmark without provenance
# is not reproducible three weeks from now.
{
    echo "date:        $(date -Is)"
    echo "host:        $(hostname)"
    echo "kernel:      $(uname -r)"
    echo "driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "git_commit:  $(git -C "$ROOT/llama.cpp" rev-parse --short HEAD 2>/dev/null)"
    echo "git_dirty:   $(git -C "$ROOT/llama.cpp" status --porcelain 2>/dev/null | wc -l) files"
    echo "model:       $MODEL"
    echo "mmproj:      $MMPROJ"
    echo "n_predict:   $NPREDICT"
    echo "n_ctx:       $NCTX"
    echo "seed:        $SEED  (greedy, --temp 0)"
    echo "videos:      $VIDEOS"
    echo "frames:      $FRAMES"
    echo "fa_modes:    $FA_MODES"
    echo "reps:        $REPS"
    echo "warmup_iters: $WARMUP  (in-process)"
} > "$OUTDIR/env.txt"
echo "== Environment recorded in $OUTDIR/env.txt"
cat "$OUTDIR/env.txt"
echo

# --- one run ----------------------------------------------------------------
# $1 video  $2 frames  $3 fa (on|off|auto)  $4 tag (warmup|r0|r1..)
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

run_one() {
    local video="$1" nf="$2" fa="$3" tag="$4"
    local stem="$OUTDIR/${video}_f${nf}_fa${fa}_${tag}"

    # Sample VRAM at 100 ms while the run is live; peak goes in .vram.
    ( while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
        sleep 0.1
      done ) > "$stem.vram.raw" 2>/dev/null &
    local poller=$!

    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" \
        --sintel-dir "$SINTEL" --video "$video" --base-frames "$nf" \
        --output-json "$stem.json" --no-flow-analysis --warmup-iters "$WARMUP" \
        -ngl "$NGL" -c "$NCTX" -n "$NPREDICT" \
        --temp 0 --seed "$SEED" -fa "$fa" "${EOS_ARGS[@]}" \
        > "$stem.log" 2>&1
    local rc=$?

    kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null
    sort -n "$stem.vram.raw" 2>/dev/null | tail -1 > "$stem.vram"
    rm -f "$stem.vram.raw"

    if [[ $rc -ne 0 ]]; then
        echo "   !! FAILED (rc=$rc) — see $stem.log"
        tail -5 "$stem.log" | sed 's/^/      /'
    fi
    return $rc
}

# --- sweep ------------------------------------------------------------------
total=0; failed=0
for nf in $FRAMES; do
    for video in $VIDEOS; do
        for fa in $FA_MODES; do
            echo "== $video @ ${nf} frames, flash=$fa"
            # No separate warmup PROCESS: --warmup-iters warms up inside each run,
            # which is both more effective and cheaper than an extra model load.
            for ((r=0; r<REPS; r++)); do
                printf "   rep %d/%d ... " "$((r+1))" "$REPS"
                if run_one "$video" "$nf" "$fa" "r$r"; then
                    # quick inline feedback so a long sweep isn't a black box
                    python3 - "$OUTDIR/${video}_f${nf}_fa${fa}_r$r.json" <<'PY' 2>/dev/null || echo "ok"
import json,sys
j=json.load(open(sys.argv[1]))
t=j["timing"]; n=j["inference"]["n_tokens"]
print(f'encode+prefill {t["encode_ms"]:.0f} ms | gen {t["generation_ms"]:.0f} ms '
      f'| {n} tok | {n/(t["generation_ms"]/1000):.1f} tok/s')
PY
                else
                    failed=$((failed+1))
                fi
                total=$((total+1))
            done
        done
    done
done

echo
echo "== Sweep done: $total runs, $failed failed."
echo "== Summarizing ..."
python3 "$ROOT/llama.cpp/dmc955/bench_summarize.py" "$OUTDIR"
