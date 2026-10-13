#!/usr/bin/env bash
# measure.sh -- one command per configuration, GPU-side stage attribution.
#
# WHY, alongside bench_serving.sh:
#
#   bench_serving.sh gives clean WALL-CLOCK latency with no profiler attached.
#   That is the right tool for "is A faster than B". But it splits ViT from
#   prefill by parsing mtmd's host-side log lines, which are wall-clock spans
#   around synchronous calls -- correct only while the phases are serial. Once
#   ViT and prefill overlap (the 578 ms optimisation), those spans overlap too
#   and the split silently becomes wrong.
#
#   This runs the same workload under nsys and attributes stages from GPU
#   activity via stage_attrib.py, which stays correct under overlap and reports
#   the overlap itself. Use it for STAGE questions; use bench_serving.sh for
#   wall-clock. nsys adds tracing overhead -- never quote wall time from here.
#
# Usage:
#   bash measure.sh <label> [-- <extra llama-baseline-vlm args>]
#
#   bash measure.sh baseline
#   bash measure.sh ub640     -- -ub 640
#   bash measure.sh f24       -- --base-frames 24
#   FRAMES=24 NCTX=32768 bash measure.sh f24
#
# Appends one row per run to results/experiments.csv, tagged with the git SHA.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./model_config.sh

NSYS="${NSYS:-/usr/local/cuda-12.6/bin/nsys}"
BIN="${BIN:-../build/bin/llama-baseline-vlm}"
SINTEL="${SINTEL:-/data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel}"
OUTDIR="${OUTDIR:-results}"
CSV="${CSV:-$OUTDIR/experiments.csv}"

VIDEO="${VIDEO:-alley_1}"
ROUNDS="${ROUNDS:-3}"        # round 0 is warmup; stage_attrib reports each request
FRAMES="${FRAMES:-13}"
NPREDICT="${NPREDICT:-128}"
NCTX="${NCTX:-16384}"
NGL="${NGL:-99}"
FA="${FA:-on}"

LABEL="${1:?usage: measure.sh <label> [-- extra args]}"; shift || true
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

mkdir -p "$OUTDIR"
REP="$OUTDIR/trace_$LABEL"
SHA="$(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "== measure: $LABEL  (video=$VIDEO frames=$FRAMES rounds=$ROUNDS n=$NPREDICT git=$SHA)"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "   extra args: ${EXTRA[*]}"

# --ignore-eos is REQUIRED: BF16 is numerically non-deterministic and generation
# length otherwise varies 9-128 tokens, making decode_ms meaningless.
"$NSYS" profile -t cuda,osrt --cuda-graph-trace=node --gpu-metrics-device=0 \
    -o "$REP" --force-overwrite=true --stats=false \
    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --sintel-dir "$SINTEL" \
    --serve-videos "$VIDEO" --serve-rounds "$ROUNDS" --base-frames "$FRAMES" \
    -n "$NPREDICT" -ngl "$NGL" -c "$NCTX" --temp 0 --seed 42 -fa "$FA" \
    --no-flow-analysis --ignore-eos --warmup-iters 0 \
    "${EXTRA[@]}" > "$REP.runlog" 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "!! run failed (rc=$RC) -- see $REP.runlog"; tail -5 "$REP.runlog" | sed 's/^/   /'; exit $RC
fi

grep -E "Request [0-9]+ done|Serving run complete" "$REP.runlog" | sed 's/^/   /'

python3 stage_attrib.py "$REP.nsys-rep" --csv "$CSV" --label "$LABEL" \
        --json "$REP.stages.json"

# The .sqlite export is ~265 MB per run (GPU_METRICS alone is millions of rows),
# so a 10-point sweep would cost ~2.6 GB. The .nsys-rep is only ~10 MB and
# stage_attrib.py re-exports on demand, so drop the sqlite unless asked to keep it.
if [[ "${KEEP_SQLITE:-0}" != "1" ]]; then
    rm -f "$REP.sqlite"
fi
echo "   trace: $REP.nsys-rep  (KEEP_SQLITE=1 to retain the .sqlite export)"
