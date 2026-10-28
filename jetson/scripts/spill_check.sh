#!/bin/bash
# Register spills of the GEMM kernels the design uses, stock (cuBLASLt heuristic) against tuned (GEMM table): one
# Nsight Compute pass per shape over jetson-gemm-tune --profile, which runs the shape once with each algorithm in the
# NVTX ranges "heuristic" and "tuned". Needs sudo (GPU counters):
#
#   sudo bash jetson/scripts/spill_check.sh [table=jetson/gemm/table_orin.txt]
#
# Shapes (m n k d_f32 bias, as jetson-gemm.cu calls them): the seven LLM projections at a 512-token micro-batch and
# the ViT projections of a full Sintel frame (2,304 patches), each with the output type and bias the design runs.
# Per kernel: registers per thread and local-memory (spill) traffic, l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum;
# zero local bytes means no spills. Register allocation is fixed at compile time, so the power mode does not matter.
# Out: jetson/experiments/profiles/spills/{spills_raw.csv,spills.md}.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCU="${NCU:-/usr/local/cuda-12.6/bin/ncu}"
TABLE="${1:-$REPO/jetson/gemm/table_orin.txt}"
OUT="$REPO/jetson/experiments/profiles/spills"
USR="${SUDO_USER:-$(id -un)}"
mkdir -p "$OUT"
SHAPES=("3584 512 3584 1 1:Q (LLM)" "512 512 3584 1 1:K, V (LLM)" "3584 512 3584 1 0:O (LLM)"
        "18944 512 3584 1 0:gate, up (LLM)" "3584 512 18944 1 0:down (LLM)"
        "1280 2304 1280 1 1:Q, K, V, O (ViT)" "3420 2304 1280 1 1:gate, up (ViT)" "1280 2304 3420 1 1:down (ViT)")
METRICS=launch__registers_per_thread,launch__grid_size,launch__block_size,launch__shared_mem_per_block,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
: > "$OUT/spills_raw.csv"
for s in "${SHAPES[@]}"; do
    shape="${s%%:*}"; role="${s#*:}"
    echo "[$(date +%T)] $role: $shape"
    # shellcheck disable=SC2086
    "$NCU" --nvtx --metrics "$METRICS" --csv --print-units base \
        "$REPO/build/bin/jetson-gemm-tune" --profile $shape "$TABLE" 2> /dev/null |
        awk -v role="$role" -v shape="$shape" '/^"ID"/ { next } /^"/ { print "\"" role "\",\"" shape "\"," $0 }' \
        >> "$OUT/spills_raw.csv"
done
python3 - "$OUT/spills_raw.csv" > "$OUT/spills.md" <<'EOF'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
k = {}
for r in rows:
    if len(r) < 4: continue
    role, shape, metric, val = r[0], r[1], r[-3], r[-1]
    name = next((x for x in r if "gemm" in x.lower() or "Kernel2" in x or "xmma" in x.lower()), "?")
    rng = "tuned" if any(x.endswith("tuned") for x in r[2:]) else "heuristic" if any(x.endswith("heuristic") for x in r[2:]) else "?"
    k.setdefault((role, shape, rng, name), {})[metric] = val
print("| GEMM | shape (m n k f32 bias) | algorithm | kernel | registers | local load B | local store B |")
print("|---|---|---|---|---|---|---|")
for (role, shape, rng, name), m in k.items():
    print(f"| {role} | {shape} | {rng} | {name[:70]} | {m.get('launch__registers_per_thread', '')} | "
          f"{m.get('l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum', '')} | {m.get('l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum', '')} |")
EOF
chown -R "$USR:" "$OUT"
cat "$OUT/spills.md"
