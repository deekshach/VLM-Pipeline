#!/usr/bin/env bash
# run_ncu_survey.sh — per-stage ncu sampling, built from the nsys (kernel,grid) census.
#
# METHOD (see PROFILING_HANDOFF.md section 6):
#   1. nsys `cuda_gpu_kern_gb_sum` gives time share per (kernel, grid) per stage.
#   2. Targets are chosen by walking that list down to a coverage goal — NOT by
#      guessing which kernel "represents" a stage.
#   3. ncu cannot filter by grid. Where a name covers several grids we either
#      accept them (if durations are homogeneous) or use -s to land on one.
#   4. Stream filtering (--kernel-id ":17::") separates ViT from LLM exactly.
#      Prefill vs decode is separated by name (verified phase-exclusive).
#   5. Attention in prefill DRIFTS +229% across the phase (KV grows per frame),
#      so it is sampled as a CURVE at four launch depths, never as one number.
#
# Output: one .ncu-rep + .log per target, named <stage>__<kernel>__<selector>.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./model_config.sh

NCU=/usr/local/cuda-12.6/bin/ncu
OUT="ncu_$(date +%Y-%m-%d)_${MODEL_TAG}"
mkdir -p "$OUT"

# Workload: 13 frames matches the 460-request characterization operating point.
# -n 16 is enough decode to sample every mul_mat_vec_f shape including lm_head.
WL=(-m "$MODEL" --mmproj "$MMPROJ"
    --sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel
    --video alley_1 --base-frames 13
    -ngl 99 -c 16384 -n 16 --temp 0 --seed 42 -fa on
    --no-flow-analysis --ignore-eos --warmup-iters 0)

# $1 output stem  $2 ncu selector args (array name)  $3 human note
run() {
    local name="$1"; shift
    local -a sel=("$@")
    [[ -f "$OUT/$name.ncu-rep" ]] && { echo "  SKIP $name (exists)"; return 0; }
    printf "  %-46s" "$name"
    local t0=$SECONDS
    "$NCU" --set full --graph-profiling node --kernel-name-base demangled \
        "${sel[@]}" --target-processes all \
        -o "$OUT/$name" --force-overwrite \
        ../build/bin/llama-baseline-vlm "${WL[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 && -f "$OUT/$name.ncu-rep" ]]; then
        echo " ok ($((SECONDS-t0))s)"
    else
        echo " FAILED rc=$rc"; tail -3 "$OUT/$name.log" | sed 's/^/       /'
    fi
}

echo "=== ViT (stream 17) — targets cover ~91% of stage GPU time ==="
run vit__cutlass_s16816gemm_bf16__29pct   --kernel-id ':17:regex:cutlass_80_tensorop_bf16_s16816gemm_bf16:' -c 4
run vit__ampere_s1688gemm_128x64__17pct   --kernel-id ':17:regex:ampere_bf16_s1688gemm_bf16_128x64:'        -c 4
run vit__flash_attn_hd80__16pct           --kernel-id ':17:regex:flash_attn_ext_f16:'                        -c 3
run vit__k_bin_bcast_add__9pct            --kernel-id ':17:regex:k_bin_bcast:'                               -c 3
run vit__convert_unary_bf16_to_f32__7pct  --kernel-id ':17:regex:convert_unary<__nv_bfloat16:'               -c 3
run vit__convert_unary_f32_to_bf16__6pct  --kernel-id ':17:regex:convert_unary<float:'                       -c 3
run vit__silu_gated__4pct                 --kernel-id ':17:regex:unary_gated_op_kernel:'                     -c 3
run vit__rope_vision__4pct                --kernel-id ':17:regex:rope_vision:'                               -c 3

echo "=== Prefill GEMMs (stream 15, stable — drift ~0%, sample anywhere) ~75% ==="
run prefill__ampere_s1688gemm_64x128__26pct  --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_64x128:'   -c 4
run prefill__mul_mat_f__17pct                -k regex:'mul_mat_f<'                          -s 400 -c 6
run prefill__cutlass_s16816gemm_relu__15pct  --kernel-id ':15:regex:cutlass_80_tensorop_bf16_s16816gemm_relu:' -c 4
run prefill__ampere_s1688gemm_128x64__12pct  --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_128x64:'   -c 4
run prefill__ampere_s1688gemm_128x128__5pct  --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_128x128:'  -c 4

echo "=== Prefill attention — CURVE, not a point (drifts +229% across the phase) ==="
for s in 60 320 640 900; do
  run "prefill_attn__hd128__ctxdepth_s${s}" \
      -k regex:'flash_attn_ext_f16<\(int\)128' -s "$s" -c 2
done

echo "=== Decode (~96% of stage) ==="
run decode__mul_mat_vec_f__shapes_3584_512_18944  -k regex:'mul_mat_vec_f'                  -c 14
run decode__mul_mat_vec_f__shape_152064_lmhead    -k regex:'mul_mat_vec_f'          -s 160  -c 4
run decode__flash_attn_vec__5pct                  -k regex:'flash_attn_ext_vec'             -c 3

echo
echo "=== done. reports in $OUT/ ==="
ls -la "$OUT"/*.ncu-rep 2>/dev/null | awk '{printf "  %6.1f MB  %s\n", $5/1e6, $9}'
