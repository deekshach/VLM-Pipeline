#!/usr/bin/env bash
# run_ncu_f8.sh -- ncu survey at the frames=8 operating point.
#
# METHOD (differs from run_ncu_survey.sh in four ways):
#  1. Targets are derived from the f8 kernel census (kernel_census.py), walking
#     each stage's time-ordered shape list to ~90%+ coverage -- not hand-picked.
#     The old hand list missed 17.0% of the ViT stage, three of those shapes being
#     the SAME kernel name at a different grid than the one profiled.
#  2. ViT/prefill are separated by CUDA stream (:17: / :15:) because they share
#     kernel names (convert_unary, k_bin_bcast, unary_gated_op_kernel, rms_norm).
#  3. ncu forbids --launch-skip together with a stream-filtered --kernel-id, so
#     the two targets needing -s use plain -k regex: on phase-exclusive names.
#  4. Analysis is ncu_bottleneck.py, which reads L2 THROUGHPUT, tensor-pipe %,
#     eligible warps and the occupancy block-limits -- not just hit rate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./model_config.sh

NCU=/usr/local/cuda-12.6/bin/ncu
OUT="${OUT:-ncu_$(date +%Y-%m-%d)_f8}"
mkdir -p "$OUT"

# -n 16 keeps each replay short; decode kernels are context-insensitive apart from
# attention, and at f8 the KV differs by ~110 of ~4800 tokens vs the -n 128 nsys run.
WL=(-m "$MODEL" --mmproj "$MMPROJ"
    --sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel
    --video alley_1 --base-frames 8
    -ngl 99 -c 16384 -n 16 --temp 0 --seed 42 -fa on
    --no-flow-analysis --ignore-eos --warmup-iters 0)

run() {
    local name="$1"; shift
    [[ -f "$OUT/$name.ncu-rep" ]] && { echo "  SKIP $name"; return 0; }
    printf "  %-44s" "$name"
    local t0=$SECONDS
    "$NCU" --set full --graph-profiling node --kernel-name-base demangled \
        "$@" --target-processes all -o "$OUT/$name" --force-overwrite \
        ../build/bin/llama-baseline-vlm "${WL[@]}" > "$OUT/$name.log" 2>&1
    if [[ -f "$OUT/$name.ncu-rep" ]]; then echo " ok ($((SECONDS-t0))s)"
    else echo " FAILED"; tail -3 "$OUT/$name.log" | sed 's/^/       /'; fi
}

echo "=== ViT (stream 17) -- census targets to ~95% of stage ==="
run vit__cutlass_gemm_bf16__30pct  --kernel-id ':17:regex:cutlass_80_tensorop_bf16_s16816gemm_bf16:' -c 6
run vit__flash_attn_hd80__16pct    --kernel-id ':17:regex:flash_attn_ext_f16:'                        -c 3
run vit__ampere_128x64__16pct      --kernel-id ':17:regex:ampere_bf16_s1688gemm_bf16_128x64:'         -c 4
run vit__k_bin_bcast__9pct         --kernel-id ':17:regex:k_bin_bcast:'                               -c 6
run vit__convert_bf16_to_f32__7pct --kernel-id ':17:regex:convert_unary<__nv_bfloat16:'               -c 6
run vit__convert_f32_to_bf16__6pct --kernel-id ':17:regex:convert_unary<float:'                       -c 4
run vit__silu_gated__4pct          --kernel-id ':17:regex:unary_gated_op_kernel:'                     -c 3
run vit__rope_vision__4pct         --kernel-id ':17:regex:rope_vision:'                               -c 3
run vit__rms_norm__3pct            --kernel-id ':17:regex:rms_norm_f32:'                              -c 4
run vit__cpy_scalar__3pct          --kernel-id ':17:regex:cpy_scalar:'                                -c 3

echo "=== prefill (stream 15) -- census targets to ~88% of stage ==="
run prefill__ampere_64x128__26pct   --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_64x128:'   -c 4
run prefill__ampere_128x64__13pct   --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_128x64:'   -c 4
run prefill__cutlass_relu__15pct    --kernel-id ':15:regex:cutlass_80_tensorop_bf16_s16816gemm_relu:' -c 6
run prefill__ampere_128x128__6pct   --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_128x128:'  -c 4
run prefill__ampere_64x64__2pct     --kernel-id ':15:regex:ampere_bf16_s16816gemm_bf16_64x64:'   -c 3
run prefill__silu_gated__3pct       --kernel-id ':15:regex:unary_gated_op_kernel:'               -c 3
run prefill__convert_bf16_to_f32__2pct --kernel-id ':15:regex:convert_unary<__nv_bfloat16:'      -c 3
# name is phase-exclusive (0 launches in ViT), so -s is safe here
run prefill__mul_mat_f__14pct       -k regex:'mul_mat_f<' -s 200 -c 8

echo "=== prefill attention -- a CURVE (8x spread at one grid: 24.4-201.5 us) ==="
for s in 40 150 280 400; do
  run "prefill_attn__hd128__depth_s${s}" -k regex:'flash_attn_ext_f16<\(int\)128' -s "$s" -c 2
done

echo "=== decode -- census targets to ~97% of stage ==="
run decode__mul_mat_vec_f__all_shapes  -k regex:'mul_mat_vec_f'      -c 14
run decode__mul_mat_vec_f__lmhead      -k regex:'mul_mat_vec_f' -s 160 -c 4
run decode__flash_attn_vec__3pct       -k regex:'flash_attn_ext_vec' -c 3

echo; echo "=== done -> $OUT ==="
ls "$OUT"/*.ncu-rep 2>/dev/null | wc -l | xargs echo "  reports:"
