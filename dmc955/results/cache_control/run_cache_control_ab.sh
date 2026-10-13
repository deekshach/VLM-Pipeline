set -uo pipefail
cd /data/spring26/CPP/llama.cpp/dmc955
source ./model_config.sh
SCR=/scratch/setupfiles/lockdir/claude-5760439/-data-spring26-CPP-llama-cpp-dmc955/361b2341-505b-45c0-80de-3d15254d849b/scratchpad
NCU=/usr/local/cuda-12.6/bin/ncu
M="lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,dram__sectors_read.sum,dram__sectors_write.sum,gpu__time_duration.sum"
WL=(-m "$MODEL" --mmproj "$MMPROJ" --sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel
    --video alley_1 --base-frames 8 -ngl 99 -c 16384 -n 4 --temp 0 --seed 42 -fa on
    --no-flow-analysis --ignore-eos --warmup-iters 0)

probe () {   # $1 tag  $2 cachectl  $3.. selector
  local tag="$1" cc="$2"; shift 2
  "$NCU" --metrics "$M" --kernel-name-base demangled --cache-control "$cc" --clock-control base \
      "$@" --target-processes all ../build/bin/llama-baseline-vlm "${WL[@]}" \
      > "$SCR/cc_${tag}_${cc}.log" 2>&1
  echo "  done $tag/$cc"
}

for cc in all none; do
  probe vit_gemm      $cc --kernel-id ':17:regex:cutlass_80_tensorop_bf16_s16816gemm_bf16:' -c 3
  probe vit_silu      $cc --kernel-id ':17:regex:unary_gated_op_kernel:'                    -c 3
  probe pre_gemm      $cc --kernel-id ':15:regex:ampere_bf16_s1688gemm_bf16_64x128:'        -c 3
  probe pre_attn      $cc -k regex:'flash_attn_ext_f16<\(int\)128' -s 150 -c 3
done
echo "=== CC PROBE DONE ==="
