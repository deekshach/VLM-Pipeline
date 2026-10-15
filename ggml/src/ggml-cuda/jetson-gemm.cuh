#pragma once
// BF16 GEMMs through cuBLASLt (Jetson study switches; all default off, so behaviour is unchanged).
//
//   GGML_CUDA_JETSON_GEMM_F32OUT=1      write the GEMM result as F32 directly instead of BF16 followed by
//                                        a separate BF16 -> F32 conversion kernel
//   GGML_CUDA_JETSON_GEMM_BIAS=1        fold a following bias ADD into the GEMM epilogue (needs F32OUT)
//   GGML_CUDA_JETSON_GEMM_TABLE=<file>  use the cuBLASLt algorithm measured fastest for each GEMM shape
//                                        (written by jetson/tools/gemm_tune) instead of the heuristic pick
//   GGML_CUDA_JETSON_GEMM_RECORD=<file> append every BF16 GEMM shape seen, as input for the tuner
//
// Any of the first three routes BF16 x F32 cuBLAS GEMMs through cuBLASLt.

#include "common.cuh"

struct ggml_cuda_jetson_gemm_switches {
    bool f32_out   = false;
    bool bias_fuse = false;
    bool table     = false;
    bool record    = false;
    bool any() const { return f32_out || bias_fuse || table; }
};

const ggml_cuda_jetson_gemm_switches & ggml_cuda_jetson_gemm_get_switches();

// D[m x n] (column-major, leading dimension ldd) = op(A)^T * B, with A BF16 [k x m] (lda = k) and
// B BF16 [k x n] (ldb = k), F32 accumulation. D is F32 when d_f32, else BF16. With bias != nullptr
// (F32, length m, needs d_f32) the bias is added to every column in the epilogue.
void ggml_cuda_jetson_gemm_bf16(
        ggml_backend_cuda_context & ctx, cudaStream_t stream,
        int64_t m, int64_t n, int64_t k,
        const nv_bfloat16 * A, const nv_bfloat16 * B,
        void * D, bool d_f32, int64_t ldd,
        const float * bias);

// Fused MUL_MAT + ADD(bias) node pair: mm = MUL_MAT(src0 BF16 weight, src1 F32), add = ADD(mm, bias).
// Returns false (and does nothing) when the pair does not qualify. src0 must not be in a split buffer.
bool ggml_cuda_jetson_gemm_try_fuse_bias(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add);
