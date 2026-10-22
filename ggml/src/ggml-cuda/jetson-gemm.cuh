#pragma once
// BF16 GEMMs through cuBLASLt (Jetson study switches; all default off, so behaviour is unchanged).
//
//   GGML_CUDA_JETSON_GEMM_F32OUT=1      write the GEMM result as F32 directly instead of BF16 followed by
//                                        a separate BF16 -> F32 conversion kernel
//   GGML_CUDA_JETSON_GEMM_BIAS=1        fold a following bias ADD into the GEMM epilogue (needs F32OUT)
//   GGML_CUDA_JETSON_GEMM_RESIDUAL=1    fold the residual ADD that follows a GEMM (after its bias ADD, if any) into the
//                                        GEMM as D = A*B (+ bias) + C, C the residual (cuBLASLt beta = 1; needs F32OUT)
//   GGML_CUDA_JETSON_GEMM_TABLE=<file>  use the cuBLASLt algorithm measured fastest for each GEMM shape
//                                        (written by jetson/tools/gemm_tune) instead of the heuristic pick
//   GGML_CUDA_JETSON_GEMM_RECORD=<file> append every BF16 GEMM shape seen, as input for the tuner
//   GGML_CUDA_JETSON_ACT_DEDUP=1        convert each F32 GEMM input to BF16 once and reuse it for every GEMM that
//                                        reads it (Q/K/V share the attention-norm output, gate/up the FFN-norm output)
//   GGML_CUDA_JETSON_ACT_FUSE=1         producers write the BF16 GEMM input directly: SwiGLU (input of the down
//                                        projection) and RMS norm x weight (input of Q/K/V and gate/up), when every
//                                        consumer is such a GEMM; the F32 tensor is then never written
//
// The two ACT switches change where the F32 -> BF16 rounding happens, not its result (bit-identical output).
// ACT_FUSE counts a tensor's readers in the graph being evaluated, so it assumes they are all in the same graph
// split: true with the whole model on the GPU (the ViT is one split; the LLM is the CPU embedding lookup + one GPU
// split). The BF16 buffers work under CUDA graph capture (never freed, only retired when they grow).
// They target the 4 MB L2: a 512-token activation (7.3 MB F32, 39 MB for the FFN) does not stay in L2 between
// kernels, so each extra conversion pass and each F32 intermediate is a full DRAM round trip.
//
// Any of the first three routes BF16 x F32 cuBLAS GEMMs through cuBLASLt.

#include "common.cuh"

struct ggml_cuda_jetson_gemm_switches {
    bool f32_out   = false;
    bool bias_fuse = false;
    bool residual  = false;
    bool table     = false;
    bool record    = false;
    bool act_dedup = false;
    bool act_fuse  = false;
    bool any() const { return f32_out || bias_fuse || table; }
    bool act() const { return act_dedup || act_fuse; }
};

const ggml_cuda_jetson_gemm_switches & ggml_cuda_jetson_gemm_get_switches();

// D[m x n] (column-major, leading dimension ldd) = op(A)^T * B, with A BF16 [k x m] (lda = k) and
// B BF16 [k x n] (ldb = k), F32 accumulation. D is F32 when d_f32, else BF16. With bias != nullptr
// (F32, length m, needs d_f32) the bias is added to every column in the epilogue. With C != nullptr (F32, the layout
// of D, needs d_f32; may be D itself) D = A*B (+ bias) + C.
void ggml_cuda_jetson_gemm_bf16(
        ggml_backend_cuda_context & ctx, cudaStream_t stream,
        int64_t m, int64_t n, int64_t k,
        const nv_bfloat16 * A, const nv_bfloat16 * B,
        void * D, bool d_f32, int64_t ldd,
        const float * bias, const float * C = nullptr);

// Fused MUL_MAT + ADD(bias) node pair: mm = MUL_MAT(src0 BF16 weight, src1 F32), add = ADD(mm, bias).
// Returns false (and does nothing) when the pair does not qualify. src0 must not be in a split buffer.
bool ggml_cuda_jetson_gemm_try_fuse_bias(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add);

// GGML_CUDA_JETSON_GEMM_RESIDUAL: mm = MUL_MAT(src0 BF16 weight, src1 F32), add_bias = ADD(mm, bias) or nullptr,
// add_res = ADD(add_bias or mm, residual) with the residual of the same shape. One cuBLASLt call writes add_res.
// Returns false (and does nothing) when the nodes do not qualify. src0 must not be in a split buffer.
bool ggml_cuda_jetson_gemm_try_fuse_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
                                             const ggml_tensor * add_bias, ggml_tensor * add_res);

// ---- BF16 activation cache (GGML_CUDA_JETSON_ACT_*), per stream, valid within one graph evaluation.
// Start of a graph evaluation: forget every cached BF16 activation of this stream.
void ggml_cuda_jetson_act_reset(cudaStream_t stream);

// BF16 copy of the F32 GEMM input src1 (contiguous, 2-D, on this device): the cached one if a producer wrote it or
// an earlier GEMM converted it; else, with ACT_DEDUP, converted now and cached; else nullptr (the caller converts).
const nv_bfloat16 * ggml_cuda_jetson_act_get(ggml_backend_cuda_context & ctx, cudaStream_t stream, const ggml_tensor * src1);

// ACT_FUSE: number of readers if node `t` (at graph index i) can be produced as BF16 only (every later reader is a
// MUL_MAT taking it as src1 that runs through the cuBLAS(Lt) BF16 path, and nothing else uses it), else 0.
int ggml_cuda_jetson_act_fusable(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_tensor * t);

// ACT_FUSE: buffer for the BF16 version of `t`, registered so its `readers` GEMMs find it (kept until the last ran).
nv_bfloat16 * ggml_cuda_jetson_act_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int readers);
