// bench_vit_gemm.cu -- settle the ViT GEMM efficiency question.
//
// WHY
//   The f8 census + FLOP arithmetic says the ViT's GEMMs sustain ~140 TFLOP/s,
//   which is 85% of the commonly quoted 165.2 TFLOP/s BF16 dense peak for a
//   4090. ncu reports the tensor pipe at 43.6% for the same kernels -- a ratio
//   of almost exactly 2x. Both cannot be right, and it decides whether those
//   GEMMs have any headroom left (i.e. whether "ViT GEMMs are L2-leaning" or
//   "ViT GEMMs are near compute peak" is the correct reading).
//
// METHOD
//   Replicate ggml's call exactly -- cublasGemmEx, CUDA_R_16BF in/out,
//   CUBLAS_COMPUTE_32F, CUBLAS_OP_T/CUBLAS_OP_N, lda=ldb=k, ldc=m
//   (ggml-cuda.cu:1277) -- at the three real ViT shapes, and ALSO measure the
//   machine's own achievable peak with a large square GEMM. Efficiency is then
//   quoted against a MEASURED ceiling, not a spec sheet.
//
//   COMPUTE_16F is measured too: if FP32 accumulate is half-rate on this part,
//   that is the 2x and it shows up here directly.
//
// ggml semantics: dst[n_out, n_tokens] = W[k, n_out]^T * x[k, n_tokens]
//   Q/K/V/O : m=1280 n=2368 k=1280
//   up/gate : m=3420 n=2368 k=1280
//   down    : m=1280 n=2368 k=3420
//
// build: /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_89 -lcublas -o bench_vit_gemm bench_vit_gemm.cu

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x)  do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CB(x)  do { cublasStatus_t cb_st_=(x); if(cb_st_){printf("cuBLAS %d @%d\n",(int)cb_st_,__LINE__);exit(1);} } while(0)

struct Shape { const char* name; int m, n, k; };

static double run(cublasHandle_t h, Shape s, cublasComputeType_t ct, int iters,
                  cudaDataType_t dt = CUDA_R_16BF) {
    __nv_bfloat16 *A, *B, *C;
    CK(cudaMalloc(&A, (size_t)s.k * s.m * 2));
    CK(cudaMalloc(&B, (size_t)s.k * s.n * 2));
    CK(cudaMalloc(&C, (size_t)s.m * s.n * 2));
    CK(cudaMemset(A, 0x3c, (size_t)s.k * s.m * 2));   // ~1.0 in bf16, avoids denormal paths
    CK(cudaMemset(B, 0x3c, (size_t)s.k * s.n * 2));

    const float  a32 = 1.0f, b32 = 0.0f;
    const __half a16 = __float2half(1.0f), b16 = __float2half(0.0f);
    const void *alpha = (ct == CUBLAS_COMPUTE_16F) ? (const void*)&a16 : (const void*)&a32;
    const void *beta  = (ct == CUBLAS_COMPUTE_16F) ? (const void*)&b16 : (const void*)&b32;

    auto call = [&]{
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                        s.m, s.n, s.k,
                        alpha, A, dt, s.k,
                               B, dt, s.k,
                        beta,  C, dt, s.m,
                        ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    // BF16 inputs admit only FP32 accumulate; report unsupported rather than abort.
    if (call() != CUBLAS_STATUS_SUCCESS) {
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
        return -1.0;
    }
    for (int i = 0; i < 20; i++) call();          // warm up + let clocks ramp
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) call();
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1));

    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    double us    = (double)ms * 1000.0 / iters;
    double flops = 2.0 * s.m * s.n * s.k;
    return flops / (us * 1e-6) / 1e12;             // TFLOP/s
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s  sm_%d%d  %d SMs  clock %.0f MHz\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate / 1000.0);

    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    std::vector<Shape> shapes = {
        {"PEAK probe 8192^3",    8192, 8192, 8192},
        {"PEAK probe 4096^3",    4096, 4096, 4096},
        {"ViT Q/K/V/O",          1280, 2368, 1280},
        {"ViT up/gate",          3420, 2368, 1280},
        {"ViT down",             1280, 2368, 3420},
    };

    printf("%-22s %6s %6s %6s %10s %12s %12s\n",
           "shape", "m", "n", "k", "GFLOP", "COMPUTE_32F", "COMPUTE_16F");
    printf("%s\n", std::string(80, '-').c_str());

    double peak32 = 0, peak16 = 0;
    std::vector<std::pair<std::string,double>> vit;
    for (auto& s : shapes) {
        int iters = ((double)s.m * s.n * s.k > 1e11) ? 30 : 300;
        double t32 = run(h, s, CUBLAS_COMPUTE_32F, iters);
        double t16 = run(h, s, CUBLAS_COMPUTE_16F, iters);
        double gf  = 2.0 * s.m * s.n * s.k / 1e9;
        printf("%-22s %6d %6d %6d %10.2f %12.1f %12s\n", s.name, s.m, s.n, s.k, gf, t32,
               t16 < 0 ? "unsupported" : (std::to_string((int)t16)).c_str());
        if (std::string(s.name).rfind("PEAK", 0) == 0) {
            if (t32 > peak32) peak32 = t32;
            if (t16 > peak16) peak16 = t16;
        } else {
            vit.push_back({s.name, t32});
        }
    }

    printf("\nMEASURED BF16 peak (best square GEMM, COMPUTE_32F): %.1f TFLOP/s\n", peak32);

    // Is FP32 accumulate reduced-rate? BF16 has no FP16-accumulate path, so ask
    // the question with FP16 inputs, where both accumulate modes are legal.
    Shape probe = {"fp16 probe", 8192, 8192, 8192};
    double f32 = run(h, probe, CUBLAS_COMPUTE_32F, 30, CUDA_R_16F);
    double f16 = run(h, probe, CUBLAS_COMPUTE_16F, 30, CUDA_R_16F);
    printf("FP16 8192^3: FP32-accum %.1f   FP16-accum %.1f TFLOP/s  -> ratio %.2fx  (%s)\n",
           f32, f16, f16 / f32,
           f16 > 1.15 * f32 ? "FP32 accumulate IS half-rate" : "FP32 accumulate is NOT reduced-rate");

    printf("\nViT GEMM efficiency, against the MEASURED COMPUTE_32F peak (what ggml uses):\n");
    for (auto& v : vit)
        printf("  %-18s %7.1f TFLOP/s = %5.1f%% of measured peak\n",
               v.first.c_str(), v.second, 100.0 * v.second / peak32);

    printf("\nFor reference, in-pipeline (nsys census, f8): Q/K/V/O 145.6, up/gate 139.4,\n");
    printf("down 134.7, all-GEMM 140.3 TFLOP/s. ncu tensor-pipe reads 43.6%%.\n");

    CB(cublasDestroy(h));
    return 0;
}
