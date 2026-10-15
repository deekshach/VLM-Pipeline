// gemm_tune -- measure every cuBLASLt algorithm the heuristic offers for each BF16 GEMM shape the
// pipeline runs, and write the fastest per shape as a table for GGML_CUDA_JETSON_GEMM_TABLE.
//
//   GGML_CUDA_JETSON_GEMM_RECORD=shapes.txt llama-jetson-vlm ...   # collect the shapes once
//   jetson-gemm-tune shapes.txt table.txt [candidates=32] [reps=10]
//   jetson-gemm-tune --profile m n k d_f32 bias table.txt   # one call with the heuristic's algorithm and
//                                                           # one with the table's, in NVTX ranges
//                                                           # "heuristic" and "tuned" (for ncu)
//
// Shapes are "m n k d_f32 bias" (the GEMM is D[m x n] = A[k x m]^T B[k x n], BF16 in, F32 accumulate,
// D F32 or BF16, optional F32 bias epilogue), exactly as ggml-cuda/jetson-gemm.cu calls it.
// Each candidate is timed (median of `reps` after 3 warm-up calls); the heuristic's first pick is
// what the engine uses without a table, so its time is the baseline in the report.
#include <cublasLt.h>
#ifdef JETSON_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLASLt error %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(1); } } while (0)

static constexpr size_t WORKSPACE = 32u << 20;   // same as jetson-gemm.cu

// --profile: run one shape once with the heuristic's first pick and once with the table's algorithm.
static int profile_shape(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s --profile m n k d_f32 bias table.txt\n", argv[0]); return 1; }
    const long long m = atoll(argv[2]), n = atoll(argv[3]), k = atoll(argv[4]);
    const int d32 = atoi(argv[5]), use_bias = atoi(argv[6]);
    cublasLtMatmulAlgo_t tuned;
    bool have_tuned = false;
    if (FILE * f = fopen(argv[7], "r")) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            long long tm, tn, tk; int td, tb; unsigned long long w[8];
            if (sscanf(line, "%lld %lld %lld %d %d %llx %llx %llx %llx %llx %llx %llx %llx", &tm, &tn, &tk, &td, &tb,
                       &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]) == 13 &&
                tm == m && tn == n && tk == k && td == d32 && tb == use_bias) {
                memcpy(tuned.data, w, sizeof(w));
                have_tuned = true;
            }
        }
        fclose(f);
    }
    if (!have_tuned) { fprintf(stderr, "shape not in %s\n", argv[7]); return 1; }
    cublasLtHandle_t h; CL(cublasLtCreate(&h));
    void * ws; CK(cudaMalloc(&ws, WORKSPACE));
    nv_bfloat16 *A, *B; void * D; float * bias = nullptr;
    CK(cudaMalloc(&A, m * k * 2)); CK(cudaMalloc(&B, k * n * 2)); CK(cudaMalloc(&D, m * n * (d32 ? 4 : 2)));
    CK(cudaMemset(A, 0, m * k * 2)); CK(cudaMemset(B, 0, k * n * 2));
    if (use_bias) { CK(cudaMalloc(&bias, m * 4)); CK(cudaMemset(bias, 0, m * 4)); }
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, ld;
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    if (use_bias) {
        const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS; const cudaDataType_t bt = CUDA_R_32F;
        CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    }
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k));
    CL(cublasLtMatrixLayoutCreate(&ld, d32 ? CUDA_R_32F : CUDA_R_16BF, m, n, m));
    cublasLtMatmulPreference_t pref; CL(cublasLtMatmulPreferenceCreate(&pref));
    const size_t wsz = WORKSPACE;
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
    cublasLtMatmulHeuristicResult_t res; int n_res = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 1, &res, &n_res));
    const float alpha = 1.0f, beta = 0.0f;
    const cublasLtMatmulAlgo_t * algos[2] = { &res.algo, &tuned };
    const char * names[2] = { "heuristic", "tuned" };
    for (int i = 0; i < 2; i++) {
        CL(cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, algos[i], ws, WORKSPACE, 0));   // warm-up
        CK(cudaDeviceSynchronize());
#ifdef JETSON_NVTX
        nvtxRangePushA(names[i]);
#endif
        CL(cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, algos[i], ws, WORKSPACE, 0));
        CK(cudaDeviceSynchronize());
#ifdef JETSON_NVTX
        nvtxRangePop();
#endif
        printf("%s done\n", names[i]);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--profile") == 0) return profile_shape(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "usage: %s <shapes.txt> <table.txt> [candidates=32] [reps=10]\n", argv[0]);
        return 1;
    }
    const int n_cand = argc > 3 ? atoi(argv[3]) : 32;
    const int reps   = argc > 4 ? atoi(argv[4]) : 10;

    std::set<std::tuple<long long, long long, long long, int, int>> shapes;
    {
        FILE * f = fopen(argv[1], "r");
        if (!f) { perror(argv[1]); return 1; }
        long long m, n, k; int d32, bias;
        while (fscanf(f, "%lld %lld %lld %d %d", &m, &n, &k, &d32, &bias) == 5) shapes.insert({m, n, k, d32, bias});
        fclose(f);
    }
    FILE * out = fopen(argv[2], "w");
    if (!out) { perror(argv[2]); return 1; }
    fprintf(out, "# m n k d_f32 bias algo[8 x hex]   (jetson-gemm-tune, %d candidates, median of %d)\n", n_cand, reps);

    cublasLtHandle_t h;
    CL(cublasLtCreate(&h));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    void * ws;
    CK(cudaMalloc(&ws, WORKSPACE));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    double total_base = 0, total_best = 0;
    printf("%6s %6s %6s %3s %4s | %10s %10s %7s | candidates\n", "m", "n", "k", "f32", "bias", "heur0 us", "best us", "gain");
    for (const auto & s : shapes) {
        const long long m = std::get<0>(s), n = std::get<1>(s), k = std::get<2>(s);
        const int d32 = std::get<3>(s), use_bias = std::get<4>(s);

        nv_bfloat16 *A, *B;
        void * D;
        float * bias = nullptr;
        CK(cudaMalloc(&A, m * k * sizeof(nv_bfloat16)));
        CK(cudaMalloc(&B, k * n * sizeof(nv_bfloat16)));
        CK(cudaMalloc(&D, m * n * (d32 ? 4 : 2)));
        {   // small random values so the kernels do real work
            std::vector<nv_bfloat16> hA(m * k), hB(k * n);
            for (auto & x : hA) x = __float2bfloat16((rand() / (float) RAND_MAX - 0.5f) * 0.1f);
            for (auto & x : hB) x = __float2bfloat16((rand() / (float) RAND_MAX - 0.5f) * 0.1f);
            CK(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
        }
        if (use_bias) {
            CK(cudaMalloc(&bias, m * sizeof(float)));
            CK(cudaMemset(bias, 0, m * sizeof(float)));
        }

        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la, lb, ld;
        CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        if (use_bias) {
            const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
            const cudaDataType_t bt = CUDA_R_32F;
            CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
            CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
            CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
        }
        CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k));
        CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k));
        CL(cublasLtMatrixLayoutCreate(&ld, d32 ? CUDA_R_32F : CUDA_R_16BF, m, n, m));

        cublasLtMatmulPreference_t pref;
        CL(cublasLtMatmulPreferenceCreate(&pref));
        const size_t wsz = WORKSPACE;
        CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
        std::vector<cublasLtMatmulHeuristicResult_t> res(n_cand);
        int n_res = 0;
        CL(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, n_cand, res.data(), &n_res));

        const float alpha = 1.0f, beta = 0.0f;
        double base_us = -1, best_us = 1e30;
        int best = -1, n_ok = 0;
        for (int i = 0; i < n_res; i++) {
            if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
            bool good = true;
            for (int w = 0; w < 3 && good; w++) {
                good = cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &res[i].algo, ws, WORKSPACE, stream) ==
                       CUBLAS_STATUS_SUCCESS;
            }
            if (!good) continue;
            std::vector<float> t(reps);
            for (int r = 0; r < reps; r++) {
                CK(cudaEventRecord(e0, stream));
                CL(cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &res[i].algo, ws, WORKSPACE, stream));
                CK(cudaEventRecord(e1, stream));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&t[r], e0, e1));
            }
            std::sort(t.begin(), t.end());
            const double us = t[reps / 2] * 1000.0;
            n_ok++;
            if (i == 0) base_us = us;
            if (us < best_us) { best_us = us; best = i; }
        }
        if (best >= 0) {
            fprintf(out, "%lld %lld %lld %d %d", m, n, k, d32, use_bias);
            for (int w = 0; w < 8; w++) fprintf(out, " %llx", (unsigned long long) res[best].algo.data[w]);
            fprintf(out, "\n");
            if (base_us > 0) { total_base += base_us; total_best += best_us; }
            printf("%6lld %6lld %6lld %3d %4d | %10.1f %10.1f %6.2fx | %d tried, pick #%d\n", m, n, k, d32, use_bias,
                   base_us, best_us, base_us > 0 ? base_us / best_us : 0.0, n_ok, best);
        } else {
            printf("%6lld %6lld %6lld %3d %4d | no working candidate\n", m, n, k, d32, use_bias);
        }
        fflush(stdout);

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(op);
        cudaFree(A); cudaFree(B); cudaFree(D);
        if (bias) cudaFree(bias);
    }
    printf("sum over shapes (one call each): heuristic %.1f us -> tuned %.1f us (%.2fx)\n",
           total_base, total_best, total_best > 0 ? total_base / total_best : 0.0);
    fclose(out);
    return 0;
}
