#include "jetson-gemm.cuh"
#include "convert.cuh"
#include "mmf.cuh"
#include "mmvf.cuh"

#include <cublasLt.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

#define CUBLASLT_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    GGML_ABORT("cuBLASLt error %d at %s:%d: %s", (int) s_, __FILE__, __LINE__, #x); } } while (0)

static constexpr size_t JETSON_GEMM_WORKSPACE = 32u << 20;

const ggml_cuda_jetson_gemm_switches & ggml_cuda_jetson_gemm_get_switches() {
    static const ggml_cuda_jetson_gemm_switches sw = [] {
        auto on = [](const char * name) { const char * v = getenv(name); return v != nullptr && atoi(v) != 0; };
        ggml_cuda_jetson_gemm_switches s;
        s.f32_out   = on("GGML_CUDA_JETSON_GEMM_F32OUT");
        s.bias_fuse = on("GGML_CUDA_JETSON_GEMM_BIAS");
        s.table     = getenv("GGML_CUDA_JETSON_GEMM_TABLE")  != nullptr;
        s.record    = getenv("GGML_CUDA_JETSON_GEMM_RECORD") != nullptr;
        s.act_dedup = on("GGML_CUDA_JETSON_ACT_DEDUP");
        s.act_fuse  = on("GGML_CUDA_JETSON_ACT_FUSE");
        if (s.bias_fuse && !s.f32_out) {
            GGML_LOG_WARN("%s: GGML_CUDA_JETSON_GEMM_BIAS needs GGML_CUDA_JETSON_GEMM_F32OUT; bias fusion off\n", __func__);
            s.bias_fuse = false;
        }
        if (s.any() || s.record) {
            GGML_LOG_INFO("%s: cuBLASLt BF16 GEMM switches: f32_out=%d bias_fuse=%d table=%s record=%s act_dedup=%d act_fuse=%d\n",
                          __func__, s.f32_out, s.bias_fuse, s.table ? getenv("GGML_CUDA_JETSON_GEMM_TABLE") : "-",
                          s.record ? getenv("GGML_CUDA_JETSON_GEMM_RECORD") : "-", s.act_dedup, s.act_fuse);
        }
        return s;
    }();
    return sw;
}

// shape key: m, n, k, D is F32, bias epilogue
using gemm_key = std::tuple<int64_t, int64_t, int64_t, int, int>;

struct jetson_gemm_state {
    std::mutex mtx;
    cublasLtHandle_t handle[GGML_CUDA_MAX_DEVICES] = {};
    std::map<cudaStream_t, void *> workspace;   // per stream: split-K keeps partial sums and counters here, so
                                                // two models running concurrently must not share one
    std::map<gemm_key, cublasLtMatmulAlgo_t> tuned;      // from GGML_CUDA_JETSON_GEMM_TABLE
    std::map<gemm_key, cublasLtMatmulAlgo_t> heuristic;  // first heuristic pick, cached per shape
    std::set<gemm_key> recorded;
    bool table_loaded = false;

    // BF16 activation cache: a few slots per stream, least recently used reused first
    struct act_slot { const ggml_tensor * t = nullptr; const void * data = nullptr; nv_bfloat16 * buf = nullptr;
                      size_t cap = 0; uint64_t used = 0; };
    struct act_cache { act_slot slot[4]; std::map<const ggml_tensor *, int> produced;   // BF16-only: readers left
                       uint64_t clock = 0; std::vector<nv_bfloat16 *> retired; };
    std::map<cudaStream_t, act_cache> act;
};

static jetson_gemm_state & state() {
    static jetson_gemm_state s;
    return s;
}

// Table lines: m n k d_f32 bias, then the 8 words of cublasLtMatmulAlgo_t in hex.
static void load_table(jetson_gemm_state & st) {
    st.table_loaded = true;
    const char * path = getenv("GGML_CUDA_JETSON_GEMM_TABLE");
    FILE * f = path ? fopen(path, "r") : nullptr;
    if (!f) {
        GGML_LOG_WARN("%s: cannot open GEMM table '%s'; using the heuristic\n", __func__, path ? path : "");
        return;
    }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        long long m, n, k; int d_f32, bias;
        unsigned long long w[8];
        if (sscanf(line, "%lld %lld %lld %d %d %llx %llx %llx %llx %llx %llx %llx %llx", &m, &n, &k, &d_f32, &bias,
                   &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]) != 13) continue;
        cublasLtMatmulAlgo_t algo;
        static_assert(sizeof(algo.data) == sizeof(w), "unexpected cublasLtMatmulAlgo_t layout");
        memcpy(algo.data, w, sizeof(w));
        st.tuned[gemm_key(m, n, k, d_f32, bias)] = algo;
    }
    fclose(f);
    GGML_LOG_INFO("%s: %zu tuned GEMM shapes from %s\n", __func__, st.tuned.size(), path);
}

static void record_shape(jetson_gemm_state & st, const gemm_key & key) {
    if (!st.recorded.insert(key).second) return;
    FILE * f = fopen(getenv("GGML_CUDA_JETSON_GEMM_RECORD"), "a");
    if (!f) return;
    fprintf(f, "%" PRId64 " %" PRId64 " %" PRId64 " %d %d\n",
            std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), std::get<4>(key));
    fclose(f);
}

void ggml_cuda_jetson_gemm_bf16(
        ggml_backend_cuda_context & ctx, cudaStream_t stream,
        int64_t m, int64_t n, int64_t k,
        const nv_bfloat16 * A, const nv_bfloat16 * B,
        void * D, bool d_f32, int64_t ldd,
        const float * bias) {
    GGML_ASSERT(bias == nullptr || d_f32);
    const auto & sw = ggml_cuda_jetson_gemm_get_switches();
    auto & st = state();
    const int device = ctx.device;
    const gemm_key key(m, n, k, d_f32 ? 1 : 0, bias ? 1 : 0);

    cublasLtHandle_t handle;
    void * workspace;
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        if (!st.handle[device]) CUBLASLT_CHECK(cublasLtCreate(&st.handle[device]));
        void *& ws = st.workspace[stream];
        if (!ws) CUDA_CHECK(cudaMalloc(&ws, JETSON_GEMM_WORKSPACE));
        if (sw.table && !st.table_loaded) load_table(st);
        if (sw.record) record_shape(st, key);
        handle = st.handle[device];
        workspace = ws;
    }

    const cudaDataType_t d_type = d_f32 ? CUDA_R_32F : CUDA_R_16BF;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    if (bias) {
        const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        const cudaDataType_t bias_type = CUDA_R_32F;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
    }
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ld, d_type, m, n, ldd));

    // algorithm: tuned for this shape if available and valid, else the heuristic's first pick
    cublasLtMatmulAlgo_t algo;
    bool have_algo = false;
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        auto it = st.tuned.find(key);
        if (it != st.tuned.end()) {
            cublasLtMatmulHeuristicResult_t chk;
            if (cublasLtMatmulAlgoCheck(handle, op, la, lb, ld, ld, &it->second, &chk) == CUBLAS_STATUS_SUCCESS &&
                chk.workspaceSize <= JETSON_GEMM_WORKSPACE) {
                algo = it->second;
                have_algo = true;
            }
        }
        if (!have_algo) {
            auto ih = st.heuristic.find(key);
            if (ih != st.heuristic.end()) {
                algo = ih->second;
                have_algo = true;
            }
        }
    }
    if (!have_algo) {
        cublasLtMatmulPreference_t pref;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        const size_t ws = JETSON_GEMM_WORKSPACE;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
        cublasLtMatmulHeuristicResult_t res;
        int n_res = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, ld, ld, pref, 1, &res, &n_res));
        cublasLtMatmulPreferenceDestroy(pref);
        GGML_ASSERT(n_res > 0 && "no cuBLASLt algorithm for BF16 GEMM");
        algo = res.algo;
        std::lock_guard<std::mutex> lk(st.mtx);
        st.heuristic[key] = algo;
    }

    const float alpha = 1.0f, beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(handle, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld,
                                  &algo, workspace, JETSON_GEMM_WORKSPACE, stream));

    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
}

// ---- BF16 activation cache

void ggml_cuda_jetson_act_reset(cudaStream_t stream) {
    auto & st = state();
    std::lock_guard<std::mutex> lk(st.mtx);
    auto & c = st.act[stream];
    GGML_ASSERT(c.produced.empty() && "Jetson ACT_FUSE: a BF16-only activation was not read by all its GEMMs");
    for (auto & sl : c.slot) { sl.t = nullptr; sl.data = nullptr; }
}

// slot for t (the least recently used one), large enough for n BF16 values; caller holds st.mtx
static nv_bfloat16 * act_slot_for(jetson_gemm_state::act_cache & c, cudaStream_t stream, const ggml_tensor * t, size_t n) {
    auto * victim = &c.slot[0];
    for (auto & sl : c.slot) if (sl.used < victim->used) victim = &sl;
    GGML_ASSERT((victim->t == nullptr || !c.produced.count(victim->t)) &&
                "Jetson ACT_FUSE: a BF16-only activation would be evicted before its GEMMs ran");
    if (victim->cap < n) {
        // a captured CUDA graph may still use the old buffer: retire it instead of freeing it (no sync, so this
        // also works during capture, which ggml runs in relaxed mode); sizes settle after the first graphs
        if (victim->buf) c.retired.push_back(victim->buf);
        CUDA_CHECK(cudaMalloc(&victim->buf, n * sizeof(nv_bfloat16)));
        victim->cap = n;
    }
    GGML_UNUSED(stream);
    victim->t = t; victim->data = t->data; victim->used = ++c.clock;
    return victim->buf;
}

static bool act_qualifies(const ggml_tensor * src1) {
    return src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[2] == 1 && src1->ne[3] == 1;
}

const nv_bfloat16 * ggml_cuda_jetson_act_get(ggml_backend_cuda_context & ctx, cudaStream_t stream, const ggml_tensor * src1) {
    const auto & sw = ggml_cuda_jetson_gemm_get_switches();
    if (!sw.act() || !act_qualifies(src1)) return nullptr;
    auto & st = state();
    std::lock_guard<std::mutex> lk(st.mtx);
    auto & c = st.act[stream];
    for (auto & sl : c.slot) {
        if (sl.t == src1 && sl.data == src1->data) {
            sl.used = ++c.clock;
            auto ip = c.produced.find(src1);
            if (ip != c.produced.end() && --ip->second == 0) c.produced.erase(ip);   // last reader: slot reusable
            return sl.buf;
        }
    }
    GGML_ASSERT(!c.produced.count(src1) && "Jetson ACT_FUSE: BF16-only activation not found");
    if (!sw.act_dedup) return nullptr;
    const size_t n = ggml_nelements(src1);
    nv_bfloat16 * buf = act_slot_for(c, stream, src1, n);
    ggml_get_to_bf16_cuda(GGML_TYPE_F32)(src1->data, buf, n, stream);
    GGML_UNUSED(ctx);
    return buf;
}

// would ggml-cuda run this MUL_MAT through ggml_cuda_op_mul_mat_cublas (where the cache is read)?
static bool act_reader_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * t) {
    const ggml_tensor * src0 = mm->src[0];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != t || mm->src[0] == t || mm->type != GGML_TYPE_F32) return false;
    if (src0->type != GGML_TYPE_BF16 || !ggml_is_contiguous(src0) || src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!src0->buffer || strstr(ggml_backend_buft_name(ggml_backend_buffer_get_type(src0->buffer)), "_Split")) return false;
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    return !ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, t->ne[1]) &&
           !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, t->ne[1], false);
}

int ggml_cuda_jetson_act_fusable(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_tensor * t) {
    if (!ggml_cuda_jetson_gemm_get_switches().act_fuse || !act_qualifies(t) || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) return 0;
    int readers = 0;
    for (int j = i + 1; j < cgraph->n_nodes; j++) {
        const ggml_tensor * nd = cgraph->nodes[j];
        if (nd->view_src == t) return 0;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (nd->src[s] != t) continue;
            if (!act_reader_ok(ctx, nd, t)) return 0;
            readers++;
        }
    }
    return readers;
}

nv_bfloat16 * ggml_cuda_jetson_act_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int readers) {
    auto & st = state();
    cudaStream_t stream = ctx.stream();
    std::lock_guard<std::mutex> lk(st.mtx);
    auto & c = st.act[stream];
    nv_bfloat16 * buf = act_slot_for(c, stream, t, ggml_nelements(t));
    c.produced[t] = readers;
    return buf;
}

bool ggml_cuda_jetson_gemm_try_fuse_bias(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * add) {
    const auto & sw = ggml_cuda_jetson_gemm_get_switches();
    if (!sw.bias_fuse || mm->op != GGML_OP_MUL_MAT || add->op != GGML_OP_ADD) return false;

    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    // (the caller rules out split buffers)
    if (src0->type != GGML_TYPE_BF16 || !ggml_is_contiguous(src0) || src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (mm->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || !ggml_are_same_shape(mm, add)) return false;

    const ggml_tensor * bias = add->src[0] == mm ? add->src[1] : add->src[1] == mm ? add->src[0] : nullptr;
    if (!bias || bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) || bias->ne[0] != mm->ne[0] ||
        ggml_nelements(bias) != bias->ne[0]) return false;

    // only where ggml would otherwise use cuBLAS (small batches go to its own kernels)
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]) ||
        ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], false)) return false;

    const int64_t m = src0->ne[1], k = src0->ne[0], n = src1->ne[1];
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<nv_bfloat16> src1_bf16(ctx.pool());
    const nv_bfloat16 * src1_ptr = ggml_cuda_jetson_act_get(ctx, stream, src1);   // ACT switches
    if (!src1_ptr) {
        src1_bf16.alloc(k * n);
        ggml_get_to_bf16_cuda(GGML_TYPE_F32)(src1->data, src1_bf16.get(), k * n, stream);
        src1_ptr = src1_bf16.get();
    }
    ggml_cuda_jetson_gemm_bf16(ctx, stream, m, n, k, (const nv_bfloat16 *) src0->data, src1_ptr,
                               add->data, true, m, (const float *) bias->data);
    return true;
}
