// LLM prefill microarchitecture probe: the prompt-processing cost of N image tokens, the quantity
// adaptive sampling changes (576 LLM tokens per 1008x448 frame).
//
// Loads the LLM fully on the GPU with the pipeline's context settings (n_ubatch 512, flash
// attention AUTO, F16 KV cache) and prefills N tokens in ceil(N/512) ubatches. The first N - 512 tokens fill the KV cache; the last ubatch (512 tokens attending to
// all N) is decoded separately inside an NVTX range "prefill_last_<N>" and the whole prefill
// inside "prefill_<N>", so nsys and ncu can select either. Token ids stand in for image
// embeddings: every kernel after the embedding lookup is the same.
//
//   jetson-prefill-probe <model.gguf> <N e.g. 1152,2304,4608> [warm=1] [reps=3]
//
// The engine switches (GGML_CUDA_JETSON_GEMM_*) apply as set in the environment; PROBE_MARK_ALL=1 marks every
// timed repetition (not only the last), for averaging in nsys.
#include "llama.h"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static const int UBATCH = 512;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool decode(llama_context * ctx, std::vector<llama_token> & tok, int pos0, int n) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = tok[pos0 + i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = i == n - 1;
    }
    b.n_tokens = n;
    const bool ok = llama_decode(ctx, b) == 0;
    llama_synchronize(ctx);
    llama_batch_free(b);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf N[,N...] [warm=1] [reps=3]\n", argv[0]); return 1; }
    const int warm = argc > 3 ? atoi(argv[3]) : 1;
    const int reps = argc > 4 ? atoi(argv[4]) : 3;
    const bool mark_all = getenv("PROBE_MARK_ALL") != nullptr;   // nsys: every timed prefill in the ranges

    std::vector<int> ns;
    std::stringstream ss(argv[2]);
    for (std::string item; std::getline(ss, item, ',');) ns.push_back(atoi(item.c_str()));
    const int n_max = *std::max_element(ns.begin(), ns.end());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_max + 64;
    cp.n_batch = std::max(2048, n_max);
    cp.n_ubatch = UBATCH;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tok(n_max);
    for (int i = 0; i < n_max; i++) tok[i] = 1000 + (int)((i * 2654435761u) % (unsigned)(n_vocab / 2));

    printf("N,ubatches,median_ms,min_ms,last_ubatch_median_ms\n");
    for (int n : ns) {
        if (n <= UBATCH) { fprintf(stderr, "N must be > %d\n", UBATCH); return 1; }
        std::vector<double> t, tl;
        for (int i = 0; i < warm + reps; i++) {
            const bool mark = i == warm + reps - 1 || (mark_all && i >= warm);
            llama_memory_clear(llama_get_memory(ctx), true);
            const std::string r_all = "prefill_" + std::to_string(n), r_last = "prefill_last_" + std::to_string(n);
            if (mark) nvtxRangePushA(r_all.c_str());
            const double t0 = now_ms();
            if (!decode(ctx, tok, 0, n - UBATCH)) return 1;
            const double t1 = now_ms();
            if (mark) nvtxRangePushA(r_last.c_str());
            if (!decode(ctx, tok, n - UBATCH, UBATCH)) return 1;
            if (mark) nvtxRangePop();
            const double t2 = now_ms();
            if (mark) nvtxRangePop();
            if (i >= warm) { t.push_back(t2 - t0); tl.push_back(t2 - t1); }
        }
        std::sort(t.begin(), t.end()); std::sort(tl.begin(), tl.end());
        printf("%d,%d,%.2f,%.2f,%.2f\n", n, (n + UBATCH - 1) / UBATCH, t[t.size() / 2], t[0], tl[tl.size() / 2]);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
