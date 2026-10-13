#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // Pyramid KV Prefill (Architecture B) — visual token dropping during prefill
    bool  pyramid_enabled       = false;
    int   pyramid_clie_layer    = 2;      // scout depth (layers 0..clie_layer-1)
    float pyramid_pvc_min       = 0.20f;  // minimum retention ratio at deepest layer
    int   pyramid_visual_start  = 0;      // KV cache cell index of first visual token
    int   pyramid_n_visual      = 0;      // total visual token count
    int   pyramid_tokens_per_frame = 0;   // tokens in frame 0 (protected)
    bool  pyramid_protect_frame0 = true;
};
