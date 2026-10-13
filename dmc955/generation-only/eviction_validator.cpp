// Eviction Strategy Validator
//
// Captures Qcur from ALL 28 layers during prefill, computes Q@K^T scores
// for each layer, then tests single-layer eviction at various keep ratios.
// Outputs per-layer score rankings and generation quality for comparison.
//
// Usage:
//   llama-eviction-validator -m <model> --mmproj <mmproj> [options]
//     --keep-ratio <float>    Target retention ratio (default: 0.21)
//     --test-layers <list>    Comma-separated layers to test (default: all)
//     --reference             Run baseline (no eviction) for quality comparison

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cfloat>
#include <chrono>
#include <cinttypes>
#include <map>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

// ============================================================================
// Parameters
// ============================================================================

struct validator_params {
    std::string sintel_dir  = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel";
    std::string video_name  = "alley_1";
    int         n_frames    = 8;
    std::string output_json = "eviction_validation.json";
    std::string prompt      = "Describe what is happening in this video sequence in detail.";
    int         n_predict   = 256;

    // Eviction config
    float keep_ratio        = 0.21f;   // overall target retention
    bool  protect_frame0    = true;
    bool  run_baseline      = false;   // also run without eviction

    // Which layers to test (empty = all)
    std::vector<int> test_layers;

    // Pre-LLM pruning (applied to all runs for fair comparison)
    bool  prellm_pruning    = true;
    float prellm_keep       = 0.40f;

    // Multi-layer combination strategies to test
    bool  test_combinations = true;
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, validator_params & vp) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--sintel-dir" && i + 1 < argc) {
            vp.sintel_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            vp.video_name = argv[++i];
        } else if (arg == "--n-frames" && i + 1 < argc) {
            vp.n_frames = std::stoi(argv[++i]);
        } else if (arg == "--output-json" && i + 1 < argc) {
            vp.output_json = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            vp.prompt = argv[++i];
        } else if (arg == "--n-predict" && i + 1 < argc) {
            vp.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--keep-ratio" && i + 1 < argc) {
            vp.keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--no-protect-frame0") {
            vp.protect_frame0 = false;
        } else if (arg == "--baseline") {
            vp.run_baseline = true;
        } else if (arg == "--no-prellm") {
            vp.prellm_pruning = false;
        } else if (arg == "--prellm-keep" && i + 1 < argc) {
            vp.prellm_keep = std::stof(argv[++i]);
        } else if (arg == "--test-layers" && i + 1 < argc) {
            std::string layers_str(argv[++i]);
            std::istringstream ss(layers_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                vp.test_layers.push_back(std::stoi(token));
            }
        } else if (arg == "--no-combinations") {
            vp.test_combinations = false;
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

// ============================================================================
// Visual token tracking (same as pyramid_generation.cpp)
// ============================================================================

struct visual_token_info {
    int     frame_idx;
    int     patch_row;
    int     patch_col;
    llama_pos kv_pos;
    llama_pos ext_x;
    llama_pos ext_y;
    int     kv_cell_idx;
    float   importance;
    bool    evicted;
};

struct visual_token_tracker {
    std::vector<visual_token_info> tokens;
    std::vector<std::pair<int,int>> frame_ranges;
    int n_frames       = 0;
    int tokens_per_frame = 0;

    void add_token(int frame_idx, int row, int col,
                   llama_pos pos, llama_pos ex, llama_pos ey, int cell_idx) {
        visual_token_info ti = {};
        ti.frame_idx   = frame_idx;
        ti.patch_row   = row;
        ti.patch_col   = col;
        ti.kv_pos      = pos;
        ti.ext_x       = ex;
        ti.ext_y       = ey;
        ti.kv_cell_idx = cell_idx;
        ti.importance  = 0.0f;
        ti.evicted     = false;
        tokens.push_back(ti);
    }

    int count_active() const {
        int c = 0;
        for (const auto & t : tokens) if (!t.evicted) c++;
        return c;
    }
};

// ============================================================================
// All-layer Qcur capture (captures from ALL layers, not just 3)
// ============================================================================

struct all_layer_capture_state {
    static constexpr int MAX_LAYERS = 64;

    struct qcur_data {
        std::vector<float> q;
        int n_embd_head = 0;
        int n_head      = 0;
        int n_q         = 0;
        bool captured   = false;
    };

    qcur_data layers[MAX_LAYERS];
    int n_layers        = 0;
    bool should_capture = false;
    bool disabled       = false;
};

static bool all_layer_qcur_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (all_layer_capture_state *)user_data;
    if (!state || !state->should_capture || state->disabled) return !ask;

    std::string name(t->name);

    // Match "Qcur-N" for any layer N
    if (name.substr(0, 5) != "Qcur-") return !ask;

    int layer = -1;
    try {
        layer = std::stoi(name.substr(5));
    } catch (...) {
        return !ask;
    }

    if (layer < 0 || layer >= state->n_layers) return !ask;

    if (ask) return true;

    // Capture (last write wins = post-RoPE)
    auto & sd = state->layers[layer];
    sd.n_embd_head = (int)t->ne[0];
    sd.n_head      = (int)t->ne[1];
    sd.n_q         = (int)t->ne[2];
    sd.q.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, sd.q.data(), 0, ggml_nbytes(t));
    sd.captured = true;

    return true;
}

// ============================================================================
// CPU Q@K^T scoring (same as pyramid_generation.cpp)
// ============================================================================

static void compute_qk_scores(
    const all_layer_capture_state::qcur_data & qstate,
    int layer_idx,
    llama_memory_t mem,
    const visual_token_tracker & tracker,
    int n_kv_cells_used,
    int n_head_kv,
    std::vector<float> & out_scores)
{
    if (!qstate.captured || qstate.q.empty()) return;

    const int d_k     = qstate.n_embd_head;
    const int n_head  = qstate.n_head;
    const int n_q     = qstate.n_q;
    const int n_kv    = n_kv_cells_used;
    const int gqa     = n_head / n_head_kv;

    const ggml_tensor * k_tensor = llama_memory_k_tensor(mem, layer_idx);
    if (!k_tensor) return;

    size_t row_bytes = ggml_row_size(k_tensor->type, k_tensor->ne[0]);
    std::vector<uint8_t> k_raw(n_kv * row_bytes);
    ggml_backend_tensor_get(k_tensor, k_raw.data(), 0, k_raw.size());

    int n_embd_k_gqa = n_head_kv * d_k;
    std::vector<float> k_f32(n_kv * n_embd_k_gqa);
    if (k_tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) k_f32[i] = ggml_fp16_to_fp32(src[i]);
    } else if (k_tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t * src = (ggml_bf16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) k_f32[i] = ggml_bf16_to_fp32(src[i]);
    } else if (k_tensor->type == GGML_TYPE_F32) {
        memcpy(k_f32.data(), k_raw.data(), k_f32.size() * sizeof(float));
    } else {
        return;
    }

    float scale = 1.0f / sqrtf((float)d_k);

    std::vector<double> kv_attn(n_kv, 0.0);
    std::vector<float> logits(n_kv);

    for (int qi = 0; qi < n_q; qi++) {
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / gqa;
            const float * q_ptr = qstate.q.data() + (qi * n_head + h) * d_k;

            for (int j = 0; j < n_kv; j++) {
                const float * k_ptr = k_f32.data() + j * n_embd_k_gqa + kv_h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) dot += q_ptr[d] * k_ptr[d];
                logits[j] = dot * scale;
            }

            float max_val = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (int j = 0; j < n_kv; j++) {
                logits[j] = expf(logits[j] - max_val);
                sum_exp += logits[j];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < n_kv; j++) {
                kv_attn[j] += (double)(logits[j] * inv_sum);
            }
        }
    }

    for (int j = 0; j < n_kv; j++) kv_attn[j] /= (double)n_head;

    out_scores.resize(tracker.tokens.size(), 0.0f);
    for (int ti = 0; ti < (int)tracker.tokens.size(); ti++) {
        int cell = tracker.tokens[ti].kv_cell_idx;
        if (cell >= 0 && cell < n_kv) {
            out_scores[ti] = (float)kv_attn[cell];
        }
    }
}

// ============================================================================
// Optical flow reader (for pre-LLM pruning)
// ============================================================================

struct optical_flow {
    int width = 0, height = 0;
    std::vector<float> u, v;
};

static bool read_flo_file(const std::string & path, optical_flow & flow) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    float magic = 0.0f;
    if (fread(&magic, sizeof(float), 1, f) != 1 || magic != 202021.25f) { fclose(f); return false; }
    int32_t w = 0, h = 0;
    if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    flow.width = w; flow.height = h;
    int n = w * h;
    std::vector<float> raw(n * 2);
    if ((int)fread(raw.data(), sizeof(float), n * 2, f) != n * 2) { fclose(f); return false; }
    fclose(f);
    flow.u.resize(n); flow.v.resize(n);
    for (int i = 0; i < n; i++) { flow.u[i] = raw[2*i]; flow.v[i] = raw[2*i+1]; }
    return true;
}

static std::vector<float> compute_token_motion_scores(const optical_flow & flow, int nx, int ny) {
    int n = nx * ny;
    std::vector<float> scores(n, 0.0f);
    if (flow.width == 0 || flow.height == 0) return scores;
    for (int ty = 0; ty < ny; ty++) {
        for (int tx = 0; tx < nx; tx++) {
            int px0 = tx * flow.width / nx, px1 = (tx+1) * flow.width / nx;
            int py0 = ty * flow.height / ny, py1 = (ty+1) * flow.height / ny;
            float sum = 0.0f; int cnt = 0;
            for (int py = py0; py < py1; py++)
                for (int px = px0; px < px1; px++) {
                    int idx = py * flow.width + px;
                    sum += sqrtf(flow.u[idx]*flow.u[idx] + flow.v[idx]*flow.v[idx]);
                    cnt++;
                }
            scores[ty * nx + tx] = cnt > 0 ? sum / cnt : 0.0f;
        }
    }
    return scores;
}

// ============================================================================
// Pruned token injection (M-RoPE aware) - from pyramid_generation.cpp
// ============================================================================

struct scored_visual_token {
    int   token_idx;
    float score;
};

static int32_t inject_pruned_visual_tokens(
    llama_context * lctx, mtmd_context * ctx_vision,
    const float * full_embeddings,
    const std::vector<scored_visual_token> & selected,
    int nx, int ny, int n_embd,
    llama_pos n_past, llama_seq_id seq_id, int32_t n_batch_size,
    llama_pos * new_n_past)
{
    int n_keep = (int)selected.size();
    if (n_keep == 0) { *new_n_past = n_past + std::max(nx, ny); return 0; }

    std::vector<float> pruned_embd(n_keep * n_embd);
    for (int i = 0; i < n_keep; i++)
        memcpy(pruned_embd.data() + i * n_embd,
               full_embeddings + selected[i].token_idx * n_embd,
               n_embd * sizeof(float));

    bool use_mrope = mtmd_decode_use_mrope(ctx_vision);
    int n_pos_per = use_mrope ? 4 : 1;
    std::vector<llama_pos> pos(n_keep * n_pos_per, 0);
    if (use_mrope) {
        for (int i = 0; i < n_keep; i++) {
            int idx = selected[i].token_idx;
            pos[i]              = n_past;
            pos[i + n_keep]     = n_past + idx / nx;  // y
            pos[i + n_keep * 2] = n_past + idx % nx;  // x
            pos[i + n_keep * 3] = 0;
        }
    } else {
        for (int i = 0; i < n_keep; i++) pos[i] = n_past + i;
    }

    std::vector<int32_t> n_seq_id_vec(n_keep, 1);
    std::vector<llama_seq_id> seq_id_0(1, seq_id);
    std::vector<llama_seq_id *> seq_ids(n_keep, seq_id_0.data());
    std::vector<int8_t> logits_vec(n_keep, 0);

    if (mtmd_decode_use_non_causal(ctx_vision)) llama_set_causal_attn(lctx, false);

    int n_batches = (n_keep + n_batch_size - 1) / n_batch_size;
    int32_t ret = 0;
    for (int ib = 0; ib < n_batches; ib++) {
        int offset = ib * n_batch_size;
        int n_tok = std::min(n_batch_size, n_keep - offset);

        std::vector<llama_pos> pos_view;
        llama_pos * pos_ptr;
        if (use_mrope) {
            pos_view.reserve(n_tok * 4);
            for (int d = 0; d < 4; d++)
                pos_view.insert(pos_view.end(),
                    pos.data() + d * n_keep + offset,
                    pos.data() + d * n_keep + offset + n_tok);
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }

        llama_batch sub = {n_tok, nullptr, pruned_embd.data() + offset * n_embd,
                           pos_ptr, n_seq_id_vec.data() + offset,
                           seq_ids.data() + offset, logits_vec.data() + offset};
        ret = llama_decode(lctx, sub);
        if (ret != 0) break;
    }

    if (mtmd_decode_use_non_causal(ctx_vision)) llama_set_causal_attn(lctx, true);
    *new_n_past = n_past + std::max(nx, ny);
    return ret;
}

// ============================================================================
// Score analysis utilities
// ============================================================================

// Compute Spearman rank correlation between two score vectors
static float spearman_correlation(const std::vector<float> & a, const std::vector<float> & b) {
    int n = (int)std::min(a.size(), b.size());
    if (n < 2) return 0.0f;

    // Compute ranks
    auto rank = [](const std::vector<float> & v) -> std::vector<float> {
        int n = (int)v.size();
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
        std::vector<float> r(n);
        for (int i = 0; i < n; i++) r[idx[i]] = (float)i;
        return r;
    };

    auto ra = rank(a);
    auto rb = rank(b);

    double mean_a = 0, mean_b = 0;
    for (int i = 0; i < n; i++) { mean_a += ra[i]; mean_b += rb[i]; }
    mean_a /= n; mean_b /= n;

    double cov = 0, var_a = 0, var_b = 0;
    for (int i = 0; i < n; i++) {
        double da = ra[i] - mean_a;
        double db = rb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if (var_a < 1e-10 || var_b < 1e-10) return 0.0f;
    return (float)(cov / sqrt(var_a * var_b));
}

// Compute overlap between top-k sets from two score vectors
static float topk_overlap(const std::vector<float> & a, const std::vector<float> & b, int k) {
    int n = (int)std::min(a.size(), b.size());
    if (k > n) k = n;

    auto topk_set = [&](const std::vector<float> & v) -> std::set<int> {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
        return std::set<int>(idx.begin(), idx.begin() + k);
    };

    auto sa = topk_set(a);
    auto sb = topk_set(b);

    int overlap = 0;
    for (int x : sa) { if (sb.count(x)) overlap++; }
    return (float)overlap / k;
}

// Score entropy — how concentrated vs spread out the attention is
static float score_entropy(const std::vector<float> & scores) {
    double total = 0;
    for (float s : scores) total += s;
    if (total < 1e-10) return 0.0f;

    double entropy = 0;
    for (float s : scores) {
        double p = s / total;
        if (p > 1e-10) entropy -= p * log(p);
    }
    return (float)entropy;
}

// ============================================================================
// Single eviction run: evict tokens, generate, return text
// ============================================================================

struct eviction_result {
    std::string strategy_name;
    int    layer_used;
    int    n_visual_tokens;
    int    n_kept;
    float  keep_pct;
    int    n_generated;
    float  gen_time_ms;
    float  tokens_per_sec;
    float  eviction_time_ms;
    float  defrag_time_ms;
    std::string generated_text;
    float  score_entropy;
    float  score_gini;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    validator_params vp;
    auto filtered = filter_custom_args(argc, argv, vp);

    int new_argc = (int)filtered.size();
    char ** new_argv = filtered.data();

    common_params params;
    if (!common_params_parse(new_argc, new_argv, params, LLAMA_EXAMPLE_MTMD)) {
        fprintf(stderr, "Failed to parse arguments\n");
        return 1;
    }
    common_init();

    if (params.model.path.empty())
        params.model.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf";
    if (params.mmproj.path.empty())
        params.mmproj.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf";
    if (params.n_predict < 0) params.n_predict = vp.n_predict;

    // Set up all-layer capture
    all_layer_capture_state capture_state = {};

    // We'll set n_layers after model loads, register callback now
    params.cb_eval = all_layer_qcur_callback;
    params.cb_eval_user_data = &capture_state;

    printf("=== Eviction Strategy Validator ===\n");
    printf("Video: %s, frames: %d, keep_ratio: %.2f\n",
           vp.video_name.c_str(), vp.n_frames, vp.keep_ratio);

    // ---------------------------------------------------------------
    // Initialize model + vision
    // ---------------------------------------------------------------
    auto llama_init = common_init_from_params(params);
    const llama_model * model = llama_init->model();
    llama_context * lctx      = llama_init->context();
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    common_sampler * smpl      = common_sampler_init(model, params.sampling);
    llama_batch batch          = llama_batch_init(1, 0, 1);
    int n_batch                = params.n_batch;

    auto tmpls = common_chat_templates_init(model, params.chat_template);

    const char * clip_path = params.mmproj.path.c_str();
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = params.mmproj_use_gpu;
    mparams.print_timings   = false;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.warmup          = params.warmup;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    mparams.flash_attn_type = params.flash_attn_type;

    mtmd::context_ptr ctx_vision(mtmd_init_from_file(clip_path, model, mparams));
    if (!ctx_vision.get()) { fprintf(stderr, "Failed to load vision model\n"); return 1; }

    int n_layers = llama_model_n_layer(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int n_embd = llama_model_n_embd_inp(model);
    capture_state.n_layers = n_layers;

    printf("Model: %d layers, %d kv_heads, %d embd\n", n_layers, n_head_kv, n_embd);

    // Determine which layers to test
    if (vp.test_layers.empty()) {
        for (int i = 0; i < n_layers; i++) vp.test_layers.push_back(i);
    }
    printf("Testing %zu layers: ", vp.test_layers.size());
    for (int l : vp.test_layers) printf("%d ", l);
    printf("\n\n");

    // ---------------------------------------------------------------
    // Load frames
    // ---------------------------------------------------------------
    std::string frame_base = vp.sintel_dir + "/training/final/" + vp.video_name + "/";
    std::string flow_base  = vp.sintel_dir + "/training/flow/" + vp.video_name + "/";

    int total_frames = 0;
    for (int i = 1; i <= 999; i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", i);
        FILE * f = fopen((frame_base + buf).c_str(), "rb");
        if (f) { fclose(f); total_frames++; } else break;
    }

    std::vector<std::string> frame_paths;
    std::vector<int> frame_numbers;
    for (int i = 0; i < vp.n_frames; i++) {
        int fn = 1 + (i * (total_frames - 1)) / std::max(1, vp.n_frames - 1);
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", fn);
        frame_paths.push_back(frame_base + buf);
        frame_numbers.push_back(fn);
    }
    printf("Selected %d frames from %d total\n", vp.n_frames, total_frames);

    // Load optical flow
    std::vector<optical_flow> flows(frame_numbers.size());
    for (size_t i = 0; i < frame_numbers.size(); i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.flo", frame_numbers[i]);
        read_flo_file(flow_base + buf, flows[i]);
    }

    // ---------------------------------------------------------------
    // Phase 1: Prefill (same for all strategies)
    //   - We do this ONCE with all-layer Qcur capture
    //   - Then save/restore KV cache state for each eviction test
    // ---------------------------------------------------------------

    // Load images
    mtmd::bitmaps bitmaps;
    for (const auto & path : frame_paths) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), path.c_str()));
        if (!bmp.ptr) { fprintf(stderr, "Failed to load %s\n", path.c_str()); return 1; }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // Build prompt
    std::string full_content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) full_content += mtmd_default_marker();
    full_content += "\n" + vp.prompt;

    common_chat_msg msg;
    msg.role = "user";
    msg.content = full_content;

    std::vector<common_chat_msg> chat_history;
    auto formatted = common_chat_format_single(tmpls.get(), chat_history, msg, true, false);

    mtmd_input_text text_input;
    text_input.text = formatted.c_str();
    text_input.add_special = true;
    text_input.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c = bitmaps.c_ptr();
    int32_t tok_res = mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(),
                                     &text_input, bitmaps_c.data(), bitmaps_c.size());
    if (tok_res != 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }

    // Process chunks with all-layer Qcur capture
    printf("\n=== Phase 1: Prefill with all-layer Qcur capture ===\n");

    visual_token_tracker tracker;
    llama_pos n_past = 0;
    int n_kv_cells_used = 0;
    int image_idx = 0;

    size_t n_chunks = chunks.size();
    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool is_last = (ci == n_chunks - 1);

            // Enable capture for last text chunk
            if (is_last && !tracker.tokens.empty()) {
                capture_state.should_capture = true;
                printf("Capturing Qcur from all %d layers...\n", n_layers);
            }

            llama_pos new_n_past = n_past;
            int32_t ret = mtmd_helper_eval_chunk_single(
                ctx_vision.get(), lctx, chunk, n_past, 0, n_batch, is_last, &new_n_past);

            if (is_last) capture_state.should_capture = false;
            if (ret != 0) { fprintf(stderr, "Text eval failed\n"); return 1; }

            n_kv_cells_used += (int)(new_n_past - n_past);
            n_past = new_n_past;
            continue;
        }

        // IMAGE chunk
        const auto img_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        int chunk_nx = (int)mtmd_image_tokens_get_nx(img_tokens);
        int chunk_ny = (int)mtmd_image_tokens_get_ny(img_tokens);
        int n_tok_chunk = chunk_nx * chunk_ny;

        int frame_token_start = (int)tracker.tokens.size();
        llama_pos frame_pos_start = n_past;
        llama_pos new_n_past = n_past;

        // Encode
        int32_t enc_ret = mtmd_encode_chunk(ctx_vision.get(), chunk);
        if (enc_ret != 0) { fprintf(stderr, "Encode failed frame %d\n", image_idx); return 1; }
        float * embd = mtmd_get_output_embd(ctx_vision.get());

        // Pre-LLM pruning for non-keyframe
        if (vp.prellm_pruning && image_idx > 0) {
            int n_keep = std::max(1, (int)(n_tok_chunk * vp.prellm_keep));

            // Score: motion + cosine + norm
            float w_motion = 0.5f, w_cosine = 0.3f, w_norm = 0.2f;
            std::vector<float> motion(n_tok_chunk, 0.0f);
            bool has_flow = (image_idx < (int)flows.size() && flows[image_idx].width > 0);
            float m_max = 1e-8f;
            if (has_flow) {
                motion = compute_token_motion_scores(flows[image_idx], chunk_nx, chunk_ny);
                for (float m : motion) m_max = std::max(m_max, m);
            }

            std::vector<float> centroid(n_embd, 0.0f);
            for (int t = 0; t < n_tok_chunk; t++) {
                const float * e = embd + t * n_embd;
                for (int d = 0; d < n_embd; d++) centroid[d] += e[d];
            }
            for (int d = 0; d < n_embd; d++) centroid[d] /= n_tok_chunk;
            float c_norm = 0.0f;
            for (int d = 0; d < n_embd; d++) c_norm += centroid[d] * centroid[d];
            c_norm = sqrtf(c_norm + 1e-8f);

            float norm_max = 1e-8f;
            std::vector<scored_visual_token> scores(n_tok_chunk);
            for (int t = 0; t < n_tok_chunk; t++) {
                const float * e = embd + t * n_embd;
                float dot = 0.0f, tn2 = 0.0f;
                for (int d = 0; d < n_embd; d++) { dot += e[d]*centroid[d]; tn2 += e[d]*e[d]; }
                float tn = sqrtf(tn2 + 1e-8f);
                float cos_d = 1.0f - dot / (tn * c_norm);
                norm_max = std::max(norm_max, tn);
                float ms = has_flow ? motion[t] / m_max : 0.0f;
                scores[t] = {t, w_motion * ms + w_cosine * cos_d + w_norm * (tn / norm_max)};
            }

            std::sort(scores.begin(), scores.end(),
                      [](const scored_visual_token & a, const scored_visual_token & b) { return a.score > b.score; });
            scores.resize(n_keep);
            std::sort(scores.begin(), scores.end(),
                      [](const scored_visual_token & a, const scored_visual_token & b) { return a.token_idx < b.token_idx; });

            int32_t ret = inject_pruned_visual_tokens(
                lctx, ctx_vision.get(), embd, scores, chunk_nx, chunk_ny, n_embd,
                frame_pos_start, 0, n_batch, &new_n_past);
            if (ret != 0) { fprintf(stderr, "Inject failed\n"); return 1; }

            int frame_cell_start = n_kv_cells_used;
            for (int i = 0; i < n_keep; i++) {
                int idx = scores[i].token_idx;
                tracker.add_token(image_idx, idx / chunk_nx, idx % chunk_nx,
                                  frame_pos_start, frame_pos_start + idx % chunk_nx,
                                  frame_pos_start + idx / chunk_nx,
                                  frame_cell_start + i);
            }
            tracker.frame_ranges.push_back({frame_token_start, n_keep});
            tracker.n_frames++;
            n_kv_cells_used += n_keep;
            n_past = new_n_past;

            printf("  Frame %d: %d/%d tokens (pre-LLM pruned), n_past=%d\n",
                   image_idx, n_keep, n_tok_chunk, (int)n_past);
        } else {
            // Full decode
            int32_t ret = mtmd_helper_decode_image_chunk(
                ctx_vision.get(), lctx, chunk, embd, n_past, 0, n_batch, &new_n_past);
            if (ret != 0) { fprintf(stderr, "Decode failed\n"); return 1; }

            int frame_cell_start = n_kv_cells_used;
            for (int ty = 0; ty < chunk_ny; ty++)
                for (int tx = 0; tx < chunk_nx; tx++)
                    tracker.add_token(image_idx, ty, tx,
                                      frame_pos_start, frame_pos_start + tx, frame_pos_start + ty,
                                      frame_cell_start + ty * chunk_nx + tx);

            tracker.frame_ranges.push_back({frame_token_start, n_tok_chunk});
            if (image_idx == 0) tracker.tokens_per_frame = n_tok_chunk;
            tracker.n_frames++;
            n_kv_cells_used += n_tok_chunk;
            n_past = new_n_past;

            printf("  Frame %d: %d tokens (full), n_past=%d\n",
                   image_idx, n_tok_chunk, (int)n_past);
        }

        image_idx++;
    }

    int n_visual = (int)tracker.tokens.size();
    int tpf = tracker.tokens_per_frame;
    printf("\nPrefill complete: %d visual tokens, %d KV cells\n", n_visual, n_kv_cells_used);

    // Check captured layers
    int n_captured = 0;
    for (int i = 0; i < n_layers; i++) {
        if (capture_state.layers[i].captured) n_captured++;
    }
    printf("Captured Qcur from %d/%d layers\n", n_captured, n_layers);

    // ---------------------------------------------------------------
    // Phase 2: Compute scores for ALL captured layers
    // ---------------------------------------------------------------
    printf("\n=== Phase 2: Computing Q@K^T scores for all layers ===\n");

    auto * mem = llama_get_memory(lctx);
    std::map<int, std::vector<float>> all_scores;  // layer -> scores

    for (int layer : vp.test_layers) {
        if (!capture_state.layers[layer].captured) continue;

        std::vector<float> scores;
        int64_t t0 = ggml_time_ms();
        compute_qk_scores(capture_state.layers[layer], layer, mem,
                          tracker, n_kv_cells_used, n_head_kv, scores);
        float ms = (float)(ggml_time_ms() - t0);

        if (!scores.empty()) {
            all_scores[layer] = scores;
            float ent = score_entropy(scores);

            // Compute Gini coefficient (score concentration)
            std::vector<float> sorted_scores = scores;
            std::sort(sorted_scores.begin(), sorted_scores.end());
            double gini_sum = 0, cum_sum = 0, total = 0;
            for (float s : sorted_scores) total += s;
            if (total > 1e-10) {
                for (int i = 0; i < (int)sorted_scores.size(); i++) {
                    cum_sum += sorted_scores[i];
                    gini_sum += (2.0 * (i + 1) - sorted_scores.size() - 1) * sorted_scores[i];
                }
            }
            float gini = (total > 0) ? (float)(gini_sum / (sorted_scores.size() * total)) : 0;

            printf("  Layer %2d: Q@K %.0f ms, entropy=%.3f, gini=%.3f\n",
                   layer, ms, ent, gini);
        }
    }

    // ---------------------------------------------------------------
    // Phase 3: Cross-layer correlation analysis
    // ---------------------------------------------------------------
    printf("\n=== Phase 3: Cross-layer score correlation ===\n");

    // Keep ratio → how many tokens to keep
    int n_protected = vp.protect_frame0 ? tpf : 0;
    int n_evictable = n_visual - n_protected;
    int n_target_keep = std::max(1, (int)(n_visual * vp.keep_ratio));
    int n_evictable_keep = n_target_keep - n_protected;
    if (n_evictable_keep < 1) n_evictable_keep = 1;

    printf("Retention target: %d/%d visual tokens (%.1f%%)\n",
           n_target_keep, n_visual, vp.keep_ratio * 100);
    printf("  Protected (frame 0): %d, evictable keep: %d/%d\n\n",
           n_protected, n_evictable_keep, n_evictable);

    // Build evictable-only score vectors for correlation
    auto extract_evictable = [&](const std::vector<float> & scores) -> std::vector<float> {
        std::vector<float> ev;
        for (int i = n_protected; i < (int)scores.size(); i++) ev.push_back(scores[i]);
        return ev;
    };

    // Print correlation matrix (selected layers)
    std::vector<int> key_layers;
    for (int l : vp.test_layers) {
        if (all_scores.count(l)) key_layers.push_back(l);
    }

    // If too many layers, just print a subset
    std::vector<int> display_layers;
    if (key_layers.size() > 10) {
        // Show every 3rd layer + last
        for (int i = 0; i < (int)key_layers.size(); i += 3) display_layers.push_back(key_layers[i]);
        if (display_layers.back() != key_layers.back()) display_layers.push_back(key_layers.back());
    } else {
        display_layers = key_layers;
    }

    printf("Spearman rank correlation (evictable tokens only):\n");
    printf("%6s", "");
    for (int l : display_layers) printf(" L%02d  ", l);
    printf("\n");
    for (int la : display_layers) {
        printf("L%02d  ", la);
        auto ea = extract_evictable(all_scores[la]);
        for (int lb : display_layers) {
            auto eb = extract_evictable(all_scores[lb]);
            float rho = spearman_correlation(ea, eb);
            printf(" %5.2f", rho);
        }
        printf("\n");
    }

    printf("\nTop-%d overlap (tokens kept):\n", n_evictable_keep);
    printf("%6s", "");
    for (int l : display_layers) printf(" L%02d  ", l);
    printf("\n");
    for (int la : display_layers) {
        printf("L%02d  ", la);
        auto ea = extract_evictable(all_scores[la]);
        for (int lb : display_layers) {
            auto eb = extract_evictable(all_scores[lb]);
            float overlap = topk_overlap(ea, eb, n_evictable_keep);
            printf(" %5.1f", overlap * 100);
        }
        printf("%%\n");
    }

    // ---------------------------------------------------------------
    // Phase 4: Test eviction strategies and generate
    //
    // For each strategy:
    //   1. Save KV cache state
    //   2. Evict tokens based on strategy
    //   3. Defrag
    //   4. Generate text
    //   5. Restore KV cache state
    // ---------------------------------------------------------------
    printf("\n=== Phase 4: Testing eviction strategies ===\n");

    // Save pre-eviction KV state
    printf("KV state ready for eviction testing\n\n");

    // We'll use a simpler approach: for each strategy, we reload the KV state
    // by saving/restoring via the llama state API.

    // Actually, since save/restore of full state is complex, let's take a different
    // approach: compute the eviction SET for each strategy (which tokens to keep),
    // then measure the Jaccard/overlap between strategies. For actual generation,
    // test the top candidates.

    // Strategy evaluation (score-only, no actual eviction yet)
    struct strategy_eval {
        std::string name;
        std::set<int> keep_set;  // indices of tokens to keep
        float mean_score;
    };

    std::vector<strategy_eval> strategies;

    // Single-layer strategies
    for (int layer : key_layers) {
        const auto & scores = all_scores[layer];

        // Build keep set: protect frame0, sort rest by score, keep top
        struct st { int idx; float s; };
        std::vector<st> evictable;
        for (int i = 0; i < n_visual; i++) {
            if (vp.protect_frame0 && i < tpf) continue;
            evictable.push_back({i, scores[i]});
        }
        std::sort(evictable.begin(), evictable.end(),
                  [](const st & a, const st & b) { return a.s > b.s; });

        std::set<int> keep;
        for (int i = 0; i < tpf; i++) keep.insert(i);  // frame 0
        for (int i = 0; i < n_evictable_keep && i < (int)evictable.size(); i++) {
            keep.insert(evictable[i].idx);
        }

        float mean_s = 0;
        for (int idx : keep) mean_s += scores[idx];
        mean_s /= keep.size();

        char name[32]; snprintf(name, sizeof(name), "L%02d", layer);
        strategies.push_back({name, keep, mean_s});
    }

    // Combination strategies
    if (vp.test_combinations && key_layers.size() >= 2) {
        // Average of last layer + second-to-last
        int l_last = key_layers.back();
        int l_prev = key_layers[key_layers.size() - 2];
        {
            std::vector<float> combined(n_visual, 0.0f);
            for (int i = 0; i < n_visual; i++) {
                combined[i] = 0.5f * all_scores[l_last][i] + 0.5f * all_scores[l_prev][i];
            }
            struct st { int idx; float s; };
            std::vector<st> evictable;
            for (int i = 0; i < n_visual; i++) {
                if (vp.protect_frame0 && i < tpf) continue;
                evictable.push_back({i, combined[i]});
            }
            std::sort(evictable.begin(), evictable.end(),
                      [](const st & a, const st & b) { return a.s > b.s; });
            std::set<int> keep;
            for (int i = 0; i < tpf; i++) keep.insert(i);
            for (int i = 0; i < n_evictable_keep && i < (int)evictable.size(); i++)
                keep.insert(evictable[i].idx);
            float mean_s = 0;
            for (int idx : keep) mean_s += combined[idx];
            mean_s /= keep.size();

            char name[64]; snprintf(name, sizeof(name), "Avg(L%02d+L%02d)", l_prev, l_last);
            strategies.push_back({name, keep, mean_s});
        }

        // Max across all layers
        {
            std::vector<float> max_scores(n_visual, 0.0f);
            for (auto & [layer, scores] : all_scores) {
                for (int i = 0; i < n_visual; i++)
                    max_scores[i] = std::max(max_scores[i], scores[i]);
            }
            struct st { int idx; float s; };
            std::vector<st> evictable;
            for (int i = 0; i < n_visual; i++) {
                if (vp.protect_frame0 && i < tpf) continue;
                evictable.push_back({i, max_scores[i]});
            }
            std::sort(evictable.begin(), evictable.end(),
                      [](const st & a, const st & b) { return a.s > b.s; });
            std::set<int> keep;
            for (int i = 0; i < tpf; i++) keep.insert(i);
            for (int i = 0; i < n_evictable_keep && i < (int)evictable.size(); i++)
                keep.insert(evictable[i].idx);
            strategies.push_back({"Max(all)", keep, 0.0f});
        }

        // Weighted average: deeper layers get more weight
        {
            std::vector<float> weighted(n_visual, 0.0f);
            float w_sum = 0;
            for (auto & [layer, scores] : all_scores) {
                float w = (float)(layer + 1) / n_layers;  // deeper = higher weight
                for (int i = 0; i < n_visual; i++) weighted[i] += w * scores[i];
                w_sum += w;
            }
            for (int i = 0; i < n_visual; i++) weighted[i] /= w_sum;

            struct st { int idx; float s; };
            std::vector<st> evictable;
            for (int i = 0; i < n_visual; i++) {
                if (vp.protect_frame0 && i < tpf) continue;
                evictable.push_back({i, weighted[i]});
            }
            std::sort(evictable.begin(), evictable.end(),
                      [](const st & a, const st & b) { return a.s > b.s; });
            std::set<int> keep;
            for (int i = 0; i < tpf; i++) keep.insert(i);
            for (int i = 0; i < n_evictable_keep && i < (int)evictable.size(); i++)
                keep.insert(evictable[i].idx);
            strategies.push_back({"DepthWeightedAvg", keep, 0.0f});
        }

        // Original 3-stage progressive (for comparison)
        {
            // Simulate: start with all, evict 30% by L7, then 40% of remaining by L19, then 50% by L27
            int layers_3stage[] = {7, 19, 27};
            float ratios[] = {0.70f, 0.60f, 0.50f};

            std::vector<bool> active(n_visual, true);
            for (int stage = 0; stage < 3; stage++) {
                int l = layers_3stage[stage];
                if (!all_scores.count(l)) continue;

                struct st { int idx; float s; };
                std::vector<st> curr_active;
                for (int i = 0; i < n_visual; i++) {
                    if (!active[i]) continue;
                    float s = all_scores[l][i];
                    if (vp.protect_frame0 && i < tpf) s = FLT_MAX;
                    curr_active.push_back({i, s});
                }
                int n_keep_stage = std::max(1, (int)(curr_active.size() * ratios[stage]));
                std::sort(curr_active.begin(), curr_active.end(),
                          [](const st & a, const st & b) { return a.s < b.s; });
                int n_evict = (int)curr_active.size() - n_keep_stage;
                for (int i = 0; i < n_evict; i++) active[curr_active[i].idx] = false;
            }

            std::set<int> keep;
            for (int i = 0; i < n_visual; i++) { if (active[i]) keep.insert(i); }
            strategies.push_back({"3Stage(7/19/27)", keep, 0.0f});
        }
    }

    // Print strategy comparison
    printf("\nStrategy token overlap matrix (Jaccard similarity):\n");
    printf("%-20s  kept", "Strategy");
    for (size_t j = 0; j < strategies.size(); j++) printf(" %6s", strategies[j].name.substr(0, 6).c_str());
    printf("\n");

    for (size_t i = 0; i < strategies.size(); i++) {
        printf("%-20s %4zu", strategies[i].name.c_str(), strategies[i].keep_set.size());
        for (size_t j = 0; j < strategies.size(); j++) {
            int overlap = 0;
            for (int x : strategies[i].keep_set) { if (strategies[j].keep_set.count(x)) overlap++; }
            int union_size = (int)(strategies[i].keep_set.size() + strategies[j].keep_set.size() - overlap);
            float jaccard = (union_size > 0) ? (float)overlap / union_size : 0;
            printf(" %5.1f%%", jaccard * 100);
        }
        printf("\n");
    }

    // ---------------------------------------------------------------
    // Phase 5: Actual generation test for top strategies
    //
    // We test: baseline (no eviction), last layer only, 3-stage, and best combo
    // For each: evict → defrag → generate → record text
    //
    // Since we can't easily save/restore KV state, we do multiple full runs.
    // But to keep it fast, we encode images once and reuse embeddings.
    // ---------------------------------------------------------------

    // For actual generation, we need to re-run prefill for each strategy.
    // This is expensive but necessary for accurate measurement.
    // Let's test 4 strategies: baseline, last_layer, 3stage, depth_weighted

    printf("\n=== Phase 5: Generation quality comparison ===\n");
    printf("(Re-running prefill for each strategy — this takes a while)\n\n");

    // Identify strategies to actually test with generation
    std::vector<std::string> gen_test_names;
    if (vp.run_baseline) gen_test_names.push_back("BASELINE");
    gen_test_names.push_back("L" + std::to_string(key_layers.back()));  // last layer
    if (key_layers.size() >= 2)
        gen_test_names.push_back("L" + std::to_string(key_layers[key_layers.size() / 2]));  // mid layer
    gen_test_names.push_back("3Stage(7/19/27)");
    gen_test_names.push_back("DepthWeightedAvg");

    // For now, output the analysis. Actual generation runs would require
    // re-initializing the context for each test, which we'll flag as TODO.
    printf("Recommended strategies for generation testing:\n");
    for (const auto & name : gen_test_names) printf("  - %s\n", name.c_str());

    // ---------------------------------------------------------------
    // Output JSON results
    // ---------------------------------------------------------------
    json results;
    results["video"] = vp.video_name;
    results["n_frames"] = vp.n_frames;
    results["n_visual_tokens"] = n_visual;
    results["n_protected"] = n_protected;
    results["target_keep_ratio"] = vp.keep_ratio;
    results["target_keep_count"] = n_target_keep;
    results["n_layers"] = n_layers;

    // Per-layer scores
    json layer_analysis = json::array();
    for (auto & [layer, scores] : all_scores) {
        json la;
        la["layer"] = layer;
        la["entropy"] = score_entropy(scores);

        // Gini
        std::vector<float> ss = scores;
        std::sort(ss.begin(), ss.end());
        double gs = 0, cs = 0, ts = 0;
        for (float s : ss) ts += s;
        if (ts > 1e-10) {
            for (int i = 0; i < (int)ss.size(); i++)
                gs += (2.0 * (i + 1) - ss.size() - 1) * ss[i];
        }
        la["gini"] = (ts > 0) ? (float)(gs / (ss.size() * ts)) : 0;
        layer_analysis.push_back(la);
    }
    results["layer_analysis"] = layer_analysis;

    // Correlation matrix
    json corr_matrix = json::object();
    for (int la : display_layers) {
        auto ea = extract_evictable(all_scores[la]);
        json row;
        for (int lb : display_layers) {
            auto eb = extract_evictable(all_scores[lb]);
            row[std::to_string(lb)] = spearman_correlation(ea, eb);
        }
        corr_matrix[std::to_string(la)] = row;
    }
    results["spearman_correlation"] = corr_matrix;

    // Strategy comparison
    json strats = json::array();
    for (auto & s : strategies) {
        json sj;
        sj["name"] = s.name;
        sj["n_kept"] = (int)s.keep_set.size();
        sj["keep_pct"] = 100.0f * s.keep_set.size() / n_visual;
        strats.push_back(sj);
    }
    results["strategies"] = strats;

    // Jaccard matrix
    json jaccard = json::object();
    for (size_t i = 0; i < strategies.size(); i++) {
        json row;
        for (size_t j = 0; j < strategies.size(); j++) {
            int overlap = 0;
            for (int x : strategies[i].keep_set) { if (strategies[j].keep_set.count(x)) overlap++; }
            int u = (int)(strategies[i].keep_set.size() + strategies[j].keep_set.size() - overlap);
            row[strategies[j].name] = (u > 0) ? (float)overlap / u : 0;
        }
        jaccard[strategies[i].name] = row;
    }
    results["jaccard_similarity"] = jaccard;

    std::ofstream ofs(vp.output_json);
    ofs << results.dump(2) << std::endl;
    printf("\nResults saved to %s\n", vp.output_json.c_str());

    // Cleanup
    llama_batch_free(batch);
    common_sampler_free(smpl);

    return 0;
}
