// Scoring Strategy Sweep
//
// Tests which tokens to keep from non-keyframes:
//   1. MOTION     — pure optical flow magnitude (keep moving patches)
//   2. STATIC     — inverse motion (keep static patches) [negative control]
//   3. COSINE_KF  — cosine distance from keyframe embeddings (keep patches that CHANGED from frame 0)
//   4. COSINE_CTR — cosine distance from frame centroid (keep outlier patches)
//   5. NORM       — embedding L2 norm (keep high-energy patches)
//   6. BLEND      — current 50/30/20 mix
//   7. RANDOM     — uniform random [baseline control]
//   8. MOTION+KF  — 50% motion + 50% cosine-from-keyframe
//
// Loads model once, runs each strategy at a fixed keep ratio, compares outputs.

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
#include <chrono>
#include <cinttypes>
#include <cfloat>
#include <sstream>
#include <map>
#include <random>
#include <set>

using json = nlohmann::ordered_json;

// ============================================================================
// Scoring strategies
// ============================================================================

enum scoring_strategy {
    SCORE_MOTION,       // pure optical flow
    SCORE_STATIC,       // inverse motion (negative control)
    SCORE_COSINE_KF,    // cosine distance from keyframe embedding
    SCORE_COSINE_CTR,   // cosine distance from frame centroid
    SCORE_NORM,         // embedding L2 norm
    SCORE_BLEND,        // current 50/30/20 (motion/cosine_ctr/norm)
    SCORE_RANDOM,       // random baseline
    SCORE_MOTION_KF,    // 50% motion + 50% cosine-from-keyframe
    SCORE_COUNT
};

static const char * strategy_name(scoring_strategy s) {
    switch (s) {
        case SCORE_MOTION:     return "MOTION";
        case SCORE_STATIC:     return "STATIC";
        case SCORE_COSINE_KF:  return "COSINE_KF";
        case SCORE_COSINE_CTR: return "COSINE_CTR";
        case SCORE_NORM:       return "NORM";
        case SCORE_BLEND:      return "BLEND";
        case SCORE_RANDOM:     return "RANDOM";
        case SCORE_MOTION_KF:  return "MOTION+KF";
        default: return "?";
    }
}

// ============================================================================
// Parameters
// ============================================================================

struct sweep_params {
    std::string sintel_dir  = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel";
    std::string video_name  = "alley_1";
    int         n_frames    = 8;
    std::string output_json = "scoring_sweep.json";
    std::string prompt      = "Describe what is happening in this video sequence in detail.";
    int         n_predict   = 256;
    float       keep_ratio  = 0.30f;
    bool        protect_frame0 = true;
};

static std::vector<char *> filter_args(int argc, char ** argv, sweep_params & sp) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--sintel-dir" && i + 1 < argc) sp.sintel_dir = argv[++i];
        else if (arg == "--video" && i + 1 < argc) sp.video_name = argv[++i];
        else if (arg == "--n-frames" && i + 1 < argc) sp.n_frames = std::stoi(argv[++i]);
        else if (arg == "--output-json" && i + 1 < argc) sp.output_json = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) sp.prompt = argv[++i];
        else if (arg == "--n-predict" && i + 1 < argc) sp.n_predict = std::stoi(argv[++i]);
        else if (arg == "--keep-ratio" && i + 1 < argc) sp.keep_ratio = std::stof(argv[++i]);
        else filtered.push_back(argv[i]);
    }
    return filtered;
}

// ============================================================================
// Optical flow helpers
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

static std::vector<float> compute_token_motion(const optical_flow & flow, int nx, int ny) {
    int n = nx * ny;
    std::vector<float> scores(n, 0.0f);
    if (flow.width == 0) return scores;
    for (int ty = 0; ty < ny; ty++)
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
    return scores;
}

// ============================================================================
// Pruned token injection (M-RoPE aware)
// ============================================================================

struct scored_token {
    int idx;
    float score;
};

static int32_t inject_pruned_tokens(
    llama_context * lctx, mtmd_context * vctx,
    const float * embd, const std::vector<scored_token> & sel,
    int nx, int ny, int n_embd,
    llama_pos n_past, int32_t n_batch, llama_pos * new_past)
{
    int n = (int)sel.size();
    if (n == 0) { *new_past = n_past + std::max(nx, ny); return 0; }

    std::vector<float> buf(n * n_embd);
    for (int i = 0; i < n; i++)
        memcpy(buf.data() + i * n_embd, embd + sel[i].idx * n_embd, n_embd * sizeof(float));

    bool mrope = mtmd_decode_use_mrope(vctx);
    int npp = mrope ? 4 : 1;
    std::vector<llama_pos> pos(n * npp, 0);
    if (mrope) {
        for (int i = 0; i < n; i++) {
            int gx = sel[i].idx % nx, gy = sel[i].idx / nx;
            pos[i]         = n_past;
            pos[i + n]     = n_past + gy;
            pos[i + n * 2] = n_past + gx;
            pos[i + n * 3] = 0;
        }
    } else {
        for (int i = 0; i < n; i++) pos[i] = n_past + i;
    }

    std::vector<int32_t> nsi(n, 1);
    std::vector<llama_seq_id> s0(1, 0);
    std::vector<llama_seq_id *> sids(n, s0.data());
    std::vector<int8_t> logits(n, 0);

    if (mtmd_decode_use_non_causal(vctx)) llama_set_causal_attn(lctx, false);

    int32_t ret = 0;
    for (int ib = 0; ib < (n + n_batch - 1) / n_batch; ib++) {
        int off = ib * n_batch;
        int cnt = std::min(n_batch, n - off);
        llama_pos * pp;
        std::vector<llama_pos> pv;
        if (mrope) {
            pv.reserve(cnt * 4);
            for (int d = 0; d < 4; d++)
                pv.insert(pv.end(), pos.data() + d*n + off, pos.data() + d*n + off + cnt);
            pp = pv.data();
        } else { pp = pos.data() + off; }
        llama_batch b = {cnt, nullptr, buf.data() + off * n_embd, pp,
                         nsi.data() + off, sids.data() + off, logits.data() + off};
        ret = llama_decode(lctx, b);
        if (ret != 0) break;
    }
    if (mtmd_decode_use_non_causal(vctx)) llama_set_causal_attn(lctx, true);
    *new_past = n_past + std::max(nx, ny);
    return ret;
}

// ============================================================================
// Score tokens by strategy
// ============================================================================

static std::vector<scored_token> score_tokens(
    scoring_strategy strategy,
    const float * embd,           // current frame embeddings [n_tok * n_embd]
    const float * kf_embd,        // keyframe embeddings (frame 0) [n_tok * n_embd], may be null
    int n_tok, int n_embd,
    const optical_flow & flow,
    int nx, int ny,
    float keep_ratio,
    int seed = 42)
{
    int n_keep = std::max(1, (int)(n_tok * keep_ratio));

    std::vector<float> raw_scores(n_tok, 0.0f);

    // --- Motion scores ---
    std::vector<float> motion(n_tok, 0.0f);
    bool has_flow = flow.width > 0;
    float m_max = 1e-8f;
    if (has_flow) {
        motion = compute_token_motion(flow, nx, ny);
        for (float m : motion) m_max = std::max(m_max, m);
    }

    // --- Centroid + per-token stats ---
    std::vector<float> centroid(n_embd, 0.0f);
    for (int t = 0; t < n_tok; t++) {
        const float * e = embd + t * n_embd;
        for (int d = 0; d < n_embd; d++) centroid[d] += e[d];
    }
    for (int d = 0; d < n_embd; d++) centroid[d] /= n_tok;
    float c_norm_sq = 0;
    for (int d = 0; d < n_embd; d++) c_norm_sq += centroid[d] * centroid[d];
    float c_norm = sqrtf(c_norm_sq + 1e-8f);

    // Per-token: cosine to centroid, cosine to keyframe, norm
    std::vector<float> cos_ctr(n_tok), cos_kf(n_tok), norms(n_tok);
    float norm_max = 1e-8f;
    float cos_kf_max = 1e-8f;

    for (int t = 0; t < n_tok; t++) {
        const float * e = embd + t * n_embd;
        float dot_c = 0, dot_kf = 0, tn2 = 0, kf_n2 = 0;
        for (int d = 0; d < n_embd; d++) {
            dot_c += e[d] * centroid[d];
            tn2 += e[d] * e[d];
        }
        float tn = sqrtf(tn2 + 1e-8f);
        cos_ctr[t] = 1.0f - dot_c / (tn * c_norm);  // cosine distance
        norms[t] = tn;
        norm_max = std::max(norm_max, tn);

        // Cosine distance from keyframe embedding at same spatial position
        if (kf_embd) {
            const float * kf_e = kf_embd + t * n_embd;
            for (int d = 0; d < n_embd; d++) {
                dot_kf += e[d] * kf_e[d];
                kf_n2 += kf_e[d] * kf_e[d];
            }
            float kf_n = sqrtf(kf_n2 + 1e-8f);
            cos_kf[t] = 1.0f - dot_kf / (tn * kf_n);  // high = changed from keyframe
            cos_kf_max = std::max(cos_kf_max, cos_kf[t]);
        }
    }

    // --- Compute strategy-specific scores ---
    switch (strategy) {
        case SCORE_MOTION:
            for (int t = 0; t < n_tok; t++)
                raw_scores[t] = has_flow ? motion[t] / m_max : 0.0f;
            break;

        case SCORE_STATIC:
            for (int t = 0; t < n_tok; t++)
                raw_scores[t] = has_flow ? 1.0f - motion[t] / m_max : 1.0f;
            break;

        case SCORE_COSINE_KF:
            if (kf_embd) {
                for (int t = 0; t < n_tok; t++)
                    raw_scores[t] = cos_kf[t] / cos_kf_max;
            }
            break;

        case SCORE_COSINE_CTR:
            for (int t = 0; t < n_tok; t++)
                raw_scores[t] = cos_ctr[t];
            break;

        case SCORE_NORM:
            for (int t = 0; t < n_tok; t++)
                raw_scores[t] = norms[t] / norm_max;
            break;

        case SCORE_BLEND:
            for (int t = 0; t < n_tok; t++)
                raw_scores[t] = 0.5f * (has_flow ? motion[t] / m_max : 0.0f)
                              + 0.3f * cos_ctr[t]
                              + 0.2f * (norms[t] / norm_max);
            break;

        case SCORE_RANDOM: {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int t = 0; t < n_tok; t++) raw_scores[t] = dist(rng);
            break;
        }

        case SCORE_MOTION_KF:
            if (kf_embd) {
                for (int t = 0; t < n_tok; t++)
                    raw_scores[t] = 0.5f * (has_flow ? motion[t] / m_max : 0.0f)
                                  + 0.5f * (cos_kf[t] / cos_kf_max);
            } else {
                for (int t = 0; t < n_tok; t++)
                    raw_scores[t] = has_flow ? motion[t] / m_max : 0.0f;
            }
            break;

        default: break;
    }

    // Spatial grid guarantee (4x4)
    int grid = 4;
    std::vector<bool> force_keep(n_tok, false);
    for (int gy = 0; gy < grid; gy++)
        for (int gx = 0; gx < grid; gx++) {
            int best = -1; float best_s = -1;
            int y0 = gy * ny / grid, y1 = (gy+1) * ny / grid;
            int x0 = gx * nx / grid, x1 = (gx+1) * nx / grid;
            for (int ty = y0; ty < y1; ty++)
                for (int tx = x0; tx < x1; tx++) {
                    int i = ty * nx + tx;
                    if (raw_scores[i] > best_s) { best_s = raw_scores[i]; best = i; }
                }
            if (best >= 0) force_keep[best] = true;
        }

    // Select top-k
    std::vector<scored_token> forced, candidates;
    for (int t = 0; t < n_tok; t++) {
        scored_token st = {t, raw_scores[t]};
        if (force_keep[t]) forced.push_back(st);
        else candidates.push_back(st);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const scored_token & a, const scored_token & b) { return a.score > b.score; });

    std::vector<scored_token> selected = forced;
    int remaining = n_keep - (int)forced.size();
    for (int i = 0; i < remaining && i < (int)candidates.size(); i++)
        selected.push_back(candidates[i]);

    std::sort(selected.begin(), selected.end(),
              [](const scored_token & a, const scored_token & b) { return a.idx < b.idx; });

    return selected;
}

// ============================================================================
// Text similarity
// ============================================================================

static std::vector<std::string> tokenize_words(const std::string & text) {
    std::vector<std::string> words;
    std::istringstream ss(text);
    std::string w;
    while (ss >> w) {
        std::string clean;
        for (char c : w) if (std::isalpha(c)) clean += std::tolower(c);
        if (!clean.empty()) words.push_back(clean);
    }
    return words;
}

static float rouge1_f1(const std::string & ref, const std::string & hyp) {
    auto rw = tokenize_words(ref), hw = tokenize_words(hyp);
    if (rw.empty() || hw.empty()) return 0;
    std::map<std::string,int> rc, hc;
    for (auto & w : rw) rc[w]++;
    for (auto & w : hw) hc[w]++;
    int overlap = 0;
    for (auto & [w,c] : rc) if (hc.count(w)) overlap += std::min(c, hc[w]);
    float p = (float)overlap / hw.size(), r = (float)overlap / rw.size();
    return (p + r > 0) ? 2*p*r/(p+r) : 0;
}

static std::set<std::string> extract_concepts(const std::string & text) {
    std::set<std::string> stops = {
        "the","and","this","that","with","from","have","has","had","was","were","are",
        "been","being","will","would","could","should","may","might","shall","can",
        "does","did","done","into","through","during","before","after","above","below",
        "between","same","each","every","both","more","most","other","some","such",
        "only","over","also","than","then","when","where","which","while","about",
        "their","there","these","those","what","first","second","third","she","her",
        "his","its","they","them","very","just","like","looking","appears","hand",
        "right","left","video","sequence","shows","begins","scene","frame"
    };
    auto words = tokenize_words(text);
    std::set<std::string> concepts;
    for (auto & w : words) if (w.size() > 3 && !stops.count(w)) concepts.insert(w);
    return concepts;
}

static float concept_recall(const std::string & ref, const std::string & hyp) {
    auto rc = extract_concepts(ref), hc = extract_concepts(hyp);
    if (rc.empty()) return 0;
    int found = 0;
    for (auto & c : rc) if (hc.count(c)) found++;
    return (float)found / rc.size();
}

// ============================================================================
// Single run
// ============================================================================

struct run_result {
    scoring_strategy strategy;
    int    n_visual_tokens;
    int    n_generated;
    float  prefill_ms;
    float  gen_ms;
    float  tps;
    std::string text;
};

static run_result run_single(
    llama_context * lctx, const llama_model * model, const llama_vocab * vocab,
    mtmd_context * vctx, common_chat_templates * tmpls, common_params & params,
    const sweep_params & sp,
    const std::vector<std::string> & frame_paths,
    const std::vector<optical_flow> & flows,
    scoring_strategy strategy,
    float keep_ratio,
    const std::vector<float> & kf_embeddings,  // keyframe embeddings (frame 0)
    int kf_n_tok)
{
    run_result res;
    res.strategy = strategy;

    auto * mem = llama_get_memory(lctx);
    llama_memory_clear(mem, true);

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    int n_batch = params.n_batch;
    int n_embd = llama_model_n_embd_inp(model);

    // Load + tokenize
    mtmd::bitmaps bitmaps;
    for (const auto & path : frame_paths) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx, path.c_str()));
        bitmaps.entries.push_back(std::move(bmp));
    }

    std::string content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) content += mtmd_default_marker();
    content += "\n" + sp.prompt;

    common_chat_msg msg; msg.role = "user"; msg.content = content;
    std::vector<common_chat_msg> history;
    auto formatted = common_chat_format_single(tmpls, history, msg, true, false);

    mtmd_input_text tin; tin.text = formatted.c_str(); tin.add_special = true; tin.parse_special = true;
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bm = bitmaps.c_ptr();
    if (mtmd_tokenize(vctx, chunks.ptr.get(), &tin, bm.data(), bm.size()) != 0) return res;

    // Prefill
    int64_t t0 = ggml_time_ms();
    llama_pos n_past = 0;
    int img_idx = 0, total_vis = 0;

    for (size_t ci = 0; ci < chunks.size(); ci++) {
        auto * chunk = chunks[ci];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos np = n_past;
            mtmd_helper_eval_chunk_single(vctx, lctx, chunk, n_past, 0, n_batch,
                                          ci == chunks.size()-1, &np);
            n_past = np;
            continue;
        }

        auto * itk = mtmd_input_chunk_get_tokens_image(chunk);
        int cnx = (int)mtmd_image_tokens_get_nx(itk);
        int cny = (int)mtmd_image_tokens_get_ny(itk);
        int ntok = cnx * cny;

        mtmd_encode_chunk(vctx, chunk);
        float * embd = mtmd_get_output_embd(vctx);
        llama_pos np = n_past;

        if (img_idx == 0 || keep_ratio >= 0.999f) {
            // Full decode for frame 0 or 100%
            mtmd_helper_decode_image_chunk(vctx, lctx, chunk, embd, n_past, 0, n_batch, &np);
            total_vis += ntok;
        } else {
            // Score and select tokens based on strategy
            const float * kf_ptr = (!kf_embeddings.empty() && kf_n_tok == ntok)
                                   ? kf_embeddings.data() : nullptr;
            const optical_flow & flow = (img_idx < (int)flows.size()) ? flows[img_idx] : flows[0];

            auto selected = score_tokens(strategy, embd, kf_ptr, ntok, n_embd,
                                         flow, cnx, cny, keep_ratio, 42 + img_idx);

            inject_pruned_tokens(lctx, vctx, embd, selected, cnx, cny, n_embd, n_past, n_batch, &np);
            total_vis += (int)selected.size();
        }
        n_past = np;
        img_idx++;
    }

    res.n_visual_tokens = total_vis;
    res.prefill_ms = (float)(ggml_time_ms() - t0);

    // Generate
    int64_t tg = ggml_time_ms();
    std::string text;
    int ngen = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < sp.n_predict; i++) {
        llama_token tok = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, tok, true);
        if (llama_vocab_is_eog(vocab, tok)) break;
        text += common_token_to_piece(lctx, tok);
        common_batch_clear(batch);
        common_batch_add(batch, tok, n_past++, {0}, true);
        if (llama_decode(lctx, batch)) break;
        ngen++;
    }

    res.n_generated = ngen;
    res.gen_ms = (float)(ggml_time_ms() - tg);
    res.tps = ngen / (res.gen_ms / 1000.0f);
    res.text = text;

    llama_batch_free(batch);
    common_sampler_free(smpl);
    return res;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    sweep_params sp;
    auto filtered = filter_args(argc, argv, sp);
    int new_argc = (int)filtered.size();

    common_params params;
    if (!common_params_parse(new_argc, filtered.data(), params, LLAMA_EXAMPLE_MTMD)) return 1;
    common_init();

    if (params.model.path.empty())
        params.model.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf";
    if (params.mmproj.path.empty())
        params.mmproj.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf";
    if (params.n_predict < 0) params.n_predict = sp.n_predict;

    printf("=== Scoring Strategy Sweep ===\n");
    printf("Video: %s, frames: %d, keep: %.0f%%\n\n", sp.video_name.c_str(), sp.n_frames, sp.keep_ratio*100);

    auto llama_init = common_init_from_params(params);
    const llama_model * model = llama_init->model();
    llama_context * lctx = llama_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, params.chat_template);

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = params.mmproj_use_gpu;
    mp.print_timings = false;
    mp.n_threads = params.cpuparams.n_threads;
    mp.warmup = params.warmup;
    mp.image_min_tokens = params.image_min_tokens;
    mp.image_max_tokens = params.image_max_tokens;
    mp.flash_attn_type = params.flash_attn_type;

    mtmd::context_ptr vctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mp));
    int n_embd = llama_model_n_embd_inp(model);

    // Load frames
    std::string fbase = sp.sintel_dir + "/training/final/" + sp.video_name + "/";
    std::string flbase = sp.sintel_dir + "/training/flow/" + sp.video_name + "/";

    int total = 0;
    for (int i = 1; i <= 999; i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", i);
        FILE * f = fopen((fbase + buf).c_str(), "rb");
        if (f) { fclose(f); total++; } else break;
    }

    std::vector<std::string> fpaths;
    std::vector<int> fnums;
    for (int i = 0; i < sp.n_frames; i++) {
        int fn = 1 + (i * (total - 1)) / std::max(1, sp.n_frames - 1);
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", fn);
        fpaths.push_back(fbase + buf);
        fnums.push_back(fn);
    }

    std::vector<optical_flow> flows(fnums.size());
    for (size_t i = 0; i < fnums.size(); i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.flo", fnums[i]);
        read_flo_file(flbase + buf, flows[i]);
    }

    printf("Frames: "); for (int fn : fnums) printf("%d ", fn); printf("\n");

    // Pre-encode keyframe to get embeddings for COSINE_KF strategy
    printf("Encoding keyframe for cross-frame comparison...\n");
    std::vector<float> kf_embeddings;
    int kf_n_tok = 0;
    {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx.get(), fpaths[0].c_str()));
        std::string marker(mtmd_default_marker());
        mtmd_input_text tin; tin.text = marker.c_str(); tin.add_special = false; tin.parse_special = true;
        mtmd::input_chunks cks(mtmd_input_chunks_init());
        const mtmd_bitmap * bp = bmp.ptr.get();
        mtmd_tokenize(vctx.get(), cks.ptr.get(), &tin, &bp, 1);

        for (size_t ci = 0; ci < cks.size(); ci++) {
            if (mtmd_input_chunk_get_type(cks[ci]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                auto * itk = mtmd_input_chunk_get_tokens_image(cks[ci]);
                int nx = (int)mtmd_image_tokens_get_nx(itk);
                int ny = (int)mtmd_image_tokens_get_ny(itk);
                kf_n_tok = nx * ny;

                mtmd_encode_chunk(vctx.get(), cks[ci]);
                float * embd = mtmd_get_output_embd(vctx.get());
                kf_embeddings.assign(embd, embd + kf_n_tok * n_embd);
                printf("  Keyframe: %dx%d = %d tokens, %d embd\n", nx, ny, kf_n_tok, n_embd);
                break;
            }
        }
    }

    // Run baseline (100%) first
    printf("\nRunning baseline (100%%)...\n");
    auto baseline = run_single(lctx, model, vocab, vctx.get(), tmpls.get(), params, sp,
                               fpaths, flows, SCORE_BLEND, 1.0f, kf_embeddings, kf_n_tok);
    printf("  Baseline: %d vis tokens, %d gen tokens\n", baseline.n_visual_tokens, baseline.n_generated);
    printf("  Text: %.200s...\n\n", baseline.text.c_str());

    // Run each strategy
    std::vector<run_result> results;
    results.push_back(baseline);

    scoring_strategy strategies[] = {
        SCORE_MOTION, SCORE_STATIC, SCORE_COSINE_KF, SCORE_COSINE_CTR,
        SCORE_NORM, SCORE_BLEND, SCORE_RANDOM, SCORE_MOTION_KF
    };

    for (auto strat : strategies) {
        printf("Running %s (%.0f%%)...\n", strategy_name(strat), sp.keep_ratio * 100);
        auto r = run_single(lctx, model, vocab, vctx.get(), tmpls.get(), params, sp,
                            fpaths, flows, strat, sp.keep_ratio, kf_embeddings, kf_n_tok);
        results.push_back(r);
        printf("  %d vis tokens, %d gen tokens, %.1f t/s\n", r.n_visual_tokens, r.n_generated, r.tps);
        printf("  Text: %.150s...\n\n", r.text.c_str());
    }

    // Summary table
    printf("\n");
    printf("╔══════════════╦═══════════╦══════╦════════╦══════════╦══════════╦═════════╗\n");
    printf("║  Strategy    ║ Vis. Tok  ║ Gen  ║  t/s   ║ ROUGE-1  ║ Concepts ║ Prefill ║\n");
    printf("╠══════════════╬═══════════╬══════╬════════╬══════════╬══════════╬═════════╣\n");

    for (auto & r : results) {
        float r1 = rouge1_f1(baseline.text, r.text);
        float cr = concept_recall(baseline.text, r.text);
        const char * name = (r.strategy == SCORE_BLEND && r.n_visual_tokens == baseline.n_visual_tokens)
                           ? "BASELINE" : strategy_name(r.strategy);
        printf("║ %-12s ║ %7d   ║ %4d ║ %6.1f ║  %5.1f%%  ║  %5.1f%%  ║ %5.0f ms║\n",
               name, r.n_visual_tokens, r.n_generated, r.tps, r1*100, cr*100, r.prefill_ms);
    }
    printf("╚══════════════╩═══════════╩══════╩════════╩══════════╩══════════╩═════════╝\n");

    // Print actual texts for comparison
    printf("\n=== Generated Texts ===\n\n");
    for (auto & r : results) {
        const char * name = (r.strategy == SCORE_BLEND && r.n_visual_tokens == baseline.n_visual_tokens)
                           ? "BASELINE" : strategy_name(r.strategy);
        printf("--- %s (%d vis tok) ---\n%s\n\n", name, r.n_visual_tokens, r.text.c_str());
    }

    // Save JSON
    json out;
    out["video"] = sp.video_name;
    out["n_frames"] = sp.n_frames;
    out["keep_ratio"] = sp.keep_ratio;
    out["reference_text"] = baseline.text;

    auto ref_concepts = extract_concepts(baseline.text);
    json jc = json::array();
    for (auto & c : ref_concepts) jc.push_back(c);
    out["reference_concepts"] = jc;

    json runs = json::array();
    for (auto & r : results) {
        json rj;
        rj["strategy"] = (r.strategy == SCORE_BLEND && r.n_visual_tokens == baseline.n_visual_tokens)
                        ? "BASELINE" : strategy_name(r.strategy);
        rj["n_visual_tokens"] = r.n_visual_tokens;
        rj["n_generated"] = r.n_generated;
        rj["prefill_ms"] = r.prefill_ms;
        rj["gen_ms"] = r.gen_ms;
        rj["tokens_per_sec"] = r.tps;
        rj["text"] = r.text;
        rj["rouge1_f1"] = rouge1_f1(baseline.text, r.text);
        rj["concept_recall"] = concept_recall(baseline.text, r.text);
        runs.push_back(rj);
    }
    out["runs"] = runs;

    std::ofstream ofs(sp.output_json);
    ofs << out.dump(2) << std::endl;
    printf("Results saved to %s\n", sp.output_json.c_str());

    return 0;
}
