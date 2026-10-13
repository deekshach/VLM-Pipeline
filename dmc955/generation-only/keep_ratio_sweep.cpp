// Keep-Ratio Sweep: Accuracy vs Token Count Study
//
// Loads the model ONCE, then for each keep ratio:
//   1. Clears KV cache
//   2. Prefills with pre-LLM pruning at the given ratio
//   3. Generates text
//   4. Records output quality and timing
//
// Outputs a JSON with all results for analysis.
//
// Usage:
//   llama-keep-sweep -m <model> --mmproj <mmproj> --video <name> [options]
//     --ratios <list>       Comma-separated keep ratios (default: 1.0,0.8,...,0.1)
//     --n-frames <int>      Number of frames (default: 8)
//     --n-predict <int>     Max tokens to generate (default: 256)

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

using json = nlohmann::ordered_json;

// ============================================================================
// Parameters
// ============================================================================

struct sweep_params {
    std::string sintel_dir  = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel";
    std::string video_name  = "alley_1";
    int         n_frames    = 8;
    std::string output_json = "keep_ratio_sweep.json";
    std::string prompt      = "Describe what is happening in this video sequence in detail.";
    int         n_predict   = 256;

    std::vector<float> ratios = {1.0f, 0.80f, 0.60f, 0.50f, 0.40f, 0.30f, 0.20f, 0.15f, 0.10f, 0.05f};

    bool protect_frame0 = true;
};

static std::vector<char *> filter_args(int argc, char ** argv, sweep_params & sp) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--sintel-dir" && i + 1 < argc) {
            sp.sintel_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            sp.video_name = argv[++i];
        } else if (arg == "--n-frames" && i + 1 < argc) {
            sp.n_frames = std::stoi(argv[++i]);
        } else if (arg == "--output-json" && i + 1 < argc) {
            sp.output_json = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            sp.prompt = argv[++i];
        } else if (arg == "--n-predict" && i + 1 < argc) {
            sp.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--ratios" && i + 1 < argc) {
            sp.ratios.clear();
            std::istringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) sp.ratios.push_back(std::stof(tok));
        } else if (arg == "--no-protect-frame0") {
            sp.protect_frame0 = false;
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

// ============================================================================
// Optical flow helpers (minimal, for pre-LLM scoring)
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
        } else {
            pp = pos.data() + off;
        }

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
// Score tokens for pre-LLM pruning (motion + cosine + norm)
// ============================================================================

static std::vector<scored_token> score_and_select(
    const float * embd, int n_tok, int n_embd,
    const optical_flow & flow, int nx, int ny,
    float keep_ratio)
{
    int n_keep = std::max(1, (int)(n_tok * keep_ratio));

    float w_motion = 0.5f, w_cosine = 0.3f, w_norm = 0.2f;

    // Motion scores
    std::vector<float> motion(n_tok, 0.0f);
    bool has_flow = flow.width > 0;
    float m_max = 1e-8f;
    if (has_flow) {
        motion = compute_token_motion(flow, nx, ny);
        for (float m : motion) m_max = std::max(m_max, m);
    }

    // Centroid
    std::vector<float> centroid(n_embd, 0.0f);
    for (int t = 0; t < n_tok; t++) {
        const float * e = embd + t * n_embd;
        for (int d = 0; d < n_embd; d++) centroid[d] += e[d];
    }
    for (int d = 0; d < n_embd; d++) centroid[d] /= n_tok;
    float c_norm = 0.0f;
    for (int d = 0; d < n_embd; d++) c_norm += centroid[d] * centroid[d];
    c_norm = sqrtf(c_norm + 1e-8f);

    // Score
    float norm_max = 1e-8f;
    std::vector<scored_token> scores(n_tok);
    for (int t = 0; t < n_tok; t++) {
        const float * e = embd + t * n_embd;
        float dot = 0.0f, tn2 = 0.0f;
        for (int d = 0; d < n_embd; d++) { dot += e[d]*centroid[d]; tn2 += e[d]*e[d]; }
        float tn = sqrtf(tn2 + 1e-8f);
        float cos_d = 1.0f - dot / (tn * c_norm);
        norm_max = std::max(norm_max, tn);
        float ms = has_flow ? motion[t] / m_max : 0.0f;
        scores[t] = {t, w_motion * ms + w_cosine * cos_d + w_norm * (tn / norm_max)};
    }

    // Spatial grid guarantee (4x4): keep best token per cell
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
                    if (scores[i].score > best_s) { best_s = scores[i].score; best = i; }
                }
            if (best >= 0) force_keep[best] = true;
        }

    std::vector<scored_token> forced, candidates;
    for (int t = 0; t < n_tok; t++) {
        if (force_keep[t]) forced.push_back(scores[t]);
        else candidates.push_back(scores[t]);
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
// Single sweep run
// ============================================================================

struct sweep_result {
    float  keep_ratio;
    int    n_visual_tokens;       // total visual tokens in KV after prefill
    int    n_tokens_per_frame;    // tokens per frame at this ratio
    int    n_tokens_frame0;       // frame 0 always full
    int    n_tokens_generated;
    float  prefill_ms;
    float  generation_ms;
    float  tokens_per_sec;
    std::string generated_text;
};

static sweep_result run_single(
    llama_context * lctx,
    const llama_model * model,
    const llama_vocab * vocab,
    mtmd_context * vctx,
    common_chat_templates * tmpls,
    const common_params & params,
    const sweep_params & sp,
    const std::vector<std::string> & frame_paths,
    const std::vector<optical_flow> & flows,
    float keep_ratio)
{
    sweep_result res;
    res.keep_ratio = keep_ratio;

    // Clear KV cache
    auto * mem = llama_get_memory(lctx);
    llama_memory_clear(mem, true);

    // Reset sampler
    common_params & mutable_params = const_cast<common_params &>(params);
    common_sampler * smpl = common_sampler_init(model, mutable_params.sampling);

    int n_batch = params.n_batch;
    int n_embd = llama_model_n_embd_inp(model);

    // Load + tokenize
    mtmd::bitmaps bitmaps;
    for (const auto & path : frame_paths) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx, path.c_str()));
        if (!bmp.ptr) { fprintf(stderr, "Failed: %s\n", path.c_str()); return res; }
        bitmaps.entries.push_back(std::move(bmp));
    }

    std::string content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) content += mtmd_default_marker();
    content += "\n" + sp.prompt;

    common_chat_msg msg;
    msg.role = "user";
    msg.content = content;

    std::vector<common_chat_msg> history;
    auto formatted = common_chat_format_single(tmpls, history, msg, true, false);

    mtmd_input_text text_in;
    text_in.text = formatted.c_str();
    text_in.add_special = true;
    text_in.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bm_ptrs = bitmaps.c_ptr();
    if (mtmd_tokenize(vctx, chunks.ptr.get(), &text_in, bm_ptrs.data(), bm_ptrs.size()) != 0) {
        fprintf(stderr, "Tokenize failed\n");
        return res;
    }

    // Prefill
    int64_t t_prefill = ggml_time_ms();
    llama_pos n_past = 0;
    int image_idx = 0;
    int total_visual = 0;

    for (size_t ci = 0; ci < chunks.size(); ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        auto ctype = mtmd_input_chunk_get_type(chunk);

        if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool is_last = (ci == chunks.size() - 1);
            llama_pos np = n_past;
            int32_t ret = mtmd_helper_eval_chunk_single(vctx, lctx, chunk, n_past, 0, n_batch, is_last, &np);
            if (ret != 0) { fprintf(stderr, "Text eval fail\n"); break; }
            n_past = np;
            continue;
        }

        // IMAGE chunk
        const auto itk = mtmd_input_chunk_get_tokens_image(chunk);
        int cnx = (int)mtmd_image_tokens_get_nx(itk);
        int cny = (int)mtmd_image_tokens_get_ny(itk);
        int n_tok = cnx * cny;

        // Encode ViT
        if (mtmd_encode_chunk(vctx, chunk) != 0) { fprintf(stderr, "Encode fail\n"); break; }
        float * embd = mtmd_get_output_embd(vctx);

        llama_pos np = n_past;

        if (image_idx == 0 || keep_ratio >= 0.999f) {
            // Frame 0 or 100% keep: full decode
            int32_t ret = mtmd_helper_decode_image_chunk(vctx, lctx, chunk, embd, n_past, 0, n_batch, &np);
            if (ret != 0) { fprintf(stderr, "Decode fail frame %d\n", image_idx); break; }
            total_visual += n_tok;
            if (image_idx == 0) { res.n_tokens_frame0 = n_tok; res.n_tokens_per_frame = n_tok; }
        } else {
            // Pre-LLM pruning
            const optical_flow & flow = (image_idx < (int)flows.size()) ? flows[image_idx] : flows[0];
            auto selected = score_and_select(embd, n_tok, n_embd, flow, cnx, cny, keep_ratio);
            int n_keep = (int)selected.size();

            int32_t ret = inject_pruned_tokens(lctx, vctx, embd, selected, cnx, cny, n_embd, n_past, n_batch, &np);
            if (ret != 0) { fprintf(stderr, "Inject fail frame %d\n", image_idx); break; }
            total_visual += n_keep;
            res.n_tokens_per_frame = n_keep;
        }

        n_past = np;
        image_idx++;
    }

    res.n_visual_tokens = total_visual;
    res.prefill_ms = (float)(ggml_time_ms() - t_prefill);

    // Generate
    int64_t t_gen = ggml_time_ms();
    std::string text;
    int n_gen = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < sp.n_predict; i++) {
        llama_token tok = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, tok, true);

        if (llama_vocab_is_eog(vocab, tok)) break;

        text += common_token_to_piece(lctx, tok);

        common_batch_clear(batch);
        common_batch_add(batch, tok, n_past++, {0}, true);
        if (llama_decode(lctx, batch)) { fprintf(stderr, "Decode fail tok %d\n", i); break; }

        n_gen++;
    }

    res.n_tokens_generated = n_gen;
    res.generation_ms = (float)(ggml_time_ms() - t_gen);
    res.tokens_per_sec = n_gen / (res.generation_ms / 1000.0f);
    res.generated_text = text;

    llama_batch_free(batch);
    common_sampler_free(smpl);

    return res;
}

// ============================================================================
// Text similarity (word overlap / ROUGE-1)
// ============================================================================

static std::vector<std::string> tokenize_words(const std::string & text) {
    std::vector<std::string> words;
    std::istringstream ss(text);
    std::string w;
    while (ss >> w) {
        // Lowercase + strip punctuation
        std::string clean;
        for (char c : w) {
            if (std::isalpha(c)) clean += std::tolower(c);
        }
        if (!clean.empty()) words.push_back(clean);
    }
    return words;
}

static float rouge1_f1(const std::string & reference, const std::string & hypothesis) {
    auto ref_words = tokenize_words(reference);
    auto hyp_words = tokenize_words(hypothesis);

    if (ref_words.empty() || hyp_words.empty()) return 0.0f;

    // Count word occurrences
    std::map<std::string, int> ref_counts, hyp_counts;
    for (const auto & w : ref_words) ref_counts[w]++;
    for (const auto & w : hyp_words) hyp_counts[w]++;

    // Overlap
    int overlap = 0;
    for (const auto & [word, count] : ref_counts) {
        if (hyp_counts.count(word)) {
            overlap += std::min(count, hyp_counts[word]);
        }
    }

    float precision = (float)overlap / hyp_words.size();
    float recall    = (float)overlap / ref_words.size();

    if (precision + recall < 1e-8f) return 0.0f;
    return 2.0f * precision * recall / (precision + recall);
}

// Key concept extraction: find important nouns/adjectives
static std::set<std::string> extract_key_concepts(const std::string & text) {
    // Simple: just use all words longer than 3 chars as "concepts"
    auto words = tokenize_words(text);
    std::set<std::string> concepts;
    // Common stop words to exclude
    std::set<std::string> stops = {
        "the", "and", "this", "that", "with", "from", "have", "has", "had",
        "was", "were", "are", "been", "being", "will", "would", "could",
        "should", "may", "might", "shall", "can", "does", "did", "done",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "same", "each", "every", "both", "more", "most", "other",
        "some", "such", "only", "over", "also", "than", "then", "when",
        "where", "which", "while", "about", "their", "there", "these",
        "those", "what", "first", "second", "third", "she", "her", "his",
        "its", "they", "them", "very", "just", "like", "looking", "appears"
    };

    for (const auto & w : words) {
        if (w.size() > 3 && !stops.count(w)) concepts.insert(w);
    }
    return concepts;
}

static float concept_recall(const std::string & reference, const std::string & hypothesis) {
    auto ref_concepts = extract_key_concepts(reference);
    auto hyp_concepts = extract_key_concepts(hypothesis);

    if (ref_concepts.empty()) return 0.0f;

    int found = 0;
    for (const auto & c : ref_concepts) {
        if (hyp_concepts.count(c)) found++;
    }

    return (float)found / ref_concepts.size();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    sweep_params sp;
    auto filtered = filter_args(argc, argv, sp);

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
    if (params.n_predict < 0) params.n_predict = sp.n_predict;

    // Sort ratios descending (run 100% first as reference)
    std::sort(sp.ratios.begin(), sp.ratios.end(), std::greater<float>());

    printf("=== Keep-Ratio Sweep ===\n");
    printf("Video: %s, frames: %d, predict: %d\n", sp.video_name.c_str(), sp.n_frames, sp.n_predict);
    printf("Ratios: ");
    for (float r : sp.ratios) printf("%.0f%% ", r * 100);
    printf("\n\n");

    // Initialize model once
    auto llama_init = common_init_from_params(params);
    const llama_model * model = llama_init->model();
    llama_context * lctx      = llama_init->context();
    const llama_vocab * vocab  = llama_model_get_vocab(model);

    auto tmpls = common_chat_templates_init(model, params.chat_template);

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu         = params.mmproj_use_gpu;
    mp.print_timings   = false;
    mp.n_threads       = params.cpuparams.n_threads;
    mp.warmup          = params.warmup;
    mp.image_min_tokens = params.image_min_tokens;
    mp.image_max_tokens = params.image_max_tokens;
    mp.flash_attn_type = params.flash_attn_type;

    mtmd::context_ptr vctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mp));
    if (!vctx.get()) { fprintf(stderr, "Vision init failed\n"); return 1; }

    printf("Model loaded: %d layers, %d embd\n\n", llama_model_n_layer(model), llama_model_n_embd(model));

    // Load frame paths
    std::string frame_base = sp.sintel_dir + "/training/final/" + sp.video_name + "/";
    std::string flow_base  = sp.sintel_dir + "/training/flow/" + sp.video_name + "/";

    int total_frames = 0;
    for (int i = 1; i <= 999; i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", i);
        FILE * f = fopen((frame_base + buf).c_str(), "rb");
        if (f) { fclose(f); total_frames++; } else break;
    }

    std::vector<std::string> frame_paths;
    std::vector<int> frame_nums;
    for (int i = 0; i < sp.n_frames; i++) {
        int fn = 1 + (i * (total_frames - 1)) / std::max(1, sp.n_frames - 1);
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.png", fn);
        frame_paths.push_back(frame_base + buf);
        frame_nums.push_back(fn);
    }
    printf("Selected %d/%d frames: ", sp.n_frames, total_frames);
    for (int fn : frame_nums) printf("%d ", fn);
    printf("\n\n");

    // Load optical flow
    std::vector<optical_flow> flows(frame_nums.size());
    for (size_t i = 0; i < frame_nums.size(); i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "frame_%04d.flo", frame_nums[i]);
        read_flo_file(flow_base + buf, flows[i]);
    }

    // Run sweep
    std::vector<sweep_result> results;
    std::string reference_text;

    for (float ratio : sp.ratios) {
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Keep ratio: %.0f%%\n", ratio * 100);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        auto res = run_single(lctx, model, vocab, vctx.get(), tmpls.get(), params, sp,
                              frame_paths, flows, ratio);

        printf("  Visual tokens: %d (frame0=%d + %d×%d)\n",
               res.n_visual_tokens, res.n_tokens_frame0,
               sp.n_frames - 1, res.n_tokens_per_frame);
        printf("  Generated: %d tokens, %.1f t/s\n", res.n_tokens_generated, res.tokens_per_sec);
        printf("  Prefill: %.0f ms, Gen: %.0f ms\n", res.prefill_ms, res.generation_ms);
        printf("  Text: %s\n\n", res.generated_text.substr(0, 200).c_str());

        if (ratio >= 0.999f) {
            reference_text = res.generated_text;
        }

        results.push_back(res);
    }

    // Analysis
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        KEEP-RATIO SWEEP RESULTS                            ║\n");
    printf("╠══════════╦═══════════╦══════╦════════╦══════════╦══════════╦════════════════╣\n");
    printf("║ Keep %%   ║ Vis. Tok  ║ Gen  ║  t/s   ║ ROUGE-1  ║ Concept  ║ Prefill+Gen ms ║\n");
    printf("╠══════════╬═══════════╬══════╬════════╬══════════╬══════════╬════════════════╣\n");

    for (const auto & r : results) {
        float r1 = rouge1_f1(reference_text, r.generated_text);
        float cr = concept_recall(reference_text, r.generated_text);

        printf("║ %6.0f%%  ║ %7d   ║ %4d ║ %6.1f ║  %5.1f%%  ║  %5.1f%%  ║  %6.0f + %-5.0f ║\n",
               r.keep_ratio * 100,
               r.n_visual_tokens,
               r.n_tokens_generated,
               r.tokens_per_sec,
               r1 * 100,
               cr * 100,
               r.prefill_ms, r.generation_ms);
    }
    printf("╚══════════╩═══════════╩══════╩════════╩══════════╩══════════╩════════════════╝\n");

    // Find quality knee point
    printf("\nQuality analysis vs baseline (100%%):\n");
    for (const auto & r : results) {
        float r1 = rouge1_f1(reference_text, r.generated_text);
        float cr = concept_recall(reference_text, r.generated_text);
        float savings = 1.0f - (float)r.n_visual_tokens / results[0].n_visual_tokens;

        printf("  %3.0f%%: ROUGE=%.1f%% Concepts=%.1f%% Savings=%.0f%% | %s\n",
               r.keep_ratio * 100, r1 * 100, cr * 100, savings * 100,
               (r1 > 0.7f && cr > 0.7f) ? "GOOD" :
               (r1 > 0.5f && cr > 0.5f) ? "OK" : "DEGRADED");
    }

    // Save JSON
    json out;
    out["video"] = sp.video_name;
    out["n_frames"] = sp.n_frames;
    out["n_predict"] = sp.n_predict;
    out["reference_text"] = reference_text;

    json runs = json::array();
    for (const auto & r : results) {
        json rj;
        rj["keep_ratio"]        = r.keep_ratio;
        rj["n_visual_tokens"]   = r.n_visual_tokens;
        rj["n_tokens_frame0"]   = r.n_tokens_frame0;
        rj["n_tokens_per_frame"] = r.n_tokens_per_frame;
        rj["n_tokens_generated"] = r.n_tokens_generated;
        rj["prefill_ms"]        = r.prefill_ms;
        rj["generation_ms"]     = r.generation_ms;
        rj["tokens_per_sec"]    = r.tokens_per_sec;
        rj["generated_text"]    = r.generated_text;
        rj["rouge1_f1"]         = rouge1_f1(reference_text, r.generated_text);
        rj["concept_recall"]    = concept_recall(reference_text, r.generated_text);
        rj["token_savings_pct"] = 100.0f * (1.0f - (float)r.n_visual_tokens / results[0].n_visual_tokens);
        runs.push_back(rj);
    }
    out["runs"] = runs;

    // Key concepts from reference
    auto ref_concepts = extract_key_concepts(reference_text);
    json concepts = json::array();
    for (const auto & c : ref_concepts) concepts.push_back(c);
    out["reference_concepts"] = concepts;

    std::ofstream ofs(sp.output_json);
    ofs << out.dump(2) << std::endl;
    printf("\nResults saved to %s\n", sp.output_json.c_str());

    return 0;
}
