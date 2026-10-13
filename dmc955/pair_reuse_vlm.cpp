// How much of a paired-frame Qwen2.5-VL request could be reused, and at what cost
//
// Baseline: frames go through the ViT two at a time (one token grid per pair),
// every pair fully encoded. For each video the pairs are encoded once, and the
// LLM is then run several times on the same prompt with the visual tokens
// altered after the fact:
//
//   stale  a token judged unchanged keeps the embedding it had when it was last
//          taken fresh; the LLM still gets every token. Upper bound for any
//          scheme that skips ViT work: fresh tokens here come from a full encode,
//          so they carry no crop error.
//   drop   a token judged unchanged is not sent to the LLM at all (the previous
//          pair's token is already in context). Shrinks prefill as well.
//
// "Unchanged" is decided by one of
//   pixel<T>  112 px window whose worst token cell moved by at most T (mean
//             |dRGB|, 0..255, either frame of the pair) since it was last fresh
//   cos<C>    token whose true embedding is within cosine C of the cached one
//             (an oracle: it needs the full encode it is trying to avoid)
//
// Error is measured at the LLM output against the baseline run of the same
// video: KL divergence of the next-token distribution along the baseline's own
// caption (teacher forced), and the caption each variant writes by itself.
// Generation is greedy, so with -fa off every number is reproducible.

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fstream>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Parameters
// ============================================================================

static const int TOKEN_PX      = 28;   // one LLM token = 2x2 patches of 14 px
static const int WINDOW_PX     = 112;  // ViT attention window
static const int WINDOW_TOKENS = WINDOW_PX / TOKEN_PX;

struct pr_params {
    std::string data_dir      = "/data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel";
    std::string frames_subdir = "training/clean";
    std::string videos        = "alley_1";   // comma-separated
    std::string output_json;
    std::string pixel_thresholds = "6,24,48";
    std::string cos_thresholds   = "0.95,0.9,0.8,0.7";
    int base_frames = 8;
    int frame_step  = 0;      // 0 = spread base_frames evenly over the clip; N = every Nth frame from the start
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, pr_params & p) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        const bool has = i + 1 < argc;
        if      (arg == "--sintel-dir" && has)        p.data_dir = argv[++i];
        else if (arg == "--frames-subdir" && has)     p.frames_subdir = argv[++i];
        else if (arg == "--videos" && has)            p.videos = argv[++i];
        else if (arg == "--output-json" && has)       p.output_json = argv[++i];
        else if (arg == "--base-frames" && has)       p.base_frames = std::stoi(argv[++i]);
        else if (arg == "--frame-step" && has)        p.frame_step = std::stoi(argv[++i]);
        else if (arg == "--pixel-thresholds" && has)  p.pixel_thresholds = argv[++i];
        else if (arg == "--cos-thresholds" && has)    p.cos_thresholds = argv[++i];
        else filtered.push_back(argv[i]);
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nReuse potential on top of paired-frame Qwen2.5-VL input\n\nUsage: %s [options]\n\n", argv[0]);
    printf("  --videos <a,b,c>          videos under <sintel-dir>/<frames-subdir>\n");
    printf("  --base-frames <n>         sampled frames per request, paired two by two (default: 8)\n");
    printf("  --frame-step <n>          take every nth frame from the start instead (default: 0 = spread evenly)\n");
    printf("  --pixel-thresholds <a,b>  window change thresholds, mean |dRGB| 0..255 (default: 6,24,48)\n");
    printf("  --cos-thresholds <a,b>    oracle token cosine thresholds (default: 0.95,0.9,0.8,0.7)\n");
    printf("  --output-json <f>         results file\n");
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stof(cur)); cur.clear(); }
        else cur += c;
    }
    return out;
}

// ============================================================================
// Section 2: Images
// ============================================================================

struct rgb_image {
    int w = 0, h = 0;
    std::vector<unsigned char> px;   // RGB, row-major
};

static bool load_image(mtmd_context * mctx, const std::string & path, rgb_image & out) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mctx, path.c_str()));
    if (!bmp.ptr) return false;
    out.w = (int)bmp.nx();
    out.h = (int)bmp.ny();
    out.px.assign(bmp.data(), bmp.data() + bmp.n_bytes());
    return true;
}

static rgb_image resize_bilinear(const rgb_image & src, int W, int H) {
    if (src.w == W && src.h == H) return src;
    rgb_image dst;
    dst.w = W; dst.h = H;
    dst.px.resize((size_t)W * H * 3);
    const float sx = (float)src.w / W, sy = (float)src.h / H;
    for (int y = 0; y < H; y++) {
        const float fy = std::min(std::max((y + 0.5f) * sy - 0.5f, 0.0f), (float)(src.h - 1));
        const int y0 = (int)fy, y1 = std::min(y0 + 1, src.h - 1);
        const float wy = fy - y0;
        for (int x = 0; x < W; x++) {
            const float fx = std::min(std::max((x + 0.5f) * sx - 0.5f, 0.0f), (float)(src.w - 1));
            const int x0 = (int)fx, x1 = std::min(x0 + 1, src.w - 1);
            const float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                const float v = (1 - wy) * ((1 - wx) * src.px[((size_t)y0 * src.w + x0) * 3 + c] + wx * src.px[((size_t)y0 * src.w + x1) * 3 + c])
                              +      wy  * ((1 - wx) * src.px[((size_t)y1 * src.w + x0) * 3 + c] + wx * src.px[((size_t)y1 * src.w + x1) * 3 + c]);
                dst.px[((size_t)y * W + x) * 3 + c] = (unsigned char)std::lround(std::min(std::max(v, 0.0f), 255.0f));
            }
        }
    }
    return dst;
}

static int round_to(int v, int f) { return std::max(f, (int)std::lround((double)v / f) * f); }

// ============================================================================
// Section 3: Model context
// ============================================================================

struct vlm_context {
    mtmd::context_ptr       ctx_vision;
    common_init_result_ptr  llama_init;

    llama_model       * model = nullptr;
    llama_context     * lctx  = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler    * smpl  = nullptr;
    llama_batch         batch;
    int                 n_batch = 2048;

    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    llama_pos n_past = 0;

    vlm_context(common_params & params)
        : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx  = llama_init->context();
        if (!model || !lctx) {
            LOG_ERR("Failed to initialize model/context\n");
            exit(1);
        }
        vocab = llama_model_get_vocab(model);
        smpl  = common_sampler_init(model, params.sampling);
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;
        tmpls = common_chat_templates_init(model, params.chat_template);

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu         = params.mmproj_use_gpu;
        mparams.print_timings   = false;
        mparams.n_threads       = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup          = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;

        ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", params.mmproj.path.c_str());
            exit(1);
        }
    }

    void reset() {
        llama_memory_clear(llama_get_memory(lctx), true);
        common_sampler_reset(smpl);
        chat_history.clear();
        n_past = 0;
    }

    ~vlm_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }
};

// ============================================================================
// Section 4: Change measures
// ============================================================================

// Per 112 px window: the largest mean |dRGB| among its token cells.
static std::vector<float> window_diff(const rgb_image & cur, const rgb_image & ref, int nwx, int nwy) {
    std::vector<float> score((size_t)nwx * nwy, 0.0f);
    for (int wy = 0; wy < nwy; wy++) {
        for (int wx = 0; wx < nwx; wx++) {
            float worst = 0.0f;
            for (int cy = 0; cy < WINDOW_TOKENS; cy++) {
                for (int cx = 0; cx < WINDOW_TOKENS; cx++) {
                    const int x0 = wx * WINDOW_PX + cx * TOKEN_PX, y0 = wy * WINDOW_PX + cy * TOKEN_PX;
                    long sum = 0;
                    for (int y = y0; y < y0 + TOKEN_PX; y++) {
                        const unsigned char * a = &cur.px[((size_t)y * cur.w + x0) * 3];
                        const unsigned char * b = &ref.px[((size_t)y * ref.w + x0) * 3];
                        for (int k = 0; k < TOKEN_PX * 3; k++) sum += std::abs((int)a[k] - (int)b[k]);
                    }
                    worst = std::max(worst, (float)sum / (TOKEN_PX * TOKEN_PX * 3));
                }
            }
            score[(size_t)wy * nwx + wx] = worst;
        }
    }
    return score;
}

static void copy_window(rgb_image & dst, const rgb_image & src, int wx, int wy) {
    for (int y = wy * WINDOW_PX; y < (wy + 1) * WINDOW_PX; y++) {
        memcpy(&dst.px[((size_t)y * dst.w + wx * WINDOW_PX) * 3], &src.px[((size_t)y * src.w + wx * WINDOW_PX) * 3],
               (size_t)WINDOW_PX * 3);
    }
}

static float token_cos(const float * a, const float * b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int k = 0; k < n; k++) { ab += (double)a[k] * b[k]; aa += (double)a[k] * a[k]; bb += (double)b[k] * b[k]; }
    return (float)(ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30));
}

// One way of deciding which tokens are unchanged, applied to a whole video.
struct criterion {
    std::string name;                               // pixel24, cos0.90
    std::vector<std::vector<uint8_t>> reused;       // [pair][token]
    std::vector<std::vector<float>>   input;        // [pair] embeddings the stale variant feeds
    double frac_reused = 0;      // of the tokens in pairs after the first
    double frac_rect   = 0;      // share of those pairs a window-aligned bounding rectangle of the fresh tokens covers
    double cos_input   = 0;      // stale input vs the true encode, pairs after the first
    double pairs_all_reused = 0; // share of those pairs with nothing fresh
};

static void finish_criterion(criterion & c, const std::vector<std::vector<float>> & truth, int nx, int ny, int n_embd) {
    const int n_tok = nx * ny, n_pairs = (int)truth.size();
    const int nwx = nx / WINDOW_TOKENS, nwy = ny / WINDOW_TOKENS;
    double reused = 0, rect = 0, cs = 0, all = 0;
    for (int k = 1; k < n_pairs; k++) {
        int n_re = 0, x0 = nwx, x1 = -1, y0 = nwy, y1 = -1;
        for (int i = 0; i < n_tok; i++) {
            cs += token_cos(&c.input[k][(size_t)i * n_embd], &truth[k][(size_t)i * n_embd], n_embd);
            if (c.reused[k][i]) { n_re++; continue; }
            const int wx = (i % nx) / WINDOW_TOKENS, wy = (i / nx) / WINDOW_TOKENS;
            x0 = std::min(x0, wx); x1 = std::max(x1, wx); y0 = std::min(y0, wy); y1 = std::max(y1, wy);
        }
        reused += (double)n_re / n_tok;
        if (x1 >= 0) rect += (double)((x1 - x0 + 1) * (y1 - y0 + 1)) / (nwx * nwy);
        else         all += 1;
    }
    const int d = std::max(n_pairs - 1, 1);
    c.frac_reused = reused / d;
    c.frac_rect = rect / d;
    c.cos_input = cs / ((double)d * n_tok);
    c.pairs_all_reused = all / d;
}

// ============================================================================
// Section 5: LLM runs
// ============================================================================

struct llm_run {
    std::vector<llama_token> tokens;      // generated, including the end token if reached
    std::string text;
    int   n_visual = 0;
    float prefill_ms = 0;
    double kl_mean = 0, kl_first = 0, top1_agree = 1;
    bool ok = false;
};

// Visual embeddings into the LLM with explicit rotary positions (t, h, w per token).
static bool decode_visual(vlm_context & ctx, const float * embd, int n_tok, int n_embd,
                          const std::vector<llama_pos> & pt, const std::vector<llama_pos> & ph,
                          const std::vector<llama_pos> & pw) {
    llama_seq_id seq0 = 0;
    for (int off = 0; off < n_tok; off += ctx.n_batch) {
        const int n = std::min(ctx.n_batch, n_tok - off);
        std::vector<llama_pos> pos((size_t)n * 4);
        for (int i = 0; i < n; i++) {
            pos[i]                 = pt[off + i];
            pos[(size_t)n + i]     = ph[off + i];
            pos[(size_t)n * 2 + i] = pw[off + i];
            pos[(size_t)n * 3 + i] = 0;   // last pos dim is unused
        }
        std::vector<int32_t>        n_seq_id(n, 1);
        std::vector<llama_seq_id *> seq_ids(n + 1, &seq0);
        std::vector<int8_t>         logits(n, 0);
        seq_ids[n] = nullptr;
        llama_batch b = {
            /*n_tokens =*/ n,
            /*tokens   =*/ nullptr,
            /*embd     =*/ const_cast<float *>(embd) + (size_t)off * n_embd,
            /*pos      =*/ pos.data(),
            /*n_seq_id =*/ n_seq_id.data(),
            /*seq_id   =*/ seq_ids.data(),
            /*logits   =*/ logits.data(),
        };
        if (llama_decode(ctx.lctx, b) != 0) return false;
    }
    return true;
}

static std::vector<float> log_softmax(const float * logits, int n) {
    double mx = logits[0];
    for (int i = 1; i < n; i++) mx = std::max(mx, (double)logits[i]);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += std::exp((double)logits[i] - mx);
    const double lz = mx + std::log(sum);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) out[i] = (float)((double)logits[i] - lz);
    return out;
}

static double kl_div(const std::vector<float> & lp, const std::vector<float> & lq) {
    double kl = 0;
    for (size_t i = 0; i < lp.size(); i++) {
        const double p = std::exp((double)lp[i]);
        if (p > 1e-12) kl += p * ((double)lp[i] - (double)lq[i]);
    }
    return kl;
}

static int argmax(const std::vector<float> & v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

static bool feed_token(vlm_context & ctx, llama_token id) {
    common_batch_clear(ctx.batch);
    common_batch_add(ctx.batch, id, ctx.n_past++, {0}, true);
    return llama_decode(ctx.lctx, ctx.batch) == 0;
}

// Prefill the prompt with the given visual tokens, then
//   - if `base` is set, walk the baseline caption and compare distributions,
//   - generate greedily (the baseline run records its distributions in `base_logp`).
// Tokens flagged in `dropped` are left out of the prompt.
static llm_run run_llm(vlm_context & ctx, const mtmd::input_chunks & chunks, int nx, int ny, int n_embd,
                       const std::vector<std::vector<float>> & input,
                       const std::vector<std::vector<uint8_t>> * dropped,
                       const llm_run * base, std::vector<std::vector<float>> * base_logp, int n_predict) {
    llm_run run;
    ctx.reset();
    const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
    const size_t n_chunks = chunks.size();
    int pair = 0;
    int64_t t0 = ggml_time_us();
    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n_past = ctx.n_past;
            if (mtmd_helper_eval_chunk_single(ctx.ctx_vision.get(), ctx.lctx, chunk, ctx.n_past, 0,
                                              ctx.n_batch, ci == n_chunks - 1, &new_n_past) != 0) return run;
            ctx.n_past = new_n_past;
            continue;
        }
        const llama_pos st = ctx.n_past;
        std::vector<float> embd;
        std::vector<llama_pos> pt, ph, pw;
        for (int i = 0; i < nx * ny; i++) {
            if (dropped && (*dropped)[pair][i]) continue;
            embd.insert(embd.end(), &input[pair][(size_t)i * n_embd], &input[pair][(size_t)(i + 1) * n_embd]);
            pt.push_back(st); ph.push_back(st + i / nx); pw.push_back(st + i % nx);
        }
        if (!pt.empty() && !decode_visual(ctx, embd.data(), (int)pt.size(), n_embd, pt, ph, pw)) return run;
        run.n_visual += (int)pt.size();
        ctx.n_past = st + std::max(nx, ny);
        pair++;
    }
    llama_synchronize(ctx.lctx);
    run.prefill_ms = (ggml_time_us() - t0) / 1000.0f;

    const llama_pos prompt_end = ctx.n_past;
    const std::vector<float> first = log_softmax(llama_get_logits_ith(ctx.lctx, -1), n_vocab);

    if (base) {
        // distribution at every step of the baseline caption
        double kl = 0;
        int agree = 0;
        const size_t steps = base->tokens.size();
        std::vector<float> lp = first;
        for (size_t s = 0; s < steps; s++) {
            const double k = kl_div((*base_logp)[s], lp);
            if (s == 0) run.kl_first = k;
            kl += k;
            agree += argmax(lp) == base->tokens[s];
            if (s + 1 < steps) {
                if (!feed_token(ctx, base->tokens[s])) return run;
                lp = log_softmax(llama_get_logits_ith(ctx.lctx, -1), n_vocab);
            }
        }
        run.kl_mean = steps ? kl / steps : 0;
        run.top1_agree = steps ? (double)agree / steps : 1;
        llama_memory_seq_rm(llama_get_memory(ctx.lctx), 0, prompt_end, -1);
        ctx.n_past = prompt_end;
    }

    std::vector<float> lp = first;
    for (int i = 0; i < n_predict; i++) {
        const llama_token id = argmax(lp);
        run.tokens.push_back(id);
        if (!base && base_logp) base_logp->push_back(lp);
        if (llama_vocab_is_eog(ctx.vocab, id)) break;
        run.text += common_token_to_piece(ctx.lctx, id);
        if (i + 1 == n_predict) break;
        if (!feed_token(ctx, id)) return run;
        lp = log_softmax(llama_get_logits_ith(ctx.lctx, -1), n_vocab);
    }
    run.ok = true;
    return run;
}

// ============================================================================
// Section 6: One video
// ============================================================================

static std::string sanitize_utf8(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        const unsigned char c = (unsigned char) in[i];
        size_t len = 0;
        if      (c < 0x80)            len = 1;
        else if ((c & 0xE0) == 0xC0)  len = 2;
        else if ((c & 0xF0) == 0xE0)  len = 3;
        else if ((c & 0xF8) == 0xF0)  len = 4;
        bool ok = len > 0 && i + len <= in.size();
        for (size_t k = 1; ok && k < len; k++) {
            if (((unsigned char) in[i + k] & 0xC0) != 0x80) ok = false;
        }
        if (ok) { out.append(in, i, len); i += len; }
        else    { out += "\xEF\xBF\xBD"; i += 1; }   // U+FFFD
    }
    return out;
}

static json run_json(const std::string & name, const std::string & kind, const criterion * c, const llm_run & r,
                     const llm_run & base, int n_tok_all) {
    json j;
    j["name"] = name;
    j["kind"] = kind;
    if (c) {
        j["criterion"] = c->name;
        j["frac_reused"] = c->frac_reused;
        j["frac_rect"] = c->frac_rect;
        j["cos_input"] = c->cos_input;
        j["pairs_all_reused"] = c->pairs_all_reused;
    }
    j["n_visual"] = r.n_visual;
    j["frac_visual"] = (double)r.n_visual / n_tok_all;
    j["prefill_ms"] = r.prefill_ms;
    j["kl_mean"] = r.kl_mean;
    j["kl_first"] = r.kl_first;
    j["top1_agree"] = r.top1_agree;
    j["same_text"] = r.tokens == base.tokens;
    j["n_tokens"] = (int)r.tokens.size();
    j["text"] = sanitize_utf8(r.text);
    return j;
}

static json run_video(vlm_context & ctx, const pr_params & p, const std::string & video,
                      const std::string & prompt, int n_predict) {
    json jv;
    jv["video"] = video;
    const std::string dir = p.data_dir + "/" + p.frames_subdir + "/" + video;

    int n_total = 0;
    for (int i = 1; ; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", i);
        FILE * f = fopen((dir + buf).c_str(), "rb");
        if (!f) break;
        fclose(f);
        n_total++;
    }
    if (n_total == 0) {
        LOG_ERR("No frames for video '%s' in %s\n", video.c_str(), dir.c_str());
        return jv;
    }
    std::vector<int> frame_idx;
    const int n_select = std::min(p.base_frames, n_total);
    for (int k = 0; k < n_select; k++) {
        int idx = p.frame_step > 0 ? 1 + k * p.frame_step
                                   : 1 + (int)std::round((double)k * (n_total - 1) / std::max(n_select - 1, 1));
        if (idx > n_total) break;
        if (frame_idx.empty() || frame_idx.back() != idx) frame_idx.push_back(idx);
    }

    // Frames on the window grid, so a window of pixels is a window of tokens.
    std::vector<rgb_image> frames(frame_idx.size());
    for (size_t i = 0; i < frame_idx.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", frame_idx[i]);
        rgb_image raw;
        if (!load_image(ctx.ctx_vision.get(), dir + buf, raw)) {
            LOG_ERR("Failed to load %s%s\n", dir.c_str(), buf);
            return jv;
        }
        frames[i] = resize_bilinear(raw, round_to(raw.w, WINDOW_PX), round_to(raw.h, WINDOW_PX));
    }
    if (frames.size() % 2) frames.push_back(frames.back());   // odd count: the last frame pairs with itself
    const int n_pairs = (int)frames.size() / 2;

    mtmd::bitmaps bitmaps;
    for (const auto & f : frames) {
        bitmaps.entries.emplace_back((uint32_t)f.w, (uint32_t)f.h, f.px.data());
    }
    std::string content;
    for (int i = 0; i < n_pairs; i++) content += mtmd_default_marker();
    content += "\n" + prompt;

    ctx.reset();
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = content;
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history, msg, true, false);

    mtmd_input_text text;
    text.text          = formatted.c_str();
    text.add_special   = true;
    text.parse_special = true;
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx.ctx_vision.get(), chunks.ptr.get(), &text, bitmaps_c.data(), bitmaps_c.size()) != 0) {
        LOG_ERR("Failed to tokenize prompt\n");
        return jv;
    }

    // every pair through the ViT once: the truth all variants are cut from
    const int n_embd = llama_model_n_embd_inp(ctx.model);
    std::vector<std::vector<float>> truth;
    int nx = 0, ny = 0;
    float vit_ms = 0;
    for (size_t ci = 0; ci < chunks.size(); ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        const auto it = mtmd_input_chunk_get_tokens_image(chunk);
        nx = (int)mtmd_image_tokens_get_nx(it);
        ny = (int)mtmd_image_tokens_get_ny(it);
        int64_t t = ggml_time_us();
        if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) {
            LOG_ERR("Failed to encode pair\n");
            return jv;
        }
        vit_ms += (ggml_time_us() - t) / 1000.0f;
        const float * e = mtmd_get_output_embd(ctx.ctx_vision.get());
        truth.emplace_back(e, e + (size_t)nx * ny * n_embd);
    }
    if ((int)truth.size() != n_pairs || nx * TOKEN_PX != frames[0].w || ny * TOKEN_PX != frames[0].h) {
        LOG_ERR("unexpected grid for %s: %zu pairs, %dx%d tokens\n", video.c_str(), truth.size(), nx, ny);
        return jv;
    }
    const int n_tok = nx * ny, nwx = nx / WINDOW_TOKENS, nwy = ny / WINDOW_TOKENS;

    std::vector<criterion> crits;
    for (float thr : parse_floats(p.pixel_thresholds)) {
        criterion c;
        char nm[32]; snprintf(nm, sizeof(nm), "pixel%g", thr);
        c.name = nm;
        rgb_image ref0 = frames[0], ref1 = frames[1];   // pixels each window was last fresh from
        std::vector<float> cache = truth[0];
        c.reused.emplace_back(n_tok, 0);
        c.input.push_back(truth[0]);
        for (int k = 1; k < n_pairs; k++) {
            const auto s0 = window_diff(frames[2 * k], ref0, nwx, nwy);
            const auto s1 = window_diff(frames[2 * k + 1], ref1, nwx, nwy);
            std::vector<uint8_t> re(n_tok, 0);
            for (int wy = 0; wy < nwy; wy++) {
                for (int wx = 0; wx < nwx; wx++) {
                    const bool same = std::max(s0[(size_t)wy * nwx + wx], s1[(size_t)wy * nwx + wx]) <= thr;
                    if (!same) {
                        copy_window(ref0, frames[2 * k], wx, wy);
                        copy_window(ref1, frames[2 * k + 1], wx, wy);
                    }
                    for (int ty = wy * WINDOW_TOKENS; ty < (wy + 1) * WINDOW_TOKENS; ty++) {
                        for (int tx = wx * WINDOW_TOKENS; tx < (wx + 1) * WINDOW_TOKENS; tx++) {
                            const int i = ty * nx + tx;
                            re[i] = same;
                            if (!same) memcpy(&cache[(size_t)i * n_embd], &truth[k][(size_t)i * n_embd], n_embd * sizeof(float));
                        }
                    }
                }
            }
            c.reused.push_back(re);
            c.input.push_back(cache);
        }
        finish_criterion(c, truth, nx, ny, n_embd);
        crits.push_back(std::move(c));
    }
    for (float thr : parse_floats(p.cos_thresholds)) {
        criterion c;
        char nm[32]; snprintf(nm, sizeof(nm), "cos%.2f", thr);
        c.name = nm;
        std::vector<float> cache = truth[0];
        c.reused.emplace_back(n_tok, 0);
        c.input.push_back(truth[0]);
        for (int k = 1; k < n_pairs; k++) {
            std::vector<uint8_t> re(n_tok, 0);
            for (int i = 0; i < n_tok; i++) {
                re[i] = token_cos(&cache[(size_t)i * n_embd], &truth[k][(size_t)i * n_embd], n_embd) >= thr;
                if (!re[i]) memcpy(&cache[(size_t)i * n_embd], &truth[k][(size_t)i * n_embd], n_embd * sizeof(float));
            }
            c.reused.push_back(re);
            c.input.push_back(cache);
        }
        finish_criterion(c, truth, nx, ny, n_embd);
        crits.push_back(std::move(c));
    }

    // how far the true tokens move from one pair to the next
    {
        std::vector<float> cs;
        for (int k = 1; k < n_pairs; k++) {
            for (int i = 0; i < n_tok; i++) {
                cs.push_back(token_cos(&truth[k - 1][(size_t)i * n_embd], &truth[k][(size_t)i * n_embd], n_embd));
            }
        }
        std::sort(cs.begin(), cs.end());
        json jc;
        if (!cs.empty()) {
            jc["median"] = cs[cs.size() / 2];
            for (float t : {0.95f, 0.90f, 0.80f, 0.70f}) {
                char nm[32]; snprintf(nm, sizeof(nm), "ge_%.2f", t);
                jc[nm] = (double)(cs.end() - std::lower_bound(cs.begin(), cs.end(), t)) / cs.size();
            }
        }
        jv["consecutive_pair_token_cos"] = jc;
    }

    const int n_tok_all = n_tok * n_pairs;
    std::vector<std::vector<float>> base_logp;
    llm_run base = run_llm(ctx, chunks, nx, ny, n_embd, truth, nullptr, nullptr, &base_logp, n_predict);
    if (!base.ok) {
        LOG_ERR("baseline run failed for %s\n", video.c_str());
        return jv;
    }
    jv["n_frames"] = (int)frame_idx.size();
    jv["n_pairs"] = n_pairs;
    jv["grid"] = {nx, ny};
    jv["vit_ms"] = vit_ms;

    json runs = json::array();
    runs.push_back(run_json("baseline", "none", nullptr, base, base, n_tok_all));
    // the baseline again: any difference here is run-to-run noise
    {
        llm_run r = run_llm(ctx, chunks, nx, ny, n_embd, truth, nullptr, &base, &base_logp, n_predict);
        if (r.ok) runs.push_back(run_json("baseline_repeat", "none", nullptr, r, base, n_tok_all));
    }
    for (const auto & c : crits) {
        llm_run s = run_llm(ctx, chunks, nx, ny, n_embd, c.input, nullptr, &base, &base_logp, n_predict);
        if (s.ok) runs.push_back(run_json("stale_" + c.name, "stale", &c, s, base, n_tok_all));
        llm_run d = run_llm(ctx, chunks, nx, ny, n_embd, truth, &c.reused, &base, &base_logp, n_predict);
        if (d.ok) runs.push_back(run_json("drop_" + c.name, "drop", &c, d, base, n_tok_all));
    }
    jv["runs"] = runs;
    return jv;
}

// ============================================================================
// Section 7: Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    pr_params p;
    auto filtered_argv = filter_custom_args(argc, argv, p);
    common_params params;
    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }
    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (p.output_json.empty()) {
        LOG_ERR("--output-json is required\n");
        return 1;
    }
    // two bitmaps per media marker: read once by the tokenizer, so set before any tokenizing
    setenv("MTMD_TEMPORAL_PAIR", "1", 1);

    std::vector<std::string> videos;
    {
        std::string cur;
        for (char c : p.videos + ",") {
            if (c == ',') { if (!cur.empty()) videos.push_back(cur); cur.clear(); }
            else cur += c;
        }
    }
    if (videos.empty()) {
        LOG_ERR("--videos parsed to an empty list\n");
        return 1;
    }

    const int n_predict = params.n_predict < 0 ? 128 : params.n_predict;
    const std::string prompt = params.prompt.empty()
        ? "Describe the scene, motion patterns, and key visual elements in these video frames."
        : params.prompt;

    vlm_context ctx(params);

    json out;
    out["base_frames"] = p.base_frames;
    out["frame_step"] = p.frame_step;
    out["n_predict"] = n_predict;
    out["pixel_thresholds"] = p.pixel_thresholds;
    out["cos_thresholds"] = p.cos_thresholds;

    json jvideos = json::array();
    for (const auto & v : videos) {
        int64_t t = ggml_time_us();
        json jv = run_video(ctx, p, v, prompt, n_predict);
        if (!jv.contains("runs")) {
            LOG_ERR("video failed: %s\n", v.c_str());
            continue;
        }
        LOG_INF("%s: %d pairs, %zu runs, %.1f s\n", v.c_str(), jv["n_pairs"].get<int>(), jv["runs"].size(),
                (ggml_time_us() - t) / 1e6);
        jvideos.push_back(jv);
    }
    out["videos"] = jvideos;

    std::ofstream f(p.output_json);
    f << out.dump(1) << std::endl;
    LOG_INF("results written to %s\n", p.output_json.c_str());
    return 0;
}
