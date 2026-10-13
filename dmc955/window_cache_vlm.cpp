// Window-aligned embedding cache for Qwen2.5-VL video inference
//
// Three per-frame ViT policies run on the SAME uniformly sampled frames inside
// one loaded model, so their latency and outputs are directly comparable:
//
//   full    every frame through the whole ViT (the reference).
//   legacy  the policy of adaptive_patch_cache_vlm.cpp: optical-flow activity
//           on an NxN image grid, token-level bounding-box crop taken from the
//           original-resolution bitmap, <5% active skips the ViT.
//   window  reuse is decided per ViT attention window (112 px = 4x4 tokens):
//           the frame is pre-resized to a multiple of 112 px, a window is dirty
//           when its pixels differ from the frame it was LAST ENCODED from, and
//           the crop is a rectangle of whole windows. A crop taken on that grid
//           is partitioned into the same windows as the full frame, so the 28
//           window-local ViT layers compute what they would have computed in
//           the full frame; only the 4 global layers see less context.
//
// --bench-vit times the ViT at every window-rectangle shape instead.
// --verify re-encodes every frame in full and reports how far the embeddings
// handed to the LLM are from that reference (not for latency runs).

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
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
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <fstream>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Parameters
// ============================================================================

static const int TOKEN_PX      = 28;   // one LLM token = 2x2 patches of 14 px
static const int WINDOW_PX     = 112;  // ViT attention window
static const int WINDOW_TOKENS = WINDOW_PX / TOKEN_PX;

struct wc_params {
    std::string sintel_dir    = "/data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel";
    std::string frames_subdir = "training/clean";
    std::string flow_subdir   = "training/flow";
    std::string videos        = "alley_1";   // comma-separated
    std::string policy        = "window";    // full | legacy | window
    std::string output_json;
    int   base_frames = 8;
    int   frame_step  = 0;      // 0 = spread base_frames evenly over the clip; N = every Nth frame from the start
    int   rounds      = 1;
    int   warmup      = 1;      // unrecorded requests before measuring
    int   align       = 0;      // 0 = let mtmd resize; 112 = pre-resize to the window grid
    float diff_threshold = 6.0f; // window policy: dirty if a token cell's mean |dRGB| exceeds this (0..255)
    int   halo        = 0;      // window policy: grow the dirty rectangle by this many windows
    bool  cost_model  = true;   // window policy: pick the crop shape from measured encode times
    float rekey_frac  = 0.75f;  // without the cost model: crops above this fraction become full encodes
    int   calib_reps  = 3;
    bool  verify      = false;
    bool  bench_vit   = false;
    bool  study       = false;
    int   bench_reps  = 6;
    // legacy policy (defaults of run_embedding_cache_experiment_v2.sh)
    int   patch_grid        = 8;
    float motion_threshold  = 2.0f;
    float spatial_threshold = 0.05f;
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, wc_params & p) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        const bool has = i + 1 < argc;
        if      (arg == "--sintel-dir" && has)        p.sintel_dir = argv[++i];
        else if (arg == "--frames-subdir" && has)     p.frames_subdir = argv[++i];
        else if (arg == "--flow-subdir" && has)       p.flow_subdir = argv[++i];
        else if (arg == "--videos" && has)            p.videos = argv[++i];
        else if (arg == "--policy" && has)            p.policy = argv[++i];
        else if (arg == "--output-json" && has)       p.output_json = argv[++i];
        else if (arg == "--base-frames" && has)       p.base_frames = std::stoi(argv[++i]);
        else if (arg == "--frame-step" && has)        p.frame_step = std::stoi(argv[++i]);
        else if (arg == "--rounds" && has)            p.rounds = std::stoi(argv[++i]);
        else if (arg == "--warmup-requests" && has)   p.warmup = std::stoi(argv[++i]);
        else if (arg == "--align" && has)             p.align = std::stoi(argv[++i]);
        else if (arg == "--diff-threshold" && has)    p.diff_threshold = std::stof(argv[++i]);
        else if (arg == "--halo" && has)              p.halo = std::stoi(argv[++i]);
        else if (arg == "--no-cost-model")            p.cost_model = false;
        else if (arg == "--rekey-frac" && has)        p.rekey_frac = std::stof(argv[++i]);
        else if (arg == "--calib-reps" && has)        p.calib_reps = std::stoi(argv[++i]);
        else if (arg == "--verify")                   p.verify = true;
        else if (arg == "--bench-vit")                p.bench_vit = true;
        else if (arg == "--study")                    p.study = true;
        else if (arg == "--bench-reps" && has)        p.bench_reps = std::stoi(argv[++i]);
        else if (arg == "--patch-grid" && has)        p.patch_grid = std::stoi(argv[++i]);
        else if (arg == "--motion-threshold" && has)  p.motion_threshold = std::stof(argv[++i]);
        else if (arg == "--spatial-threshold" && has) p.spatial_threshold = std::stof(argv[++i]);
        else filtered.push_back(argv[i]);
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nWindow-aligned embedding cache benchmark\n\nUsage: %s [options]\n\n", argv[0]);
    printf("  --policy <full|legacy|window>  per-frame ViT policy (default: window)\n");
    printf("  --videos <a,b,c>      videos under <sintel-dir>/<frames-subdir>\n");
    printf("  --base-frames <n>     uniformly sampled frames per request (default: 8)\n");
    printf("  --frame-step <n>      take every nth frame from the start instead (default: 0 = spread evenly)\n");
    printf("  --rounds <n>          passes over the video list (default: 1)\n");
    printf("  --warmup-requests <n> unrecorded requests first (default: 1)\n");
    printf("  --align <0|112>       pre-resize frames to a multiple of this many px (default: 0)\n");
    printf("  --diff-threshold <f>  window policy dirty threshold, mean |dRGB| 0..255 (default: 6)\n");
    printf("  --halo <n>            window policy: grow the dirty rectangle by n windows\n");
    printf("  --no-cost-model       window policy: always crop unless above --rekey-frac\n");
    printf("  --verify              compare LLM-input embeddings with a full encode of each frame\n");
    printf("  --bench-vit           time the ViT at every window-rectangle shape and exit\n");
    printf("  --study               measure what reuse and cropping do to the embeddings and exit\n");
    printf("  --output-json <f>     results file\n");
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
        float fy = std::max(0.0f, (y + 0.5f) * sy - 0.5f);
        int y0 = std::min((int)fy, src.h - 1), y1 = std::min(y0 + 1, src.h - 1);
        float wy = fy - y0;
        for (int x = 0; x < W; x++) {
            float fx = std::max(0.0f, (x + 0.5f) * sx - 0.5f);
            int x0 = std::min((int)fx, src.w - 1), x1 = std::min(x0 + 1, src.w - 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float a = src.px[((size_t)y0 * src.w + x0) * 3 + c], b = src.px[((size_t)y0 * src.w + x1) * 3 + c];
                float d = src.px[((size_t)y1 * src.w + x0) * 3 + c], e = src.px[((size_t)y1 * src.w + x1) * 3 + c];
                float v = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
                dst.px[((size_t)y * W + x) * 3 + c] = (unsigned char)std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
    return dst;
}

static rgb_image crop_image(const rgb_image & src, int x, int y, int w, int h) {
    rgb_image dst;
    dst.w = w; dst.h = h;
    dst.px.resize((size_t)w * h * 3);
    for (int r = 0; r < h; r++) {
        memcpy(&dst.px[(size_t)r * w * 3], &src.px[((size_t)(y + r) * src.w + x) * 3], (size_t)w * 3);
    }
    return dst;
}

static void copy_region(rgb_image & dst, const rgb_image & src, int x, int y, int w, int h) {
    for (int r = 0; r < h; r++) {
        size_t off = ((size_t)(y + r) * src.w + x) * 3;
        memcpy(&dst.px[off], &src.px[off], (size_t)w * 3);
    }
}

static int round_to(int v, int f) { return std::max(f, (int)std::lround((double)v / f) * f); }

// Per-window change score: the largest mean |dRGB| among the window's 4x4 token
// cells. Taking the max over cells keeps a small moving object from being
// averaged away by the static remainder of its window.
static std::vector<float> window_diff(const rgb_image & cur, const rgb_image & ref, int nwx, int nwy) {
    std::vector<float> score((size_t)nwx * nwy, 0.0f);
    const int ntx = nwx * WINDOW_TOKENS, nty = nwy * WINDOW_TOKENS;
    for (int ty = 0; ty < nty; ty++) {
        std::vector<uint32_t> acc(ntx, 0);
        for (int r = 0; r < TOKEN_PX; r++) {
            const unsigned char * a = &cur.px[(size_t)(ty * TOKEN_PX + r) * cur.w * 3];
            const unsigned char * b = &ref.px[(size_t)(ty * TOKEN_PX + r) * ref.w * 3];
            for (int tx = 0; tx < ntx; tx++) {
                uint32_t s = 0;
                for (int k = tx * TOKEN_PX * 3; k < (tx + 1) * TOKEN_PX * 3; k++) {
                    s += (uint32_t)std::abs((int)a[k] - (int)b[k]);
                }
                acc[tx] += s;
            }
        }
        for (int tx = 0; tx < ntx; tx++) {
            float m = (float)acc[tx] / (TOKEN_PX * TOKEN_PX * 3);
            float & dst = score[(size_t)(ty / WINDOW_TOKENS) * nwx + tx / WINDOW_TOKENS];
            dst = std::max(dst, m);
        }
    }
    return score;
}

// ============================================================================
// Section 3: Optical flow (legacy policy only)
// ============================================================================

struct optical_flow {
    int width = 0, height = 0;
    std::vector<float> uv;   // interleaved (u, v)
};

static bool read_flo_file(const std::string & path, optical_flow & flow) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    float magic = 0.0f;
    int32_t w = 0, h = 0;
    bool ok = fread(&magic, sizeof(float), 1, f) == 1 && magic == 202021.25f
           && fread(&w, sizeof(int32_t), 1, f) == 1 && fread(&h, sizeof(int32_t), 1, f) == 1;
    if (ok) {
        flow.width = w; flow.height = h;
        flow.uv.resize((size_t)w * h * 2);
        ok = fread(flow.uv.data(), sizeof(float), flow.uv.size(), f) == flow.uv.size();
    }
    fclose(f);
    return ok;
}

// Same selection rule as analyze_spatial_patches() in adaptive_patch_cache_vlm.cpp,
// including its floor of 25% of cells forced active.
static std::vector<bool> legacy_active_cells(const optical_flow & flow, int grid,
                                             float spatial_threshold, float motion_threshold) {
    std::vector<bool> sel(grid * grid, false);
    std::vector<float> mean(grid * grid, 0.0f);
    const int pw = flow.width / grid, ph = flow.height / grid;
    for (int gy = 0; gy < grid; gy++) {
        for (int gx = 0; gx < grid; gx++) {
            const int x0 = gx * pw, y0 = gy * ph;
            const int x1 = (gx == grid - 1) ? flow.width : x0 + pw;
            const int y1 = (gy == grid - 1) ? flow.height : y0 + ph;
            float sum = 0.0f, mx = 0.0f;
            int high = 0, total = 0;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    const float u = flow.uv[((size_t)y * flow.width + x) * 2];
                    const float v = flow.uv[((size_t)y * flow.width + x) * 2 + 1];
                    const float mag = std::sqrt(u * u + v * v);
                    sum += mag;
                    mx = std::max(mx, mag);
                    if (mag > motion_threshold) high++;
                    total++;
                }
            }
            const float mean_m = total > 0 ? sum / total : 0.0f;
            const float ratio  = total > 0 ? (float)high / total : 0.0f;
            mean[gy * grid + gx] = mean_m;
            sel[gy * grid + gx] = ratio > spatial_threshold || mean_m > motion_threshold || mx > motion_threshold * 2.0f;
        }
    }
    int n_sel = (int)std::count(sel.begin(), sel.end(), true);
    const int min_cells = std::max(1, grid * grid / 4);
    if (n_sel < min_cells) {
        std::vector<int> idx(grid * grid);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return mean[a] > mean[b]; });
        for (int i : idx) {
            if (n_sel >= min_cells) break;
            if (!sel[i]) { sel[i] = true; n_sel++; }
        }
    }
    return sel;
}

// ============================================================================
// Section 4: Model context
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
// Section 5: Encoding one image region
// ============================================================================

// A standalone image tokenized on its own, kept alive while its chunk is in use.
struct encoded_region {
    mtmd::input_chunks chunks{mtmd_input_chunks_init()};
    const mtmd_input_chunk * chunk = nullptr;
    int nx = 0, ny = 0;
};

static bool tokenize_region(vlm_context & ctx, const rgb_image & img, encoded_region & out) {
    mtmd::bitmap bmp((uint32_t)img.w, (uint32_t)img.h, img.px.data());
    std::string marker(mtmd_default_marker());
    mtmd_input_text text;
    text.text          = marker.c_str();
    text.add_special   = false;
    text.parse_special = true;
    const mtmd_bitmap * bptr = bmp.ptr.get();
    if (mtmd_tokenize(ctx.ctx_vision.get(), out.chunks.ptr.get(), &text, &bptr, 1) != 0) return false;
    for (size_t k = 0; k < out.chunks.size(); k++) {
        if (mtmd_input_chunk_get_type(out.chunks[k]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            out.chunk = out.chunks[k];
            const auto it = mtmd_input_chunk_get_tokens_image(out.chunk);
            out.nx = (int)mtmd_image_tokens_get_nx(it);
            out.ny = (int)mtmd_image_tokens_get_ny(it);
            return true;
        }
    }
    return false;
}

// Measured encode time per window-rectangle shape. `first` is the call right
// after a different shape was encoded (it pays for rebuilding the backend's
// cached execution plan), `steady` is a repeat of the same shape.
struct cost_table {
    int nwx = 0, nwy = 0;
    std::vector<float> first_ms, steady_ms;
    bool valid = false;
    float & first(int w, int h)  { return first_ms[(size_t)(h - 1) * nwx + (w - 1)]; }
    float & steady(int w, int h) { return steady_ms[(size_t)(h - 1) * nwx + (w - 1)]; }
    float cost(int w, int h, int last_w, int last_h) {
        return (w == last_w && h == last_h) ? steady(w, h) : first(w, h);
    }
};

static json bench_shapes(vlm_context & ctx, const rgb_image & frame, int nwx, int nwy, int reps, cost_table * table) {
    json rows = json::array();
    if (table) {
        table->nwx = nwx; table->nwy = nwy;
        table->first_ms.assign((size_t)nwx * nwy, 0.0f);
        table->steady_ms.assign((size_t)nwx * nwy, 0.0f);
    }
    for (int h = 1; h <= nwy; h++) {
        for (int w = 1; w <= nwx; w++) {
            rgb_image crop = crop_image(frame, 0, 0, w * WINDOW_PX, h * WINDOW_PX);
            json times = json::array();
            double prep_ms = 0.0;
            for (int r = 0; r < reps; r++) {
                encoded_region reg;
                int64_t t0 = ggml_time_us();
                if (!tokenize_region(ctx, crop, reg) || reg.nx != w * WINDOW_TOKENS || reg.ny != h * WINDOW_TOKENS) {
                    LOG_ERR("bench: shape %dx%d windows tokenized to %dx%d tokens\n", w, h, reg.nx, reg.ny);
                    return rows;
                }
                int64_t t1 = ggml_time_us();
                if (mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk) != 0) {
                    LOG_ERR("bench: encode failed for %dx%d\n", w, h);
                    return rows;
                }
                int64_t t2 = ggml_time_us();
                prep_ms += (t1 - t0) / 1000.0;
                times.push_back((t2 - t1) / 1000.0);
            }
            double steady = 1e30;
            for (int r = 1; r < reps; r++) steady = std::min(steady, times[r].get<double>());
            if (reps < 2) steady = times[0].get<double>();
            if (table) {
                table->first(w, h)  = (float)(times[0].get<double>() + prep_ms / reps);
                table->steady(w, h) = (float)(steady + prep_ms / reps);
            }
            json row;
            row["w"] = w; row["h"] = h; row["windows"] = w * h; row["tokens"] = w * h * WINDOW_TOKENS * WINDOW_TOKENS;
            row["prep_ms"] = prep_ms / reps;
            row["first_ms"] = times[0];
            row["steady_ms"] = steady;
            row["encode_ms"] = times;
            rows.push_back(row);
        }
    }
    if (table) table->valid = true;
    return rows;
}

// ============================================================================
// Section 6: One request
// ============================================================================

struct frame_record {
    std::string action;          // key | full | crop | skip
    int   n_dirty        = 0;    // dirty windows (window) or active tokens (legacy)
    int   crop_w = 0, crop_h = 0;  // crop size in tokens
    int   tokens_encoded = 0;    // tokens that went through the ViT
    int   tokens_updated = 0;    // tokens replaced in the cache
    float analysis_ms = 0.0f;
    float prep_ms     = 0.0f;    // crop + tokenize of a crop
    float encode_ms   = 0.0f;
    float prefill_ms  = 0.0f;
    // --verify
    float cos_mean = -1.0f, cos_min = -1.0f, cos_p05 = -1.0f;
    float cos_reused = -1.0f, cos_fresh = -1.0f;
    float frac_below_090 = -1.0f;
};

struct request_result {
    std::string text;
    int   n_tokens = 0;
    int   n_frames = 0, nx = 0, ny = 0;
    float load_ms = 0, tokenize_ms = 0, analysis_ms = 0, vit_ms = 0, prefill_ms = 0, decode_ms = 0, total_ms = 0;
    std::vector<frame_record> frames;
    bool ok = false;
};

static void verify_against_full(vlm_context & ctx, const mtmd_input_chunk * chunk, const std::vector<float> & cache,
                                const std::vector<bool> & fresh, int n_tok, int n_embd, frame_record & fr) {
    if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) return;
    const float * ref = mtmd_get_output_embd(ctx.ctx_vision.get());
    std::vector<float> cs(n_tok);
    double s_all = 0, s_re = 0, s_fr = 0;
    int n_re = 0, n_fr = 0, n_low = 0;
    for (int t = 0; t < n_tok; t++) {
        const float * a = &cache[(size_t)t * n_embd];
        const float * b = ref + (size_t)t * n_embd;
        double ab = 0, aa = 0, bb = 0;
        for (int k = 0; k < n_embd; k++) { ab += (double)a[k] * b[k]; aa += (double)a[k] * a[k]; bb += (double)b[k] * b[k]; }
        cs[t] = (float)(ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30));
        s_all += cs[t];
        if (fresh[t]) { s_fr += cs[t]; n_fr++; } else { s_re += cs[t]; n_re++; }
        if (cs[t] < 0.90f) n_low++;
    }
    std::vector<float> sorted = cs;
    std::sort(sorted.begin(), sorted.end());
    fr.cos_mean = (float)(s_all / n_tok);
    fr.cos_min  = sorted.front();
    fr.cos_p05  = sorted[(size_t)(0.05 * (n_tok - 1))];
    fr.cos_reused = n_re > 0 ? (float)(s_re / n_re) : -1.0f;
    fr.cos_fresh  = n_fr > 0 ? (float)(s_fr / n_fr) : -1.0f;
    fr.frac_below_090 = (float)n_low / n_tok;
}

static request_result run_request(vlm_context & ctx, const wc_params & p, const std::string & video,
                                  const std::string & prompt, int n_predict, cost_table & table,
                                  int & last_w, int & last_h) {
    request_result res;
    const int64_t t_start = ggml_time_us();
    const std::string dir = p.sintel_dir + "/" + p.frames_subdir + "/" + video;

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
        return res;
    }
    std::vector<int> frame_idx;
    const int n_select = std::min(p.base_frames, n_total);
    for (int k = 0; k < n_select; k++) {
        int idx = p.frame_step > 0 ? 1 + k * p.frame_step
                                   : 1 + (int)std::round((double)k * (n_total - 1) / std::max(n_select - 1, 1));
        if (idx > n_total) break;
        if (frame_idx.empty() || frame_idx.back() != idx) frame_idx.push_back(idx);
    }

    // Load, and with --align put every frame on the window grid up front so the
    // full frame and every crop of it share one pixel grid.
    std::vector<rgb_image> frames(frame_idx.size());
    for (size_t i = 0; i < frame_idx.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", frame_idx[i]);
        rgb_image raw;
        if (!load_image(ctx.ctx_vision.get(), dir + buf, raw)) {
            LOG_ERR("Failed to load %s%s\n", dir.c_str(), buf);
            return res;
        }
        frames[i] = p.align > 0 ? resize_bilinear(raw, round_to(raw.w, p.align), round_to(raw.h, p.align)) : raw;
    }
    res.load_ms = (ggml_time_us() - t_start) / 1000.0f;

    int64_t t0 = ggml_time_us();
    mtmd::bitmaps bitmaps;
    for (const auto & f : frames) {
        bitmaps.entries.emplace_back((uint32_t)f.w, (uint32_t)f.h, f.px.data());
    }
    std::string content;
    for (size_t i = 0; i < frames.size(); i++) content += mtmd_default_marker();
    content += "\n" + prompt;

    common_chat_msg msg;
    msg.role    = "user";
    msg.content = content;
    const bool add_bos = ctx.chat_history.empty();
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history, msg, true, false);
    ctx.chat_history.push_back(msg);

    mtmd_input_text text;
    text.text          = formatted.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx.ctx_vision.get(), chunks.ptr.get(), &text, bitmaps_c.data(), bitmaps_c.size()) != 0) {
        LOG_ERR("Failed to tokenize prompt\n");
        return res;
    }
    res.tokenize_ms = (ggml_time_us() - t0) / 1000.0f;

    const int n_embd = llama_model_n_embd_inp(ctx.model);
    const bool is_window = p.policy == "window";
    const bool is_legacy = p.policy == "legacy";

    std::vector<float> cache;       // embeddings handed to the LLM, raster order
    rgb_image ref;                  // window policy: pixels each window was last encoded from
    int nx = 0, ny = 0;
    int image_idx = 0;
    const size_t n_chunks = chunks.size();

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            int64_t t = ggml_time_us();
            llama_pos new_n_past = ctx.n_past;
            if (mtmd_helper_eval_chunk_single(ctx.ctx_vision.get(), ctx.lctx, chunk, ctx.n_past, 0,
                                              ctx.n_batch, ci == n_chunks - 1, &new_n_past) != 0) {
                LOG_ERR("Failed to evaluate text chunk %zu\n", ci);
                return res;
            }
            llama_synchronize(ctx.lctx);
            ctx.n_past = new_n_past;
            res.prefill_ms += (ggml_time_us() - t) / 1000.0f;
            continue;
        }

        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        const int cnx = (int)mtmd_image_tokens_get_nx(image_tokens);
        const int cny = (int)mtmd_image_tokens_get_ny(image_tokens);
        const int n_tok = cnx * cny;
        const rgb_image & img = frames[image_idx];
        frame_record fr;
        std::vector<bool> fresh(n_tok, false);

        auto encode_full = [&](const char * action) -> bool {
            int64_t t = ggml_time_us();
            if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) return false;
            fr.encode_ms = (ggml_time_us() - t) / 1000.0f;
            fr.action = action;
            fr.tokens_encoded = n_tok;
            return true;
        };

        const bool have_cache = image_idx > 0 && !cache.empty() && cnx == nx && cny == ny;

        if (!have_cache || p.policy == "full") {
            if (!encode_full(image_idx == 0 ? "key" : "full")) return res;
            const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
            nx = cnx; ny = cny;
            cache.assign(embd, embd + (size_t)n_tok * n_embd);
            fresh.assign(n_tok, true);
            fr.tokens_updated = n_tok;
            if (is_window) ref = img;
            last_w = cnx / WINDOW_TOKENS; last_h = cny / WINDOW_TOKENS;

        } else if (is_window) {
            if (img.w % WINDOW_PX || img.h % WINDOW_PX || cnx * TOKEN_PX != img.w || cny * TOKEN_PX != img.h) {
                LOG_ERR("window policy needs --align %d (frame %dx%d px, %dx%d tokens)\n", WINDOW_PX, img.w, img.h, cnx, cny);
                return res;
            }
            const int nwx = cnx / WINDOW_TOKENS, nwy = cny / WINDOW_TOKENS;
            int64_t t = ggml_time_us();
            std::vector<float> score = window_diff(img, ref, nwx, nwy);
            int x0 = nwx, x1 = -1, y0 = nwy, y1 = -1;
            for (int wy = 0; wy < nwy; wy++) {
                for (int wx = 0; wx < nwx; wx++) {
                    if (score[(size_t)wy * nwx + wx] > p.diff_threshold) {
                        fr.n_dirty++;
                        x0 = std::min(x0, wx); x1 = std::max(x1, wx);
                        y0 = std::min(y0, wy); y1 = std::max(y1, wy);
                    }
                }
            }
            fr.analysis_ms = (ggml_time_us() - t) / 1000.0f;

            if (fr.n_dirty == 0) {
                fr.action = "skip";
            } else {
                x0 = std::max(0, x0 - p.halo); y0 = std::max(0, y0 - p.halo);
                x1 = std::min(nwx - 1, x1 + p.halo); y1 = std::min(nwy - 1, y1 + p.halo);
                int bw = x1 - x0 + 1, bh = y1 - y0 + 1;

                // Any rectangle containing the dirty one is a valid choice,
                // including the whole frame and the shape encoded last time
                // (which skips the plan rebuild). Take the cheapest.
                int cw = bw, ch = bh;
                if (p.cost_model && table.valid) {
                    float best = 1e30f;
                    for (int h = bh; h <= nwy; h++) {
                        for (int w = bw; w <= nwx; w++) {
                            float c = table.cost(w, h, last_w, last_h);
                            if (c < best) { best = c; cw = w; ch = h; }
                        }
                    }
                } else if ((float)(bw * bh) > p.rekey_frac * nwx * nwy) {
                    cw = nwx; ch = nwy;
                }

                if (cw == nwx && ch == nwy) {
                    if (!encode_full("full")) return res;
                    const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
                    cache.assign(embd, embd + (size_t)n_tok * n_embd);
                    fresh.assign(n_tok, true);
                    fr.tokens_updated = n_tok;
                    ref = img;
                } else {
                    // Keep the dirty rectangle inside the (possibly larger) chosen one.
                    int cx = std::min(x0, nwx - cw), cy = std::min(y0, nwy - ch);
                    t = ggml_time_us();
                    rgb_image crop = crop_image(img, cx * WINDOW_PX, cy * WINDOW_PX, cw * WINDOW_PX, ch * WINDOW_PX);
                    encoded_region reg;
                    if (!tokenize_region(ctx, crop, reg) || reg.nx != cw * WINDOW_TOKENS || reg.ny != ch * WINDOW_TOKENS) {
                        LOG_ERR("crop %dx%d windows tokenized to %dx%d tokens\n", cw, ch, reg.nx, reg.ny);
                        return res;
                    }
                    fr.prep_ms = (ggml_time_us() - t) / 1000.0f;
                    t = ggml_time_us();
                    if (mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk) != 0) return res;
                    fr.encode_ms = (ggml_time_us() - t) / 1000.0f;
                    const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
                    for (int r = 0; r < reg.ny; r++) {
                        const int ty = cy * WINDOW_TOKENS + r, tx = cx * WINDOW_TOKENS;
                        memcpy(&cache[((size_t)ty * nx + tx) * n_embd], embd + (size_t)r * reg.nx * n_embd,
                               (size_t)reg.nx * n_embd * sizeof(float));
                        for (int c = 0; c < reg.nx; c++) fresh[(size_t)ty * nx + tx + c] = true;
                    }
                    copy_region(ref, img, cx * WINDOW_PX, cy * WINDOW_PX, cw * WINDOW_PX, ch * WINDOW_PX);
                    fr.action = "crop";
                    fr.crop_w = reg.nx; fr.crop_h = reg.ny;
                    fr.tokens_encoded = reg.nx * reg.ny;
                    fr.tokens_updated = reg.nx * reg.ny;
                }
                last_w = cw; last_h = ch;
            }

        } else if (is_legacy) {
            int64_t t = ggml_time_us();
            char buf[64];
            snprintf(buf, sizeof(buf), "/frame_%04d.flo", frame_idx[image_idx]);
            optical_flow flow;
            const bool have_flow = read_flo_file(p.sintel_dir + "/" + p.flow_subdir + "/" + video + buf, flow);
            std::vector<bool> active(n_tok, true);
            if (have_flow) {
                auto cells = legacy_active_cells(flow, p.patch_grid, p.spatial_threshold, p.motion_threshold);
                for (int ty = 0; ty < cny; ty++) {
                    for (int tx = 0; tx < cnx; tx++) {
                        int gx = std::min(tx * p.patch_grid / cnx, p.patch_grid - 1);
                        int gy = std::min(ty * p.patch_grid / cny, p.patch_grid - 1);
                        active[(size_t)ty * cnx + tx] = cells[gy * p.patch_grid + gx];
                    }
                }
            }
            fr.n_dirty = (int)std::count(active.begin(), active.end(), true);
            fr.analysis_ms = (ggml_time_us() - t) / 1000.0f;

            if (have_flow && (float)fr.n_dirty / n_tok < 0.05f) {
                fr.action = "skip";
            } else {
                int tx0 = cnx, tx1 = -1, ty0 = cny, ty1 = -1;
                for (int ty = 0; ty < cny; ty++) {
                    for (int tx = 0; tx < cnx; tx++) {
                        if (active[(size_t)ty * cnx + tx]) {
                            tx0 = std::min(tx0, tx); tx1 = std::max(tx1, tx);
                            ty0 = std::min(ty0, ty); ty1 = std::max(ty1, ty);
                        }
                    }
                }
                const int bnx = tx1 - tx0 + 1, bny = ty1 - ty0 + 1;
                // token box -> pixels of the original image, as the legacy code does
                const float isx = (float)img.w / (cnx * TOKEN_PX), isy = (float)img.h / (cny * TOKEN_PX);
                int px = std::max(0, std::min((int)(tx0 * TOKEN_PX * isx), img.w - 1));
                int py = std::max(0, std::min((int)(ty0 * TOKEN_PX * isy), img.h - 1));
                int pw = std::min((int)(bnx * TOKEN_PX * isx), img.w - px);
                int ph = std::min((int)(bny * TOKEN_PX * isy), img.h - py);

                t = ggml_time_us();
                rgb_image crop = crop_image(img, px, py, pw, ph);
                encoded_region reg;
                const bool crop_ok = have_flow && tokenize_region(ctx, crop, reg) && reg.nx == bnx && reg.ny == bny;
                fr.prep_ms = (ggml_time_us() - t) / 1000.0f;

                if (crop_ok) {
                    t = ggml_time_us();
                    if (mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk) != 0) return res;
                    fr.encode_ms = (ggml_time_us() - t) / 1000.0f;
                    const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
                    for (int r = 0; r < bny; r++) {
                        for (int c = 0; c < bnx; c++) {
                            const size_t full = (size_t)(ty0 + r) * cnx + tx0 + c;
                            if (!active[full]) continue;
                            memcpy(&cache[full * n_embd], embd + ((size_t)r * bnx + c) * n_embd, n_embd * sizeof(float));
                            fresh[full] = true;
                            fr.tokens_updated++;
                        }
                    }
                    fr.action = "crop";
                    fr.crop_w = bnx; fr.crop_h = bny;
                    fr.tokens_encoded = bnx * bny;
                } else {
                    if (!encode_full("full")) return res;
                    const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());
                    for (int i = 0; i < n_tok; i++) {
                        if (!active[i]) continue;
                        memcpy(&cache[(size_t)i * n_embd], embd + (size_t)i * n_embd, n_embd * sizeof(float));
                        fresh[i] = true;
                        fr.tokens_updated++;
                    }
                }
            }
        }

        res.analysis_ms += fr.analysis_ms;
        res.vit_ms += fr.prep_ms + fr.encode_ms;

        if (p.verify) {
            verify_against_full(ctx, chunk, cache, fresh, n_tok, n_embd, fr);
            last_w = cnx / WINDOW_TOKENS; last_h = cny / WINDOW_TOKENS;
        }

        int64_t t = ggml_time_us();
        llama_pos new_n_past = ctx.n_past;
        if (mtmd_helper_decode_image_chunk(ctx.ctx_vision.get(), ctx.lctx, chunk, cache.data(),
                                           ctx.n_past, 0, ctx.n_batch, &new_n_past) != 0) {
            LOG_ERR("Failed to decode image chunk %zu\n", ci);
            return res;
        }
        llama_synchronize(ctx.lctx);
        ctx.n_past = new_n_past;
        fr.prefill_ms = (ggml_time_us() - t) / 1000.0f;
        res.prefill_ms += fr.prefill_ms;

        res.frames.push_back(fr);
        image_idx++;
    }
    res.n_frames = image_idx;
    res.nx = nx; res.ny = ny;

    int64_t t_gen = ggml_time_us();
    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        common_sampler_accept(ctx.smpl, id, true);
        res.n_tokens++;
        if (llama_vocab_is_eog(ctx.vocab, id)) break;
        res.text += common_token_to_piece(ctx.lctx, id);
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("Failed to decode token\n");
            break;
        }
    }
    llama_synchronize(ctx.lctx);
    res.decode_ms = (ggml_time_us() - t_gen) / 1000.0f;
    res.total_ms  = (ggml_time_us() - t_start) / 1000.0f;
    res.ok = true;
    return res;
}


// ============================================================================
// Section 6b: Embedding study
// ============================================================================
//
// What the reuse decision needs to know, measured directly on full encodes:
//   pairs    for consecutive selected frames, each window's pixel-change score
//            next to how far its embedding actually moved.
//   crops    how close a crop's embeddings are to the same tokens of the full
//            frame, for crops on the window grid and shifted half a window off it.
//   mosaics  the same for scattered windows packed into one rectangle.

static bool encode_image(vlm_context & ctx, const rgb_image & img, std::vector<float> & out, int & nx, int & ny) {
    encoded_region reg;
    if (!tokenize_region(ctx, img, reg)) return false;
    if (mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk) != 0) return false;
    const int n_embd = llama_model_n_embd_inp(ctx.model);
    const float * e = mtmd_get_output_embd(ctx.ctx_vision.get());
    nx = reg.nx; ny = reg.ny;
    out.assign(e, e + (size_t)nx * ny * n_embd);
    return true;
}

static float token_cos(const float * a, const float * b, int n_embd) {
    double ab = 0, aa = 0, bb = 0;
    for (int k = 0; k < n_embd; k++) { ab += (double)a[k] * b[k]; aa += (double)a[k] * a[k]; bb += (double)b[k] * b[k]; }
    return (float)(ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30));
}

static json cos_stats(std::vector<float> cs) {
    std::sort(cs.begin(), cs.end());
    double s = 0;
    for (float c : cs) s += c;
    json j;
    j["n"] = cs.size();
    j["mean"] = s / cs.size();
    j["min"] = cs.front();
    j["p05"] = cs[(size_t)(0.05 * (cs.size() - 1))];
    j["median"] = cs[cs.size() / 2];
    return j;
}

static json run_study(vlm_context & ctx, const wc_params & p, const std::string & video) {
    json out;
    out["video"] = video;
    const std::string dir = p.sintel_dir + "/" + p.frames_subdir + "/" + video;
    const int n_embd = llama_model_n_embd_inp(ctx.model);

    int n_total = 0;
    for (int i = 1; ; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", i);
        FILE * f = fopen((dir + buf).c_str(), "rb");
        if (!f) break;
        fclose(f);
        n_total++;
    }
    std::vector<int> frame_idx;
    const int n_select = std::min(p.base_frames, n_total);
    for (int k = 0; k < n_select; k++) {
        int idx = p.frame_step > 0 ? 1 + k * p.frame_step
                                   : 1 + (int)std::round((double)k * (n_total - 1) / std::max(n_select - 1, 1));
        if (idx > n_total) break;
        if (frame_idx.empty() || frame_idx.back() != idx) frame_idx.push_back(idx);
    }

    std::vector<rgb_image> frames(frame_idx.size());
    std::vector<std::vector<float>> embd(frame_idx.size());
    int nx = 0, ny = 0;
    for (size_t i = 0; i < frame_idx.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", frame_idx[i]);
        rgb_image raw;
        if (!load_image(ctx.ctx_vision.get(), dir + buf, raw)) return out;
        frames[i] = resize_bilinear(raw, round_to(raw.w, WINDOW_PX), round_to(raw.h, WINDOW_PX));
        if (!encode_image(ctx, frames[i], embd[i], nx, ny)) return out;
    }
    const int nwx = nx / WINDOW_TOKENS, nwy = ny / WINDOW_TOKENS;
    out["grid_windows"] = {nwx, nwy};

    json pairs = json::array();
    for (size_t i = 1; i < frames.size(); i++) {
        std::vector<float> score = window_diff(frames[i], frames[i - 1], nwx, nwy);
        for (int wy = 0; wy < nwy; wy++) {
            for (int wx = 0; wx < nwx; wx++) {
                double s = 0;
                float mn = 1.0f;
                for (int r = 0; r < WINDOW_TOKENS; r++) {
                    for (int c = 0; c < WINDOW_TOKENS; c++) {
                        const size_t t = (size_t)(wy * WINDOW_TOKENS + r) * nx + wx * WINDOW_TOKENS + c;
                        float cs = token_cos(&embd[i][t * n_embd], &embd[i - 1][t * n_embd], n_embd);
                        s += cs;
                        mn = std::min(mn, cs);
                    }
                }
                pairs.push_back({score[(size_t)wy * nwx + wx], s / (WINDOW_TOKENS * WINDOW_TOKENS), mn});
            }
        }
    }
    out["pairs_score_cosmean_cosmin"] = pairs;

    // Crops of the middle frame against its own full encode.
    const size_t mid = frames.size() / 2;
    const rgb_image & img = frames[mid];
    const std::vector<float> & full = embd[mid];
    struct rect { int x, y, w, h; };
    std::vector<rect> rects = {
        {0, 0, std::max(1, nwx / 2), nwy}, {0, 0, nwx, std::max(1, nwy / 2)},
        {nwx / 4, nwy / 4, std::max(1, nwx / 2), std::max(1, nwy / 2)},
        {0, 0, 1, 1}, {nwx / 2, nwy / 2, 1, 1}, {nwx / 3, 0, std::min(2, nwx), std::min(2, nwy)},
    };
    json crops = json::array();
    for (const auto & r : rects) {
        for (int shift = 0; shift <= 1; shift++) {
            // shift = 1 moves the crop half a window off the grid, keeping it on the token grid
            const int off = shift * WINDOW_PX / 2;
            int px = r.x * WINDOW_PX + off, py = r.y * WINDOW_PX + off;
            if (px + r.w * WINDOW_PX > img.w) px = r.x * WINDOW_PX - off;
            if (py + r.h * WINDOW_PX > img.h) py = r.y * WINDOW_PX - off;
            if (px < 0 || py < 0) continue;
            std::vector<float> ce;
            int cnx = 0, cny = 0;
            if (!encode_image(ctx, crop_image(img, px, py, r.w * WINDOW_PX, r.h * WINDOW_PX), ce, cnx, cny)) continue;
            std::vector<float> cs;
            for (int rr = 0; rr < cny; rr++) {
                for (int c = 0; c < cnx; c++) {
                    const size_t t = (size_t)(py / TOKEN_PX + rr) * nx + px / TOKEN_PX + c;
                    cs.push_back(token_cos(&ce[((size_t)rr * cnx + c) * n_embd], &full[t * n_embd], n_embd));
                }
            }
            // Sanity for the token mapping: how often is a crop token's nearest
            // full-frame token the one at its own position?
            int n_match = 0;
            for (int rr = 0; rr < cny; rr++) {
                for (int c = 0; c < cnx; c++) {
                    const size_t want = (size_t)(py / TOKEN_PX + rr) * nx + px / TOKEN_PX + c;
                    size_t best = 0;
                    float best_cs = -2.0f;
                    for (size_t t = 0; t < (size_t)nx * ny; t++) {
                        float v = token_cos(&ce[((size_t)rr * cnx + c) * n_embd], &full[t * n_embd], n_embd);
                        if (v > best_cs) { best_cs = v; best = t; }
                    }
                    if (best == want) n_match++;
                }
            }
            json j = cos_stats(cs);
            j["nearest_is_self"] = (double)n_match / (cnx * cny);
            j["windows"] = {r.w, r.h};
            j["on_grid"] = shift == 0;
            crops.push_back(j);
        }
    }
    out["crops"] = crops;

    // Scattered windows packed into one rectangle.
    json mosaics = json::array();
    const std::vector<rect> packs = {{0, 0, 2, 1}, {0, 0, 2, 2}, {0, 0, 3, 3}, {0, 0, 6, 3}, {0, 0, 9, 3}};
    uint32_t rng = 12345;
    for (const auto & pk : packs) {
        const int k = pk.w * pk.h;
        if (k >= nwx * nwy) continue;
        std::vector<int> ids(nwx * nwy);
        std::iota(ids.begin(), ids.end(), 0);
        for (int i = (int)ids.size() - 1; i > 0; i--) {
            rng = rng * 1664525u + 1013904223u;
            std::swap(ids[i], ids[(rng >> 8) % (i + 1)]);
        }
        ids.resize(k);
        std::sort(ids.begin(), ids.end());
        rgb_image mos;
        mos.w = pk.w * WINDOW_PX; mos.h = pk.h * WINDOW_PX;
        mos.px.resize((size_t)mos.w * mos.h * 3);
        for (int i = 0; i < k; i++) {
            const int sx = (ids[i] % nwx) * WINDOW_PX, sy = (ids[i] / nwx) * WINDOW_PX;
            const int dx = (i % pk.w) * WINDOW_PX, dy = (i / pk.w) * WINDOW_PX;
            for (int r = 0; r < WINDOW_PX; r++) {
                memcpy(&mos.px[((size_t)(dy + r) * mos.w + dx) * 3], &img.px[((size_t)(sy + r) * img.w + sx) * 3], (size_t)WINDOW_PX * 3);
            }
        }
        std::vector<float> me;
        int mnx = 0, mny = 0;
        if (!encode_image(ctx, mos, me, mnx, mny)) continue;
        std::vector<float> cs;
        for (int i = 0; i < k; i++) {
            for (int r = 0; r < WINDOW_TOKENS; r++) {
                for (int c = 0; c < WINDOW_TOKENS; c++) {
                    const size_t tm = (size_t)((i / pk.w) * WINDOW_TOKENS + r) * mnx + (i % pk.w) * WINDOW_TOKENS + c;
                    const size_t tf = (size_t)((ids[i] / nwx) * WINDOW_TOKENS + r) * nx + (ids[i] % nwx) * WINDOW_TOKENS + c;
                    cs.push_back(token_cos(&me[tm * n_embd], &full[tf * n_embd], n_embd));
                }
            }
        }
        json j = cos_stats(cs);
        j["windows"] = k;
        mosaics.push_back(j);
    }
    out["mosaics"] = mosaics;
    return out;
}

// ============================================================================
// Section 7: Output
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

static json request_to_json(const request_result & r, const std::string & video, int round) {
    json j;
    j["video"] = video;
    j["round"] = round;
    j["n_frames"] = r.n_frames;
    j["grid"] = {r.nx, r.ny};
    j["load_ms"] = r.load_ms;
    j["tokenize_ms"] = r.tokenize_ms;
    j["analysis_ms"] = r.analysis_ms;
    j["vit_ms"] = r.vit_ms;
    j["prefill_ms"] = r.prefill_ms;
    j["decode_ms"] = r.decode_ms;
    j["total_ms"] = r.total_ms;
    j["n_tokens"] = r.n_tokens;
    j["text"] = sanitize_utf8(r.text);
    json frames = json::array();
    for (const auto & f : r.frames) {
        json jf;
        jf["action"] = f.action;
        jf["n_dirty"] = f.n_dirty;
        jf["crop"] = {f.crop_w, f.crop_h};
        jf["tokens_encoded"] = f.tokens_encoded;
        jf["tokens_updated"] = f.tokens_updated;
        jf["analysis_ms"] = f.analysis_ms;
        jf["prep_ms"] = f.prep_ms;
        jf["encode_ms"] = f.encode_ms;
        jf["prefill_ms"] = f.prefill_ms;
        if (f.cos_mean >= 0.0f) {
            jf["cos_mean"] = f.cos_mean;
            jf["cos_min"] = f.cos_min;
            jf["cos_p05"] = f.cos_p05;
            jf["cos_reused"] = f.cos_reused;
            jf["cos_fresh"] = f.cos_fresh;
            jf["frac_below_090"] = f.frac_below_090;
        }
        frames.push_back(jf);
    }
    j["frames"] = frames;
    return j;
}

// ============================================================================
// Section 8: Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    wc_params p;
    auto filtered_argv = filter_custom_args(argc, argv, p);
    common_params params;
    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }
    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (p.policy != "full" && p.policy != "legacy" && p.policy != "window") {
        LOG_ERR("unknown --policy '%s'\n", p.policy.c_str());
        return 1;
    }
    if (p.policy == "window" && !p.bench_vit && !p.study && p.align != WINDOW_PX) {
        LOG_ERR("--policy window requires --align %d\n", WINDOW_PX);
        return 1;
    }
    if (p.output_json.empty()) {
        LOG_ERR("--output-json is required\n");
        return 1;
    }

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
    out["policy"] = p.policy;
    out["align"] = p.align;
    out["base_frames"] = p.base_frames;
    out["frame_step"] = p.frame_step;
    out["n_predict"] = n_predict;
    out["diff_threshold"] = p.diff_threshold;
    out["halo"] = p.halo;
    out["cost_model"] = p.cost_model;
    out["verify"] = p.verify;
    out["cuda_graphs_env_disabled"] = getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr;

    // A frame on the window grid, for --bench-vit and for calibrating the cost model.
    rgb_image probe;
    int nwx = 0, nwy = 0;
    if (p.bench_vit || (p.policy == "window" && p.cost_model && !p.study)) {
        rgb_image raw;
        const std::string path = p.sintel_dir + "/" + p.frames_subdir + "/" + videos[0] + "/frame_0001.png";
        if (!load_image(ctx.ctx_vision.get(), path, raw)) {
            LOG_ERR("Failed to load %s\n", path.c_str());
            return 1;
        }
        probe = resize_bilinear(raw, round_to(raw.w, WINDOW_PX), round_to(raw.h, WINDOW_PX));
        nwx = probe.w / WINDOW_PX; nwy = probe.h / WINDOW_PX;
    }

    if (p.bench_vit) {
        // Unaligned reference: the frame as mtmd would resize it on its own.
        rgb_image raw;
        load_image(ctx.ctx_vision.get(), p.sintel_dir + "/" + p.frames_subdir + "/" + videos[0] + "/frame_0001.png", raw);
        json native = json::array();
        {
            encoded_region reg;
            if (tokenize_region(ctx, raw, reg)) {
                for (int r = 0; r < p.bench_reps; r++) {
                    int64_t t = ggml_time_us();
                    mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk);
                    native.push_back((ggml_time_us() - t) / 1000.0);
                }
                out["native_grid"] = {reg.nx, reg.ny};
            }
        }
        out["native_encode_ms"] = native;
        out["window_grid"] = {nwx, nwy};
        out["shapes"] = bench_shapes(ctx, probe, nwx, nwy, p.bench_reps, nullptr);

        // Alternate two shapes so every call follows a different one: the cost
        // of never repeating a shape, and whether the backend gives up on
        // caching plans for the ViT.
        auto alternate = [&](int w1, int h1, int w2, int h2, int n) {
            json a = json::array();
            for (int i = 0; i < n; i++) {
                const int w = i % 2 ? w2 : w1, h = i % 2 ? h2 : h1;
                rgb_image crop = crop_image(probe, 0, 0, w * WINDOW_PX, h * WINDOW_PX);
                encoded_region reg;
                if (!tokenize_region(ctx, crop, reg)) break;
                int64_t t = ggml_time_us();
                mtmd_encode_chunk(ctx.ctx_vision.get(), reg.chunk);
                a.push_back({{"w", w}, {"h", h}, {"ms", (ggml_time_us() - t) / 1000.0}});
            }
            return a;
        };
        out["alternating"] = alternate(nwx, nwy, std::max(1, nwx / 2), std::max(1, nwy / 2), 16);
        out["shapes_after_alternating"] = bench_shapes(ctx, probe, nwx, nwy, p.bench_reps, nullptr);

        std::ofstream f(p.output_json);
        f << out.dump(2) << std::endl;
        LOG_INF("bench-vit written to %s\n", p.output_json.c_str());
        return 0;
    }

    if (p.study) {
        json studies = json::array();
        for (const auto & v : videos) studies.push_back(run_study(ctx, p, v));
        out["studies"] = studies;
        std::ofstream f(p.output_json);
        f << out.dump() << std::endl;
        LOG_INF("study written to %s\n", p.output_json.c_str());
        return 0;
    }

    cost_table table;
    if (p.policy == "window" && p.cost_model) {
        int64_t t = ggml_time_us();
        out["calibration"] = bench_shapes(ctx, probe, nwx, nwy, p.calib_reps, &table);
        out["calibration_ms"] = (ggml_time_us() - t) / 1000.0;
    }

    int last_w = 0, last_h = 0;
    for (int i = 0; i < p.warmup; i++) {
        ctx.reset();
        run_request(ctx, p, videos[0], prompt, n_predict, table, last_w, last_h);
    }

    json requests = json::array();
    for (int round = 0; round < p.rounds; round++) {
        for (const auto & v : videos) {
            ctx.reset();
            request_result r = run_request(ctx, p, v, prompt, n_predict, table, last_w, last_h);
            if (!r.ok) {
                LOG_ERR("request failed: %s\n", v.c_str());
                continue;
            }
            LOG_INF("[%s r%d] %s: vit %.1f prefill %.1f decode %.1f total %.1f ms, %d tok\n",
                    p.policy.c_str(), round, v.c_str(), r.vit_ms, r.prefill_ms, r.decode_ms, r.total_ms, r.n_tokens);
            requests.push_back(request_to_json(r, v, round));
        }
    }
    out["requests"] = requests;

    std::ofstream f(p.output_json);
    f << out.dump(2) << std::endl;
    LOG_INF("results written to %s\n", p.output_json.c_str());
    return 0;
}
