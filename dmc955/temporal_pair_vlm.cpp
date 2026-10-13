// Qwen2.5-VL video input the way the reference model takes it
//
// The reference encodes a video two frames at a time: the patch embedding is a
// Conv3d with temporal kernel 2, so each PAIR of frames becomes one grid of
// visual tokens. All pairs of one video sit in a single vision block and share
// their height/width rotary positions; only the temporal position advances.
//
// Modes, same sampled frames for each:
//   images  one frame per ViT pass, one vision block per frame (stock llama.cpp)
//   pairs   two frames per ViT pass, but still one vision block per pair with
//           image-style positions (what MTMD_TEMPORAL_PAIR=1 gives on its own)
//   video   the reference layout: two frames per ViT pass, one vision block for
//           the whole video, video rotary positions
//
// Reference positions (transformers <= 4.5x, Qwen2_5_VLModel.get_rope_index),
// with st = position of the first visual token:
//   t = st + floor(k * second_per_grid_t * tokens_per_second)   k = pair index
//   h = st + row,  w = st + column                               same for every pair
//   next text position = max over all three + 1
// where second_per_grid_t = temporal_patch_size / sampling fps.
//
// --dump-dir writes what the first request per video fed the LLM (frames,
// embeddings, token ids, positions, first-step logits) for verify_temporal_pair.py.

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

struct tp_params {
    std::string data_dir      = "/data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel";
    std::string frames_subdir = "training/clean";
    std::string videos        = "alley_1";   // comma-separated
    std::string mode          = "video";     // images | pairs | video
    std::string output_json;
    std::string dump_dir;
    int   base_frames = 8;
    int   frame_step  = 0;      // 0 = spread base_frames evenly over the clip; N = every Nth frame from the start
    int   rounds      = 1;
    int   warmup      = 1;      // unrecorded requests before measuring
    int   align       = 0;      // 0 = let mtmd resize; N = pre-resize to a multiple of N px
    float video_fps   = 2.0f;   // sampling rate the frames represent (reference default)
    int   tokens_per_second = 2;  // vision_config.tokens_per_second of Qwen2.5-VL
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, tp_params & p) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        const bool has = i + 1 < argc;
        if      (arg == "--sintel-dir" && has)        p.data_dir = argv[++i];
        else if (arg == "--frames-subdir" && has)     p.frames_subdir = argv[++i];
        else if (arg == "--videos" && has)            p.videos = argv[++i];
        else if (arg == "--mode" && has)              p.mode = argv[++i];
        else if (arg == "--output-json" && has)       p.output_json = argv[++i];
        else if (arg == "--dump-dir" && has)          p.dump_dir = argv[++i];
        else if (arg == "--base-frames" && has)       p.base_frames = std::stoi(argv[++i]);
        else if (arg == "--frame-step" && has)        p.frame_step = std::stoi(argv[++i]);
        else if (arg == "--rounds" && has)            p.rounds = std::stoi(argv[++i]);
        else if (arg == "--warmup-requests" && has)   p.warmup = std::stoi(argv[++i]);
        else if (arg == "--align" && has)             p.align = std::stoi(argv[++i]);
        else if (arg == "--video-fps" && has)         p.video_fps = std::stof(argv[++i]);
        else if (arg == "--tokens-per-second" && has) p.tokens_per_second = std::stoi(argv[++i]);
        else filtered.push_back(argv[i]);
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nQwen2.5-VL video input: per-frame images vs reference frame pairs\n\nUsage: %s [options]\n\n", argv[0]);
    printf("  --mode <images|pairs|video>  how frames reach the model (default: video)\n");
    printf("  --videos <a,b,c>      videos under <sintel-dir>/<frames-subdir>\n");
    printf("  --base-frames <n>     uniformly sampled frames per request (default: 8)\n");
    printf("  --frame-step <n>      take every nth frame from the start instead (default: 0 = spread evenly)\n");
    printf("  --video-fps <f>       sampling rate the frames represent, sets the temporal position step (default: 2)\n");
    printf("  --rounds <n>          passes over the video list (default: 1)\n");
    printf("  --warmup-requests <n> unrecorded requests first (default: 1)\n");
    printf("  --align <n>           pre-resize frames to a multiple of n px (default: 0 = mtmd resizes)\n");
    printf("  --dump-dir <d>        write frames, embeddings, token ids, positions and logits per video\n");
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
// Section 4: One request
// ============================================================================

struct request_result {
    std::string text;
    int   n_tokens = 0;          // generated
    int   n_frames = 0, n_groups = 0, nx = 0, ny = 0;
    int   n_visual_tokens = 0, n_prompt_tokens = 0;
    float load_ms = 0, tokenize_ms = 0, vit_ms = 0, prefill_ms = 0, decode_ms = 0, total_ms = 0;
    std::vector<float> group_encode_ms;
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

static void write_bin(const std::string & path, const void * data, size_t n_bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *)data, (std::streamsize)n_bytes);
}

static request_result run_request(vlm_context & ctx, const tp_params & p, const std::string & video,
                                  const std::string & prompt, int n_predict, bool dump) {
    request_result res;
    const int64_t t_start = ggml_time_us();
    const std::string dir = p.data_dir + "/" + p.frames_subdir + "/" + video;
    const bool paired   = p.mode != "images";
    const bool is_video = p.mode == "video";

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
    res.n_frames = (int)frames.size();
    res.load_ms = (ggml_time_us() - t_start) / 1000.0f;

    // One marker per ViT pass. In the paired modes the tokenizer takes two
    // bitmaps per marker and repeats the last frame of an odd count, as the
    // reference does.
    int64_t t0 = ggml_time_us();
    mtmd::bitmaps bitmaps;
    for (const auto & f : frames) {
        bitmaps.entries.emplace_back((uint32_t)f.w, (uint32_t)f.h, f.px.data());
    }
    const size_t n_groups = paired ? (frames.size() + 1) / 2 : frames.size();
    std::string content;
    for (size_t i = 0; i < n_groups; i++) content += mtmd_default_marker();
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
    const size_t n_chunks = chunks.size();
    size_t first_img = n_chunks, last_img = 0;
    for (size_t ci = 0; ci < n_chunks; ci++) {
        if (mtmd_input_chunk_get_type(chunks[ci]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            first_img = std::min(first_img, ci);
            last_img  = ci;
        }
    }

    // --dump-dir: everything the LLM was given, in order
    std::vector<llama_token> dump_tokens;      // -1 marks a visual token
    std::vector<llama_pos>   dump_pos;         // t, h, w per prompt token
    std::vector<float>       dump_embd;

    llama_pos video_st = -1;     // position of the first visual token of the video
    llama_pos video_max = 0;     // largest position used by the video so far
    int group = 0;

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_text = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n_text);
            if (is_video && ci > first_img && ci < last_img) {
                // The separator mtmd puts between two media markers closes one
                // vision block and opens the next. A video is one block, so the
                // separator is dropped; anything else here would be lost text.
                std::string piece;
                for (size_t k = 0; k < n_text; k++) piece += common_token_to_piece(ctx.lctx, toks[k], true);
                if (piece != "<|vision_end|><|vision_start|>") {
                    LOG_ERR("unexpected text between video frames: '%s'\n", piece.c_str());
                    return res;
                }
                continue;
            }
            if (is_video && video_st >= 0 && ci > last_img && ctx.n_past <= video_max) {
                ctx.n_past = video_max + 1;
            }
            int64_t t = ggml_time_us();
            llama_pos new_n_past = ctx.n_past;
            if (mtmd_helper_eval_chunk_single(ctx.ctx_vision.get(), ctx.lctx, chunk, ctx.n_past, 0,
                                              ctx.n_batch, ci == n_chunks - 1, &new_n_past) != 0) {
                LOG_ERR("Failed to evaluate text chunk %zu\n", ci);
                return res;
            }
            llama_synchronize(ctx.lctx);
            for (size_t k = 0; k < n_text; k++) {
                dump_tokens.push_back(toks[k]);
                for (int d = 0; d < 3; d++) dump_pos.push_back(ctx.n_past + (llama_pos)k);
            }
            ctx.n_past = new_n_past;
            res.n_prompt_tokens += (int)n_text;
            res.prefill_ms += (ggml_time_us() - t) / 1000.0f;
            continue;
        }

        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        const int nx = (int)mtmd_image_tokens_get_nx(image_tokens);
        const int ny = (int)mtmd_image_tokens_get_ny(image_tokens);
        const int n_tok = nx * ny;
        if (res.nx != 0 && (nx != res.nx || ny != res.ny)) {
            LOG_ERR("frames of one video must share a token grid (%dx%d vs %dx%d)\n", nx, ny, res.nx, res.ny);
            return res;
        }
        res.nx = nx; res.ny = ny;

        int64_t t = ggml_time_us();
        if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) {
            LOG_ERR("Failed to encode image chunk %zu\n", ci);
            return res;
        }
        const float enc_ms = (ggml_time_us() - t) / 1000.0f;
        res.group_encode_ms.push_back(enc_ms);
        res.vit_ms += enc_ms;
        const float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

        // rotary positions for this grid
        llama_pos st, pos_t;
        if (is_video) {
            if (video_st < 0) video_st = ctx.n_past;
            st = video_st;
            const double second_per_grid_t = 2.0 / p.video_fps;   // temporal_patch_size / fps
            pos_t = st + (llama_pos)std::floor(group * second_per_grid_t * p.tokens_per_second);
        } else {
            st = ctx.n_past;
            pos_t = st;
        }
        std::vector<llama_pos> pt(n_tok), ph(n_tok), pw(n_tok);
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                const int i = y * nx + x;
                pt[i] = pos_t; ph[i] = st + y; pw[i] = st + x;
            }
        }

        t = ggml_time_us();
        if (!decode_visual(ctx, embd, n_tok, n_embd, pt, ph, pw)) {
            LOG_ERR("Failed to decode image chunk %zu\n", ci);
            return res;
        }
        llama_synchronize(ctx.lctx);
        res.prefill_ms += (ggml_time_us() - t) / 1000.0f;

        if (is_video) {
            video_max = std::max(video_max, std::max(pos_t, st + (llama_pos)std::max(nx, ny) - 1));
        } else {
            ctx.n_past = st + std::max(nx, ny);
        }
        if (dump) {
            dump_embd.insert(dump_embd.end(), embd, embd + (size_t)n_tok * n_embd);
            for (int i = 0; i < n_tok; i++) {
                dump_tokens.push_back(-1);
                dump_pos.push_back(pt[i]); dump_pos.push_back(ph[i]); dump_pos.push_back(pw[i]);
            }
        }
        res.n_visual_tokens += n_tok;
        res.n_prompt_tokens += n_tok;
        group++;
    }
    res.n_groups = group;

    if (dump) {
        const std::string base = p.dump_dir + "/" + video + "." + p.mode;
        const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
        const float * logits = llama_get_logits_ith(ctx.lctx, -1);
        write_bin(base + ".logits.bin", logits, (size_t)n_vocab * sizeof(float));
        write_bin(base + ".embd.bin", dump_embd.data(), dump_embd.size() * sizeof(float));
        std::vector<unsigned char> all;
        for (const auto & f : frames) all.insert(all.end(), f.px.begin(), f.px.end());
        write_bin(base + ".frames.bin", all.data(), all.size());
        json j;
        j["video"] = video;
        j["mode"] = p.mode;
        j["frame_idx"] = frame_idx;
        j["frame_size"] = {frames[0].w, frames[0].h};
        j["grid"] = {res.nx, res.ny};
        j["n_groups"] = res.n_groups;
        j["n_embd"] = n_embd;
        j["n_vocab"] = n_vocab;
        j["video_fps"] = p.video_fps;
        j["tokens_per_second"] = p.tokens_per_second;
        j["tokens"] = dump_tokens;
        j["pos"] = dump_pos;
        j["next_pos"] = ctx.n_past;
        std::ofstream f(base + ".json");
        f << j.dump() << std::endl;
    }

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
// Section 5: Output
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
    j["n_groups"] = r.n_groups;
    j["grid"] = {r.nx, r.ny};
    j["n_visual_tokens"] = r.n_visual_tokens;
    j["n_prompt_tokens"] = r.n_prompt_tokens;
    j["load_ms"] = r.load_ms;
    j["tokenize_ms"] = r.tokenize_ms;
    j["vit_ms"] = r.vit_ms;
    j["prefill_ms"] = r.prefill_ms;
    j["decode_ms"] = r.decode_ms;
    j["total_ms"] = r.total_ms;
    j["group_encode_ms"] = r.group_encode_ms;
    j["n_tokens"] = r.n_tokens;
    j["text"] = sanitize_utf8(r.text);
    return j;
}

// ============================================================================
// Section 6: Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    tp_params p;
    auto filtered_argv = filter_custom_args(argc, argv, p);
    common_params params;
    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }
    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (p.mode != "images" && p.mode != "pairs" && p.mode != "video") {
        LOG_ERR("unknown --mode '%s'\n", p.mode.c_str());
        return 1;
    }
    if (p.output_json.empty()) {
        LOG_ERR("--output-json is required\n");
        return 1;
    }
    if (p.video_fps <= 0.0f) {
        LOG_ERR("--video-fps must be positive\n");
        return 1;
    }
    // The tokenizer reads this once, on its first media marker: two bitmaps per
    // marker when set. It must match the mode before anything is tokenized.
    if (p.mode == "images") unsetenv("MTMD_TEMPORAL_PAIR");
    else                    setenv("MTMD_TEMPORAL_PAIR", "1", 1);

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
    out["mode"] = p.mode;
    out["align"] = p.align;
    out["base_frames"] = p.base_frames;
    out["frame_step"] = p.frame_step;
    out["video_fps"] = p.video_fps;
    out["tokens_per_second"] = p.tokens_per_second;
    out["n_predict"] = n_predict;

    for (int i = 0; i < p.warmup; i++) {
        ctx.reset();
        run_request(ctx, p, videos[0], prompt, n_predict, false);
    }

    json requests = json::array();
    for (int round = 0; round < p.rounds; round++) {
        for (const auto & v : videos) {
            ctx.reset();
            request_result r = run_request(ctx, p, v, prompt, n_predict, !p.dump_dir.empty() && round == 0);
            if (!r.ok) {
                LOG_ERR("request failed: %s\n", v.c_str());
                continue;
            }
            LOG_INF("[%s r%d] %s: %d frames -> %d visual tokens, vit %.1f prefill %.1f decode %.1f total %.1f ms, %d tok\n",
                    p.mode.c_str(), round, v.c_str(), r.n_frames, r.n_visual_tokens,
                    r.vit_ms, r.prefill_ms, r.decode_ms, r.total_ms, r.n_tokens);
            requests.push_back(request_to_json(r, v, round));
        }
    }
    out["requests"] = requests;

    std::ofstream f(p.output_json);
    f << out.dump(2) << std::endl;
    LOG_INF("results written to %s\n", p.output_json.c_str());
    return 0;
}
