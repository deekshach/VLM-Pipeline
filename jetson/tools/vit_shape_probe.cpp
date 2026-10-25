// ViT microarchitecture probe: full-frame encode vs window-aligned crops.
//
// Keeps only the vision tower on the GPU (the LLM is loaded with 0 GPU layers,
// memory-mapped, only for mtmd's n_embd check), so ncu's kernel replay has little
// device memory to save and restore. Resizes one frame to the 112 px window grid as window_cache_vlm.cpp
// does with --align 112, and encodes crops of whole windows. Each shape is
// encoded `warm` times untimed, then `reps` times timed; the last timed encode
// of each shape is wrapped in an NVTX range "vit_<W>x<H>" (W x H windows) so
// nsys and ncu can select it.
//
//   jetson-vit-probe <model.gguf> <mmproj.gguf> <frame.png> <shapes e.g. 9x4,5x2> [warm=2] [reps=5] [fa=on|off]
//   jetson-vit-probe <model.gguf> <mmproj.gguf> <frame.png> w:<k list e.g. 1,4,9,18,36> [warm] [reps] [fa]
//     per-window mode: the whole frame is the key frame, then k of its windows re-encoded (the window cache's
//     per-window path, which needs MTMD_VIT_WINDOW_BATCH=1); 16 tokens per window
//
// Environment: PROBE_DUMP_PREFIX=<p> writes each shape's embeddings to <p>_<shape>.f32 (for comparing
// switches); PROBE_CPU=1 runs the ViT on the CPU (F32-output reference); PROBE_MARK_ALL=1 puts every timed
// encode (not only the last) in the NVTX range, for averaging in nsys. The engine switches
// (MTMD_VIT_WINDOW_BATCH, GGML_CUDA_JETSON_GEMM_*) apply as set in the environment.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static const int WINDOW_PX = 112;
static const int TOKEN_PX  = 28;

struct rgb { int w = 0, h = 0; std::vector<unsigned char> px; };

static rgb resize_bilinear(const rgb & s, int W, int H) {   // same as window_cache_vlm.cpp
    rgb d; d.w = W; d.h = H; d.px.resize((size_t)W * H * 3);
    const float sx = (float)s.w / W, sy = (float)s.h / H;
    for (int y = 0; y < H; y++) {
        float fy = std::max(0.0f, (y + 0.5f) * sy - 0.5f);
        int y0 = std::min((int)fy, s.h - 1), y1 = std::min(y0 + 1, s.h - 1);
        float wy = fy - y0;
        for (int x = 0; x < W; x++) {
            float fx = std::max(0.0f, (x + 0.5f) * sx - 0.5f);
            int x0 = std::min((int)fx, s.w - 1), x1 = std::min(x0 + 1, s.w - 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float a = s.px[((size_t)y0 * s.w + x0) * 3 + c], b = s.px[((size_t)y0 * s.w + x1) * 3 + c];
                float e = s.px[((size_t)y1 * s.w + x0) * 3 + c], f = s.px[((size_t)y1 * s.w + x1) * 3 + c];
                float v = (a * (1 - wx) + b * wx) * (1 - wy) + (e * (1 - wx) + f * wx) * wy;
                d.px[((size_t)y * W + x) * 3 + c] = (unsigned char)std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
    return d;
}

static rgb crop(const rgb & s, int w, int h) {
    rgb d; d.w = w; d.h = h; d.px.resize((size_t)w * h * 3);
    for (int r = 0; r < h; r++) memcpy(&d.px[(size_t)r * w * 3], &s.px[(size_t)r * s.w * 3], (size_t)w * 3);
    return d;
}

// No-op eval callback. This fork's mtmd ignores mtmd_context_params.flash_attn_type
// for the vision encoder (it always passes AUTO, which enables flash attention) and
// only disables flash attention when cb_eval is set. Asking for no tensors keeps the
// scheduler computing each split in one call, as without a callback.
static bool no_observe(struct ggml_tensor *, bool ask, void *) { return !ask; }

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model mmproj frame.png shapes [warm] [reps]\n", argv[0]); return 1; }
    const int warm = argc > 5 ? atoi(argv[5]) : 2;
    const int reps = argc > 6 ? atoi(argv[6]) : 5;
    const bool fa  = !(argc > 7 && strcmp(argv[7], "off") == 0);
    const bool mark_all = getenv("PROBE_MARK_ALL") != nullptr;   // nsys: every timed encode in the range

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = getenv("PROBE_CPU") == nullptr;   // PROBE_CPU=1: CPU reference (F32 accumulation and output)
    vp.print_timings = false; vp.warmup = false;
    vp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (!fa) vp.cb_eval = no_observe;
    mtmd_context * vctx = mtmd_init_from_file(argv[2], model, vp);
    if (!vctx) return 1;

    mtmd_bitmap * b = mtmd_helper_bitmap_init_from_file(vctx, argv[3]);
    if (!b) return 1;
    rgb raw; raw.w = mtmd_bitmap_get_nx(b); raw.h = mtmd_bitmap_get_ny(b);
    raw.px.assign(mtmd_bitmap_get_data(b), mtmd_bitmap_get_data(b) + mtmd_bitmap_get_n_bytes(b));
    mtmd_bitmap_free(b);
    auto round_to = [](int v) { return std::max(WINDOW_PX, (int)std::lround((double)v / WINDOW_PX) * WINDOW_PX); };
    const rgb frame = resize_bilinear(raw, round_to(raw.w), round_to(raw.h));
    printf("frame %dx%d px -> %dx%d windows, flash attention %s\n", frame.w, frame.h, frame.w / WINDOW_PX, frame.h / WINDOW_PX, fa ? "on" : "off");

    if (strncmp(argv[4], "w:", 2) == 0) {
        // Per-window re-encode (the window cache's --wc-per-window path): encode the whole frame once through
        // mtmd_encode_chunk_windows, which fills the encoder's K/V cache as a key frame does, then re-encode
        // exactly k windows, spread evenly over the frame, for each k in the list. NVTX range "vitw_<k>".
        mtmd_bitmap * fb = mtmd_bitmap_init(frame.w, frame.h, frame.px.data());
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text = { mtmd_default_marker(), false, true };
        const mtmd_bitmap * fbp = fb;
        if (mtmd_tokenize(vctx, chunks, &text, &fbp, 1) != 0) return 1;
        const mtmd_input_chunk * img = nullptr;
        for (size_t k = 0; k < mtmd_input_chunks_size(chunks); k++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, k);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = ch;
        }
        if (!img) return 1;
        const int nw = (frame.w / WINDOW_PX) * (frame.h / WINDOW_PX);
        std::vector<int32_t> all(nw);
        for (int w = 0; w < nw; w++) all[w] = w;
        // a plain encode first: it allocates the encoder's buffers (its warm-up graph knows no per-window state)
        if (mtmd_encode_chunk(vctx, img) != 0) return 1;
        for (int i = 0; i < warm; i++) if (mtmd_encode_chunk_windows(vctx, img, all.data(), nw) != 0) return 1;
        {   // the whole frame through the per-window path, timed: the key-frame cost the k-window encodes compare to
            std::vector<double> t;
            for (int i = 0; i < reps; i++) {
                const double t0 = now_ms();
                if (mtmd_encode_chunk_windows(vctx, img, all.data(), nw) != 0) return 1;
                t.push_back(now_ms() - t0);
            }
            std::sort(t.begin(), t.end());
            printf("key frame (all %d windows): median %.2f ms, min %.2f ms\n", nw, t[t.size() / 2], t[0]);
        }
        printf("windows,of,tokens,median_ms,min_ms\n");
        std::stringstream ws(argv[4] + 2);
        std::string item;
        while (std::getline(ws, item, ',')) {
            const int k = std::min(nw, std::max(1, atoi(item.c_str())));
            std::vector<int32_t> wins;
            for (int i = 0; i < k; i++) wins.push_back((int32_t)((long)i * nw / k));
            if (mtmd_encode_chunk_windows(vctx, img, all.data(), nw) != 0) return 1;   // a fresh key frame
            for (int i = 0; i < warm; i++) if (mtmd_encode_chunk_windows(vctx, img, wins.data(), k) != 0) return 1;
            std::vector<double> t;
            const std::string range = "vitw_" + std::to_string(k);
            for (int i = 0; i < reps; i++) {
                const bool mark = i == reps - 1 || mark_all;
                if (mark) nvtxRangePushA(range.c_str());
                const double t0 = now_ms();
                if (mtmd_encode_chunk_windows(vctx, img, wins.data(), k) != 0) return 1;
                t.push_back(now_ms() - t0);
                if (mark) nvtxRangePop();
            }
            std::vector<double> s = t; std::sort(s.begin(), s.end());
            printf("%d,%d,%d,%.2f,%.2f\n", k, nw, k * 16, s[s.size() / 2], s[0]);
            fflush(stdout);
        }
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(fb);
        mtmd_free(vctx);
        llama_model_free(model);
        return 0;
    }

    printf("shape,windows,tokens,patches,median_ms,min_ms\n");
    std::stringstream ss(argv[4]);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int ww = 0, wh = 0;
        if (sscanf(item.c_str(), "%dx%d", &ww, &wh) != 2) { fprintf(stderr, "bad shape %s\n", item.c_str()); return 1; }
        const rgb c = crop(frame, ww * WINDOW_PX, wh * WINDOW_PX);
        mtmd_bitmap * cb = mtmd_bitmap_init(c.w, c.h, c.px.data());
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text = { mtmd_default_marker(), false, true };
        const mtmd_bitmap * cbp = cb;
        if (mtmd_tokenize(vctx, chunks, &text, &cbp, 1) != 0) return 1;
        const mtmd_input_chunk * img = nullptr;
        for (size_t k = 0; k < mtmd_input_chunks_size(chunks); k++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, k);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = ch;
        }
        if (!img) return 1;
        const auto * it = mtmd_input_chunk_get_tokens_image(img);
        const int ntok = (int)(mtmd_image_tokens_get_nx(it) * mtmd_image_tokens_get_ny(it));
        if ((int)mtmd_image_tokens_get_nx(it) != ww * WINDOW_PX / TOKEN_PX) {
            fprintf(stderr, "shape %s tokenized off-grid\n", item.c_str()); return 1;
        }
        for (int i = 0; i < warm; i++) mtmd_encode_chunk(vctx, img);
        std::vector<double> t;
        const std::string range = "vit_" + item;
        for (int i = 0; i < reps; i++) {
            const bool mark = i == reps - 1 || mark_all;
            if (mark) nvtxRangePushA(range.c_str());
            const double t0 = now_ms();
            if (mtmd_encode_chunk(vctx, img) != 0) return 1;
            t.push_back(now_ms() - t0);
            if (mark) nvtxRangePop();
        }
        if (const char * dump = getenv("PROBE_DUMP_PREFIX")) {   // embeddings of the last encode, for comparisons
            const std::string path = std::string(dump) + "_" + item + ".f32";
            FILE * df = fopen(path.c_str(), "wb");
            if (df) {
                fwrite(mtmd_get_output_embd(vctx), sizeof(float), (size_t)ntok * llama_model_n_embd_inp(model), df);
                fclose(df);
            }
        }
        std::vector<double> s = t; std::sort(s.begin(), s.end());
        printf("%s,%d,%d,%d,%.2f,%.2f\n", item.c_str(), ww * wh, ntok, ntok * 4, s[s.size() / 2], s[0]);
        fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(cb);
    }
    mtmd_free(vctx);
    llama_model_free(model);
    return 0;
}
