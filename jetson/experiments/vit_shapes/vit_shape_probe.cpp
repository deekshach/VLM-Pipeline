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
//   vit_shape_probe <model.gguf> <mmproj.gguf> <frame.png> <shapes e.g. 9x4,5x2> [warm=2] [reps=5] [fa=on|off]
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
            const bool mark = i == reps - 1;
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
