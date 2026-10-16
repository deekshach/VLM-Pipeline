// Dense ViT reference for the accuracy study: encode every frame of a clip as one full image (no window cache)
// and write the merged visual tokens the LLM would receive.
//
//   jetson-embed-frames <model.gguf> <mmproj.gguf> <frames dir> <out dir> [native|aligned] [step=1]
//
// native:  the frame as the baseline pipeline passes it (mtmd's own resize; 1024x436 -> 592 tokens on Sintel)
// aligned: resized to the 112 px window grid first, as --window-align does (1024x436 -> 1008x448, 576 tokens)
// Writes <out>/frame_NNNN.f32 (nx*ny x n_embd, F32, raster order) and <out>/meta.json. ViT flash attention off
// (the study setting); engine switches apply as set in the environment.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static const int WINDOW_PX = 112;

struct rgb { int w = 0, h = 0; std::vector<unsigned char> px; };

static rgb resize_bilinear(const rgb & s, int W, int H) {   // same as jetson_vlm.cpp
    if (s.w == W && s.h == H) return s;
    rgb d; d.w = W; d.h = H; d.px.resize((size_t) W * H * 3);
    const float sx = (float) s.w / W, sy = (float) s.h / H;
    for (int y = 0; y < H; y++) {
        float fy = std::max(0.0f, (y + 0.5f) * sy - 0.5f);
        int y0 = std::min((int) fy, s.h - 1), y1 = std::min(y0 + 1, s.h - 1);
        float wy = fy - y0;
        for (int x = 0; x < W; x++) {
            float fx = std::max(0.0f, (x + 0.5f) * sx - 0.5f);
            int x0 = std::min((int) fx, s.w - 1), x1 = std::min(x0 + 1, s.w - 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float a = s.px[((size_t) y0 * s.w + x0) * 3 + c], b = s.px[((size_t) y0 * s.w + x1) * 3 + c];
                float e = s.px[((size_t) y1 * s.w + x0) * 3 + c], f = s.px[((size_t) y1 * s.w + x1) * 3 + c];
                float v = (a * (1 - wx) + b * wx) * (1 - wy) + (e * (1 - wx) + f * wx) * wy;
                d.px[((size_t) y * W + x) * 3 + c] = (unsigned char) std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
    return d;
}

static int round_to(int v, int f) { return std::max(f, (int) std::lround((double) v / f) * f); }

// see vit_shape_probe.cpp: a no-op eval callback is what turns the ViT's flash attention off in this fork
static bool no_observe(struct ggml_tensor *, bool ask, void *) { return !ask; }

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model mmproj frames_dir out_dir [native|aligned] [step=1]\n", argv[0]);
        return 1;
    }
    const std::string in_dir = argv[3], out_dir = argv[4];
    const bool aligned = argc > 5 && strcmp(argv[5], "aligned") == 0;
    const int step = argc > 6 ? std::max(1, atoi(argv[6])) : 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // only the vision tower runs; the LLM is needed for mtmd's n_embd check
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = true; vp.print_timings = false; vp.warmup = false;
    vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vp.cb_eval = no_observe;
    mtmd_context * vctx = mtmd_init_from_file(argv[2], model, vp);
    if (!vctx) return 1;
    const int n_embd = llama_model_n_embd_inp(model);

    std::vector<std::string> frames;
    for (const auto & e : std::filesystem::directory_iterator(in_dir)) {
        if (e.path().extension() == ".png" || e.path().extension() == ".jpg") frames.push_back(e.path().filename().string());
    }
    std::sort(frames.begin(), frames.end());
    std::filesystem::create_directories(out_dir);
    FILE * meta = fopen((out_dir + "/meta.json").c_str(), "w");
    fprintf(meta, "{\"mode\":\"%s\",\"n_embd\":%d,\"frames\":[", aligned ? "aligned" : "native", n_embd);

    int written = 0;
    for (size_t i = 0; i < frames.size(); i += step) {
        mtmd_bitmap * b = mtmd_helper_bitmap_init_from_file(vctx, (in_dir + "/" + frames[i]).c_str());
        if (!b) return 1;
        if (aligned) {
            rgb s; s.w = mtmd_bitmap_get_nx(b); s.h = mtmd_bitmap_get_ny(b);
            s.px.assign(mtmd_bitmap_get_data(b), mtmd_bitmap_get_data(b) + mtmd_bitmap_get_n_bytes(b));
            mtmd_bitmap_free(b);
            const rgb a = resize_bilinear(s, round_to(s.w, WINDOW_PX), round_to(s.h, WINDOW_PX));
            b = mtmd_bitmap_init(a.w, a.h, a.px.data());
        }
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text = { mtmd_default_marker(), false, true };
        const mtmd_bitmap * bp = b;
        if (mtmd_tokenize(vctx, chunks, &text, &bp, 1) != 0) return 1;
        const mtmd_input_chunk * img = nullptr;
        for (size_t k = 0; k < mtmd_input_chunks_size(chunks); k++) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, k);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c;
        }
        if (!img || mtmd_encode_chunk(vctx, img) != 0) return 1;
        const auto * it = mtmd_input_chunk_get_tokens_image(img);
        const int nx = (int) mtmd_image_tokens_get_nx(it), ny = (int) mtmd_image_tokens_get_ny(it);
        const std::string stem = frames[i].substr(0, frames[i].find_last_of('.'));
        if (FILE * f = fopen((out_dir + "/" + stem + ".f32").c_str(), "wb")) {
            fwrite(mtmd_get_output_embd(vctx), sizeof(float), (size_t) nx * ny * n_embd, f);
            fclose(f);
        }
        fprintf(meta, "%s{\"frame\":\"%s\",\"nx\":%d,\"ny\":%d}", written++ ? "," : "", stem.c_str(), nx, ny);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(b);
    }
    fprintf(meta, "]}\n");
    fclose(meta);
    printf("%d frames -> %s (%s)\n", written, out_dir.c_str(), aligned ? "aligned" : "native");
    mtmd_free(vctx);
    llama_model_free(model);
    return 0;
}
