// Quick check: can ViT encode and LLM prefill overlap on one GPU?
// Runs 8 ViT encodes and 8 image-chunk prefills (a) one after the other, (b) on two threads at once.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model mmproj image [frames] [rounds]\n", argv[0]); return 1; }
    const int N      = argc > 4 ? atoi(argv[4]) : 8;
    const int ROUNDS = argc > 5 ? atoi(argv[5]) : 3;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16384; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) return 1;

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = true; vp.print_timings = false;
    vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    mtmd_context * vctx = mtmd_init_from_file(argv[2], model, vp);
    if (!vctx) return 1;

    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(vctx, argv[3]);
    if (!bmp) return 1;
    std::string prompt = mtmd_default_marker();
    mtmd_input_text text; text.text = prompt.c_str(); text.add_special = false; text.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bmps[1] = { bmp };
    if (mtmd_tokenize(vctx, chunks, &text, bmps, 1) != 0) return 1;
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c;
    }
    if (!img) return 1;
    const size_t n_tok = mtmd_input_chunk_get_n_tokens(img);
    const int n_embd = llama_model_n_embd_inp(model);
    printf("image chunk: %zu tokens, n_embd %d, frames/round %d\n", n_tok, n_embd, N);

    // one encode to get embeddings for the prefill thread (also serves as warmup)
    if (mtmd_encode_chunk(vctx, img) != 0) return 1;
    std::vector<float> embd(mtmd_get_output_embd(vctx), mtmd_get_output_embd(vctx) + n_tok * n_embd);

    auto run_vit = [&](std::vector<double> & t) {
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            mtmd_encode_chunk(vctx, img);
            t.push_back(now_ms() - t0);
        }
    };
    auto run_prefill = [&](std::vector<double> & t) {
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_pos n_past = 0;
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            llama_pos np = n_past;
            mtmd_helper_decode_image_chunk(vctx, lctx, img, embd.data(), n_past, 0, 2048, &np);
            llama_synchronize(lctx);
            n_past = np;
            t.push_back(now_ms() - t0);
        }
    };
    auto sum = [](const std::vector<double> & v) { double s = 0; for (double x : v) s += x; return s; };

    { std::vector<double> a, b; run_vit(a); run_prefill(b); } // warmup round

    for (int r = 0; r < ROUNDS; r++) {
        std::vector<double> v, p, vc, pc;
        double t0 = now_ms(); run_vit(v);     double tv = now_ms() - t0;
        t0 = now_ms();        run_prefill(p); double tp = now_ms() - t0;
        t0 = now_ms();
        std::thread a([&] { run_vit(vc); });
        std::thread b([&] { run_prefill(pc); });
        a.join(); b.join();
        double tc = now_ms() - t0;
        printf("round %d | solo: vit %.0f ms (%.0f/frame)  prefill %.0f ms (%.0f/frame)  serial sum %.0f ms\n",
               r, tv, tv / N, tp, tp / N, tv + tp);
        printf("        | concurrent: wall %.0f ms  (vit thread %.0f ms, %.0f/frame; prefill thread %.0f ms, %.0f/frame)\n",
               tc, sum(vc), sum(vc) / N, sum(pc), sum(pc) / N);
        printf("        | speedup vs serial %.3fx   saved %.0f ms   ideal(max) %.0f ms\n",
               (tv + tp) / tc, tv + tp - tc, tv > tp ? tv : tp);
        fflush(stdout);
    }
    return 0;
}
