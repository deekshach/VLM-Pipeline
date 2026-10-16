// ViT encode / LLM prefill overlap probe: can frame k's prefill run while frame k+1 is encoded, on the one GPU?
//
// Each frame: the ViT encodes a window-cache crop of <shape> windows (9x4 = full frame), then the LLM prefills
// one full frame's image tokens (576 at 1008x448, embeddings from one full-frame encode). Per round:
//   serial      all encodes, then all prefills (what the pipeline does)
//   pipelined   two threads; prefill of frame k waits for encode k (the real dependency)
// and reports the wall time, each stage's time and the speedup. ViT flash attention off.
//
//   jetson-overlap-probe <model.gguf> <mmproj.gguf> <frame.png> <shape e.g. 5x2> [frames=8] [rounds=3]
//
// Engine switches apply as set in the environment (MTMD_VIT_WINDOW_BATCH, GGML_CUDA_JETSON_GEMM_*).
// OVERLAP_VIT_PRIO / OVERLAP_LLM_PRIO=<n>: CUDA stream priority of each model's streams (lower = higher
// priority; on Orin -5..0), through ggml_backend_cuda_set_thread_stream_priority. Default: both 0.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const int WINDOW_PX = 112;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// see vit_shape_probe.cpp: a no-op eval callback is what turns the ViT's flash attention off in this fork
static bool no_observe(struct ggml_tensor *, bool ask, void *) { return !ask; }

static mtmd_input_chunks * tokenize(mtmd_context * vctx, const std::vector<unsigned char> & px, int w, int h,
                                    const mtmd_input_chunk ** img) {
    mtmd_bitmap * b = mtmd_bitmap_init(w, h, px.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text text = { mtmd_default_marker(), false, true };
    const mtmd_bitmap * bp = b;
    if (mtmd_tokenize(vctx, chunks, &text, &bp, 1) != 0) exit(1);
    mtmd_bitmap_free(b);
    *img = nullptr;
    for (size_t k = 0; k < mtmd_input_chunks_size(chunks); k++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, k);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) *img = c;
    }
    if (!*img) exit(1);
    return chunks;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model mmproj frame.png shape [frames] [rounds]\n", argv[0]); return 1; }
    int ww = 0, wh = 0;
    if (sscanf(argv[4], "%dx%d", &ww, &wh) != 2) return 1;
    const int N      = argc > 5 ? atoi(argv[5]) : 8;
    const int ROUNDS = argc > 6 ? atoi(argv[6]) : 3;
    const int vit_prio = getenv("OVERLAP_VIT_PRIO") ? atoi(getenv("OVERLAP_VIT_PRIO")) : 0;
    const int llm_prio = getenv("OVERLAP_LLM_PRIO") ? atoi(getenv("OVERLAP_LLM_PRIO")) : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    // streams are created on first use, on the thread that first uses them: set each model's priority
    // right before its first computation
    ggml_backend_cuda_set_thread_stream_priority(llm_prio);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N * 600 + 256; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) return 1;

    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu = true; vp.print_timings = false; vp.warmup = false;
    vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    vp.cb_eval = no_observe;
    mtmd_context * vctx = mtmd_init_from_file(argv[2], model, vp);
    if (!vctx) return 1;

    // the frame, resized to the 112 px window grid (as the pipeline's --align 112), and the crop
    mtmd_bitmap * raw = mtmd_helper_bitmap_init_from_file(vctx, argv[3]);
    if (!raw) return 1;
    const int fw = 9 * WINDOW_PX, fh = 4 * WINDOW_PX;   // 1008 x 448
    std::vector<unsigned char> frame((size_t)fw * fh * 3);
    {
        const int sw = mtmd_bitmap_get_nx(raw), sh = mtmd_bitmap_get_ny(raw);
        const unsigned char * s = mtmd_bitmap_get_data(raw);
        for (int y = 0; y < fh; y++) for (int x = 0; x < fw; x++) for (int c = 0; c < 3; c++)   // nearest
            frame[((size_t)y * fw + x) * 3 + c] = s[((size_t)(y * sh / fh) * sw + x * sw / fw) * 3 + c];
        mtmd_bitmap_free(raw);
    }
    std::vector<unsigned char> cpx((size_t)ww * WINDOW_PX * wh * WINDOW_PX * 3);
    for (int r = 0; r < wh * WINDOW_PX; r++)
        memcpy(&cpx[(size_t)r * ww * WINDOW_PX * 3], &frame[(size_t)r * fw * 3], (size_t)ww * WINDOW_PX * 3);
    const mtmd_input_chunk * full = nullptr, * crop = nullptr;
    mtmd_input_chunks * cf = tokenize(vctx, frame, fw, fh, &full);
    mtmd_input_chunks * cc = tokenize(vctx, cpx, ww * WINDOW_PX, wh * WINDOW_PX, &crop);
    const size_t n_tok = mtmd_input_chunk_get_n_tokens(full);
    const int n_embd = llama_model_n_embd_inp(model);

    ggml_backend_cuda_set_thread_stream_priority(vit_prio);
    if (mtmd_encode_chunk(vctx, full) != 0) return 1;   // embeddings for the prefills; creates the ViT streams
    std::vector<float> embd(mtmd_get_output_embd(vctx), mtmd_get_output_embd(vctx) + n_tok * n_embd);
    ggml_backend_cuda_set_thread_stream_priority(0);
    printf("crop %s (%zu ViT tokens), prefill %zu tokens per frame, %d frames, stream priority vit %d llm %d\n",
           argv[4], mtmd_input_chunk_get_n_tokens(crop), n_tok, N, vit_prio, llm_prio);

    auto encode = [&](int) { if (mtmd_encode_chunk(vctx, crop) != 0) exit(1); };
    llama_pos n_past = 0;
    auto prefill = [&](int) {
        llama_pos np = n_past;
        if (mtmd_helper_decode_image_chunk(vctx, lctx, full, embd.data(), n_past, 0, 2048, &np) != 0) exit(1);
        llama_synchronize(lctx);
        n_past = np;
    };
    auto reset = [&] { llama_memory_clear(llama_get_memory(lctx), true); n_past = 0; };

    for (int i = 0; i < 2; i++) { encode(0); }   // warm-up
    reset(); for (int i = 0; i < 2; i++) prefill(i);

    for (int r = 0; r < ROUNDS; r++) {
        // serial: all encodes, then all prefills
        reset();
        double t0 = now_ms();
        for (int i = 0; i < N; i++) encode(i);
        const double t_vit = now_ms() - t0;
        t0 = now_ms();
        for (int i = 0; i < N; i++) prefill(i);
        const double t_pre = now_ms() - t0;

        // pipelined: prefill k after encode k, on two threads
        reset();
        std::mutex m; std::condition_variable cv; int encoded = 0;
        double v_busy = 0, p_busy = 0, p_wait = 0;
        t0 = now_ms();
        std::thread tv([&] {
            for (int i = 0; i < N; i++) {
                const double a = now_ms(); encode(i); v_busy += now_ms() - a;
                { std::lock_guard<std::mutex> l(m); encoded = i + 1; }
                cv.notify_one();
            }
        });
        std::thread tp([&] {
            for (int i = 0; i < N; i++) {
                const double a = now_ms();
                { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return encoded > i; }); }
                const double b = now_ms(); p_wait += b - a;
                prefill(i); p_busy += now_ms() - b;
            }
        });
        tv.join(); tp.join();
        const double t_pipe = now_ms() - t0;
        // the ideal: the longer stage plus one frame of the other (the pipeline fill)
        const double ideal = std::max(t_vit + t_pre / N, t_pre + t_vit / N);
        printf("round %d | serial %.0f ms (vit %.0f = %.1f/frame, prefill %.0f = %.1f/frame)\n", r, t_vit + t_pre,
               t_vit, t_vit / N, t_pre, t_pre / N);
        printf("        | pipelined %.0f ms, speedup %.3fx (ideal %.0f ms, %.3fx) | concurrent: vit %.1f/frame (%.2fx solo), "
               "prefill %.1f/frame (%.2fx solo), prefill waited %.0f ms\n", t_pipe, (t_vit + t_pre) / t_pipe, ideal,
               (t_vit + t_pre) / ideal, v_busy / N, v_busy / t_vit, p_busy / N, p_busy / t_pre, p_wait);
        fflush(stdout);
    }
    mtmd_input_chunks_free(cf); mtmd_input_chunks_free(cc);
    mtmd_free(vctx);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
