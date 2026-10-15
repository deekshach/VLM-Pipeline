// llama-jetson-vlm -- the Jetson AGX Orin video VLM pipeline (Qwen2.5-VL via libmtmd), with every
// optimization behind its own switch so each one's effect can be measured on its own:
//
//   frame selection   --baseline (uniform) | adaptive (default): motion-guided sampling from optical flow
//   flow source       Sintel .flo files (default) | --live-flow: computed on the OFA (VPI), pipelined
//                     (--live-flow-serial: one stage at a time, for the breakdown)
//   anchor            --anchor-first: frame 1 always selected and encoded while the flow is analysed
//   window grid       --window-align: frames resized to a multiple of the 112 px ViT attention window
//   window cache      --window-cache: re-encode only the windows that changed since they were last
//                     encoded (dirty-window crop), reuse the rest of the embeddings (implies --window-align)
//   ViT attention     --vit-flash-attn on|off; --vit-window-batch: windowed layers as a batch of windows
//                     (engine switch MTMD_VIT_WINDOW_BATCH; needs aligned frames)
//   BF16 GEMMs        --gemm-f32-out, --gemm-bias-fuse, --gemm-table <file>, --gemm-record <file>
//                     (engine switches GGML_CUDA_JETSON_GEMM_*, see ggml-cuda/jetson-gemm.cuh)
//
// The JSON output records every switch and a TTFT breakdown; NVTX ranges (request, flow, select,
// anchor, vit, prefill_img, prefill_txt, wc_diff, decode) let nsys/ncu attribute GPU work to stages.

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef JETSON_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void) 0)
#define NVTX_POP()      ((void) 0)
#endif
#ifdef HAVE_VPI
#include "ofa_live_flow.h"
#else
struct rgb_frame {
    int w = 0, h = 0;
    std::vector<unsigned char> px; // RGB8, tightly packed
};
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <chrono>
#include <cinttypes>
#include <future>
#include <memory>
#include <cstdlib>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Custom parameters for Sintel dataset processing
// ============================================================================

struct sintel_params {
    std::string sintel_dir    = "/home/nvidia/data/MPI-Sintel";
    std::string video_name    = "alley_1";
    int         min_frames    = 3;
    int         max_frames    = 24;
    int         base_frames   = 16;
    float       motion_threshold  = 2.0f;
    float       motion_percentile = 0.95f;
    float       selection_percentile = 70.0f; // percentile for importance weight thresholding
    int         patch_grid    = 4;
    float       spatial_threshold = 0.05f;  // patch motion ratio threshold (Python: patch_motion_ratio)
    std::string output_json;
    bool        baseline      = false; // if true, use uniform frame sampling (no motion analysis)
    bool        anchor_first  = false; // adaptive only: always keep frame 1 and encode it while the CPU analyses motion
    bool        live_flow     = false; // compute flow from the frames on the OFA instead of reading training/flow
    bool        live_flow_serial = false; // live flow one stage at a time (for the breakdown), not pipelined
    int         decode_threads = 6;    // live flow: PNG decode threads
    int         ofa_grid      = 2;     // live flow: one motion vector per grid x grid pixels
    int         ofa_levels    = 4;     // live flow: pyramid levels (more = longer search range)
    bool        ofa_prep_cpu  = true;  // live flow: grayscale + pyramid on the CPU (false: CUDA), keeping the GPU free
    bool        vit_fa_off    = false; // run the ViT without flash attention
    // window grid, window cache, engine switches
    bool        window_align  = false; // resize frames to a multiple of the 112 px attention window
    bool        window_cache  = false; // dirty-window crop re-encoding with embedding reuse
    float       wc_threshold  = 6.0f;  // window is dirty if a token cell's mean |dRGB| (0..255) exceeds this
    int         wc_halo       = 0;     // grow the dirty rectangle by this many windows
    float       wc_rekey_frac = 0.75f; // without the cost model: crops above this fraction of the frame become full encodes
    bool        wc_cost_model = false; // choose the crop shape from measured encode times (calibrated at setup)
    int         wc_calib_reps = 3;
    bool        vit_window_batch = false;
    bool        gemm_f32_out  = false;
    bool        gemm_bias_fuse = false;
    std::string gemm_table;
    std::string gemm_record;
};

// Strip custom arguments from argv before passing to common_params_parse.
// Returns the filtered argv (without sintel-specific flags).
static std::vector<char *> filter_custom_args(int argc, char ** argv, sintel_params & sparams) {
    std::vector<char *> filtered;
    for (int i = 0; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--sintel-dir" && i + 1 < argc) {
            sparams.sintel_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            sparams.video_name = argv[++i];
        } else if (arg == "--min-frames" && i + 1 < argc) {
            sparams.min_frames = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            sparams.max_frames = std::stoi(argv[++i]);
        } else if (arg == "--base-frames" && i + 1 < argc) {
            sparams.base_frames = std::stoi(argv[++i]);
        } else if (arg == "--motion-threshold" && i + 1 < argc) {
            sparams.motion_threshold = std::stof(argv[++i]);
        } else if (arg == "--motion-percentile" && i + 1 < argc) {
            sparams.motion_percentile = std::stof(argv[++i]);
        } else if (arg == "--patch-grid" && i + 1 < argc) {
            sparams.patch_grid = std::stoi(argv[++i]);
        } else if (arg == "--spatial-threshold" && i + 1 < argc) {
            sparams.spatial_threshold = std::stof(argv[++i]);
        } else if (arg == "--selection-percentile" && i + 1 < argc) {
            sparams.selection_percentile = std::stof(argv[++i]);
        } else if (arg == "--output-json" && i + 1 < argc) {
            sparams.output_json = argv[++i];
        } else if (arg == "--baseline") {
            sparams.baseline = true;
        } else if (arg == "--anchor-first") {
            sparams.anchor_first = true;
        } else if (arg == "--live-flow") {
            sparams.live_flow = true;
        } else if (arg == "--live-flow-serial") {
            sparams.live_flow = true;
            sparams.live_flow_serial = true;
        } else if (arg == "--decode-threads" && i + 1 < argc) {
            sparams.decode_threads = std::stoi(argv[++i]);
        } else if (arg == "--ofa-grid" && i + 1 < argc) {
            sparams.ofa_grid = std::stoi(argv[++i]);
        } else if (arg == "--ofa-levels" && i + 1 < argc) {
            sparams.ofa_levels = std::stoi(argv[++i]);
        } else if (arg == "--ofa-prep" && i + 1 < argc) {
            sparams.ofa_prep_cpu = std::string(argv[++i]) != "cuda";
        } else if (arg == "--vit-flash-attn" && i + 1 < argc) {
            sparams.vit_fa_off = std::string(argv[++i]) == "off";
        } else if (arg == "--window-align") {
            sparams.window_align = true;
        } else if (arg == "--window-cache") {
            sparams.window_cache = true;
            sparams.window_align = true;
        } else if (arg == "--wc-threshold" && i + 1 < argc) {
            sparams.wc_threshold = std::stof(argv[++i]);
        } else if (arg == "--wc-halo" && i + 1 < argc) {
            sparams.wc_halo = std::stoi(argv[++i]);
        } else if (arg == "--wc-rekey-frac" && i + 1 < argc) {
            sparams.wc_rekey_frac = std::stof(argv[++i]);
        } else if (arg == "--wc-cost-model") {
            sparams.wc_cost_model = true;
        } else if (arg == "--wc-calib-reps" && i + 1 < argc) {
            sparams.wc_calib_reps = std::stoi(argv[++i]);
        } else if (arg == "--vit-window-batch") {
            sparams.vit_window_batch = true;
        } else if (arg == "--gemm-f32-out") {
            sparams.gemm_f32_out = true;
        } else if (arg == "--gemm-bias-fuse") {
            sparams.gemm_bias_fuse = true;
        } else if (arg == "--gemm-table" && i + 1 < argc) {
            sparams.gemm_table = argv[++i];
        } else if (arg == "--gemm-record" && i + 1 < argc) {
            sparams.gemm_record = argv[++i];
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nJetson video VLM pipeline (MPI-Sintel), every optimization behind a switch\n\n");
    printf("Usage: %s [options]\n\n", argv[0]);
    printf("Required llama.cpp options:\n");
    printf("  -m <model>          Path to Qwen2.5-VL GGUF model\n");
    printf("  --mmproj <mmproj>   Path to Qwen2.5-VL mmproj GGUF\n\n");
    printf("Sintel-specific options:\n");
    printf("  --sintel-dir <dir>  MPI-Sintel dataset directory\n");
    printf("  --video <name>      Video name (e.g., alley_1)\n");
    printf("  --min-frames <n>    Minimum frames to select (default: 3)\n");
    printf("  --max-frames <n>    Maximum frames to select (default: 15)\n");
    printf("  --base-frames <n>   Base frame count (default: 8)\n");
    printf("  --motion-threshold  Motion magnitude threshold (default: 2.0)\n");
    printf("  --motion-percentile Percentile for motion stats (default: 0.95)\n");
    printf("  --selection-percentile Percentile for weight thresholding (default: 70)\n");
    printf("  --patch-grid <n>    Spatial patch grid size NxN (default: 4)\n");
    printf("  --spatial-threshold Patch motion ratio threshold (default: 0.05)\n");
    printf("  --output-json <f>   Output JSON file path\n");
    printf("  --baseline          Use uniform frame sampling (no motion analysis)\n");
    printf("  --anchor-first      Adaptive only: always select frame 1 and encode it on the GPU\n");
    printf("                      while the CPU runs the motion analysis\n");
    printf("  --live-flow         Adaptive only: compute optical flow from the frames on the OFA\n");
    printf("                      (Jetson, NVIDIA VPI) instead of reading training/flow\n");
    printf("  --live-flow-serial  As --live-flow, one stage at a time (stage breakdown, no overlap)\n");
    printf("  --decode-threads <n> Live flow: PNG decode threads (default: 6)\n");
    printf("  --ofa-grid <n>      Live flow: one vector per n x n pixels, 1/2/4/8 (default: 2)\n");
    printf("  --ofa-levels <n>    Live flow: pyramid levels (default: 4)\n");
    printf("  --ofa-prep <cpu|cuda> Live flow: where grayscale + pyramid run (default: cpu, keeps the GPU free)\n");
    printf("  --vit-flash-attn <on|off> ViT flash attention (default: on; mtmd ignores -fa for the ViT)\n");
    printf("  --window-align      Resize frames to a multiple of the 112 px ViT attention window\n");
    printf("  --window-cache      Re-encode only windows that changed since last encoded (implies --window-align)\n");
    printf("  --wc-threshold <f>  Window cache: dirty if a token cell's mean |dRGB| exceeds this (default: 6)\n");
    printf("  --wc-halo <n>       Window cache: grow the dirty rectangle by n windows (default: 0)\n");
    printf("  --wc-rekey-frac <f> Window cache: crops above this frame fraction are full encodes (default: 0.75)\n");
    printf("  --wc-cost-model     Window cache: pick the crop shape from encode times measured at setup\n");
    printf("  --vit-window-batch  ViT windowed layers as a batch of windows (needs --window-align)\n");
    printf("  --gemm-f32-out      BF16 GEMMs write F32 directly (no BF16->F32 conversion kernel)\n");
    printf("  --gemm-bias-fuse    Fold the bias ADD after a BF16 GEMM into the GEMM (needs --gemm-f32-out)\n");
    printf("  --gemm-table <f>    Per-shape cuBLASLt algorithms from jetson-gemm-tune\n");
    printf("  --gemm-record <f>   Append the BF16 GEMM shapes this run uses (input for jetson-gemm-tune)\n\n");
    printf("Standard llama.cpp options (-p, -n, -ngl, -c, etc.) are also supported.\n");
}

// ============================================================================
// Section 2: Optical flow (.flo) file reader
// ============================================================================

struct optical_flow {
    int width  = 0;
    int height = 0;
    std::vector<float> u; // horizontal flow component, size = width * height
    std::vector<float> v; // vertical flow component,   size = width * height
};

static bool read_flo_file(const std::string & path, optical_flow & flow) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERR("Cannot open .flo file: %s\n", path.c_str());
        return false;
    }

    // Read and verify magic number
    float magic = 0.0f;
    if (fread(&magic, sizeof(float), 1, f) != 1 || magic != 202021.25f) {
        LOG_ERR("Invalid .flo magic in %s (got %f)\n", path.c_str(), magic);
        fclose(f);
        return false;
    }

    int32_t w = 0, h = 0;
    if (fread(&w, sizeof(int32_t), 1, f) != 1 || fread(&h, sizeof(int32_t), 1, f) != 1) {
        LOG_ERR("Cannot read .flo dimensions from %s\n", path.c_str());
        fclose(f);
        return false;
    }

    flow.width  = w;
    flow.height = h;
    const int n_pixels = w * h;

    // Read interleaved (u, v) pairs
    std::vector<float> raw(n_pixels * 2);
    if ((int)fread(raw.data(), sizeof(float), n_pixels * 2, f) != n_pixels * 2) {
        LOG_ERR("Incomplete flow data in %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fclose(f);

    flow.u.resize(n_pixels);
    flow.v.resize(n_pixels);
    for (int i = 0; i < n_pixels; i++) {
        flow.u[i] = raw[2 * i];
        flow.v[i] = raw[2 * i + 1];
    }

    return true;
}

// ============================================================================
// Section 3: Motion vector analysis
// ============================================================================

struct motion_stats {
    float mean_magnitude  = 0.0f;
    float p95_magnitude   = 0.0f;
    float max_magnitude   = 0.0f;
    float variance        = 0.0f;
    float high_motion_ratio = 0.0f;
    float complexity_score  = 0.0f;
};

static motion_stats compute_motion_stats(const optical_flow & flow,
                                         float motion_threshold,
                                         float percentile) {
    const int n = flow.width * flow.height;
    std::vector<float> mag(n);

    double sum = 0.0;
    int high_count = 0;
    for (int i = 0; i < n; i++) {
        mag[i] = std::sqrt(flow.u[i] * flow.u[i] + flow.v[i] * flow.v[i]);
        sum += mag[i];
        if (mag[i] > motion_threshold) {
            high_count++;
        }
    }

    motion_stats stats;
    stats.mean_magnitude = (float)(sum / n);
    stats.high_motion_ratio = (float)high_count / n;

    // Percentile by partial selection: the same element a full sort would put at p_idx
    stats.max_magnitude = *std::max_element(mag.begin(), mag.end());

    int p_idx = std::min((int)(percentile * n), n - 1);
    std::vector<float> sel = mag;
    std::nth_element(sel.begin(), sel.begin() + p_idx, sel.end());
    stats.p95_magnitude = sel[p_idx];

    // Variance
    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = mag[i] - stats.mean_magnitude;
        var_sum += diff * diff;
    }
    stats.variance = (float)(var_sum / n);

    return stats;
}

struct video_motion_profile {
    std::vector<motion_stats> frame_stats;
    float global_complexity = 0.0f;
};

static void finalize_motion_profile(video_motion_profile & profile);

static video_motion_profile build_motion_profile(const std::string & flow_dir,
                                                  const std::string & video_name,
                                                  float motion_threshold,
                                                  float percentile) {
    video_motion_profile profile;

    // Enumerate flow files: frame_0001.flo, frame_0002.flo, ...
    std::string dir = flow_dir + "/" + video_name;
    std::vector<std::string> flow_files;
    for (int i = 1; ; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.flo", i);
        std::string path = dir + buf;
        FILE * test = fopen(path.c_str(), "rb");
        if (!test) break;
        fclose(test);
        flow_files.push_back(path);
    }

    if (flow_files.empty()) {
        LOG_ERR("No flow files found in %s\n", dir.c_str());
        return profile;
    }

    LOG_INF("Found %zu flow files for video '%s'\n", flow_files.size(), video_name.c_str());

    // Read all flow files and compute per-frame stats
    for (const auto & flo_path : flow_files) {
        optical_flow flow;
        if (!read_flo_file(flo_path, flow)) {
            LOG_ERR("Skipping invalid flow file: %s\n", flo_path.c_str());
            motion_stats empty;
            profile.frame_stats.push_back(empty);
            continue;
        }
        profile.frame_stats.push_back(compute_motion_stats(flow, motion_threshold, percentile));
    }

    finalize_motion_profile(profile);
    return profile;
}

// Per-frame complexity and global complexity from the per-frame stats (shared by the
// .flo-file path and live flow).
static void finalize_motion_profile(video_motion_profile & profile) {
    if (profile.frame_stats.empty()) return;

    // Compute per-frame complexity using min-max normalization (for JSON diagnostics)
    float mean_min = 1e9f, mean_max = 0.0f;
    float var_min  = 1e9f, var_max  = 0.0f;
    float range_min = 1e9f, range_max = 0.0f;

    for (auto & s : profile.frame_stats) {
        mean_min = std::min(mean_min, s.mean_magnitude);
        mean_max = std::max(mean_max, s.mean_magnitude);
        var_min  = std::min(var_min, s.variance);
        var_max  = std::max(var_max, s.variance);
        float range = s.max_magnitude;
        range_min = std::min(range_min, range);
        range_max = std::max(range_max, range);
    }

    for (auto & s : profile.frame_stats) {
        float mean_norm = (mean_max > mean_min) ? (s.mean_magnitude - mean_min) / (mean_max - mean_min) : 0.0f;
        float var_norm  = (var_max > var_min) ? (s.variance - var_min) / (var_max - var_min) : 0.0f;
        float range     = s.max_magnitude;
        float range_norm = (range_max > range_min) ? (range - range_min) / (range_max - range_min) : 0.0f;
        mean_norm  = std::min(std::max(mean_norm, 0.0f), 1.0f);
        var_norm   = std::min(std::max(var_norm, 0.0f), 1.0f);
        range_norm = std::min(std::max(range_norm, 0.0f), 1.0f);
        s.complexity_score = 0.4f * mean_norm + 0.3f * var_norm + 0.2f * range_norm + 0.1f * s.high_motion_ratio;
        s.complexity_score = std::min(s.complexity_score, 1.0f);
    }

    // Compute GLOBAL complexity using Python's algorithm:
    // Uses p95 magnitudes per frame with absolute normalization constants
    {
        std::vector<float> p95_values;
        for (const auto & s : profile.frame_stats) {
            p95_values.push_back(s.p95_magnitude);
        }
        const int nf = (int)p95_values.size();

        // Component 1: Mean of p95 values / 100
        double p95_mean = 0.0;
        for (float v : p95_values) p95_mean += v;
        p95_mean /= nf;
        float mean_norm = std::min((float)(p95_mean / 100.0), 1.0f);

        // Component 2: Variance of p95 values / 5000
        double p95_var = 0.0;
        for (float v : p95_values) {
            double d = v - p95_mean;
            p95_var += d * d;
        }
        p95_var /= nf;
        float var_norm = std::min((float)(p95_var / 5000.0), 1.0f);

        // Component 3: Range of p95 values / 150
        float p95_min = *std::min_element(p95_values.begin(), p95_values.end());
        float p95_max = *std::max_element(p95_values.begin(), p95_values.end());
        float p95_range = p95_max - p95_min;
        float range_norm = std::min(p95_range / 150.0f, 1.0f);

        // Component 4: Fraction of p95 values above the 75th percentile of p95 values
        std::vector<float> sorted_p95 = p95_values;
        std::sort(sorted_p95.begin(), sorted_p95.end());
        int p75_idx = std::min((int)(0.75f * nf), nf - 1);
        float p75_threshold = sorted_p95[p75_idx];
        int above_count = 0;
        for (float v : p95_values) {
            if (v > p75_threshold) above_count++;
        }
        float high_motion_ratio = (float)above_count / nf;

        profile.global_complexity = 0.4f * mean_norm + 0.3f * var_norm + 0.2f * range_norm + 0.1f * high_motion_ratio;
        profile.global_complexity = std::min(profile.global_complexity, 1.0f);
    }

    LOG_INF("Motion profile: %zu frames, global complexity = %.3f\n",
            profile.frame_stats.size(), profile.global_complexity);
}

// ============================================================================
// Section 3b: Count total PNG frames in a video directory
// ============================================================================

static int count_video_frames(const std::string & sintel_dir, const std::string & video_name) {
    std::string dir = sintel_dir + "/training/clean/" + video_name;
    int count = 0;
    for (int i = 1; ; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", i);
        FILE * test = fopen((dir + buf).c_str(), "rb");
        if (!test) break;
        fclose(test);
        count++;
    }
    return count;
}

// ============================================================================
// Section 4: Motion-guided frame selector
// ============================================================================

struct selected_frame {
    int   frame_index;   // 1-based (matches frame_XXXX.png)
    float motion_weight;
    float complexity_score;
};

static std::vector<selected_frame> select_frames(const video_motion_profile & profile,
                                                  int min_frames,
                                                  int max_frames,
                                                  int base_frames,
                                                  float global_complexity,
                                                  float selection_percentile,
                                                  bool anchor_first = false) {
    const int n_flow = (int)profile.frame_stats.size();
    if (n_flow == 0) return {};

    // Adaptive frame count using aggressive step function
    // More aggressive reductions (-4/-8) for low-complexity content
    int n_select;
    if (global_complexity > 0.8f) {
        n_select = std::min(base_frames + 6, max_frames);
    } else if (global_complexity > 0.6f) {
        n_select = base_frames + 3;
    } else if (global_complexity > 0.4f) {
        n_select = base_frames;
    } else if (global_complexity > 0.2f) {
        n_select = std::max(base_frames - 4, min_frames);
    } else {
        n_select = std::max(base_frames - 8, min_frames);
    }
    n_select = std::min(n_select, n_flow); // cannot exceed available flow frames

    LOG_INF("Adaptive frame count: %d (complexity=%.4f, base=%d, range=[%d,%d])\n",
            n_select, global_complexity, base_frames, min_frames, max_frames);

    // Build importance weights with percentile-based thresholding (Python algorithm)
    // Collect p95 values and compute the selection threshold
    std::vector<float> p95_values;
    for (int i = 0; i < n_flow; i++) {
        p95_values.push_back(profile.frame_stats[i].p95_magnitude);
    }
    std::vector<float> sorted_p95 = p95_values;
    std::sort(sorted_p95.begin(), sorted_p95.end());
    int thresh_idx = std::min((int)(selection_percentile / 100.0f * n_flow), n_flow - 1);
    float threshold = sorted_p95[thresh_idx];

    LOG_INF("Selection threshold: %.3f (%.0fth percentile of p95 values)\n",
            threshold, selection_percentile);

    // Frames above threshold keep their p95 weight, below get 0.1
    std::vector<double> weights(n_flow);
    for (int i = 0; i < n_flow; i++) {
        if (p95_values[i] >= threshold) {
            weights[i] = (double)p95_values[i];
        } else {
            weights[i] = 0.1;
        }
    }

    // Normalize to probabilities
    double prob_sum = 0.0;
    for (auto w : weights) prob_sum += w;
    std::vector<double> probabilities(n_flow);
    if (prob_sum > 0.0) {
        for (int i = 0; i < n_flow; i++) {
            probabilities[i] = weights[i] / prob_sum;
        }
    } else {
        for (int i = 0; i < n_flow; i++) {
            probabilities[i] = 1.0 / n_flow;
        }
    }

    // Weighted sampling without replacement using discrete_distribution
    std::mt19937 rng(42); // deterministic seed
    std::vector<int> selected_indices;
    std::vector<double> current_probs = probabilities;

    // With anchor_first, frame 1 takes one of the n_select slots unconditionally
    // (its encode has already started); the rest are sampled as usual.
    if (anchor_first) {
        selected_indices.push_back(0);
        current_probs[0] = 0.0;
    }

    for (int k = (int)selected_indices.size(); k < n_select; k++) {
        double total = 0.0;
        for (auto p : current_probs) total += p;
        if (total <= 0.0) break;

        // Re-normalize
        for (int i = 0; i < n_flow; i++) {
            current_probs[i] /= total;
        }

        std::discrete_distribution<int> dist(current_probs.begin(), current_probs.end());
        int idx = dist(rng);

        selected_indices.push_back(idx);
        current_probs[idx] = 0.0; // remove from future selection
    }

    // Sort in temporal order (NO first/last anchoring — matching Python)
    std::sort(selected_indices.begin(), selected_indices.end());

    // Remove duplicates
    selected_indices.erase(std::unique(selected_indices.begin(), selected_indices.end()),
                           selected_indices.end());

    // Build result (flow index i -> frame index i+1)
    std::vector<selected_frame> result;
    for (int idx : selected_indices) {
        selected_frame sf;
        sf.frame_index     = idx + 1; // 1-based frame numbering
        sf.motion_weight   = (float)weights[idx];
        sf.complexity_score = profile.frame_stats[idx].complexity_score;
        result.push_back(sf);
    }

    LOG_INF("Selected %zu frames: ", result.size());
    for (const auto & sf : result) {
        LOG_INF("%d ", sf.frame_index);
    }
    LOG_INF("\n");

    return result;
}

// Uniform frame selection for baseline mode
static std::vector<selected_frame> select_frames_uniform(int n_total_frames, int n_select) {
    n_select = std::min(n_select, n_total_frames);
    std::vector<selected_frame> result;

    if (n_select <= 0 || n_total_frames <= 0) return result;

    // Evenly spaced frame indices
    for (int k = 0; k < n_select; k++) {
        selected_frame sf;
        sf.frame_index = 1 + (int)std::round((double)k * (n_total_frames - 1) / std::max(n_select - 1, 1));
        sf.motion_weight = 1.0f;
        sf.complexity_score = 0.0f;
        result.push_back(sf);
    }

    // Remove duplicates (can happen if n_select > n_total_frames)
    std::sort(result.begin(), result.end(), [](const selected_frame & a, const selected_frame & b) {
        return a.frame_index < b.frame_index;
    });
    result.erase(std::unique(result.begin(), result.end(), [](const selected_frame & a, const selected_frame & b) {
        return a.frame_index == b.frame_index;
    }), result.end());

    LOG_INF("Uniform selection: %zu frames from %d total: ", result.size(), n_total_frames);
    for (const auto & sf : result) {
        LOG_INF("%d ", sf.frame_index);
    }
    LOG_INF("\n");

    return result;
}

// ============================================================================
// Section 5: Spatial patch analyzer
// ============================================================================

struct patch_info {
    int   grid_x, grid_y;
    float motion_ratio;
    float mean_motion;
    float max_motion;
    bool  selected;
};

struct frame_spatial_analysis {
    int frame_index;
    std::vector<patch_info> patches;
    int n_selected_patches;
    int n_total_patches;
    float spatial_savings;
};

static frame_spatial_analysis analyze_spatial_patches(const optical_flow & flow,
                                                      int frame_index,
                                                      int grid_size,
                                                      float spatial_threshold,
                                                      float motion_threshold) {
    frame_spatial_analysis result;
    result.frame_index = frame_index;
    result.n_total_patches = grid_size * grid_size;

    int pw = flow.width / grid_size;
    int ph = flow.height / grid_size;

    for (int gy = 0; gy < grid_size; gy++) {
        for (int gx = 0; gx < grid_size; gx++) {
            int x0 = gx * pw;
            int y0 = gy * ph;
            int x1 = (gx == grid_size - 1) ? flow.width : x0 + pw;
            int y1 = (gy == grid_size - 1) ? flow.height : y0 + ph;

            float sum_mag = 0.0f;
            float max_mag = 0.0f;
            int high_count = 0;
            int total = 0;

            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int idx = y * flow.width + x;
                    float mag = std::sqrt(flow.u[idx] * flow.u[idx] + flow.v[idx] * flow.v[idx]);
                    sum_mag += mag;
                    max_mag = std::max(max_mag, mag);
                    if (mag > motion_threshold) high_count++;
                    total++;
                }
            }

            patch_info pi;
            pi.grid_x = gx;
            pi.grid_y = gy;
            pi.mean_motion = (total > 0) ? sum_mag / total : 0.0f;
            pi.max_motion = max_mag;
            pi.motion_ratio = (total > 0) ? (float)high_count / total : 0.0f;

            // 3-criteria encoding decision
            pi.selected = (pi.motion_ratio > spatial_threshold) ||
                          (pi.mean_motion > motion_threshold) ||
                          (pi.max_motion > motion_threshold * 2.0f);

            result.patches.push_back(pi);
        }
    }

    // Guarantee minimum patches (25% of grid)
    result.n_selected_patches = 0;
    for (const auto & p : result.patches) {
        if (p.selected) result.n_selected_patches++;
    }

    int min_patches = std::max(1, result.n_total_patches / 4);
    if (result.n_selected_patches < min_patches) {
        // Sort by mean_motion descending, select top ones
        std::vector<int> indices(result.patches.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return result.patches[a].mean_motion > result.patches[b].mean_motion;
        });
        for (int idx : indices) {
            if (result.n_selected_patches >= min_patches) break;
            if (!result.patches[idx].selected) {
                result.patches[idx].selected = true;
                result.n_selected_patches++;
            }
        }
    }

    result.spatial_savings = 1.0f - (float)result.n_selected_patches / result.n_total_patches;

    return result;
}

// ============================================================================
// Section 6: Quality metrics
// ============================================================================

static double compute_shannon_entropy(const std::string & text) {
    if (text.empty()) return 0.0;

    int freq[256] = {};
    for (unsigned char c : text) {
        freq[c]++;
    }

    double entropy = 0.0;
    double n = (double)text.size();
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            double p = freq[i] / n;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

static double compute_motion_coverage(const video_motion_profile & profile,
                                       const std::vector<selected_frame> & selected) {
    double total_motion = 0.0;
    double selected_motion = 0.0;

    for (const auto & s : profile.frame_stats) {
        total_motion += s.p95_magnitude;
    }

    for (const auto & sf : selected) {
        int flow_idx = sf.frame_index - 1; // convert 1-based to 0-based
        if (flow_idx >= 0 && flow_idx < (int)profile.frame_stats.size()) {
            selected_motion += profile.frame_stats[flow_idx].p95_magnitude;
        }
    }

    return (total_motion > 0.0) ? selected_motion / total_motion : 0.0;
}

// ============================================================================
// Section 7: Inference context (mirrors mtmd-cli.cpp)
// ============================================================================

struct adaptive_vlm_context {
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

    adaptive_vlm_context(common_params & params, bool vit_fa_off = false)
        : llama_init(common_init_from_params(params)) {

        model = llama_init->model();
        lctx  = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl  = common_sampler_init(model, params.sampling);
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (!model || !lctx) {
            LOG_ERR("Failed to initialize model/context\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);

        // Initialize vision context
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu         = params.mmproj_use_gpu;
        mparams.print_timings   = true;
        mparams.n_threads       = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup          = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (vit_fa_off) {
            // mtmd ignores flash_attn_type for the vision encoder and only disables flash
            // attention when an eval callback is set. Asking to observe no tensors keeps the
            // scheduler computing each split in one call, as without a callback.
            mparams.cb_eval = [](struct ggml_tensor *, bool ask, void *) { return !ask; };
        }

        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }

        LOG_INF("Model and vision context initialized successfully\n");
    }

    ~adaptive_vlm_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }
};

// ============================================================================
// Section 8: Inference function
// ============================================================================

// Where encode time goes: ViT, prefill of image tokens, prefill of text tokens.
struct encode_breakdown {
    float vit_ms         = 0.0f;
    float prefill_img_ms = 0.0f;
    float prefill_txt_ms = 0.0f;
    int   n_images       = 0;
    // window cache
    float wc_diff_ms     = 0.0f;   // pixel diff + rectangle choice (CPU)
    float wc_prep_ms     = 0.0f;   // cropping + tokenizing the crop (CPU)
    int   n_full = 0, n_crop = 0, n_skip = 0;
    int   tokens_encoded = 0;      // tokens that went through the ViT
};

// mtmd_helper_eval_chunk_single, with the ViT encode and the LLM prefill timed separately
// (each followed by a sync so the times do not bleed into the next stage).
static int32_t eval_chunk_timed(mtmd_context * mctx, llama_context * lctx, const mtmd_input_chunk * chunk,
                                llama_pos & n_past, int32_t n_batch, bool logits_last, encode_breakdown & b) {
    int64_t t0 = ggml_time_us();
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        NVTX_PUSH("prefill_txt");
        int32_t ret = mtmd_helper_eval_chunk_single(mctx, lctx, chunk, n_past, 0, n_batch, logits_last, &n_past);
        llama_synchronize(lctx);
        NVTX_POP();
        b.prefill_txt_ms += (ggml_time_us() - t0) / 1000.0f;
        return ret;
    }
    NVTX_PUSH("vit");
    int32_t ret = mtmd_encode_chunk(mctx, chunk);
    NVTX_POP();
    if (ret != 0) return ret;
    int64_t t1 = ggml_time_us();
    NVTX_PUSH("prefill_img");
    ret = mtmd_helper_decode_image_chunk(mctx, lctx, chunk, mtmd_get_output_embd(mctx), n_past, 0, n_batch, &n_past);
    llama_synchronize(lctx);
    NVTX_POP();
    b.vit_ms         += (t1 - t0) / 1000.0f;
    b.prefill_img_ms += (ggml_time_us() - t1) / 1000.0f;
    b.n_images++;
    b.tokens_encoded += (int) mtmd_input_chunk_get_n_tokens(chunk);
    b.n_full++;
    return ret;
}

// ============================================================================
// Window grid and window cache (ported from dmc955/window_cache_vlm.cpp)
// ============================================================================
//
// Qwen2.5-VL's ViT attends within 112 px windows (4x4 LLM tokens) in 28 of its 32 layers. With frames
// resized to a multiple of 112 px, a crop of whole windows is partitioned into the same windows as the
// full frame, so re-encoding only the windows whose pixels changed since they were last encoded, and
// reusing the cached embeddings for the rest, reproduces the windowed layers exactly; only the 4
// global layers see less context. The LLM still prefills every token of every frame.

static const int TOKEN_PX      = 28;   // one LLM token = 2x2 patches of 14 px
static const int WINDOW_PX     = 112;  // ViT attention window
static const int WINDOW_TOKENS = WINDOW_PX / TOKEN_PX;

static bool load_rgb(const std::string & path, rgb_frame & f) {
    // nullptr context: for image files the helper only decodes
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(nullptr, path.c_str()));
    if (!bmp.ptr) return false;
    f.w = (int) bmp.nx();
    f.h = (int) bmp.ny();
    f.px.assign(bmp.data(), bmp.data() + bmp.n_bytes());
    return true;
}

static int round_to(int v, int f) { return std::max(f, (int) std::lround((double) v / f) * f); }

static rgb_frame resize_bilinear(const rgb_frame & src, int W, int H) {
    if (src.w == W && src.h == H) return src;
    rgb_frame dst;
    dst.w = W; dst.h = H;
    dst.px.resize((size_t) W * H * 3);
    const float sx = (float) src.w / W, sy = (float) src.h / H;
    for (int y = 0; y < H; y++) {
        float fy = std::max(0.0f, (y + 0.5f) * sy - 0.5f);
        int y0 = std::min((int) fy, src.h - 1), y1 = std::min(y0 + 1, src.h - 1);
        float wy = fy - y0;
        for (int x = 0; x < W; x++) {
            float fx = std::max(0.0f, (x + 0.5f) * sx - 0.5f);
            int x0 = std::min((int) fx, src.w - 1), x1 = std::min(x0 + 1, src.w - 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float a = src.px[((size_t) y0 * src.w + x0) * 3 + c], b = src.px[((size_t) y0 * src.w + x1) * 3 + c];
                float d = src.px[((size_t) y1 * src.w + x0) * 3 + c], e = src.px[((size_t) y1 * src.w + x1) * 3 + c];
                float v = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
                dst.px[((size_t) y * W + x) * 3 + c] = (unsigned char) std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
    return dst;
}

static rgb_frame align_to_windows(const rgb_frame & f) {
    return resize_bilinear(f, round_to(f.w, WINDOW_PX), round_to(f.h, WINDOW_PX));
}

static rgb_frame crop_rgb(const rgb_frame & src, int x, int y, int w, int h) {
    rgb_frame dst;
    dst.w = w; dst.h = h;
    dst.px.resize((size_t) w * h * 3);
    for (int r = 0; r < h; r++) {
        memcpy(&dst.px[(size_t) r * w * 3], &src.px[((size_t) (y + r) * src.w + x) * 3], (size_t) w * 3);
    }
    return dst;
}

static void copy_region(rgb_frame & dst, const rgb_frame & src, int x, int y, int w, int h) {
    for (int r = 0; r < h; r++) {
        size_t off = ((size_t) (y + r) * src.w + x) * 3;
        memcpy(&dst.px[off], &src.px[off], (size_t) w * 3);
    }
}

// Per-window change score: the largest mean |dRGB| among the window's 4x4 token cells, so a small
// moving object is not averaged away by the static remainder of its window.
static std::vector<float> window_diff(const rgb_frame & cur, const rgb_frame & ref, int nwx, int nwy) {
    std::vector<float> score((size_t) nwx * nwy, 0.0f);
    const int ntx = nwx * WINDOW_TOKENS, nty = nwy * WINDOW_TOKENS;
    for (int ty = 0; ty < nty; ty++) {
        std::vector<uint32_t> acc(ntx, 0);
        for (int r = 0; r < TOKEN_PX; r++) {
            const unsigned char * a = &cur.px[(size_t) (ty * TOKEN_PX + r) * cur.w * 3];
            const unsigned char * b = &ref.px[(size_t) (ty * TOKEN_PX + r) * ref.w * 3];
            for (int tx = 0; tx < ntx; tx++) {
                uint32_t sum = 0;
                for (int k = tx * TOKEN_PX * 3; k < (tx + 1) * TOKEN_PX * 3; k++) {
                    sum += (uint32_t) std::abs((int) a[k] - (int) b[k]);
                }
                acc[tx] += sum;
            }
        }
        for (int tx = 0; tx < ntx; tx++) {
            float m = (float) acc[tx] / (TOKEN_PX * TOKEN_PX * 3);
            float & dst = score[(size_t) (ty / WINDOW_TOKENS) * nwx + tx / WINDOW_TOKENS];
            dst = std::max(dst, m);
        }
    }
    return score;
}

struct encoded_region {
    mtmd::input_chunks chunks{mtmd_input_chunks_init()};
    const mtmd_input_chunk * chunk = nullptr;
    int nx = 0, ny = 0;
};

static bool tokenize_region(mtmd_context * mctx, const rgb_frame & img, encoded_region & out) {
    mtmd::bitmap bmp((uint32_t) img.w, (uint32_t) img.h, img.px.data());
    std::string marker(mtmd_default_marker());
    mtmd_input_text text;
    text.text          = marker.c_str();
    text.add_special   = false;
    text.parse_special = true;
    const mtmd_bitmap * bptr = bmp.ptr.get();
    if (mtmd_tokenize(mctx, out.chunks.ptr.get(), &text, &bptr, 1) != 0) return false;
    for (size_t k = 0; k < out.chunks.size(); k++) {
        if (mtmd_input_chunk_get_type(out.chunks[k]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            out.chunk = out.chunks[k];
            const auto it = mtmd_input_chunk_get_tokens_image(out.chunk);
            out.nx = (int) mtmd_image_tokens_get_nx(it);
            out.ny = (int) mtmd_image_tokens_get_ny(it);
            return true;
        }
    }
    return false;
}

// Measured encode time per window-rectangle shape. `first` follows a different shape (pays for the
// backend rebuilding its cached execution plan), `steady` repeats the same shape.
struct cost_table {
    int nwx = 0, nwy = 0;
    std::vector<float> first_ms, steady_ms;
    bool valid = false;
    float & first(int w, int h)  { return first_ms[(size_t) (h - 1) * nwx + (w - 1)]; }
    float & steady(int w, int h) { return steady_ms[(size_t) (h - 1) * nwx + (w - 1)]; }
    float cost(int w, int h, int last_w, int last_h) { return (w == last_w && h == last_h) ? steady(w, h) : first(w, h); }
};

static bool calibrate_cost_table(mtmd_context * mctx, const rgb_frame & frame, int reps, cost_table & table) {
    table.nwx = frame.w / WINDOW_PX;
    table.nwy = frame.h / WINDOW_PX;
    table.first_ms.assign((size_t) table.nwx * table.nwy, 0.0f);
    table.steady_ms.assign((size_t) table.nwx * table.nwy, 0.0f);
    for (int h = 1; h <= table.nwy; h++) {
        for (int w = 1; w <= table.nwx; w++) {
            rgb_frame crop = crop_rgb(frame, 0, 0, w * WINDOW_PX, h * WINDOW_PX);
            double prep_ms = 0.0, first = 0.0, steady = 1e30;
            for (int r = 0; r < std::max(2, reps); r++) {
                encoded_region reg;
                int64_t t0 = ggml_time_us();
                if (!tokenize_region(mctx, crop, reg)) return false;
                int64_t t1 = ggml_time_us();
                if (mtmd_encode_chunk(mctx, reg.chunk) != 0) return false;
                const double ms = (ggml_time_us() - t1) / 1000.0;
                prep_ms += (t1 - t0) / 1000.0;
                if (r == 0) first = ms; else steady = std::min(steady, ms);
            }
            prep_ms /= std::max(2, reps);
            table.first(w, h)  = (float) (first + prep_ms);
            table.steady(w, h) = (float) (steady + prep_ms);
        }
    }
    table.valid = true;
    return true;
}

struct wc_frame_record {
    std::string action;            // key | full | crop | skip
    int   n_dirty = 0, crop_w = 0, crop_h = 0, tokens_encoded = 0;
    float diff_ms = 0.0f, prep_ms = 0.0f, encode_ms = 0.0f;
};

struct wc_state {
    bool  enabled    = false;
    float threshold  = 6.0f;
    int   halo       = 0;
    float rekey_frac = 0.75f;
    cost_table table;
    std::vector<float> cache;      // embeddings handed to the LLM, raster order
    int   nx = 0, ny = 0;          // token grid of the cached frame
    rgb_frame ref;                 // pixels each window was last encoded from
    int   last_w = 0, last_h = 0;  // window shape encoded last (cost model)
    std::vector<wc_frame_record> frames;
};

// Encode an image chunk through the window cache, then prefill the frame from the cache.
// `img` is the (window-aligned) frame the chunk was tokenized from.
static int32_t eval_image_chunk_wc(mtmd_context * mctx, llama_context * lctx, const mtmd_input_chunk * chunk,
                                   const rgb_frame & img, int n_embd, wc_state & wc,
                                   llama_pos & n_past, int32_t n_batch, encode_breakdown & b) {
    const auto * it = mtmd_input_chunk_get_tokens_image(chunk);
    const int cnx = (int) mtmd_image_tokens_get_nx(it), cny = (int) mtmd_image_tokens_get_ny(it);
    const int n_tok = cnx * cny;
    if (img.w % WINDOW_PX || img.h % WINDOW_PX || cnx * TOKEN_PX != img.w || cny * TOKEN_PX != img.h) {
        LOG_ERR("window cache needs window-aligned frames (frame %dx%d px, %dx%d tokens)\n", img.w, img.h, cnx, cny);
        return -1;
    }
    const int nwx = cnx / WINDOW_TOKENS, nwy = cny / WINDOW_TOKENS;
    wc_frame_record fr;

    auto encode_full = [&](const char * action) -> bool {
        int64_t t = ggml_time_us();
        NVTX_PUSH("vit");
        const bool ok = mtmd_encode_chunk(mctx, chunk) == 0;
        NVTX_POP();
        if (!ok) return false;
        fr.encode_ms = (ggml_time_us() - t) / 1000.0f;
        const float * embd = mtmd_get_output_embd(mctx);
        wc.cache.assign(embd, embd + (size_t) n_tok * n_embd);
        wc.nx = cnx; wc.ny = cny;
        wc.ref = img;
        wc.last_w = nwx; wc.last_h = nwy;
        fr.action = action;
        fr.tokens_encoded = n_tok;
        return true;
    };

    const bool have_cache = !wc.cache.empty() && cnx == wc.nx && cny == wc.ny;
    if (!have_cache) {
        if (!encode_full("key")) return -1;
    } else {
        int64_t t = ggml_time_us();
        NVTX_PUSH("wc_diff");
        std::vector<float> score = window_diff(img, wc.ref, nwx, nwy);
        int x0 = nwx, x1 = -1, y0 = nwy, y1 = -1;
        for (int wy = 0; wy < nwy; wy++) {
            for (int wx = 0; wx < nwx; wx++) {
                if (score[(size_t) wy * nwx + wx] > wc.threshold) {
                    fr.n_dirty++;
                    x0 = std::min(x0, wx); x1 = std::max(x1, wx);
                    y0 = std::min(y0, wy); y1 = std::max(y1, wy);
                }
            }
        }
        int cw = 0, ch = 0;
        if (fr.n_dirty > 0) {
            x0 = std::max(0, x0 - wc.halo); y0 = std::max(0, y0 - wc.halo);
            x1 = std::min(nwx - 1, x1 + wc.halo); y1 = std::min(nwy - 1, y1 + wc.halo);
            const int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
            cw = bw; ch = bh;
            if (wc.table.valid) {   // cheapest rectangle containing the dirty one (incl. the full frame)
                float best = 1e30f;
                for (int h = bh; h <= nwy; h++) {
                    for (int w = bw; w <= nwx; w++) {
                        float c = wc.table.cost(w, h, wc.last_w, wc.last_h);
                        if (c < best) { best = c; cw = w; ch = h; }
                    }
                }
            } else if ((float) (bw * bh) > wc.rekey_frac * nwx * nwy) {
                cw = nwx; ch = nwy;
            }
        }
        NVTX_POP();
        fr.diff_ms = (ggml_time_us() - t) / 1000.0f;

        if (fr.n_dirty == 0) {
            fr.action = "skip";
        } else if (cw == nwx && ch == nwy) {
            if (!encode_full("full")) return -1;
        } else {
            const int cx = std::min(x0, nwx - cw), cy = std::min(y0, nwy - ch);
            t = ggml_time_us();
            rgb_frame crop = crop_rgb(img, cx * WINDOW_PX, cy * WINDOW_PX, cw * WINDOW_PX, ch * WINDOW_PX);
            encoded_region reg;
            if (!tokenize_region(mctx, crop, reg) || reg.nx != cw * WINDOW_TOKENS || reg.ny != ch * WINDOW_TOKENS) {
                LOG_ERR("crop %dx%d windows tokenized to %dx%d tokens\n", cw, ch, reg.nx, reg.ny);
                return -1;
            }
            fr.prep_ms = (ggml_time_us() - t) / 1000.0f;
            t = ggml_time_us();
            NVTX_PUSH("vit");
            const bool ok = mtmd_encode_chunk(mctx, reg.chunk) == 0;
            NVTX_POP();
            if (!ok) return -1;
            fr.encode_ms = (ggml_time_us() - t) / 1000.0f;
            const float * embd = mtmd_get_output_embd(mctx);
            for (int r = 0; r < reg.ny; r++) {
                const int ty = cy * WINDOW_TOKENS + r, tx = cx * WINDOW_TOKENS;
                memcpy(&wc.cache[((size_t) ty * wc.nx + tx) * n_embd], embd + (size_t) r * reg.nx * n_embd,
                       (size_t) reg.nx * n_embd * sizeof(float));
            }
            copy_region(wc.ref, img, cx * WINDOW_PX, cy * WINDOW_PX, cw * WINDOW_PX, ch * WINDOW_PX);
            fr.action = "crop";
            fr.crop_w = cw; fr.crop_h = ch;
            fr.tokens_encoded = reg.nx * reg.ny;
            wc.last_w = cw; wc.last_h = ch;
        }
    }

    b.vit_ms     += fr.encode_ms;
    b.wc_diff_ms += fr.diff_ms;
    b.wc_prep_ms += fr.prep_ms;
    b.tokens_encoded += fr.tokens_encoded;
    if (fr.action == "skip") b.n_skip++; else if (fr.action == "crop") b.n_crop++; else b.n_full++;

    int64_t t1 = ggml_time_us();
    NVTX_PUSH("prefill_img");
    const int32_t ret = mtmd_helper_decode_image_chunk(mctx, lctx, chunk, wc.cache.data(), n_past, 0, n_batch, &n_past);
    llama_synchronize(lctx);
    NVTX_POP();
    b.prefill_img_ms += (ggml_time_us() - t1) / 1000.0f;
    b.n_images++;
    wc.frames.push_back(fr);
    return ret;
}

struct inference_result {
    std::string generated_text;
    int   n_tokens_generated = 0;
    float encode_time_ms     = 0.0f;
    float inference_time_ms  = 0.0f;
    int64_t t_encode_end_ms  = 0;    // absolute time the first token became sampleable
    float image_load_ms      = 0.0f; // loading the remaining frames' bitmaps
    float tokenize_ms        = 0.0f; // chat formatting + mtmd tokenization
    encode_breakdown rest;           // everything evaluated by run_inference (after the anchor prefix, if any)
};

// State left behind by eval_anchor_prefix() for run_inference() to continue from.
struct anchor_prefix {
    bool        ok = false;
    std::string text;                // formatted prompt up to and including the first media marker
    llama_tokens tail;               // text tokens after the image, held back to share a batch with the next text chunk
    float       eval_ms = 0.0f;
    float       load_ms = 0.0f;      // decoding frame 1
    int64_t     t_done_ms = 0;       // absolute time the prefix was in the KV cache
    encode_breakdown bd;
};

// Evaluate the part of the prompt that does not depend on frame selection: the chat
// header and frame 1 (ViT encode + prefill). Runs on a worker thread while the main
// thread does the CPU-only motion analysis.
static anchor_prefix eval_anchor_prefix(adaptive_vlm_context & ctx,
                                        const std::string & frame_path,
                                        const std::string & prompt,
                                        bool align = false,
                                        wc_state * wc = nullptr) {
    anchor_prefix pre;
    int64_t t_start = ggml_time_ms();
    NVTX_PUSH("anchor");
    struct nvtx_pop_guard { ~nvtx_pop_guard() { NVTX_POP(); } } nvtx_guard;

    rgb_frame img;
    if (!load_rgb(frame_path, img)) {
        LOG_ERR("Failed to load image: %s\n", frame_path.c_str());
        return pre;
    }
    if (align) img = align_to_windows(img);
    mtmd::bitmap bmp((uint32_t) img.w, (uint32_t) img.h, img.px.data());
    pre.load_ms = (float)(ggml_time_ms() - t_start);

    // The prompt is known up front, only the number of markers is not, so format
    // with a single marker and cut right after it.
    const std::string marker = mtmd_default_marker();
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = marker + "\n" + prompt;
    auto formatted = common_chat_format_single(
        ctx.tmpls.get(), ctx.chat_history, msg, true /* add_ass */, false /* use_jinja */);
    size_t marker_pos = formatted.find(marker);
    if (marker_pos == std::string::npos) {
        LOG_ERR("Media marker not found in formatted prompt\n");
        return pre;
    }
    pre.text = formatted.substr(0, marker_pos + marker.size());

    mtmd_input_text text;
    text.text          = pre.text.c_str();
    text.add_special   = ctx.chat_history.empty();
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    const mtmd_bitmap * bmp_c_ptr = bmp.ptr.get();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(), chunks.ptr.get(), &text, &bmp_c_ptr, 1);
    if (res != 0) {
        LOG_ERR("Failed to tokenize anchor prefix (error %d)\n", res);
        return pre;
    }

    bool seen_image = false;
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        const auto type = mtmd_input_chunk_get_type(chunk);
        if (seen_image && type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            pre.tail.insert(pre.tail.end(), tokens, tokens + n_tokens);
            continue;
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) seen_image = true;
        const int32_t rc = type == MTMD_INPUT_CHUNK_TYPE_IMAGE && wc && wc->enabled
            ? eval_image_chunk_wc(ctx.ctx_vision.get(), ctx.lctx, chunk, img, llama_model_n_embd_inp(ctx.model), *wc,
                                  ctx.n_past, ctx.n_batch, pre.bd)
            : eval_chunk_timed(ctx.ctx_vision.get(), ctx.lctx, chunk, ctx.n_past, ctx.n_batch, false /* logits_last */, pre.bd);
        if (rc) {
            LOG_ERR("Failed to evaluate anchor prefix chunk %zu\n", i);
            return pre;
        }
    }
    llama_synchronize(ctx.lctx);

    pre.t_done_ms = ggml_time_ms();
    pre.eval_ms = (float)(pre.t_done_ms - t_start);
    pre.ok      = true;
    LOG_INF("Anchor prefix evaluated. n_past = %d, %.1f ms\n", (int)ctx.n_past, pre.eval_ms);
    return pre;
}

static inference_result run_inference(adaptive_vlm_context & ctx,
                                      const std::vector<std::string> & frame_paths,
                                      const std::string & prompt,
                                      int n_predict,
                                      const anchor_prefix * prefix = nullptr,
                                      const std::vector<const rgb_frame *> * preloaded = nullptr,
                                      bool align = false,
                                      wc_state * wc = nullptr) {
    inference_result result;
    int64_t t_start = ggml_time_ms();

    // Load frames (with a prefix, frame_paths[0] is already in the KV cache). Frames already
    // decoded for live flow are reused instead of decoding the PNG again; with `align` every
    // frame is resized to the 112 px window grid first.
    std::vector<rgb_frame> imgs;
    mtmd::bitmaps bitmaps;
    for (size_t pi = prefix ? 1 : 0; pi < frame_paths.size(); pi++) {
        const auto & path = frame_paths[pi];
        const rgb_frame * pre = preloaded && pi < preloaded->size() ? (*preloaded)[pi] : nullptr;
        rgb_frame img;
        if (pre && !pre->px.empty()) {
            img = *pre;
        } else if (!load_rgb(path, img)) {
            LOG_ERR("Failed to load image: %s\n", path.c_str());
            continue;
        }
        imgs.push_back(align ? align_to_windows(img) : std::move(img));
    }
    for (const auto & img : imgs) {
        bitmaps.entries.emplace_back((uint32_t) img.w, (uint32_t) img.h, img.px.data());
    }
    result.image_load_ms = (float)(ggml_time_ms() - t_start);
    const int n_embd = llama_model_n_embd_inp(ctx.model);
    size_t img_i = 0;   // next frame in imgs, in chunk order
    auto eval_chunk = [&](const mtmd_input_chunk * chunk, bool logits_last) -> int32_t {
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE && wc && wc->enabled) {
            GGML_ASSERT(img_i < imgs.size());
            return eval_image_chunk_wc(ctx.ctx_vision.get(), ctx.lctx, chunk, imgs[img_i++], n_embd, *wc,
                                       ctx.n_past, ctx.n_batch, result.rest);
        }
        return eval_chunk_timed(ctx.ctx_vision.get(), ctx.lctx, chunk, ctx.n_past, ctx.n_batch, logits_last, result.rest);
    };

    if (bitmaps.entries.empty() && !prefix) {
        LOG_ERR("No images loaded for inference\n");
        return result;
    }

    LOG_INF("Loaded %zu images for inference\n", bitmaps.entries.size());

    // Build prompt with media markers (one per image)
    std::string full_content;
    for (size_t i = 0; i < bitmaps.entries.size() + (prefix ? 1 : 0); i++) {
        full_content += mtmd_default_marker();
    }
    full_content += "\n" + prompt;

    // Format as chat message
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = full_content;

    bool add_bos = ctx.chat_history.empty();
    auto formatted = common_chat_format_single(
        ctx.tmpls.get(), ctx.chat_history, msg, true /* add_ass */, false /* use_jinja */);
    ctx.chat_history.push_back(msg);

    LOG_INF("Formatted prompt length: %zu chars\n", formatted.size());

    // With a prefix, only what follows the first marker is still to be evaluated
    if (prefix) {
        if (formatted.compare(0, prefix->text.size(), prefix->text) != 0) {
            LOG_ERR("Anchor prefix does not match the formatted prompt\n");
            return result;
        }
        formatted = formatted.substr(prefix->text.size());
        add_bos   = false;
    }

    // Tokenize with media
    mtmd_input_text text;
    text.text        = formatted.c_str();
    text.add_special = add_bos;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                                chunks.ptr.get(),
                                &text,
                                bitmaps_c_ptr.data(),
                                bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Failed to tokenize prompt (error %d)\n", res);
        return result;
    }

    // Evaluate all chunks (encode images + decode text)
    int64_t t_encode_start = ggml_time_ms();
    result.tokenize_ms = (float)(t_encode_start - t_start) - result.image_load_ms;
    if (prefix) {
        const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());

        // The held-back tail of the prefix and the first text chunk here are adjacent
        // text; decode them as one batch, as a single tokenization would have.
        llama_tokens first = prefix->tail;
        size_t i_next = 0;
        if (n_chunks > 0) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), 0);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                first.insert(first.end(), tokens, tokens + n_tokens);
                i_next = 1;
            }
        }
        if (!first.empty()) {
            int64_t t_txt = ggml_time_us();
            llama_batch text_batch = llama_batch_init((int32_t)first.size(), 0, 1);
            for (size_t k = 0; k < first.size(); k++) {
                bool want_logits = (i_next == n_chunks) && (k + 1 == first.size());
                common_batch_add(text_batch, first[k], ctx.n_past++, {0}, want_logits);
            }
            int32_t ret = llama_decode(ctx.lctx, text_batch);
            llama_batch_free(text_batch);
            if (ret != 0) {
                LOG_ERR("Failed to decode text after anchor prefix\n");
                return result;
            }
            llama_synchronize(ctx.lctx);
            result.rest.prefill_txt_ms += (ggml_time_us() - t_txt) / 1000.0f;
        }
        for (size_t i = i_next; i < n_chunks; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
            if (eval_chunk(chunk, i == n_chunks - 1 /* logits_last */)) {
                LOG_ERR("Failed to evaluate chunk %zu\n", i);
                return result;
            }
        }
    } else {
        // Same as mtmd_helper_eval_chunks (logits on the last chunk only), with stage timing.
        const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
        for (size_t i = 0; i < n_chunks; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
            if (eval_chunk(chunk, i == n_chunks - 1 /* logits_last */)) {
                LOG_ERR("Failed to evaluate chunk %zu\n", i);
                return result;
            }
        }
    }
    result.t_encode_end_ms = ggml_time_ms();
    result.encode_time_ms  = (float)(result.t_encode_end_ms - t_encode_start) + (prefix ? prefix->eval_ms : 0.0f);

    LOG_INF("Chunks evaluated. n_past = %d, encode_time = %.1f ms\n",
            (int)ctx.n_past, result.encode_time_ms);

    // Generate tokens
    NVTX_PUSH("decode");
    int64_t t_gen_start = ggml_time_ms();
    std::string generated_text;
    llama_tokens generated_tokens;

    for (int i = 0; i < n_predict; i++) {
        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            break;
        }

        std::string piece = common_token_to_piece(ctx.lctx, token_id);
        generated_text += piece;
        printf("%s", piece.c_str());
        fflush(stdout);

        // Prepare next batch
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("Failed to decode token\n");
            break;
        }
    }
    printf("\n");
    NVTX_POP();

    result.generated_text    = generated_text;
    result.n_tokens_generated = (int)generated_tokens.size();
    result.inference_time_ms  = (float)(ggml_time_ms() - t_gen_start);

    LOG_INF("Generated %d tokens in %.1f ms (%.1f t/s)\n",
            result.n_tokens_generated, result.inference_time_ms,
            result.n_tokens_generated / (result.inference_time_ms / 1000.0f));

    return result;
}

// ============================================================================
// Section 9: JSON output
// ============================================================================

static void write_results_json(const std::string & output_path,
                                const sintel_params & sparams,
                                int n_total_frames,
                                const video_motion_profile & profile,
                                const std::vector<selected_frame> & selected,
                                const std::vector<frame_spatial_analysis> & spatial,
                                const inference_result & inf_result,
                                double motion_coverage,
                                double entropy,
                                float flow_time_ms,
                                float select_time_ms,
                                float ttft_ms,
                                float total_time_ms,
                                const json & breakdown = json()) {
    json j;

    // Mode and video info
    j["mode"]             = sparams.baseline ? "baseline" : "adaptive";
    j["video_name"]       = sparams.video_name;
    j["n_flow_files"]     = (int)profile.frame_stats.size();
    j["n_frames_total"]   = n_total_frames;
    j["global_complexity"] = profile.global_complexity;
    j["n_frames_selected"] = (int)selected.size();

    // Selected frames
    json sel_arr = json::array();
    for (const auto & sf : selected) {
        json sf_j;
        sf_j["index"]      = sf.frame_index;
        sf_j["weight"]     = sf.motion_weight;
        sf_j["complexity"] = sf.complexity_score;
        sel_arr.push_back(sf_j);
    }
    j["selected_frames"] = sel_arr;

    // Motion profile
    json mp_arr = json::array();
    for (int i = 0; i < (int)profile.frame_stats.size(); i++) {
        const auto & s = profile.frame_stats[i];
        json m;
        m["frame"]      = i + 1;
        m["mean_mag"]   = s.mean_magnitude;
        m["p95_mag"]    = s.p95_magnitude;
        m["variance"]   = s.variance;
        m["complexity"] = s.complexity_score;
        mp_arr.push_back(m);
    }
    j["motion_profile"] = mp_arr;

    // Spatial analysis
    json sp_arr = json::array();
    for (const auto & sa : spatial) {
        json sa_j;
        sa_j["frame_index"]       = sa.frame_index;
        sa_j["n_selected_patches"] = sa.n_selected_patches;
        sa_j["n_total_patches"]    = sa.n_total_patches;
        sa_j["spatial_savings"]    = sa.spatial_savings;

        json patches_arr = json::array();
        for (const auto & p : sa.patches) {
            json pj;
            pj["grid_x"]       = p.grid_x;
            pj["grid_y"]       = p.grid_y;
            pj["motion_ratio"] = p.motion_ratio;
            pj["mean_motion"]  = p.mean_motion;
            pj["max_motion"]   = p.max_motion;
            pj["selected"]     = p.selected;
            patches_arr.push_back(pj);
        }
        sa_j["patches"] = patches_arr;
        sp_arr.push_back(sa_j);
    }
    j["spatial_analysis"] = sp_arr;

    // Inference results
    j["inference"]["generated_text"]    = inf_result.generated_text;
    j["inference"]["n_tokens"]          = inf_result.n_tokens_generated;
    j["inference"]["encode_time_ms"]    = inf_result.encode_time_ms;
    j["inference"]["generation_time_ms"] = inf_result.inference_time_ms;

    // Quality metrics
    j["metrics"]["motion_coverage"]   = motion_coverage;
    j["metrics"]["shannon_entropy"]   = entropy;
    j["metrics"]["temporal_reduction"] = 1.0 - (double)selected.size() / n_total_frames;

    double avg_spatial_savings = 0.0;
    if (!spatial.empty()) {
        for (const auto & sa : spatial) avg_spatial_savings += sa.spatial_savings;
        avg_spatial_savings /= spatial.size();
    }
    j["metrics"]["avg_spatial_savings"] = avg_spatial_savings;

    // Timing
    j["timing"]["flow_analysis_ms"]    = flow_time_ms;
    j["timing"]["frame_selection_ms"]  = select_time_ms;
    j["timing"]["encode_ms"]           = inf_result.encode_time_ms;
    j["timing"]["generation_ms"]       = inf_result.inference_time_ms;
    j["timing"]["ttft_ms"]             = ttft_ms;
    j["timing"]["total_ms"]            = total_time_ms;
    if (!breakdown.is_null()) {
        j["timing"]["breakdown"]       = breakdown;
    }

    // Parameters
    j["params"]["sintel_dir"]         = sparams.sintel_dir;
    j["switches"] = {
        {"selection",        sparams.baseline ? "uniform" : "adaptive"},
        {"flow",             sparams.baseline ? "none" : sparams.live_flow ? (sparams.live_flow_serial ? "live_serial" : "live_pipelined") : "files"},
        {"anchor_first",     sparams.anchor_first && !sparams.baseline},
        {"window_align",     sparams.window_align},
        {"window_cache",     sparams.window_cache},
        {"wc_threshold",     sparams.wc_threshold},
        {"wc_halo",          sparams.wc_halo},
        {"wc_rekey_frac",    sparams.wc_rekey_frac},
        {"wc_cost_model",    sparams.wc_cost_model},
        {"vit_flash_attn",   sparams.vit_fa_off ? "off" : "on"},
        {"vit_window_batch", sparams.vit_window_batch},
        {"gemm_f32_out",     sparams.gemm_f32_out},
        {"gemm_bias_fuse",   sparams.gemm_bias_fuse && sparams.gemm_f32_out},
        {"gemm_table",       sparams.gemm_table},
    };
    j["params"]["video_name"]         = sparams.video_name;
    j["params"]["min_frames"]         = sparams.min_frames;
    j["params"]["max_frames"]         = sparams.max_frames;
    j["params"]["base_frames"]        = sparams.base_frames;
    j["params"]["motion_threshold"]   = sparams.motion_threshold;
    j["params"]["motion_percentile"]  = sparams.motion_percentile;
    j["params"]["patch_grid"]         = sparams.patch_grid;
    j["params"]["spatial_threshold"]  = sparams.spatial_threshold;

    // Write to file
    std::ofstream out(output_path);
    if (!out.is_open()) {
        LOG_ERR("Cannot write JSON to %s\n", output_path.c_str());
        return;
    }
    out << j.dump(2) << std::endl;
    out.close();

    LOG_INF("Results written to %s\n", output_path.c_str());
}

// ============================================================================
// Section 10: Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    // Parse custom sintel arguments first (they are stripped from argv)
    sintel_params sparams;
    auto filtered_argv = filter_custom_args(argc, argv, sparams);
    int filtered_argc = (int)filtered_argv.size();

    // Parse standard llama.cpp parameters
    common_params params;
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    // Engine switches are read from the environment once, on first use, so set them before any
    // model or backend initialisation.
    if (sparams.vit_window_batch) setenv("MTMD_VIT_WINDOW_BATCH", "1", 1);
    if (sparams.gemm_f32_out)     setenv("GGML_CUDA_JETSON_GEMM_F32OUT", "1", 1);
    if (sparams.gemm_bias_fuse)   setenv("GGML_CUDA_JETSON_GEMM_BIAS", "1", 1);
    if (!sparams.gemm_table.empty())  setenv("GGML_CUDA_JETSON_GEMM_TABLE", sparams.gemm_table.c_str(), 1);
    if (!sparams.gemm_record.empty()) setenv("GGML_CUDA_JETSON_GEMM_RECORD", sparams.gemm_record.c_str(), 1);

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (sparams.vit_window_batch && !sparams.window_align) {
        LOG_WRN("--vit-window-batch only applies to frames that are a whole number of windows; "
                "without --window-align Sintel frames (1024x436) use the dense path\n");
    }
    if (sparams.gemm_bias_fuse && !sparams.gemm_f32_out) {
        LOG_WRN("--gemm-bias-fuse needs --gemm-f32-out; bias fusion stays off\n");
    }

    // Set default model paths if not provided
    if (params.model.path.empty()) {
        params.model.path = "/home/nvidia/data/models/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf";
    }
    if (params.mmproj.path.empty()) {
        params.mmproj.path = "/home/nvidia/data/models/Qwen2.5VL/mmproj-BF16.gguf";
    }

    int n_predict = params.n_predict < 0 ? 512 : params.n_predict;

    std::string prompt = params.prompt.empty()
        ? "Analyze these video frames from MPI-Sintel. Describe the scene, motion patterns, and key visual elements."
        : params.prompt;

    const char * mode_str = sparams.baseline ? "BASELINE (uniform)" : "ADAPTIVE (motion-guided)";

    LOG_INF("\n=== Spatio-Temporal VLM Optimizer [%s] ===\n", mode_str);
    LOG_INF("Video: %s\n", sparams.video_name.c_str());
    LOG_INF("Dataset: %s\n", sparams.sintel_dir.c_str());

    int64_t t_total_start = ggml_time_ms();

    // Count total frames available
    int n_total_frames = count_video_frames(sparams.sintel_dir, sparams.video_name);
    if (n_total_frames == 0) {
        LOG_ERR("No frames found for video '%s'. Exiting.\n", sparams.video_name.c_str());
        return 1;
    }
    LOG_INF("Total frames available: %d\n", n_total_frames);

    std::string flow_dir = sparams.sintel_dir + "/training/flow";

    // --live-flow: optical flow is computed from the frames on the OFA during the request.
    // The engine (VPI streams, buffers, OFA warm-up) is set up here, outside the request,
    // like the model load.
    const bool live_flow = sparams.live_flow && !sparams.baseline;
    std::vector<std::string> all_frame_paths;
    for (int i = 1; i <= n_total_frames; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", i);
        all_frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }
#ifdef HAVE_VPI
    ofa_flow_engine ofa;
    if (live_flow) {
        mtmd::bitmap probe(mtmd_helper_bitmap_init_from_file(nullptr, all_frame_paths[0].c_str()));
        if (!probe.ptr) {
            LOG_ERR("Cannot read %s\n", all_frame_paths[0].c_str());
            return 1;
        }
        int64_t t0 = ggml_time_ms();
        if (!ofa.init((int)probe.nx(), (int)probe.ny(), n_total_frames, sparams.ofa_grid, sparams.ofa_levels,
                      VPI_OPTICAL_FLOW_QUALITY_HIGH, sparams.ofa_prep_cpu)) {
            LOG_ERR("OFA engine initialisation failed\n");
            return 1;
        }
        LOG_INF("OFA engine ready: %ux%u frames, %dx%d vectors (grid %d, %d levels, high quality), %" PRId64 " ms\n",
                probe.nx(), probe.ny(), ofa.mv_width(), ofa.mv_height(), sparams.ofa_grid, sparams.ofa_levels,
                ggml_time_ms() - t0);
    }
#else
    if (live_flow) {
        LOG_ERR("--live-flow needs a build with NVIDIA VPI (HAVE_VPI)\n");
        return 1;
    }
#endif
    std::vector<optical_flow> live_flows;   // live flow: pair k is the flow from frame k+1 to k+2
    std::vector<rgb_frame>    live_frames;  // live flow: decoded frames, reused for inference
    json live_timing;

    video_motion_profile profile;
    std::vector<selected_frame> selected;
    std::vector<frame_spatial_analysis> spatial_analyses;
    float flow_time_ms = 0.0f;
    float select_time_ms = 0.0f;

    // The model is loaded before the request in every mode (setup, not part of TTFT). With
    // --anchor-first, frame 1 is always selected, so its ViT encode + prefill run on the GPU
    // while the motion analysis below runs on the CPU / OFA.
    const bool anchor_first = sparams.anchor_first && !sparams.baseline;
    auto ctx_holder = std::make_unique<adaptive_vlm_context>(params, sparams.vit_fa_off);
    adaptive_vlm_context & ctx = *ctx_holder;
    std::future<anchor_prefix> prefix_future;

    // Window cache state; with --wc-cost-model the crop-shape encode times are measured here.
    wc_state wc;
    wc.enabled    = sparams.window_cache;
    wc.threshold  = sparams.wc_threshold;
    wc.halo       = sparams.wc_halo;
    wc.rekey_frac = sparams.wc_rekey_frac;
    float calib_ms = 0.0f;
    if (wc.enabled && sparams.wc_cost_model) {
        rgb_frame probe;
        if (!load_rgb(all_frame_paths[0], probe)) {
            LOG_ERR("Cannot read %s\n", all_frame_paths[0].c_str());
            return 1;
        }
        int64_t t0 = ggml_time_ms();
        if (!calibrate_cost_table(ctx.ctx_vision.get(), align_to_windows(probe), sparams.wc_calib_reps, wc.table)) {
            LOG_ERR("Window cache cost-model calibration failed\n");
            return 1;
        }
        calib_ms = (float)(ggml_time_ms() - t0);
        LOG_INF("Window cache cost model: %dx%d window shapes measured in %.0f ms\n", wc.table.nwx, wc.table.nwy, calib_ms);
    }

    NVTX_PUSH("request");
    int64_t t_request_start = ggml_time_ms();
    if (anchor_first) {
        adaptive_vlm_context * ctx_ptr = &ctx;
        wc_state * wc_ptr = &wc;
        const bool align = sparams.window_align;
        std::string anchor_path = sparams.sintel_dir + "/training/clean/" + sparams.video_name + "/frame_0001.png";
        prefix_future = std::async(std::launch::async, [ctx_ptr, anchor_path, prompt, align, wc_ptr]() {
            return eval_anchor_prefix(*ctx_ptr, anchor_path, prompt, align, wc_ptr);
        });
    }

    if (sparams.baseline) {
        // ============ BASELINE MODE ============
        LOG_INF("\n--- Baseline: Uniform Frame Selection ---\n");
        int64_t t0 = ggml_time_ms();

        selected = select_frames_uniform(n_total_frames, sparams.base_frames);
        select_time_ms = (float)(ggml_time_ms() - t0);

        // Still build motion profile for comparison metrics (motion coverage)
        LOG_INF("\n--- Building motion profile for metrics ---\n");
        t0 = ggml_time_ms();
        profile = build_motion_profile(flow_dir, sparams.video_name,
                                        sparams.motion_threshold,
                                        sparams.motion_percentile);
        flow_time_ms = (float)(ggml_time_ms() - t0);

    } else {
        // ============ ADAPTIVE MODE ============
        LOG_INF("Frame range: [%d, %d], base: %d\n", sparams.min_frames, sparams.max_frames, sparams.base_frames);

        // Step 1: Build motion profile
        LOG_INF("\n--- Step 1: Motion Analysis ---\n");
        int64_t t0 = ggml_time_ms();
        NVTX_PUSH("flow");

        if (live_flow) {
#ifdef HAVE_VPI
            const int n_pairs = n_total_frames - 1;
            profile.frame_stats.assign(n_pairs, motion_stats());
            live_flows.assign(n_pairs, optical_flow());
            ofa_flow_timing ft;
            const bool ok = ofa.run(all_frame_paths, !sparams.live_flow_serial, sparams.decode_threads,
                [&](int p, int w, int h, std::vector<float> && u, std::vector<float> && v) {
                    optical_flow & f = live_flows[p];
                    f.width = w; f.height = h; f.u = std::move(u); f.v = std::move(v);
                    profile.frame_stats[p] = compute_motion_stats(f, sparams.motion_threshold, sparams.motion_percentile);
                }, live_frames, ft);
            if (!ok) {
                LOG_ERR("Live optical flow failed\n");
                return 1;
            }
            finalize_motion_profile(profile);
            live_timing = {
                {"schedule", sparams.live_flow_serial ? "serial" : "pipelined"},
                {"decode_threads", sparams.live_flow_serial ? 1 : sparams.decode_threads},
                {"grid", sparams.ofa_grid}, {"levels", sparams.ofa_levels},
                {"prep_backend", sparams.ofa_prep_cpu ? "cpu" : "cuda"},
                {"n_frames", ft.n_frames}, {"n_pairs", ft.n_pairs},
                {"wall_ms", ft.wall_ms}, {"decode_wall_ms", ft.decode_wall_ms}, {"decode_sum_ms", ft.decode_sum_ms},
                {"prep_ms", ft.prep_ms}, {"ofa_ms", ft.ofa_ms}, {"readback_ms", ft.readback_ms},
                {"stats_ms", ft.consumer_ms},
            };
            LOG_INF("Live flow (%s): %d pairs in %.1f ms (decode sum %.1f ms, wall %.1f ms; prep %.1f, OFA %.1f, "
                    "readback %.1f, stats %.1f ms)\n", sparams.live_flow_serial ? "serial" : "pipelined", ft.n_pairs,
                    ft.wall_ms, ft.decode_sum_ms, ft.decode_wall_ms, ft.prep_ms, ft.ofa_ms, ft.readback_ms, ft.consumer_ms);
#endif
        } else {
            profile = build_motion_profile(flow_dir, sparams.video_name,
                                            sparams.motion_threshold,
                                            sparams.motion_percentile);
        }
        flow_time_ms = (float)(ggml_time_ms() - t0);
        LOG_INF("Flow analysis completed in %.1f ms\n", flow_time_ms);

        if (profile.frame_stats.empty()) {
            LOG_ERR("No motion data available. Exiting.\n");
            return 1;
        }

        NVTX_POP();   // flow

        // Step 2: Select frames adaptively
        LOG_INF("\n--- Step 2: Adaptive Frame Selection ---\n");
        t0 = ggml_time_ms();
        NVTX_PUSH("select");

        selected = select_frames(profile,
                                  sparams.min_frames,
                                  sparams.max_frames,
                                  sparams.base_frames,
                                  profile.global_complexity,
                                  sparams.selection_percentile,
                                  anchor_first);
        select_time_ms = (float)(ggml_time_ms() - t0);

        if (selected.empty()) {
            LOG_ERR("No frames selected. Exiting.\n");
            return 1;
        }

        // Step 3: Post-selection complexity filtering
        // Remove frames with very low per-frame complexity (near-static frames
        // that slipped through probabilistic sampling)
        {
            const float complexity_floor = 0.05f;
            int before_count = (int)selected.size();
            std::vector<selected_frame> filtered;
            for (const auto & sf : selected) {
                if (sf.complexity_score >= complexity_floor || (int)filtered.size() < sparams.min_frames ||
                    (anchor_first && sf.frame_index == 1)) {
                    filtered.push_back(sf);
                } else {
                    LOG_INF("  Dropping frame %d (complexity %.4f < %.4f)\n",
                            sf.frame_index, sf.complexity_score, complexity_floor);
                }
            }
            // Always keep at least min_frames
            if ((int)filtered.size() < sparams.min_frames) {
                filtered = selected;
                if ((int)filtered.size() > sparams.min_frames) {
                    // Sort by complexity descending, keep top min_frames
                    std::sort(filtered.begin(), filtered.end(), [](const selected_frame & a, const selected_frame & b) {
                        return a.complexity_score > b.complexity_score;
                    });
                    filtered.resize(sparams.min_frames);
                    // Re-sort in temporal order
                    std::sort(filtered.begin(), filtered.end(), [](const selected_frame & a, const selected_frame & b) {
                        return a.frame_index < b.frame_index;
                    });
                }
            }
            selected = filtered;
            LOG_INF("\n--- Step 3: Post-selection filtering: %d -> %zu frames ---\n",
                    before_count, selected.size());
        }

        // Step 4: Spatial patch analysis
        LOG_INF("\n--- Step 4: Spatial Patch Analysis ---\n");
        for (const auto & sf : selected) {
            int flow_idx = sf.frame_index - 1;
            if (flow_idx < 0 || flow_idx >= (int)profile.frame_stats.size()) continue;

            char buf[64];
            snprintf(buf, sizeof(buf), "/frame_%04d.flo", sf.frame_index);
            std::string flo_path = flow_dir + "/" + sparams.video_name + buf;

            optical_flow flow;
            if (live_flow) {
                flow = live_flows[flow_idx];
            }
            if (live_flow ? !flow.u.empty() : read_flo_file(flo_path, flow)) {
                auto sa = analyze_spatial_patches(flow, sf.frame_index,
                                                  sparams.patch_grid,
                                                  sparams.spatial_threshold,
                                                  sparams.motion_threshold);
                spatial_analyses.push_back(sa);

                LOG_INF("  Frame %d: %d/%d patches active (%.0f%% savings)\n",
                        sf.frame_index, sa.n_selected_patches, sa.n_total_patches,
                        sa.spatial_savings * 100.0f);
            }
        }

        // Step 5: Spatial-aware frame skipping
        // Drop frames where <25% of spatial patches have motion (mostly static)
        {
            const float min_active_ratio = 0.25f;
            int before_count = (int)selected.size();
            std::vector<selected_frame> spatially_filtered;
            std::vector<frame_spatial_analysis> kept_spatial;

            for (size_t i = 0; i < selected.size(); i++) {
                bool keep = true;
                // Find matching spatial analysis (the anchor frame is never skipped)
                for (const auto & sa : spatial_analyses) {
                    if (anchor_first && selected[i].frame_index == 1) break;
                    if (sa.frame_index == selected[i].frame_index) {
                        float active_ratio = (float)sa.n_selected_patches / sa.n_total_patches;
                        if (active_ratio < min_active_ratio && (int)spatially_filtered.size() >= sparams.min_frames) {
                            LOG_INF("  Spatial skip: frame %d (%d/%d patches = %.0f%% active < %.0f%% threshold)\n",
                                    selected[i].frame_index, sa.n_selected_patches, sa.n_total_patches,
                                    active_ratio * 100.0f, min_active_ratio * 100.0f);
                            keep = false;
                        }
                        break;
                    }
                }
                if (keep) {
                    spatially_filtered.push_back(selected[i]);
                }
            }

            // Always keep at least min_frames
            if ((int)spatially_filtered.size() < sparams.min_frames) {
                spatially_filtered = selected; // revert if too aggressive
            }

            // Rebuild spatial_analyses for kept frames only
            kept_spatial.clear();
            for (const auto & sa : spatial_analyses) {
                for (const auto & sf : spatially_filtered) {
                    if (sa.frame_index == sf.frame_index) {
                        kept_spatial.push_back(sa);
                        break;
                    }
                }
            }

            selected = spatially_filtered;
            spatial_analyses = kept_spatial;
            LOG_INF("\n--- Step 5: Spatial-aware filtering: %d -> %zu frames ---\n",
                    before_count, selected.size());
        }
        NVTX_POP();   // select
    }

    const int64_t t_analysis_end = ggml_time_ms();
    float analysis_time_ms = (float)(t_analysis_end - t_request_start);

    if (anchor_first && (selected.empty() || selected[0].frame_index != 1)) {
        LOG_ERR("Anchor frame 1 missing from the selection. Exiting.\n");
        return 1;
    }

    // Build frame paths for inference
    std::vector<std::string> frame_paths;
    for (const auto & sf : selected) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
        frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }

    // Run inference
    LOG_INF("\n--- Model Inference ---\n");

    anchor_prefix prefix;
    if (anchor_first) {
        prefix = prefix_future.get();
        if (!prefix.ok) {
            LOG_ERR("Anchor prefix evaluation failed. Exiting.\n");
            return 1;
        }
    }

    LOG_INF("Running inference with %zu frames [%s]...\n", frame_paths.size(), mode_str);
    LOG_INF("Prompt: %s\n", prompt.c_str());

    int64_t t_infer_start = ggml_time_ms();
    std::vector<const rgb_frame *> preloaded;
    if (live_flow) {
        for (const auto & sf : selected) {
            const int k = sf.frame_index - 1;
            preloaded.push_back(k >= 0 && k < (int)live_frames.size() ? &live_frames[k] : nullptr);
        }
    }
    auto inf_result = run_inference(ctx, frame_paths, prompt, n_predict, anchor_first ? &prefix : nullptr,
                                    live_flow ? &preloaded : nullptr, sparams.window_align, &wc);
    NVTX_POP();   // request (first token ready; decode was its own range)

    // Time to first token, excluding model load. The baseline's flow analysis is
    // reporting-only, so it is not charged; the adaptive analysis is.
    float ttft_ms;
    if (anchor_first) {
        ttft_ms = (float)(inf_result.t_encode_end_ms - t_request_start);
    } else {
        ttft_ms = (float)(inf_result.t_encode_end_ms - t_infer_start) + (sparams.baseline ? 0.0f : analysis_time_ms);
    }

    // Compute metrics
    LOG_INF("\n--- Metrics ---\n");
    double motion_cov = compute_motion_coverage(profile, selected);
    double entropy = compute_shannon_entropy(inf_result.generated_text);
    float total_time_ms = (float)(ggml_time_ms() - t_total_start);

    double temporal_reduction = 1.0 - (double)selected.size() / n_total_frames;

    LOG_INF("Mode:               %s\n", mode_str);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Shannon entropy:    %.2f bits\n", entropy);
    LOG_INF("Temporal reduction: %.1f%%\n", temporal_reduction * 100.0);
    LOG_INF("Tokens generated:   %d\n", inf_result.n_tokens_generated);
    LOG_INF("TTFT (no load):     %.1f ms\n", ttft_ms);
    LOG_INF("Total time:         %.1f ms\n", total_time_ms);

    // Write JSON output
    std::string json_path = sparams.output_json.empty()
        ? ("results_" + sparams.video_name + (sparams.baseline ? "_baseline" : "_adaptive") + ".json")
        : sparams.output_json;

    // TTFT breakdown, all relative to the request start where a timeline is meaningful.
    auto bd_json = [](const encode_breakdown & b) {
        return json{{"vit_ms", b.vit_ms}, {"prefill_img_ms", b.prefill_img_ms},
                    {"prefill_txt_ms", b.prefill_txt_ms}, {"n_images", b.n_images},
                    {"tokens_encoded", b.tokens_encoded}, {"n_full", b.n_full}, {"n_crop", b.n_crop},
                    {"n_skip", b.n_skip}, {"wc_diff_ms", b.wc_diff_ms}, {"wc_prep_ms", b.wc_prep_ms}};
    };
    json breakdown = {
        {"analysis_ms", analysis_time_ms},          // request start -> selection + spatial filtering done
        {"flow_ms", flow_time_ms},                  // motion profile (file read or live flow + stats)
        {"rest_image_load_ms", inf_result.image_load_ms},
        {"rest_tokenize_ms", inf_result.tokenize_ms},
        {"rest", bd_json(inf_result.rest)},
        {"vit_flash_attn", sparams.vit_fa_off ? "off" : "on"},
    };
    if (wc.enabled) {
        json frames = json::array();
        for (const auto & f : wc.frames) {
            frames.push_back({{"action", f.action}, {"n_dirty", f.n_dirty}, {"crop", {f.crop_w, f.crop_h}},
                              {"tokens_encoded", f.tokens_encoded}, {"diff_ms", f.diff_ms}, {"prep_ms", f.prep_ms},
                              {"encode_ms", f.encode_ms}});
        }
        breakdown["window_cache"] = {{"frames", frames}, {"calibration_ms", calib_ms}};
    }
    if (!live_timing.is_null()) breakdown["live_flow"] = live_timing;
    if (anchor_first) {
        breakdown["anchor"] = bd_json(prefix.bd);
        breakdown["anchor"]["load_ms"]  = prefix.load_ms;
        breakdown["anchor"]["total_ms"] = prefix.eval_ms;
        breakdown["anchor"]["done_at_ms"] = (double)(prefix.t_done_ms - t_request_start);
        breakdown["wait_for_anchor_ms"] = (double)(t_infer_start - t_analysis_end);
    }

    write_results_json(json_path, sparams, n_total_frames, profile, selected, spatial_analyses,
                       inf_result, motion_cov, entropy, flow_time_ms, select_time_ms,
                       ttft_ms, total_time_ms, breakdown);

    // Print performance summary
    LOG_INF("\n=== Summary [%s] ===\n", mode_str);
    LOG_INF("Video:              %s\n", sparams.video_name.c_str());
    LOG_INF("Frames:             %zu / %d selected\n", selected.size(), n_total_frames);
    LOG_INF("Global complexity:  %.3f\n", profile.global_complexity);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Temporal reduction: %.1f%%\n", temporal_reduction * 100.0);
    LOG_INF("Output:             %s\n", json_path.c_str());

    llama_perf_context_print(ctx.lctx);

    return 0;
}
