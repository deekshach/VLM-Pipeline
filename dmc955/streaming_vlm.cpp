// Streaming VLM Benchmark
//
// Simulates 4fps video frame arrival from MPI-Sintel and makes online frame
// selection decisions using cumulative motion thresholding. Supports two modes:
//   --baseline:   uniform frame selection (no motion analysis, no cache, no pruning)
//   (default):    online cumulative-motion selection + embedding cache + pruning
//
// The online selector is causally correct: each frame decision only uses data
// from the current and preceding frames. A two-pass architecture is required
// because mtmd_tokenize() needs to know the image count upfront.
//
// Output: JSON with per-frame streaming decisions, timing, and quality metrics.

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

#ifdef HAVE_CUDA_KV_REUSE
#include "kv_cache_rerotate.cuh"
#include <cuda_runtime.h>
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
#include <unordered_map>
#include <map>
#include <set>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Parameters
// ============================================================================

struct streaming_params {
    // Inherited from sintel_params
    std::string sintel_dir    = "/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel";
    std::string video_name    = "alley_1";
    int         min_frames    = 3;
    int         max_frames    = 24;
    int         base_frames   = 16;
    float       motion_threshold  = 2.0f;
    float       motion_percentile = 0.95f;
    float       selection_percentile = 70.0f;
    int         patch_grid    = 4;
    float       spatial_threshold = 0.05f;
    std::string output_json;
    bool        baseline      = false;
    int         crop_padding  = 16;
    int         min_crop_dim  = 112;
    float       patch_reuse_motion = 0.0f;

    // LLM-side progressive visual token pruning
    int         prune_layer          = 2;
    int         attn_sample_interval = 5;
    int         first_prune_step     = 5;
    int         second_prune_step    = 15;
    float       patch_keep_ratio     = 0.5f;
    float       token_keep_ratio     = 0.7f;
    float       motion_weight        = 0.3f;
    bool        enable_pruning       = false;

    // Prefill-stage motion pruning
    bool        enable_prefill_pruning = false;
    float       prefill_keep_ratio     = 0.6f;

    // Pre-LLM visual token pruning
    bool        enable_pre_llm_pruning  = false;
    float       pre_llm_keep_ratio      = 0.5f;
    float       pre_llm_weight_motion   = 0.3f;
    float       pre_llm_weight_norm     = 0.3f;
    float       pre_llm_weight_attn     = 0.4f;
    std::string pre_llm_strategy        = "combined"; // combined, random, motion-only, norm-only, attn-only, motion-norm

    // Post-prefill KV eviction
    bool        enable_postprefill_eviction = false;
    float       postprefill_keep_ratio      = 0.5f;

    // Attention correlation analysis
    bool        analyze_attention_correlation = false;

    // KV cache reuse (CUDA): copy K/V from keyframe with temporal RoPE re-rotation
    bool        enable_kv_cache_reuse = false;

    // Streaming-specific parameters
    float       fps                    = 4.0f;
    float       cumulative_threshold   = 15.0f;
    int         max_selected_frames    = 10;
    int         baseline_uniform_count = 16;
};

static std::vector<char *> filter_custom_args(int argc, char ** argv, streaming_params & sparams) {
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
        } else if (arg == "--crop-padding" && i + 1 < argc) {
            sparams.crop_padding = std::stoi(argv[++i]);
        } else if (arg == "--min-crop-dim" && i + 1 < argc) {
            sparams.min_crop_dim = std::stoi(argv[++i]);
        } else if (arg == "--patch-reuse-motion" && i + 1 < argc) {
            sparams.patch_reuse_motion = std::stof(argv[++i]);
        } else if (arg == "--enable-pruning") {
            sparams.enable_pruning = true;
        } else if (arg == "--prune-layer" && i + 1 < argc) {
            sparams.prune_layer = std::stoi(argv[++i]);
        } else if (arg == "--attn-sample-interval" && i + 1 < argc) {
            sparams.attn_sample_interval = std::stoi(argv[++i]);
        } else if (arg == "--first-prune-step" && i + 1 < argc) {
            sparams.first_prune_step = std::stoi(argv[++i]);
        } else if (arg == "--second-prune-step" && i + 1 < argc) {
            sparams.second_prune_step = std::stoi(argv[++i]);
        } else if (arg == "--patch-keep-ratio" && i + 1 < argc) {
            sparams.patch_keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--token-keep-ratio" && i + 1 < argc) {
            sparams.token_keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--motion-weight" && i + 1 < argc) {
            sparams.motion_weight = std::stof(argv[++i]);
        } else if (arg == "--enable-prefill-pruning") {
            sparams.enable_prefill_pruning = true;
        } else if (arg == "--prefill-keep-ratio" && i + 1 < argc) {
            sparams.prefill_keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--enable-pre-llm-pruning") {
            sparams.enable_pre_llm_pruning = true;
        } else if (arg == "--pre-llm-keep-ratio" && i + 1 < argc) {
            sparams.pre_llm_keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--pre-llm-weight-motion" && i + 1 < argc) {
            sparams.pre_llm_weight_motion = std::stof(argv[++i]);
        } else if (arg == "--pre-llm-weight-norm" && i + 1 < argc) {
            sparams.pre_llm_weight_norm = std::stof(argv[++i]);
        } else if (arg == "--pre-llm-weight-attn" && i + 1 < argc) {
            sparams.pre_llm_weight_attn = std::stof(argv[++i]);
        } else if (arg == "--pre-llm-strategy" && i + 1 < argc) {
            sparams.pre_llm_strategy = argv[++i];
        } else if (arg == "--enable-postprefill-eviction") {
            sparams.enable_postprefill_eviction = true;
        } else if (arg == "--postprefill-keep-ratio" && i + 1 < argc) {
            sparams.postprefill_keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--analyze-attention-correlation") {
            sparams.analyze_attention_correlation = true;
        } else if (arg == "--enable-kv-cache-reuse") {
            sparams.enable_kv_cache_reuse = true;
        } else if (arg == "--fps" && i + 1 < argc) {
            sparams.fps = std::stof(argv[++i]);
        } else if (arg == "--cumulative-threshold" && i + 1 < argc) {
            sparams.cumulative_threshold = std::stof(argv[++i]);
        } else if (arg == "--max-selected-frames" && i + 1 < argc) {
            sparams.max_selected_frames = std::stoi(argv[++i]);
        } else if (arg == "--baseline-uniform-count" && i + 1 < argc) {
            sparams.baseline_uniform_count = std::stoi(argv[++i]);
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nStreaming VLM Benchmark\n\n");
    printf("Usage: %s [options]\n\n", argv[0]);
    printf("Required llama.cpp options:\n");
    printf("  -m <model>          Path to Qwen2.5-VL GGUF model\n");
    printf("  --mmproj <mmproj>   Path to Qwen2.5-VL mmproj GGUF\n\n");
    printf("Streaming-specific options:\n");
    printf("  --fps <f>                    Simulated frame rate (default: 4.0)\n");
    printf("  --cumulative-threshold <f>   Motion accumulation threshold (default: 15.0)\n");
    printf("  --max-selected-frames <n>    Max frames to select online (default: 16)\n");
    printf("  --baseline-uniform-count <n> Frame count for baseline mode (default: 12)\n");
    printf("  --baseline                   Use uniform frame sampling (baseline mode)\n\n");
    printf("Sintel-specific options:\n");
    printf("  --sintel-dir <dir>  MPI-Sintel dataset directory\n");
    printf("  --video <name>      Video name (e.g., alley_1)\n");
    printf("  --patch-grid <n>    Spatial patch grid size NxN (default: 4)\n");
    printf("  --spatial-threshold Patch motion ratio threshold (default: 0.05)\n");
    printf("  --crop-padding <n>  Pixel padding around crop region (default: 16)\n");
    printf("  --min-crop-dim <n>  Minimum crop dimension in pixels (default: 112)\n");
    printf("  --patch-reuse-motion <f> Mean motion threshold for patch re-encode (default: 0)\n");
    printf("  --output-json <f>   Output JSON file path\n\n");
    printf("LLM-side progressive visual token pruning:\n");
    printf("  --enable-pruning           Enable LLM-side visual token pruning\n");
    printf("  --prune-layer <n>          Transformer layer for attention capture (default: 2)\n");
    printf("  --attn-sample-interval <n> Sample attention every N gen steps (default: 5)\n");
    printf("  --first-prune-step <n>     Generation step for patch-level pruning (default: 5)\n");
    printf("  --second-prune-step <n>    Generation step for token-level pruning (default: 15)\n");
    printf("  --patch-keep-ratio <f>     Fraction of patches to keep in round 1 (default: 0.5)\n");
    printf("  --token-keep-ratio <f>     Fraction of tokens to keep in round 2 (default: 0.7)\n");
    printf("  --motion-weight <f>        Weight of motion vs attention scoring (default: 0.3)\n\n");
    printf("Prefill-stage motion pruning:\n");
    printf("  --enable-prefill-pruning   Prune low-motion tokens between frame decodes\n");
    printf("  --prefill-keep-ratio <f>   Fraction of tokens to keep per non-keyframe (default: 0.6)\n\n");
    printf("Pre-LLM visual token pruning:\n");
    printf("  --enable-pre-llm-pruning          Enable pre-LLM visual token pruning\n");
    printf("  --pre-llm-keep-ratio <f>          Fraction of tokens to keep (default: 0.5)\n");
    printf("  --pre-llm-weight-motion <f>       Weight for motion signal (default: 0.3)\n");
    printf("  --pre-llm-weight-norm <f>         Weight for embedding norm signal (default: 0.3)\n");
    printf("  --pre-llm-weight-attn <f>         Weight for ViT attention signal (default: 0.4)\n");
    printf("  --pre-llm-strategy <s>           Strategy: combined, random, motion-only, norm-only, attn-only, motion-norm (default: combined)\n\n");
    printf("Standard llama.cpp options (-p, -n, -ngl, -c, etc.) are also supported.\n");
}

// ============================================================================
// Section 2: Optical flow (.flo) file reader
// ============================================================================

struct optical_flow {
    int width  = 0;
    int height = 0;
    std::vector<float> u;
    std::vector<float> v;
};

static bool read_flo_file(const std::string & path, optical_flow & flow) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERR("Cannot open .flo file: %s\n", path.c_str());
        return false;
    }

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

    std::vector<float> sorted_mag = mag;
    std::sort(sorted_mag.begin(), sorted_mag.end());
    stats.max_magnitude = sorted_mag.back();

    int p_idx = std::min((int)(percentile * n), n - 1);
    stats.p95_magnitude = sorted_mag[p_idx];

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

static video_motion_profile build_motion_profile(const std::string & flow_dir,
                                                  const std::string & video_name,
                                                  float motion_threshold,
                                                  float percentile) {
    video_motion_profile profile;

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

    // Per-frame complexity using min-max normalization
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

    // Global complexity
    {
        std::vector<float> p95_values;
        for (const auto & s : profile.frame_stats) {
            p95_values.push_back(s.p95_magnitude);
        }
        const int nf = (int)p95_values.size();

        double p95_mean = 0.0;
        for (float v : p95_values) p95_mean += v;
        p95_mean /= nf;
        float mn = std::min((float)(p95_mean / 100.0), 1.0f);

        double p95_var = 0.0;
        for (float v : p95_values) {
            double d = v - p95_mean;
            p95_var += d * d;
        }
        p95_var /= nf;
        float vn = std::min((float)(p95_var / 5000.0), 1.0f);

        float p95_min = *std::min_element(p95_values.begin(), p95_values.end());
        float p95_max = *std::max_element(p95_values.begin(), p95_values.end());
        float p95_range = p95_max - p95_min;
        float rn = std::min(p95_range / 150.0f, 1.0f);

        std::vector<float> sorted_p95 = p95_values;
        std::sort(sorted_p95.begin(), sorted_p95.end());
        int p75_idx = std::min((int)(0.75f * nf), nf - 1);
        float p75_threshold = sorted_p95[p75_idx];
        int above_count = 0;
        for (float v : p95_values) {
            if (v > p75_threshold) above_count++;
        }
        float high_motion_ratio = (float)above_count / nf;

        profile.global_complexity = 0.4f * mn + 0.3f * vn + 0.2f * rn + 0.1f * high_motion_ratio;
        profile.global_complexity = std::min(profile.global_complexity, 1.0f);
    }

    LOG_INF("Motion profile: %zu frames, global complexity = %.3f\n",
            profile.frame_stats.size(), profile.global_complexity);

    return profile;
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
// Section 4: Frame selection (uniform for baseline)
// ============================================================================

struct selected_frame {
    int   frame_index;
    float motion_weight;
    float complexity_score;
};

static std::vector<selected_frame> select_frames_uniform(int n_total_frames, int n_select) {
    n_select = std::min(n_select, n_total_frames);
    std::vector<selected_frame> result;

    if (n_select <= 0 || n_total_frames <= 0) return result;

    for (int k = 0; k < n_select; k++) {
        selected_frame sf;
        sf.frame_index = 1 + (int)std::round((double)k * (n_total_frames - 1) / std::max(n_select - 1, 1));
        sf.motion_weight = 1.0f;
        sf.complexity_score = 0.0f;
        result.push_back(sf);
    }

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

            pi.selected = (pi.motion_ratio > spatial_threshold) ||
                          (pi.mean_motion > motion_threshold) ||
                          (pi.max_motion > motion_threshold * 2.0f);

            result.patches.push_back(pi);
        }
    }

    result.n_selected_patches = 0;
    for (const auto & p : result.patches) {
        if (p.selected) result.n_selected_patches++;
    }

    int min_patches = std::max(1, result.n_total_patches / 4);
    if (result.n_selected_patches < min_patches) {
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
// Section 5b: Spatial crop computation
// ============================================================================

struct frame_crop_info {
    int  frame_index;
    int  orig_w, orig_h;
    int  crop_x, crop_y, crop_w, crop_h;
    bool was_cropped;
    int  estimated_tokens_full;
    int  estimated_tokens_crop;
};

// ============================================================================
// Section 5b-1: Embedding cache for encoder-level spatial redundancy
// ============================================================================

struct embedding_cache {
    int nx_full  = 0;
    int ny_full  = 0;
    int n_embd   = 0;
    std::vector<float> embeddings;
    bool valid = false;

    void allocate(int nx, int ny, int embd) {
        nx_full = nx;
        ny_full = ny;
        n_embd  = embd;
        embeddings.resize(nx * ny * embd, 0.0f);
        valid = false;
    }

    float * at(int gy, int gx) {
        return embeddings.data() + (gy * nx_full + gx) * n_embd;
    }

    void store_full(const float * src) {
        memcpy(embeddings.data(), src, embeddings.size() * sizeof(float));
        valid = true;
    }

    void selective_update(const float * new_embd,
                          const std::vector<bool> & patch_active,
                          int & n_updated, int & n_kept) {
        n_updated = 0;
        n_kept = 0;
        int n_total = nx_full * ny_full;
        for (int i = 0; i < n_total; i++) {
            if (patch_active[i]) {
                memcpy(embeddings.data() + i * n_embd,
                       new_embd + i * n_embd,
                       n_embd * sizeof(float));
                n_updated++;
            } else {
                n_kept++;
            }
        }
        valid = true;
    }
};

// ============================================================================
// Section 5b-1b: KV cache snapshot for CUDA KV reuse
// ============================================================================

struct kv_cache_snapshot {
    bool valid = false;
    llama_pos temporal_pos;           // the temporal position of the keyframe
    int n_visual_tokens;              // total visual tokens in keyframe
    std::vector<int32_t> cell_indices; // KV cell indices for each visual token
    int nx, ny;                       // grid dimensions

    void capture(llama_pos temp_pos, int grid_nx, int grid_ny,
                 const int32_t * cells, int n_tokens) {
        temporal_pos = temp_pos;
        nx = grid_nx;
        ny = grid_ny;
        n_visual_tokens = n_tokens;
        cell_indices.assign(cells, cells + n_tokens);
        valid = true;
    }

    void invalidate() {
        valid = false;
        cell_indices.clear();
    }
};

// ============================================================================
// Section 5b-2: Per-patch motion mapping for ViT token grid
// ============================================================================

static std::vector<bool> build_patch_active_mask(
    const frame_spatial_analysis & sa,
    int grid_size,
    int vit_nx, int vit_ny,
    float patch_reuse_motion = 0.0f) {

    int n_tokens = vit_nx * vit_ny;
    std::vector<bool> active(n_tokens, true);

    std::vector<bool> sa_active(grid_size * grid_size, false);
    for (const auto & p : sa.patches) {
        int idx = p.grid_y * grid_size + p.grid_x;
        if (patch_reuse_motion > 0.0f) {
            sa_active[idx] = (p.mean_motion >= patch_reuse_motion);
        } else {
            sa_active[idx] = p.selected;
        }
    }

    for (int ty = 0; ty < vit_ny; ty++) {
        for (int tx = 0; tx < vit_nx; tx++) {
            int sa_gx = tx * grid_size / vit_nx;
            int sa_gy = ty * grid_size / vit_ny;
            sa_gx = std::min(sa_gx, grid_size - 1);
            sa_gy = std::min(sa_gy, grid_size - 1);
            active[ty * vit_nx + tx] = sa_active[sa_gy * grid_size + sa_gx];
        }
    }

    return active;
}

// ============================================================================
// Section 5b-3: Visual token map for LLM-side pruning
// ============================================================================

struct visual_token_map {
    struct token_info {
        int frame_idx;
        int patch_row;
        int patch_col;
        llama_pos kv_pos;
        llama_pos ext_x;
        llama_pos ext_y;
        int kv_cell_idx;       // actual KV cache cell index (for attention correlation)
        float motion_score;
        float attn_score;
        int attn_samples;
        bool pruned;
        float pre_llm_score;   // pre-LLM combined score (for correlation analysis)
        bool would_prune;      // true if this token would be pruned by pre-LLM scorer
        float prefill_attn_score; // attention received from prompt queries during prefill
    };

    std::vector<token_info> tokens;
    std::map<std::string, std::vector<int>> patch_to_token_indices;

    struct frame_range {
        llama_pos start_pos;
        llama_pos end_pos;
        int nx, ny;
        int token_start_idx;
        int token_count;
    };
    std::vector<frame_range> frame_ranges;

    void add_token(llama_pos kv_pos, llama_pos ext_x, llama_pos ext_y,
                   int frame_idx, int row, int col, float motion,
                   int sa_grid_row, int sa_grid_col, int cell_idx = -1) {
        token_info ti;
        ti.frame_idx    = frame_idx;
        ti.patch_row    = row;
        ti.patch_col    = col;
        ti.kv_pos       = kv_pos;
        ti.ext_x        = ext_x;
        ti.ext_y        = ext_y;
        ti.kv_cell_idx  = cell_idx;
        ti.motion_score = motion;
        ti.attn_score   = 0.0f;
        ti.attn_samples = 0;
        ti.pruned       = false;
        ti.pre_llm_score = 0.0f;
        ti.would_prune   = false;
        ti.prefill_attn_score = 0.0f;

        int idx = (int)tokens.size();
        tokens.push_back(ti);

        std::string key = std::to_string(frame_idx) + "_" +
                          std::to_string(sa_grid_row) + "_" + std::to_string(sa_grid_col);
        patch_to_token_indices[key].push_back(idx);
    }

    int count_active() const {
        int count = 0;
        for (const auto & t : tokens) {
            if (!t.pruned) count++;
        }
        return count;
    }
};

// ============================================================================
// Section 5b-4: Attention capture state for cb_eval callback
// ============================================================================

struct attention_capture_state {
    int target_layer;
    int sample_interval;
    int current_gen_step;
    bool should_capture;

    visual_token_map * token_map;

    llama_pos visual_pos_min;
    llama_pos visual_pos_max;
    llama_pos text_pos_start;

    bool initialized;
    bool disabled;
    bool prefill_mode;  // true during last text chunk processing (prompt attending to visual tokens)
};

static bool attn_softmax_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (attention_capture_state *)user_data;

    if (!state || !state->initialized || state->disabled || !state->should_capture) {
        return !ask;
    }

    // Build exact target name for this layer
    std::string target = "kq_soft_max-" + std::to_string(state->target_layer);

    if (ask) {
        std::string name(t->name);
        // Exact match required — avoid "kq_soft_max-2" matching "kq_soft_max-20"
        if (name == target) {
            return true;
        }
        return false;
    }

    // Verify this is still the right tensor in the data callback
    {
        std::string name(t->name);
        if (name != target) return true;
    }

    if (!state->token_map) return true;

    // Tensor layout: [n_kv, n_q, n_heads, 1]
    int64_t n_kv    = t->ne[0];
    int64_t n_q     = t->ne[1];
    int64_t n_heads = t->ne[2];

    if (t->type != GGML_TYPE_F32) return true;
    if (n_kv <= 0 || n_heads <= 0) return true;

    // Prefill path: n_q > 1 (prompt tokens attending to visual tokens in KV cache)
    // Sum attention from all prompt queries over all heads for each visual token
    if (state->prefill_mode && n_q > 1) {
        size_t n_bytes = ggml_nbytes(t);
        std::vector<float> attn_data(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, attn_data.data(), 0, n_bytes);

        const float * data_ptr = attn_data.data();

        for (auto & tok : state->token_map->tokens) {
            if (tok.pruned) continue;

            int cell = tok.kv_cell_idx;
            if (cell < 0 || cell >= (int)n_kv) continue;

            float total_attn = 0.0f;
            for (int h = 0; h < (int)n_heads; h++) {
                for (int q = 0; q < (int)n_q; q++) {
                    // Layout: [n_kv, n_q, n_heads, 1] contiguous
                    // offset = (h * n_q + q) * n_kv + cell
                    total_attn += data_ptr[(h * n_q + q) * n_kv + cell];
                }
            }
            // Average over heads (but sum over queries — total attention budget from prompt)
            tok.prefill_attn_score += total_attn / (float)n_heads;
        }

        return true;
    }

    // During autoregressive decode, n_q == 1 (single new token)
    // During prefill, n_q > 1 — skip those (too much data, not meaningful for per-token decode attention)
    if (n_q != 1) return true;

    // Copy tensor data from GPU to CPU using ggml_backend_tensor_get
    // (t->data may point to GPU memory and is NOT directly accessible from CPU)
    size_t n_bytes = ggml_nbytes(t);
    std::vector<float> attn_data(n_bytes / sizeof(float));
    ggml_backend_tensor_get(t, attn_data.data(), 0, n_bytes);

    const float * data_ptr = attn_data.data();

    for (auto & tok : state->token_map->tokens) {
        if (tok.pruned) continue;

        // Use KV cell index (not M-RoPE position) to index into attention weights
        int cell = tok.kv_cell_idx;
        if (cell < 0 || cell >= (int)n_kv) continue;

        float total_attn = 0.0f;
        for (int h = 0; h < (int)n_heads; h++) {
            // Layout: [n_kv, n_q=1, n_heads] contiguous -> offset = h * n_kv + cell
            total_attn += data_ptr[h * n_kv + cell];
        }
        tok.attn_score += total_attn / (float)n_heads;
        tok.attn_samples++;
    }

    return true;
}

// ============================================================================
// Section 5b-4b: Flash-compatible Qcur capture for post-prefill eviction
// ============================================================================

struct qcur_capture_state {
    int target_layer = -1;
    bool should_capture = false;     // set true for last text chunk
    std::vector<float> q_data;       // captured Q: [n_embd_head, n_head, n_text_tokens]
    int n_text_tokens = 0;
    int n_head = 0;
    int n_embd_head = 0;
    bool captured = false;           // true after successful capture
};

static bool qcur_capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (qcur_capture_state *)user_data;
    if (!state || !state->should_capture) return !ask;

    std::string target = "Qcur-" + std::to_string(state->target_layer);
    std::string name(t->name);

    if (ask) {
        // Accept ALL occurrences of Qcur at our target layer.
        // The last write wins (post-RoPE), which is the one we want.
        return (name == target);
    }

    if (name != target) return true;

    // Copy Qcur data from GPU: shape [n_embd_head, n_head, n_tokens]
    state->n_embd_head   = (int)t->ne[0];  // 128
    state->n_head        = (int)t->ne[1];  // 28
    state->n_text_tokens = (int)t->ne[2];
    state->q_data.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, state->q_data.data(), 0, ggml_nbytes(t));
    state->captured = true;
    return true;
}

// Compute attention scores on CPU using captured Qcur and K cache tensor.
// Writes per-visual-token prefill_attn_score into vis_token_map.
static void compute_cpu_attention_scores(
        const qcur_capture_state & qstate,
        visual_token_map & vis_token_map,
        llama_memory_t mem,
        int target_layer,
        int n_kv_cells_used,
        int n_head_kv) {
    if (!qstate.captured || qstate.q_data.empty() || qstate.n_text_tokens == 0) {
        LOG_WRN("Qcur capture empty — skipping CPU attention computation\n");
        return;
    }

    const int d_k     = qstate.n_embd_head;  // 128
    const int n_head  = qstate.n_head;        // 28 (Q heads)
    const int n_q     = qstate.n_text_tokens;
    const int n_kv    = n_kv_cells_used;
    const int gqa_ratio = n_head / n_head_kv; // 28 / 4 = 7

    LOG_INF("CPU Q@K^T: n_q=%d, n_kv=%d, n_head=%d, n_head_kv=%d, d_k=%d\n",
            n_q, n_kv, n_head, n_head_kv, d_k);

    // Read K cache from GPU
    const ggml_tensor * k_tensor = llama_memory_k_tensor(mem, target_layer);
    if (!k_tensor) {
        LOG_ERR("Failed to get K tensor for layer %d\n", target_layer);
        return;
    }

    // K tensor shape: [n_embd_k_gqa, kv_size, 1] where n_embd_k_gqa = d_k * n_head_kv
    // We only need the first n_kv rows
    size_t row_bytes = ggml_row_size(k_tensor->type, k_tensor->ne[0]);
    std::vector<uint8_t> k_raw(n_kv * row_bytes);
    ggml_backend_tensor_get(k_tensor, k_raw.data(), 0, k_raw.size());

    // Convert K to f32: [n_kv, n_head_kv, d_k]
    int n_embd_k_gqa = n_head_kv * d_k;
    std::vector<float> k_f32(n_kv * n_embd_k_gqa);
    if (k_tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t * k_f16 = (ggml_fp16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) {
            k_f32[i] = ggml_fp16_to_fp32(k_f16[i]);
        }
    } else if (k_tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t * k_bf16 = (ggml_bf16_t *)k_raw.data();
        for (size_t i = 0; i < k_f32.size(); i++) {
            k_f32[i] = ggml_bf16_to_fp32(k_bf16[i]);
        }
    } else if (k_tensor->type == GGML_TYPE_F32) {
        memcpy(k_f32.data(), k_raw.data(), k_f32.size() * sizeof(float));
    } else {
        LOG_ERR("Unsupported K cache type: %d\n", (int)k_tensor->type);
        return;
    }

    float scale = 1.0f / sqrtf((float)d_k);

    // Compute per-KV-position attention score:
    // For each query token i, each head h:
    //   score_raw[h][i][j] = dot(Q[i,h,:], K[j, h/gqa_ratio, :]) * scale
    // Then softmax across j for each (i,h), then sum over i and h for each j
    // Result: per KV position j, total attention received from all queries

    // Accumulate total attention per KV position
    std::vector<double> kv_attn(n_kv, 0.0);

    // Q layout: [d_k, n_head, n_q] contiguous (captured from Qcur tensor)
    // Q[i,h,d] = q_data[(i * n_head + h) * d_k + d]
    // K layout after conversion: [n_kv, n_head_kv * d_k] row-major
    // K[j, kv_h, d] = k_f32[j * n_embd_k_gqa + kv_h * d_k + d]

    // Process per query token, per head
    std::vector<float> logits(n_kv);

    for (int qi = 0; qi < n_q; qi++) {
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / gqa_ratio;

            // Compute Q[qi,h,:] @ K[:, kv_h, :]^T
            const float * q_ptr = qstate.q_data.data() + (qi * n_head + h) * d_k;
            for (int j = 0; j < n_kv; j++) {
                const float * k_ptr = k_f32.data() + j * n_embd_k_gqa + kv_h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                logits[j] = dot * scale;
            }

            // Softmax over KV positions
            float max_val = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (int j = 0; j < n_kv; j++) {
                logits[j] = expf(logits[j] - max_val);
                sum_exp += logits[j];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < n_kv; j++) {
                kv_attn[j] += (double)(logits[j] * inv_sum);
            }
        }
    }

    // Normalize: average over heads (but sum over queries, matching legacy behavior)
    // Legacy: tok.prefill_attn_score += total_attn / n_heads  (summed over queries)
    // Here we already summed over queries and heads, so divide by n_head
    for (int j = 0; j < n_kv; j++) {
        kv_attn[j] /= (double)n_head;
    }

    // Assign scores to visual tokens via kv_cell_idx
    int scored = 0;
    for (auto & tok : vis_token_map.tokens) {
        if (tok.pruned) continue;
        int cell = tok.kv_cell_idx;
        if (cell >= 0 && cell < n_kv) {
            tok.prefill_attn_score = (float)kv_attn[cell];
            scored++;
        }
    }

    LOG_INF("CPU attention scored %d visual tokens (%.1f ms compute would vary)\n", scored, 0.0f);
}

// ============================================================================
// Section 5b-5: Token estimation
// ============================================================================

static int estimate_tokens(int w, int h, int align = 28) {
    int w_a = std::max(align, (int)(std::round((float)w / align) * align));
    int h_a = std::max(align, (int)(std::round((float)h / align) * align));
    return (w_a / align) * (h_a / align);
}

static frame_crop_info compute_crop_region(const frame_spatial_analysis & sa,
                                            int img_w, int img_h,
                                            int grid_size,
                                            int crop_padding,
                                            int min_crop_dim) {
    frame_crop_info ci;
    ci.frame_index = sa.frame_index;
    ci.orig_w = img_w;
    ci.orig_h = img_h;
    ci.estimated_tokens_full = estimate_tokens(img_w, img_h);

    int gx_min = grid_size, gx_max = -1;
    int gy_min = grid_size, gy_max = -1;

    for (const auto & p : sa.patches) {
        if (p.selected) {
            gx_min = std::min(gx_min, p.grid_x);
            gx_max = std::max(gx_max, p.grid_x);
            gy_min = std::min(gy_min, p.grid_y);
            gy_max = std::max(gy_max, p.grid_y);
        }
    }

    if (gx_max < 0) {
        ci.crop_x = 0; ci.crop_y = 0;
        ci.crop_w = img_w; ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
        return ci;
    }

    int pw = img_w / grid_size;
    int ph = img_h / grid_size;

    int px_min = gx_min * pw;
    int py_min = gy_min * ph;
    int px_max = (gx_max == grid_size - 1) ? img_w : (gx_max + 1) * pw;
    int py_max = (gy_max == grid_size - 1) ? img_h : (gy_max + 1) * ph;

    px_min = std::max(0, px_min - crop_padding);
    py_min = std::max(0, py_min - crop_padding);
    px_max = std::min(img_w, px_max + crop_padding);
    py_max = std::min(img_h, py_max + crop_padding);

    int cw = px_max - px_min;
    int ch = py_max - py_min;

    if (cw < min_crop_dim) {
        int expand = min_crop_dim - cw;
        int left  = expand / 2;
        int right = expand - left;
        px_min = std::max(0, px_min - left);
        px_max = std::min(img_w, px_max + right);
        cw = px_max - px_min;
        if (cw < min_crop_dim) {
            if (px_min > 0) px_min = std::max(0, px_max - min_crop_dim);
            else px_max = std::min(img_w, px_min + min_crop_dim);
            cw = px_max - px_min;
        }
    }
    if (ch < min_crop_dim) {
        int expand = min_crop_dim - ch;
        int top    = expand / 2;
        int bottom = expand - top;
        py_min = std::max(0, py_min - top);
        py_max = std::min(img_h, py_max + bottom);
        ch = py_max - py_min;
        if (ch < min_crop_dim) {
            if (py_min > 0) py_min = std::max(0, py_max - min_crop_dim);
            else py_max = std::min(img_h, py_min + min_crop_dim);
            ch = py_max - py_min;
        }
    }

    ci.crop_x = px_min;
    ci.crop_y = py_min;
    ci.crop_w = cw;
    ci.crop_h = ch;

    double crop_area = (double)cw * ch;
    double full_area = (double)img_w * img_h;
    if (crop_area >= 0.9 * full_area) {
        ci.crop_x = 0; ci.crop_y = 0;
        ci.crop_w = img_w; ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
    } else {
        ci.was_cropped = true;
        ci.estimated_tokens_crop = estimate_tokens(cw, ch);
    }

    return ci;
}

static mtmd::bitmap crop_bitmap(const mtmd::bitmap & full, const frame_crop_info & ci) {
    const unsigned char * src = full.data();
    uint32_t src_w = full.nx();
    std::vector<unsigned char> cropped(ci.crop_w * ci.crop_h * 3);
    for (int row = 0; row < ci.crop_h; row++) {
        memcpy(&cropped[row * ci.crop_w * 3],
               &src[((ci.crop_y + row) * src_w + ci.crop_x) * 3],
               ci.crop_w * 3);
    }
    return mtmd::bitmap((uint32_t)ci.crop_w, (uint32_t)ci.crop_h, cropped.data());
}

// ============================================================================
// Section 5c: Pre-LLM visual token pruning
// ============================================================================

// ViT attention capture state — captures attention weights during ViT encoding
struct vit_attention_capture_state {
    bool active = false;
    int n_vit_layers = 32;           // Qwen2.5-VL ViT has 32 layers
    int target_layer;                // = n_vit_layers - 1 (last layer)
    std::vector<float> token_attn_scores;  // [n_tokens] averaged attention received
    int n_tokens = 0;
    int n_heads = 0;

    void prepare(int expected_tokens) {
        target_layer = n_vit_layers - 1;
        n_tokens = expected_tokens;
        token_attn_scores.assign(expected_tokens, 0.0f);
        n_heads = 0;
    }
};

// cb_eval callback for ViT encoding — captures attention from last ViT layer
static bool vit_attn_softmax_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (vit_attention_capture_state *)user_data;

    if (!state || !state->active) {
        return !ask;
    }

    if (ask) {
        std::string name(t->name);
        std::string target = "vit_kq_soft_max-" + std::to_string(state->target_layer);
        return (name == target);
    }

    // Tensor shape: [n_kv, n_heads, n_q] but after softmax it's [n_tokens, n_heads, n_tokens]
    // Actually for ViT self-attention: ne[0]=n_kv (keys), ne[1]=n_q (queries per head), ne[2]=n_heads
    // But the layout is: attn[head][query][key] = probability that query q attends to key k
    int64_t n_kv    = t->ne[0];
    int64_t n_q     = t->ne[1];
    int64_t n_heads = t->ne[2];

    if (n_kv != n_q || t->type != GGML_TYPE_F32) {
        // Not a square self-attention matrix or wrong type, skip
        return true;
    }

    state->n_heads = (int)n_heads;
    int n_tokens = (int)n_kv;

    if (n_tokens != state->n_tokens) {
        // Dimension mismatch, resize
        state->n_tokens = n_tokens;
        state->token_attn_scores.assign(n_tokens, 0.0f);
    }

    const float * data = (const float *)t->data;

    // Compute "attention received" for each token j:
    // score[j] = mean over heads of sum_over_queries(attn[head][query][j])
    for (int h = 0; h < (int)n_heads; h++) {
        const float * head_data = data + h * n_q * n_kv;
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k < n_tokens; k++) {
                state->token_attn_scores[k] += head_data[q * n_kv + k];
            }
        }
    }

    // Normalize by n_heads (will be per-token sum of attention received across all queries)
    for (int k = 0; k < n_tokens; k++) {
        state->token_attn_scores[k] /= (float)n_heads;
    }

    return true;
}

// Pre-LLM token scorer — combines motion, embedding norm, and ViT attention
struct pre_llm_token_scorer {
    float weight_motion;
    float weight_norm;
    float weight_attn;
    float keep_ratio;
    std::string strategy;  // combined, random, motion-only, norm-only, attn-only, motion-norm

    struct scored_token {
        int token_idx;       // index in the full [n_tokens] embedding buffer
        float score;
    };

    pre_llm_token_scorer(float wm, float wn, float wa, float kr, const std::string & strat = "combined")
        : weight_motion(wm), weight_norm(wn), weight_attn(wa), keep_ratio(kr), strategy(strat) {}

    std::vector<scored_token> score_and_select(
        const float * embeddings, int n_tokens, int n_embd,
        int nx, int ny,
        const frame_spatial_analysis & sa,
        const vit_attention_capture_state & vit_capture,
        int patch_grid,
        int frame_seed = 0) {

        int n_keep = std::max(1, (int)(n_tokens * keep_ratio));

        // Random strategy: select random tokens (deterministic per-frame)
        if (strategy == "random") {
            std::vector<scored_token> all_tokens(n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                all_tokens[i].token_idx = i;
            }
            std::mt19937 rng(42 + frame_seed);
            std::shuffle(all_tokens.begin(), all_tokens.end(), rng);
            all_tokens.resize(n_keep);
            std::sort(all_tokens.begin(), all_tokens.end(),
                [](const scored_token & a, const scored_token & b) { return a.token_idx < b.token_idx; });
            for (auto & t : all_tokens) t.score = 1.0f;
            return all_tokens;
        }

        // Compute per-token scores from three signals
        std::vector<float> motion_scores(n_tokens, 0.0f);
        std::vector<float> norm_scores(n_tokens, 0.0f);
        std::vector<float> attn_scores(n_tokens, 0.0f);

        // Signal A: Motion — map spatial analysis patches to ViT tokens
        for (int ty = 0; ty < ny; ty++) {
            for (int tx = 0; tx < nx; tx++) {
                int idx = ty * nx + tx;
                int sa_gx = tx * patch_grid / nx;
                int sa_gy = ty * patch_grid / ny;
                sa_gx = std::min(sa_gx, patch_grid - 1);
                sa_gy = std::min(sa_gy, patch_grid - 1);

                float tok_motion = 0.0f;
                for (const auto & p : sa.patches) {
                    if (p.grid_x == sa_gx && p.grid_y == sa_gy) {
                        tok_motion = p.mean_motion;
                        break;
                    }
                }
                motion_scores[idx] = tok_motion;
            }
        }

        // Signal B: Embedding L2 norm
        for (int i = 0; i < n_tokens; i++) {
            const float * emb = embeddings + i * n_embd;
            float sum_sq = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                sum_sq += emb[d] * emb[d];
            }
            norm_scores[i] = std::sqrt(sum_sq);
        }

        // Signal C: ViT attention received
        if ((int)vit_capture.token_attn_scores.size() >= n_tokens) {
            for (int i = 0; i < n_tokens; i++) {
                attn_scores[i] = vit_capture.token_attn_scores[i];
            }
        }

        // Min-max normalize each signal to [0, 1]
        auto normalize = [](std::vector<float> & scores) {
            if (scores.empty()) return;
            float mn = *std::min_element(scores.begin(), scores.end());
            float mx = *std::max_element(scores.begin(), scores.end());
            float range = mx - mn;
            if (range > 1e-8f) {
                for (auto & s : scores) {
                    s = (s - mn) / range;
                }
            } else {
                for (auto & s : scores) {
                    s = 0.5f;
                }
            }
        };

        normalize(motion_scores);
        normalize(norm_scores);
        normalize(attn_scores);

        // Determine effective weights based on strategy
        float eff_wm = weight_motion, eff_wn = weight_norm, eff_wa = weight_attn;
        if (strategy == "motion-only") {
            eff_wm = 1.0f; eff_wn = 0.0f; eff_wa = 0.0f;
        } else if (strategy == "norm-only") {
            eff_wm = 0.0f; eff_wn = 1.0f; eff_wa = 0.0f;
        } else if (strategy == "attn-only") {
            eff_wm = 0.0f; eff_wn = 0.0f; eff_wa = 1.0f;
        } else if (strategy == "motion-norm") {
            eff_wm = 0.5f; eff_wn = 0.5f; eff_wa = 0.0f;
        }
        // "combined" uses the original weights

        // Combine scores
        std::vector<scored_token> all_tokens(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            all_tokens[i].token_idx = i;
            all_tokens[i].score = eff_wm * motion_scores[i]
                                + eff_wn * norm_scores[i]
                                + eff_wa * attn_scores[i];
        }

        // Sort by score descending, keep top keep_ratio
        std::sort(all_tokens.begin(), all_tokens.end(),
            [](const scored_token & a, const scored_token & b) { return a.score > b.score; });

        all_tokens.resize(n_keep);

        // Sort by index to preserve spatial ordering for batch injection
        std::sort(all_tokens.begin(), all_tokens.end(),
            [](const scored_token & a, const scored_token & b) { return a.token_idx < b.token_idx; });

        return all_tokens;
    }

    // Score ALL tokens without filtering — returns scores for every token
    // Used for correlation analysis (compare pre-LLM scores with LLM attention)
    std::vector<scored_token> score_all(
        const float * embeddings, int n_tokens, int n_embd,
        int nx, int ny,
        const frame_spatial_analysis & sa,
        const vit_attention_capture_state & vit_capture,
        int patch_grid) {

        std::vector<float> motion_scores(n_tokens, 0.0f);
        std::vector<float> norm_scores(n_tokens, 0.0f);
        std::vector<float> attn_scores(n_tokens, 0.0f);

        // Signal A: Motion
        for (int ty = 0; ty < ny; ty++) {
            for (int tx = 0; tx < nx; tx++) {
                int idx = ty * nx + tx;
                int sa_gx = tx * patch_grid / nx;
                int sa_gy = ty * patch_grid / ny;
                sa_gx = std::min(sa_gx, patch_grid - 1);
                sa_gy = std::min(sa_gy, patch_grid - 1);
                float tok_motion = 0.0f;
                for (const auto & p : sa.patches) {
                    if (p.grid_x == sa_gx && p.grid_y == sa_gy) {
                        tok_motion = p.mean_motion;
                        break;
                    }
                }
                motion_scores[idx] = tok_motion;
            }
        }

        // Signal B: Embedding L2 norm
        for (int i = 0; i < n_tokens; i++) {
            const float * emb = embeddings + i * n_embd;
            float sum_sq = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                sum_sq += emb[d] * emb[d];
            }
            norm_scores[i] = std::sqrt(sum_sq);
        }

        // Signal C: ViT attention received
        if ((int)vit_capture.token_attn_scores.size() >= n_tokens) {
            for (int i = 0; i < n_tokens; i++) {
                attn_scores[i] = vit_capture.token_attn_scores[i];
            }
        }

        auto normalize = [](std::vector<float> & scores) {
            if (scores.empty()) return;
            float mn = *std::min_element(scores.begin(), scores.end());
            float mx = *std::max_element(scores.begin(), scores.end());
            float range = mx - mn;
            if (range > 1e-8f) {
                for (auto & s : scores) s = (s - mn) / range;
            } else {
                for (auto & s : scores) s = 0.5f;
            }
        };

        normalize(motion_scores);
        normalize(norm_scores);
        normalize(attn_scores);

        float eff_wm = weight_motion, eff_wn = weight_norm, eff_wa = weight_attn;
        if (strategy == "motion-only")  { eff_wm = 1.0f; eff_wn = 0.0f; eff_wa = 0.0f; }
        else if (strategy == "norm-only")   { eff_wm = 0.0f; eff_wn = 1.0f; eff_wa = 0.0f; }
        else if (strategy == "attn-only")   { eff_wm = 0.0f; eff_wn = 0.0f; eff_wa = 1.0f; }
        else if (strategy == "motion-norm") { eff_wm = 0.5f; eff_wn = 0.5f; eff_wa = 0.0f; }

        std::vector<scored_token> all_tokens(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            all_tokens[i].token_idx = i;
            all_tokens[i].score = eff_wm * motion_scores[i]
                                + eff_wn * norm_scores[i]
                                + eff_wa * attn_scores[i];
        }
        return all_tokens;
    }
};

// Inject a subset of visual tokens with correct M-RoPE positions
static int32_t inject_pruned_visual_tokens(
    llama_context * lctx,
    mtmd_context * ctx_vision,
    const float * full_embeddings,
    const std::vector<pre_llm_token_scorer::scored_token> & selected,
    int nx, int ny, int n_embd,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch_size,
    llama_pos * new_n_past) {

    int n_keep = (int)selected.size();
    if (n_keep == 0) {
        *new_n_past = n_past + std::max(nx, ny);
        return 0;
    }

    // Build subset embedding buffer
    std::vector<float> pruned_embd(n_keep * n_embd);
    for (int i = 0; i < n_keep; i++) {
        memcpy(pruned_embd.data() + i * n_embd,
               full_embeddings + selected[i].token_idx * n_embd,
               n_embd * sizeof(float));
    }

    // Build M-RoPE positions for subset tokens (4 dims per token)
    bool use_mrope = mtmd_decode_use_mrope(ctx_vision);
    int n_pos_per_embd = use_mrope ? 4 : 1;
    std::vector<llama_pos> pos(n_keep * n_pos_per_embd, 0);

    if (use_mrope) {
        for (int i = 0; i < n_keep; i++) {
            int idx = selected[i].token_idx;
            int grid_y = idx / nx;
            int grid_x = idx % nx;
            pos[i]                = n_past;            // temporal (same for all)
            pos[i + n_keep]       = n_past + grid_y;   // y position
            pos[i + n_keep * 2]   = n_past + grid_x;   // x position
            pos[i + n_keep * 3]   = 0;                  // unused dim
        }
    } else {
        for (int i = 0; i < n_keep; i++) {
            pos[i] = n_past + i;
        }
    }

    // Build batch metadata
    std::vector<int32_t> n_seq_id_vec(n_keep, 1);
    std::vector<llama_seq_id> seq_id_0(1, seq_id);
    std::vector<llama_seq_id *> seq_ids(n_keep, seq_id_0.data());
    std::vector<int8_t> logits(n_keep, 0);

    llama_batch batch = {
        /*n_tokens       =*/ n_keep,
        /*tokens         =*/ nullptr,
        /*embd           =*/ pruned_embd.data(),
        /*pos            =*/ pos.data(),
        /*n_seq_id       =*/ n_seq_id_vec.data(),
        /*seq_id         =*/ seq_ids.data(),
        /*logits         =*/ logits.data(),
    };

    // Non-causal attention for image tokens
    if (mtmd_decode_use_non_causal(ctx_vision)) {
        llama_set_causal_attn(lctx, false);
    }

    // Decode in sub-batches
    int n_img_batches = (n_keep + n_batch_size - 1) / n_batch_size;
    for (int ib = 0; ib < n_img_batches; ib++) {
        int offset = ib * n_batch_size;
        int n_tokens_batch = std::min(n_batch_size, n_keep - offset);

        // Build view for this sub-batch
        std::vector<llama_pos> pos_view;
        llama_pos * pos_ptr;
        if (use_mrope) {
            pos_view.reserve(n_tokens_batch * 4);
            for (int d = 0; d < 4; d++) {
                size_t src_idx = d * n_keep + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens_batch);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }

        llama_batch sub_batch = {
            /*n_tokens       =*/ n_tokens_batch,
            /*tokens         =*/ nullptr,
            /*embd           =*/ pruned_embd.data() + offset * n_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ n_seq_id_vec.data() + offset,
            /*seq_id         =*/ seq_ids.data() + offset,
            /*logits         =*/ logits.data() + offset,
        };

        LOG_INF("decoding pruned image batch %d/%d, n_tokens_batch = %d\n",
                ib + 1, n_img_batches, n_tokens_batch);

        int32_t ret = llama_decode(lctx, sub_batch);
        if (ret != 0) {
            LOG_ERR("failed to decode pruned image batch\n");
            if (mtmd_decode_use_non_causal(ctx_vision)) {
                llama_set_causal_attn(lctx, true);
            }
            return ret;
        }
    }

    if (mtmd_decode_use_non_causal(ctx_vision)) {
        llama_set_causal_attn(lctx, true);
    }

    // CRITICAL: n_past advances by max(nx, ny) regardless of n_keep
    *new_n_past = n_past + std::max(nx, ny);
    return 0;
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
        int flow_idx = sf.frame_index - 1;
        if (flow_idx >= 0 && flow_idx < (int)profile.frame_stats.size()) {
            selected_motion += profile.frame_stats[flow_idx].p95_magnitude;
        }
    }

    return (total_motion > 0.0) ? selected_motion / total_motion : 0.0;
}

// ============================================================================
// Section 6b: LLM-side progressive visual token pruning
// ============================================================================

static int do_patch_level_pruning(
    llama_context * lctx,
    visual_token_map & token_map,
    const attention_capture_state & capture_state,
    const streaming_params & sparams) {

    (void)capture_state;

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        LOG_ERR("Failed to get memory for pruning\n");
        return 0;
    }

    struct patch_score {
        std::string key;
        float score;
        std::vector<int> token_indices;
    };
    std::vector<patch_score> patches;

    for (auto & kv : token_map.patch_to_token_indices) {
        float total_motion = 0.0f;
        float total_attn = 0.0f;
        int attn_count = 0;
        int active_count = 0;
        std::vector<int> active_indices;

        for (int idx : kv.second) {
            auto & t = token_map.tokens[idx];
            if (t.pruned) continue;
            active_count++;
            active_indices.push_back(idx);
            total_motion += t.motion_score;
            if (t.attn_samples > 0) {
                total_attn += t.attn_score / t.attn_samples;
                attn_count++;
            }
        }
        if (active_count == 0) continue;

        float avg_motion = total_motion / active_count;
        float avg_attn = (attn_count > 0) ? total_attn / attn_count : 0.0f;

        float norm_motion = std::min(avg_motion / 10.0f, 1.0f);
        float combined = (1.0f - sparams.motion_weight) * avg_attn +
                         sparams.motion_weight * norm_motion;

        patch_score ps;
        ps.key = kv.first;
        ps.score = combined;
        ps.token_indices = active_indices;
        patches.push_back(ps);
    }

    if (patches.empty()) return 0;

    std::sort(patches.begin(), patches.end(),
        [](const patch_score & a, const patch_score & b) { return a.score < b.score; });

    int n_total = (int)patches.size();
    int n_keep = std::max(1, (int)(n_total * sparams.patch_keep_ratio));
    int n_prune = n_total - n_keep;

    int n_tokens_pruned = 0;
    int n_patches_pruned = 0;

    for (int i = 0; i < n_prune; i++) {
        const auto & ps = patches[i];
        for (int idx : ps.token_indices) {
            auto & t = token_map.tokens[idx];
            if (t.pruned) continue;

            bool ok = llama_memory_seq_rm_pos_ext(mem, 0, t.kv_pos, t.ext_x, t.ext_y);
            if (ok) {
                t.pruned = true;
                n_tokens_pruned++;
            }
        }
        n_patches_pruned++;
    }

    LOG_INF("Pruning round 1: removed %d visual tokens (%d/%d patches pruned, keep_ratio=%.2f)\n",
            n_tokens_pruned, n_patches_pruned, n_total, sparams.patch_keep_ratio);

    return n_tokens_pruned;
}

static int do_token_level_pruning(
    llama_context * lctx,
    visual_token_map & token_map,
    const attention_capture_state & capture_state,
    const streaming_params & sparams) {

    (void)capture_state;

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        LOG_ERR("Failed to get memory for pruning\n");
        return 0;
    }

    struct token_score {
        int idx;
        float score;
    };
    std::vector<token_score> active_tokens;

    for (int i = 0; i < (int)token_map.tokens.size(); i++) {
        auto & t = token_map.tokens[i];
        if (t.pruned) continue;

        float score = 0.0f;
        if (t.attn_samples > 0) {
            score = t.attn_score / t.attn_samples;
        } else {
            score = std::min(t.motion_score / 10.0f, 1.0f);
        }

        token_score ts;
        ts.idx = i;
        ts.score = score;
        active_tokens.push_back(ts);
    }

    if (active_tokens.empty()) return 0;

    std::sort(active_tokens.begin(), active_tokens.end(),
        [](const token_score & a, const token_score & b) { return a.score < b.score; });

    int n_total = (int)active_tokens.size();
    int n_keep = std::max(1, (int)(n_total * sparams.token_keep_ratio));
    int n_prune = n_total - n_keep;

    int n_tokens_pruned = 0;

    for (int i = 0; i < n_prune; i++) {
        auto & t = token_map.tokens[active_tokens[i].idx];
        if (t.pruned) continue;

        bool ok = llama_memory_seq_rm_pos_ext(mem, 0, t.kv_pos, t.ext_x, t.ext_y);
        if (ok) {
            t.pruned = true;
            n_tokens_pruned++;
        }
    }

    LOG_INF("Pruning round 2: removed %d/%d remaining visual tokens (keep_ratio=%.2f)\n",
            n_tokens_pruned, n_total, sparams.token_keep_ratio);

    return n_tokens_pruned;
}

static int do_prefill_motion_pruning(
    llama_context * lctx,
    visual_token_map & token_map,
    int frame_idx,
    float keep_ratio) {

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        LOG_ERR("Failed to get memory for prefill pruning\n");
        return 0;
    }

    if (frame_idx < 0 || frame_idx >= (int)token_map.frame_ranges.size()) {
        return 0;
    }

    const auto & fr = token_map.frame_ranges[frame_idx];

    struct token_score {
        int idx;
        float motion;
    };
    std::vector<token_score> candidates;

    int end_idx = fr.token_start_idx + fr.token_count;
    for (int i = fr.token_start_idx; i < end_idx && i < (int)token_map.tokens.size(); i++) {
        auto & t = token_map.tokens[i];
        if (t.pruned) continue;
        if (t.frame_idx != frame_idx) continue;
        token_score ts;
        ts.idx = i;
        ts.motion = t.motion_score;
        candidates.push_back(ts);
    }

    if (candidates.empty()) return 0;

    std::sort(candidates.begin(), candidates.end(),
        [](const token_score & a, const token_score & b) { return a.motion < b.motion; });

    int n_total = (int)candidates.size();
    int n_keep = std::max(1, (int)(n_total * keep_ratio));
    int n_prune = n_total - n_keep;

    int n_tokens_pruned = 0;
    for (int i = 0; i < n_prune; i++) {
        auto & t = token_map.tokens[candidates[i].idx];
        if (t.pruned) continue;

        bool ok = llama_memory_seq_rm_pos_ext(mem, 0, t.kv_pos, t.ext_x, t.ext_y);
        if (ok) {
            t.pruned = true;
            n_tokens_pruned++;
        }
    }

    LOG_INF("Prefill pruning frame %d: removed %d/%d visual tokens (keep_ratio=%.2f)\n",
            frame_idx, n_tokens_pruned, n_total, keep_ratio);

    return n_tokens_pruned;
}

// ============================================================================
// Section 7: Inference context
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

    adaptive_vlm_context(common_params & params,
                         ggml_backend_sched_eval_callback vit_cb_eval = nullptr,
                         void * vit_cb_eval_user_data = nullptr)
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

        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu         = params.mmproj_use_gpu;
        mparams.print_timings   = true;
        mparams.n_threads       = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup          = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;

        // Register ViT attention capture callback if provided
        if (vit_cb_eval) {
            mparams.cb_eval = vit_cb_eval;
            mparams.cb_eval_user_data = vit_cb_eval_user_data;
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
// Section 7b: CUDA KV cache reuse helper
// ============================================================================

#ifdef HAVE_CUDA_KV_REUSE
// Perform KV cache reuse: reserve new cells and GPU-copy K/V from keyframe
// with temporal RoPE re-rotation on K values.
// Returns the number of new cell indices written, or -1 on failure.
static int perform_kv_cache_reuse(
    llama_context * lctx,
    const llama_model * model,
    const kv_cache_snapshot & snapshot,
    llama_seq_id seq_id,
    llama_pos new_temporal_pos,
    int nx, int ny,
    int n_tokens,                        // number of static tokens to reuse
    const int * token_indices,           // [n_tokens] indices into keyframe grid (row-major)
    std::vector<int32_t> & out_new_cells) // output: new cell indices
{
    int64_t t0 = ggml_time_us();

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) return -1;

    // Build M-RoPE positions for the new cells
    std::vector<llama_pos> pos(n_tokens * 4, 0);
    for (int i = 0; i < n_tokens; i++) {
        int idx = token_indices[i];
        int grid_y = idx / nx;
        int grid_x = idx % nx;
        pos[i]                 = new_temporal_pos;            // temporal
        pos[i + n_tokens]      = new_temporal_pos + grid_y;   // y
        pos[i + n_tokens * 2]  = new_temporal_pos + grid_x;   // x
        pos[i + n_tokens * 3]  = 0;                           // unused
    }

    int64_t t1 = ggml_time_us();

    // Reserve new KV cells
    out_new_cells.resize(n_tokens);
    if (!llama_memory_reserve_cells(mem, seq_id, n_tokens, pos.data(), true, out_new_cells.data())) {
        LOG_ERR("KV cache reuse: failed to reserve %d cells\n", n_tokens);
        return -1;
    }

    int64_t t2 = ggml_time_us();

    // Build source cell indices from snapshot
    std::vector<int32_t> src_cells(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        src_cells[i] = snapshot.cell_indices[token_indices[i]];
    }

    // Upload cell index arrays to GPU
    int * d_src_cells = nullptr;
    int * d_dst_cells = nullptr;
    cudaMalloc(&d_src_cells, n_tokens * sizeof(int));
    cudaMalloc(&d_dst_cells, n_tokens * sizeof(int));
    cudaMemcpy(d_src_cells, src_cells.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_cells, out_new_cells.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    int64_t t3 = ggml_time_us();

    // Query model parameters
    int32_t n_layers   = llama_model_n_layer(model);
    int32_t n_kv_heads = llama_model_n_head_kv(model);
    int32_t n_embd_hk  = llama_model_n_embd_head_k(model);
    float   freq_base  = llama_model_rope_freq_base_train(model);
    int32_t sections[4];
    llama_model_rope_sections(model, sections);
    bool v_trans = llama_memory_v_trans(mem);

    // Collect K/V data pointers from tensors
    std::vector<void *> k_ptrs(n_layers);
    std::vector<void *> v_ptrs(n_layers);
    // Also need kv_size from the first K tensor dimensions
    int kv_size = 0;
    int n_embd_hv = n_embd_hk; // assume same for now

    for (int il = 0; il < n_layers; il++) {
        const ggml_tensor * k_t = llama_memory_k_tensor(mem, il);
        const ggml_tensor * v_t = llama_memory_v_tensor(mem, il);
        if (!k_t || !v_t) {
            LOG_ERR("KV cache reuse: null tensor at layer %d\n", il);
            cudaFree(d_src_cells);
            cudaFree(d_dst_cells);
            return -1;
        }
        k_ptrs[il] = k_t->data;
        v_ptrs[il] = v_t->data;

        if (il == 0) {
            // K tensor: [n_embd_k_gqa, kv_size, n_stream]
            // V tensor: [n_embd_v_gqa, kv_size, n_stream]
            kv_size = (int)k_t->ne[1];
            // n_embd_v_gqa = n_embd_head_v * n_kv_heads
            int n_embd_v_gqa = (int)v_t->ne[0];
            n_embd_hv = n_embd_v_gqa / n_kv_heads;
        }
    }

    int64_t t4 = ggml_time_us();

    // Build kernel params
    kv_reuse_params params;
    params.n_tokens        = n_tokens;
    params.n_layers        = n_layers;
    params.n_kv_heads      = n_kv_heads;
    params.n_embd_head_k   = n_embd_hk;
    params.n_embd_head_v   = n_embd_hv;
    params.kv_size         = kv_size;
    params.rope_sections[0] = sections[0];
    params.rope_sections[1] = sections[1];
    params.rope_sections[2] = sections[2];
    params.rope_sections[3] = sections[3];
    params.freq_base       = freq_base;
    params.old_temporal_pos = (int)snapshot.temporal_pos;
    params.new_temporal_pos = (int)new_temporal_pos;
    params.v_trans         = v_trans;
    params.src_cells       = d_src_cells;
    params.dst_cells       = d_dst_cells;

    // Launch on default stream (synchronous with respect to other GPU work)
    launch_kv_reuse(params, k_ptrs.data(), v_ptrs.data(), 0);

    // Synchronize to ensure copy is complete before proceeding
    cudaDeviceSynchronize();

    int64_t t5 = ggml_time_us();

    // Cleanup GPU buffers
    cudaFree(d_src_cells);
    cudaFree(d_dst_cells);

    int64_t t6 = ggml_time_us();

    LOG_INF("    KV reuse breakdown (%d tokens): pos_build=%.1f ms, reserve_cells=%.1f ms, "
            "cuda_upload=%.1f ms, param_collect=%.1f ms, kernel+sync=%.1f ms, cleanup=%.1f ms, "
            "total=%.1f ms\n",
            n_tokens,
            (t1 - t0) / 1000.0f, (t2 - t1) / 1000.0f,
            (t3 - t2) / 1000.0f, (t4 - t3) / 1000.0f,
            (t5 - t4) / 1000.0f, (t6 - t5) / 1000.0f,
            (t6 - t0) / 1000.0f);

    return n_tokens;
}
#endif // HAVE_CUDA_KV_REUSE

// ============================================================================
// Section 8: Inference result
// ============================================================================

struct inference_result {
    std::string generated_text;
    int   n_tokens_generated = 0;
    float encode_time_ms     = 0.0f;
    float inference_time_ms  = 0.0f;

    int   total_tokens_encoded = 0;
    int   total_tokens_cached  = 0;
    float total_encode_time_ms = 0.0f;
    float estimated_full_encode_time_ms = 0.0f;

    struct per_frame_cache_info {
        int  frame_index     = 0;
        bool is_keyframe     = false;
        bool used_cache      = false;
        bool is_full_encode  = false;
        int  n_tokens_encoded = 0;
        int  n_tokens_cached  = 0;
        float encode_time_ms  = 0.0f;
        float splice_time_ms  = 0.0f;
        int  n_tokens_pre_llm_pruned = 0;
        int  n_tokens_injected       = 0;
    };
    std::vector<per_frame_cache_info> frame_cache_infos;

    int   n_visual_tokens_initial      = 0;
    int   n_visual_tokens_after_round1 = 0;
    int   n_visual_tokens_after_round2 = 0;
    float pruning_overhead_ms          = 0.0f;
    int   n_tokens_pruned_round1       = 0;
    int   n_tokens_pruned_round2       = 0;

    int   n_tokens_pruned_prefill       = 0;
    float prefill_pruning_overhead_ms   = 0.0f;

    // Pre-LLM pruning stats
    int   pre_llm_total_tokens_before   = 0;
    int   pre_llm_total_tokens_injected = 0;
    int   pre_llm_total_tokens_pruned   = 0;

    // Post-prefill eviction stats
    int   postprefill_n_tokens_before    = 0;
    int   postprefill_n_tokens_evicted   = 0;
    float postprefill_eviction_ms        = 0.0f;
    float postprefill_attn_retained_pct  = 0.0f;
    bool  postprefill_needs_defrag       = false;

    // Attention correlation analysis results
    struct correlation_result {
        int    n_tokens_analyzed = 0;
        float  pearson_r         = 0.0f;
        float  spearman_rho      = 0.0f;
        float  avg_attn_kept     = 0.0f;   // avg LLM attention for tokens scorer would keep
        float  avg_attn_pruned   = 0.0f;   // avg LLM attention for tokens scorer would prune
        float  attn_ratio        = 0.0f;   // kept/pruned ratio (>1 = scoring works)
        int    n_kept            = 0;
        int    n_pruned          = 0;

        // Prefill attention retention analysis
        float  prefill_attn_retained_pct = 0.0f;
        float  prefill_attn_total        = 0.0f;
        float  prefill_attn_on_kept      = 0.0f;
        int    prefill_n_visual_tokens   = 0;

        struct per_frame_corr {
            int   frame_idx      = 0;
            int   n_tokens       = 0;
            float pearson_r      = 0.0f;
            float avg_attn_kept  = 0.0f;
            float avg_attn_pruned = 0.0f;
            float attn_ratio     = 0.0f;
            float prefill_attn_retained_pct = 0.0f;
        };
        std::vector<per_frame_corr> per_frame;
    };
    correlation_result correlation;
};

// ============================================================================
// Section 8b: Inference with encoder-level embedding cache
// ============================================================================

static inference_result run_inference_with_embedding_cache(
    adaptive_vlm_context & ctx,
    const std::vector<std::string> & frame_paths,
    const std::vector<frame_spatial_analysis> & spatial_analyses,
    const streaming_params & sparams,
    const std::string & prompt, int n_predict,
    std::vector<frame_crop_info> & crop_infos_out,
    attention_capture_state * ext_capture_state = nullptr,
    vit_attention_capture_state * vit_capture_state = nullptr,
    qcur_capture_state * ext_qcur_state = nullptr) {

    inference_result result;
    int64_t t_start = ggml_time_ms();
    crop_infos_out.clear();

    const llama_model * model = llama_get_model(ctx.lctx);
    int n_embd = llama_model_n_embd_inp(model);

    embedding_cache cache;
    mtmd::input_chunk_ptr keyframe_chunk_ptr(nullptr);
    int nx_full = 0, ny_full = 0;
    kv_cache_snapshot kv_snapshot;  // for CUDA KV cache reuse

    // Phase 1: Load ALL frames as full-size bitmaps
    mtmd::bitmaps full_bitmaps;
    for (size_t i = 0; i < frame_paths.size(); i++) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), frame_paths[i].c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", frame_paths[i].c_str());
            inference_result empty;
            return empty;
        }
        full_bitmaps.entries.push_back(std::move(bmp));
    }

    // Phase 2: Build prompt with media markers and tokenize
    std::string full_content;
    for (size_t i = 0; i < full_bitmaps.entries.size(); i++) {
        full_content += mtmd_default_marker();
    }
    full_content += "\n" + prompt;

    common_chat_msg msg;
    msg.role    = "user";
    msg.content = full_content;

    bool add_bos = ctx.chat_history.empty();
    auto formatted = common_chat_format_single(
        ctx.tmpls.get(), ctx.chat_history, msg, true, false);
    ctx.chat_history.push_back(msg);

    LOG_INF("Formatted prompt length: %zu chars\n", formatted.size());

    mtmd_input_text text_input;
    text_input.text          = formatted.c_str();
    text_input.add_special   = add_bos;
    text_input.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = full_bitmaps.c_ptr();
    int32_t tok_res = mtmd_tokenize(ctx.ctx_vision.get(),
                                     chunks.ptr.get(),
                                     &text_input,
                                     bitmaps_c_ptr.data(),
                                     bitmaps_c_ptr.size());
    if (tok_res != 0) {
        LOG_ERR("Failed to tokenize prompt (error %d)\n", tok_res);
        return result;
    }

    // Phase 3: Walk chunks
    const float skip_threshold = 0.05f;

    visual_token_map vis_token_map;

    attention_capture_state local_capture_state = {};
    attention_capture_state & capture_state = ext_capture_state ? *ext_capture_state : local_capture_state;
    capture_state.target_layer = sparams.prune_layer;
    capture_state.sample_interval = sparams.attn_sample_interval;
    capture_state.current_gen_step = 0;
    capture_state.should_capture = false;
    capture_state.token_map = &vis_token_map;
    capture_state.visual_pos_min = 0;
    capture_state.visual_pos_max = 0;
    capture_state.text_pos_start = 0;
    capture_state.initialized = false;
    capture_state.prefill_mode = false;

    int64_t t_encode_total_start = ggml_time_ms();
    size_t n_chunks = chunks.size();
    int image_idx = 0;
    bool encoding_failed = false;
    int n_kv_cells_used = 0;  // running count of actual KV cache cells used

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool is_last_chunk = (ci == n_chunks - 1);

            // Enable prefill attention capture for the last text chunk (user prompt)
            // By this point, all image tokens and their kv_cell_idx values are in the visual token map
            if (is_last_chunk && (sparams.analyze_attention_correlation || sparams.enable_postprefill_eviction) && !vis_token_map.tokens.empty()) {
                if (ext_qcur_state) {
                    // Flash-compatible mode: capture Qcur for CPU Q@K^T
                    ext_qcur_state->should_capture = true;
                    ext_qcur_state->captured = false;
                } else {
                    // Legacy mode: capture kq_soft_max
                    capture_state.initialized = true;
                    capture_state.should_capture = true;
                    capture_state.prefill_mode = true;
                }
            }

            llama_pos new_n_past = ctx.n_past;
            int32_t ret = mtmd_helper_eval_chunk_single(
                ctx.ctx_vision.get(), ctx.lctx, chunk,
                ctx.n_past, 0, ctx.n_batch, is_last_chunk, &new_n_past);

            // Disable prefill capture after eval
            if (is_last_chunk && capture_state.prefill_mode) {
                capture_state.prefill_mode = false;
                capture_state.should_capture = false;
            }
            if (is_last_chunk && ext_qcur_state) {
                ext_qcur_state->should_capture = false;  // ensure disabled after capture
            }

            if (ret != 0) {
                LOG_ERR("Failed to evaluate text chunk %zu (context full?)\n", ci);
                encoding_failed = true; break;
            }
            // For text, each token = 1 KV cell (1:1 mapping with n_past)
            n_kv_cells_used += (int)(new_n_past - ctx.n_past);
            ctx.n_past = new_n_past;
            continue;
        }

        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (!image_tokens) {
            LOG_ERR("Image tokens null for chunk %zu\n", ci);
            encoding_failed = true; break;
        }

        int chunk_nx = (int)mtmd_image_tokens_get_nx(image_tokens);
        int chunk_ny = (int)mtmd_image_tokens_get_ny(image_tokens);
        int n_tokens_chunk = chunk_nx * chunk_ny;

        int img_w = (int)full_bitmaps.entries[image_idx].nx();
        int img_h = (int)full_bitmaps.entries[image_idx].ny();

        inference_result::per_frame_cache_info fci;
        fci.frame_index = (image_idx < (int)spatial_analyses.size())
            ? spatial_analyses[image_idx].frame_index : (int)(image_idx + 1);

        frame_crop_info crop_ci;
        crop_ci.frame_index = fci.frame_index;
        crop_ci.orig_w = img_w;
        crop_ci.orig_h = img_h;
        crop_ci.estimated_tokens_full = n_tokens_chunk;
        crop_ci.crop_x = 0; crop_ci.crop_y = 0;
        crop_ci.crop_w = img_w; crop_ci.crop_h = img_h;
        crop_ci.was_cropped = false;
        crop_ci.estimated_tokens_crop = n_tokens_chunk;

        bool can_use_cache = cache.valid && keyframe_chunk_ptr
                             && (chunk_nx == nx_full) && (chunk_ny == ny_full);

        llama_pos frame_pos_start = ctx.n_past;

        // Track which token indices survived pre-LLM pruning (empty = all survived)
        std::vector<int> pre_llm_surviving_indices;

        // For correlation analysis: store pre-LLM scores for ALL tokens in this frame
        // (scored_token has token_idx and score)
        std::vector<pre_llm_token_scorer::scored_token> pre_llm_all_scores;

        if (image_idx == 0) {
            // KEYFRAME
            fci.is_keyframe = true;
            fci.is_full_encode = true;

            // Skip ViT capture for keyframe (frame 0) — no motion reference available
            // ViT capture is only needed for non-keyframes where motion data exists

            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                LOG_ERR("Failed to encode keyframe\n");
                encoding_failed = true; break;
            }
            fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

            if (vit_capture_state) vit_capture_state->active = false;

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

            nx_full = chunk_nx;
            ny_full = chunk_ny;
            cache.allocate(nx_full, ny_full, n_embd);
            cache.store_full(embd);

            // Score all keyframe tokens for correlation analysis
            if (sparams.analyze_attention_correlation && image_idx < (int)spatial_analyses.size()) {
                vit_attention_capture_state kf_vit;
                if (vit_capture_state) kf_vit = *vit_capture_state;
                else kf_vit.prepare(n_tokens_chunk);

                pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                            sparams.pre_llm_weight_norm,
                                            sparams.pre_llm_weight_attn,
                                            sparams.pre_llm_keep_ratio,
                                            sparams.pre_llm_strategy);
                pre_llm_all_scores = scorer.score_all(
                    cache.embeddings.data(), n_tokens_chunk, n_embd,
                    chunk_nx, chunk_ny, spatial_analyses[image_idx],
                    kf_vit, sparams.patch_grid);
            }

            keyframe_chunk_ptr.reset(mtmd_input_chunk_copy(chunk));

            fci.n_tokens_encoded = n_tokens_chunk;
            fci.n_tokens_cached  = 0;
            fci.used_cache = false;

            llama_pos new_n_past = ctx.n_past;
            ret = mtmd_helper_decode_image_chunk(
                ctx.ctx_vision.get(), ctx.lctx, chunk,
                cache.embeddings.data(), ctx.n_past, 0, ctx.n_batch, &new_n_past);
            if (ret != 0) {
                LOG_ERR("Failed to decode keyframe embeddings (context full? n_past=%d)\n", (int)ctx.n_past);
                encoding_failed = true; break;
            }
            ctx.n_past = new_n_past;

            // Capture KV cache snapshot for CUDA reuse (if enabled)
            if (sparams.enable_kv_cache_reuse) {
                // After keyframe decode, the tokens occupy sequential KV cells
                // starting from n_kv_cells_used (which tracks the running cell count)
                std::vector<int32_t> kf_cells(n_tokens_chunk);
                for (int i = 0; i < n_tokens_chunk; i++) {
                    kf_cells[i] = n_kv_cells_used + i;
                }
                kv_snapshot.capture(frame_pos_start, chunk_nx, chunk_ny,
                                    kf_cells.data(), n_tokens_chunk);
                LOG_INF("  KV snapshot captured: %d cells at temporal_pos=%d\n",
                        n_tokens_chunk, (int)frame_pos_start);
            }

            LOG_INF("  Frame %d: KEYFRAME (%dx%d, %dx%d grid, %d tokens, encode %.1f ms)\n",
                    fci.frame_index, img_w, img_h, chunk_nx, chunk_ny,
                    n_tokens_chunk, fci.encode_time_ms);

        } else if (can_use_cache && image_idx < (int)spatial_analyses.size()) {
            // PER-PATCH CACHE
            fci.is_keyframe = false;

            auto patch_active = build_patch_active_mask(
                spatial_analyses[image_idx], sparams.patch_grid, chunk_nx, chunk_ny,
                sparams.patch_reuse_motion);

            int n_active = 0;
            for (bool a : patch_active) { if (a) n_active++; }
            float active_ratio = (float)n_active / n_tokens_chunk;

            if (active_ratio < skip_threshold) {
                // SKIP ViT: reuse entire cache
                fci.is_full_encode = false;
                fci.used_cache = true;
                fci.encode_time_ms = 0.0f;
                fci.n_tokens_encoded = 0;
                fci.n_tokens_cached  = n_tokens_chunk;

                llama_pos new_n_past = ctx.n_past;
                int32_t ret = 0;
                bool used_kv_reuse = false;

#ifdef HAVE_CUDA_KV_REUSE
                // Try CUDA KV cache reuse: GPU-copy K/V from keyframe with RoPE correction
                if (sparams.enable_kv_cache_reuse && kv_snapshot.valid
                    && chunk_nx == kv_snapshot.nx && chunk_ny == kv_snapshot.ny
                    && !sparams.enable_pre_llm_pruning) {
                    int64_t t_reuse = ggml_time_ms();

                    // All tokens are static in the SKIP path
                    std::vector<int> all_indices(n_tokens_chunk);
                    for (int i = 0; i < n_tokens_chunk; i++) all_indices[i] = i;

                    std::vector<int32_t> new_cells;
                    int reused = perform_kv_cache_reuse(
                        ctx.lctx, model, kv_snapshot,
                        0, // seq_id
                        ctx.n_past,
                        chunk_nx, chunk_ny,
                        n_tokens_chunk, all_indices.data(),
                        new_cells);

                    if (reused > 0) {
                        used_kv_reuse = true;
                        new_n_past = ctx.n_past + std::max(chunk_nx, chunk_ny);
                        float reuse_ms = (float)(ggml_time_ms() - t_reuse);
                        LOG_INF("  Frame %d: KV REUSE (%.0f%% active < %.0f%% threshold, "
                                "%d tokens reused via CUDA copy, %.1f ms)\n",
                                fci.frame_index, active_ratio * 100.0f,
                                skip_threshold * 100.0f, reused, reuse_ms);
                    } else {
                        LOG_WRN("  KV cache reuse failed, falling back to decode\n");
                    }
                }
#endif // HAVE_CUDA_KV_REUSE

                if (!used_kv_reuse) {
                    if (sparams.enable_pre_llm_pruning && image_idx > 0 && image_idx < (int)spatial_analyses.size()) {
                        // Pre-LLM pruning: score with motion + norm (no ViT attn for skip path)
                        vit_attention_capture_state empty_vit;
                        empty_vit.prepare(n_tokens_chunk);
                        pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                    sparams.pre_llm_weight_norm,
                                                    0.0f,  // no ViT attn for skip path
                                                    sparams.pre_llm_keep_ratio,
                                                    sparams.pre_llm_strategy);
                        auto selected = scorer.score_and_select(
                            cache.embeddings.data(), n_tokens_chunk, n_embd,
                            chunk_nx, chunk_ny, spatial_analyses[image_idx],
                            empty_vit, sparams.patch_grid, image_idx);
                        fci.n_tokens_pre_llm_pruned = n_tokens_chunk - (int)selected.size();
                        fci.n_tokens_injected = (int)selected.size();
                        result.pre_llm_total_tokens_before += n_tokens_chunk;
                        result.pre_llm_total_tokens_injected += (int)selected.size();
                        result.pre_llm_total_tokens_pruned += fci.n_tokens_pre_llm_pruned;
                        for (const auto & s : selected) pre_llm_surviving_indices.push_back(s.token_idx);

                        ret = inject_pruned_visual_tokens(
                            ctx.lctx, ctx.ctx_vision.get(),
                            cache.embeddings.data(), selected,
                            chunk_nx, chunk_ny, n_embd,
                            ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    } else {
                        // Score all tokens for correlation analysis (without pruning)
                        if (sparams.analyze_attention_correlation && image_idx < (int)spatial_analyses.size()) {
                            vit_attention_capture_state empty_vit;
                            empty_vit.prepare(n_tokens_chunk);
                            pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                        sparams.pre_llm_weight_norm,
                                                        0.0f,
                                                        sparams.pre_llm_keep_ratio,
                                                        sparams.pre_llm_strategy);
                            pre_llm_all_scores = scorer.score_all(
                                cache.embeddings.data(), n_tokens_chunk, n_embd,
                                chunk_nx, chunk_ny, spatial_analyses[image_idx],
                                empty_vit, sparams.patch_grid);
                        }
                        ret = mtmd_helper_decode_image_chunk(
                            ctx.ctx_vision.get(), ctx.lctx,
                            keyframe_chunk_ptr.get(),
                            cache.embeddings.data(),
                            ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    }
                    if (ret != 0) {
                        LOG_ERR("Failed to decode cached frame %d (context full? n_past=%d)\n", fci.frame_index, (int)ctx.n_past);
                        encoding_failed = true; break;
                    }
                }
                ctx.n_past = new_n_past;

                crop_ci.estimated_tokens_crop = 0;
                crop_ci.was_cropped = true;

                if (used_kv_reuse) {
                    // Already logged above
                } else if (fci.n_tokens_injected > 0) {
                    LOG_INF("  Frame %d: SKIP ViT + PRE-LLM PRUNING (%.0f%% active < %.0f%% threshold, "
                            "0/%d tokens encoded, all cached, %d/%d injected)\n",
                            fci.frame_index, active_ratio * 100.0f,
                            skip_threshold * 100.0f, n_tokens_chunk,
                            fci.n_tokens_injected, n_tokens_chunk);
                } else {
                    LOG_INF("  Frame %d: SKIP ViT (%.0f%% active < %.0f%% threshold, "
                            "0/%d tokens encoded, all cached)\n",
                            fci.frame_index, active_ratio * 100.0f,
                            skip_threshold * 100.0f, n_tokens_chunk);
                }

            } else {
                // CROP-ENCODE active region, then SELECTIVE splice
                int tx_min = chunk_nx, tx_max = -1;
                int ty_min = chunk_ny, ty_max = -1;
                for (int ty = 0; ty < chunk_ny; ty++) {
                    for (int tx = 0; tx < chunk_nx; tx++) {
                        if (patch_active[ty * chunk_nx + tx]) {
                            tx_min = std::min(tx_min, tx);
                            tx_max = std::max(tx_max, tx);
                            ty_min = std::min(ty_min, ty);
                            ty_max = std::max(ty_max, ty);
                        }
                    }
                }

                int bb_nx = tx_max - tx_min + 1;
                int bb_ny = ty_max - ty_min + 1;
                int bb_tokens = bb_nx * bb_ny;

                float inv_sx = (float)img_w / (chunk_nx * 28);
                float inv_sy = (float)img_h / (chunk_ny * 28);
                int raw_cx = (int)(tx_min * 28 * inv_sx);
                int raw_cy = (int)(ty_min * 28 * inv_sy);
                int raw_cw = (int)(bb_nx * 28 * inv_sx);
                int raw_ch = (int)(bb_ny * 28 * inv_sy);

                raw_cx = std::max(0, std::min(raw_cx, img_w - 1));
                raw_cy = std::max(0, std::min(raw_cy, img_h - 1));
                raw_cw = std::min(raw_cw, img_w - raw_cx);
                raw_ch = std::min(raw_ch, img_h - raw_cy);

                frame_crop_info tmp_ci;
                tmp_ci.crop_x = raw_cx; tmp_ci.crop_y = raw_cy;
                tmp_ci.crop_w = raw_cw; tmp_ci.crop_h = raw_ch;
                mtmd::bitmap cropped_bmp = crop_bitmap(full_bitmaps.entries[image_idx], tmp_ci);

                std::string crop_marker(mtmd_default_marker());
                mtmd_input_text crop_text;
                crop_text.text          = crop_marker.c_str();
                crop_text.add_special   = false;
                crop_text.parse_special = true;

                mtmd::input_chunks crop_chunks(mtmd_input_chunks_init());
                const mtmd_bitmap * crop_bmp_ptr = cropped_bmp.ptr.get();
                int32_t crop_tok_res = mtmd_tokenize(ctx.ctx_vision.get(),
                                                      crop_chunks.ptr.get(),
                                                      &crop_text,
                                                      &crop_bmp_ptr, 1);

                const mtmd_input_chunk * crop_chunk = nullptr;
                int crop_enc_nx = 0, crop_enc_ny = 0;
                if (crop_tok_res == 0) {
                    for (size_t k = 0; k < crop_chunks.size(); k++) {
                        if (mtmd_input_chunk_get_type(crop_chunks[k]) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                            crop_chunk = crop_chunks[k];
                            break;
                        }
                    }
                    if (crop_chunk) {
                        const auto cit = mtmd_input_chunk_get_tokens_image(crop_chunk);
                        crop_enc_nx = (int)mtmd_image_tokens_get_nx(cit);
                        crop_enc_ny = (int)mtmd_image_tokens_get_ny(cit);
                    }
                }

                bool crop_ok = crop_chunk && (crop_enc_nx == bb_nx) && (crop_enc_ny == bb_ny);

                if (crop_ok) {
                    fci.is_full_encode = false;

                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), crop_chunk);
                    if (ret != 0) {
                        LOG_ERR("Failed to encode crop for frame %d\n", fci.frame_index);
                        crop_ok = false;
                    } else {
                        fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

                        float * crop_embd = mtmd_get_output_embd(ctx.ctx_vision.get());

                        int64_t t_splice = ggml_time_ms();
                        int n_updated = 0, n_kept_in_bb = 0;
                        for (int cy = 0; cy < bb_ny; cy++) {
                            for (int cx = 0; cx < bb_nx; cx++) {
                                int full_tx = tx_min + cx;
                                int full_ty = ty_min + cy;
                                int full_idx = full_ty * chunk_nx + full_tx;
                                if (patch_active[full_idx]) {
                                    memcpy(cache.at(full_ty, full_tx),
                                           crop_embd + (cy * bb_nx + cx) * n_embd,
                                           n_embd * sizeof(float));
                                    n_updated++;
                                } else {
                                    n_kept_in_bb++;
                                }
                            }
                        }
                        int n_kept_outside = n_tokens_chunk - bb_tokens;
                        int n_kept_total = n_kept_outside + n_kept_in_bb;
                        fci.splice_time_ms = (float)(ggml_time_ms() - t_splice);

                        fci.used_cache = true;
                        fci.n_tokens_encoded = n_updated;
                        fci.n_tokens_cached  = n_kept_total;
                        cache.valid = true;

                        llama_pos new_n_past = ctx.n_past;
                        bool used_hybrid_kv_reuse = false;
                        int32_t ret = 0;

#ifdef HAVE_CUDA_KV_REUSE
                        // Hybrid KV reuse: CUDA-copy static patches, llama_decode only active patches
                        if (sparams.enable_kv_cache_reuse && kv_snapshot.valid
                            && chunk_nx == kv_snapshot.nx && chunk_ny == kv_snapshot.ny
                            && !sparams.enable_pre_llm_pruning
                            && n_active < n_tokens_chunk) {  // skip if all active (no benefit)
                            int64_t t_hybrid = ggml_time_ms();

                            // Partition into static and active token indices
                            std::vector<int> static_indices;
                            std::vector<int> active_indices;
                            for (int i = 0; i < n_tokens_chunk; i++) {
                                if (patch_active[i]) {
                                    active_indices.push_back(i);
                                } else {
                                    static_indices.push_back(i);
                                }
                            }

                            // Phase 1: CUDA-copy static tokens from keyframe snapshot
                            std::vector<int32_t> static_new_cells;
                            int reused = perform_kv_cache_reuse(
                                ctx.lctx, model, kv_snapshot,
                                0,  // seq_id
                                ctx.n_past,
                                chunk_nx, chunk_ny,
                                (int)static_indices.size(),
                                static_indices.data(),
                                static_new_cells);

                            if (reused > 0) {
                                // Phase 2: llama_decode only active tokens
                                int n_act = (int)active_indices.size();

                                // Build subset embeddings from cache
                                std::vector<float> active_embd(n_act * n_embd);
                                for (int i = 0; i < n_act; i++) {
                                    memcpy(active_embd.data() + i * n_embd,
                                           cache.embeddings.data() + active_indices[i] * n_embd,
                                           n_embd * sizeof(float));
                                }

                                // Build M-RoPE positions (4 dims per token)
                                std::vector<llama_pos> pos_active(n_act * 4, 0);
                                for (int i = 0; i < n_act; i++) {
                                    int idx = active_indices[i];
                                    int grid_y = idx / chunk_nx;
                                    int grid_x = idx % chunk_nx;
                                    pos_active[i]              = ctx.n_past;              // temporal
                                    pos_active[i + n_act]      = ctx.n_past + grid_y;     // y
                                    pos_active[i + n_act * 2]  = ctx.n_past + grid_x;     // x
                                    pos_active[i + n_act * 3]  = 0;                        // unused
                                }

                                // Batch metadata
                                std::vector<int32_t> n_seq_id_vec(n_act, 1);
                                std::vector<llama_seq_id> seq_id_0(1, 0);
                                std::vector<llama_seq_id *> seq_ids_vec(n_act, seq_id_0.data());
                                std::vector<int8_t> logits_vec(n_act, 0);

                                // Non-causal attention for image tokens
                                if (mtmd_decode_use_non_causal(ctx.ctx_vision.get())) {
                                    llama_set_causal_attn(ctx.lctx, false);
                                }

                                // Decode active tokens in sub-batches
                                int n_img_batches = (n_act + ctx.n_batch - 1) / ctx.n_batch;
                                for (int ib = 0; ib < n_img_batches; ib++) {
                                    int offset = ib * ctx.n_batch;
                                    int n_tokens_batch = std::min(ctx.n_batch, n_act - offset);

                                    // Build M-RoPE pos view for sub-batch
                                    std::vector<llama_pos> pos_view;
                                    pos_view.reserve(n_tokens_batch * 4);
                                    for (int d = 0; d < 4; d++) {
                                        size_t src_idx = d * n_act + offset;
                                        pos_view.insert(pos_view.end(),
                                            pos_active.data() + src_idx,
                                            pos_active.data() + src_idx + n_tokens_batch);
                                    }

                                    llama_batch sub_batch = {
                                        /*n_tokens       =*/ n_tokens_batch,
                                        /*tokens         =*/ nullptr,
                                        /*embd           =*/ active_embd.data() + offset * n_embd,
                                        /*pos            =*/ pos_view.data(),
                                        /*n_seq_id       =*/ n_seq_id_vec.data() + offset,
                                        /*seq_id         =*/ seq_ids_vec.data() + offset,
                                        /*logits         =*/ logits_vec.data() + offset,
                                    };

                                    ret = llama_decode(ctx.lctx, sub_batch);
                                    if (ret != 0) {
                                        LOG_ERR("Hybrid KV reuse: failed to decode active batch %d/%d\n", ib + 1, n_img_batches);
                                        break;
                                    }
                                }

                                if (mtmd_decode_use_non_causal(ctx.ctx_vision.get())) {
                                    llama_set_causal_attn(ctx.lctx, true);
                                }

                                if (ret == 0) {
                                    used_hybrid_kv_reuse = true;
                                    new_n_past = ctx.n_past + std::max(chunk_nx, chunk_ny);
                                    float hybrid_ms = (float)(ggml_time_ms() - t_hybrid);
                                    LOG_INF("  Frame %d: HYBRID KV REUSE (%d/%d active decoded, "
                                            "%d static CUDA-copied, %.1f ms total)\n",
                                            fci.frame_index, n_act, n_tokens_chunk,
                                            (int)static_indices.size(), hybrid_ms);
                                }
                            }

                            if (!used_hybrid_kv_reuse) {
                                LOG_WRN("  Hybrid KV reuse failed, falling back to full decode\n");
                            }
                        }
#endif // HAVE_CUDA_KV_REUSE

                        if (!used_hybrid_kv_reuse) {
                        if (sparams.enable_pre_llm_pruning && image_idx > 0 && image_idx < (int)spatial_analyses.size()) {
                            // Pre-LLM pruning on spliced cache (motion + norm, no ViT attn for crop path)
                            vit_attention_capture_state empty_vit;
                            empty_vit.prepare(n_tokens_chunk);
                            pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                        sparams.pre_llm_weight_norm,
                                                        0.0f,
                                                        sparams.pre_llm_keep_ratio,
                                                        sparams.pre_llm_strategy);
                            auto selected = scorer.score_and_select(
                                cache.embeddings.data(), n_tokens_chunk, n_embd,
                                chunk_nx, chunk_ny, spatial_analyses[image_idx],
                                empty_vit, sparams.patch_grid, image_idx);
                            fci.n_tokens_pre_llm_pruned = n_tokens_chunk - (int)selected.size();
                            fci.n_tokens_injected = (int)selected.size();
                            result.pre_llm_total_tokens_before += n_tokens_chunk;
                            result.pre_llm_total_tokens_injected += (int)selected.size();
                            result.pre_llm_total_tokens_pruned += fci.n_tokens_pre_llm_pruned;
                            for (const auto & s : selected) pre_llm_surviving_indices.push_back(s.token_idx);

                            ret = inject_pruned_visual_tokens(
                                ctx.lctx, ctx.ctx_vision.get(),
                                cache.embeddings.data(), selected,
                                chunk_nx, chunk_ny, n_embd,
                                ctx.n_past, 0, ctx.n_batch, &new_n_past);
                        } else {
                            // Score all tokens for correlation analysis (without pruning)
                            if (sparams.analyze_attention_correlation && image_idx < (int)spatial_analyses.size()) {
                                vit_attention_capture_state empty_vit;
                                empty_vit.prepare(n_tokens_chunk);
                                pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                            sparams.pre_llm_weight_norm,
                                                            0.0f,
                                                            sparams.pre_llm_keep_ratio,
                                                            sparams.pre_llm_strategy);
                                pre_llm_all_scores = scorer.score_all(
                                    cache.embeddings.data(), n_tokens_chunk, n_embd,
                                    chunk_nx, chunk_ny, spatial_analyses[image_idx],
                                    empty_vit, sparams.patch_grid);
                            }
                            ret = mtmd_helper_decode_image_chunk(
                                ctx.ctx_vision.get(), ctx.lctx,
                                keyframe_chunk_ptr.get(),
                                cache.embeddings.data(),
                                ctx.n_past, 0, ctx.n_batch, &new_n_past);
                        }
                        if (ret != 0) {
                            LOG_ERR("Failed to decode frame %d (context full? n_past=%d)\n", fci.frame_index, (int)ctx.n_past);
                            encoding_failed = true; break;
                        }
                        } // !used_hybrid_kv_reuse
                        ctx.n_past = new_n_past;

                        crop_ci.crop_x = raw_cx; crop_ci.crop_y = raw_cy;
                        crop_ci.crop_w = raw_cw; crop_ci.crop_h = raw_ch;
                        crop_ci.was_cropped = true;
                        crop_ci.estimated_tokens_crop = bb_tokens;

                        if (used_hybrid_kv_reuse) {
                            // Already logged above in the hybrid path
                        } else if (fci.n_tokens_injected > 0) {
                            LOG_INF("  Frame %d: CROP+SPLICE + PRE-LLM PRUNING (%d/%d patches updated, %d cached, "
                                    "crop %dx%d tokens, %.0f%% active, encode %.1f ms, splice %.1f ms, "
                                    "%d/%d injected)\n",
                                    fci.frame_index, n_updated, n_tokens_chunk, n_kept_total,
                                    bb_nx, bb_ny, active_ratio * 100.0f,
                                    fci.encode_time_ms, fci.splice_time_ms,
                                    fci.n_tokens_injected, n_tokens_chunk);
                        } else {
                            LOG_INF("  Frame %d: CROP+SPLICE (%d/%d patches updated, %d cached "
                                    "[%d outside BB + %d inactive in BB], crop %dx%d tokens, "
                                    "%.0f%% active, encode %.1f ms, splice %.1f ms)\n",
                                    fci.frame_index, n_updated, n_tokens_chunk, n_kept_total,
                                    n_kept_outside, n_kept_in_bb, bb_nx, bb_ny,
                                    active_ratio * 100.0f, fci.encode_time_ms, fci.splice_time_ms);
                        }
                    }
                }

                if (!crop_ok) {
                    // FALLBACK: encode full frame, selective update
                    fci.is_full_encode = true;

                    // Activate ViT capture if pre-LLM pruning or correlation analysis enabled
                    if (vit_capture_state && (sparams.enable_pre_llm_pruning || sparams.analyze_attention_correlation) && image_idx > 0) {
                        vit_capture_state->prepare(n_tokens_chunk);
                        vit_capture_state->active = true;
                    }

                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
                    if (ret != 0) {
                        LOG_ERR("Failed to encode frame %d (fallback)\n", fci.frame_index);
                        if (vit_capture_state) vit_capture_state->active = false;
                        encoding_failed = true; break;
                    }
                    fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

                    if (vit_capture_state) vit_capture_state->active = false;

                    float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

                    int64_t t_splice = ggml_time_ms();
                    int n_updated = 0, n_kept = 0;
                    cache.selective_update(embd, patch_active, n_updated, n_kept);
                    fci.splice_time_ms = (float)(ggml_time_ms() - t_splice);

                    fci.used_cache = (n_kept > 0);
                    fci.n_tokens_encoded = n_updated;
                    fci.n_tokens_cached  = n_kept;

                    llama_pos new_n_past = ctx.n_past;

                    if (sparams.enable_pre_llm_pruning && image_idx > 0 && image_idx < (int)spatial_analyses.size()) {
                        // Pre-LLM pruning with ViT attention from full encode
                        vit_attention_capture_state fallback_vit;
                        if (vit_capture_state) {
                            fallback_vit = *vit_capture_state;
                        } else {
                            fallback_vit.prepare(n_tokens_chunk);
                        }
                        pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                    sparams.pre_llm_weight_norm,
                                                    sparams.pre_llm_weight_attn,
                                                    sparams.pre_llm_keep_ratio,
                                                    sparams.pre_llm_strategy);
                        auto selected = scorer.score_and_select(
                            cache.embeddings.data(), n_tokens_chunk, n_embd,
                            chunk_nx, chunk_ny, spatial_analyses[image_idx],
                            fallback_vit, sparams.patch_grid, image_idx);
                        fci.n_tokens_pre_llm_pruned = n_tokens_chunk - (int)selected.size();
                        fci.n_tokens_injected = (int)selected.size();
                        result.pre_llm_total_tokens_before += n_tokens_chunk;
                        result.pre_llm_total_tokens_injected += (int)selected.size();
                        result.pre_llm_total_tokens_pruned += fci.n_tokens_pre_llm_pruned;
                        for (const auto & s : selected) pre_llm_surviving_indices.push_back(s.token_idx);

                        ret = inject_pruned_visual_tokens(
                            ctx.lctx, ctx.ctx_vision.get(),
                            cache.embeddings.data(), selected,
                            chunk_nx, chunk_ny, n_embd,
                            ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    } else {
                        // Score all tokens for correlation analysis (without pruning)
                        if (sparams.analyze_attention_correlation && image_idx < (int)spatial_analyses.size()) {
                            vit_attention_capture_state fallback_vit;
                            if (vit_capture_state) fallback_vit = *vit_capture_state;
                            else fallback_vit.prepare(n_tokens_chunk);
                            pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                        sparams.pre_llm_weight_norm,
                                                        sparams.pre_llm_weight_attn,
                                                        sparams.pre_llm_keep_ratio,
                                                        sparams.pre_llm_strategy);
                            pre_llm_all_scores = scorer.score_all(
                                cache.embeddings.data(), n_tokens_chunk, n_embd,
                                chunk_nx, chunk_ny, spatial_analyses[image_idx],
                                fallback_vit, sparams.patch_grid);
                        }
                        ret = mtmd_helper_decode_image_chunk(
                            ctx.ctx_vision.get(), ctx.lctx,
                            keyframe_chunk_ptr.get(),
                            cache.embeddings.data(),
                            ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    }
                    if (ret != 0) {
                        LOG_ERR("Failed to decode frame %d (fallback, context full? n_past=%d)\n", fci.frame_index, (int)ctx.n_past);
                        encoding_failed = true; break;
                    }
                    ctx.n_past = new_n_past;

                    crop_ci.estimated_tokens_crop = n_updated;
                    crop_ci.was_cropped = (n_kept > 0);

                    if (fci.n_tokens_injected > 0) {
                        LOG_INF("  Frame %d: FALLBACK full encode + PRE-LLM PRUNING (%d/%d updated, %d cached, "
                                "encode %.1f ms, %d/%d injected)\n",
                                fci.frame_index, n_updated, n_tokens_chunk, n_kept,
                                fci.encode_time_ms, fci.n_tokens_injected, n_tokens_chunk);
                    } else {
                        LOG_INF("  Frame %d: FALLBACK full encode (%d/%d updated, %d cached, "
                                "crop dim mismatch %dx%d vs expected %dx%d, encode %.1f ms)\n",
                                fci.frame_index, n_updated, n_tokens_chunk, n_kept,
                                crop_enc_nx, crop_enc_ny, bb_nx, bb_ny,
                                fci.encode_time_ms);
                    }
                }
            }

        } else {
            // NO CACHE AVAILABLE: full encode
            fci.is_keyframe = false;
            fci.is_full_encode = true;

            // Activate ViT capture if pre-LLM pruning or correlation analysis enabled
            if (vit_capture_state && (sparams.enable_pre_llm_pruning || sparams.analyze_attention_correlation) && image_idx > 0) {
                vit_capture_state->prepare(n_tokens_chunk);
                vit_capture_state->active = true;
            }

            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                LOG_ERR("Failed to encode frame %d\n", fci.frame_index);
                if (vit_capture_state) vit_capture_state->active = false;
                encoding_failed = true; break;
            }
            fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

            if (vit_capture_state) vit_capture_state->active = false;

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

            if (chunk_nx == nx_full && chunk_ny == ny_full) {
                cache.store_full(embd);
            }

            fci.n_tokens_encoded = n_tokens_chunk;
            fci.n_tokens_cached  = 0;
            fci.used_cache = false;

            llama_pos new_n_past = ctx.n_past;
            const float * decode_embd = (chunk_nx == nx_full && chunk_ny == ny_full)
                ? cache.embeddings.data() : embd;

            if (sparams.enable_pre_llm_pruning && image_idx > 0 && image_idx < (int)spatial_analyses.size()) {
                // Pre-LLM pruning with ViT attention from full encode
                vit_attention_capture_state nocache_vit;
                if (vit_capture_state) {
                    nocache_vit = *vit_capture_state;
                } else {
                    nocache_vit.prepare(n_tokens_chunk);
                }
                pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                            sparams.pre_llm_weight_norm,
                                            sparams.pre_llm_weight_attn,
                                            sparams.pre_llm_keep_ratio,
                                            sparams.pre_llm_strategy);
                auto selected = scorer.score_and_select(
                    decode_embd, n_tokens_chunk, n_embd,
                    chunk_nx, chunk_ny, spatial_analyses[image_idx],
                    nocache_vit, sparams.patch_grid, image_idx);
                fci.n_tokens_pre_llm_pruned = n_tokens_chunk - (int)selected.size();
                fci.n_tokens_injected = (int)selected.size();
                result.pre_llm_total_tokens_before += n_tokens_chunk;
                result.pre_llm_total_tokens_injected += (int)selected.size();
                result.pre_llm_total_tokens_pruned += fci.n_tokens_pre_llm_pruned;
                for (const auto & s : selected) pre_llm_surviving_indices.push_back(s.token_idx);

                ret = inject_pruned_visual_tokens(
                    ctx.lctx, ctx.ctx_vision.get(),
                    decode_embd, selected,
                    chunk_nx, chunk_ny, n_embd,
                    ctx.n_past, 0, ctx.n_batch, &new_n_past);
            } else {
                // Score all tokens for correlation analysis (without pruning)
                if (sparams.analyze_attention_correlation && image_idx < (int)spatial_analyses.size()) {
                    vit_attention_capture_state nocache_vit;
                    if (vit_capture_state) nocache_vit = *vit_capture_state;
                    else nocache_vit.prepare(n_tokens_chunk);
                    pre_llm_token_scorer scorer(sparams.pre_llm_weight_motion,
                                                sparams.pre_llm_weight_norm,
                                                sparams.pre_llm_weight_attn,
                                                sparams.pre_llm_keep_ratio,
                                                sparams.pre_llm_strategy);
                    pre_llm_all_scores = scorer.score_all(
                        decode_embd, n_tokens_chunk, n_embd,
                        chunk_nx, chunk_ny, spatial_analyses[image_idx],
                        nocache_vit, sparams.patch_grid);
                }
                const mtmd_input_chunk * decode_chunk = keyframe_chunk_ptr ? keyframe_chunk_ptr.get() : chunk;
                ret = mtmd_helper_decode_image_chunk(
                    ctx.ctx_vision.get(), ctx.lctx, decode_chunk,
                    const_cast<float *>(decode_embd),
                    ctx.n_past, 0, ctx.n_batch, &new_n_past);
            }
            if (ret != 0) {
                LOG_ERR("Failed to decode frame %d (context full? n_past=%d)\n", fci.frame_index, (int)ctx.n_past);
                encoding_failed = true; break;
            }
            ctx.n_past = new_n_past;

            if (fci.n_tokens_injected > 0) {
                LOG_INF("  Frame %d: FULL ENCODE + PRE-LLM PRUNING (no spatial data, %d tokens, "
                        "%.1f ms, %d/%d injected)\n",
                        fci.frame_index, n_tokens_chunk, fci.encode_time_ms,
                        fci.n_tokens_injected, n_tokens_chunk);
            } else {
                LOG_INF("  Frame %d: FULL ENCODE (no spatial data, %d tokens, %.1f ms)\n",
                        fci.frame_index, n_tokens_chunk, fci.encode_time_ms);
            }
        }

        result.frame_cache_infos.push_back(fci);
        result.total_tokens_encoded += fci.n_tokens_encoded;
        result.total_tokens_cached  += fci.n_tokens_cached;
        result.total_encode_time_ms += fci.encode_time_ms;
        crop_infos_out.push_back(crop_ci);

        // Populate visual token map for pruning, correlation analysis, post-prefill eviction, or KV cache reuse
        if (sparams.enable_pruning || sparams.enable_prefill_pruning || sparams.analyze_attention_correlation || sparams.enable_postprefill_eviction || sparams.enable_kv_cache_reuse) {
            llama_pos frame_n_pos = (llama_pos)mtmd_input_chunk_get_n_pos(chunk);
            visual_token_map::frame_range fr;
            fr.start_pos = frame_pos_start;
            fr.end_pos   = frame_pos_start + frame_n_pos;
            fr.nx = chunk_nx;
            fr.ny = chunk_ny;
            fr.token_start_idx = (int)vis_token_map.tokens.size();
            fr.token_count = chunk_nx * chunk_ny;
            vis_token_map.frame_ranges.push_back(fr);

            float default_motion = 0.0f;
            if (image_idx < (int)spatial_analyses.size()) {
                const auto & sa = spatial_analyses[image_idx];
                for (const auto & p : sa.patches) {
                    default_motion += p.mean_motion;
                }
                default_motion /= std::max(1, (int)sa.patches.size());
            }

            // Build set of surviving token indices for quick lookup
            std::set<int> surviving_set;
            if (!pre_llm_surviving_indices.empty()) {
                surviving_set.insert(pre_llm_surviving_indices.begin(),
                                     pre_llm_surviving_indices.end());
            }

            int frame_idx = (int)vis_token_map.frame_ranges.size() - 1;
            int frame_cell_start = n_kv_cells_used;  // KV cell index where this frame starts
            int actual_token_count = 0;
            for (int ty = 0; ty < chunk_ny; ty++) {
                for (int tx = 0; tx < chunk_nx; tx++) {
                    int tok_idx = ty * chunk_nx + tx;

                    // Skip tokens that were pruned pre-LLM
                    if (!surviving_set.empty() && surviving_set.find(tok_idx) == surviving_set.end()) {
                        continue;
                    }

                    llama_pos tok_pos = frame_pos_start;
                    llama_pos tok_ext_x = frame_pos_start + tx;
                    llama_pos tok_ext_y = frame_pos_start + ty;

                    int sa_gx = tx * sparams.patch_grid / chunk_nx;
                    int sa_gy = ty * sparams.patch_grid / chunk_ny;
                    sa_gx = std::min(sa_gx, sparams.patch_grid - 1);
                    sa_gy = std::min(sa_gy, sparams.patch_grid - 1);

                    float tok_motion = default_motion;
                    if (image_idx < (int)spatial_analyses.size()) {
                        const auto & sa = spatial_analyses[image_idx];
                        for (const auto & p : sa.patches) {
                            if (p.grid_x == sa_gx && p.grid_y == sa_gy) {
                                tok_motion = p.mean_motion;
                                break;
                            }
                        }
                    }

                    int cell_idx = frame_cell_start + actual_token_count;
                    vis_token_map.add_token(tok_pos, tok_ext_x, tok_ext_y,
                                            frame_idx, ty, tx, tok_motion,
                                            sa_gy, sa_gx, cell_idx);
                    actual_token_count++;
                }
            }
            n_kv_cells_used += actual_token_count;
            fr.token_count = actual_token_count;

            // Apply pre-LLM scores to token map entries for correlation analysis
            if (sparams.analyze_attention_correlation && !pre_llm_all_scores.empty()) {
                int n_keep_threshold = std::max(1, (int)(pre_llm_all_scores.size() * sparams.pre_llm_keep_ratio));

                // Sort by score to determine the threshold
                std::vector<float> sorted_scores;
                for (const auto & st : pre_llm_all_scores) {
                    sorted_scores.push_back(st.score);
                }
                std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<float>());
                float threshold = (n_keep_threshold < (int)sorted_scores.size())
                    ? sorted_scores[n_keep_threshold - 1] : 0.0f;

                // Build a map from token_idx -> score
                std::unordered_map<int, float> score_map;
                for (const auto & st : pre_llm_all_scores) {
                    score_map[st.token_idx] = st.score;
                }

                // Apply scores to the tokens just added to the map
                int token_start = fr.token_start_idx;
                int token_end = token_start + fr.token_count;
                int tok_linear = 0;
                for (int ty = 0; ty < chunk_ny; ty++) {
                    for (int tx = 0; tx < chunk_nx; tx++) {
                        int tok_idx = ty * chunk_nx + tx;
                        // Skip tokens pruned pre-LLM (they're not in the map)
                        if (!surviving_set.empty() && surviving_set.find(tok_idx) == surviving_set.end()) {
                            continue;
                        }
                        int map_idx = token_start + tok_linear;
                        if (map_idx < token_end && map_idx < (int)vis_token_map.tokens.size()) {
                            auto it = score_map.find(tok_idx);
                            if (it != score_map.end()) {
                                vis_token_map.tokens[map_idx].pre_llm_score = it->second;
                                vis_token_map.tokens[map_idx].would_prune = (it->second < threshold);
                            }
                        }
                        tok_linear++;
                    }
                }
            }
        }

        // Prefill-stage pruning
        if (sparams.enable_prefill_pruning && image_idx > 0 && !vis_token_map.tokens.empty()) {
            int frame_idx = (int)vis_token_map.frame_ranges.size() - 1;
            int64_t t_pp = ggml_time_ms();
            int n_pruned = do_prefill_motion_pruning(
                ctx.lctx, vis_token_map, frame_idx, sparams.prefill_keep_ratio);
            if (n_pruned > 0) {
                llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
            }
            result.n_tokens_pruned_prefill += n_pruned;
            result.prefill_pruning_overhead_ms += (float)(ggml_time_ms() - t_pp);
        }

        image_idx++;
    }

    result.encode_time_ms = (float)(ggml_time_ms() - t_encode_total_start);

    if (encoding_failed) {
        LOG_ERR("\nEncoding failed after %d/%zu frames (n_past=%d). "
                "Context window may be too small — try increasing -c.\n",
                image_idx, frame_paths.size(), (int)ctx.n_past);
        LOG_ERR("Continuing with %d successfully decoded frames.\n", image_idx);
    }

    // Estimate what full encoding would have cost
    if (!result.frame_cache_infos.empty()) {
        float keyframe_ms_per_token = 0.0f;
        for (const auto & fci : result.frame_cache_infos) {
            if (fci.is_keyframe && fci.n_tokens_encoded > 0) {
                keyframe_ms_per_token = fci.encode_time_ms / fci.n_tokens_encoded;
                break;
            }
        }
        int total_full_tokens = 0;
        for (const auto & fci : result.frame_cache_infos) {
            total_full_tokens += fci.n_tokens_encoded + fci.n_tokens_cached;
        }
        result.estimated_full_encode_time_ms = keyframe_ms_per_token * total_full_tokens;
    }

    LOG_INF("Embedding cache: %d tokens encoded, %d tokens cached (%.1f%% savings)\n",
            result.total_tokens_encoded, result.total_tokens_cached,
            (result.total_tokens_encoded + result.total_tokens_cached > 0)
                ? 100.0f * result.total_tokens_cached / (result.total_tokens_encoded + result.total_tokens_cached)
                : 0.0f);
    LOG_INF("Encode time: %.1f ms (estimated full: %.1f ms, speedup: %.2fx)\n",
            result.total_encode_time_ms, result.estimated_full_encode_time_ms,
            (result.total_encode_time_ms > 0)
                ? result.estimated_full_encode_time_ms / result.total_encode_time_ms
                : 1.0f);

    if (sparams.enable_pre_llm_pruning && result.pre_llm_total_tokens_before > 0) {
        float savings_pct = 100.0f * result.pre_llm_total_tokens_pruned / result.pre_llm_total_tokens_before;
        LOG_INF("Pre-LLM pruning: %d/%d tokens kept (%.1f%% pruned, keep_ratio=%.2f)\n",
                result.pre_llm_total_tokens_injected,
                result.pre_llm_total_tokens_before,
                savings_pct, sparams.pre_llm_keep_ratio);
    }

    // Phase 4: Generate tokens (with optional progressive pruning)
    if (sparams.enable_pruning && !vis_token_map.tokens.empty()) {
        capture_state.initialized = true;
        result.n_visual_tokens_initial = vis_token_map.count_active();
        LOG_INF("\n--- Progressive Visual Token Pruning enabled ---\n");
        LOG_INF("  Visual tokens: %d across %zu frames\n",
                result.n_visual_tokens_initial, vis_token_map.frame_ranges.size());
        LOG_INF("  Prune layer: %d, sample interval: %d\n",
                sparams.prune_layer, sparams.attn_sample_interval);
        LOG_INF("  Round 1 at step %d (patch_keep_ratio=%.2f, motion_weight=%.2f)\n",
                sparams.first_prune_step, sparams.patch_keep_ratio, sparams.motion_weight);
        LOG_INF("  Round 2 at step %d (token_keep_ratio=%.2f)\n",
                sparams.second_prune_step, sparams.token_keep_ratio);
    }

    // Enable attention capture for correlation analysis (capture LLM attention without pruning)
    if (sparams.analyze_attention_correlation) {
        if (vis_token_map.tokens.empty()) {
            LOG_ERR("WARNING: visual token map is empty for correlation analysis!\n");
        } else {
            capture_state.initialized = true;
            capture_state.sample_interval = 1;  // capture every step for correlation
            LOG_INF("\n--- Attention Correlation Analysis enabled ---\n");
            LOG_INF("  Visual tokens: %d across %zu frames\n",
                    vis_token_map.count_active(), vis_token_map.frame_ranges.size());
            LOG_INF("  Capturing LLM attention at layer %d every step\n", sparams.prune_layer);
            LOG_INF("  token_map ptr: %p, initialized: %d\n",
                    (void*)capture_state.token_map, capture_state.initialized);
        }
    }

    // ================================================================
    // Flash-compatible CPU Q@K^T attention computation
    // ================================================================
    if (ext_qcur_state && sparams.enable_postprefill_eviction && !vis_token_map.tokens.empty()) {
        auto * mem = llama_get_memory(ctx.lctx);
        int n_head_kv = llama_model_n_head_kv(llama_get_model(ctx.lctx));
        int64_t t_cpu_attn_start = ggml_time_ms();
        compute_cpu_attention_scores(*ext_qcur_state, vis_token_map, mem,
                                     sparams.prune_layer, n_kv_cells_used, n_head_kv);
        LOG_INF("CPU Q@K^T computation took %.1f ms\n", (float)(ggml_time_ms() - t_cpu_attn_start));
    }

    // ================================================================
    // Post-prefill KV Eviction
    // ================================================================
    if (sparams.enable_postprefill_eviction && !vis_token_map.tokens.empty()) {
        int64_t t_evict_start = ggml_time_ms();

        // Collect active (non-pruned) tokens with their prefill attention scores
        struct scored_token { int idx; float score; };
        std::vector<scored_token> active_tokens;
        for (int i = 0; i < (int)vis_token_map.tokens.size(); i++) {
            const auto & tok = vis_token_map.tokens[i];
            if (!tok.pruned) {
                active_tokens.push_back({i, tok.prefill_attn_score});
            }
        }

        int n_active = (int)active_tokens.size();
        result.postprefill_n_tokens_before = n_active;

        if (n_active > 1) {
            // Sort by score ascending (lowest attention first = evict first)
            std::sort(active_tokens.begin(), active_tokens.end(),
                      [](const scored_token & a, const scored_token & b) { return a.score < b.score; });

            int n_keep = std::max(1, (int)(n_active * sparams.postprefill_keep_ratio));
            int n_to_evict = n_active - n_keep;

            // Compute total attention before eviction
            float total_attn = 0.0f;
            for (const auto & st : active_tokens) {
                total_attn += st.score;
            }

            // Evict the bottom n_to_evict tokens
            auto * mem = llama_get_memory(ctx.lctx);
            int n_evicted = 0;
            for (int i = 0; i < n_to_evict; i++) {
                auto & tok = vis_token_map.tokens[active_tokens[i].idx];
                if (tok.pruned) continue;
                bool ok = llama_memory_seq_rm_pos_ext(mem, 0, tok.kv_pos, tok.ext_x, tok.ext_y);
                if (ok) {
                    tok.pruned = true;
                    n_evicted++;
                }
            }

            // Note: defer defrag to after the first decode step.
            // Running defrag here (between prefill and first sample) invalidates
            // the cached logits from prefill, causing the model to sample EOS.
            result.postprefill_needs_defrag = (n_evicted > 0);

            // Compute retained attention
            float kept_attn = 0.0f;
            for (const auto & st : active_tokens) {
                if (!vis_token_map.tokens[st.idx].pruned) {
                    kept_attn += st.score;
                }
            }
            float retained_pct = (total_attn > 1e-10f) ? 100.0f * kept_attn / total_attn : 0.0f;

            result.postprefill_n_tokens_evicted = n_evicted;
            result.postprefill_attn_retained_pct = retained_pct;
            result.postprefill_eviction_ms = (float)(ggml_time_ms() - t_evict_start);

            LOG_INF("\n--- Post-prefill eviction: kept %d/%d tokens, %.1f%% attention retained (%.1f ms) ---\n",
                    n_active - n_evicted, n_active, retained_pct, result.postprefill_eviction_ms);
        } else {
            LOG_INF("\n--- Post-prefill eviction: skipped (only %d active tokens) ---\n", n_active);
        }
    }

    // Disable the attention callback for decode if we only needed it for post-prefill eviction
    if (sparams.enable_postprefill_eviction && !sparams.enable_pruning && !sparams.analyze_attention_correlation) {
        capture_state.should_capture = false;
        capture_state.disabled = true;
        // Note: cannot re-enable flash attention at runtime — V cache layout is set at context creation.
        // The callback is disabled so it returns immediately, adding negligible overhead.
    }

    int64_t t_gen_start = ggml_time_ms();
    int64_t t_prune_total = 0;
    std::string generated_text;
    llama_tokens generated_tokens;

    for (int i = 0; i < n_predict; i++) {
        if ((sparams.enable_pruning || sparams.analyze_attention_correlation) && capture_state.initialized) {
            capture_state.current_gen_step = i;
            capture_state.should_capture = sparams.analyze_attention_correlation
                ? true  // capture every step for correlation
                : (i % sparams.attn_sample_interval == 0);
        }

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

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("Failed to decode token\n");
            break;
        }

        // Deferred defrag: compact KV cache after first decode consumes the prefill logits
        if (i == 0 && result.postprefill_needs_defrag) {
            auto * mem = llama_get_memory(ctx.lctx);
            llama_memory_defragment(mem, ctx.lctx);
            result.postprefill_needs_defrag = false;
            LOG_INF("Post-prefill KV defrag completed (compacted after first decode)\n");
        }

        // Progressive pruning checkpoints
        if (sparams.enable_pruning && capture_state.initialized) {
            if (i == sparams.first_prune_step) {
                int64_t t_prune = ggml_time_ms();
                result.n_tokens_pruned_round1 = do_patch_level_pruning(
                    ctx.lctx, vis_token_map, capture_state, sparams);
                result.n_visual_tokens_after_round1 = vis_token_map.count_active();
                llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
                t_prune_total += ggml_time_ms() - t_prune;
            }
            if (i == sparams.second_prune_step) {
                int64_t t_prune = ggml_time_ms();
                result.n_tokens_pruned_round2 = do_token_level_pruning(
                    ctx.lctx, vis_token_map, capture_state, sparams);
                result.n_visual_tokens_after_round2 = vis_token_map.count_active();
                llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
                t_prune_total += ggml_time_ms() - t_prune;
                capture_state.disabled = true;
            }
        }
    }
    printf("\n");

    result.generated_text    = generated_text;
    result.n_tokens_generated = (int)generated_tokens.size();
    result.inference_time_ms  = (float)(ggml_time_ms() - t_gen_start);
    result.pruning_overhead_ms = (float)t_prune_total;

    if (sparams.enable_pruning) {
        if (result.n_visual_tokens_after_round1 == 0 && result.n_tokens_pruned_round1 == 0) {
            result.n_visual_tokens_after_round1 = result.n_visual_tokens_initial;
        }
        if (result.n_visual_tokens_after_round2 == 0 && result.n_tokens_pruned_round2 == 0) {
            result.n_visual_tokens_after_round2 = result.n_visual_tokens_after_round1;
        }
    }

    LOG_INF("Generated %d tokens in %.1f ms (%.1f t/s)\n",
            result.n_tokens_generated, result.inference_time_ms,
            result.n_tokens_generated / (result.inference_time_ms / 1000.0f));

    if (sparams.enable_pruning && result.n_visual_tokens_initial > 0) {
        LOG_INF("Pruning summary: %d -> %d -> %d visual tokens (%.1f%% total reduction)\n",
                result.n_visual_tokens_initial,
                result.n_visual_tokens_after_round1,
                result.n_visual_tokens_after_round2,
                (1.0f - (float)result.n_visual_tokens_after_round2 / result.n_visual_tokens_initial) * 100.0f);
        LOG_INF("Pruning overhead: %.1f ms\n", result.pruning_overhead_ms);
    }

    // ================================================================
    // Attention Correlation Analysis
    // ================================================================
    if (sparams.analyze_attention_correlation && !vis_token_map.tokens.empty()) {
        LOG_INF("\n--- Attention Correlation Analysis ---\n");
        LOG_INF("  Token map: %zu tokens, %zu frame ranges\n",
                vis_token_map.tokens.size(), vis_token_map.frame_ranges.size());

        // Debug: count tokens with attn_samples > 0
        int n_with_attn = 0, n_with_score = 0;
        for (const auto & tok : vis_token_map.tokens) {
            if (tok.attn_samples > 0) n_with_attn++;
            if (tok.pre_llm_score > 0) n_with_score++;
        }
        LOG_INF("  Tokens with LLM attention: %d, with pre-LLM scores: %d\n",
                n_with_attn, n_with_score);

        // Collect tokens with both pre-LLM scores and LLM attention
        std::vector<float> pre_llm_scores_vec;
        std::vector<float> llm_attn_vec;
        std::vector<bool>  would_prune_vec;

        // Per-frame analysis
        for (size_t fi = 0; fi < vis_token_map.frame_ranges.size(); fi++) {
            const auto & fr = vis_token_map.frame_ranges[fi];
            std::vector<float> frame_prellm, frame_llm_attn;
            std::vector<bool>  frame_would_prune;

            for (int ti = fr.token_start_idx; ti < fr.token_start_idx + fr.token_count; ti++) {
                if (ti >= (int)vis_token_map.tokens.size()) break;
                const auto & tok = vis_token_map.tokens[ti];
                if (tok.pruned || tok.attn_samples == 0) continue;

                float avg_llm_attn = tok.attn_score / std::max(1, tok.attn_samples);
                frame_prellm.push_back(tok.pre_llm_score);
                frame_llm_attn.push_back(avg_llm_attn);
                frame_would_prune.push_back(tok.would_prune);

                pre_llm_scores_vec.push_back(tok.pre_llm_score);
                llm_attn_vec.push_back(avg_llm_attn);
                would_prune_vec.push_back(tok.would_prune);
            }

            // Per-frame Pearson correlation
            if (frame_prellm.size() >= 5) {
                inference_result::correlation_result::per_frame_corr pfc;
                pfc.frame_idx = fr.start_pos;
                pfc.n_tokens = (int)frame_prellm.size();

                // Pearson r
                float sum_x = 0, sum_y = 0;
                for (size_t i = 0; i < frame_prellm.size(); i++) {
                    sum_x += frame_prellm[i];
                    sum_y += frame_llm_attn[i];
                }
                float mean_x = sum_x / frame_prellm.size();
                float mean_y = sum_y / frame_prellm.size();
                float cov = 0, var_x = 0, var_y = 0;
                for (size_t i = 0; i < frame_prellm.size(); i++) {
                    float dx = frame_prellm[i] - mean_x;
                    float dy = frame_llm_attn[i] - mean_y;
                    cov += dx * dy;
                    var_x += dx * dx;
                    var_y += dy * dy;
                }
                float denom = std::sqrt(var_x * var_y);
                pfc.pearson_r = (denom > 1e-10f) ? (cov / denom) : 0.0f;

                // Split averages
                float sum_kept = 0, sum_pruned = 0;
                int n_kept = 0, n_pruned = 0;
                for (size_t i = 0; i < frame_prellm.size(); i++) {
                    if (frame_would_prune[i]) {
                        sum_pruned += frame_llm_attn[i];
                        n_pruned++;
                    } else {
                        sum_kept += frame_llm_attn[i];
                        n_kept++;
                    }
                }
                pfc.avg_attn_kept = (n_kept > 0) ? sum_kept / n_kept : 0.0f;
                pfc.avg_attn_pruned = (n_pruned > 0) ? sum_pruned / n_pruned : 0.0f;
                pfc.attn_ratio = (pfc.avg_attn_pruned > 1e-10f)
                    ? pfc.avg_attn_kept / pfc.avg_attn_pruned : 0.0f;

                // Per-frame prefill attention retention
                {
                    float frame_prefill_total = 0.0f, frame_prefill_kept = 0.0f;
                    for (int ti2 = fr.token_start_idx; ti2 < fr.token_start_idx + fr.token_count; ti2++) {
                        if (ti2 >= (int)vis_token_map.tokens.size()) break;
                        const auto & tok2 = vis_token_map.tokens[ti2];
                        if (tok2.pruned) continue;
                        frame_prefill_total += tok2.prefill_attn_score;
                        if (!tok2.would_prune) {
                            frame_prefill_kept += tok2.prefill_attn_score;
                        }
                    }
                    pfc.prefill_attn_retained_pct = (frame_prefill_total > 1e-10f)
                        ? 100.0f * frame_prefill_kept / frame_prefill_total : 0.0f;
                }

                result.correlation.per_frame.push_back(pfc);

                LOG_INF("  Frame pos=%d: %d tokens, Pearson r=%.4f, "
                        "avg_attn kept=%.6f pruned=%.6f ratio=%.2f prefill_retained=%.1f%%\n",
                        (int)fr.start_pos, pfc.n_tokens, pfc.pearson_r,
                        pfc.avg_attn_kept, pfc.avg_attn_pruned, pfc.attn_ratio,
                        pfc.prefill_attn_retained_pct);
            }
        }

        // Global correlation
        int n = (int)pre_llm_scores_vec.size();
        result.correlation.n_tokens_analyzed = n;

        if (n >= 5) {
            // Pearson r
            float sum_x = 0, sum_y = 0;
            for (int i = 0; i < n; i++) {
                sum_x += pre_llm_scores_vec[i];
                sum_y += llm_attn_vec[i];
            }
            float mean_x = sum_x / n;
            float mean_y = sum_y / n;
            float cov = 0, var_x = 0, var_y = 0;
            for (int i = 0; i < n; i++) {
                float dx = pre_llm_scores_vec[i] - mean_x;
                float dy = llm_attn_vec[i] - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            float denom = std::sqrt(var_x * var_y);
            result.correlation.pearson_r = (denom > 1e-10f) ? (cov / denom) : 0.0f;

            // Spearman rho (rank correlation)
            auto rank = [](const std::vector<float> & vals) -> std::vector<float> {
                int m = (int)vals.size();
                std::vector<int> indices(m);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(),
                    [&vals](int a, int b) { return vals[a] < vals[b]; });
                std::vector<float> ranks(m);
                for (int i = 0; i < m; i++) ranks[indices[i]] = (float)(i + 1);
                return ranks;
            };
            auto ranks_x = rank(pre_llm_scores_vec);
            auto ranks_y = rank(llm_attn_vec);
            float rs_sum_x = 0, rs_sum_y = 0;
            for (int i = 0; i < n; i++) {
                rs_sum_x += ranks_x[i];
                rs_sum_y += ranks_y[i];
            }
            float rs_mean_x = rs_sum_x / n;
            float rs_mean_y = rs_sum_y / n;
            float rs_cov = 0, rs_var_x = 0, rs_var_y = 0;
            for (int i = 0; i < n; i++) {
                float dx = ranks_x[i] - rs_mean_x;
                float dy = ranks_y[i] - rs_mean_y;
                rs_cov += dx * dy;
                rs_var_x += dx * dx;
                rs_var_y += dy * dy;
            }
            float rs_denom = std::sqrt(rs_var_x * rs_var_y);
            result.correlation.spearman_rho = (rs_denom > 1e-10f) ? (rs_cov / rs_denom) : 0.0f;

            // Split: kept vs pruned average LLM attention
            float sum_kept = 0, sum_pruned = 0;
            int n_kept = 0, n_pruned = 0;
            for (int i = 0; i < n; i++) {
                if (would_prune_vec[i]) {
                    sum_pruned += llm_attn_vec[i];
                    n_pruned++;
                } else {
                    sum_kept += llm_attn_vec[i];
                    n_kept++;
                }
            }
            result.correlation.n_kept = n_kept;
            result.correlation.n_pruned = n_pruned;
            result.correlation.avg_attn_kept = (n_kept > 0) ? sum_kept / n_kept : 0.0f;
            result.correlation.avg_attn_pruned = (n_pruned > 0) ? sum_pruned / n_pruned : 0.0f;
            result.correlation.attn_ratio = (result.correlation.avg_attn_pruned > 1e-10f)
                ? result.correlation.avg_attn_kept / result.correlation.avg_attn_pruned : 0.0f;

            LOG_INF("\n  === Global Correlation (n=%d tokens) ===\n", n);
            LOG_INF("  Pearson r:          %.4f\n", result.correlation.pearson_r);
            LOG_INF("  Spearman rho:       %.4f\n", result.correlation.spearman_rho);
            LOG_INF("  Kept tokens:        %d (avg LLM attn: %.6f)\n",
                    n_kept, result.correlation.avg_attn_kept);
            LOG_INF("  Pruned tokens:      %d (avg LLM attn: %.6f)\n",
                    n_pruned, result.correlation.avg_attn_pruned);
            LOG_INF("  Attention ratio:    %.2fx (kept/pruned, >1 = scoring works)\n",
                    result.correlation.attn_ratio);
        }

        // Prefill attention retention analysis
        {
            float prefill_total = 0.0f, prefill_kept = 0.0f;
            int prefill_n = 0;
            for (const auto & tok : vis_token_map.tokens) {
                if (tok.pruned) continue;
                prefill_total += tok.prefill_attn_score;
                prefill_n++;
                if (!tok.would_prune) {
                    prefill_kept += tok.prefill_attn_score;
                }
            }
            result.correlation.prefill_attn_total = prefill_total;
            result.correlation.prefill_attn_on_kept = prefill_kept;
            result.correlation.prefill_n_visual_tokens = prefill_n;
            result.correlation.prefill_attn_retained_pct = (prefill_total > 1e-10f)
                ? 100.0f * prefill_kept / prefill_total : 0.0f;

            float keep_ratio = (prefill_n > 0)
                ? (float)(prefill_n - result.correlation.n_pruned) / prefill_n : 0.0f;

            LOG_INF("\n  === Prefill Attention Retention ===\n");
            LOG_INF("  Visual tokens:      %d\n", prefill_n);
            LOG_INF("  Total prefill attn: %.4f\n", prefill_total);
            LOG_INF("  Attn on kept:       %.4f\n", prefill_kept);
            LOG_INF("  Keeping %.0f%% of tokens retains %.1f%% of prompt visual attention\n",
                    keep_ratio * 100.0f, result.correlation.prefill_attn_retained_pct);
        }
    }

    return result;
}

// ============================================================================
// Section 9: Online frame selector (Core new algorithm)
// ============================================================================

struct frame_decision {
    int   frame_index;         // 1-based
    float arrival_time_s;      // simulated arrival time
    float p95_magnitude;       // motion from previous frame
    float cumulative_motion;   // accumulated since last selection
    bool  selected;
    float selection_latency_ms; // time to make this decision
};

struct online_frame_selector {
    float threshold;
    int   max_frames;
    float fps;
    float motion_threshold;
    float motion_percentile;

    std::vector<frame_decision> decisions;
    std::vector<selected_frame> selected_frames;

    void init(float thresh, int max_sel, float f, float mt, float mp) {
        threshold = thresh;
        max_frames = max_sel;
        fps = f;
        motion_threshold = mt;
        motion_percentile = mp;
        decisions.clear();
        selected_frames.clear();
    }

    void run(const std::string & flow_dir, const std::string & video_name, int n_total_frames) {
        float cumulative = 0.0f;
        int n_selected = 0;

        for (int i = 1; i <= n_total_frames; i++) {
            frame_decision fd;
            fd.frame_index = i;
            fd.arrival_time_s = (float)(i - 1) / fps;
            fd.p95_magnitude = 0.0f;
            fd.cumulative_motion = 0.0f;
            fd.selected = false;
            fd.selection_latency_ms = 0.0f;

            int64_t t_dec = ggml_time_ms();

            if (i == 1) {
                // Always select first frame as keyframe
                fd.selected = true;
                fd.cumulative_motion = 0.0f;
                cumulative = 0.0f;
                n_selected++;
            } else {
                // Read flow from frame i-1 to i
                char buf[64];
                snprintf(buf, sizeof(buf), "/frame_%04d.flo", i - 1);
                std::string flo_path = flow_dir + "/" + video_name + buf;

                optical_flow flow;
                if (read_flo_file(flo_path, flow)) {
                    motion_stats stats = compute_motion_stats(flow, motion_threshold, motion_percentile);
                    fd.p95_magnitude = stats.p95_magnitude;
                    cumulative += stats.p95_magnitude;
                    fd.cumulative_motion = cumulative;

                    if (cumulative >= threshold && n_selected < max_frames) {
                        fd.selected = true;
                        cumulative = 0.0f;
                        n_selected++;
                    }
                }
            }

            fd.selection_latency_ms = (float)(ggml_time_ms() - t_dec);
            decisions.push_back(fd);

            if (fd.selected) {
                selected_frame sf;
                sf.frame_index = i;
                sf.motion_weight = fd.p95_magnitude;
                sf.complexity_score = 0.0f;
                selected_frames.push_back(sf);
            }
        }

        // Always select last frame if not already selected (temporal anchor)
        if (!decisions.empty() && !decisions.back().selected && n_selected < max_frames) {
            decisions.back().selected = true;
            selected_frame sf;
            sf.frame_index = n_total_frames;
            sf.motion_weight = decisions.back().p95_magnitude;
            sf.complexity_score = 0.0f;
            selected_frames.push_back(sf);
        }

        LOG_INF("Online selection: %zu frames selected from %d total (threshold=%.1f, fps=%.1f)\n",
                selected_frames.size(), n_total_frames, threshold, fps);
        LOG_INF("  Selected frames: ");
        for (const auto & sf : selected_frames) {
            LOG_INF("%d ", sf.frame_index);
        }
        LOG_INF("\n");
    }
};

// ============================================================================
// Section 10: Streaming pipelines
// ============================================================================

static void run_streaming_optimized(
    const streaming_params & sparams,
    common_params & params,
    const std::string & prompt,
    int n_predict) {

    int64_t t_total_start = ggml_time_ms();

    int n_total_frames = count_video_frames(sparams.sintel_dir, sparams.video_name);
    if (n_total_frames == 0) {
        LOG_ERR("No frames found for video '%s'. Exiting.\n", sparams.video_name.c_str());
        return;
    }
    LOG_INF("Total frames available: %d\n", n_total_frames);

    std::string flow_dir = sparams.sintel_dir + "/training/flow";

    // ---------------------------------------------------------------
    // Pass 1: Online frame selection (causally correct)
    // ---------------------------------------------------------------
    LOG_INF("\n--- Pass 1: Online Frame Selection ---\n");
    int64_t t_sel_start = ggml_time_ms();

    online_frame_selector selector;
    selector.init(sparams.cumulative_threshold, sparams.max_selected_frames,
                  sparams.fps, sparams.motion_threshold, sparams.motion_percentile);
    selector.run(flow_dir, sparams.video_name, n_total_frames);

    float selection_time_ms = (float)(ggml_time_ms() - t_sel_start);
    LOG_INF("Selection completed in %.1f ms\n", selection_time_ms);

    if (selector.selected_frames.empty()) {
        LOG_ERR("No frames selected. Exiting.\n");
        return;
    }

    // Spatial analysis for selected frames
    LOG_INF("\n--- Spatial Analysis for Selected Frames ---\n");
    std::vector<frame_spatial_analysis> spatial_analyses;
    for (const auto & sf : selector.selected_frames) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.flo", sf.frame_index);
        std::string flo_path = flow_dir + "/" + sparams.video_name + buf;

        optical_flow flow;
        if (read_flo_file(flo_path, flow)) {
            auto sa = analyze_spatial_patches(flow, sf.frame_index,
                                              sparams.patch_grid,
                                              sparams.spatial_threshold,
                                              sparams.motion_threshold);
            spatial_analyses.push_back(sa);

            LOG_INF("  Frame %d: %d/%d patches active (%.0f%% savings)\n",
                    sf.frame_index, sa.n_selected_patches, sa.n_total_patches,
                    sa.spatial_savings * 100.0f);
        } else {
            // No flow for this frame (e.g., first frame), push empty analysis
            frame_spatial_analysis empty_sa;
            empty_sa.frame_index = sf.frame_index;
            empty_sa.n_total_patches = sparams.patch_grid * sparams.patch_grid;
            empty_sa.n_selected_patches = empty_sa.n_total_patches;
            empty_sa.spatial_savings = 0.0f;
            spatial_analyses.push_back(empty_sa);
        }
    }

    // Build frame paths
    std::vector<std::string> frame_paths;
    for (const auto & sf : selector.selected_frames) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
        frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }

    // ---------------------------------------------------------------
    // Pass 2: Inference with embedding cache + pruning
    // ---------------------------------------------------------------
    LOG_INF("\n--- Pass 2: Model Inference with Embedding Cache ---\n");

    attention_capture_state main_capture_state = {};
    main_capture_state.target_layer = sparams.prune_layer;
    main_capture_state.sample_interval = sparams.attn_sample_interval;
    main_capture_state.initialized = false;
    main_capture_state.should_capture = false;
    main_capture_state.disabled = false;
    main_capture_state.token_map = nullptr;

    qcur_capture_state qcur_state = {};
    qcur_state.target_layer = sparams.prune_layer;

    // Mode selection: legacy (kq_soft_max capture, flash off) vs flash-compatible (Qcur capture, flash on)
    bool need_legacy_capture = sparams.enable_pruning || sparams.analyze_attention_correlation;
    bool need_flash_eviction = sparams.enable_postprefill_eviction && !need_legacy_capture;

    if (need_legacy_capture) {
        // Legacy mode: disable flash attention, capture kq_soft_max
        // Also handles postprefill_eviction if combined with pruning/correlation
        params.cb_eval = attn_softmax_callback;
        params.cb_eval_user_data = &main_capture_state;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        LOG_INF("Legacy attention capture mode (flash attn disabled, layer %d)%s\n",
                sparams.prune_layer,
                sparams.analyze_attention_correlation ? " [correlation analysis]" : "");
    } else if (need_flash_eviction) {
        // Flash-compatible mode: keep flash attention, capture Qcur for CPU Q@K^T
        params.cb_eval = qcur_capture_callback;
        params.cb_eval_user_data = &qcur_state;
        // Do NOT disable flash attention — that's the whole point
        LOG_INF("Flash-compatible eviction mode (flash attn enabled, capturing Qcur at layer %d)\n",
                sparams.prune_layer);
    }

    // Set up ViT attention capture for pre-LLM pruning or correlation analysis
    vit_attention_capture_state vit_capture;
    vit_capture.active = false;

    ggml_backend_sched_eval_callback vit_cb = nullptr;
    void * vit_cb_data = nullptr;
    if (sparams.enable_pre_llm_pruning || sparams.analyze_attention_correlation) {
        vit_cb = vit_attn_softmax_callback;
        vit_cb_data = &vit_capture;
        LOG_INF("Registered ViT attention capture callback%s\n",
                sparams.analyze_attention_correlation ? " [correlation analysis]" : "");
    }

    adaptive_vlm_context ctx(params, vit_cb, vit_cb_data);

    LOG_INF("Running inference with %zu frames [STREAMING_OPTIMIZED]...\n", frame_paths.size());
    LOG_INF("Prompt: %s\n", prompt.c_str());

    std::vector<frame_crop_info> crop_infos;
    bool need_capture = need_legacy_capture || (sparams.enable_postprefill_eviction && need_legacy_capture);
    auto inf_result = run_inference_with_embedding_cache(
        ctx, frame_paths, spatial_analyses, sparams, prompt, n_predict,
        crop_infos, need_capture ? &main_capture_state : nullptr,
        (sparams.enable_pre_llm_pruning || sparams.analyze_attention_correlation)
            ? &vit_capture : nullptr,
        need_flash_eviction ? &qcur_state : nullptr);

    // Build motion profile post-hoc for metrics
    LOG_INF("\n--- Building motion profile for metrics ---\n");
    auto profile = build_motion_profile(flow_dir, sparams.video_name,
                                         sparams.motion_threshold,
                                         sparams.motion_percentile);

    // Compute metrics
    double motion_cov = compute_motion_coverage(profile, selector.selected_frames);
    double entropy = compute_shannon_entropy(inf_result.generated_text);
    float total_time_ms = (float)(ggml_time_ms() - t_total_start);

    // ---------------------------------------------------------------
    // Write extended JSON
    // ---------------------------------------------------------------
    std::string json_path = sparams.output_json.empty()
        ? ("streaming_optimized_" + sparams.video_name + ".json")
        : sparams.output_json;

    json j;
    j["mode"] = "streaming_optimized";
    j["video_name"] = sparams.video_name;
    j["n_frames_total"] = n_total_frames;
    j["n_frames_selected"] = (int)selector.selected_frames.size();
    j["global_complexity"] = profile.global_complexity;

    // Streaming info
    j["streaming"]["fps"] = sparams.fps;
    j["streaming"]["video_duration_s"] = (float)(n_total_frames - 1) / sparams.fps;
    j["streaming"]["online_selection"]["method"] = "cumulative_motion";
    j["streaming"]["online_selection"]["cumulative_threshold"] = sparams.cumulative_threshold;
    j["streaming"]["online_selection"]["max_selected_frames"] = sparams.max_selected_frames;

    // Per-frame decisions
    json dec_arr = json::array();
    for (const auto & fd : selector.decisions) {
        json dj;
        dj["frame_index"] = fd.frame_index;
        dj["arrival_time_s"] = fd.arrival_time_s;
        dj["p95_magnitude"] = fd.p95_magnitude;
        dj["cumulative_motion"] = fd.cumulative_motion;
        dj["selected"] = fd.selected;
        dj["selection_latency_ms"] = fd.selection_latency_ms;
        dec_arr.push_back(dj);
    }
    j["streaming"]["online_selection"]["decisions"] = dec_arr;

    // Selected frames
    json sel_arr = json::array();
    for (const auto & sf : selector.selected_frames) {
        json sfj;
        sfj["index"] = sf.frame_index;
        sfj["weight"] = sf.motion_weight;
        sel_arr.push_back(sfj);
    }
    j["selected_frames"] = sel_arr;

    // Inference results
    j["inference"]["generated_text"] = inf_result.generated_text;
    j["inference"]["n_tokens"] = inf_result.n_tokens_generated;
    j["inference"]["encode_time_ms"] = inf_result.encode_time_ms;
    j["inference"]["generation_time_ms"] = inf_result.inference_time_ms;

    // Metrics
    j["metrics"]["motion_coverage"] = motion_cov;
    j["metrics"]["shannon_entropy"] = entropy;
    j["metrics"]["temporal_reduction"] = 1.0 - (double)selector.selected_frames.size() / n_total_frames;
    j["metrics"]["total_tokens_encoded"] = inf_result.total_tokens_encoded;
    j["metrics"]["total_tokens_cached"] = inf_result.total_tokens_cached;
    int total_enc_tokens = inf_result.total_tokens_encoded + inf_result.total_tokens_cached;
    j["metrics"]["encoder_savings_pct"] = (total_enc_tokens > 0)
        ? 100.0 * inf_result.total_tokens_cached / total_enc_tokens : 0.0;
    j["metrics"]["total_encode_time_ms"] = inf_result.total_encode_time_ms;
    j["metrics"]["estimated_full_encode_time_ms"] = inf_result.estimated_full_encode_time_ms;
    j["metrics"]["encode_speedup"] = (inf_result.total_encode_time_ms > 0)
        ? inf_result.estimated_full_encode_time_ms / inf_result.total_encode_time_ms : 1.0;

    // Per-frame cache info
    json cache_arr = json::array();
    for (const auto & fci : inf_result.frame_cache_infos) {
        json fc;
        fc["frame_index"] = fci.frame_index;
        fc["is_keyframe"] = fci.is_keyframe;
        fc["used_cache"] = fci.used_cache;
        fc["is_full_encode"] = fci.is_full_encode;
        fc["n_tokens_encoded"] = fci.n_tokens_encoded;
        fc["n_tokens_cached"] = fci.n_tokens_cached;
        fc["encode_time_ms"] = fci.encode_time_ms;
        fc["splice_time_ms"] = fci.splice_time_ms;
        if (sparams.enable_pre_llm_pruning) {
            fc["n_tokens_pre_llm_pruned"] = fci.n_tokens_pre_llm_pruned;
            fc["n_tokens_injected"] = fci.n_tokens_injected;
        }
        cache_arr.push_back(fc);
    }
    j["per_frame_cache"] = cache_arr;

    // Pruning
    if (sparams.enable_pruning) {
        j["pruning"]["enabled"] = true;
        j["pruning"]["n_visual_tokens_initial"] = inf_result.n_visual_tokens_initial;
        j["pruning"]["n_visual_tokens_after_round1"] = inf_result.n_visual_tokens_after_round1;
        j["pruning"]["n_visual_tokens_after_round2"] = inf_result.n_visual_tokens_after_round2;
        j["pruning"]["n_tokens_pruned_round1"] = inf_result.n_tokens_pruned_round1;
        j["pruning"]["n_tokens_pruned_round2"] = inf_result.n_tokens_pruned_round2;
        j["pruning"]["pruning_overhead_ms"] = inf_result.pruning_overhead_ms;
        j["pruning"]["total_reduction_pct"] = (inf_result.n_visual_tokens_initial > 0)
            ? (1.0 - (double)inf_result.n_visual_tokens_after_round2 / inf_result.n_visual_tokens_initial) * 100.0
            : 0.0;
    } else {
        j["pruning"]["enabled"] = false;
    }

    // Prefill pruning
    if (sparams.enable_prefill_pruning) {
        j["prefill_pruning"]["enabled"] = true;
        j["prefill_pruning"]["n_tokens_pruned"] = inf_result.n_tokens_pruned_prefill;
        j["prefill_pruning"]["overhead_ms"] = inf_result.prefill_pruning_overhead_ms;
        j["prefill_pruning"]["config"]["keep_ratio"] = sparams.prefill_keep_ratio;
    } else {
        j["prefill_pruning"]["enabled"] = false;
    }

    // Post-prefill eviction
    if (sparams.enable_postprefill_eviction) {
        bool used_flash_mode = !sparams.enable_pruning && !sparams.analyze_attention_correlation;
        j["postprefill_eviction"]["enabled"] = true;
        j["postprefill_eviction"]["mode"] = used_flash_mode ? "flash_qcur" : "legacy_softmax";
        j["postprefill_eviction"]["keep_ratio"] = sparams.postprefill_keep_ratio;
        j["postprefill_eviction"]["n_tokens_before"] = inf_result.postprefill_n_tokens_before;
        j["postprefill_eviction"]["n_tokens_evicted"] = inf_result.postprefill_n_tokens_evicted;
        j["postprefill_eviction"]["attn_retained_pct"] = inf_result.postprefill_attn_retained_pct;
        j["postprefill_eviction"]["eviction_overhead_ms"] = inf_result.postprefill_eviction_ms;
    } else {
        j["postprefill_eviction"]["enabled"] = false;
    }

    // Pre-LLM pruning
    if (sparams.enable_pre_llm_pruning) {
        j["pre_llm_pruning"]["enabled"] = true;
        j["pre_llm_pruning"]["strategy"] = sparams.pre_llm_strategy;
        j["pre_llm_pruning"]["keep_ratio"] = sparams.pre_llm_keep_ratio;
        j["pre_llm_pruning"]["weights"]["motion"] = sparams.pre_llm_weight_motion;
        j["pre_llm_pruning"]["weights"]["norm"] = sparams.pre_llm_weight_norm;
        j["pre_llm_pruning"]["weights"]["attn"] = sparams.pre_llm_weight_attn;
        j["pre_llm_pruning"]["total_tokens_before"] = inf_result.pre_llm_total_tokens_before;
        j["pre_llm_pruning"]["total_tokens_injected"] = inf_result.pre_llm_total_tokens_injected;
        j["pre_llm_pruning"]["total_tokens_pruned"] = inf_result.pre_llm_total_tokens_pruned;
        j["pre_llm_pruning"]["savings_pct"] = (inf_result.pre_llm_total_tokens_before > 0)
            ? 100.0 * inf_result.pre_llm_total_tokens_pruned / inf_result.pre_llm_total_tokens_before
            : 0.0;
    } else {
        j["pre_llm_pruning"]["enabled"] = false;
    }

    // Attention correlation analysis
    if (sparams.analyze_attention_correlation && inf_result.correlation.n_tokens_analyzed > 0) {
        auto & corr = inf_result.correlation;
        j["attention_correlation"]["n_tokens_analyzed"] = corr.n_tokens_analyzed;
        j["attention_correlation"]["pearson_r"] = corr.pearson_r;
        j["attention_correlation"]["spearman_rho"] = corr.spearman_rho;
        j["attention_correlation"]["avg_attn_kept"] = corr.avg_attn_kept;
        j["attention_correlation"]["avg_attn_pruned"] = corr.avg_attn_pruned;
        j["attention_correlation"]["attn_ratio"] = corr.attn_ratio;
        j["attention_correlation"]["n_kept"] = corr.n_kept;
        j["attention_correlation"]["n_pruned"] = corr.n_pruned;
        j["attention_correlation"]["strategy"] = sparams.pre_llm_strategy;
        j["attention_correlation"]["keep_ratio"] = sparams.pre_llm_keep_ratio;

        json per_frame_arr = json::array();
        for (const auto & pf : corr.per_frame) {
            json pf_j;
            pf_j["frame_pos"] = pf.frame_idx;
            pf_j["n_tokens"] = pf.n_tokens;
            pf_j["pearson_r"] = pf.pearson_r;
            pf_j["avg_attn_kept"] = pf.avg_attn_kept;
            pf_j["avg_attn_pruned"] = pf.avg_attn_pruned;
            pf_j["attn_ratio"] = pf.attn_ratio;
            pf_j["prefill_attn_retained_pct"] = pf.prefill_attn_retained_pct;
            per_frame_arr.push_back(pf_j);
        }
        j["attention_correlation"]["per_frame"] = per_frame_arr;

        // Prefill attention retention
        float keep_ratio = (corr.prefill_n_visual_tokens > 0)
            ? (float)(corr.prefill_n_visual_tokens - corr.n_pruned) / corr.prefill_n_visual_tokens : 0.0f;
        j["attention_correlation"]["prefill"]["attn_retained_pct"] = corr.prefill_attn_retained_pct;
        j["attention_correlation"]["prefill"]["attn_total"] = corr.prefill_attn_total;
        j["attention_correlation"]["prefill"]["attn_on_kept"] = corr.prefill_attn_on_kept;
        j["attention_correlation"]["prefill"]["n_visual_tokens"] = corr.prefill_n_visual_tokens;
        j["attention_correlation"]["prefill"]["keep_ratio"] = keep_ratio;

        json prefill_per_frame_arr = json::array();
        for (const auto & pf : corr.per_frame) {
            json ppf_j;
            ppf_j["frame_pos"] = pf.frame_idx;
            ppf_j["n_tokens"] = pf.n_tokens;
            ppf_j["prefill_attn_retained_pct"] = pf.prefill_attn_retained_pct;
            prefill_per_frame_arr.push_back(ppf_j);
        }
        j["attention_correlation"]["prefill"]["per_frame"] = prefill_per_frame_arr;
    }

    // Timing
    j["timing"]["selection_ms"] = selection_time_ms;
    j["timing"]["encode_ms"] = inf_result.encode_time_ms;
    j["timing"]["generation_ms"] = inf_result.inference_time_ms;
    j["timing"]["pruning_overhead_ms"] = inf_result.pruning_overhead_ms;
    j["timing"]["prefill_pruning_overhead_ms"] = inf_result.prefill_pruning_overhead_ms;
    j["timing"]["postprefill_eviction_ms"] = inf_result.postprefill_eviction_ms;
    j["timing"]["total_ms"] = total_time_ms;

    // Parameters
    j["params"]["sintel_dir"] = sparams.sintel_dir;
    j["params"]["video_name"] = sparams.video_name;
    j["params"]["fps"] = sparams.fps;
    j["params"]["cumulative_threshold"] = sparams.cumulative_threshold;
    j["params"]["max_selected_frames"] = sparams.max_selected_frames;
    j["params"]["patch_grid"] = sparams.patch_grid;
    j["params"]["spatial_threshold"] = sparams.spatial_threshold;
    j["params"]["patch_reuse_motion"] = sparams.patch_reuse_motion;
    j["params"]["enable_pruning"] = sparams.enable_pruning;
    j["params"]["enable_prefill_pruning"] = sparams.enable_prefill_pruning;
    j["params"]["enable_kv_cache_reuse"] = sparams.enable_kv_cache_reuse;
    j["params"]["enable_postprefill_eviction"] = sparams.enable_postprefill_eviction;
    if (sparams.enable_postprefill_eviction) {
        j["params"]["postprefill_keep_ratio"] = sparams.postprefill_keep_ratio;
    }
    j["params"]["enable_pre_llm_pruning"] = sparams.enable_pre_llm_pruning;
    if (sparams.enable_pre_llm_pruning) {
        j["params"]["pre_llm_keep_ratio"] = sparams.pre_llm_keep_ratio;
        j["params"]["pre_llm_weight_motion"] = sparams.pre_llm_weight_motion;
        j["params"]["pre_llm_weight_norm"] = sparams.pre_llm_weight_norm;
        j["params"]["pre_llm_weight_attn"] = sparams.pre_llm_weight_attn;
    }

    std::ofstream out(json_path);
    if (!out.is_open()) {
        LOG_ERR("Cannot write JSON to %s\n", json_path.c_str());
        return;
    }
    out << j.dump(2) << std::endl;
    out.close();

    LOG_INF("\nResults written to %s\n", json_path.c_str());

    // Print summary
    LOG_INF("\n=== Summary [STREAMING_OPTIMIZED] ===\n");
    LOG_INF("Video:              %s\n", sparams.video_name.c_str());
    LOG_INF("Frames:             %zu / %d selected\n", selector.selected_frames.size(), n_total_frames);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Encoder savings:    %.1f%%\n",
            (total_enc_tokens > 0) ? 100.0 * inf_result.total_tokens_cached / total_enc_tokens : 0.0);
    LOG_INF("Encode speedup:     %.2fx\n",
            (inf_result.total_encode_time_ms > 0)
                ? inf_result.estimated_full_encode_time_ms / inf_result.total_encode_time_ms : 1.0f);

    if (sparams.enable_pruning && inf_result.n_visual_tokens_initial > 0) {
        LOG_INF("LLM pruning:        %d -> %d -> %d tokens (%.1f%% reduction)\n",
                inf_result.n_visual_tokens_initial,
                inf_result.n_visual_tokens_after_round1,
                inf_result.n_visual_tokens_after_round2,
                (1.0 - (double)inf_result.n_visual_tokens_after_round2 / inf_result.n_visual_tokens_initial) * 100.0);
    }

    if (sparams.enable_pre_llm_pruning && inf_result.pre_llm_total_tokens_before > 0) {
        LOG_INF("Pre-LLM pruning:    %d/%d tokens kept (%.1f%% pruned)\n",
                inf_result.pre_llm_total_tokens_injected,
                inf_result.pre_llm_total_tokens_before,
                100.0 * inf_result.pre_llm_total_tokens_pruned / inf_result.pre_llm_total_tokens_before);
    }

    if (sparams.enable_postprefill_eviction && inf_result.postprefill_n_tokens_before > 0) {
        LOG_INF("Post-prefill evict: %d/%d tokens kept (%.1f%% attn retained, %.1f ms)\n",
                inf_result.postprefill_n_tokens_before - inf_result.postprefill_n_tokens_evicted,
                inf_result.postprefill_n_tokens_before,
                inf_result.postprefill_attn_retained_pct,
                inf_result.postprefill_eviction_ms);
    }

    LOG_INF("Total time:         %.1f ms\n", total_time_ms);
    LOG_INF("Output:             %s\n", json_path.c_str());

    llama_perf_context_print(ctx.lctx);
}

static void run_baseline_pipeline(
    const streaming_params & sparams,
    common_params & params,
    const std::string & prompt,
    int n_predict) {

    int64_t t_total_start = ggml_time_ms();

    int n_total_frames = count_video_frames(sparams.sintel_dir, sparams.video_name);
    if (n_total_frames == 0) {
        LOG_ERR("No frames found for video '%s'. Exiting.\n", sparams.video_name.c_str());
        return;
    }
    LOG_INF("Total frames available: %d\n", n_total_frames);

    std::string flow_dir = sparams.sintel_dir + "/training/flow";

    // Uniform frame selection
    LOG_INF("\n--- Baseline: Uniform Frame Selection ---\n");
    int64_t t_sel_start = ggml_time_ms();
    auto selected = select_frames_uniform(n_total_frames, sparams.baseline_uniform_count);
    float selection_time_ms = (float)(ggml_time_ms() - t_sel_start);

    // Build frame paths
    std::vector<std::string> frame_paths;
    for (const auto & sf : selected) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
        frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }

    // Empty spatial analyses -> forces full encode for every frame, no cache, no pruning
    std::vector<frame_spatial_analysis> spatial_analyses;

    // Initialize model and run inference
    LOG_INF("\n--- Model Inference (Baseline) ---\n");

    adaptive_vlm_context ctx(params);

    LOG_INF("Running inference with %zu frames [STREAMING_BASELINE]...\n", frame_paths.size());
    LOG_INF("Prompt: %s\n", prompt.c_str());

    // Create a streaming_params copy with pruning disabled for baseline
    streaming_params baseline_sparams = sparams;
    baseline_sparams.enable_pruning = false;
    baseline_sparams.enable_prefill_pruning = false;

    std::vector<frame_crop_info> crop_infos;
    auto inf_result = run_inference_with_embedding_cache(
        ctx, frame_paths, spatial_analyses, baseline_sparams, prompt, n_predict, crop_infos);

    // Build motion profile for metrics
    LOG_INF("\n--- Building motion profile for metrics ---\n");
    auto profile = build_motion_profile(flow_dir, sparams.video_name,
                                         sparams.motion_threshold,
                                         sparams.motion_percentile);

    // Compute metrics
    double motion_cov = compute_motion_coverage(profile, selected);
    double entropy = compute_shannon_entropy(inf_result.generated_text);
    float total_time_ms = (float)(ggml_time_ms() - t_total_start);

    // Write JSON
    std::string json_path = sparams.output_json.empty()
        ? ("streaming_baseline_" + sparams.video_name + ".json")
        : sparams.output_json;

    json j;
    j["mode"] = "streaming_baseline";
    j["video_name"] = sparams.video_name;
    j["n_frames_total"] = n_total_frames;
    j["n_frames_selected"] = (int)selected.size();
    j["global_complexity"] = profile.global_complexity;

    // Streaming info
    j["streaming"]["fps"] = sparams.fps;
    j["streaming"]["video_duration_s"] = (float)(n_total_frames - 1) / sparams.fps;
    j["streaming"]["online_selection"]["method"] = "uniform";
    j["streaming"]["online_selection"]["baseline_uniform_count"] = sparams.baseline_uniform_count;

    // Selected frames
    json sel_arr = json::array();
    for (const auto & sf : selected) {
        json sfj;
        sfj["index"] = sf.frame_index;
        sfj["weight"] = sf.motion_weight;
        sel_arr.push_back(sfj);
    }
    j["selected_frames"] = sel_arr;

    // Inference
    j["inference"]["generated_text"] = inf_result.generated_text;
    j["inference"]["n_tokens"] = inf_result.n_tokens_generated;
    j["inference"]["encode_time_ms"] = inf_result.encode_time_ms;
    j["inference"]["generation_time_ms"] = inf_result.inference_time_ms;

    // Metrics
    j["metrics"]["motion_coverage"] = motion_cov;
    j["metrics"]["shannon_entropy"] = entropy;
    j["metrics"]["temporal_reduction"] = 1.0 - (double)selected.size() / n_total_frames;
    j["metrics"]["total_tokens_encoded"] = inf_result.total_tokens_encoded;
    j["metrics"]["total_tokens_cached"] = inf_result.total_tokens_cached;
    int total_enc_tokens = inf_result.total_tokens_encoded + inf_result.total_tokens_cached;
    j["metrics"]["encoder_savings_pct"] = (total_enc_tokens > 0)
        ? 100.0 * inf_result.total_tokens_cached / total_enc_tokens : 0.0;
    j["metrics"]["total_encode_time_ms"] = inf_result.total_encode_time_ms;
    j["metrics"]["estimated_full_encode_time_ms"] = inf_result.estimated_full_encode_time_ms;
    j["metrics"]["encode_speedup"] = (inf_result.total_encode_time_ms > 0)
        ? inf_result.estimated_full_encode_time_ms / inf_result.total_encode_time_ms : 1.0;

    // Per-frame cache info
    json cache_arr = json::array();
    for (const auto & fci : inf_result.frame_cache_infos) {
        json fc;
        fc["frame_index"] = fci.frame_index;
        fc["is_keyframe"] = fci.is_keyframe;
        fc["used_cache"] = fci.used_cache;
        fc["is_full_encode"] = fci.is_full_encode;
        fc["n_tokens_encoded"] = fci.n_tokens_encoded;
        fc["n_tokens_cached"] = fci.n_tokens_cached;
        fc["encode_time_ms"] = fci.encode_time_ms;
        fc["splice_time_ms"] = fci.splice_time_ms;
        cache_arr.push_back(fc);
    }
    j["per_frame_cache"] = cache_arr;

    // Pruning (disabled for baseline)
    j["pruning"]["enabled"] = false;
    j["prefill_pruning"]["enabled"] = false;

    // Timing
    j["timing"]["selection_ms"] = selection_time_ms;
    j["timing"]["encode_ms"] = inf_result.encode_time_ms;
    j["timing"]["generation_ms"] = inf_result.inference_time_ms;
    j["timing"]["total_ms"] = total_time_ms;

    // Parameters
    j["params"]["sintel_dir"] = sparams.sintel_dir;
    j["params"]["video_name"] = sparams.video_name;
    j["params"]["fps"] = sparams.fps;
    j["params"]["baseline_uniform_count"] = sparams.baseline_uniform_count;

    std::ofstream out(json_path);
    if (!out.is_open()) {
        LOG_ERR("Cannot write JSON to %s\n", json_path.c_str());
        return;
    }
    out << j.dump(2) << std::endl;
    out.close();

    LOG_INF("\nResults written to %s\n", json_path.c_str());

    // Print summary
    LOG_INF("\n=== Summary [STREAMING_BASELINE] ===\n");
    LOG_INF("Video:              %s\n", sparams.video_name.c_str());
    LOG_INF("Frames:             %zu / %d selected\n", selected.size(), n_total_frames);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Total time:         %.1f ms\n", total_time_ms);
    LOG_INF("Output:             %s\n", json_path.c_str());

    llama_perf_context_print(ctx.lctx);
}

// ============================================================================
// Section 11: Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    streaming_params sparams;
    auto filtered_argv = filter_custom_args(argc, argv, sparams);
    int filtered_argc = (int)filtered_argv.size();

    common_params params;
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.model.path.empty()) {
        params.model.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf";
    }
    if (params.mmproj.path.empty()) {
        params.mmproj.path = "/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf";
    }

    int n_predict = params.n_predict < 0 ? 512 : params.n_predict;

    std::string prompt = params.prompt.empty()
        ? "You are watching sequential frames from an animated short film. "
          "Provide a detailed and comprehensive description of everything you observe: "
          "the characters and their appearances, the environment and setting, "
          "lighting and color palette, camera movement and angles, "
          "character actions and body language, any visible motion or animation dynamics, "
          "the mood and atmosphere of the scene, and any narrative or story elements you can infer. "
          "Be thorough and descriptive."
        : params.prompt;

    const char * mode_str = sparams.baseline ? "STREAMING_BASELINE" : "STREAMING_OPTIMIZED";

    LOG_INF("\n=== Streaming VLM Benchmark [%s] ===\n", mode_str);
    LOG_INF("Video: %s\n", sparams.video_name.c_str());
    LOG_INF("FPS: %.1f\n", sparams.fps);
    if (sparams.baseline) {
        LOG_INF("Baseline uniform count: %d\n", sparams.baseline_uniform_count);
    } else {
        LOG_INF("Cumulative threshold: %.1f\n", sparams.cumulative_threshold);
        LOG_INF("Max selected frames: %d\n", sparams.max_selected_frames);
    }

    if (sparams.baseline) {
        run_baseline_pipeline(sparams, params, prompt, n_predict);
    } else {
        run_streaming_optimized(sparams, params, prompt, n_predict);
    }

    return 0;
}
