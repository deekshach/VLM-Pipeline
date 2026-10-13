// Adaptive Spatio-Temporal VLM Optimizer with Encoder-Level Embedding Cache
//
// Extends adaptive temporal sampling with encoder-level patch embedding caching:
// The keyframe is fully encoded and its per-patch embeddings are cached. For
// subsequent frames, only the changed patches (identified via spatial analysis)
// are re-encoded as a smaller 28px-aligned crop, then spliced into the cached
// full-grid embedding array. The LLM always sees the complete token grid with
// correct M-RoPE positions, preserving spatial context.
//
// Key insight: Cache embeddings at the encoder output level, not pixels.
// The LLM always receives a full nx*ny embedding grid. Only the ViT encode
// is reduced for unchanged regions.

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
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <chrono>
#include <cinttypes>
#include <unordered_map>
#include <map>

using json = nlohmann::ordered_json;

// ============================================================================
// Section 1: Custom parameters for Sintel dataset processing
// ============================================================================

struct sintel_params {
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
    // Spatial cropping parameters
    int         crop_padding  = 16;   // pixel padding around active-patch bounding box
    int         min_crop_dim  = 112;  // minimum crop dimension to avoid upscaling penalty
    // Patch cache reuse threshold: patches with mean_motion below this reuse cached embeddings.
    // Higher value = more aggressive caching (tolerate more motion before re-encoding).
    // 0 = use binary spatial_analysis decision (original behavior).
    float       patch_reuse_motion = 0.0f;

    // LLM-side progressive visual token pruning parameters
    int         prune_layer          = 2;     // which transformer layer's attention to capture
    int         attn_sample_interval = 5;     // sample attention every N generation steps
    int         first_prune_step     = 5;     // when to do coarse patch pruning
    int         second_prune_step    = 15;    // when to do fine token pruning
    float       patch_keep_ratio     = 0.5f;  // fraction of patches to keep in round 1
    float       token_keep_ratio     = 0.7f;  // fraction of remaining tokens to keep in round 2
    float       motion_weight        = 0.3f;  // weight of motion vs attention in hybrid scoring
    bool        enable_pruning       = false; // enable LLM-side visual token pruning

    // Prefill-stage motion pruning parameters
    bool        enable_prefill_pruning = false; // prune low-motion tokens between frame decodes
    float       prefill_keep_ratio     = 0.6f;  // fraction of tokens to keep per non-keyframe
};

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
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nAdaptive Spatio-Temporal VLM Optimizer with Spatial Cropping\n\n");
    printf("Usage: %s [options]\n\n", argv[0]);
    printf("Required llama.cpp options:\n");
    printf("  -m <model>          Path to Qwen2.5-VL GGUF model\n");
    printf("  --mmproj <mmproj>   Path to Qwen2.5-VL mmproj GGUF\n\n");
    printf("Sintel-specific options:\n");
    printf("  --sintel-dir <dir>  MPI-Sintel dataset directory\n");
    printf("  --video <name>      Video name (e.g., alley_1)\n");
    printf("  --min-frames <n>    Minimum frames to select (default: 3)\n");
    printf("  --max-frames <n>    Maximum frames to select (default: 24)\n");
    printf("  --base-frames <n>   Base frame count (default: 16)\n");
    printf("  --motion-threshold  Motion magnitude threshold (default: 2.0)\n");
    printf("  --motion-percentile Percentile for motion stats (default: 0.95)\n");
    printf("  --selection-percentile Percentile for weight thresholding (default: 70)\n");
    printf("  --patch-grid <n>    Spatial patch grid size NxN (default: 4)\n");
    printf("  --spatial-threshold Patch motion ratio threshold (default: 0.05)\n");
    printf("  --crop-padding <n>  Pixel padding around crop region (default: 16)\n");
    printf("  --min-crop-dim <n>  Minimum crop dimension in pixels (default: 112)\n");
    printf("  --patch-reuse-motion <f> Mean motion threshold for patch re-encode (default: 0 = use binary)\n");
    printf("                       Higher values tolerate more motion before re-encoding a patch\n");
    printf("  --output-json <f>   Output JSON file path\n");
    printf("  --baseline          Use uniform frame sampling (no motion analysis)\n\n");
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
    printf("  --enable-prefill-pruning   Prune low-motion tokens between frame decodes during prefill\n");
    printf("  --prefill-keep-ratio <f>   Fraction of tokens to keep per non-keyframe (default: 0.6)\n\n");
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

    // Global complexity using Python's algorithm
    {
        std::vector<float> p95_values;
        for (const auto & s : profile.frame_stats) {
            p95_values.push_back(s.p95_magnitude);
        }
        const int nf = (int)p95_values.size();

        double p95_mean = 0.0;
        for (float v : p95_values) p95_mean += v;
        p95_mean /= nf;
        float mean_norm = std::min((float)(p95_mean / 100.0), 1.0f);

        double p95_var = 0.0;
        for (float v : p95_values) {
            double d = v - p95_mean;
            p95_var += d * d;
        }
        p95_var /= nf;
        float var_norm = std::min((float)(p95_var / 5000.0), 1.0f);

        float p95_min = *std::min_element(p95_values.begin(), p95_values.end());
        float p95_max = *std::max_element(p95_values.begin(), p95_values.end());
        float p95_range = p95_max - p95_min;
        float range_norm = std::min(p95_range / 150.0f, 1.0f);

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
// Section 4: Motion-guided frame selector
// ============================================================================

struct selected_frame {
    int   frame_index;
    float motion_weight;
    float complexity_score;
};

static std::vector<selected_frame> select_frames(const video_motion_profile & profile,
                                                  int min_frames,
                                                  int max_frames,
                                                  int base_frames,
                                                  float global_complexity,
                                                  float selection_percentile) {
    const int n_flow = (int)profile.frame_stats.size();
    if (n_flow == 0) return {};

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
    n_select = std::min(n_select, n_flow);

    LOG_INF("Adaptive frame count: %d (complexity=%.4f, base=%d, range=[%d,%d])\n",
            n_select, global_complexity, base_frames, min_frames, max_frames);

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

    std::vector<double> weights(n_flow);
    for (int i = 0; i < n_flow; i++) {
        if (p95_values[i] >= threshold) {
            weights[i] = (double)p95_values[i];
        } else {
            weights[i] = 0.1;
        }
    }

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

    std::mt19937 rng(42);
    std::vector<int> selected_indices;
    std::vector<double> current_probs = probabilities;

    for (int k = 0; k < n_select; k++) {
        double total = 0.0;
        for (auto p : current_probs) total += p;
        if (total <= 0.0) break;

        for (int i = 0; i < n_flow; i++) {
            current_probs[i] /= total;
        }

        std::discrete_distribution<int> dist(current_probs.begin(), current_probs.end());
        int idx = dist(rng);

        selected_indices.push_back(idx);
        current_probs[idx] = 0.0;
    }

    std::sort(selected_indices.begin(), selected_indices.end());
    selected_indices.erase(std::unique(selected_indices.begin(), selected_indices.end()),
                           selected_indices.end());

    std::vector<selected_frame> result;
    for (int idx : selected_indices) {
        selected_frame sf;
        sf.frame_index     = idx + 1;
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

    // Per-patch selective update: for each token, if patch_active[i] is true,
    // copy from new_embd; otherwise keep the cached value.
    // Returns counts of updated and kept patches.
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
// Section 5b-2: Per-patch motion mapping for ViT token grid
// ============================================================================

// Build a per-token (ViT grid) active mask from the spatial analysis grid.
// Maps each ViT token at (tx, ty) in [0..vit_nx) x [0..vit_ny) to the
// corresponding spatial analysis patch and returns true if that patch needs re-encoding.
//
// When patch_reuse_motion > 0, uses the continuous mean_motion value:
//   patch is active (needs re-encode) only if mean_motion >= patch_reuse_motion.
// When patch_reuse_motion == 0, uses the binary p.selected flag (original behavior).
static std::vector<bool> build_patch_active_mask(
    const frame_spatial_analysis & sa,
    int grid_size,
    int vit_nx, int vit_ny,
    float patch_reuse_motion = 0.0f) {

    int n_tokens = vit_nx * vit_ny;
    std::vector<bool> active(n_tokens, true); // default to active (safe fallback)

    // Build a grid_size x grid_size lookup from the spatial analysis
    // Use continuous mean_motion when patch_reuse_motion > 0, else binary selected flag
    std::vector<bool> sa_active(grid_size * grid_size, false);
    for (const auto & p : sa.patches) {
        int idx = p.grid_y * grid_size + p.grid_x;
        if (patch_reuse_motion > 0.0f) {
            // Patch needs re-encoding only if mean motion exceeds the reuse threshold
            sa_active[idx] = (p.mean_motion >= patch_reuse_motion);
        } else {
            // Original binary behavior
            sa_active[idx] = p.selected;
        }
    }

    // Map each ViT token to its spatial analysis patch
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
    // Per-token info for each visual token in the KV cache
    struct token_info {
        int frame_idx;
        int patch_row;       // in spatial grid (ViT token y)
        int patch_col;       // in spatial grid (ViT token x)
        llama_pos kv_pos;    // primary KV cache position (temporal, = pos_0 for M-RoPE)
        llama_pos ext_x;     // M-RoPE spatial x coordinate (pos_0 + col)
        llama_pos ext_y;     // M-RoPE spatial y coordinate (pos_0 + row)
        float motion_score;  // from optical flow analysis (mean_motion of parent patch)
        float attn_score;    // accumulated attention from text tokens
        int attn_samples;    // number of times attention was sampled
        bool pruned;
    };

    // All visual tokens, indexed sequentially
    std::vector<token_info> tokens;

    // Reverse map: spatial analysis patch key "frame_gridRow_gridCol" → token indices
    std::map<std::string, std::vector<int>> patch_to_token_indices;

    // Range of visual token positions per frame: [start_pos, end_pos)
    struct frame_range {
        llama_pos start_pos;  // pos_0 for this frame
        llama_pos end_pos;    // pos_0 + max(nx, ny)
        int nx, ny;           // ViT grid dimensions
        int token_start_idx;  // index into tokens[] for first token of this frame
        int token_count;      // number of tokens in this frame
    };
    std::vector<frame_range> frame_ranges;

    void add_token(llama_pos kv_pos, llama_pos ext_x, llama_pos ext_y,
                   int frame_idx, int row, int col, float motion,
                   int sa_grid_row, int sa_grid_col) {
        token_info ti;
        ti.frame_idx    = frame_idx;
        ti.patch_row    = row;
        ti.patch_col    = col;
        ti.kv_pos       = kv_pos;
        ti.ext_x        = ext_x;
        ti.ext_y        = ext_y;
        ti.motion_score = motion;
        ti.attn_score   = 0.0f;
        ti.attn_samples = 0;
        ti.pruned       = false;

        int idx = (int)tokens.size();
        tokens.push_back(ti);

        // Group by spatial analysis patch for coarse pruning
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
    int target_layer;           // which layer's attention to capture
    int sample_interval;        // every N generation steps
    int current_gen_step;       // current generation step
    bool should_capture;        // true when current step % interval == 0

    visual_token_map * token_map;  // pointer to the visual token map

    // Track the range of visual token positions for efficient lookup
    llama_pos visual_pos_min;   // minimum visual token position
    llama_pos visual_pos_max;   // maximum visual token position (exclusive)
    llama_pos text_pos_start;   // where text tokens begin (after all visual tokens)

    bool initialized;
    bool disabled;      // set to true after pruning is complete — callback becomes a no-op
};

// The cb_eval callback function for attention capture
static bool attn_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (attention_capture_state *)user_data;

    if (!state || !state->initialized || state->disabled || !state->should_capture) {
        // Return false for ask=true: "I don't want to observe any tensors"
        // Return true for ask=false: "continue computation" (shouldn't happen, but safe)
        return !ask;
    }

    if (ask) {
        // Check if this is the KQV merged tensor at our target layer
        // In llama.cpp, the attention output tensor is named like "kqv_merged_cont-{layer}"
        std::string name(t->name);
        std::string target = "kqv_merged_cont-" + std::to_string(state->target_layer);
        return name.find(target) != std::string::npos;
    }

    // ask == false: tensor data is available
    // The kqv_merged_cont tensor has shape [n_embd, n_tokens] after attention
    // We need the raw attention weights, but kqv_merged_cont is post-softmax * V
    //
    // Alternative approach: capture the softmax output "attn-" tensor
    // For now, we use a simpler heuristic: the L2 norm of the output embedding
    // for each token position serves as a proxy for attention importance.
    //
    // The tensor data is on GPU, we need to read it back
    int n_elements = (int)ggml_nelements(t);
    if (n_elements <= 0) return true;

    // Get tensor dimensions
    int64_t ne0 = t->ne[0]; // n_embd (or n_head * head_dim)
    int64_t ne1 = t->ne[1]; // n_tokens in this batch

    // During generation, ne1 == 1 (single token). The new text token attends
    // to all KV cache entries. The kqv_merged_cont represents the weighted sum
    // of values. We can't directly extract per-position attention weights from it.
    //
    // Instead, we use a different strategy: look for the "attn_soft_max" or
    // "kq_masked" tensor to get the raw attention weights.
    // For a simpler first pass, we just note that the callback was triggered
    // and increment the sample count for all visual tokens.
    // The actual scoring will rely more on motion priors.

    if (state->token_map && ne1 == 1) {
        // Single-token generation step: the attention was computed
        // Increment sample count for all active visual tokens
        for (auto & t : state->token_map->tokens) {
            if (!t.pruned) {
                t.attn_samples++;
            }
        }
    }

    return true;
}

// Alternative callback that captures the attention softmax output
static bool attn_softmax_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (attention_capture_state *)user_data;

    if (!state || !state->initialized || state->disabled || !state->should_capture) {
        // Return false for ask=true: "don't observe" — avoids GPU sync overhead
        // Return true for ask=false: "continue computation"
        return !ask;
    }

    if (ask) {
        // Look for softmax attention weights tensor
        // In llama.cpp these are named "kq_soft_max-{layer}" or similar
        std::string name(t->name);
        std::string target = "kq_soft_max-" + std::to_string(state->target_layer);
        if (name.find(target) != std::string::npos) {
            return true;  // observe this tensor
        }
        return false;  // skip others
    }

    // ask == false: we have the softmax attention weights
    // Shape: [n_kv, n_heads, n_tokens_q] where n_tokens_q=1 during generation
    // Each head has attention weights over all KV cache positions

    if (!state->token_map) return true;

    int64_t n_kv    = t->ne[0]; // number of KV cache positions
    int64_t n_heads = t->ne[1]; // number of attention heads
    int64_t n_q     = t->ne[2]; // number of query tokens (1 during generation)

    if (n_q != 1) return true; // only process single-token generation

    // Read the attention weights from the tensor
    // Note: tensor may be on GPU, need to copy to CPU
    size_t data_size = ggml_nbytes(t);
    std::vector<float> attn_data(n_kv * n_heads);

    // The data might be in f32 or f16 format
    if (t->type == GGML_TYPE_F32) {
        memcpy(attn_data.data(), t->data, std::min(data_size, attn_data.size() * sizeof(float)));
    } else {
        // For non-f32 types, we'd need conversion - skip for safety
        return true;
    }

    // Accumulate attention scores for visual tokens
    // Sum across all heads for each KV position
    // Note: with M-RoPE, multiple tokens may share the same kv_pos, but each
    // has a unique (kv_pos, ext_x, ext_y) tuple. The KV cache index for a token
    // is not directly available here, so we use kv_pos as a proxy for attention.
    // Tokens sharing a kv_pos will get the same attention score contribution.
    for (auto & t : state->token_map->tokens) {
        if (t.pruned) continue;

        llama_pos pos = t.kv_pos;
        if (pos < 0 || pos >= n_kv) continue;

        float total_attn = 0.0f;
        for (int h = 0; h < n_heads; h++) {
            total_attn += attn_data[h * n_kv + pos];
        }
        t.attn_score += total_attn / n_heads; // average across heads
        t.attn_samples++;
    }

    return true;
}

// Estimate Qwen2.5-VL token count for a given image dimension.
// Qwen2.5-VL uses a 14px patch size with 2x2 merge, so effective tile = 28px.
static int estimate_tokens(int w, int h, int align = 28) {
    int w_a = std::max(align, (int)(std::round((float)w / align) * align));
    int h_a = std::max(align, (int)(std::round((float)h / align) * align));
    return (w_a / align) * (h_a / align);
}

// Compute the crop region from spatial analysis of a frame.
// Returns a frame_crop_info with the bounding box of active patches + padding.
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

    // Find bounding box of active (selected) patches in grid coordinates
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

    // If no patches selected (shouldn't happen due to minimum guarantee), use full image
    if (gx_max < 0) {
        ci.crop_x = 0;
        ci.crop_y = 0;
        ci.crop_w = img_w;
        ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
        return ci;
    }

    // Convert grid coordinates to pixel coordinates
    int pw = img_w / grid_size;
    int ph = img_h / grid_size;

    int px_min = gx_min * pw;
    int py_min = gy_min * ph;
    // For the max, account for the last column/row extending to image edge
    int px_max = (gx_max == grid_size - 1) ? img_w : (gx_max + 1) * pw;
    int py_max = (gy_max == grid_size - 1) ? img_h : (gy_max + 1) * ph;

    // Add padding
    px_min = std::max(0, px_min - crop_padding);
    py_min = std::max(0, py_min - crop_padding);
    px_max = std::min(img_w, px_max + crop_padding);
    py_max = std::min(img_h, py_max + crop_padding);

    int cw = px_max - px_min;
    int ch = py_max - py_min;

    // Enforce minimum dimensions (expand symmetrically if needed)
    if (cw < min_crop_dim) {
        int expand = min_crop_dim - cw;
        int left  = expand / 2;
        int right = expand - left;
        px_min = std::max(0, px_min - left);
        px_max = std::min(img_w, px_max + right);
        cw = px_max - px_min;
        // If still too small after clamping, expand in the other direction
        if (cw < min_crop_dim) {
            if (px_min > 0) {
                px_min = std::max(0, px_max - min_crop_dim);
            } else {
                px_max = std::min(img_w, px_min + min_crop_dim);
            }
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
            if (py_min > 0) {
                py_min = std::max(0, py_max - min_crop_dim);
            } else {
                py_max = std::min(img_h, py_min + min_crop_dim);
            }
            ch = py_max - py_min;
        }
    }

    ci.crop_x = px_min;
    ci.crop_y = py_min;
    ci.crop_w = cw;
    ci.crop_h = ch;

    // If crop covers >= 90% of image area, skip cropping (overhead not worth it)
    double crop_area = (double)cw * ch;
    double full_area = (double)img_w * img_h;
    if (crop_area >= 0.9 * full_area) {
        ci.crop_x = 0;
        ci.crop_y = 0;
        ci.crop_w = img_w;
        ci.crop_h = img_h;
        ci.was_cropped = false;
        ci.estimated_tokens_crop = ci.estimated_tokens_full;
    } else {
        ci.was_cropped = true;
        ci.estimated_tokens_crop = estimate_tokens(cw, ch);
    }

    return ci;
}

// Crop a bitmap to the specified region.
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

// Individual visual token pruning using llama_memory_seq_rm_pos_ext().
// With M-RoPE, all image tokens in a frame share the same temporal position
// (pos_0), but each has unique 2D spatial coordinates (ext_x, ext_y).
// The new API matches on (pos, ext_x, ext_y) to target individual tokens.

// Round 1: Coarse patch-level pruning via motion priors + attention
// Groups visual tokens by their spatial analysis patch, computes a per-patch
// hybrid score (motion + attention), and removes the lowest-scoring patches.
static int do_patch_level_pruning(
    llama_context * lctx,
    visual_token_map & token_map,
    const attention_capture_state & capture_state,
    const sintel_params & sparams) {

    (void)capture_state;

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        LOG_ERR("Failed to get memory for pruning\n");
        return 0;
    }

    // Score each spatial analysis patch by combining motion + attention
    struct patch_score {
        std::string key;
        float score;
        std::vector<int> token_indices; // indices into token_map.tokens
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

        // Normalize motion score to [0, 1] range
        float norm_motion = std::min(avg_motion / 10.0f, 1.0f);

        // Combined score: higher = more important
        float combined = (1.0f - sparams.motion_weight) * avg_attn +
                         sparams.motion_weight * norm_motion;

        patch_score ps;
        ps.key = kv.first;
        ps.score = combined;
        ps.token_indices = active_indices;
        patches.push_back(ps);
    }

    if (patches.empty()) return 0;

    // Sort by score (ascending = least important first)
    std::sort(patches.begin(), patches.end(),
        [](const patch_score & a, const patch_score & b) { return a.score < b.score; });

    // Determine how many patches to prune
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

            // Remove individual token from KV cache using 2D-aware removal
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

// Round 2: Fine per-token pruning via accumulated attention scores
// Among remaining (non-pruned) visual tokens, rank by individual attention
// score and remove the lowest-scoring ones.
static int do_token_level_pruning(
    llama_context * lctx,
    visual_token_map & token_map,
    const attention_capture_state & capture_state,
    const sintel_params & sparams) {

    (void)capture_state;

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        LOG_ERR("Failed to get memory for pruning\n");
        return 0;
    }

    // Collect all remaining (non-pruned) visual tokens with their scores
    struct token_score {
        int idx;       // index into token_map.tokens
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
            // Fall back to motion score if no attention data
            score = std::min(t.motion_score / 10.0f, 1.0f);
        }

        token_score ts;
        ts.idx = i;
        ts.score = score;
        active_tokens.push_back(ts);
    }

    if (active_tokens.empty()) return 0;

    // Sort by score (ascending = least important first)
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

// Prefill-stage pruning: remove low-motion tokens from a single frame
// immediately after it is decoded into the KV cache, before the next frame
// is processed. This reduces the KV cache size that subsequent frames
// attend to during prefill, directly lowering TTFT.
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

    // Collect non-pruned tokens for this frame
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

    // Sort by motion ascending (lowest motion = least important = prune first)
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

    adaptive_vlm_context(common_params & params)
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
// Section 8: Inference function with spatial cropping
// ============================================================================

struct inference_result {
    std::string generated_text;
    int   n_tokens_generated = 0;
    float encode_time_ms     = 0.0f;
    float inference_time_ms  = 0.0f;

    // Embedding cache metrics (populated by run_inference_with_embedding_cache)
    int   total_tokens_encoded = 0;   // tokens actually sent through ViT
    int   total_tokens_cached  = 0;   // tokens reused from cache
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
    };
    std::vector<per_frame_cache_info> frame_cache_infos;

    // LLM-side visual token pruning metrics
    int   n_visual_tokens_initial      = 0;  // before any pruning
    int   n_visual_tokens_after_round1 = 0;  // after patch pruning
    int   n_visual_tokens_after_round2 = 0;  // after token pruning
    float pruning_overhead_ms          = 0.0f; // time spent on scoring + pruning
    int   n_tokens_pruned_round1       = 0;
    int   n_tokens_pruned_round2       = 0;

    // Prefill-stage pruning metrics
    int   n_tokens_pruned_prefill       = 0;
    float prefill_pruning_overhead_ms   = 0.0f;
};

static inference_result run_inference_with_crops(
    adaptive_vlm_context & ctx,
    const std::vector<std::string> & frame_paths,
    const std::vector<frame_spatial_analysis> & spatial_analyses,
    const sintel_params & sparams,
    const std::string & prompt, int n_predict,
    std::vector<frame_crop_info> & crop_infos_out) {

    inference_result result;
    int64_t t_start = ggml_time_ms();

    // Load frames as bitmaps, applying spatial crops where beneficial
    mtmd::bitmaps bitmaps;
    crop_infos_out.clear();

    for (size_t i = 0; i < frame_paths.size(); i++) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), frame_paths[i].c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", frame_paths[i].c_str());
            continue;
        }

        int img_w = (int)bmp.nx();
        int img_h = (int)bmp.ny();

        // Frame 0 is always the keyframe: send full image for scene context
        if (i == 0) {
            frame_crop_info ci;
            ci.frame_index = (i < spatial_analyses.size()) ? spatial_analyses[i].frame_index : (int)(i + 1);
            ci.orig_w = img_w;
            ci.orig_h = img_h;
            ci.crop_x = 0;
            ci.crop_y = 0;
            ci.crop_w = img_w;
            ci.crop_h = img_h;
            ci.was_cropped = false;
            ci.estimated_tokens_full = estimate_tokens(img_w, img_h);
            ci.estimated_tokens_crop = ci.estimated_tokens_full;
            crop_infos_out.push_back(ci);
            bitmaps.entries.push_back(std::move(bmp));
            LOG_INF("  Frame %d: KEYFRAME (full %dx%d, ~%d tokens)\n",
                    ci.frame_index, img_w, img_h, ci.estimated_tokens_full);
            continue;
        }

        // For subsequent frames, check if we have spatial analysis and can crop
        bool cropped = false;
        if (i < spatial_analyses.size()) {
            const auto & sa = spatial_analyses[i];
            frame_crop_info ci = compute_crop_region(sa, img_w, img_h,
                                                      sparams.patch_grid,
                                                      sparams.crop_padding,
                                                      sparams.min_crop_dim);

            if (ci.was_cropped) {
                mtmd::bitmap cropped_bmp = crop_bitmap(bmp, ci);
                bitmaps.entries.push_back(std::move(cropped_bmp));
                cropped = true;
                LOG_INF("  Frame %d: CROPPED %dx%d -> %dx%d @ (%d,%d) (~%d -> ~%d tokens, %.0f%% reduction)\n",
                        ci.frame_index, img_w, img_h, ci.crop_w, ci.crop_h,
                        ci.crop_x, ci.crop_y,
                        ci.estimated_tokens_full, ci.estimated_tokens_crop,
                        (1.0f - (float)ci.estimated_tokens_crop / ci.estimated_tokens_full) * 100.0f);
            } else {
                bitmaps.entries.push_back(std::move(bmp));
                LOG_INF("  Frame %d: FULL (crop not beneficial, %dx%d, ~%d tokens)\n",
                        ci.frame_index, img_w, img_h, ci.estimated_tokens_full);
            }
            crop_infos_out.push_back(ci);
        } else {
            // No spatial analysis available, send full frame
            frame_crop_info ci;
            ci.frame_index = (int)(i + 1);
            ci.orig_w = img_w;
            ci.orig_h = img_h;
            ci.crop_x = 0;
            ci.crop_y = 0;
            ci.crop_w = img_w;
            ci.crop_h = img_h;
            ci.was_cropped = false;
            ci.estimated_tokens_full = estimate_tokens(img_w, img_h);
            ci.estimated_tokens_crop = ci.estimated_tokens_full;
            crop_infos_out.push_back(ci);
            bitmaps.entries.push_back(std::move(bmp));
            LOG_INF("  Frame %d: FULL (no spatial data, %dx%d, ~%d tokens)\n",
                    ci.frame_index, img_w, img_h, ci.estimated_tokens_full);
        }
    }

    if (bitmaps.entries.empty()) {
        LOG_ERR("No images loaded for inference\n");
        return result;
    }

    LOG_INF("Loaded %zu images for inference (with spatial cropping)\n", bitmaps.entries.size());

    // Build prompt with media markers
    std::string full_content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) {
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

    // Evaluate all chunks
    int64_t t_encode_start = ggml_time_ms();
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                                ctx.lctx,
                                chunks.ptr.get(),
                                ctx.n_past,
                                0,
                                ctx.n_batch,
                                true,
                                &new_n_past)) {
        LOG_ERR("Failed to evaluate chunks\n");
        return result;
    }
    ctx.n_past = new_n_past;
    result.encode_time_ms = (float)(ggml_time_ms() - t_encode_start);

    LOG_INF("Chunks evaluated. n_past = %d, encode_time = %.1f ms\n",
            (int)ctx.n_past, result.encode_time_ms);

    // Generate tokens
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

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("Failed to decode token\n");
            break;
        }
    }
    printf("\n");

    result.generated_text    = generated_text;
    result.n_tokens_generated = (int)generated_tokens.size();
    result.inference_time_ms  = (float)(ggml_time_ms() - t_gen_start);

    LOG_INF("Generated %d tokens in %.1f ms (%.1f t/s)\n",
            result.n_tokens_generated, result.inference_time_ms,
            result.n_tokens_generated / (result.inference_time_ms / 1000.0f));

    return result;
}

// ============================================================================
// Section 8b: Inference with encoder-level embedding cache
// ============================================================================

static inference_result run_inference_with_embedding_cache(
    adaptive_vlm_context & ctx,
    const std::vector<std::string> & frame_paths,
    const std::vector<frame_spatial_analysis> & spatial_analyses,
    const sintel_params & sparams,
    const std::string & prompt, int n_predict,
    std::vector<frame_crop_info> & crop_infos_out,
    attention_capture_state * ext_capture_state = nullptr) {

    inference_result result;
    int64_t t_start = ggml_time_ms();
    crop_infos_out.clear();

    const llama_model * model = llama_get_model(ctx.lctx);
    int n_embd = llama_model_n_embd_inp(model);

    embedding_cache cache;
    mtmd::input_chunk_ptr keyframe_chunk_ptr(nullptr);
    int nx_full = 0, ny_full = 0;

    // ---------------------------------------------------------------
    // Phase 1: Load ALL frames as full-size bitmaps for tokenization.
    // We need every frame full-size so mtmd_tokenize sees the correct
    // number of <media> markers and produces the right chunk sequence.
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // Phase 2: Build prompt with media markers and tokenize
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // Phase 3: Walk chunks — text chunks go through normal eval,
    //          image chunks go through per-patch cache path.
    //
    // Per-patch strategy:
    //   Keyframe: full ViT encode → cache all patch embeddings
    //   Subsequent: if active_ratio < skip_threshold → skip ViT, reuse cache
    //               else → full ViT encode → selective_update (only active patches)
    //   The LLM always sees the full nx*ny grid via keyframe_chunk metadata.
    // ---------------------------------------------------------------
    const float skip_threshold = 0.05f; // skip ViT if <5% of patches are active

    // Visual token map for LLM-side pruning
    visual_token_map vis_token_map;

    // Use external capture state (from main, registered with cb_eval) or a local dummy
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

    int64_t t_encode_total_start = ggml_time_ms();
    size_t n_chunks = chunks.size();
    int image_idx = 0;

    for (size_t ci = 0; ci < n_chunks; ci++) {
        const mtmd_input_chunk * chunk = chunks[ci];
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool is_last_chunk = (ci == n_chunks - 1);
            llama_pos new_n_past = ctx.n_past;
            int32_t ret = mtmd_helper_eval_chunk_single(
                ctx.ctx_vision.get(), ctx.lctx, chunk,
                ctx.n_past, 0, ctx.n_batch, is_last_chunk, &new_n_past);
            if (ret != 0) {
                LOG_ERR("Failed to evaluate text chunk %zu\n", ci);
                return result;
            }
            ctx.n_past = new_n_past;
            continue;
        }

        // --- Image chunk: per-patch cache encoding ---
        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (!image_tokens) {
            LOG_ERR("Image tokens null for chunk %zu\n", ci);
            return result;
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
        crop_ci.crop_x = 0;
        crop_ci.crop_y = 0;
        crop_ci.crop_w = img_w;
        crop_ci.crop_h = img_h;
        crop_ci.was_cropped = false;
        crop_ci.estimated_tokens_crop = n_tokens_chunk;

        bool can_use_cache = cache.valid && keyframe_chunk_ptr
                             && (chunk_nx == nx_full) && (chunk_ny == ny_full);

        // Record the start position for visual token map (before decode)
        llama_pos frame_pos_start = ctx.n_past;

        if (image_idx == 0) {
            // ============ KEYFRAME: encode full, store all patches in cache ============
            fci.is_keyframe = true;
            fci.is_full_encode = true;

            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                LOG_ERR("Failed to encode keyframe\n");
                return result;
            }
            fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

            nx_full = chunk_nx;
            ny_full = chunk_ny;
            cache.allocate(nx_full, ny_full, n_embd);
            cache.store_full(embd);

            keyframe_chunk_ptr.reset(mtmd_input_chunk_copy(chunk));

            fci.n_tokens_encoded = n_tokens_chunk;
            fci.n_tokens_cached  = 0;
            fci.used_cache = false;

            llama_pos new_n_past = ctx.n_past;
            ret = mtmd_helper_decode_image_chunk(
                ctx.ctx_vision.get(), ctx.lctx, chunk,
                cache.embeddings.data(), ctx.n_past, 0, ctx.n_batch, &new_n_past);
            if (ret != 0) {
                LOG_ERR("Failed to decode keyframe embeddings\n");
                return result;
            }
            ctx.n_past = new_n_past;

            LOG_INF("  Frame %d: KEYFRAME (%dx%d, %dx%d grid, %d tokens, encode %.1f ms)\n",
                    fci.frame_index, img_w, img_h, chunk_nx, chunk_ny,
                    n_tokens_chunk, fci.encode_time_ms);

        } else if (can_use_cache && image_idx < (int)spatial_analyses.size()) {
            // ============ PER-PATCH CACHE DECISION ============
            fci.is_keyframe = false;

            // Build per-token active mask from spatial analysis
            auto patch_active = build_patch_active_mask(
                spatial_analyses[image_idx], sparams.patch_grid, chunk_nx, chunk_ny,
                sparams.patch_reuse_motion);

            int n_active = 0;
            for (bool a : patch_active) { if (a) n_active++; }
            float active_ratio = (float)n_active / n_tokens_chunk;

            if (active_ratio < skip_threshold) {
                // ---- SKIP ViT: reuse entire cache as-is ----
                fci.is_full_encode = false;
                fci.used_cache = true;
                fci.encode_time_ms = 0.0f;
                fci.n_tokens_encoded = 0;
                fci.n_tokens_cached  = n_tokens_chunk;

                llama_pos new_n_past = ctx.n_past;
                int32_t ret = mtmd_helper_decode_image_chunk(
                    ctx.ctx_vision.get(), ctx.lctx,
                    keyframe_chunk_ptr.get(),
                    cache.embeddings.data(),
                    ctx.n_past, 0, ctx.n_batch, &new_n_past);
                if (ret != 0) {
                    LOG_ERR("Failed to decode cached frame %d\n", fci.frame_index);
                    return result;
                }
                ctx.n_past = new_n_past;

                crop_ci.estimated_tokens_crop = 0; // 0 tokens encoded
                crop_ci.was_cropped = true;

                LOG_INF("  Frame %d: SKIP ViT (%.0f%% active < %.0f%% threshold, "
                        "0/%d tokens encoded, all cached)\n",
                        fci.frame_index, active_ratio * 100.0f,
                        skip_threshold * 100.0f, n_tokens_chunk);

            } else {
                // ---- CROP-ENCODE active region, then SELECTIVE splice ----
                // 1. Find bounding box of active patches in the ViT token grid
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

                // Bounding box in ViT tokens (inclusive)
                int bb_nx = tx_max - tx_min + 1;
                int bb_ny = ty_max - ty_min + 1;
                int bb_tokens = bb_nx * bb_ny;

                // Convert token BB to pixel coords in original image
                // Each ViT token = 28px in preprocessed space
                float inv_sx = (float)img_w / (chunk_nx * 28);
                float inv_sy = (float)img_h / (chunk_ny * 28);
                int raw_cx = (int)(tx_min * 28 * inv_sx);
                int raw_cy = (int)(ty_min * 28 * inv_sy);
                int raw_cw = (int)(bb_nx * 28 * inv_sx);
                int raw_ch = (int)(bb_ny * 28 * inv_sy);

                // Clamp to image bounds
                raw_cx = std::max(0, std::min(raw_cx, img_w - 1));
                raw_cy = std::max(0, std::min(raw_cy, img_h - 1));
                raw_cw = std::min(raw_cw, img_w - raw_cx);
                raw_ch = std::min(raw_ch, img_h - raw_cy);

                // Crop the bitmap to the active region
                frame_crop_info tmp_ci;
                tmp_ci.crop_x = raw_cx;
                tmp_ci.crop_y = raw_cy;
                tmp_ci.crop_w = raw_cw;
                tmp_ci.crop_h = raw_ch;
                mtmd::bitmap cropped_bmp = crop_bitmap(full_bitmaps.entries[image_idx], tmp_ci);

                // Tokenize the crop as standalone image
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

                // Find the image chunk in crop tokenization result
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

                // Verify crop dimensions match expected bounding box
                bool crop_ok = crop_chunk && (crop_enc_nx == bb_nx) && (crop_enc_ny == bb_ny);

                if (crop_ok) {
                    // ---- Encode the CROP (smaller, faster ViT) ----
                    fci.is_full_encode = false;

                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), crop_chunk);
                    if (ret != 0) {
                        LOG_ERR("Failed to encode crop for frame %d\n", fci.frame_index);
                        // Fall through to full encode below
                        crop_ok = false;
                    } else {
                        fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

                        float * crop_embd = mtmd_get_output_embd(ctx.ctx_vision.get());

                        // Splice: for each patch in the crop's grid, only copy to cache
                        // if the per-patch mask says it's active. Inactive patches within
                        // the bounding box keep their cached values.
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

                        // Decode the full cache (keyframe grid) to LLM
                        llama_pos new_n_past = ctx.n_past;
                        ret = mtmd_helper_decode_image_chunk(
                            ctx.ctx_vision.get(), ctx.lctx,
                            keyframe_chunk_ptr.get(),
                            cache.embeddings.data(),
                            ctx.n_past, 0, ctx.n_batch, &new_n_past);
                        if (ret != 0) {
                            LOG_ERR("Failed to decode frame %d\n", fci.frame_index);
                            return result;
                        }
                        ctx.n_past = new_n_past;

                        crop_ci.crop_x = raw_cx;
                        crop_ci.crop_y = raw_cy;
                        crop_ci.crop_w = raw_cw;
                        crop_ci.crop_h = raw_ch;
                        crop_ci.was_cropped = true;
                        crop_ci.estimated_tokens_crop = bb_tokens;

                        LOG_INF("  Frame %d: CROP+SPLICE (%d/%d patches updated, %d cached "
                                "[%d outside BB + %d inactive in BB], crop %dx%d tokens, "
                                "%.0f%% active, encode %.1f ms, splice %.1f ms)\n",
                                fci.frame_index, n_updated, n_tokens_chunk, n_kept_total,
                                n_kept_outside, n_kept_in_bb, bb_nx, bb_ny,
                                active_ratio * 100.0f, fci.encode_time_ms, fci.splice_time_ms);
                    }
                }

                if (!crop_ok) {
                    // ---- FALLBACK: encode full frame, selective update ----
                    fci.is_full_encode = true;

                    int64_t t_enc = ggml_time_ms();
                    int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
                    if (ret != 0) {
                        LOG_ERR("Failed to encode frame %d (fallback)\n", fci.frame_index);
                        return result;
                    }
                    fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

                    float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

                    int64_t t_splice = ggml_time_ms();
                    int n_updated = 0, n_kept = 0;
                    cache.selective_update(embd, patch_active, n_updated, n_kept);
                    fci.splice_time_ms = (float)(ggml_time_ms() - t_splice);

                    fci.used_cache = (n_kept > 0);
                    fci.n_tokens_encoded = n_updated;
                    fci.n_tokens_cached  = n_kept;

                    llama_pos new_n_past = ctx.n_past;
                    ret = mtmd_helper_decode_image_chunk(
                        ctx.ctx_vision.get(), ctx.lctx,
                        keyframe_chunk_ptr.get(),
                        cache.embeddings.data(),
                        ctx.n_past, 0, ctx.n_batch, &new_n_past);
                    if (ret != 0) {
                        LOG_ERR("Failed to decode frame %d (fallback)\n", fci.frame_index);
                        return result;
                    }
                    ctx.n_past = new_n_past;

                    crop_ci.estimated_tokens_crop = n_updated;
                    crop_ci.was_cropped = (n_kept > 0);

                    LOG_INF("  Frame %d: FALLBACK full encode (%d/%d updated, %d cached, "
                            "crop dim mismatch %dx%d vs expected %dx%d, encode %.1f ms)\n",
                            fci.frame_index, n_updated, n_tokens_chunk, n_kept,
                            crop_enc_nx, crop_enc_ny, bb_nx, bb_ny,
                            fci.encode_time_ms);
                }
            }

        } else {
            // ============ NO CACHE AVAILABLE: full encode ============
            fci.is_keyframe = false;
            fci.is_full_encode = true;

            int64_t t_enc = ggml_time_ms();
            int32_t ret = mtmd_encode_chunk(ctx.ctx_vision.get(), chunk);
            if (ret != 0) {
                LOG_ERR("Failed to encode frame %d\n", fci.frame_index);
                return result;
            }
            fci.encode_time_ms = (float)(ggml_time_ms() - t_enc);

            float * embd = mtmd_get_output_embd(ctx.ctx_vision.get());

            if (chunk_nx == nx_full && chunk_ny == ny_full) {
                cache.store_full(embd);
            }

            fci.n_tokens_encoded = n_tokens_chunk;
            fci.n_tokens_cached  = 0;
            fci.used_cache = false;

            llama_pos new_n_past = ctx.n_past;
            const mtmd_input_chunk * decode_chunk = keyframe_chunk_ptr ? keyframe_chunk_ptr.get() : chunk;
            ret = mtmd_helper_decode_image_chunk(
                ctx.ctx_vision.get(), ctx.lctx, decode_chunk,
                (chunk_nx == nx_full && chunk_ny == ny_full) ? cache.embeddings.data() : embd,
                ctx.n_past, 0, ctx.n_batch, &new_n_past);
            if (ret != 0) {
                LOG_ERR("Failed to decode frame %d\n", fci.frame_index);
                return result;
            }
            ctx.n_past = new_n_past;

            LOG_INF("  Frame %d: FULL ENCODE (no spatial data, %d tokens, %.1f ms)\n",
                    fci.frame_index, n_tokens_chunk, fci.encode_time_ms);
        }

        result.frame_cache_infos.push_back(fci);
        result.total_tokens_encoded += fci.n_tokens_encoded;
        result.total_tokens_cached  += fci.n_tokens_cached;
        result.total_encode_time_ms += fci.encode_time_ms;
        crop_infos_out.push_back(crop_ci);

        // Populate visual token map for pruning
        if (sparams.enable_pruning || sparams.enable_prefill_pruning) {
            // Record frame range: M-RoPE uses n_pos = max(nx, ny) positions
            llama_pos frame_n_pos = (llama_pos)mtmd_input_chunk_get_n_pos(chunk);
            visual_token_map::frame_range fr;
            fr.start_pos = frame_pos_start;
            fr.end_pos   = frame_pos_start + frame_n_pos;
            fr.nx = chunk_nx;
            fr.ny = chunk_ny;
            fr.token_start_idx = (int)vis_token_map.tokens.size();
            fr.token_count = chunk_nx * chunk_ny;
            vis_token_map.frame_ranges.push_back(fr);

            // Get per-patch motion scores from spatial analysis
            float default_motion = 0.0f;
            if (image_idx < (int)spatial_analyses.size()) {
                // Use average motion from spatial analysis
                const auto & sa = spatial_analyses[image_idx];
                for (const auto & p : sa.patches) {
                    default_motion += p.mean_motion;
                }
                default_motion /= std::max(1, (int)sa.patches.size());
            }

            // Record each token in the grid with M-RoPE ext coordinates
            int frame_idx = (int)vis_token_map.frame_ranges.size() - 1;
            for (int ty = 0; ty < chunk_ny; ty++) {
                for (int tx = 0; tx < chunk_nx; tx++) {
                    // M-RoPE positions: pos_0 (temporal), pos_0+ty (ext_y), pos_0+tx (ext_x)
                    llama_pos tok_pos = frame_pos_start;
                    llama_pos tok_ext_x = frame_pos_start + tx;
                    llama_pos tok_ext_y = frame_pos_start + ty;

                    // Map this ViT token to its spatial analysis grid cell
                    int sa_gx = tx * sparams.patch_grid / chunk_nx;
                    int sa_gy = ty * sparams.patch_grid / chunk_ny;
                    sa_gx = std::min(sa_gx, sparams.patch_grid - 1);
                    sa_gy = std::min(sa_gy, sparams.patch_grid - 1);

                    // Compute per-token motion from spatial analysis patches
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

                    vis_token_map.add_token(tok_pos, tok_ext_x, tok_ext_y,
                                            frame_idx, ty, tx, tok_motion,
                                            sa_gy, sa_gx);
                }
            }
        }

        // Prefill-stage pruning: prune low-motion tokens from this frame
        // before the next frame is decoded, reducing KV cache size for
        // subsequent prefill steps. Skip keyframe (image_idx 0).
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

    // Estimate what full encoding would have cost
    if (!result.frame_cache_infos.empty()) {
        // Use keyframe encode time per token as baseline
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

    // ---------------------------------------------------------------
    // Phase 4: Generate tokens (with optional progressive pruning)
    // ---------------------------------------------------------------

    // Finalize attention capture state now that all images are decoded
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

    int64_t t_gen_start = ggml_time_ms();
    int64_t t_prune_total = 0;
    std::string generated_text;
    llama_tokens generated_tokens;

    for (int i = 0; i < n_predict; i++) {
        // Update attention capture state
        if (sparams.enable_pruning && capture_state.initialized) {
            capture_state.current_gen_step = i;
            capture_state.should_capture = (i % sparams.attn_sample_interval == 0);
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

        // Progressive pruning checkpoints
        if (sparams.enable_pruning && capture_state.initialized) {
            if (i == sparams.first_prune_step) {
                int64_t t_prune = ggml_time_ms();
                result.n_tokens_pruned_round1 = do_patch_level_pruning(
                    ctx.lctx, vis_token_map, capture_state, sparams);
                result.n_visual_tokens_after_round1 = vis_token_map.count_active();
                // Compact KV cache so n_kv shrinks and attention is actually faster
                llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
                t_prune_total += ggml_time_ms() - t_prune;
            }
            if (i == sparams.second_prune_step) {
                int64_t t_prune = ggml_time_ms();
                result.n_tokens_pruned_round2 = do_token_level_pruning(
                    ctx.lctx, vis_token_map, capture_state, sparams);
                result.n_visual_tokens_after_round2 = vis_token_map.count_active();
                // Compact KV cache again
                llama_memory_defragment(llama_get_memory(ctx.lctx), ctx.lctx);
                t_prune_total += ggml_time_ms() - t_prune;
                // Disable callback — pruning is done, no more overhead per token
                capture_state.disabled = true;
            }
        }
    }
    printf("\n");

    result.generated_text    = generated_text;
    result.n_tokens_generated = (int)generated_tokens.size();
    result.inference_time_ms  = (float)(ggml_time_ms() - t_gen_start);
    result.pruning_overhead_ms = (float)t_prune_total;

    // Fill in pruning metrics that weren't triggered
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

    return result;
}

// ============================================================================
// Section 9: JSON output (extended with crop metrics)
// ============================================================================

static void write_results_json(const std::string & output_path,
                                const sintel_params & sparams,
                                int n_total_frames,
                                const video_motion_profile & profile,
                                const std::vector<selected_frame> & selected,
                                const std::vector<frame_spatial_analysis> & spatial,
                                const std::vector<frame_crop_info> & crop_infos,
                                const inference_result & inf_result,
                                double motion_coverage,
                                double entropy,
                                float flow_time_ms,
                                float select_time_ms,
                                float total_time_ms) {
    json j;

    // Mode and video info
    j["mode"]             = sparams.baseline ? "baseline" : "adaptive_embedding_cache";
    j["video_name"]       = sparams.video_name;
    j["n_flow_files"]     = (int)profile.frame_stats.size();
    j["n_frames_total"]   = n_total_frames;
    j["global_complexity"] = profile.global_complexity;
    j["n_frames_selected"] = (int)selected.size();

    // Crop configuration
    j["crop_config"]["crop_padding"]  = sparams.crop_padding;
    j["crop_config"]["min_crop_dim"]  = sparams.min_crop_dim;

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

    // Per-frame crop info
    json crop_arr = json::array();
    int total_tokens_full = 0;
    int total_tokens_crop = 0;
    int n_frames_cropped  = 0;
    for (const auto & ci : crop_infos) {
        json cj;
        cj["frame_index"]           = ci.frame_index;
        cj["orig_w"]                = ci.orig_w;
        cj["orig_h"]                = ci.orig_h;
        cj["crop_x"]                = ci.crop_x;
        cj["crop_y"]                = ci.crop_y;
        cj["crop_w"]                = ci.crop_w;
        cj["crop_h"]                = ci.crop_h;
        cj["was_cropped"]           = ci.was_cropped;
        cj["estimated_tokens_full"] = ci.estimated_tokens_full;
        cj["estimated_tokens_crop"] = ci.estimated_tokens_crop;
        crop_arr.push_back(cj);

        total_tokens_full += ci.estimated_tokens_full;
        total_tokens_crop += ci.estimated_tokens_crop;
        if (ci.was_cropped) n_frames_cropped++;
    }
    j["per_frame_crops"] = crop_arr;

    // Inference results
    j["inference"]["generated_text"]    = inf_result.generated_text;
    j["inference"]["n_tokens"]          = inf_result.n_tokens_generated;
    j["inference"]["encode_time_ms"]    = inf_result.encode_time_ms;
    j["inference"]["generation_time_ms"] = inf_result.inference_time_ms;

    // Quality metrics (extended with crop stats)
    j["metrics"]["motion_coverage"]   = motion_coverage;
    j["metrics"]["shannon_entropy"]   = entropy;
    j["metrics"]["temporal_reduction"] = 1.0 - (double)selected.size() / n_total_frames;

    double avg_spatial_savings = 0.0;
    if (!spatial.empty()) {
        for (const auto & sa : spatial) avg_spatial_savings += sa.spatial_savings;
        avg_spatial_savings /= spatial.size();
    }
    j["metrics"]["avg_spatial_savings"] = avg_spatial_savings;

    // Crop-specific metrics
    j["metrics"]["total_tokens_full"]       = total_tokens_full;
    j["metrics"]["total_tokens_cropped"]    = total_tokens_crop;
    j["metrics"]["total_token_savings_pct"] = (total_tokens_full > 0)
        ? (1.0 - (double)total_tokens_crop / total_tokens_full) * 100.0 : 0.0;
    j["metrics"]["n_frames_cropped"]        = n_frames_cropped;
    j["metrics"]["n_frames_total_selected"] = (int)crop_infos.size();

    // Embedding cache metrics
    j["metrics"]["total_tokens_encoded"]        = inf_result.total_tokens_encoded;
    j["metrics"]["total_tokens_cached"]         = inf_result.total_tokens_cached;
    int total_encoder_tokens = inf_result.total_tokens_encoded + inf_result.total_tokens_cached;
    j["metrics"]["encoder_savings_pct"]         = (total_encoder_tokens > 0)
        ? 100.0 * inf_result.total_tokens_cached / total_encoder_tokens : 0.0;
    j["metrics"]["total_encode_time_ms"]        = inf_result.total_encode_time_ms;
    j["metrics"]["estimated_full_encode_time_ms"] = inf_result.estimated_full_encode_time_ms;
    j["metrics"]["encode_speedup"]              = (inf_result.total_encode_time_ms > 0)
        ? inf_result.estimated_full_encode_time_ms / inf_result.total_encode_time_ms : 1.0;

    // Per-frame cache info
    json cache_arr = json::array();
    for (const auto & fci : inf_result.frame_cache_infos) {
        json fc;
        fc["frame_index"]      = fci.frame_index;
        fc["is_keyframe"]      = fci.is_keyframe;
        fc["used_cache"]       = fci.used_cache;
        fc["is_full_encode"]   = fci.is_full_encode;
        fc["n_tokens_encoded"] = fci.n_tokens_encoded;
        fc["n_tokens_cached"]  = fci.n_tokens_cached;
        fc["encode_time_ms"]   = fci.encode_time_ms;
        fc["splice_time_ms"]   = fci.splice_time_ms;
        cache_arr.push_back(fc);
    }
    j["per_frame_cache"] = cache_arr;

    // LLM-side pruning metrics
    if (sparams.enable_pruning) {
        j["pruning"]["enabled"]                      = true;
        j["pruning"]["n_visual_tokens_initial"]      = inf_result.n_visual_tokens_initial;
        j["pruning"]["n_visual_tokens_after_round1"] = inf_result.n_visual_tokens_after_round1;
        j["pruning"]["n_visual_tokens_after_round2"] = inf_result.n_visual_tokens_after_round2;
        j["pruning"]["n_tokens_pruned_round1"]       = inf_result.n_tokens_pruned_round1;
        j["pruning"]["n_tokens_pruned_round2"]       = inf_result.n_tokens_pruned_round2;
        j["pruning"]["pruning_overhead_ms"]          = inf_result.pruning_overhead_ms;
        j["pruning"]["total_reduction_pct"]          = (inf_result.n_visual_tokens_initial > 0)
            ? (1.0 - (double)inf_result.n_visual_tokens_after_round2 / inf_result.n_visual_tokens_initial) * 100.0
            : 0.0;
        j["pruning"]["config"]["prune_layer"]          = sparams.prune_layer;
        j["pruning"]["config"]["attn_sample_interval"] = sparams.attn_sample_interval;
        j["pruning"]["config"]["first_prune_step"]     = sparams.first_prune_step;
        j["pruning"]["config"]["second_prune_step"]    = sparams.second_prune_step;
        j["pruning"]["config"]["patch_keep_ratio"]     = sparams.patch_keep_ratio;
        j["pruning"]["config"]["token_keep_ratio"]     = sparams.token_keep_ratio;
        j["pruning"]["config"]["motion_weight"]        = sparams.motion_weight;
    } else {
        j["pruning"]["enabled"] = false;
    }

    // Prefill-stage pruning metrics
    if (sparams.enable_prefill_pruning) {
        j["prefill_pruning"]["enabled"]                = true;
        j["prefill_pruning"]["n_tokens_pruned"]        = inf_result.n_tokens_pruned_prefill;
        j["prefill_pruning"]["overhead_ms"]            = inf_result.prefill_pruning_overhead_ms;
        j["prefill_pruning"]["config"]["keep_ratio"]   = sparams.prefill_keep_ratio;
    } else {
        j["prefill_pruning"]["enabled"] = false;
    }

    // Timing
    j["timing"]["flow_analysis_ms"]           = flow_time_ms;
    j["timing"]["frame_selection_ms"]         = select_time_ms;
    j["timing"]["encode_ms"]                  = inf_result.encode_time_ms;
    j["timing"]["generation_ms"]              = inf_result.inference_time_ms;
    j["timing"]["pruning_overhead_ms"]        = inf_result.pruning_overhead_ms;
    j["timing"]["prefill_pruning_overhead_ms"]= inf_result.prefill_pruning_overhead_ms;
    j["timing"]["total_ms"]                   = total_time_ms;

    // Parameters
    j["params"]["sintel_dir"]            = sparams.sintel_dir;
    j["params"]["video_name"]            = sparams.video_name;
    j["params"]["min_frames"]            = sparams.min_frames;
    j["params"]["max_frames"]            = sparams.max_frames;
    j["params"]["base_frames"]           = sparams.base_frames;
    j["params"]["motion_threshold"]      = sparams.motion_threshold;
    j["params"]["motion_percentile"]     = sparams.motion_percentile;
    j["params"]["patch_grid"]            = sparams.patch_grid;
    j["params"]["spatial_threshold"]     = sparams.spatial_threshold;
    j["params"]["crop_padding"]          = sparams.crop_padding;
    j["params"]["min_crop_dim"]          = sparams.min_crop_dim;
    j["params"]["patch_reuse_motion"]    = sparams.patch_reuse_motion;
    j["params"]["enable_pruning"]        = sparams.enable_pruning;
    j["params"]["enable_prefill_pruning"]= sparams.enable_prefill_pruning;
    j["params"]["prefill_keep_ratio"]    = sparams.prefill_keep_ratio;

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

    sintel_params sparams;
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
        ? "Analyze these video frames from MPI-Sintel. Describe the scene, motion patterns, and key visual elements."
        : params.prompt;

    const char * mode_str = sparams.baseline ? "BASELINE (uniform)" :
                            (sparams.enable_pruning ? "ADAPTIVE+EMBEDDING_CACHE+PRUNING" : "ADAPTIVE+EMBEDDING_CACHE");

    LOG_INF("\n=== Spatio-Temporal VLM Optimizer with Embedding Cache [%s] ===\n", mode_str);
    LOG_INF("Video: %s\n", sparams.video_name.c_str());
    LOG_INF("Dataset: %s\n", sparams.sintel_dir.c_str());
    LOG_INF("Crop padding: %d px, min crop dim: %d px\n", sparams.crop_padding, sparams.min_crop_dim);

    int64_t t_total_start = ggml_time_ms();

    int n_total_frames = count_video_frames(sparams.sintel_dir, sparams.video_name);
    if (n_total_frames == 0) {
        LOG_ERR("No frames found for video '%s'. Exiting.\n", sparams.video_name.c_str());
        return 1;
    }
    LOG_INF("Total frames available: %d\n", n_total_frames);

    std::string flow_dir = sparams.sintel_dir + "/training/flow";
    video_motion_profile profile;
    std::vector<selected_frame> selected;
    std::vector<frame_spatial_analysis> spatial_analyses;
    std::vector<frame_crop_info> crop_infos;
    float flow_time_ms = 0.0f;
    float select_time_ms = 0.0f;

    if (sparams.baseline) {
        // ============ BASELINE MODE ============
        LOG_INF("\n--- Baseline: Uniform Frame Selection ---\n");
        int64_t t0 = ggml_time_ms();

        selected = select_frames_uniform(n_total_frames, sparams.base_frames);
        select_time_ms = (float)(ggml_time_ms() - t0);

        LOG_INF("\n--- Building motion profile for metrics ---\n");
        t0 = ggml_time_ms();
        profile = build_motion_profile(flow_dir, sparams.video_name,
                                        sparams.motion_threshold,
                                        sparams.motion_percentile);
        flow_time_ms = (float)(ggml_time_ms() - t0);

    } else {
        // ============ ADAPTIVE + EMBEDDING CACHE MODE ============
        LOG_INF("Frame range: [%d, %d], base: %d\n", sparams.min_frames, sparams.max_frames, sparams.base_frames);

        // Step 1: Build motion profile
        LOG_INF("\n--- Step 1: Motion Analysis ---\n");
        int64_t t0 = ggml_time_ms();

        profile = build_motion_profile(flow_dir, sparams.video_name,
                                        sparams.motion_threshold,
                                        sparams.motion_percentile);
        flow_time_ms = (float)(ggml_time_ms() - t0);
        LOG_INF("Flow analysis completed in %.1f ms\n", flow_time_ms);

        if (profile.frame_stats.empty()) {
            LOG_ERR("No motion data available. Exiting.\n");
            return 1;
        }

        // Step 2: Select frames adaptively
        LOG_INF("\n--- Step 2: Adaptive Frame Selection ---\n");
        t0 = ggml_time_ms();

        selected = select_frames(profile,
                                  sparams.min_frames,
                                  sparams.max_frames,
                                  sparams.base_frames,
                                  profile.global_complexity,
                                  sparams.selection_percentile);
        select_time_ms = (float)(ggml_time_ms() - t0);

        if (selected.empty()) {
            LOG_ERR("No frames selected. Exiting.\n");
            return 1;
        }

        // Step 3: Post-selection complexity filtering
        {
            const float complexity_floor = 0.05f;
            int before_count = (int)selected.size();
            std::vector<selected_frame> filtered;
            for (const auto & sf : selected) {
                if (sf.complexity_score >= complexity_floor || (int)filtered.size() < sparams.min_frames) {
                    filtered.push_back(sf);
                } else {
                    LOG_INF("  Dropping frame %d (complexity %.4f < %.4f)\n",
                            sf.frame_index, sf.complexity_score, complexity_floor);
                }
            }
            if ((int)filtered.size() < sparams.min_frames) {
                filtered = selected;
                if ((int)filtered.size() > sparams.min_frames) {
                    std::sort(filtered.begin(), filtered.end(), [](const selected_frame & a, const selected_frame & b) {
                        return a.complexity_score > b.complexity_score;
                    });
                    filtered.resize(sparams.min_frames);
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
            if (read_flo_file(flo_path, flow)) {
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
        {
            const float min_active_ratio = 0.25f;
            int before_count = (int)selected.size();
            std::vector<selected_frame> spatially_filtered;
            std::vector<frame_spatial_analysis> kept_spatial;

            for (size_t i = 0; i < selected.size(); i++) {
                bool keep = true;
                for (const auto & sa : spatial_analyses) {
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

            if ((int)spatially_filtered.size() < sparams.min_frames) {
                spatially_filtered = selected;
            }

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
    }

    // Build frame paths for inference
    std::vector<std::string> frame_paths;
    for (const auto & sf : selected) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
        frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }

    // Initialize model and run inference with embedding cache
    LOG_INF("\n--- Step 6: Model Inference with Embedding Cache ---\n");

    // Set up the attention capture callback for pruning if enabled
    // The callback state will be populated later in run_inference_with_embedding_cache
    // but needs to be registered before context creation
    attention_capture_state main_capture_state = {};
    main_capture_state.target_layer = sparams.prune_layer;
    main_capture_state.sample_interval = sparams.attn_sample_interval;
    main_capture_state.initialized = false;  // will be set true during inference
    main_capture_state.should_capture = false;
    main_capture_state.disabled = false;
    main_capture_state.token_map = nullptr;  // will be set during inference

    if (sparams.enable_pruning) {
        params.cb_eval = attn_softmax_callback;
        params.cb_eval_user_data = &main_capture_state;
        LOG_INF("Registered attention capture callback (layer %d)\n", sparams.prune_layer);
    }

    adaptive_vlm_context ctx(params);

    LOG_INF("Running inference with %zu frames [%s]...\n", frame_paths.size(), mode_str);
    LOG_INF("Prompt: %s\n", prompt.c_str());

    auto inf_result = run_inference_with_embedding_cache(ctx, frame_paths, spatial_analyses,
                                                          sparams, prompt, n_predict,
                                                          crop_infos,
                                                          sparams.enable_pruning ? &main_capture_state : nullptr);

    // Compute metrics
    LOG_INF("\n--- Metrics ---\n");
    double motion_cov = compute_motion_coverage(profile, selected);
    double entropy = compute_shannon_entropy(inf_result.generated_text);
    float total_time_ms = (float)(ggml_time_ms() - t_total_start);

    double temporal_reduction = 1.0 - (double)selected.size() / n_total_frames;

    // Crop summary
    int total_tokens_full = 0, total_tokens_crop = 0, n_cropped = 0;
    for (const auto & ci : crop_infos) {
        total_tokens_full += ci.estimated_tokens_full;
        total_tokens_crop += ci.estimated_tokens_crop;
        if (ci.was_cropped) n_cropped++;
    }
    double token_savings_pct = (total_tokens_full > 0)
        ? (1.0 - (double)total_tokens_crop / total_tokens_full) * 100.0 : 0.0;

    // Encoder cache summary
    int total_enc_tokens = inf_result.total_tokens_encoded + inf_result.total_tokens_cached;
    double encoder_savings_pct = (total_enc_tokens > 0)
        ? 100.0 * inf_result.total_tokens_cached / total_enc_tokens : 0.0;

    LOG_INF("Mode:               %s\n", mode_str);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Shannon entropy:    %.2f bits\n", entropy);
    LOG_INF("Temporal reduction: %.1f%%\n", temporal_reduction * 100.0);
    LOG_INF("Tokens generated:   %d\n", inf_result.n_tokens_generated);
    LOG_INF("Frames cropped:     %d / %zu\n", n_cropped, crop_infos.size());
    LOG_INF("Est. token savings: %.1f%% (%d -> %d tokens)\n",
            token_savings_pct, total_tokens_full, total_tokens_crop);
    LOG_INF("Encoder savings:    %.1f%% (%d encoded, %d cached)\n",
            encoder_savings_pct, inf_result.total_tokens_encoded, inf_result.total_tokens_cached);
    LOG_INF("Encode speedup:     %.2fx (%.1f ms vs est. %.1f ms full)\n",
            (inf_result.total_encode_time_ms > 0)
                ? inf_result.estimated_full_encode_time_ms / inf_result.total_encode_time_ms : 1.0f,
            inf_result.total_encode_time_ms, inf_result.estimated_full_encode_time_ms);
    LOG_INF("Total time:         %.1f ms\n", total_time_ms);

    // Write JSON output
    std::string json_path = sparams.output_json.empty()
        ? ("results_" + sparams.video_name + (sparams.baseline ? "_baseline" :
            (sparams.enable_pruning ? "_embedding_cache_pruning" : "_embedding_cache")) + ".json")
        : sparams.output_json;

    write_results_json(json_path, sparams, n_total_frames, profile, selected, spatial_analyses,
                       crop_infos, inf_result, motion_cov, entropy, flow_time_ms, select_time_ms,
                       total_time_ms);

    // Print performance summary
    LOG_INF("\n=== Summary [%s] ===\n", mode_str);
    LOG_INF("Video:              %s\n", sparams.video_name.c_str());
    LOG_INF("Frames:             %zu / %d selected\n", selected.size(), n_total_frames);
    LOG_INF("Frames cropped:     %d / %zu\n", n_cropped, crop_infos.size());
    LOG_INF("Global complexity:  %.3f\n", profile.global_complexity);
    LOG_INF("Motion coverage:    %.1f%%\n", motion_cov * 100.0);
    LOG_INF("Temporal reduction: %.1f%%\n", temporal_reduction * 100.0);
    LOG_INF("Token savings:      %.1f%%\n", token_savings_pct);
    LOG_INF("Encoder savings:    %.1f%%\n", encoder_savings_pct);

    if (sparams.enable_pruning && inf_result.n_visual_tokens_initial > 0) {
        double prune_pct = (1.0 - (double)inf_result.n_visual_tokens_after_round2
                                  / inf_result.n_visual_tokens_initial) * 100.0;
        LOG_INF("LLM pruning:        %d -> %d -> %d tokens (%.1f%% reduction)\n",
                inf_result.n_visual_tokens_initial,
                inf_result.n_visual_tokens_after_round1,
                inf_result.n_visual_tokens_after_round2,
                prune_pct);
        LOG_INF("Pruning overhead:   %.1f ms\n", inf_result.pruning_overhead_ms);
    }

    if (sparams.enable_prefill_pruning && inf_result.n_tokens_pruned_prefill > 0) {
        LOG_INF("Prefill pruning:    %d tokens removed (keep_ratio=%.2f, overhead=%.1f ms)\n",
                inf_result.n_tokens_pruned_prefill,
                sparams.prefill_keep_ratio,
                inf_result.prefill_pruning_overhead_ms);
    }

    LOG_INF("Output:             %s\n", json_path.c_str());

    llama_perf_context_print(ctx.lctx);

    return 0;
}
