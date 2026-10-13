// Adaptive Temporal Sampling VLM for MPI-Sintel
//
// Motion-guided frame selection with full ViT encoding. Uses optical flow
// analysis to adaptively determine the number of frames and which frames
// to select based on scene complexity. All selected frames are fully encoded
// through the vision transformer (no patch-level caching).
//
// Strategy: Analyze optical flow to compute per-frame complexity scores,
// select frames proportional to motion importance, filter low-complexity
// frames, then run standard full-frame ViT encoding + LLM inference.

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
    float       selection_percentile = 70.0f; // percentile for importance weight thresholding
    int         patch_grid    = 4;
    float       spatial_threshold = 0.05f;  // patch motion ratio threshold (Python: patch_motion_ratio)
    std::string output_json;
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
        } else {
            filtered.push_back(argv[i]);
        }
    }
    return filtered;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\nAdaptive Spatio-Temporal VLM Optimizer for MPI-Sintel\n\n");
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
    printf("  --output-json <f>   Output JSON file path\n\n");
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

    // Sort for percentile computation
    std::vector<float> sorted_mag = mag;
    std::sort(sorted_mag.begin(), sorted_mag.end());
    stats.max_magnitude = sorted_mag.back();

    int p_idx = std::min((int)(percentile * n), n - 1);
    stats.p95_magnitude = sorted_mag[p_idx];

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
    int   frame_index;   // 1-based (matches frame_XXXX.png)
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

    for (int k = 0; k < n_select; k++) {
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

struct inference_result {
    std::string generated_text;
    int   n_tokens_generated = 0;
    float encode_time_ms     = 0.0f;
    float inference_time_ms  = 0.0f;
};

static inference_result run_inference(adaptive_vlm_context & ctx,
                                      const std::vector<std::string> & frame_paths,
                                      const std::string & prompt,
                                      int n_predict) {
    inference_result result;
    int64_t t_start = ggml_time_ms();

    // Load frames as bitmaps
    mtmd::bitmaps bitmaps;
    for (const auto & path : frame_paths) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", path.c_str());
            continue;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    if (bitmaps.entries.empty()) {
        LOG_ERR("No images loaded for inference\n");
        return result;
    }

    LOG_INF("Loaded %zu images for inference\n", bitmaps.entries.size());

    // Build prompt with media markers (one per image)
    std::string full_content;
    for (size_t i = 0; i < bitmaps.entries.size(); i++) {
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
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                                ctx.lctx,
                                chunks.ptr.get(),
                                ctx.n_past,
                                0,         // seq_id
                                ctx.n_batch,
                                true,      // logits_last
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

        // Prepare next batch
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
                                float total_time_ms) {
    json j;

    // Mode and video info
    j["mode"]             = "adaptive";
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
    j["timing"]["total_ms"]            = total_time_ms;

    // Parameters
    j["params"]["sintel_dir"]         = sparams.sintel_dir;
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

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    // Set default model paths if not provided
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

    const char * mode_str = "ADAPTIVE (motion-guided)";

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
    video_motion_profile profile;
    std::vector<selected_frame> selected;
    std::vector<frame_spatial_analysis> spatial_analyses;
    float flow_time_ms = 0.0f;
    float select_time_ms = 0.0f;

    // ============ ADAPTIVE MODE: Motion-Guided Frame Selection ============
    LOG_INF("Frame range: [%d, %d], base: %d\n", sparams.min_frames, sparams.max_frames, sparams.base_frames);

    // Step 1: Build motion profile from optical flow
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

    // Step 2: Select frames adaptively based on motion complexity
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
    // Remove frames with very low per-frame complexity (near-static frames
    // that slipped through probabilistic sampling)
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

    // Step 4: Spatial patch analysis (for metrics, not used for encoding decisions)
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
    // Drop frames where <25% of spatial patches have motion (mostly static)
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

    // Build frame paths for inference
    std::vector<std::string> frame_paths;
    for (const auto & sf : selected) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/frame_%04d.png", sf.frame_index);
        frame_paths.push_back(sparams.sintel_dir + "/training/clean/" + sparams.video_name + buf);
    }

    // Initialize model and run inference
    LOG_INF("\n--- Model Inference ---\n");

    adaptive_vlm_context ctx(params);

    LOG_INF("Running inference with %zu frames [%s]...\n", frame_paths.size(), mode_str);
    LOG_INF("Prompt: %s\n", prompt.c_str());

    auto inf_result = run_inference(ctx, frame_paths, prompt, n_predict);

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
    LOG_INF("Total time:         %.1f ms\n", total_time_ms);

    // Write JSON output
    std::string json_path = sparams.output_json.empty()
        ? ("results_" + sparams.video_name + "_adaptive.json")
        : sparams.output_json;

    write_results_json(json_path, sparams, n_total_frames, profile, selected, spatial_analyses,
                       inf_result, motion_cov, entropy, flow_time_ms, select_time_ms,
                       total_time_ms);

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
