// ofa_live_flow.h -- dense optical flow between consecutive video frames, computed live on the
// Jetson Orin's optical-flow accelerator (OFA) through NVIDIA VPI.
//
// Per frame: PNG decode (CPU) -> RGB8 to Y8 and Gaussian pyramid (CPU or CUDA) -> block-linear pyramid (VIC).
// With the CPU for that step, the flow path never touches the GPU, so it does not slow (or get slowed
// by) ViT work running at the same time.
// Per pair (k, k+1): dense flow on the OFA -> motion vectors back to pitch-linear (VIC) -> S10.5 to
// float on the CPU. Output is flow from frame k to k+1 (as Sintel's frame_000k.flo), one vector per
// grid x grid pixels, in full-resolution pixel units.
//
// Two schedules:
//   pipelined  decode threads, a submit thread and a consumer thread run concurrently. Frame
//              preparation (CUDA/VIC) and the OFA run on separate VPI streams joined by events,
//              so preparing frame k+1 overlaps the flow of pair (k-1, k).
//   serial     one step at a time, each stage timed on its own (for the breakdown).
//
// VPI buffers live in rings of RING slots, reused once the events say a slot is free: keeping one
// pyramid and motion-vector image per frame exhausts the driver's buffer pool after ~40 frames
// (NVMEDIA_STATUS_ERROR from the OFA). Decoded pixels are copied into VPI-owned images rather than
// wrapped: creating and destroying a wrapper around host memory waits for the GPU to go idle,
// which stalls the flow behind any ViT work running at the same time. Everything is allocated
// in init(), so a request only submits work.
#pragma once

#include "mtmd.h"
#include "mtmd-helper.h"

#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct rgb_frame {
    int w = 0, h = 0;
    std::vector<unsigned char> px; // RGB8, tightly packed
};

struct ofa_flow_timing {
    // pipelined and serial
    double wall_ms        = 0; // run() start to the last on_flow() return
    double decode_wall_ms = 0; // run() start to the last frame decoded
    double decode_sum_ms  = 0; // CPU time summed over frames
    double consumer_ms    = 0; // on_flow() callbacks, summed
    double readback_ms    = 0; // lock + S10.5 -> float, summed
    // serial only (each stage synchronised on its own)
    double prep_ms = 0;        // RGB -> Y8 -> pyramid (CPU or CUDA) -> block-linear (VIC), summed over frames
    double ofa_ms  = 0;        // OFA + motion-vector layout conversion, summed over pairs
    int n_frames = 0, n_pairs = 0;
};

#define OFA_OK(expr) ok_((expr), #expr)

class ofa_flow_engine {
public:
    using flow_cb = std::function<void(int pair, int w, int h, std::vector<float> && u, std::vector<float> && v)>;
    static constexpr int RING = 4;

    ~ofa_flow_engine() { release(); }

    bool init(int width, int height, int max_frames, int grid, int levels, VPIOpticalFlowQuality quality,
              bool prep_on_cpu = true) {
        w_ = width; h_ = height; max_frames_ = max_frames;
        prep_backend_ = prep_on_cpu ? VPI_BACKEND_CPU : VPI_BACKEND_CUDA;
        mvw_ = (w_ + grid - 1) / grid;
        mvh_ = (h_ + grid - 1) / grid;
        std::vector<int32_t> grids(levels, grid);
        if (!OFA_OK(vpiStreamCreate(prep_backend_ | VPI_BACKEND_VIC, &s_prep_)) ||
            !OFA_OK(vpiStreamCreate(VPI_BACKEND_OFA | VPI_BACKEND_VIC, &s_ofa_)) ||
            !OFA_OK(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, w_, h_, VPI_IMAGE_FORMAT_Y8_ER_BL, grids.data(), levels,
                                              quality, &payload_)) ||
            !OFA_OK(vpiImageCreate(w_, h_, VPI_IMAGE_FORMAT_Y8_ER, 0, &y_tmp_)) ||
            !OFA_OK(vpiPyramidCreate(w_, h_, VPI_IMAGE_FORMAT_Y8_ER, levels, 0.5, 0, &pyr_tmp_))) {
            return false;
        }
        for (int r = 0; r < RING; r++) {
            if (!OFA_OK(vpiPyramidCreate(w_, h_, VPI_IMAGE_FORMAT_Y8_ER_BL, levels, 0.5, 0, &pyr_[r])) ||
                !OFA_OK(vpiImageCreate(mvw_, mvh_, VPI_IMAGE_FORMAT_2S16_BL, 0, &mv_bl_[r])) ||
                !OFA_OK(vpiImageCreate(mvw_, mvh_, VPI_IMAGE_FORMAT_2S16, 0, &mv_pl_[r])) ||
                !OFA_OK(vpiImageCreate(w_, h_, VPI_IMAGE_FORMAT_RGB8, 0, &rgb_[r]))) {
                return false;
            }
        }
        ev_prep_.assign(max_frames, nullptr);
        ev_flow_.assign(max_frames, nullptr);
        for (int i = 0; i < max_frames; i++) {
            if (!OFA_OK(vpiEventCreate(0, &ev_prep_[i])) || !OFA_OK(vpiEventCreate(0, &ev_flow_[i]))) return false;
        }
        // Warm up every engine once (CUDA context, VIC, OFA firmware) outside any request.
        std::vector<rgb_frame> f(2);
        for (auto & x : f) { x.w = w_; x.h = h_; x.px.assign((size_t)w_ * h_ * 3, 0); }
        return submit_prep(f[0], 0) && submit_prep(f[1], 1) && submit_flow(0) && OFA_OK(vpiEventSync(ev_flow_[0]));
    }

    // Decode frames[0..n), compute the flow of every consecutive pair and hand each to on_flow(),
    // in pair order, from a single thread. Decoded frames are returned for reuse.
    bool run(const std::vector<std::string> & paths, bool pipelined, int decode_threads,
             const flow_cb & on_flow, std::vector<rgb_frame> & frames, ofa_flow_timing & t) {
        const int n = (int)paths.size();
        t = ofa_flow_timing();
        t.n_frames = n; t.n_pairs = n > 0 ? n - 1 : 0;
        if (n > max_frames_) { fprintf(stderr, "ofa: %d frames > max %d\n", n, max_frames_); return false; }
        frames.assign(n, rgb_frame());
        consumed_ = 0;
        const auto t0 = now();
        const bool good = pipelined ? run_pipelined(paths, decode_threads, on_flow, frames, t, t0)
                                    : run_serial(paths, on_flow, frames, t, t0);
        vpiStreamSync(s_prep_); vpiStreamSync(s_ofa_);
        t.wall_ms = ms(t0, now());
        return good;
    }

    int mv_width() const { return mvw_; }
    int mv_height() const { return mvh_; }

private:
    using clk = std::chrono::steady_clock;
    static clk::time_point now() { return clk::now(); }
    static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

    static bool ok_(VPIStatus s, const char * what) {
        if (s == VPI_SUCCESS) return true;
        char msg[VPI_MAX_STATUS_MESSAGE_LENGTH];
        vpiGetLastStatusMessage(msg, sizeof(msg));
        fprintf(stderr, "VPI error %s: %s\n  in %s\n", vpiStatusGetName(s), msg, what);
        return false;
    }

    bool decode(const std::string & path, rgb_frame & f) const {
        // nullptr context: for image files the helper only decodes (the context is used for audio).
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(nullptr, path.c_str()));
        if (!bmp.ptr) return false;
        f.w = (int)bmp.nx(); f.h = (int)bmp.ny();
        f.px.assign(bmp.data(), bmp.data() + bmp.n_bytes());
        return f.w == w_ && f.h == h_;
    }

    // Frame i: copy its pixels into ring slot i % RING and queue RGB -> Y8 -> pyramid -> block-linear.
    // The slot's previous user, frame i - RING, must be done with both of its pairs.
    bool submit_prep(const rgb_frame & f, int i) {
        const int r = i % RING;
        // frame i - RING is used by pairs i - RING - 1 and i - RING (in that order on the OFA stream)
        if (i >= RING && !OFA_OK(vpiEventSync(ev_flow_[i - RING]))) return false;
        VPIImageData d;
        if (!OFA_OK(vpiImageLockData(rgb_[r], VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d))) return false;
        const auto & pl = d.buffer.pitch.planes[0];
        for (int y = 0; y < h_; y++) {
            memcpy((uint8_t *)pl.data + (size_t)y * pl.pitchBytes, f.px.data() + (size_t)y * w_ * 3, (size_t)w_ * 3);
        }
        return OFA_OK(vpiImageUnlock(rgb_[r])) &&
               OFA_OK(vpiSubmitConvertImageFormat(s_prep_, prep_backend_, rgb_[r], y_tmp_, nullptr)) &&
               OFA_OK(vpiSubmitGaussianPyramidGenerator(s_prep_, prep_backend_, y_tmp_, pyr_tmp_, VPI_BORDER_CLAMP)) &&
               OFA_OK(vpiSubmitConvertImageFormatPyramid(s_prep_, VPI_BACKEND_VIC, pyr_tmp_, pyr_[r], nullptr)) &&
               OFA_OK(vpiEventRecord(ev_prep_[i], s_prep_));
    }

    // Pair p = (p, p+1) into motion-vector slot p % RING; the caller makes sure pair p - RING
    // has been read back before this is called.
    bool submit_flow(int p) {
        const int r = p % RING;
        return OFA_OK(vpiStreamWaitEvent(s_ofa_, ev_prep_[p + 1])) &&
               (p > 0 || OFA_OK(vpiStreamWaitEvent(s_ofa_, ev_prep_[p]))) &&
               OFA_OK(vpiSubmitOpticalFlowDensePyramid(s_ofa_, VPI_BACKEND_OFA, payload_, pyr_[p % RING],
                                                       pyr_[(p + 1) % RING], mv_bl_[r])) &&
               OFA_OK(vpiSubmitConvertImageFormat(s_ofa_, VPI_BACKEND_VIC, mv_bl_[r], mv_pl_[r], nullptr)) &&
               OFA_OK(vpiEventRecord(ev_flow_[p], s_ofa_));
    }

    bool read_flow(int p, std::vector<float> & u, std::vector<float> & v) {
        VPIImage mv = mv_pl_[p % RING];
        VPIImageData d;
        if (!OFA_OK(vpiImageLockData(mv, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d))) return false;
        const auto & pl = d.buffer.pitch.planes[0];
        u.resize((size_t)mvw_ * mvh_); v.resize((size_t)mvw_ * mvh_);
        for (int y = 0; y < mvh_; y++) {
            const int16_t * row = (const int16_t *)((const uint8_t *)pl.data + (size_t)y * pl.pitchBytes);
            for (int x = 0; x < mvw_; x++) {
                u[(size_t)y * mvw_ + x] = row[2 * x] * (1.0f / 32.0f);   // S10.5 fixed point
                v[(size_t)y * mvw_ + x] = row[2 * x + 1] * (1.0f / 32.0f);
            }
        }
        return OFA_OK(vpiImageUnlock(mv));
    }

    bool consume(int p, const flow_cb & on_flow, ofa_flow_timing & t) {
        std::vector<float> u, v;
        auto a = now();
        if (!read_flow(p, u, v)) return false;
        auto b = now();
        {
            std::lock_guard<std::mutex> lk(m_);
            consumed_ = p + 1;   // motion-vector slot p % RING is free again
        }
        cv_.notify_all();
        on_flow(p, mvw_, mvh_, std::move(u), std::move(v));
        t.readback_ms += ms(a, b);
        t.consumer_ms += ms(b, now());
        return true;
    }

    bool run_serial(const std::vector<std::string> & paths, const flow_cb & on_flow, std::vector<rgb_frame> & frames,
                    ofa_flow_timing & t, clk::time_point t0) {
        for (int i = 0; i < (int)paths.size(); i++) {
            auto a = now();
            if (!decode(paths[i], frames[i])) { fprintf(stderr, "ofa: cannot decode %s\n", paths[i].c_str()); return false; }
            auto b = now();
            t.decode_sum_ms += ms(a, b);
            t.decode_wall_ms = ms(t0, b);
            if (!submit_prep(frames[i], i) || !OFA_OK(vpiEventSync(ev_prep_[i]))) return false;
            auto c = now();
            t.prep_ms += ms(b, c);
            if (i == 0) continue;
            if (!submit_flow(i - 1) || !OFA_OK(vpiEventSync(ev_flow_[i - 1]))) return false;
            t.ofa_ms += ms(c, now());
            if (!consume(i - 1, on_flow, t)) return false;
        }
        return true;
    }

    bool run_pipelined(const std::vector<std::string> & paths, int decode_threads, const flow_cb & on_flow,
                       std::vector<rgb_frame> & frames, ofa_flow_timing & t, clk::time_point t0) {
        const int n = (int)paths.size();
        std::vector<char> decoded(n, 0);          // 1 = ok, 2 = failed
        std::vector<double> dec_ms(n, 0.0);
        std::atomic<int> next{0};
        std::atomic<bool> failed{false};
        int submitted = 0;                         // pairs submitted so far (guarded by m_)
        clk::time_point last_decoded = t0;

        // Stage 1: decode on several CPU threads, in roughly frame order.
        std::vector<std::thread> workers;
        for (int k = 0; k < std::max(1, decode_threads); k++) {
            workers.emplace_back([&]() {
                for (int i; (i = next.fetch_add(1)) < n;) {
                    auto a = now();
                    const bool good = decode(paths[i], frames[i]);
                    auto b = now();
                    std::lock_guard<std::mutex> lk(m_);
                    decoded[i] = good ? 1 : 2;
                    dec_ms[i] = ms(a, b);
                    if (b > last_decoded) last_decoded = b;
                    cv_.notify_all();
                }
            });
        }

        // Stage 4 (consumer): wait for each pair's flow in order, read it back, analyse it.
        std::thread consumer([&]() {
            for (int p = 0; p < n - 1; p++) {
                {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait(lk, [&] { return submitted > p || failed.load(); });
                }
                if (failed.load()) return;
                if (!OFA_OK(vpiEventSync(ev_flow_[p])) || !consume(p, on_flow, t)) {
                    failed = true;
                    cv_.notify_all();
                    return;
                }
            }
        });

        // Stages 2-3 (this thread): as frames arrive in order, queue their preparation (CUDA/VIC
        // stream) and the flow of the pair they complete (OFA stream, waits on the prep event).
        for (int i = 0; i < n && !failed.load(); i++) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return decoded[i] != 0; });
                if (decoded[i] == 2) { fprintf(stderr, "ofa: cannot decode %s\n", paths[i].c_str()); failed = true; break; }
            }
            if (!submit_prep(frames[i], i)) { failed = true; break; }
            if (i == 0) continue;
            const int p = i - 1;
            {
                std::unique_lock<std::mutex> lk(m_);   // motion-vector slot p % RING must have been read back
                cv_.wait(lk, [&] { return consumed_ > p - RING || failed.load(); });
            }
            if (failed.load() || !submit_flow(p)) { failed = true; break; }
            {
                std::lock_guard<std::mutex> lk(m_);
                submitted = p + 1;
            }
            cv_.notify_all();
        }
        cv_.notify_all();
        for (auto & w : workers) w.join();
        consumer.join();
        for (double d : dec_ms) t.decode_sum_ms += d;
        t.decode_wall_ms = ms(t0, last_decoded);
        return !failed.load();
    }

    void release() {
        if (s_prep_) vpiStreamSync(s_prep_);
        if (s_ofa_) vpiStreamSync(s_ofa_);
        for (int r = 0; r < RING; r++) {
            if (rgb_[r]) vpiImageDestroy(rgb_[r]);
            if (pyr_[r]) vpiPyramidDestroy(pyr_[r]);
            if (mv_bl_[r]) vpiImageDestroy(mv_bl_[r]);
            if (mv_pl_[r]) vpiImageDestroy(mv_pl_[r]);
            pyr_[r] = nullptr; rgb_[r] = mv_bl_[r] = mv_pl_[r] = nullptr;
        }
        for (auto e : ev_prep_) if (e) vpiEventDestroy(e);
        for (auto e : ev_flow_) if (e) vpiEventDestroy(e);
        ev_prep_.clear(); ev_flow_.clear();
        if (pyr_tmp_) vpiPyramidDestroy(pyr_tmp_);
        if (y_tmp_) vpiImageDestroy(y_tmp_);
        if (payload_) vpiPayloadDestroy(payload_);
        if (s_prep_) vpiStreamDestroy(s_prep_);
        if (s_ofa_) vpiStreamDestroy(s_ofa_);
        s_prep_ = s_ofa_ = nullptr; payload_ = nullptr; y_tmp_ = nullptr; pyr_tmp_ = nullptr;
    }

    int w_ = 0, h_ = 0, max_frames_ = 0, mvw_ = 0, mvh_ = 0;
    uint64_t prep_backend_ = VPI_BACKEND_CPU;
    VPIStream s_prep_ = nullptr, s_ofa_ = nullptr;
    VPIPayload payload_ = nullptr;
    VPIImage y_tmp_ = nullptr;
    VPIPyramid pyr_tmp_ = nullptr;
    VPIPyramid pyr_[RING] = {};
    VPIImage rgb_[RING] = {}, mv_bl_[RING] = {}, mv_pl_[RING] = {};
    std::vector<VPIEvent> ev_prep_, ev_flow_;
    std::mutex m_;
    std::condition_variable cv_;
    int consumed_ = 0;   // pairs read back so far (guarded by m_)
};

#undef OFA_OK
