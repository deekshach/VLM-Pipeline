# Session: CUDA KV Cache Reuse with Temporal RoPE Re-rotation

**Date**: 2026-03-13
**Status**: Working, benchmarked, all three optimizations composing correctly

---

## 1. What Was Built

A GPU-only KV cache reuse mechanism for the streaming VLM benchmark. For static (unchanged) visual patches between video frames, instead of running `llama_decode` (full 28-layer transformer), we directly copy K/V values from the keyframe's KV cache positions to new cell positions via CUDA kernels, applying temporal RoPE correction to K values only.

**This is a prefill optimization.** Generation speed is unaffected because new KV cells are still allocated (the KV cache ends up the same size).

---

## 2. Files Created

### `dmc955/kv_cache_rerotate.cuh`
Header with `kv_reuse_params` struct and `launch_kv_reuse()` declaration.

```cpp
struct kv_reuse_params {
    int n_tokens;        // number of static tokens to copy
    int n_layers;        // number of KV layers
    int n_kv_heads;      // GQA head count
    int n_embd_head_k;   // dimension per head (K)
    int n_embd_head_v;   // dimension per head (V)
    int kv_size;         // total KV cache size (for stride computation)
    int rope_sections[4]; // [temporal, y, x, unused]
    float freq_base;
    int old_temporal_pos;
    int new_temporal_pos;
    bool v_trans;
    const int * src_cells;  // [n_tokens] GPU array of source cell indices
    const int * dst_cells;  // [n_tokens] GPU array of destination cell indices
};

void launch_kv_reuse(
    const kv_reuse_params & params,
    void ** k_data_ptrs,   // [n_layers] GPU data pointers for K tensors
    void ** v_data_ptrs,   // [n_layers] GPU data pointers for V tensors
    cudaStream_t stream);
```

### `dmc955/kv_cache_rerotate.cu`
Three CUDA kernels:

1. **`kv_cache_copy_rerotate_k`** - Grid(n_tokens, n_kv_heads), Block(n_pairs=64). Copies K values with temporal RoPE correction for pairs in temporal sector (pair_idx < rope_sec_temporal). Layout: cell c at offset `c * n_embd_k_gqa + head * n_embd_head_k + pair*2`.

2. **`kv_cache_copy_v_nontrans`** - Grid(n_tokens), Block(n_embd_v_gqa). Contiguous copy per cell at `c * n_embd_v_gqa + elem`.

3. **`kv_cache_copy_v_trans`** - Grid(n_tokens), Block(n_embd_v_gqa). Scattered copy: element j of cell c at `j * kv_size + c`.

4. **`launch_kv_reuse`** - Host wrapper that iterates all layers, launching K and V kernels per layer on a single CUDA stream.

---

## 3. Files Modified

### `src/llama-memory.h`
Added virtual methods to `llama_memory_i` base class:
```cpp
virtual const ggml_tensor * get_v_tensor(int32_t il) const { return nullptr; }
virtual bool get_v_trans() const { return false; }
virtual bool reserve_cells(llama_seq_id seq_id, int32_t n_tokens,
                           const llama_pos * pos, bool is_pos_2d,
                           int32_t * out_cell_indices) { return false; }
```

### `src/llama-kv-cache.h`
Added override declarations in `llama_kv_cache` class:
```cpp
const ggml_tensor * get_v_tensor(int32_t il) const override;
bool get_v_trans() const override;
bool reserve_cells(llama_seq_id seq_id, int32_t n_tokens,
                   const llama_pos * pos, bool is_pos_2d,
                   int32_t * out_cell_indices) override;
```

### `src/llama-kv-cache.cpp`
Implemented all three methods. Key implementation: `reserve_cells()` constructs a synthetic `llama_ubatch` with all required fields (`b_equal_seqs`, `n_seqs_unq`, `seq_id_unq`, `seq_idx`, etc.), calls `find_slot` and `apply_ubatch` to allocate cells and set metadata (pos, seq_id, ext) without running the transformer.

### `include/llama.h`
Added C API declarations:
```cpp
LLAMA_API const struct ggml_tensor * llama_memory_v_tensor(llama_memory_t mem, int32_t il);
LLAMA_API bool llama_memory_v_trans(llama_memory_t mem);
LLAMA_API bool llama_memory_reserve_cells(llama_memory_t mem, llama_seq_id seq_id,
               int32_t n_tokens, const llama_pos * pos, bool is_pos_2d, int32_t * out_cell_indices);
LLAMA_API float llama_model_rope_freq_base_train(const struct llama_model * model);
LLAMA_API void llama_model_rope_sections(const struct llama_model * model, int32_t sections[4]);
LLAMA_API int32_t llama_model_n_embd_head_k(const struct llama_model * model);
```

### `src/llama-context.cpp`
Added forwarding C API implementations for `llama_memory_v_tensor`, `llama_memory_v_trans`, `llama_memory_reserve_cells`.

### `src/llama-model.cpp`
Added RoPE parameter accessors reading from `model->hparams`:
```cpp
float llama_model_rope_freq_base_train(const llama_model * model) {
    return model->hparams.rope_freq_base_train;
}
void llama_model_rope_sections(const llama_model * model, int32_t sections[4]) {
    for (int i = 0; i < 4; i++) sections[i] = model->hparams.rope_sections[i];
}
int32_t llama_model_n_embd_head_k(const llama_model * model) {
    return model->hparams.n_embd_head_k;
}
```

### `dmc955/CMakeLists.txt`
Added CUDA-enabled target:
```cmake
if(GGML_CUDA)
    enable_language(CUDA)
    set(TARGET5 llama-streaming-vlm-cuda)
    add_executable(${TARGET5} streaming_vlm.cpp kv_cache_rerotate.cu)
    set_target_properties(${TARGET5} PROPERTIES
        OUTPUT_NAME llama-streaming-vlm-cuda
        CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TARGET5} PRIVATE common mtmd Threads::Threads)
    target_compile_features(${TARGET5} PRIVATE cxx_std_17)
    target_include_directories(${TARGET5} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
    target_compile_definitions(${TARGET5} PRIVATE HAVE_CUDA_KV_REUSE)
endif()
```

### `dmc955/streaming_vlm.cpp`
Major additions:
- `#include "kv_cache_rerotate.cuh"` and `#include <cuda_runtime.h>` guarded by `#ifdef HAVE_CUDA_KV_REUSE`
- `enable_kv_cache_reuse` flag in `streaming_params` + CLI parsing (`--enable-kv-cache-reuse`)
- `kv_cache_snapshot` struct to capture keyframe's KV cell indices and temporal position
- `perform_kv_cache_reuse()` function (~90 lines): builds M-RoPE positions, calls `llama_memory_reserve_cells`, uploads cell indices to GPU, queries model params, collects K/V data pointers, builds `kv_reuse_params`, calls `launch_kv_reuse`, synchronizes, frees GPU buffers
- Keyframe path captures snapshot after decode
- SKIP ViT path tries CUDA KV reuse before falling back to `llama_decode`

---

## 4. Bugs Found and Fixed During Implementation

### Bug 1: CUDA Illegal Memory Access at Frame 21
**Symptom**: Crash with CUDA illegal memory access after 6 successful KV REUSE frames.

**Root causes**:
1. `kv_size` was extracted as `k_t->ne[2]` but K tensor layout is `[n_embd_k_gqa, kv_size, n_stream]`, so correct is `kv_size = k_t->ne[1]`.
2. V transposed kernel assumed layout `[kv_size, n_embd_head_v, n_kv_heads]` with per-head strides, but actual storage is always `[n_embd_v_gqa, kv_size]`. The transposition only affects how the tensor is VIEWED during attention, not how it's stored. For transposed V, element j of cell c is at `j * kv_size + c`.

**Fix**: Rewrote all three CUDA kernels with correct layouts, changed kv_size extraction.

### Bug 2: No SKIP ViT Triggered on alley_1
**Issue**: `alley_1` has high motion (all frames >50% active), so `active_ratio < skip_threshold` (5%) never triggered.

**Fix**: Switched to `bamboo_1` with `--patch-reuse-motion 5.0` to force more patches static.

---

## 5. Tensor Layout Reference (Qwen2.5-VL 7B)

- **K tensor**: `[n_embd_k_gqa=128*4=512, kv_size, 1]` — cell c, head h, pair p at offset `c * 512 + h * 128 + p * 2`
- **V tensor (non-transposed)**: `[n_embd_v_gqa=128*4=512, kv_size, 1]` — cell c, element j at `c * 512 + j`
- **V tensor (transposed view)**: same storage, but viewed as `[kv_size, 512]` during attention — element j of cell c at `j * kv_size + c`
- **M-RoPE sections**: `[16, 24, 24, 0]` pairs = `[temporal, Y, X, unused]` for Qwen2.5-VL
- **rope_freq_base_train**: 1000000.0

---

## 6. How to Build

```bash
cd /data/spring26/CPP/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target llama-streaming-vlm-cuda -j$(nproc)
# Also builds non-CUDA variant:
cmake --build build --target llama-streaming-vlm -j$(nproc)
```

---

## 7. How to Run

```bash
cd /data/spring26/CPP/llama.cpp

# With KV reuse (requires CUDA build):
./build/bin/llama-streaming-vlm-cuda \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --sintel-dir /data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel \
  --video bamboo_1 --max-frames 10 \
  -ngl 99 \
  --patch-reuse-motion 4.0 \
  --enable-kv-cache-reuse \
  -t 4

# Without KV reuse (baseline comparison):
./build/bin/llama-streaming-vlm \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --sintel-dir /data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel \
  --video bamboo_1 --max-frames 10 \
  -ngl 99 --baseline --baseline-uniform-count 10 \
  -t 4
```

---

## 8. Benchmark Results (bamboo_1, 10 frames, RTX 4090)

### Test A: KV Reuse Only (--patch-reuse-motion 5.0, all 9 non-keyframes static)

| Metric | Without KV Reuse | With KV Reuse | Speedup |
|--------|-----------------|---------------|---------|
| LLM prefill | 1129 ms (6028 tok) | 420 ms (700 tok) | **2.69x** |
| Per-frame static cost | ~59 ms (llama_decode) | ~15 ms (CUDA copy) | **3.9x** |

### Test B: All Three Optimizations Combined (--patch-reuse-motion 4.0)

Frame behavior at motion=4.0:
- Frame 1: KEYFRAME (full encode 160ms)
- Frame 5: **KV REUSE** (0% active, 15ms CUDA copy)
- Frames 9,12,15,21: **CROP+SPLICE** (7% active, crop 10x4, encode ~19-25ms)
- Frame 18: **CROP+SPLICE** (14% active, crop 10x12, encode 26ms)
- Frames 24,27: **KV REUSE** (0% active, 15ms CUDA copy)
- Frame 30: **CROP+SPLICE** (6% active, crop 9x4, encode 19ms)

| Config | LLM Prefill | Tokens through LLM | ViT Encode | Speedup vs Baseline |
|--------|------------|---------------------|------------|---------------------|
| Baseline (uniform, full) | 1562 ms | 6028 | ~1600 ms | 1.00x |
| Adaptive + Spatial (no KV reuse) | 1192 ms | 6028 | 310 ms | **1.31x** |
| Adaptive + Spatial + KV Reuse | 953 ms | 4252 | 310 ms | **1.64x** |

### Speedup Attribution

| Optimization Source | Prefill Reduction | Mechanism |
|--------------------|-------------------|-----------|
| Spatial patch encode | 370 ms (1562->1192) | 3 SKIP ViT frames skip llama_decode; 6 frames use small crops (10x4) instead of full (37x16) |
| KV cache reuse | 239 ms (1192->953) | 3 SKIP ViT frames use 15ms CUDA copy instead of ~60ms llama_decode, removing 1776 tokens from LLM |
| **Combined** | **609 ms (1562->953)** | **1.64x total prefill speedup** |

### Generation (identical across all configs)
- 16.05 ms/tok, 62.3 t/s
- Not affected because KV cache ends up same size (new cells allocated for reused frames)

---

## 9. Key Architecture Details

### KV Cache Reuse Flow
1. **Keyframe decode**: Frame 1 goes through full ViT encode + `llama_decode`. After decode, `kv_cache_snapshot` captures the KV cell indices and temporal position.
2. **Static frame (SKIP ViT + KV REUSE)**: When `active_ratio < 5%`:
   - Build M-RoPE position array for new temporal position
   - Call `llama_memory_reserve_cells()` to allocate new KV cells (metadata only, no compute)
   - Upload src/dst cell index arrays to GPU via `cudaMalloc`/`cudaMemcpy`
   - Collect K/V data pointers for all 28 layers
   - Call `launch_kv_reuse()` — launches kernels for ALL layers on one CUDA stream
   - K kernel: copies with temporal RoPE correction (only first 16 pairs)
   - V kernel: straight copy (no RoPE on V)
   - `cudaStreamSynchronize` + free GPU buffers
3. **Partially changed frame (CROP+SPLICE)**: When `5% <= active_ratio`:
   - Crop the image to bounding box of active patches
   - ViT-encode only the crop region
   - Splice new embeddings into the embedding cache
   - `llama_decode` all 592 tokens (full cache) through LLM

### reserve_cells Implementation
Constructs a synthetic `llama_ubatch` with:
- `b_equal_seqs = 1`, `n_seqs = 1`, `n_seqs_unq = 1`
- `n_pos = 4` (M-RoPE: temporal, y, x, unused)
- Position array with interleaved [t, y, x, 0] for each token
- Dummy `seq_id`, `seq_id_unq`, `seq_idx`, `output` arrays
- `token = nullptr`, `embd = nullptr` (no actual data)

Calls `find_slot(ubatch, false)` then `apply_ubatch(sinfo, ubatch)` to write cell metadata.

### M-RoPE Position Layout for Visual Tokens
For a grid of nx=37, ny=16 at temporal position T:
- Token at (tx, ty): pos = [T, ty, tx, 0] (interleaved as 4 consecutive llama_pos values)
- `n_past` advances by `max(nx, ny) = 37` regardless of how many tokens are actually decoded

---

## 10. Important Constraints and Gotchas

1. **Defrag corrupts M-RoPE**: NEVER call `llama_memory_defragment()` after KV reuse. M-RoPE ext_x/ext_y positions are not preserved during compaction.
2. **KV reuse is prefill-only**: New cells are allocated, so KV cache size is identical during generation. To optimize generation, would need to make static frames SHARE keyframe cells (not implemented).
3. **`enable_pre_llm_pruning` and KV reuse are mutually exclusive** in current code (guard at line 2451).
4. **V transposition**: The `v_trans` flag affects the VIEW during attention but not storage layout. Both kernels exist but non-transposed is typical.
5. **Half precision only**: Kernels use `half` (`__half`). If model uses different precision, kernels need updating.
6. **Cell index validity**: Assumes keyframe cells haven't been evicted. If KV cache is full and cells get evicted, snapshot becomes invalid.

---

## 11. Possible Next Steps

1. **Generation optimization**: Make static frames share keyframe KV cells directly instead of allocating new ones. This would shrink the effective KV cache during generation, speeding up attention. Requires changes to position tracking and seq_id management.
2. **Per-patch KV reuse**: Currently KV reuse only works for fully-static frames (SKIP ViT path). Could extend to partially-changed frames: CUDA-copy static patches, `llama_decode` only changed patches.
3. **Benchmark on more videos**: Test on alley_1 (high motion), sleeping_1 (low motion), etc.
4. **Profile CUDA kernel overhead**: The ~15ms per KV reuse frame includes `cudaMalloc`/`cudaMemcpy`/`cudaFree` overhead. Pre-allocating GPU buffers could reduce this.
5. **Support other precisions**: BF16, Q8_0, etc.

---

## 12. Model and Path Reference

- **Model**: `/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf`
- **mmproj**: `/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf`
- **Sintel dataset**: `/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel`
- **Good test video**: `bamboo_1` (low motion, many static frames with `--patch-reuse-motion >= 4.0`)
- **Build dir**: `/data/spring26/CPP/llama.cpp/build`
- **Project dir**: `/data/spring26/CPP/llama.cpp/dmc955`
