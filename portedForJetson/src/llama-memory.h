#pragma once

#include "llama.h"

#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;

    // Remove a single KV entry matching exact position and 2D spatial coordinates (M-RoPE).
    // Returns true if found and removed. Default implementation returns false (not supported).
    virtual bool seq_rm_pos_ext(llama_seq_id seq_id, llama_pos pos, llama_pos ext_x, llama_pos ext_y) {
        GGML_UNUSED(seq_id); GGML_UNUSED(pos); GGML_UNUSED(ext_x); GGML_UNUSED(ext_y);
        return false;
    }

    // compact the cache by moving occupied cells to fill gaps, reducing effective n_kv
    // if lctx is provided, uses GPU-native defrag via ggml compute graph; otherwise CPU bulk transfer
    virtual void defragment(llama_context * lctx = nullptr) { GGML_UNUSED(lctx); }

    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    // Get the K cache tensor for a given model layer (for manual attention computation)
    // Returns nullptr if not supported by this memory type
    virtual const ggml_tensor * get_k_tensor(int32_t il) const { GGML_UNUSED(il); return nullptr; }

    // Get the V cache tensor for a given model layer
    // Returns nullptr if not supported by this memory type
    virtual const ggml_tensor * get_v_tensor(int32_t il) const { GGML_UNUSED(il); return nullptr; }

    // Returns true if the V cache is stored in transposed layout
    virtual bool get_v_trans() const { return false; }

    // Reserve KV cache cells with metadata (pos, seq_id, ext) WITHOUT running the transformer.
    // This allocates cells and sets their positions/sequences but does not compute K/V values.
    // The caller is responsible for filling in K/V data (e.g. via CUDA copy kernels).
    // out_cell_indices: if non-null, filled with the allocated cell indices [n_tokens]
    // Returns true on success, false if no slot available.
    virtual bool reserve_cells(llama_seq_id seq_id, int32_t n_tokens,
                               const llama_pos * pos, bool is_pos_2d,
                               int32_t * out_cell_indices) {
        GGML_UNUSED(seq_id); GGML_UNUSED(n_tokens); GGML_UNUSED(pos);
        GGML_UNUSED(is_pos_2d); GGML_UNUSED(out_cell_indices);
        return false;
    }

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
