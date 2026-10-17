#include "models.h"

ggml_cgraph * clip_graph_qwen2vl::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;

    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS // qwen 2.5 vl
        : NORM_TYPE_NORMAL; // qwen 2 vl

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // Second temporal slice of the patch embedding.
    //
    // The original model's patch_embed is one Conv3d with temporal kernel 2;
    // convert_hf_to_gguf.py splits it into patch_embeddings_0 (t=0) and
    // patch_embeddings_1 (t=1). Feeding a DISTINCT frame to each slot reproduces
    // that Conv3d exactly:   out = W0 (*) f0 + W1 (*) f1
    // clip fills inp_raw_1 with the paired frame when one is supplied, and with a
    // copy of inp_raw otherwise -- which is what the reference does for a still
    // image, and keeps single-image behaviour bit-identical.
    ggml_tensor * inp_raw_1 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, 3);
    ggml_set_name(inp_raw_1, "inp_raw_1");
    ggml_set_input(inp_raw_1);

    // second conv dimension
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw_1, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // Batched windows (Jetson study switch, see clip_vit_window_batch_enabled): when the image is a
    // whole number of attention windows, the windowed layers attend within each window as a batch
    // of independent sequences, so neither the dense mask nor the masked-out scores exist.
    const int  win_merged   = (hparams.attn_window_size > 0 ? hparams.attn_window_size : 112) / patch_size / 2;
    const bool window_batch = use_window_attn && proj_type == PROJECTOR_TYPE_QWEN25VL && clip_vit_window_batch_enabled() &&
                              (n_patches_x / 2) % win_merged == 0 && (n_patches_y / 2) % win_merged == 0;
    const int  win_patches  = win_merged * win_merged * 4;   // patches per window (64 for 112 px)

    // Per-window encode (Jetson study, see clip_pw_state): only the listed windows' patches go through the layers.
    const bool pw_on     = pw != nullptr && !pw->windows.empty();
    GGML_ASSERT(!pw_on || window_batch);
    const int  n_windows = pw_on ? (int) pw->windows.size() : n_patches / win_patches;
    const int  n_tok     = pw_on ? n_windows * win_patches : n_patches;   // patches through the layers
    const int  num_position_ids = n_tok * 4; // m-rope requires 4 dim per position
    ggml_tensor * pw_kv_rows = nullptr;   // the listed windows' patch rows in the K/V cache (window order)
    if (pw_on) {
        pw_kv_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok);
        ggml_set_name(pw_kv_rows, "pw_kv_rows");
        ggml_set_input(pw_kv_rows);
    }

    ggml_tensor * inpL           = inp;
    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    if (use_window_attn) {
        // handle window attention inputs
        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);
        // mask for window attention
        if (!window_batch) {
            window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_name(window_mask, "window_mask");
            ggml_set_input(window_mask);

            // if flash attn is used, we need to pad the mask and cast to f16
            if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                window_mask = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);
            }
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        if (pw_on) {   // keep the listed windows (merged-token rows, window order)
            ggml_tensor * pw_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok / 4);
            ggml_set_name(pw_rows, "pw_rows");
            ggml_set_input(pw_rows);
            inpL = ggml_get_rows(ctx0, inpL, pw_rows);
        }
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_tok, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_tok);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_tok);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_tok);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            if (!full_attn && window_batch) {
                // [d_head, n_head, n_patches] -> [d_head, n_head, win_patches, n_windows]
                Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, win_patches, n_windows);
                Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head, win_patches, n_windows);
                Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head, win_patches, n_windows);
                attn_mask = nullptr;
            } else if (full_attn && pw_on) {
                // refresh the listed windows' rows of the frame's K/V cache, then attend over the whole frame
                ggml_tensor * Kall = ggml_set_rows(ctx0, pw->k[il], ggml_reshape_2d(ctx0, Kcur, n_embd, n_tok), pw_kv_rows);
                ggml_tensor * Vall = ggml_set_rows(ctx0, pw->v[il], ggml_reshape_2d(ctx0, Vcur, n_embd, n_tok), pw_kv_rows);
                Kcur = ggml_reshape_3d(ctx0, Kall, d_head, n_head, n_patches);
                Vcur = ggml_reshape_3d(ctx0, Vall, d_head, n_head, n_patches);
            }

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_tok / 4, batch_size);
    embeddings = build_ffn(embeddings,
                        model.mm_0_w, model.mm_0_b,
                        nullptr, nullptr,
                        model.mm_1_w, model.mm_1_b,
                        FFN_GELU,
                        -1);

    if (use_window_attn && !pw_on) {   // per-window output stays in window order
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);

        // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
