# Implementing temporal patching — scope and validation plan

Goal: make llama.cpp feed the vision tower a **pair of distinct frames** per Conv3d
evaluation, as Qwen2.5-VL intends, instead of the same frame twice. Then verify it
is faithful before using it as a baseline.

Measured prize (from the f4 row, which has the identical compute shape):
**611.6 ms/request at 8 frames, −18.3%**. See `FINDINGS.md` and the artifact.

Written 2026-09-16.

> **Provenance note.** `AGENTS.md` asks that llama.cpp code be authored by the
> human contributor. This is research instrumentation in a private fork. If any of
> it is later proposed upstream it should be rewritten by the contributor and the
> AI assistance disclosed.

---

## 1. What is actually wrong today

The original model's patch embedding is one **Conv3d**, kernel
`[out, in, kt=2, 14, 14]`, consuming two consecutive frames. llama.cpp's converter
splits it along the temporal axis (`convert_hf_to_gguf.py:4569`):

```python
yield (".weight",   w[:, :, 0, ...])   # t=0 slice
yield (".weight.1", w[:, :, 1, ...])   # t=1 slice
```

and `clip_graph_qwen2vl::build()` (`tools/mtmd/models/qwen2vl.cpp:20-28`) applies
**both to the same `inp_raw`** and sums:

```
reference   out = W₀ ⊛ f₀ + W₁ ⊛ f₁
llama.cpp   out = W₀ ⊛ f  + W₁ ⊛ f
```

For a still image this is correct — a lone frame has no partner and the reference
duplicates it too. For video it is not: mtmd hands the tower one frame at a time,
so the t=1 slot always carries a duplicate and every frame burns a whole temporal
slot. Result: 2× the tower passes and 2× the tokens.

---

## 2. The good news — positions need no change

`mtmd.cpp:37` — *"use M-RoPE position counting (**the whole image is 1 temporal
position**)"*. An image chunk already maps to exactly one temporal position, and
`mtmd_image_tokens_get_n_pos` returns `max(nx, ny)` = 37 for this geometry
regardless of content.

So pairing does not touch position logic at all: a chunk still covers one temporal
position, there are simply half as many chunks, and `n_past` halves on its own.
**This was the highest-risk part of the change and it is already correct by
construction.**

---

## 3. Change surface — five places

| # | file | change | size |
|---|---|---|---|
| 1 | `tools/mtmd/models/qwen2vl.cpp` | two `build_inp_raw()` tensors (`inp_raw`, `inp_raw_1`); conv0 reads the first, conv1 the second | ~8 lines |
| 2 | `tools/mtmd/clip.cpp:3596` | upload entry[0] → `inp_raw`, entry[1] → `inp_raw_1` | ~12 lines |
| 3 | `tools/mtmd/clip.cpp` preprocess | emit a **2-entry** `clip_image_f32_batch` for a frame pair | ~20 lines |
| 4 | `tools/mtmd/mtmd.cpp:644` | pair consecutive image bitmaps into one chunk; duplicate the last frame if the count is odd (what the reference does) | ~40 lines |
| 5 | `dmc955/baseline_vlm.cpp` | pass frames as a video sequence rather than N independent images | ~15 lines |

**The one real design decision is #4**: mtmd has no video concept at all —
`mtmd_bitmap` is image-or-audio and the chunk types are only TEXT / IMAGE / AUDIO.
Two options:

- **(a) Caller-side pairing.** `baseline_vlm.cpp` stacks two frames into one
  `mtmd_bitmap` of height `2*ny` (or a 6-channel bitmap) before handing it over.
  No mtmd API change; the pairing convention lives in our harness. Least invasive,
  keeps the diff inside our fork, and is enough to measure and validate.
- **(b) mtmd-level video input.** Add a bitmap flag or a `mtmd_bitmap_init_from_video`
  so mtmd pairs internally. Correct place for it architecturally, larger blast
  radius, and the right shape if this were ever upstreamed.

**Recommendation: (a) first.** It answers the research question with the smallest
change and the fewest confounds. (b) only if the result justifies upstreaming.

---

## 4. Validation ladder

"Same result as published accuracy" is the top rung, and it is much larger than
the implementation. The lower rungs are cheap and catch nearly everything.

### Gate 0 — construction *(free, already established)*
The two GGUF weights **are** the Conv3d temporal slices, verified in the converter
source. Feeding `f₀→W₀`, `f₁→W₁` and summing *is* the Conv3d by definition. There
is no approximation to test — only whether the plumbing routes the right frame to
the right slot.

### Gate 1 — degenerate regression *(hours, no external reference)*
Run the new path with `f₁ = f₀` (the same frame in both slots). It must reproduce
today's output **bit-for-bit**. This proves the plumbing without needing any
reference model, and it is the single highest-value test.

Requires `--temp 0 --seed 42`; note BF16 is numerically non-deterministic across
*reduction-order* changes, so compare logits on the first token rather than the
full generated string.

### Gate 2 — reference equivalence *(1-2 days setup)*
Compare vision-tower output embeddings against HF `transformers` on the same frame
pair. This is the real correctness proof.

Cost: neither `torch` nor `transformers` is installed here, and the environment is
PEP 668-managed, so it needs a venv; the HF-format weights are not on disk either
(~16 GB download; 11 TB free, so space is fine). Compare
`[3584 × 592]` embeddings, expect agreement to BF16 rounding.

### Gate 3 — published accuracy *(separate project)*
Qwen publishes Video-MME / MVBench / etc. Reproducing those numbers needs the
benchmark datasets and a scoring harness — **Sintel has no ground truth**, so none
of the existing infrastructure helps. This validates the whole stack, not this
change, and should be scoped on its own.

**Recommended path: Gate 0 → 1 → 2.** Gate 1 alone catches every plumbing error;
Gate 2 settles faithfulness. Gate 3 is worth doing but is a different project, and
it would be answering "is llama.cpp's Qwen2.5-VL correct overall", not "is this
change correct".

---

## 5. Sequencing

1. **Land the two-input graph change** (#1, #2) with the second input wired to the
   *same* frame. No behaviour change. Verify Gate 1 passes — that is the refactor
   gate.
2. **Add caller-side pairing** (#3, #5 option a). Behaviour now changes; token
   count halves; re-measure ViT / prefill / decode with `measure.sh`.
3. **Confirm the predicted cost.** Expect the f4 row: ViT ≈ 208 ms, prefill ≈ 386,
   request ≈ 2733 at 8 frames. If it lands materially off, the implementation is
   doing something other than intended.
4. **Gate 2** if the numbers hold.
5. **Then** rebase the optimization budget — and note it shrinks, because temporal
   pairing absorbs most of what the chunking and pipelining fixes were recovering
   (406.6 ms → 224.6 ms of remaining opportunity; co-scheduling ceiling
   1.37× → 1.26×).

## 6. Open questions for the contributor

- **Odd frame counts**: duplicate the last frame (reference behaviour) or drop it?
- **`--fps` sampling**: with pairing, does `--base-frames 8` mean 8 frames (4 groups)
  or 8 groups (16 frames)? The sweep's meaning changes either way, and every prior
  measurement is labelled by frame count.
- **`second_per_grid_ts`**: HF scales temporal positions by real elapsed time
  between groups. llama.cpp advances one temporal position per chunk regardless.
  Out of scope for the cost question, but it is a fidelity gap worth recording —
  and it may matter for Gate 2 agreement.
