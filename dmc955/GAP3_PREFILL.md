# Gap 3 in detail — why prefill leaves compute on the table

Qwen2.5-VL-7B-BF16, RTX 4090 (sm_89), 13 frames of Sintel `alley_1`, 592 visual
tokens/frame. Sources: `profiling_results/handshake_7B.sqlite` (nsys),
`ncu_2026-09-02_7B-BF16/` (ncu), and a new validated A/B (below).

Written 2026-09-15. Supersedes the one-paragraph Gap 3 in `RESEARCH_GAPS.md`.

---

## The one-line version

Gap 3 is **two** problems, not one. Two thirds of prefill's LLM evaluations run
below the roofline ridge point purely because of how the batch is chunked (**383
ms/request of excess, 9.3%**), and the remaining third — the part that *is*
batched properly — reaches only 48% of tensor peak because of an L2 bandwidth
limit plus an occupancy ceiling that ncu pins to a specific hardware resource.

The first is a configuration and chunking problem, partially fixed by one flag
(**measured: 90 ms/request**). The second is a kernel-selection problem.

---

## The governing quantity: arithmetic intensity = tokens per ubatch

For a prefill GEMM, each weight byte is read once and reused once per token in the
ubatch. With BF16 (2 bytes/element, 2 FLOP/element):

```
arithmetic intensity  =  n_tokens_in_ubatch   FLOP/byte
ridge point (4090)    =  165.2 TFLOPS / 1008 GB/s  =  164 FLOP/byte
```

**Any prefill ubatch below ~164 tokens is memory-bound by construction**, no
matter how good the kernel is. That single number organises everything below.

---

## Part 3a — the engine performs 39 LLM evaluations to prefill 7,749 tokens

Recovered from the trace by grouping `rms_norm_f32` launches (its grid equals the
ubatch's token count) into contiguous runs, then measuring stream-15 GPU-busy time
between the first and last kernel of each run. Request 1:

| n_tok | evals | GPU ms | ms/eval | **µs/token** | AI | vs ridge 164 |
|---|---|---|---|---|---|---|
| **512** | 13 | 753.4 | 58.0 | **113.2** | 512 | compute-bound |
| **80** | 13 | 274.4 | 21.1 | **263.8** | 80 | **memory-bound** |
| 29 | 1 | 18.4 | 18.4 | 633.5 | 29 | memory-bound |
| **2** | 12 | 214.3 | 17.9 | **8,930** | 2 | **memory-bound** |
| total | 39 | 1260.5 | | 162.7 | | |

Taking the 512-token pass (113.2 µs/token) as the achievable marginal rate:

| class | actual | at the 512-token rate | **excess** |
|---|---|---|---|
| 13 x 80 tok | 274.4 ms | 117.7 ms | **156.7 ms** |
| 12 x 2 tok | 214.3 ms | 2.7 ms | **211.6 ms** |
| 1 x 29 tok | 18.4 ms | 3.3 ms | 15.1 ms |
| | | | **383 ms/request = 9.3%** |

A 2-token evaluation costs 17.9 ms — **79x the marginal per-token cost** — because
it runs all 28 layers and streams the entire 15.24 GB weight set to do almost no
arithmetic. It costs the same as decoding a token, which is exactly what it is.

### Corroboration: weight re-streaming dominates encode DRAM traffic

```
39 evaluations x 15.24 GB weight set          = 594 GB
measured DRAM traffic during encode
  (43.8% x 1008 GB/s x 1.948 s, nsys hw)      = 860 GB
  -> weight re-streaming is 69% of it
```

At `n_ubatch = 2048` the same tokens would need ~4 evaluations = 61 GB. **The
engine moves ~10x more weight bytes through prefill than the work requires.**

### Cause 1: `n_ubatch` = 512 against 592 tokens per frame — VALIDATED FIX

`mtmd_helper` submits all 592 tokens in **one** `llama_decode`
(`mtmd-helper.cpp:278`, and the runlog confirms `n_tokens_batch = 592`). The split
happens *inside* llama.cpp, which chops the batch at `n_ubatch = 512` — giving a
512-token pass plus an 80-token remainder, **per frame**. Upstream already knows:
`mtmd-helper.cpp:275` reads *"TODO @ngxson : ... n_ubatch must be enough to hold
the image"*.

So it is a pure context parameter. Measured, 3 rounds, steady-state rounds 1-2:

| `-ub` | encode ms | gen ms | request ms |
|---|---|---|---|
| 512 (default) | 1945, 1944 | 2233, 2227 | 4313, 4307 |
| **640** | **1856, 1853** | 2233, 2233 | **4225, 4223** |

**90 ms/request off encode (4.6% of encode, 2.1% of request), decode unchanged**
as it must be. One flag.

That is below my 157 ms estimate: the marginal-rate model is an upper bound,
because part of the 80-token pass was already overlapped with ViT work on the
other stream. Direction and order of magnitude confirmed; the ceiling was
optimistic.

### Cause 2: twelve 2-token separator chunks — needs real work

`mtmd_tokenize` emits one chunk per frame plus the text between them. Between
consecutive frames that text is the `<|vision_end|><|vision_start|>` pair — **2
tokens** (confirmed: `rms_norm` grid = 2, 12 occurrences per request for 13
frames). Each pair triggers a full 28-layer evaluation: 193 `mul_mat_f` launches,
28 layers x 7 matmuls (3 at grid 112, 2 at grid 16, 2 at grid 592).

Cost: **214 ms/request to process 24 tokens.**

This one cannot be fixed by a flag. Merging those tokens into the adjacent image
ubatch requires a ubatch holding *both* token ids and embeddings, and
`llama_batch` carries `token` **or** `embd`, not both. The workable route is to
look up the separator tokens' embedding rows and concatenate them with the image
embeddings into a single `embd` batch — a genuine change to `mtmd_helper`, not a
parameter.

### Where the small-ubatch work lands in the kernel dispatch

Worth knowing, because it explains why two matmul kernels with opposite profiles
coexist in one phase. `ggml_cuda_should_use_mmf` (`mmf.cu:176`) returns false when
`src1_ncols > 16`, so ggml's own `mul_mat_f` **never** sees a 592- or 512-token
chunk — those fall through to cuBLAS. `mul_mat_f` is running the 2-token
separators, and its grid is `N_out / MMF_ROWS_PER_BLOCK` with
`MMF_ROWS_PER_BLOCK = 32` (`mmf.cuh:9`), which decodes the layer exactly:

| grid | x32 | weight |
|---|---|---|
| 592 | 18944 | FFN gate / up |
| 112 | 3584 | Q proj / O proj / down proj |
| 16 | 512 | K proj / V proj (n_head_kv 4 x 128) |

Hence `mul_mat_f @592` at **93.7% DRAM, 6.8% L2 hit, 13.5% SM** — it is a
decode-shaped kernel, correctly chosen for a 2-token batch. The kernel is not the
problem; being asked to run at all is.

---

## Part 3b — at 512 tokens the GEMM still reaches only 48% of tensor peak

This is the original Gap 3, now correctly scoped to the **60% of prefill time**
that is properly batched (13 x 512-token evaluations, AI 512, 3.1x above ridge —
it *should* be solidly compute-bound).

### What ncu measures, per launch (4 launches, spread < 1%)

| | `ampere_64x128` @1184 | `ampere_128x64` @896 | ViT `cutlass` @1184 |
|---|---|---|---|
| duration | 534.7 µs | 501.8 µs | 177.0 µs |
| **tensor pipe** | **48.4%** | 48.2% | 43.4% |
| SM throughput | 44.9% | 47.9% | 41.6% |
| **L2 throughput** | **68.8%** | **73.8%** | **64.6%** |
| DRAM | 32.4% | 37.1% | 16.9% |
| L2 hit rate | 91.5% | 90.3% | 97.0% |
| eligible warps/sched | **0.15** | 0.15 | 0.12 |
| issue slots busy | **10.2%** | 10.1% | 11.6% |

ncu's own `SOLBottleneck` rule on both prefill GEMMs: *"Memory is more heavily
utilized than Compute: ... identify the **L2 bottleneck**."*

### The occupancy ceiling, pinned to a specific resource

ncu's `Block Limit *` metrics give blocks/SM permitted by each resource; the
minimum binds. This is the part worth knowing precisely, because the two kernel
families are limited by **different** hardware:

| kernel | block | regs/thr | blk/SM by reg | by smem | **limiter** | warps/SM | occ | warps/sched |
|---|---|---|---|---|---|---|---|---|
| prefill `ampere_64x128` @1184 | 128 | 230 | **2** | 3 | **REGISTERS** | 8 | 16.7% | **2.00** |
| prefill `ampere_128x64` @896 | 128 | 230 | **2** | 3 | **REGISTERS** | 8 | 16.7% | 2.00 |
| prefill `cutlass_relu` @592 | 128 | 88 | 5 | **2** | **SHARED MEM** | 8 | 16.7% | 2.00 |
| ViT `cutlass` @1184 | 128 | 156 | 3 | **1** | **SHARED MEM** | 4 | **8.3%** | **1.00** |
| `mul_mat_f` @592 | 224 | 92 | **2** | 3 | REGISTERS | 14 | 29.2% | 3.50 |
| elementwise | 256 | 16-20 | 10-21 | 16-32 | WARPS | 48 | 100% | 12.00 |

The register arithmetic checks out exactly: 230 regs x 128 threads = 29,440
registers/block; the SM's 65,536 / 29,440 = 2.22 -> **2 blocks**, 8 warps, 16.7%
of the 48-warp maximum. The ViT's cutlass kernel uses 72 KB of shared memory per
block against ~100 KB/SM -> **1 block**, 4 warps, 8.3% — the worst in the
pipeline, and shared-memory-bound rather than register-bound.

Achieved occupancy equals theoretical in every case, so **this is a ceiling, not a
failure to reach one.** No amount of launch-configuration tuning helps; the tile
shape itself sets it.

### Is 16.7% occupancy actually a defect? — the honest answer

**Not on its own.** Big-tile tensor GEMMs are *designed* for low occupancy: they
hide MMA latency with register blocking and deep software pipelining, not with
warp count, and well-tuned kernels routinely exceed 80% of peak at 12.5%
occupancy. Anyone claiming "16.7% occupancy, therefore slow" is reasoning wrongly.

The evidence that something *is* being left behind is the **conjunction**:

1. tensor pipe at only 48% while it is the highest-utilised pipe;
2. L2 at 69-74% — ncu names it the bottleneck;
3. **0.15 eligible warps per scheduler** out of 2 resident, i.e. for ~92% of
   cycles no warp can issue at all;
4. `CPIStall`: *"each warp spends 11.7 cycles stalled waiting for the execution
   pipe ... all active warps execute their next instruction on a specific,
   oversubscribed math pipeline."*

Read together: with only 2 warps per scheduler there is not enough thread-level
parallelism to cover MMA latency *while* the L2 is also 69% busy feeding the
tiles. If the tensor pipe were genuinely the binding limit it would read near
100%, not 48%. It is a joint latency-hiding shortfall, and L2 is the more loaded
of the two resources.

### The open experiment for 3b

These are **cuBLAS heuristic-selected** kernels for shape
`512 x {3584, 18944} x {3584, 18944}`. Nobody chose them for this shape. Three
measurable questions, none yet answered:

1. **Does a different tile do better at exactly this shape?** A tile trading
   register/shared-memory footprint for occupancy, or a larger tile that cuts L2
   traffic per FLOP. Testable standalone against cuBLAS with CUTLASS profiler —
   no llama.cpp changes needed to *measure* it.
2. **How much L2 traffic per FLOP is actually required?** The 4090 has 72 MB of L2
   and a 512-token activation tile is only 512 x 3584 x 2 = 3.7 MB, so this is a
   bandwidth problem, not capacity. How far above the compulsory minimum is the
   69%? Unmeasured.
3. **Would raising `n_ubatch` further help 3b as well as 3a?** AI rises with ubatch
   size, so a 2048-token ubatch has AI 2048 (12.5x ridge) and amortises weights 4x
   better than 512. The `-ub 640` experiment only removed the *remainder*; it did
   not test whether larger tiles-per-launch improve the GEMM's own efficiency.
   Cheap to run: sweep `-ub` over 592 / 1024 / 2048 and watch encode time and
   `Tensor Active`.

**Caveat when quoting the L2 numbers**: ncu defaults to `--cache-control all`, so
L1/L2 are flushed before each replay pass and the 90-97% hit rates reflect
intra-kernel reuse only. Real inter-kernel reuse can only be higher, so the L2
pressure reported here is a lower bound.

---

## Recommended order

1. **Ship `-ub 640`** (or `>= 592`). Validated: 90 ms/request, one flag, decode
   untouched. Note it must be re-validated for any other frame geometry, since the
   tokens-per-frame figure changes with resolution.
2. **Sweep `-ub` to 1024 / 2048.** Tests 3a's remainder *and* 3b question 3 at
   once. Costs one afternoon.
3. **Merge the separator tokens into the image ubatch.** 214 ms/request available,
   but needs a mixed token/embedding batch path in `mtmd_helper`.
4. **Tile study for 3b.** Highest microarchitectural interest, most work, and the
   one item whose prize is genuinely unknown until measured.

Items 1-3 are all arithmetic-intensity fixes: they raise tokens per ubatch so the
GEMM crosses the ridge. Item 4 is the only one that asks whether the kernel is
right once it is on the correct side of the roofline.
