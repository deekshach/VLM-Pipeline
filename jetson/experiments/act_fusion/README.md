# GEMM-input fusion (ACT switches) and the wider GEMM configuration search

Motivation: the ncu ablation (`../profiles/ablation/`) found the elementwise kernels between the prefill GEMMs at 23%
of a 512-token ubatch. They are bound by DRAM (L2 hit 33-52%, long-scoreboard stalls), because a 512-token activation
(7.3 MB F32; 39 MB for the FFN intermediate) is larger than the 4 MB L2: every hand-off between kernels goes
through DRAM (warm vs flushed caches: 1-2% reuse). Since the L2 cannot grow, the intermediates have to go:

| switch | flag / env | what it removes |
|---|---|---|
| D | `--act-dedup` / `GGML_CUDA_JETSON_ACT_DEDUP=1` | each GEMM converted its own F32 input to BF16: the attention-norm output 3x (Q, K, V), the FFN-norm output 2x (gate, up). Now converted once and shared |
| G | `--act-fuse` / `GGML_CUDA_JETSON_ACT_FUSE=1` | RMS norm x weight and SwiGLU write the BF16 GEMM input directly; their F32 output (and the conversion pass) is never written |

Both are bit-identical: they move the F32 -> BF16 rounding, not change it. Verified: LLM logits (FNV hash of the last
token's logits, `PROBE_LOGITS=1` in `jetson-prefill-probe`) and ViT embeddings (`PROBE_DUMP_PREFIX`) are byte-equal
with and without. Flash attention for the LLM prefill stays on (AUTO); the ViT keeps it off.

## Kernel time (nsys, mean of 3 prefills / 5 encodes, all other switches on)

LLM prefill, 8 frames (4,608 tokens), ms (round with no clock drift between runs):

| class | G off | G on | change |
|---|---|---|---|
| cast (F32 -> BF16) | 278 | 26 | -253 |
| SwiGLU | 192 | 169 | -23 (writes BF16: half the bytes) |
| norm | 82 | 77 | -5 |
| GEMM, attention, add, rope | 2,277 | 2,280 | unchanged |
| all | 2,830 | 2,552 | -9.8% |

The remaining 26 ms of casts are the attention output (input of the O projection), which flash attention writes F32.

ViT (window batch on): 9x4 410 -> 382 ms (-7%), 5x2 113 -> 101 ms (-11%); casts 44 -> 6 / 13 -> 2 ms. The ViT
runs as a captured CUDA graph; the BF16 buffers work under capture (never freed, only retired when they grow).

D alone gives a smaller gain (probe 8 frames: 2,820 -> 2,753 ms) and nothing on top of G: G's BF16 norm output is
already shared by Q/K/V and gate/up. Use G.

## TTFT (`run_act_sweep.sh`, 6 clips, `summary.txt`)

| config | TTFT ms | ViT ms | image prefill ms | text prefill ms |
|---|---|---|---|---|
| all switches + anchor-first | 4,700 | 2,136 | 1,976 | 488 |
| + act-dedup | 4,655 | 2,096 | 1,969 | 491 |
| + act-fuse | 4,407 (-6.2%) | 1,979 | 1,840 | 487 |
| uniform 8 frames + all switches | 7,772 | 3,453 | 3,342 | 738 |
| + act-fuse | 7,234 (-6.9%) | 3,156 | 3,103 | 739 |

Every clip is faster with act-fuse (per-clip numbers in `summary.txt`).

## Wider GEMM configuration search (`jetson-gemm-tune ... mode=2`)

The table held the fastest of cuBLASLt's heuristic picks (about 8 per shape). Mode 2 enumerates every algorithm id
x tile x stages x split-K (1-8) x reduction scheme x CTA swizzle that `cublasLtMatmulAlgoCheck` accepts (900-2,400
per shape), screens them by a 3-call median and re-times the best 12. Prefill shapes at 512 tokens:

| GEMM (m n k) | previous table | search | |
|---|---|---|---|
| K/V 512 512 3584 (+bias) | 84.8 us | 74.4 us | -12% |
| gate/up 18944 512 3584 | 1,997 us | 1,956 us | -2% |
| down 3584 512 18944 | 2,588 us | 2,569 us | same |
| Q/O 3584 512 3584 | ~421 us | ~420 us | same |

Two rows updated in `table_orin.txt` (K/V and gate/up); prefill 8 frames 2,824 -> 2,820 ms (-0.15%). The down
projection stays at 63% of the BF16 tensor peak (gate/up: 82%) even with split-K variants that make its 56 tiles a
multiple of the 16 SMs, so its limit is the long K = 18,944 reduction, not the last wave. CTA swizzling alone
(mode 1) changed nothing (0.2%): the GEMMs are compute-bound, so the 1.4-1.5x weight re-reads from DRAM cost no time.

## Also measured: per-frame ubatches

The pipeline prefills each frame's 576 tokens as a 512 + 64 ubatch pair, and the 64-token ubatch streams all
weights for few tokens (about 90-100 ms; probe: N=576 392 ms vs N=1024 546 ms). `-ub 576` (one ubatch per frame;
GEMM shapes for 576 tuned and added to the table) gives uniform 8 frames, alley_1: image prefill 3,101 -> 3,014 ms
(-2.8%). 576 = 4.5 tiles of 128 tokens, so the 576-token GEMMs are 30% slower than the 512-token ones for 12.5% more
tokens; that eats most of the gain. Packing several frames into full 512-token ubatches would need the text marker
tokens between frames passed as embeddings (not done).
