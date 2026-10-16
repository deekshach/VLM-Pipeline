# Switch ablation: which optimization causes which change

`jetson/scripts/run_ablation.sh` runs every valid switch combination; `jetson/tools/ablation_analyze.py` writes
`report.md` (method in its docstring). Switches: W = batched-window attention, F = GEMM F32 output, B = bias
fusion (needs F), T = tuned GEMM table, C = window-cache crop (9x4 -> 5x2 windows of one frame). LLM prefill:
N = 1152 / 2304 / 4608 image tokens = 2 / 4 / 8 frames (adaptive sampling). ViT flash attention off.

Attribution = Shapley value: a switch's marginal effect averaged over every order in which the switches can
be enabled, so the values sum to the total base -> all-on change. "Alone" vs "Shapley" shows interaction.

## Real-clock kernel time (nsys, 2026-10-15; mean of 5 ViT / 3 prefill repetitions)

ViT, one frame, base 978.7 ms -> all switches 113.0 ms:

| switch | alone | Shapley | kernel classes it moves (Shapley, ms) |
|---|---|---|---|
| W batched windows | -455 | -257 (30%) | softmax -193, attention matmul -63; nothing else |
| F F32 output | -49 | -36 (4%) | casts -42, GEMM +8 (writes F32) |
| B bias fusion | (needs F) | -30 (3%) | adds -32 |
| T tuned table | -6 | -7 (1%) | split-K GEMM -21, GEMM +15 (the tuned picks drop split-K) |
| C crop 5x2 | -771 | -536 (62%) | every class, in proportion to the windows encoded |

- W and C overlap: both remove dense-attention work, so together they save less than the sum of their "alone"
  values. At a 1x2 crop, W changes nothing (31.8 -> 31.6 ms): the dense attention is already small.
- Each switch moves only its own kernel classes (off-target entries are below 1 ms), so the effects add up
  apart from the W x C overlap.

LLM prefill, 8 frames, base 4,292 ms -> all 2,834 ms: T -1,163 (80%, split-K GEMM 2,863 -> 678 ms), F -265
(casts -378, GEMM +96), B -31 (adds -29). The attention kernels do not change with any switch.

Adaptive sampling: in the last 512-token ubatch only the attention time grows with the KV length (18 -> 29 -> 54 ms
for 2 / 4 / 8 frames); every other class is constant (GEMMs always see 512 rows). The whole prefill is
1,075 / 2,117 / 4,292 ms: close to linear in frames, plus the attention term that grows with the KV length.

## Hardware counters (ncu, 2026-10-16; clocks at base, caches flushed before each kernel)

ViT, one frame (28 windowed + 4 global layers composed from two profiled layers), base -> all on:
986 -> 118 ms, DRAM read 56.3 -> 6.7 GB, L2 requests 154 -> 18 GB.

| switch | time | DRAM read | L2 requests | where (DRAM, Shapley) | mechanism |
|---|---|---|---|---|---|
| C crop 5x2 | 62% | 64% | 66% | all classes | fewer windows encoded |
| W batched windows | 29% | 28% (-14.0 GB) | 27% | softmax -8.6 GB, attn matmul -5.4 GB | 340 MB/layer score matrix -> 16 KB per (head, window) block |
| F F32 output | 5% | 2% (-1.2 GB) | 2% | casts -1.5 GB, GEMM +0.35 GB | one cast per GEMM removed; GEMM writes F32 |
| B bias fusion | 4% | 4% (-2.2 GB) | 3% | adds -2.0 GB | bias add in the GEMM epilogue |
| T tuned table | 1% | 1% (-0.6 GB) | 2% (-3.1 GB) | split-K -1.3 GB, GEMM +0.7 GB | less split-K partial-sum traffic |

Effects on efficiency (full frame):
- W: softmax L2 hit 39% -> 50%; the attention matrix multiplies go from barrier-bound (16% occupancy, 36% tensor)
  to latency-bound (long scoreboard, 28% tensor) but take 29 instead of 148 ms per frame. The 4 global layers stay
  dense and are now most of the remaining attention time.
- F + B: the remaining casts and adds have lower L2 hit (52/49% -> 33%); what was removed were the
  same-size streaming kernels. The whole ViT then stalls less on memory (long scoreboard 39% -> 32%).
- After all switches, the GEMMs are the largest class (145 of 421 ms per full frame, 49% tensor pipe).

Window cache x batched windows (DRAM read per encoded window):

| | full frame (36) | 5x2 (10) | 1x2 (2) |
|---|---|---|---|
| base | 1,564 MB | 1,018 MB | 1,172 MB |
| W | 826 MB | 884 MB | 1,160 MB |

- Without W, a crop's dense attention working set is smaller (softmax L2 hit 39% -> 59%), so each window costs
  35% less DRAM traffic: the L2 working-set effect.
- With W, the attention is already per window, so crops gain nothing per window; small crops cost *more* per
  window because the weights (about 39 MB per layer pass, read once per encode) are spread over fewer tokens. At
  1x2 the GEMMs reach 12% occupancy and 28% tensor pipe. W has no effect at 1x2 (41.5 vs 41.3 ms).

LLM prefill, last 512-token ubatch (x 28 layers), 8 frames: 540 -> 361 ms.

- T gives 81% of the time saved: tensor pipe 50% -> 86% on the weight GEMMs and L2 requests -33 GB (-22%),
  while DRAM traffic stays the same (28.7 -> 29.2 GB). Prefill GEMMs at 512 rows are limited by L2 traffic and
  instruction issue (MIO throttle), not by DRAM, so the gain comes from tiles and split-K, not from fewer bytes.
- F (17%) removes 1.4 GB of casts but the split-K GEMMs read 2.4 GB more (F32 partial sums), so DRAM read
  rises; time still drops 31 ms. B saves 4 ms (adds).
- No switch changes the attention kernel.

Adaptive sampling (KV length), last ubatch, base:

| frames (KV tokens) | KV per layer | attention ms | attention DRAM | L2 hit | rest of ubatch |
|---|---|---|---|---|---|
| 2 (1,152) | 2.4 MB | 19 | 430 MB | 95% | 484 ms |
| 4 (2,304) | 4.7 MB | 31 | 672 MB | 96% | 489 ms |
| 8 (4,608) | 9.4 MB | 57 | 1,263 MB | 96% | 484 ms |

- Attention traffic grows linearly with the KV length and the L2 hit rate does not drop when the KV per layer
  outgrows the 4 MB L2: flash attention streams K/V tiles, so no cliff appears at the L2 size. Fewer frames
  mainly remove whole ubatches (linear work); the KV term is 4-11% of a ubatch.

Pending: `sudo bash jetson/scripts/run_ablation.sh warm` (L2 reuse between kernels; report section 3).
