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

## L2 reuse between kernels (warm vs flushed caches, 2026-10-16; report section 3)

Does a kernel find its producer's output still in the 4 MB L2? Section 2 flushes the caches before every
kernel, so it cannot tell. Two measurements:

- Time: nsys kernel time (real run, caches as the previous kernel left them) vs ncu time (flushed), per class, at
  the same clock (the base frame totals agree within 1%).
- DRAM bytes, LLM only: `run_ablation.sh warm` profiles the whole last-ubatch range in one result (app-range
  replay, caches not flushed).

Per-kernel ncu with `--cache-control none` was tried and dropped. Kernel replay saves and restores memory around
each kernel. Per-kernel application replay showed no change at all, even at 1x2 where the time shows casts 44%
faster. A microbenchmark confirms that L2 contents do survive a kernel boundary (a 2 MB buffer re-read
kernel-to-kernel at 160-220 GB/s, vs 75 GB/s DRAM), so ncu's own work between kernels disturbs L2. It also hung
on the ViT configs without F and on the LLM flash-attention kernel. app-range replay crashes the ViT probe.

| | full frame 9x4 | crop 5x2 | crop 1x2 |
|---|---|---|---|
| ViT base, warm vs flushed | -1% | -4% | -23% |
| ViT all switches (WFBT) | -2% | -5% | -19% |

| | 2 frames | 4 frames | 8 frames |
|---|---|---|---|
| LLM last ubatch, time | -9% | -9% | -8% |
| LLM last ubatch, DRAM read | -2% | -2% | -1% |

- Full frame: there is almost no reuse between kernels. The activations (2,304 patches x 1280 x 4 B = 11.8 MB per
  tensor) are about 3x the L2, so each kernel streams its input from DRAM. Only the casts and the GLU, which
  read what the previous kernel just wrote, gain 9-13%. The two largest classes (softmax, attention matmul)
  do not change.
- Crops: the window cache shrinks the working set below the L2 size, so the producer's output is still in L2.
  At 1x2 (128 patches, 0.6 MB per tensor), casts, adds and norms are 40-50% faster and softmax and the attention
  matmul 20-37% faster. In the real run, then, the 1x2 crop costs 24 ms, not the 30-41 ms the flushed counters
  imply. This gain from cross-kernel reuse is an extra effect of the window cache on top of the smaller work.
  The weight GEMMs gain little (-3 to -5%): their weights (about 39 MB per layer pass) are read once per encode.
- F + B remove the cast and add kernels, the classes that profit most from warm L2 at crops. So fusion saves
  less in the real run than in the flushed counters (1x2: FB -17% vs base -23% warm gain).
- LLM prefill: the warm run is 8-9% faster, but DRAM bytes drop only 1-2% (L2 hit 85-87%). A 512-token activation is
  3.7 MB in BF16, 7.3 MB in F32: the same size as or larger than the L2, and the weights (13 GB per ubatch) are
  read once. So the time saved is mostly latency (warm TLB and instruction caches, first wave of each kernel),
  not fewer bytes. Rope (-22%) and the GLU (-13%), which consume the previous kernel's output directly, gain the
  most. The gain does not depend on the KV length (2/4/8 frames).
- The microarchitectural findings of section 2 (which switch moves which bytes) stand. Their absolute ms at crops
  are upper bounds; use the nsys times (section 1) for TTFT.
