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

## Hardware counters

`sudo bash jetson/scripts/run_ablation.sh ncu` (DRAM read, L2 requests and hit rate, tensor pipe, occupancy,
stalls per class) and `sudo bash jetson/scripts/run_ablation.sh warm` (L2 reuse from one kernel to the
next); results land in `report.md` sections 2-3 when re-run.
