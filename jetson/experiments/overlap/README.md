# ViT encode / LLM prefill overlap on one GPU

Can frame k's prefill run while frame k+1 is encoded? `out.txt` / `overlap_check.cpp` (2026-10-13) tested
full frames with no switches (ViT flash attention on): 0.99-1.02x. `out_switches.txt` (2026-10-16) re-tests with
`jetson-overlap-probe` (`jetson/tools/overlap_probe.cpp`):
- all switches on (W, F, B, T) and ViT flash attention off;
- window-cache crops (9x4 = full frame, 5x2, 1x2);
- the real dependency (prefill k waits for encode k);
- CUDA stream priorities per model (new off-by-default switch, `ggml_backend_cuda_set_thread_stream_priority`).

8 frames, one 576-token prefill per frame; round 1 of 3 (the rounds agree within 1%):

| crop | priority | serial ms | pipelined ms | speedup | ideal | ViT concurrent / solo | prefill concurrent / solo |
|---|---|---|---|---|---|---|---|
| 9x4 | equal | 6,528 | 6,424 | 1.016x | 1.77x | 1.62x | 1.83x |
| 9x4 | ViT high | 6,524 | 6,480 | 1.007x | 1.77x | 1.09x | 1.85x |
| 9x4 | LLM high | 6,534 | 6,437 | 1.015x | 1.77x | 1.85x | 1.07x |
| 5x2 | equal | 4,177 | 4,096 | 1.020x | 1.23x | 3.03x | 1.21x |
| 5x2 | ViT high | 4,177 | 4,116 | 1.015x | 1.23x | 1.18x | 1.22x |
| 5x2 | LLM high | 4,184 | 4,107 | 1.019x | 1.23x | 4.09x | 1.08x |
| 1x2 | equal | 3,477 | 3,434 | 1.012x | 1.05x | 6.73x | 1.04x |
| 1x2 | ViT high | 3,485 | 3,446 | 1.011x | 1.05x | 1.72x | 1.04x |
| 1x2 | LLM high | 3,503 | 4,082 | 0.858x | 1.05x | 9.27x | 1.23x |

(1x2 with the LLM at high priority varies between rounds: 1.003x, 0.858x and 1.001x.)

- The overlap still gains only 1-2%, with any crop and any priority. The GPU's work is conserved: whichever
  stage gets priority runs at 1.07-1.18x its solo time, and the other absorbs the rest.
- Why: the prefill leaves no SM capacity free. Its GEMMs launch 56-296 tiles of 256x128 (one CTA per SM, most of
  the shared memory) and run at 63-82% of the BF16 tensor peak, so the ViT's blocks most likely get an SM only
  between GEMM waves rather than beside them (not measured). Either way, the two stages time-share the SMs.
  The 1-2% comes from the gaps between kernels and from the partial last wave of each GEMM.
- So overlapping on this GPU can only hide work if one stage stops using the SMs. The OFA flow (a separate
  engine) pipelines because it uses no SMs; the ViT and the LLM both run on the SMs.

Bug found and fixed on the way: `ggml-cuda/jetson-gemm.cu` kept one cuBLASLt workspace per device. Split-K GEMMs
keep partial sums and reduction counters there, so a ViT and an LLM GEMM running at the same time shared it: the
first sweep deadlocked (1x2, third round) and could have computed wrong values. The workspace is now per stream.
The serial pipeline was not affected (one stream at a time).

    MTMD_VIT_WINDOW_BATCH=1 GGML_CUDA_JETSON_GEMM_F32OUT=1 GGML_CUDA_JETSON_GEMM_BIAS=1 \
    GGML_CUDA_JETSON_GEMM_TABLE=jetson/gemm/table_orin.txt [OVERLAP_VIT_PRIO=-5] [OVERLAP_LLM_PRIO=-5] \
    build/bin/jetson-overlap-probe <llm.gguf> <mmproj.gguf> <frame.png> 5x2 8 3
