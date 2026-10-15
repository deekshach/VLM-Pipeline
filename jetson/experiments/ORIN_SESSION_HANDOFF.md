# Orin session handoff (2026-10-13)

Give this file to a new Claude Code session to pick up where the last one stopped.
Full transcript of the previous session:
`/home/nvidia/.claude/projects/-home-nvidia-Desktop-Projects-VLM/5b030ded-dc0b-41a7-a96b-ff80f004e4f9.jsonl`
(`claude --resume` from `/home/nvidia/Desktop/Projects/VLM` also reopens it).

## Standing rules

- **All analysis is framed in terms of TTFT** (time to first token) = image load + tokenize + ViT + prefill. Decode is excluded.
- `AGENTS.md` (upstream llama.cpp policy) says AI agents should not write code for upstream contributions. Edits so far are to the user's own experiment files in `dmc955/` at their explicit request; this was disclosed.
- "f8" means the 8-frame operating point, not a model precision.

## Setup

- Board: Jetson AGX Orin Developer Kit, 61 GB unified RAM, 12 cores, L4T R36.4.7, CUDA 12.6 at `/usr/local/cuda-12.6`. Power mode was MAXN; **check it after reboot** (`nvpmodel -q`), since timings below assume MAXN.
- No passwordless sudo. `/data` is root-only, so everything lives under `/home/nvidia/data`.
- Repo: `/home/nvidia/Desktop/Projects/VLM/VLM-Pipeline`, branch `dmc955-vlm`, fork of llama.cpp at `b55dcdef`, remote `https://github.com/deekshach/VLM-Pipeline.git`.

| What | Where |
|---|---|
| MPI-Sintel (23 videos) | `/home/nvidia/data/MPI-Sintel/training/{clean,final,flow}` |
| MSVD | `/home/nvidia/data/MSVD` (`YouTubeClips/`, `msvd_{test,train,val}.json`) |
| MSVD prepared (20 test videos) | `/home/nvidia/data/MSVD/MSVD_prepared` (`frames/`, `flow/`, `manifest.json`, plus symlinks `training/final`, `training/flow`) |
| Model | `/home/nvidia/data/models/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf` and `mmproj-BF16.gguf` |
| Original archives (~7.5 GB, deletable) | `/home/nvidia/data/downloads` |
| Python env for MSVD prep (opencv, bundled ffmpeg) | `/home/nvidia/data/envs/msvd-prep` |

## Build

```
export PATH=/usr/local/cuda-12.6/bin:$PATH CPATH=/usr/local/cuda-12.6/include
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j10
```

- `LLAMA_BUILD_TESTS=OFF` is needed because `include/llama.h:725` has a C++ default argument that breaks `tests/test-c.c`.
- `CPATH` is needed so the C++ files of CUDA targets find `cuda_runtime.h`.
- Binaries are in `build/bin/` and survive the reboot. `dmc955/model_config.sh` defines `$MODEL` and `$MMPROJ`.

## Uncommitted changes in the repo (53 modified files, nothing committed)

1. **Path rewrite** in 49 files under `dmc955/`: old-machine paths (`/data/Fall25/...`, `/data/spring26/...`) replaced with the paths above.
2. **`--anchor-first`** in `dmc955/adaptive_sintel_vlm.cpp` (opt-in; behaviour unchanged without it). Frame 1 is always selected, and its chat header + ViT + prefill run on a worker thread while the main thread does the CPU flow analysis. The remaining frames are then appended to the same context. Also adds `timing.ttft_ms` to the result JSON.
3. New folders: `dmc955/results_orin_adaptive/` (sweep results, `anchor/` subfolder) and `dmc955/orin_session/` (baseline log, ViT/prefill overlap test source and output).

## Results so far

**Baseline** (`llama-baseline-vlm`, `alley_1`, 8 frames, mean of 3 warm requests):
ViT 4,848 ms (49.6% of TTFT), image prefill 3,945 ms (40.4%), text/other 977 ms (10.0%), TTFT about 9,770 ms. Decode 93 ms/token.

**ViT/prefill overlap test** (`orin_session/overlap/`): running the two concurrently on the one GPU gives a 0.99 to 1.02x speedup; each stage slows about 2x. Pipelining them gives no gain on this board.

**Adaptive sampling sweep**, 23 Sintel videos, versus uniform 8 frames:

| Group | Videos | Frames selected | TTFT change |
|---|---|---|---|
| Low motion | 14 | 3 to 4 | 32% to 48% faster |
| Medium motion | 2 | 8 | 13% to 17% slower |
| High motion | 7 | 11 to 14 | 41% to 90% slower |
| All | 23 | | mean 10,320 ms to 9,398 ms (9% faster) |

Flow analysis averaged 1,487 ms per video. Sweep TTFT was computed as flow analysis + selection + encode, because that binary predates the `ttft_ms` field.

**`--anchor-first` versus plain adaptive** (one cold run per mode):

| Video | Frames | Adaptive TTFT | Anchored TTFT | Saved |
|---|---|---|---|---|
| `alley_1` | 3 | 5,906 ms | 4,124 ms | 1,782 ms (30%) |
| `sleeping_1` | 3 | 5,790 ms | 4,134 ms | 1,656 ms (29%) |
| `ambush_7` | 4 | 7,045 ms | 5,357 ms | 1,688 ms (24%) |
| `temple_2` | 8 | 12,040 ms | 10,336 ms | 1,704 ms (14%) |
| `cave_2` | 11 | 15,947 ms | 14,210 ms | 1,737 ms (11%) |

Command used (per video, add `--anchor-first` for the anchored mode; see the `.log` files in `results_orin_adaptive/anchor/` for the exact invocation):
`--base-frames 8 -p "Describe the scene, motion patterns, and key visual elements in these video frames." -n 150 -ngl 99`

## Caveats and open items

- Anchoring changes which frames are picked: frame 1 takes one slot and the sampling sequence shifts (`alley_1`: 6, 24, 25 became 1, 6, 25). Output quality has not been scored for either mode.
- Anchored mode was run on 5 videos only, one cold run each. Not yet run across all 23.
- High-motion regressions come from the frame budget (up to 14 frames versus 8). Capping `--max-frames` at 8 would remove them; left unchanged because it is a policy choice.
- Ideas offered but not requested: cheaper flow analysis (subsampling, `nth_element` instead of a full sort, multithreading), Q8_0/Q4 model comparison, deleting `~/data/downloads`.
