#!/usr/bin/env python
"""Check temporal_pair_vlm.cpp against the reference Qwen2.5-VL (Hugging Face).

Reads what `llama-temporal-pair-vlm --dump-dir D` fed the LLM and gives the
reference model the very same pixels and token ids:

  positions   reference get_rope_index vs the positions llama.cpp was given
  vision      reference vision tower output vs llama.cpp embeddings (per-token cosine),
              with two controls: the pair in swapped order, and frame 0 twice
  logits      reference next-token logits vs llama.cpp, plus a greedy continuation

Needs transformers <= 4.5x: later releases changed the video position formula.
Run without the user site so the env's own transformers is used:

    /scratch/softwares/anaconda/envs/vlm/bin/python -s verify_temporal_pair.py D video1,video2 [modes]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
P, MERGE, TP = 14, 2, 2


def patchify(x):
    """(T, C, H, W) with even T -> flattened patches, as Qwen2VLImageProcessor._preprocess"""
    t, c, h, w = x.shape
    gt, gh, gw = t // TP, h // P, w // P
    p = x.reshape(gt, TP, c, gh // MERGE, MERGE, P, gw // MERGE, MERGE, P)
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return p.reshape(gt * gh * gw, c * TP * P * P), (gt, gh, gw)


def cos_rows(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def main():
    dump, videos = sys.argv[1], sys.argv[2].split(",")
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["images", "pairs", "video"]
    print(f"transformers {transformers.__version__}, weights {SNAP}")
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = model.config
    dev = "cuda"
    report = []

    for video in videos:
        for mode in modes:
            base = os.path.join(dump, f"{video}.{mode}")
            j = json.load(open(base + ".json"))
            w, h = j["frame_size"]
            n_frames = len(j["frame_idx"])
            frames = np.fromfile(base + ".frames.bin", dtype=np.uint8).reshape(n_frames, h, w, 3)
            embd = np.fromfile(base + ".embd.bin", dtype=np.float32).reshape(-1, j["n_embd"])
            logits_cpp = np.fromfile(base + ".logits.bin", dtype=np.float32)
            tokens = np.array(j["tokens"])
            pos_cpp = np.array(j["pos"]).reshape(-1, 3).T          # (3, n)
            nx, ny = j["grid"]
            n_tok = nx * ny

            x = ((frames.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)   # T,C,H,W
            if mode == "images":
                groups = [np.stack([x[i], x[i]]) for i in range(n_frames)]   # a still image is repeated
            else:
                if n_frames % 2:
                    x = np.concatenate([x, x[-1:]])                          # odd count: repeat the last frame
                groups = [x[i:i + 2] for i in range(0, len(x), 2)]

            is_video = mode == "video"
            flat, grid = patchify(np.concatenate(groups))
            gt, gh, gw = grid
            pixel = torch.from_numpy(flat).to(dev, torch.bfloat16)
            grid_one = torch.tensor([[gt, gh, gw]], device=dev)                          # one video
            grid_each = torch.tensor([[1, gh, gw]] * gt, device=dev)                     # gt separate blocks

            # ---- vision tower
            with torch.no_grad():
                ref = model.visual(pixel, grid_thw=grid_one)
                ref = getattr(ref, "pooler_output", ref).float().cpu().numpy()
            c = cos_rows(ref, embd)
            line = {"video": video, "mode": mode, "vision_cos_mean": float(c.mean()), "vision_cos_min": float(c.min())}
            if mode != "images":
                f0, f1 = groups[0][0], groups[0][1]
                with torch.no_grad():
                    for name, pair in (("swapped", np.stack([f1, f0])), ("frame0_twice", np.stack([f0, f0]))):
                        fl, g = patchify(pair)
                        o = model.visual(torch.from_numpy(fl).to(dev, torch.bfloat16),
                                         grid_thw=torch.tensor([list(g)], device=dev))
                        o = getattr(o, "pooler_output", o).float().cpu().numpy()
                        line[f"control_{name}_cos"] = float(cos_rows(o, embd[:n_tok]).mean())

            # ---- prompt as the reference sees it
            pad = cfg.video_token_id if mode != "images" else cfg.image_token_id
            ids = torch.tensor([[int(t) if t >= 0 else pad for t in tokens]], device=dev)
            mask = torch.ones_like(ids)
            kw = dict(input_ids=ids, attention_mask=mask)
            if mode == "images":
                kw.update(pixel_values=pixel, image_grid_thw=grid_each)
                rope = dict(image_grid_thw=grid_each)
            else:
                g = grid_one if is_video else grid_each
                spg = torch.tensor([TP / j["video_fps"]] * len(g), device=dev)
                kw.update(pixel_values_videos=pixel, video_grid_thw=g, second_per_grid_ts=spg)
                rope = dict(video_grid_thw=g, second_per_grid_ts=spg)

            owner = model.model if hasattr(model.model, "get_rope_index") else model
            pos_ref, _ = owner.get_rope_index(ids, attention_mask=mask, **rope)
            pos_ref = pos_ref[:, 0].cpu().numpy()
            line["positions_equal"] = bool(pos_ref.shape == pos_cpp.shape and (pos_ref == pos_cpp).all())
            if not line["positions_equal"]:
                bad = np.nonzero((pos_ref != pos_cpp).any(0))[0]
                line["positions_first_mismatch"] = [int(bad[0]), pos_ref[:, bad[0]].tolist(), pos_cpp[:, bad[0]].tolist()]
            line["positions_next_ref"] = int(pos_ref.max() + 1)
            line["positions_next_cpp"] = int(j["next_pos"])

            # ---- logits at the first generated token
            with torch.no_grad():
                out = model(**kw)
                lr = out.logits[0, -1].float().cpu().numpy()
                gen = model.generate(**kw, max_new_tokens=48, do_sample=False)
            lc = logits_cpp[:len(lr)]
            pr = torch.log_softmax(torch.from_numpy(lr), -1)
            pc = torch.log_softmax(torch.from_numpy(lc), -1)
            line["logit_top1_ref"] = tok.decode([int(lr.argmax())])
            line["logit_top1_cpp"] = tok.decode([int(lc.argmax())])
            line["logit_top5_overlap"] = len(set(np.argsort(-lr)[:5]) & set(np.argsort(-lc)[:5]))
            line["logit_pearson"] = float(np.corrcoef(lr, lc)[0, 1])
            line["logit_kl_ref_cpp"] = float((pr.exp() * (pr - pc)).sum())
            line["ref_text"] = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            report.append(line)
            print(json.dumps(line, ensure_ascii=False))

    json.dump(report, open(os.path.join(dump, "verify_report.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
