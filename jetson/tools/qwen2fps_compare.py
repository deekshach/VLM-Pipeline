#!/usr/bin/env python3
"""Accuracy and token diversity of Qwen2.5-VL's own 2 fps video input vs the uniform 8-frame baseline vs
motion-selected frames.
Python env: /home/nvidia/data/envs/vlm-eval.

    python qwen2fps_compare.py [out.md]

Configs, per clip (text from runs_eval, else the earlier runs; decoding is greedy, so answers are reproducible):
  base          8 frames spread evenly, one image per frame (llama-jetson-vlm --baseline)
  qwen2fps      every frame of the 2 fps sampling, the reference video input (scripts/run_eval.sh e8)
  ours_nocache  motion-selected frames (--live-flow --anchor-first), no window cache: the frame choice alone
  pw24          the design: motion-selected frames + per-window cache at threshold 24
  pwd24         the design + drop-clean: later frames prefill only their re-encoded windows' tokens
Accuracy, over the clips every config with runs has:
  multiple choice (NExT-QA, Video-MME, Perception Test)  option letter correct
  captions with references (MSVD, Charades, TACoS)        CIDEr, METEOR (corpus, pycocoevalcap), SBERT: best cosine
                                                           to a reference (all-mpnet-base-v2)
  Sintel (no references)                                   SBERT cosine to the base caption
  ROUGE-L is corpus-level (pycocoevalcap) like CIDEr and METEOR.
Visual tokens handed to the LLM (needs the --dump-dir of the run: img_NNN.f32, or <clip>.video.embd.bin):
  tokens   how many
  spec_H   spectral entropy in bits (analyze_accuracy.spectral_entropy): 2**spec_H = effective rank, the number of
           independent directions the token set spans; at most log2(min(tokens, 3584)), so it grows with the token
           count by construction
"""
import json, os, re, statistics as st, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy"))
import analyze_accuracy as aa
import vision_metrics as vm

ACC = "/home/nvidia/data/accuracy"
SOURCES = [os.environ.get("EVAL_RUNS", f"{ACC}/runs_eval"), f"{ACC}/runs"]
vm.SOURCES = SOURCES
EVAL = "/mnt/external_ssd/VLM-datasets/eval"
DATASETS = ["sintel", "msvd", "charades", "tacos", "nextqa", "videomme", "perception"]
CFGS = ["base", "qwen2fps", "ours_nocache", "pw24", "pwd24"]


def clips(ds):
    if ds == "msvd":
        return open(f"{ACC}/msvd_clips_45.txt").read().strip().split(",")
    root = "/home/nvidia/data/MPI-Sintel" if ds == "sintel" else f"{EVAL}/{ds}"
    return sorted(os.listdir(f"{root}/training/clean"))


def run(ds, cfg, clip):
    """(text, frames, dump dir or None) or None"""
    for src in SOURCES:
        f = f"{src}/{ds}/{cfg}/{clip}.json"
        if os.path.exists(f):
            j = json.load(open(f))
            d = f"{src}/{ds}/{cfg}/{clip}"
            d = d if os.path.isdir(d) else None
            if "requests" in j:
                if not j["requests"]:
                    return None
                r = j["requests"][0]
                return r["text"].strip(), r["n_frames"], d
            return j.get("inference", {}).get("generated_text", "").strip(), len(j.get("selected_frames", [])), d
    return None


def visual_tokens(ds, cfg, clip):
    """the visual tokens the LLM received (vision_metrics.load: drop-clean sends only the re-encoded windows),
    (n, 3584) float32, or None without a dump"""
    r = vm.load(ds, cfg, clip)
    return None if r is None or not r["groups"] else np.concatenate([g["sent"] for g in r["groups"]])


def spectral_entropy(x):
    """analyze_accuracy.spectral_entropy with the Gram matrix in float32 (tens of thousands of tokens)"""
    u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    g = (u.T @ u if u.shape[0] > u.shape[1] else u @ u.T).astype(np.float64)
    ev = np.clip(np.linalg.eigvalsh(g), 0, None)
    p = ev / ev.sum()
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def letter(text):
    m = re.match(r"\s*\(?([A-E])\b", text or "")
    return m[1] if m else None


def main():
    out = ["| dataset | clips | metric | " + " | ".join(CFGS) + " |", "|---|---|---|" + "---|" * len(CFGS)]
    msvd_refs = json.load(open(f"{ACC}/msvd/captions.json"))
    for ds in DATASETS:
        runs = {c: {v: run(ds, c, v) for v in clips(ds)} for c in CFGS}
        have = [c for c in CFGS if any(runs[c].values())]
        common = [v for v in clips(ds) if all(runs[c][v] for c in have)]
        if not common:
            continue
        items = json.load(open(f"{EVAL}/{ds}/items.json")) if os.path.exists(f"{EVAL}/{ds}/items.json") else {}
        rows = {"frames": {c: st.mean(runs[c][v][1] for v in common) for c in have}, "tokens": {}, "spec_H": {}}
        for c in have:
            tok, sh = [], []
            for v in common:
                x = visual_tokens(ds, c, v)
                if x is not None:
                    tok.append(len(x))
                    sh.append(spectral_entropy(x))
            if len(sh) == len(common):   # only when every clip has its dump
                rows["tokens"][c], rows["spec_H"][c] = st.mean(tok), st.mean(sh)
        if items and "answer" in next(iter(items.values())):
            rows["accuracy %"] = {c: 100 * st.mean(letter(runs[c][v][0]) == items[v]["answer"] for v in common) for c in have}
        elif ds == "sintel":
            if "base" in have:
                b = aa.sbert([runs["base"][v][0] for v in common])
                rows["SBERT vs base"] = {c: float((aa.sbert([runs[c][v][0] for v in common]) * b).sum(1).mean())
                                         for c in have}
        else:
            refs = {v: (msvd_refs[v] if ds == "msvd" else items[v]["refs"]) for v in common}
            rows.update({"CIDEr": {}, "METEOR": {}, "ROUGE-L": {}, "SBERT vs refs": {}})
            for c in have:
                preds = {v: runs[c][v][0] for v in common}
                sc, _ = aa.coco_scores(preds, refs)
                rows["CIDEr"][c], rows["METEOR"][c], rows["ROUGE-L"][c] = sc["CIDEr"], sc["METEOR"], sc["ROUGE-L"]
                rows["SBERT vs refs"][c] = st.mean(float((aa.sbert(refs[v]) @ aa.sbert([preds[v]])[0]).max()) for v in common)
        for k, vals in rows.items():
            out.append(f"| {ds} | {len(common)} | {k} | " +
                       " | ".join(f"{vals[c]:.3g}" if c in vals else "" for c in CFGS) + " |")
    text = "\n".join(out) + "\n"
    print(text)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(text)


if __name__ == "__main__":
    main()
