#!/usr/bin/env python3
"""Accuracy of Qwen2.5-VL's own 2 fps video input vs the uniform 8-frame baseline vs motion-selected frames.
Python env: /home/nvidia/data/envs/vlm-eval.

    python qwen2fps_compare.py [out.md]

Configs, per clip (text from runs_eval, else the earlier runs; decoding is greedy, so answers are reproducible):
  base          8 frames spread evenly, one image per frame (llama-jetson-vlm --baseline)
  qwen2fps      every frame of the 2 fps sampling, the reference video input (scripts/run_eval.sh e8)
  ours_nocache  motion-selected frames (--live-flow --anchor-first), no window cache: the frame choice alone
  pw24          the design: motion-selected frames + per-window cache at threshold 24
Accuracy, over the clips every config with runs has:
  multiple choice (NExT-QA, Video-MME, Perception Test)  option letter correct
  captions with references (MSVD, Charades, TACoS)        CIDEr, METEOR (corpus, pycocoevalcap), SBERT: best cosine
                                                           to a reference (all-mpnet-base-v2)
  Sintel (no references)                                   SBERT cosine to the base caption
"""
import json, os, re, statistics as st, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy"))
import analyze_accuracy as aa

ACC = "/home/nvidia/data/accuracy"
SOURCES = [os.environ.get("EVAL_RUNS", f"{ACC}/runs_eval"), f"{ACC}/runs"]
EVAL = "/mnt/external_ssd/VLM-datasets/eval"
DATASETS = ["sintel", "msvd", "charades", "tacos", "nextqa", "videomme", "perception"]
CFGS = ["base", "qwen2fps", "ours_nocache", "pw24"]


def clips(ds):
    if ds == "msvd":
        return open(f"{ACC}/msvd_clips_45.txt").read().strip().split(",")
    root = "/home/nvidia/data/MPI-Sintel" if ds == "sintel" else f"{EVAL}/{ds}"
    return sorted(os.listdir(f"{root}/training/clean"))


def run(ds, cfg, clip):
    """(text, frames, visual tokens) or None"""
    for src in SOURCES:
        f = f"{src}/{ds}/{cfg}/{clip}.json"
        if os.path.exists(f):
            j = json.load(open(f))
            if "requests" in j:
                if not j["requests"]:
                    return None
                r = j["requests"][0]
                return r["text"].strip(), r["n_frames"], r["n_visual_tokens"]
            return (j.get("inference", {}).get("generated_text", "").strip(), len(j.get("selected_frames", [])),
                    j["timing"]["breakdown"].get("rest", {}).get("tokens_encoded"))
    return None


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
        rows = {"frames": {c: st.mean(runs[c][v][1] for v in common) for c in have}}
        if items and "answer" in next(iter(items.values())):
            rows["accuracy %"] = {c: 100 * st.mean(letter(runs[c][v][0]) == items[v]["answer"] for v in common) for c in have}
        elif ds == "sintel":
            if "base" in have:
                b = aa.sbert([runs["base"][v][0] for v in common])
                rows["SBERT vs base"] = {c: float((aa.sbert([runs[c][v][0] for v in common]) * b).sum(1).mean())
                                         for c in have}
        else:
            refs = {v: (msvd_refs[v] if ds == "msvd" else items[v]["refs"]) for v in common}
            rows.update({"CIDEr": {}, "METEOR": {}, "SBERT vs refs": {}})
            for c in have:
                preds = {v: runs[c][v][0] for v in common}
                sc, _ = aa.coco_scores(preds, refs)
                rows["CIDEr"][c], rows["METEOR"][c] = sc["CIDEr"], sc["METEOR"]
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
