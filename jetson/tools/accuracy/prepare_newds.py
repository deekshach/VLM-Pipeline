#!/usr/bin/env python3
"""Small evaluation sets from NExT-QA, Video-MME (short) and Charades-STA in the layout llama-jetson-vlm reads.

Two steps, because the annotations need pandas (env vlm-eval) and the videos need OpenCV (conda env VLM):

    python prepare_newds.py select  <nextqa|videomme|charades> <root> [n_clips=10] [seed=0]
    python prepare_newds.py extract <root>

select writes <root>/items.json: per clip the source video, the prompt, and the answer (multiple choice: the option
letter) or the reference captions (Charades: every Charades-STA sentence of the video). extract writes the frames to
<root>/training/clean/<clip>/ with the MSVD sampling (prepare_msvd.py): 4 frames per second, at most 50 frames spread
over the clip, frames wider than 640 px scaled to 640 px wide.
"""
import json, os, random, sys, tempfile, zipfile

DATA = "/mnt/external_ssd/VLM-datasets"
FPS, MAX_FRAMES, MAX_W = 4.0, 50, 640
MC_TAIL = "Answer with the option's letter from the given choices directly."


def select(ds, root, n, seed):
    import pandas as pd
    rng = random.Random(seed)
    items = {}
    if ds == "nextqa":
        d = pd.read_parquet(f"{DATA}/nextqa/nextqa_mc.parquet")
        vids = sorted(d["video"].unique())
        rng.shuffle(vids)
        for v in vids[:n]:
            q = d[d["video"] == v].sample(1, random_state=seed).iloc[0]
            opts = "\n".join(f"{'ABCDE'[i]}. {q[f'a{i}']}" for i in range(5))
            items[str(v)] = dict(src=f"zip:{DATA}/nextqa/videos.zip:NExTVideo/{v}.mp4", answer="ABCDE"[int(q["answer"])],
                                 qtype=q["type"], prompt=f"{q['question']}?\nOptions:\n{opts}\n{MC_TAIL}")
    elif ds == "videomme":
        d = pd.read_parquet(f"{DATA}/videomme/videomme.parquet")
        d = d[d["duration"] == "short"]
        vids = sorted(v for v in d["videoID"].unique() if os.path.exists(f"{DATA}/videomme/short/{v}.mp4"))
        rng.shuffle(vids)
        for v in vids[:n]:
            q = d[d["videoID"] == v].sample(1, random_state=seed).iloc[0]
            items[v] = dict(src=f"{DATA}/videomme/short/{v}.mp4", answer=q["answer"], qtype=q["task_type"],
                            prompt=f"{q['question']}\nOptions:\n" + "\n".join(q["options"]) + f"\n{MC_TAIL}")
    elif ds == "charades":
        d = pd.read_parquet(f"{DATA}/charades_sta/charades.parquet")
        members = {}
        for z in sorted(f for f in os.listdir(f"{DATA}/charades_sta") if f.endswith(".zip")):
            for name in zipfile.ZipFile(f"{DATA}/charades_sta/{z}").namelist():
                members[os.path.basename(name)] = f"zip:{DATA}/charades_sta/{z}:{name}"
        refs = d.groupby("video")["caption"].apply(list)
        vids = sorted(v for v, c in refs.items() if len(c) >= 3 and v in members)
        rng.shuffle(vids)
        for v in vids[:n]:
            items[v[:-4]] = dict(src=members[v], refs=[c.strip() for c in refs[v]],
                                 prompt="Describe what the person is doing in this video in one sentence. Answer in English.")
    os.makedirs(root, exist_ok=True)
    json.dump(items, open(f"{root}/items.json", "w"), indent=1)
    print(f"{len(items)} items -> {root}/items.json")


def frames_of(video, out_dir):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = max(1, int(round(total / fps * FPS)))
    idx = [min(total - 1, int(round(i * fps / FPS))) for i in range(n)]
    if len(idx) > MAX_FRAMES:
        idx = [idx[int(round(i * (len(idx) - 1) / (MAX_FRAMES - 1)))] for i in range(MAX_FRAMES)]
    os.makedirs(out_dir, exist_ok=True)
    want, k, i = set(idx), 0, 0
    ok, img = cap.read()
    while ok and k < len(idx):   # decode sequentially: seeking is unreliable on some of these encodes
        if i in want:
            for _ in range(idx.count(i)):
                if img.shape[1] > MAX_W:
                    img = cv2.resize(img, (MAX_W, int(round(img.shape[0] * MAX_W / img.shape[1]))),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(out_dir, f"frame_{k + 1:04d}.png"), img)
                k += 1
        ok, img = cap.read()
        i += 1
    return k


def extract(root):
    items = json.load(open(f"{root}/items.json"))
    keep = {}
    with tempfile.TemporaryDirectory() as tmp:
        for clip, it in items.items():
            src = it["src"]
            if src.startswith("zip:"):
                _, zpath, member = src.split(":", 2)
                src = zipfile.ZipFile(zpath).extract(member, tmp)
            n = frames_of(src, f"{root}/training/clean/{clip}")
            print(f"{clip}: {n} frames", flush=True)
            if n >= 8:   # the baseline samples 8 frames
                keep[clip] = it
    json.dump(keep, open(f"{root}/items.json", "w"), indent=1)
    print(f"{len(keep)} clips -> {root}")


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10,
               int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        extract(sys.argv[2])
