#!/usr/bin/env python3
"""Small evaluation sets from NExT-QA, Video-MME (short), Charades-STA, Perception Test (sample split) and TACoS
(Multi-Level, MPII Cooking 2 videos) in the layout llama-jetson-vlm reads.

Two steps, because the annotations need pandas (env vlm-eval) and the videos need OpenCV (conda env VLM):

    python prepare_newds.py select  <nextqa|videomme|charades|perception|tacos> <root> [n_clips=10] [seed=0]
    python prepare_newds.py extract <root> [max_width=640]

select writes <root>/items.json: per clip the source video, the prompt, and the answer (multiple choice: the option
letter) or the reference captions (Charades: every Charades-STA sentence of the video; TACoS: every sentence whose
interval overlaps the clip's by IoU >= 0.5). extract writes the frames to <root>/training/clean/<clip>/ with the MSVD
sampling (prepare_msvd.py): 4 frames per second, at most 50 frames spread over the clip (TACoS: the clip's interval
of the video), frames wider than max_width scaled to max_width.

Perception Test: every static-camera sample video, n_clips multiple-choice questions per video, each its own clip (<video>_q<id>, the
frames shared through a symlink). TACoS: segments of 5-15 s of the Detailed descriptions, n_clips videos (downloaded
to tacos/videos/), two segments per video.
"""
import json, os, random, sys, tempfile, zipfile

DATA = "/mnt/external_ssd/VLM-datasets"
FPS, MAX_FRAMES = 4.0, 50
TACOS_URL = "http://datasets.d2.mpi-inf.mpg.de/MPII-Cooking-2/videos"
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
    elif ds == "perception":
        d = json.load(open(f"{DATA}/perception_test/sample.json"))
        for v, a in sorted(d.items()):
            if a["metadata"]["is_camera_moving"]:
                continue
            for q in rng.sample(a["mc_question"], min(n, len(a["mc_question"]))):
                opts = "\n".join(f"{'ABC'[i]}. {o}" for i, o in enumerate(q["options"]))
                items[f"{v}_q{q['id']}"] = dict(src=f"zip:{DATA}/perception_test/sample_videos.zip:videos/{v}.mp4",
                                                answer="ABC"[q["answer_id"]], qtype=q["area"] + "/" + q["reasoning"],
                                                prompt=f"{q['question']}\nOptions:\n{opts}\n{MC_TAIL}")
    elif ds == "tacos":
        import csv, urllib.request
        rows = []
        for f in ("annosDetailed-processed.csv", "annosShort-processed.csv"):
            for r in csv.reader(open(f"{DATA}/tacos/TACoS-Multi-Level/corpus/{f}"), delimiter="\t"):
                rows.append((r[0], int(r[1]), int(r[2]), r[6].strip(), f.startswith("annosDetailed")))
        vids = sorted({r[0] for r in rows})
        rng.shuffle(vids)
        iou = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))
        os.makedirs(f"{DATA}/tacos/videos", exist_ok=True)
        for v in vids[:n]:
            path = f"{DATA}/tacos/videos/{v}-cam-002.avi"
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"{TACOS_URL}/{v}-cam-002.avi", path)
            segs = sorted({(s, e) for f, s, e, _, det in rows if f == v and det and 5 * 29.4 <= e - s <= 15 * 29.4})
            picked = 0
            for s, e in rng.sample(segs, len(segs)):
                refs = sorted({t for f, s2, e2, t, _ in rows if f == v and iou((s, e), (s2, e2)) >= 0.5})
                if len(refs) >= 3:
                    items[f"{v}_{s}"] = dict(src=path, start_frame=s, end_frame=e, refs=refs,
                                             prompt="Describe what the person is doing in this video in one sentence. Answer in English.")
                    picked += 1
                if picked == 2:
                    break
    os.makedirs(root, exist_ok=True)
    json.dump(items, open(f"{root}/items.json", "w"), indent=1)
    print(f"{len(items)} items -> {root}/items.json")


def frames_of(video, out_dir, max_w, first=0, last=None):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total - 1, last if last is not None else total - 1)
    n = max(1, int(round((last - first + 1) / fps * FPS)))
    idx = [min(last, first + int(round(i * fps / FPS))) for i in range(n)]
    if len(idx) > MAX_FRAMES:
        idx = [idx[int(round(i * (len(idx) - 1) / (MAX_FRAMES - 1)))] for i in range(MAX_FRAMES)]
    os.makedirs(out_dir, exist_ok=True)
    want, k, i = set(idx), 0, 0
    ok, img = cap.read()
    while ok and k < len(idx):   # decode sequentially: seeking is unreliable on some of these encodes
        if i in want:
            for _ in range(idx.count(i)):
                if img.shape[1] > max_w:
                    img = cv2.resize(img, (max_w, int(round(img.shape[0] * max_w / img.shape[1]))),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(out_dir, f"frame_{k + 1:04d}.png"), img)
                k += 1
        ok, img = cap.read()
        i += 1
    return k


def extract(root, max_w):
    items = json.load(open(f"{root}/items.json"))
    keep, done = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for clip, it in items.items():
            out = f"{root}/training/clean/{clip}"
            key = (it["src"], it.get("start_frame"), it.get("end_frame"))
            if key in done:   # another question on the same video: share the frames
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.lexists(out):
                    os.symlink(os.path.basename(done[key][0]), out)
                n = done[key][1]
            else:
                src = it["src"]
                if src.startswith("zip:"):
                    _, zpath, member = src.split(":", 2)
                    src = zipfile.ZipFile(zpath).extract(member, tmp)
                n = frames_of(src, out, max_w, it.get("start_frame", 0), it.get("end_frame"))
                done[key] = (out, n)
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
        extract(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 640)
