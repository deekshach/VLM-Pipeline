#!/usr/bin/env python3
"""Extract MSVD test clips as frame folders in the layout llama-jetson-vlm reads (<root>/training/clean/<id>/).

Same sampling as /home/nvidia/data/MSVD/MSVD_prepared: 4 frames per second, at most 50 frames (spread uniformly over
the clip when it is longer). Frames wider than 640 px (MSVD sources range from 320x240 to 1920x1080) are scaled to
640 px wide, aspect kept: with ViT flash attention off a 1280x720 frame (1,196 tokens) costs ~3 s of dense attention. Captions go to <root>/captions.json. Needs OpenCV
(conda env VLM). Live OFA flow (--live-flow) computes the flow from the frames, so no flow files are needed.

    python prepare_msvd.py <root> [n_clips=100] [seed=0]
"""
import json, os, random, sys
import cv2

MSVD = "/home/nvidia/data/MSVD"
FPS, MAX_FRAMES, MAX_W = 4.0, 50, 640


def extract(video, out_dir):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    ok, img = cap.read()
    while ok:
        frames.append(img)
        ok, img = cap.read()
    if not frames:
        return 0
    n = max(1, int(round(len(frames) / fps * FPS)))
    idx = [min(len(frames) - 1, int(round(i * fps / FPS))) for i in range(n)]
    if len(idx) > MAX_FRAMES:
        idx = [idx[int(round(i * (len(idx) - 1) / (MAX_FRAMES - 1)))] for i in range(MAX_FRAMES)]
    os.makedirs(out_dir, exist_ok=True)
    for k, i in enumerate(idx):
        img = frames[i]
        if img.shape[1] > MAX_W:
            img = cv2.resize(img, (MAX_W, int(round(img.shape[0] * MAX_W / img.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"frame_{k + 1:04d}.png"), img)
    return len(idx)


def main():
    root = sys.argv[1]
    n_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    test = json.load(open(f"{MSVD}/msvd_test.json"))
    random.Random(seed).shuffle(test)
    captions = {}
    for item in test:
        if len(captions) >= n_clips:
            break
        vid = item["video_id"]
        n = extract(f"{MSVD}/YouTubeClips/{item['video']}", f"{root}/training/clean/{vid}")
        if n >= 8:   # the baseline samples 8 frames
            captions[vid] = item["caption"]
            print(f"{vid}: {n} frames", flush=True)
    json.dump(captions, open(f"{root}/captions.json", "w"), indent=1)
    print(f"{len(captions)} clips -> {root}")


if __name__ == "__main__":
    main()
