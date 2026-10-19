#!/usr/bin/env python3
"""How redundant are the frames a content-blind sampler hands the VLM? Motivation data, no pipeline involved.

    python video_redundancy.py [out_dir=experiments/video_redundancy] [workers=12]     (conda env VLM: OpenCV, PyAV)

Sampling: Qwen2.5-VL's reference video preprocessing (qwen-vl-utils smart_nframes): 2 fps, rounded to an even
count, at least 4 frames, indices spread uniformly over the clip -- blind to the content. Frames are decoded from
each clip's source video (Sintel: its 24 fps frames), resized as Qwen does (sides multiples of 28 px, the size of
one merged visual token), long side <= 644 px.

Per sampled frame after the first, against the previous sampled frame:
  codec   x264 (CRF 23, preset medium, no B frames, no scene cut) codes the sampled frames as one sequence. Its
          per-frame stats split the 16x16 macroblocks into intra (not predictable from the previous sampled frame:
          new content), inter (predicted from it, with motion and a residual) and skip (copied unchanged from the
          same place). new_info = the frame's size in that sequence / its size coded on its own (all-intra encode).
  tokens  the share of 28x28 token patches unchanged at the same position: patch PSNR >= 30 / 35 / 40 dB.
Writes per_frame.csv, per_clip.csv and summary.md. FRAMES_DIR=<dir>: also save the sampled frames as
<dir>/<dataset>/<clip>/frame_NNNN.png (input for jetson-embed-frames; video_redundancy_embed.py).
"""
import csv, glob, json, math, os, re, subprocess, sys, tempfile, zipfile
from multiprocessing import Pool
import av
import cv2
import numpy as np

EVAL = "/mnt/external_ssd/VLM-datasets/eval"
MSVD = "/home/nvidia/data/MSVD/YouTubeClips"
SINTEL = "/home/nvidia/data/MPI-Sintel/training/clean"
DATASETS = ["msvd", "sintel", "charades", "nextqa", "videomme", "perception", "tacos"]
QWEN_FPS, MIN_FRAMES, MAX_FRAMES, TOK, LONG = 2.0, 4, 768, 28, 644
PSNR = (30, 35, 40)


def qwen_indices(total, fps, first=0):
    """qwen-vl-utils smart_nframes + linspace: the frame indices Qwen2.5-VL samples from `total` frames at `fps`."""
    n = round(total / fps * QWEN_FPS / 2) * 2
    n = int(min(max(n, MIN_FRAMES), MAX_FRAMES, total // 2 * 2))
    return [first + int(round(x)) for x in np.linspace(0, total - 1, n)]


def resize(im):
    h, w = im.shape[:2]
    s = min(1.0, LONG / max(h, w))
    W, H = (max(TOK, int(round(x * s / TOK)) * TOK) for x in (w, h))
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)


def decode(src, tmp, first=0, last=None):
    if src.startswith("zip:"):
        _, z, member = src.split(":", 2)
        src = zipfile.ZipFile(z).extract(member, tmp)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total - 1, last if last is not None else total - 1)
    want = qwen_indices(last - first + 1, fps, first)
    ims, i = [], 0
    ok, im = cap.read()
    while ok and len(ims) < len(want):   # sequential decode: seeking is unreliable on some of these encodes
        ims += [resize(im)] * want.count(i)
        ok, im = cap.read()
        i += 1
    return ims


def encode(tmp, keyint):
    """Encode <tmp>/frames.npy with x264 into <tmp>/x264_<keyint>.log; its own process, since x264 completes the
    stats file only when the encoder is freed."""
    ims = np.load(os.path.join(tmp, "frames.npy"))
    out = av.open(os.path.join(tmp, f"v{keyint}.mp4"), "w")
    s = out.add_stream("libx264", rate=2)
    s.height, s.width = ims[0].shape[:2]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "23", "preset": "medium", "bf": "0",
                 "x264-params": f"pass=1:stats={tmp}/x264_{keyint}.log:keyint={keyint}:min-keyint=1:scenecut=0"}
    for im in ims:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(im[..., ::-1]), format="rgb24")):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()


def x264_stats(ims, tmp, keyint):
    np.save(os.path.join(tmp, "frames.npy"), np.stack(ims))
    subprocess.run([sys.executable, os.path.abspath(__file__), "--encode", tmp, str(keyint)], check=True)
    fr = {}
    for l in open(os.path.join(tmp, f"x264_{keyint}.log")):
        m = re.match(r"in:(\d+) .*type:(\w).*tex:(\d+) mv:(\d+) misc:(\d+) imb:(\d+) pmb:(\d+) smb:(\d+)", l)
        if m:
            fr[int(m[1])] = dict(bits=int(m[3]) + int(m[4]) + int(m[5]), imb=int(m[6]), pmb=int(m[7]), smb=int(m[8]))
    return [fr[i] for i in range(len(ims))]


def patch_psnr(a, b):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    H, W = d.shape[:2]
    mse = d.reshape(H // TOK, TOK, W // TOK, TOK, 3).mean(axis=(1, 3, 4))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-3))


def clip_stats(job):
    ds, clip, spec = job
    with tempfile.TemporaryDirectory() as tmp:
        if ds == "sintel":
            fs = sorted(glob.glob(f"{SINTEL}/{clip}/*.png"))
            ims = [resize(cv2.imread(fs[i])) for i in qwen_indices(len(fs), 24.0)]
        else:
            ims = decode(spec["src"], tmp, spec.get("start_frame", 0), spec.get("end_frame"))
        seq, intra = x264_stats(ims, tmp, 100000), x264_stats(ims, tmp, 1)
    if os.environ.get("FRAMES_DIR"):
        d = f"{os.environ['FRAMES_DIR']}/{ds}/{clip}"
        os.makedirs(d, exist_ok=True)
        for i, im in enumerate(ims):
            cv2.imwrite(f"{d}/frame_{i + 1:04d}.png", im)
    rows = []
    for t in range(1, len(ims)):
        mb = seq[t]["imb"] + seq[t]["pmb"] + seq[t]["smb"]
        p = patch_psnr(ims[t], ims[t - 1])
        rows.append(dict(dataset=ds, clip=clip, t=t, intra=seq[t]["imb"] / mb, inter=seq[t]["pmb"] / mb, skip=seq[t]["smb"] / mb,
                         new_info=seq[t]["bits"] / intra[t]["bits"], **{f"same{q}": float((p >= q).mean()) for q in PSNR}))
    return rows


def jobs():
    for ds in DATASETS:
        if ds == "msvd":
            for clip in sorted(os.listdir("/home/nvidia/data/accuracy/msvd/training/clean")):
                yield ds, clip, dict(src=f"{MSVD}/{clip}.avi")
        elif ds == "sintel":
            for clip in sorted(os.listdir(SINTEL)):
                yield ds, clip, None
        else:
            seen = set()
            for clip, it in json.load(open(f"{EVAL}/{ds}/items.json")).items():
                key = (it["src"], it.get("start_frame"))
                if key not in seen:   # Perception Test: several questions on one video
                    seen.add(key)
                    yield ds, clip, it


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/video_redundancy")
    os.makedirs(out, exist_ok=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 12) as pool:
        res = [r for r in pool.map(clip_stats, list(jobs()), chunksize=1) if r]
    frames = [r for rs in res for r in rs]
    keys = ["intra", "inter", "skip", "new_info"] + [f"same{q}" for q in PSNR]
    clips = []
    for rs in res:
        c = dict(dataset=rs[0]["dataset"], clip=rs[0]["clip"], sampled=len(rs) + 1)
        c.update({k: float(np.mean([r[k] for r in rs])) for k in keys})
        b = np.sort([r["new_info"] for r in rs])[::-1]
        c["info_top25"] = float(b[:max(1, math.ceil(len(b) / 4))].sum() / b.sum())   # share of new information in the top quarter
        clips.append(c)
    for name, data in (("per_frame.csv", frames), ("per_clip.csv", clips)):
        with open(f"{out}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in d.items()} for d in data)
    cols = ["sampled"] + keys + ["info_top25"]
    lines = ["| dataset | clips | frames | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for ds in DATASETS + ["all"]:
        C = [c for c in clips if ds in ("all", c["dataset"])]
        F = [f for f in frames if ds in ("all", f["dataset"])]
        lines.append(f"| {ds} | {len(C)} | {len(F)} | " + " | ".join(f"{np.mean([c[k] for c in C]):.3g}" for k in cols) + " |")
    lines.append("\nMeans over clips; frames = sampled frames after each clip's first. Shares 0-1; new_info = a frame's size coded "
                 "after the previous sampled frame / coded on its own; info_top25 = share of a clip's new information in its "
                 "top quarter of frames.")
    open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    encode(sys.argv[2], int(sys.argv[3])) if sys.argv[1:2] == ["--encode"] else main()
