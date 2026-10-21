#!/usr/bin/env python3
"""Frames at Qwen2.5-VL's own per-frame resolution for every arm of the main evaluation. Needs OpenCV (conda env VLM).

    python prepare_qres.py [root=/mnt/external_ssd/VLM-datasets/qres]

Qwen's reference video preprocessing (qwen-vl-utils fetch_video) resizes each frame with smart_resize: sides rounded
to multiples of 28 px, and scaled down to at most max_pixels = min(VIDEO_MAX_PIXELS 768*28*28, VIDEO_TOTAL_PIXELS /
frames * 2) = 602,112 px (~768 visual tokens) for up to 300 frames, so for every arm here. The earlier frame sets
were sized otherwise: the pools (prepare_newds.py, prepare_msvd.py) scaled wide sources to 640 px (MSVD, Video-MME)
or 1280 px (Perception Test, TACoS), and the Qwen 2 fps frames (video_redundancy.py) to a long side of 644 px.
Here every clip gets, from its source video:
  pool      the frames llama-jetson-vlm selects from, same indices as before (4 fps, at most 50 frames over the clip)
  qwen2fps  Qwen's 2 fps sampling (video_redundancy.qwen_indices), the input of scripts/run_eval.sh e8
both at smart_resize(source), which also scales frames under min_pixels (128*28*28) up: most MSVD sources
(320x240 -> 392x280). Where the old frames already reach the model at that size (mtmd rounds the sides to 28 px and
does nothing else), the old directory is linked instead, so its runs stay valid. Each regenerated set is
checked against the old one: same frame count, and each new frame scaled to the old size matches the old frame
(PSNR >= 30 dB: same indices). Writes <root>/pool/<dataset>/{training/clean/<clip>,items.json},
<root>/qwen2fps/<dataset>/<clip>, and <root>/affected.txt (dataset clip pool|qwen2fps: what was regenerated).
"""
import glob, json, math, os, shutil, sys, tempfile, zipfile
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import video_redundancy as vr

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/external_ssd/VLM-datasets/qres"
ACC = "/home/nvidia/data/accuracy"
EVAL = "/mnt/external_ssd/VLM-datasets/eval"
OLD_Q = "/mnt/external_ssd/VLM-datasets/qwen2fps/frames"
SINTEL = "/home/nvidia/data/MPI-Sintel/training/clean"
MSVD = "/home/nvidia/data/MSVD/YouTubeClips"
FACTOR, MIN_PIXELS, MAX_PIXELS = 28, 128 * 28 * 28, 768 * 28 * 28
FPS, MAX_FRAMES = 4.0, 50


def smart_resize(h, w):
    """qwen-vl-utils smart_resize: (height, width)"""
    hb, wb = max(FACTOR, round(h / FACTOR) * FACTOR), max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        hb, wb = math.floor(h / beta / FACTOR) * FACTOR, math.floor(w / beta / FACTOR) * FACTOR
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb, wb = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def resize(im, hw):
    if im.shape[:2] == hw:
        return im
    up = hw[0] * hw[1] > im.shape[0] * im.shape[1]
    return cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)


def open_video(src, tmp):
    if src.startswith("zip:"):
        _, z, member = src.split(":", 2)
        src = zipfile.ZipFile(z).extract(member, tmp)
    return src


def video_info(src):
    cap = cv2.VideoCapture(src)
    return cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def decode_count(src):
    """frames a sequential decode yields (prepare_msvd counts these)"""
    cap, n = cv2.VideoCapture(src), 0
    while cap.grab():
        n += 1
    return n


def decode_some(src, first, want):
    """the frames at the absolute indices `want`, decoding sequentially as the earlier tools (seeking is unreliable on
    some of these encodes)"""
    cap, got, i = cv2.VideoCapture(src), {}, 0
    last = max(want)
    while i <= last and cap.grab():
        if i in want:
            got[i] = cap.retrieve()[1]
        i += 1
    return got


def pool_indices(ds, src, it):
    """absolute indices of the pool frames: prepare_msvd.extract (MSVD) / prepare_newds.frames_of (the others)"""
    fps, total = video_info(src)
    if ds == "msvd":
        n_dec = decode_count(src)
        n = max(1, int(round(n_dec / fps * FPS)))
        idx = [min(n_dec - 1, int(round(i * fps / FPS))) for i in range(n)]
    else:
        first = it.get("start_frame", 0)
        last = min(total - 1, it["end_frame"] if it.get("end_frame") is not None else total - 1)
        n = max(1, int(round((last - first + 1) / fps * FPS)))
        idx = [min(last, first + int(round(i * fps / FPS))) for i in range(n)]
    if len(idx) > MAX_FRAMES:
        idx = [idx[int(round(i * (len(idx) - 1) / (MAX_FRAMES - 1)))] for i in range(MAX_FRAMES)]
    return idx


def qwen_indices(src, it):
    """absolute indices of Qwen's 2 fps sampling, as video_redundancy.decode"""
    fps, total = video_info(src)
    first = it.get("start_frame", 0)
    last = min(total - 1, it["end_frame"] if it.get("end_frame") is not None else total - 1)
    return vr.qwen_indices(last - first + 1, fps, first)


def frames_in(d):
    return sorted(glob.glob(f"{d}/frame_*.png"))


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()), 1e-3))


def write_checked(ims, hw, out, old):
    """write the frames at hw, after checking them against the old set (count, content)"""
    olds = frames_in(old)
    if len(olds) != len(ims):
        raise RuntimeError(f"{out}: {len(ims)} frames, the old set has {len(olds)}")
    worst = 99.0
    for k in range(0, len(ims), max(1, len(ims) // 6)):
        o = cv2.imread(olds[k])
        worst = min(worst, psnr(cv2.resize(ims[k], (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA), o))
    if worst < 30:
        raise RuntimeError(f"{out}: differs from the old frames (PSNR {worst:.1f} dB): not the same indices")
    os.makedirs(out, exist_ok=True)
    for k, im in enumerate(ims):
        cv2.imwrite(f"{out}/frame_{k + 1:04d}.png", resize(im, hw))
    return worst


def link(target, at):
    os.makedirs(os.path.dirname(at), exist_ok=True)
    if not os.path.lexists(at):
        os.symlink(os.path.realpath(target), at)


def size_of(d):
    im = cv2.imread(frames_in(d)[0])
    return im.shape[:2]


def main():
    affected = []
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    for ds in ["sintel", "msvd", "charades", "nextqa", "videomme", "perception", "tacos"]:
        if ds == "sintel":
            specs = {c: None for c in sorted(os.listdir(SINTEL))}
            old_pool = lambda c: f"{SINTEL}/{c}"
        elif ds == "msvd":
            specs = {c: dict(src=f"{MSVD}/{c}.avi") for c in open(f"{ACC}/msvd_clips_45.txt").read().strip().split(",")}
            old_pool = lambda c: f"{ACC}/msvd/training/clean/{c}"
        else:
            specs = json.load(open(f"{EVAL}/{ds}/items.json"))
            old_pool = lambda c, ds=ds: f"{EVAL}/{ds}/training/clean/{c}"
            os.makedirs(f"{ROOT}/pool/{ds}", exist_ok=True)
            shutil.copy(f"{EVAL}/{ds}/items.json", f"{ROOT}/pool/{ds}/items.json")
        done = {}   # Perception Test: questions on one video share its frames
        for c, it in specs.items():
            pool_out, q_out = f"{ROOT}/pool/{ds}/training/clean/{c}", f"{ROOT}/qwen2fps/{ds}/{c}"
            key = None if it is None else (it["src"], it.get("start_frame"), it.get("end_frame"))
            if key in done:
                link(done[key][0], pool_out); link(done[key][1], q_out)
                for what in done[key][2]:
                    affected.append(f"{ds} {c} {what}")
                continue
            if ds == "sintel":
                fs = sorted(glob.glob(f"{SINTEL}/{c}/*.png"))
                src_hw = cv2.imread(fs[0]).shape[:2]
            else:
                src = open_video(it["src"], tmp)
                pidx, qidx = pool_indices(ds, src, it), qwen_indices(src, it)
                got = decode_some(src, 0, set(pidx) | set(qidx))
                src_hw = next(iter(got.values())).shape[:2]
            hw = smart_resize(*src_hw)
            what = []
            # pool: regenerate when Qwen's size differs from what mtmd makes of the old pool (sides rounded to 28 px,
            # no cap, no minimum: 320x240 stays 308x252 where Qwen scales it up to 392x280)
            if tuple(round(x / FACTOR) * FACTOR for x in size_of(old_pool(c))) == hw:
                link(old_pool(c), pool_out)
            else:
                ims = [got[i] for i in pidx if i in got]   # a container count past the end: those are not written
                p = write_checked(ims, hw, pool_out, old_pool(c))
                what.append("pool")
                print(f"{ds} {c}: pool {len(ims)} frames {size_of(old_pool(c))} -> {hw} (check {p:.0f} dB)", flush=True)
            # Qwen 2 fps frames
            old_q = f"{OLD_Q}/{ds}/{c}"
            if size_of(old_q) == hw:
                link(old_q, q_out)
            else:
                if ds == "sintel":
                    q = [cv2.imread(fs[i]) for i in vr.qwen_indices(len(fs), 24.0)]
                else:
                    q = [got[i] for i in qidx if i in got]
                p = write_checked(q, hw, q_out, old_q)
                what.append("qwen2fps")
                print(f"{ds} {c}: qwen2fps {len(q)} frames {size_of(old_q)} -> {hw} (check {p:.0f} dB)", flush=True)
            if ds != "sintel":
                os.remove(src) if src.startswith(tmp) else None
            if key is not None:
                done[key] = (pool_out, q_out, what)
            affected += [f"{ds} {c} {w}" for w in what]
    shutil.rmtree(tmp, ignore_errors=True)
    open(f"{ROOT}/affected.txt", "w").write("\n".join(affected) + "\n")
    print(f"{len(affected)} regenerated -> {ROOT}/affected.txt")


if __name__ == "__main__":
    main()
