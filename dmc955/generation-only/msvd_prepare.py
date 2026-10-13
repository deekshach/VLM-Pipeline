#!/usr/bin/env python3
"""
MSVD dataset preparation: extract frames + compute optical flow for selected test videos.
Outputs a directory structure compatible with pyramid_generation's frame loading.
"""

import os
import sys
import json
import subprocess
import struct
import numpy as np
import cv2

MSVD_DIR = "/data/spring26/CPP/datasets/MSVD"
OUTPUT_DIR = "/data/spring26/CPP/datasets/MSVD/MSVD_prepared"
VIDEO_DIR = os.path.join(MSVD_DIR, "YouTubeClips")
N_VIDEOS = 20  # number of test videos to prepare

def extract_frames(video_path, out_dir, max_frames=50):
    """Extract frames from video using ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=4",  # 4 fps sampling
        "-frames:v", str(max_frames),
        os.path.join(out_dir, "frame_%04d.png"),
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    # Count extracted frames
    count = 0
    for i in range(1, max_frames + 1):
        if os.path.exists(os.path.join(out_dir, f"frame_{i:04d}.png")):
            count = i
        else:
            break
    return count


def write_flo_file(path, flow):
    """Write optical flow in Middlebury .flo format."""
    h, w = flow.shape[:2]
    with open(path, 'wb') as f:
        f.write(struct.pack('f', 202021.25))  # magic number
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))
        # Interleave u,v
        for y in range(h):
            for x in range(w):
                f.write(struct.pack('ff', flow[y, x, 0], flow[y, x, 1]))


def compute_optical_flow(frame_dir, flow_dir, n_frames):
    """Compute Farneback optical flow between consecutive frames."""
    os.makedirs(flow_dir, exist_ok=True)

    prev_gray = None
    for i in range(1, n_frames + 1):
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        img = cv2.imread(frame_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            flo_path = os.path.join(flow_dir, f"frame_{i-1:04d}.flo")
            write_flo_file(flo_path, flow)

        prev_gray = gray

    return n_frames - 1  # number of flow files


def main():
    # Load test split
    with open(os.path.join(MSVD_DIR, "msvd_test.json")) as f:
        test_data = json.load(f)

    print(f"MSVD test set: {len(test_data)} videos")

    # Select N_VIDEOS diverse test videos (pick ones with different content)
    # Sort by video_id for reproducibility, pick evenly spaced
    test_data.sort(key=lambda x: x['video_id'])
    step = max(1, len(test_data) // N_VIDEOS)
    selected = [test_data[i * step] for i in range(N_VIDEOS) if i * step < len(test_data)]

    print(f"Selected {len(selected)} videos for evaluation")

    # Prepare output
    manifest = []

    for idx, item in enumerate(selected):
        vid = item['video_id']
        video_path = os.path.join(VIDEO_DIR, item['video'])

        if not os.path.exists(video_path):
            print(f"  [{idx+1}/{len(selected)}] SKIP {vid}: video not found")
            continue

        frame_dir = os.path.join(OUTPUT_DIR, "frames", vid)
        flow_dir = os.path.join(OUTPUT_DIR, "flow", vid)

        print(f"  [{idx+1}/{len(selected)}] {vid}...", end=" ", flush=True)

        # Extract frames
        n_frames = extract_frames(video_path, frame_dir)
        if n_frames < 2:
            print(f"SKIP (only {n_frames} frames)")
            continue

        # Compute optical flow
        n_flows = compute_optical_flow(frame_dir, flow_dir, n_frames)

        # Get reference captions
        captions = item.get('caption', [])

        manifest.append({
            'video_id': vid,
            'n_frames': n_frames,
            'n_flows': n_flows,
            'frame_dir': frame_dir,
            'flow_dir': flow_dir,
            'captions': captions,
        })

        print(f"{n_frames} frames, {n_flows} flows, {len(captions)} captions")

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPrepared {len(manifest)} videos. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
