#!/bin/bash
# ViT embeddings of the Qwen 2 fps frames (tools/video_redundancy.py with FRAMES_DIR) for the motivation study:
# a contiguous block of at most 40 sampled frames from the middle of each clip, encoded whole (jetson-embed-frames,
# native). MSVD and Sintel are skipped: the accuracy study's dense references already cover them.
#
#   bash jetson/scripts/embed_qwen2fps.sh [root=/mnt/external_ssd/VLM-datasets/qwen2fps]
set -u
ROOT="${1:-/mnt/external_ssd/VLM-datasets/qwen2fps}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
N=40
for DS in tacos charades perception nextqa videomme; do
    for C in "$ROOT/frames/$DS"/*/; do
        V=$(basename "$C"); OUT=$ROOT/embed/$DS/$V
        [ -f $OUT/meta.json ] && continue
        SUB=$ROOT/block/$DS/$V; rm -rf $SUB; mkdir -p $SUB
        mapfile -t F < <(ls "$C" | sort)
        S=$(( ${#F[@]} > N ? (${#F[@]} - N) / 2 : 0 ))
        for f in "${F[@]:$S:$N}"; do ln -s "$C$f" $SUB/$f; done
        "$REPO/build/bin/jetson-embed-frames" "$MODEL" "$MMPROJ" $SUB $OUT native > /dev/null 2>&1 || echo "FAIL $DS $V"
        echo "$DS $V $(ls $SUB | wc -l) frames from $S"
    done
done
