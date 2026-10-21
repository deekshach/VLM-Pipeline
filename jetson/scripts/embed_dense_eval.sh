#!/bin/bash
# Dense ViT references for the five evaluation datasets, as run_accuracy.sh builds them for Sintel and MSVD: every
# frame of the clip's pool (training/clean, at most 50) encoded whole, native and window-aligned
# (jetson-embed-frames). The reference of tools/accuracy/vision_metrics.py: coverage of the whole clip, and the
# window cache's tokens against the same frame encoded whole. ViT only (no LLM), so the board guard can stay on.
#
#   bash jetson/scripts/embed_dense_eval.sh
#
# Results: /home/nvidia/data/accuracy/dense/<dataset>/<clip>/{native,aligned}/ (meta.json + frame_NNNN.f32);
# Perception Test questions on the same video share one reference (symlink). Finished clips are skipped.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
ACC=/home/nvidia/data/accuracy
EVAL=/mnt/external_ssd/VLM-datasets/eval
for DS in charades nextqa videomme perception tacos; do
    for C in "$EVAL/$DS/training/clean"/*; do
        V=$(basename "$C"); OUT=$ACC/dense/$DS/$V
        if [ -L "$C" ]; then   # a question on a video already encoded under another name
            mkdir -p "$ACC/dense/$DS"; [ -e "$OUT" ] || ln -s "$(readlink "$C")" "$OUT"
            continue
        fi
        for MODE in native aligned; do
            [ -f "$OUT/$MODE/meta.json" ] && continue
            "$REPO/build/bin/jetson-embed-frames" "$MODEL" "$MMPROJ" "$C" "$OUT/$MODE" $MODE > /dev/null 2>&1 \
                || echo "FAIL $DS $V $MODE"
        done
        echo "$DS $V $(ls "$C" | wc -l) frames"
    done
done
