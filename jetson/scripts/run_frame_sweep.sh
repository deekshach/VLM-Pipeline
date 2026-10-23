#!/bin/bash
# Characterization: TTFT and its stages vs the number of uniformly sampled frames, at baseline (stock kernels, no
# switches; ViT flash attention off, or on with VIT_FA=on: results in <out_dir>_fa). tegrastats runs beside each request (100 ms) for power and memory, and the
# board's over-current event counter (oc3, the hardware throttle) is recorded before and after it (<run>.oc3).
# The board guard pauses requests on every oc3 event, which would add its pauses to the timings: stop it for the
# sweep and restart it afterwards.
#
#   bash jetson/scripts/run_frame_sweep.sh [out_dir] [video=alley_1] [frame counts, comma-separated]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
VIT_FA="${VIT_FA:-off}"
OUT="${1:-$REPO/jetson/experiments/frame_sweep}"
[ "$VIT_FA" = on ] && [ $# -lt 1 ] && OUT="${OUT}_fa"
V="${2:-alley_1}"
NS="${3:-1,2,4,8,16,24,32,48}"
mkdir -p "$OUT"
OC=$(grep -l soctherm_oc /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/oc3_event_cnt
P="Describe the scene, motion patterns, and key visual elements in these video frames."
for N in ${NS//,/ }; do
    J="$OUT/${V}__f${N}.json"
    [ -f "$J" ] && continue
    echo "before $(cat $OC)" > "$OUT/${V}__f${N}.oc3"
    tegrastats --interval 100 --logfile "$OUT/${V}__f${N}.tegrastats" &
    # -c 32768 for every N: 48 frames are ~28k image tokens
    "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --video "$V" --baseline --base-frames $N -p "$P" -n 8 -ngl 99 -c 32768 \
        --vit-flash-attn $VIT_FA --output-json "$J" > "$OUT/${V}__f${N}.log" 2>&1 || echo "FAIL $V f$N"
    tegrastats --stop
    echo "after $(cat $OC)" >> "$OUT/${V}__f${N}.oc3"
    sleep 2
done
ls "$OUT"/*.json | wc -l
