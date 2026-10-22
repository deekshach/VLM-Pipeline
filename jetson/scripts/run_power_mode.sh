#!/bin/bash
# Characterization under one nvpmodel power mode: what the mode sets, its measured ceilings, TTFT at 8 and 32 frames
# (baseline, with power and over-current events), an nsys trace at 8 frames, and optionally the full design against
# the baseline on six Sintel clips. Switch the mode first (sudo nvpmodel -m <id>, reboot if asked, sudo jetson_clocks).
#
#   bash jetson/scripts/run_power_mode.sh <tag, e.g. 50w> [--design]
#
# The board guard keeps running: a request it pauses has a wall-time gap in its timings, so any request with a new
# "paused" line in the guard log is moved aside (<name>.paused.*) and run again, at most 3 times.
# Results: jetson/experiments/power_modes/<tag>/ (nsys traces in experiments/profiles/nsys/pm_<tag>_f8.*).
# VIT_FA=on: ViT flash attention on for every request (the design also gains --vit-rope-f16 --gemm-residual-fuse);
# results in power_modes_fa/<tag>/ and traces pm_fa_<tag>_f8.*. Default off, as the results in power_modes/.
set -u
TAG="${1:?mode tag, e.g. 50w}"
DESIGN="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
VIT_FA="${VIT_FA:-off}"
OUT="$REPO/jetson/experiments/power_modes/$TAG"
NSN="pm_${TAG}_f8"
if [ "$VIT_FA" = on ]; then OUT="$REPO/jetson/experiments/power_modes_fa/$TAG"; NSN="pm_fa_${TAG}_f8"; fi
GUARD=/home/nvidia/data/accuracy/board_guard.log
OC=$(grep -l soctherm_oc /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/oc3_event_cnt
mkdir -p "$OUT/design"
# the ceilings do not depend on ViT flash attention: reuse the mode's earlier measurement
if [ "$VIT_FA" = on ]; then
    for f in peaks.json l2_bandwidth.csv; do
        [ -f "$REPO/jetson/experiments/power_modes/$TAG/$f" ] && [ ! -f "$OUT/$f" ] && cp "$REPO/jetson/experiments/power_modes/$TAG/$f" "$OUT/$f"
    done
fi
pgrep -f board_guard.sh > /dev/null || echo "WARNING: board guard not running"

# what the mode sets
{
    nvpmodel -q 2>/dev/null
    echo "gpu_max_freq $(cat /sys/class/devfreq/17000000.gpu/max_freq) cur $(cat /sys/class/devfreq/17000000.gpu/cur_freq)"
    echo "tpc_pg_mask $(cat /sys/devices/platform/gpu.0/tpc_pg_mask 2>/dev/null || cat /sys/devices/gpu.0/tpc_pg_mask)"
    echo "emc_cap $(cat /sys/kernel/nvpmodel_clk_cap/emc)"
    echo "cpu_online $(cat /sys/devices/system/cpu/online)"
} > "$OUT/mode.txt"
cat "$OUT/mode.txt"

# ceilings (short GPU jobs, no retry: a pause during a best-of-N benchmark only drops that repetition)
[ -f "$OUT/peaks.json" ] || "$REPO/build/bin/dmc-bench-peaks" > "$OUT/peaks.json" 2> "$OUT/peaks.log"
[ -f "$OUT/l2_bandwidth.csv" ] || "$REPO/build/bin/l2-bandwidth" > "$OUT/l2_bandwidth.csv" 2> "$OUT/l2_bandwidth.log"

# guarded <name> <cmd...>: run, and retry while the guard paused it
guarded() {
    local name="$1"; shift
    for try in 1 2 3; do
        local before; before=$(grep -c paused "$GUARD" 2>/dev/null || echo 0)
        echo "before $(cat $OC)" > "$name.oc3"
        "$@"
        echo "after $(cat $OC)" >> "$name.oc3"
        [ "$(grep -c paused "$GUARD" 2>/dev/null || echo 0)" = "$before" ] && return 0
        echo "paused by the guard: $name (try $try)"
        for f in "$name".*; do mv "$f" "${f%.*}.paused$try.${f##*.}"; done
        sleep 30
    done
    echo "FAIL $name: paused 3 times"
}

P="Describe the scene, motion patterns, and key visual elements in these video frames."
# TTFT at 8 and 32 frames, baseline (as scripts/run_frame_sweep.sh)
mkdir -p "$OUT/sweep"
for N in 8 32; do
    B="$OUT/sweep/alley_1__f$N"
    [ -f "$B.json" ] && continue
    run_sweep() {
        tegrastats --interval 100 --logfile "$B.tegrastats" &
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --video alley_1 --baseline --base-frames $N -p "$P" -n 8 -ngl 99 -c 32768 \
            --vit-flash-attn $VIT_FA --output-json "$B.json" > "$B.log" 2>&1
        tegrastats --stop
    }
    guarded "$B" run_sweep
    sleep 2
done
python3 "$REPO/jetson/tools/frame_sweep_summary.py" "$OUT/sweep"

# nsys trace at 8 frames: kernel classes and GPU idle
NS="$REPO/jetson/experiments/profiles/nsys/$NSN"
if [ ! -f "$NS.json" ]; then
    guarded "$NS" bash "$HERE/profile_nsys.sh" "$NSN" alley_1 --baseline --vit-flash-attn $VIT_FA
    python3 "$REPO/jetson/tools/nsys_idle.py" "$NS.sqlite" > "$OUT/nsys_idle.md"
    python3 "$REPO/jetson/tools/nsys_stage_kernels.py" "$NS.sqlite" > "$OUT/nsys_stage_kernels.md"
fi

# full design vs baseline (the report's appendix command; Sintel: 4 OFA pyramid levels)
if [ "$DESIGN" = --design ]; then
    TABLE="$REPO/jetson/gemm/table_orin.txt"
    FULL="--live-flow --ofa-levels 4 --anchor-first --window-align --vit-window-batch --gemm-f32-out --gemm-bias-fuse \
--gemm-table $TABLE --act-fuse --wc-per-window --wc-threshold 24"
    [ "$VIT_FA" = on ] && FULL="$FULL --vit-rope-f16 --gemm-residual-fuse"
    for V in alley_1 sleeping_1 bandage_1 temple_2 ambush_5 cave_2; do
        for C in "uniform:--baseline" "design:$FULL"; do
            NAME="${C%%:*}"; FLAGS="${C#*:}"; B="$OUT/design/${V}__$NAME"
            [ -f "$B.json" ] && continue
            # shellcheck disable=SC2086
            guarded "$B" sh -c "\"$BIN\" -m \"$MODEL\" --mmproj \"$MMPROJ\" --video $V --base-frames 8 -p \"$P\" -n 8 -ngl 99 \
                --vit-flash-attn $VIT_FA $FLAGS --output-json \"$B.json\" > \"$B.log\" 2>&1"
        done
    done
    python3 - "$OUT/design" <<'EOF'
import glob, json, sys
d = sys.argv[1]; rows = []
for j in sorted(glob.glob(f"{d}/*__uniform.json")):
    v = j.split("/")[-1].split("__")[0]
    u = json.load(open(j))["timing"]["ttft_ms"]
    try:
        s = json.load(open(f"{d}/{v}__design.json"))["timing"]["ttft_ms"]
    except FileNotFoundError:
        continue
    rows.append((v, u, s))
lines = ["| clip | uniform (baseline) ms | full design ms | speed-up |", "|---|---|---|---|"]
lines += [f"| {v} | {u:.0f} | {s:.0f} | {u / s:.2f}x |" for v, u, s in rows]
if rows:
    U, S = sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)
    lines.append(f"| mean | {U:.0f} | {S:.0f} | {U / S:.2f}x |")
open(f"{d}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
fi
echo "done: $OUT"
