#!/bin/bash
# Board guard for long unattended runs on the Orin dev kit: pauses the GPU workloads when the board over-heats or
# trips its over-current throttle, and resumes them after it has recovered. No sudo needed.
#
#   nohup bash jetson/scripts/board_guard.sh [log] &      # stop with: kill <pid>
#
# Every 2 s: junction temperature (thermal zone tj-thermal) and the over-current event counter oc3 of soctherm_oc
# (hwmon). On a new over-current event or tj > TJ_HOT (85 C) the pipeline processes (llama-jetson-vlm,
# jetson-embed-frames, jetson-*-probe) get SIGSTOP (kernels in flight finish, the host side stops issuing work);
# they get SIGCONT once tj < TJ_COOL (70 C) and no event for QUIET (30) s. Every pause is logged.
set -u
LOG="${1:-/home/nvidia/data/accuracy/board_guard.log}"
TJ_HOT="${TJ_HOT:-85}"; TJ_COOL="${TJ_COOL:-70}"; QUIET="${QUIET:-30}"
OC=$(grep -l soctherm_oc /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/oc3_event_cnt
TZ=$(grep -l tj-thermal /sys/devices/virtual/thermal/thermal_zone*/type | head -1 | xargs dirname)/temp
NAMES="llama-jetson-vl jetson-embed-fr jetson-prefill- jetson-vit-prob jetson-overlap-"   # comm (15 chars)
pids() { for n in $NAMES; do pgrep -x "$n"; done; }
log() { echo "[$(date +%T)] $*" >> "$LOG"; }
read_tj() { local t; t=$(cat "$TZ" 2>/dev/null) && echo $((t / 1000)) || echo 0; }
last=$(cat "$OC"); paused=0; t_event=0
log "guard started: tj hot ${TJ_HOT} C, cool ${TJ_COOL} C, quiet ${QUIET} s, oc3 count $last"
while true; do
    now=$(date +%s); oc=$(cat "$OC"); tj=$(read_tj)
    if [ "$oc" -gt "$last" ]; then t_event=$now; log "over-current: oc3 +$((oc - last)) (tj ${tj} C)"; fi
    last=$oc
    if [ $paused = 0 ] && { [ $t_event = $now ] || [ "$tj" -gt "$TJ_HOT" ]; }; then
        P=$(pids); if [ -n "$P" ]; then kill -STOP $P 2>/dev/null; paused=1; log "paused: $(echo $P | tr '\n' ' ') (tj ${tj} C)"; fi
    elif [ $paused = 1 ] && [ "$tj" -lt "$TJ_COOL" ] && [ $((now - t_event)) -ge "$QUIET" ]; then
        P=$(pids); [ -n "$P" ] && kill -CONT $P 2>/dev/null; paused=0; log "resumed (tj ${tj} C)"
    fi
    sleep 2
done
