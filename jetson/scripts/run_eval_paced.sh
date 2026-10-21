#!/bin/bash
# The main evaluation one dataset at a time, with the board at rest in between (over-current throttling at MAXN
# builds up over long runs). For each dataset, each experiment in turn (scripts/run_eval.sh with DATASETS=<dataset>,
# its own 20-minute bursts inside), then a pause: the board guard back on, at least PAUSE_MIN (10) minutes and until
# the junction is under REST_TJ (50 C, given up after REST_MAX 30 minutes). The oc3 over-current events of every
# dataset go to the log.
#
#   bash jetson/scripts/run_eval_paced.sh "<experiments, e.g. e8 e9>" [datasets...]
#
# Datasets default to all seven, in the order of run_eval.sh. A dataset that runs out of time (DS_MINUTES, 240) is
# run again after a pause; a run that stops hot (STOP_TJ) ends everything. Finished jobs are skipped, so this resumes.
set -u
EXPS="${1:?experiments, e.g. \"e8 e9\"}"; shift
DSS="${*:-sintel msvd charades nextqa videomme perception tacos}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAUSE_MIN="${PAUSE_MIN:-10}"; REST_TJ="${REST_TJ:-50}"; REST_MAX="${REST_MAX:-30}"; DS_MINUTES="${DS_MINUTES:-240}"
OC=$(grep -l soctherm_oc /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/oc3_event_cnt
TZ=$(grep -l tj-thermal /sys/devices/virtual/thermal/thermal_zone*/type | head -1 | xargs dirname)/temp
tj() { echo $(( $(cat "$TZ") / 1000 )); }
log() { echo "[$(date +%T)] $*"; }

rest() {
    log "pause: $PAUSE_MIN min, then junction < $REST_TJ C (now $(tj) C)"
    sleep $(( PAUSE_MIN * 60 ))
    local t0; t0=$(date +%s)
    while [ "$(tj)" -ge "$REST_TJ" ] && [ $(( $(date +%s) - t0 )) -lt $(( REST_MAX * 60 )) ]; do sleep 30; done
    log "rested: junction $(tj) C"
}

for D in $DSS; do
    for E in $EXPS; do
        while :; do
            oc0=$(cat "$OC"); t0=$(date +%s)
            log "== $E $D (junction $(tj) C)"
            DATASETS=$D bash "$HERE/run_eval.sh" "$E" "$DS_MINUTES"
            rc=$?
            log "== $E $D: exit $rc, $(( ($(date +%s) - t0) / 60 )) min, oc3 events $(( $(cat "$OC") - oc0 ))"
            [ $rc = 2 ] && { log "stopped hot: ending"; exit 2; }
            [ $rc = 1 ] && { log "run_eval.sh refused (power mode / clocks): ending"; exit 1; }
            [ $(( $(date +%s) - t0 )) -ge 60 ] && rest   # nothing was left to run: no pause
            [ $rc = 0 ] && break
        done
    done
done
log "all done: $EXPS on $DSS"
