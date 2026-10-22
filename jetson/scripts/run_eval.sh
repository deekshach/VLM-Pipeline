#!/bin/bash
# Main evaluation (MAXN, clocks pinned), in short bursts. One request per job, exactly as run_accuracy.sh runs it
# (-c 16384, greedy, --dump-dir for the accuracy tools), plus tegrastats and the oc3 over-current counter per run.
#
#   bash jetson/scripts/run_eval.sh <e1|e3|e4|e6|e8|e9|e10|e11|e12> [minutes=60]
#
#   e1  headline TTFT + accuracy: base, uni_arch, pw24 (the design), pwd24 (+ drop-clean) on all 123 clips of the
#       seven datasets (MSVD: the 45-clip set)
#   e3  attribution ladder, one switch added per step, on six Sintel clips, 3 repetitions
#   e4  accuracy vs TTFT: uniform 4 / 16 frames with the kernel switches (uni4, uni16), cache threshold 12 / 36
#       (pw12, pw36) on NExT-QA, Video-MME, Perception Test and Charades
#   e6  base and pw24 on NExT-QA and TACoS, run under 30W (the script then expects MODE_30W)
#   e8  qwen2fps on all 123 clips: Qwen2.5-VL's own video input -- every frame of its 2 fps sampling (qwen-vl-utils
#       smart_nframes, frames from video_redundancy.py, long side <= 644 px) fed as the reference does, two frames per
#       ViT pass and one video block with video rotary positions (llama-temporal-pair-vlm --mode video). Accuracy arm:
#       its JSON is that tool's (requests[0]: vit_ms, prefill_ms, text); LLM and ViT flash attention both on (-fa).
#       The LLM's visual input is dumped (<clip>/<clip>.video.embd.bin) for the spectral entropy of the tokens.
#   e9  the design and the design + drop-clean (pw24, pwd24: later frames prefill only their re-encoded windows'
#       tokens) on all 123 clips -- the E1 runs of those two configs, ahead of E1
#   e10 the clips whose frames tools/accuracy/prepare_qres.py regenerated at Qwen's per-frame resolution (its
#       affected.txt): base, ours_nocache, pw24, pwd24 where the pool changed, qwen2fps where its frames changed
#   e11 uni4 on all 123 clips: 4 uniform frames with the kernel switches and window alignment, the frame count the
#       motion selection sends -- against ours_nocache it separates the frame choice from the frame count
#   e12 ViT flash attention on (base_fa, pw24_fa: base and the design with --vit-flash-attn on) on six Sintel clips and
#       the 10 Video-MME clips; the flash-attention-off runs of base and pw24 on the same clips come from e1 / e9
#
# Frames: every arm reads the Qwen-resolution sets of prepare_qres.py (QRES): the pools under $QRES/pool/<dataset>
# and the 2 fps frames under $QRES/qwen2fps, each frame at qwen-vl-utils smart_resize of its source.
#
# Results: /home/nvidia/data/accuracy/runs_eval/<dataset>/<config>[_r<rep>]/<clip>.{json,log,tegrastats,oc3} (+ dump
# dir for repetition 1). Finished jobs are skipped, so the script resumes where it stopped; it exits after <minutes>.
# DATASETS="msvd charades": only those datasets (scripts/run_eval_paced.sh runs one dataset at a time).
# Exit status: 0 all jobs done, 3 jobs left (out of time), 2 stopped hot (STOP_TJ).
#
# Board safety (the board guard pauses every MAXN request on its oc3 events, so it is stopped here and restarted on
# exit): before each request the junction must be under COOL_TJ (55 C); after BURST_MIN (20) minutes of requests the
# script idles COOL_MIN (5) minutes; above STOP_TJ (80 C) after a request it stops.
set -u
EXP="${1:?e1|e3|e4|e6|e8|e9|e10|e11|e12}"
MINUTES="${2:-60}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/dmc955/model_config.sh"
BIN="$REPO/build/bin/llama-jetson-vlm"
QBIN="$REPO/build/bin/llama-temporal-pair-vlm"
QRES="${QRES:-/mnt/external_ssd/VLM-datasets/qres}"
QFRAMES=$QRES/qwen2fps
ACC=/home/nvidia/data/accuracy
OUTROOT=$ACC/runs_eval
TABLE="$REPO/jetson/gemm/table_orin.txt"
GUARD_LOG=$ACC/board_guard.log
COOL_TJ="${COOL_TJ:-55}"; STOP_TJ="${STOP_TJ:-80}"; BURST_MIN="${BURST_MIN:-20}"; COOL_MIN="${COOL_MIN:-5}"
OC=$(grep -l soctherm_oc /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/oc3_event_cnt
TZ=$(grep -l tj-thermal /sys/devices/virtual/thermal/thermal_zone*/type | head -1 | xargs dirname)/temp
tj() { echo $(( $(cat "$TZ") / 1000 )); }
log() { echo "[$(date +%T)] $*"; }

# the board must be in the expected mode with clocks pinned
WANT_MODE=MAXN; [ "$EXP" = e6 ] && WANT_MODE=MODE_30W
MODE=$(nvpmodel -q 2>/dev/null | head -1 | sed 's/.*: //')
F=/sys/class/devfreq/17000000.gpu
if [ "$MODE" != "$WANT_MODE" ] || [ "$(cat $F/cur_freq)" != "$(cat $F/max_freq)" ] || [ "$(cat $F/min_freq)" != "$(cat $F/max_freq)" ]; then
    echo "board is '$MODE' at $(cat $F/cur_freq) Hz (min $(cat $F/min_freq), max $(cat $F/max_freq)); want $WANT_MODE with clocks pinned:"
    echo "  sudo nvpmodel -m <id>; reboot if asked; sudo jetson_clocks"
    exit 1
fi

# stop the board guard for the run, restart it on exit
GPID=$(pgrep -x bash -a | awk '/board_guard.sh \/home/ {print $1}')
if [ -n "$GPID" ]; then
    kill "$GPID" && echo "[$(date +%T)] guard stopped by run_eval.sh $EXP (restarted on exit)" >> "$GUARD_LOG"
fi
restart_guard() {
    pgrep -x bash -a | grep -q "board_guard.sh /home" && return
    (cd "$REPO" && setsid nohup bash jetson/scripts/board_guard.sh "$GUARD_LOG" > /dev/null 2>&1 < /dev/null &)
    log "guard restarted"
}
trap restart_guard EXIT

ARCH="--vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE --act-fuse"
# dataset settings, as run_accuracy.sh
setds() {
    DS=$1
    if [ "$DS" = sintel ]; then
        ROOT=$QRES/pool/sintel; OFA="--ofa-levels 4"; N=96
        P="Describe the scene, motion patterns, and key visual elements in these video frames. Answer in English."
    elif [ "$DS" = msvd ]; then
        ROOT=$QRES/pool/msvd; OFA="--ofa-levels 3"; N=48
        P="Describe what is happening in this video in one sentence. Answer in English."
    else
        ROOT=$QRES/pool/$DS; OFA="--ofa-levels 3"; N=48
        grep -q '"refs"' $ROOT/items.json || N=8
        P=""
    fi
}
clips() {
    case "$1" in
        msvd) tr ',' ' ' < $ACC/msvd_clips_45.txt ;;
        sintel) ls /home/nvidia/data/MPI-Sintel/training/clean | tr '\n' ' ' ;;
        *) ls /mnt/external_ssd/VLM-datasets/eval/$1/training/clean | tr '\n' ' ' ;;
    esac
}
# config -> "<frames> <flags>"
cfg() {
    local L="--live-flow $OFA"
    case "$1" in
        base)      echo "8 --baseline" ;;
        uni_arch)  echo "8 --baseline --window-align $ARCH" ;;
        uni4)      echo "4 --baseline --window-align $ARCH" ;;
        uni16)     echo "16 --baseline --window-align $ARCH" ;;
        ours_nocache) echo "8 --live-flow $OFA --window-align $ARCH --anchor-first" ;;
        # E3 ladder: each step adds to the previous one
        l1_sel)      echo "8 $L" ;;
        l2_anchor)   echo "8 $L --anchor-first" ;;
        l3_align)    echo "8 $L --anchor-first --window-align" ;;
        l4_winbatch) echo "8 $L --anchor-first --window-align --vit-window-batch" ;;
        l5_f32bias)  echo "8 $L --anchor-first --window-align --vit-window-batch --gemm-f32-out --gemm-bias-fuse" ;;
        l6_table)    echo "8 $L --anchor-first --window-align --vit-window-batch --gemm-f32-out --gemm-bias-fuse --gemm-table $TABLE" ;;
        l7_actfuse)  echo "8 $L --anchor-first --window-align $ARCH" ;;
        # ViT flash attention on (the run's --vit-flash-attn off comes first; the last one given wins)
        base_fa)   echo "8 --baseline --vit-flash-attn on" ;;
        pw24_fa)   echo "8 $L --window-align --wc-per-window --wc-threshold 24 $ARCH --anchor-first --vit-flash-attn on" ;;
        pwd*)      echo "8 $L --window-align --wc-per-window --wc-drop-clean --wc-threshold ${1#pwd} $ARCH --anchor-first" ;;
        pw*)       echo "8 $L --window-align --wc-per-window --wc-threshold ${1#pw} $ARCH --anchor-first" ;;
    esac
}
# job list: "<dataset> <config> <rep> <clip>", clip-major so a partial run leaves complete clips
jobs() {
    case "$EXP" in
        e1) for D in sintel msvd charades nextqa videomme perception tacos; do
                for V in $(clips $D); do for C in base uni_arch pw24 pwd24; do echo "$D $C 1 $V"; done; done; done ;;
        e3) for R in 1 2 3; do for V in alley_1 sleeping_1 bandage_1 temple_2 ambush_5 cave_2; do
                for C in base l1_sel l2_anchor l3_align l4_winbatch l5_f32bias l6_table l7_actfuse pw24 pwd24; do
                    echo "sintel $C $R $V"; done; done; done ;;
        e4) for D in nextqa videomme perception charades; do
                for V in $(clips $D); do for C in uni4 uni16 pw12 pw36; do echo "$D $C 1 $V"; done; done; done ;;
        e6) for D in nextqa tacos; do
                for V in $(clips $D); do for C in base pw24; do echo "$D $C 1 $V"; done; done; done ;;
        e8) for D in sintel msvd charades nextqa videomme perception tacos; do
                for V in $(clips $D); do echo "$D qwen2fps 1 $V"; done; done ;;
        e9) for D in sintel msvd charades nextqa videomme perception tacos; do
                for V in $(clips $D); do for C in pw24 pwd24; do echo "$D $C 1 $V"; done; done; done ;;
        e12) for V in alley_1 sleeping_1 bandage_1 temple_2 ambush_5 cave_2; do
                for C in base_fa pw24_fa; do echo "sintel $C 1 $V"; done; done
             for V in $(clips videomme); do for C in base_fa pw24_fa; do echo "videomme $C 1 $V"; done; done ;;
        e11) for D in sintel msvd charades nextqa videomme perception tacos; do
                for V in $(clips $D); do echo "$D uni4 1 $V"; done; done ;;
        e10) for D in sintel msvd charades nextqa videomme perception tacos; do
                awk -v d=$D '$1 == d' $QRES/affected.txt | while read -r _ V WHAT; do
                    if [ "$WHAT" = pool ]; then for C in base ours_nocache pw24 pwd24; do echo "$D $C 1 $V"; done
                    else echo "$D qwen2fps 1 $V"; fi
                done; done ;;
    esac
}

T_END=$(( $(date +%s) + MINUTES * 60 )); T_BURST=$(date +%s); DONE=0; TODO=0; HOT=0
TAG=""; [ "$EXP" = e6 ] && TAG="_30w"
while read -r D C R V <&3; do
    [ -n "${DATASETS:-}" ] && [[ " $DATASETS " != *" $D "* ]] && continue
    setds "$D"
    SUF=""; [ "$R" != 1 ] && SUF="_r$R"
    OUT=$OUTROOT/$D/$C$TAG$SUF; mkdir -p "$OUT"
    [ -f "$OUT/$V.json" ] && continue
    TODO=$((TODO + 1))
    [ "$(date +%s)" -ge "$T_END" ] && continue   # out of time: just count what is left
    if [ $(( $(date +%s) - T_BURST )) -ge $(( BURST_MIN * 60 )) ]; then
        log "burst done: idling $COOL_MIN min (tj $(tj) C)"; sleep $(( COOL_MIN * 60 )); T_BURST=$(date +%s)
    fi
    while [ "$(tj)" -ge "$COOL_TJ" ]; do sleep 10; done
    [ -f $ROOT/items.json ] && P="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["prompt"])' $ROOT/items.json $V)"
    echo "before $(cat $OC) tj $(tj)" > "$OUT/$V.oc3"
    tegrastats --interval 100 --logfile "$OUT/$V.tegrastats" &
    if [ "$C" = qwen2fps ]; then
        # all frames of the clip's 2 fps sampling: up to 210 frames, 105 pairs of up to ~770 tokens at Qwen's resolution
        # (Video-MME: ~76k visual tokens), so -c 81920
        mkdir -p "$OUT/$V"
        "$QBIN" -m "$MODEL" --mmproj "$MMPROJ" -c 81920 --no-mmap --sintel-dir $QFRAMES --frames-subdir $D --videos $V \
            --base-frames 100000 --mode video --video-fps 2 --warmup-requests 0 -p "$P" -n $N --temp 0 -ngl 99 \
            --dump-dir "$OUT/$V" --output-json "$OUT/$V.json" > "$OUT/$V.log" 2>&1 || log "FAIL $D $C $R $V"
        rm -f "$OUT/$V/$V.video.frames.bin"   # the input frames, up to 150 MB; they are on the SSD already
        TT='r = j["requests"][0]; print(round(r["vit_ms"] + r["prefill_ms"]))'
    else
        read -r NF FLAGS <<< "$(cfg $C)"
        DUMP=""; [ "$R" = 1 ] && DUMP="--dump-dir $OUT/$V"
        # shellcheck disable=SC2086
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" -c 16384 --no-mmap --sintel-dir $ROOT --video $V --base-frames $NF \
            -p "$P" -n $N --temp 0 -ngl 99 --vit-flash-attn off $FLAGS $DUMP --decode-threads 3 \
            --output-json "$OUT/$V.json" > "$OUT/$V.log" 2>&1 || log "FAIL $D $C $R $V"
        TT='print(j["timing"]["ttft_ms"])'
    fi
    tegrastats --stop
    echo "after $(cat $OC) tj $(tj)" >> "$OUT/$V.oc3"
    DONE=$((DONE + 1)); TODO=$((TODO - 1))
    log "$D $C r$R $V  ttft $(python3 -c "import json,sys; j = json.load(open(sys.argv[1])); $TT" "$OUT/$V.json" 2>/dev/null) ms  tj $(tj) C"
    if [ "$(tj)" -ge "$STOP_TJ" ]; then log "junction $(tj) C >= $STOP_TJ C: stopping"; HOT=1; break; fi
    sleep 5
done 3< <(jobs)
log "$EXP${DATASETS:+ ($DATASETS)}: $DONE run now, $TODO left"
[ "$HOT" = 1 ] && exit 2
[ "$TODO" -gt 0 ] && exit 3
exit 0
