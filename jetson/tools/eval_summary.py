#!/usr/bin/env python3
"""Tables for the main evaluation (scripts/run_eval.sh), from /home/nvidia/data/accuracy/runs_eval.

    python3 eval_summary.py <e1|e3|e4|e5|e6> [out.md]

e1/e6  per dataset and config: clips, mean and median TTFT, speed-up over base (mean of per-clip ratios, clips run in
       both), frames and the tokens the ViT encoded (anchor + later frames; with the cache, only re-encoded windows), energy to first token, oc3 events per run; multiple-choice accuracy where the
       items have an answer letter. Caption metrics (CIDEr, METEOR, SBERT): accuracy/newds_compare.py and
       accuracy/analyze_accuracy.py with ACC_RUNS=/home/nvidia/data/accuracy/runs_eval.
       qwen2fps (Qwen's own 2 fps video input, e8) and uni4 (e11) are listed beside the E1 configs; its TTFT is
       ViT + prefill + load + tokenize (total minus decode).
e5     overheads and where the time goes, per dataset and config: mean optical flow (live OFA, CPU decode overlapped),
       ViT, image and text prefill, load + tokenize, the anchor frame's work hidden behind the flow (anchor-first:
       anchor total minus the wait for it), the window cache's change test, and frames selected against the clip's
       motion complexity.
e3     the attribution ladder on six Sintel clips: per step the median over the 3 repetitions of each clip, averaged
       over clips, and the saving of each step.
e4     accuracy vs TTFT points for the multiple-choice sets: uniform 4 / 8 / 16 frames with the kernel switches and the
       design at cache threshold 12 / 24 / 36 (uni_arch and pw24 come from e1).
Energy = module power (GPU/SoC + CPU/CV + 5 V rails, mean over samples with the GPU >= 50% busy) x TTFT.
"""
import glob, json, os, re, statistics as st, sys

R = "/home/nvidia/data/accuracy/runs_eval"
EVAL = "/mnt/external_ssd/VLM-datasets/eval"
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def run(ds, cfg, clip):
    p = f"{R}/{ds}/{cfg}/{clip}"
    if not os.path.exists(p + ".json"):
        return None
    j = json.load(open(p + ".json"))
    if "requests" in j:   # llama-temporal-pair-vlm (qwen2fps)
        if not j["requests"]:
            return None
        q = j["requests"][0]
        t = {"ttft_ms": q["total_ms"] - q["decode_ms"], "breakdown": {}}
        j = {"inference": {"generated_text": q["text"]}, "selected_frames": [None] * q["n_frames"]}
        t["breakdown"]["rest"] = {"tokens_encoded": q["n_visual_tokens"]}
    else:
        t = j["timing"]
    w = []
    if os.path.exists(p + ".tegrastats"):
        for l in open(p + ".tegrastats"):
            g = re.search(r"GR3D_FREQ (\d+)%", l)
            if g and int(g[1]) >= 50:
                w.append(sum(int(re.search(rf"{r} (\d+)mW", l)[1]) for r in RAILS) / 1000)
    oc = None
    if os.path.exists(p + ".oc3"):
        v = [int(l.split()[1]) for l in open(p + ".oc3")]
        oc = v[1] - v[0] if len(v) == 2 else None
    text = j.get("inference", {}).get("generated_text", "").strip()
    b = t["breakdown"]
    tokens = b.get("rest", {}).get("tokens_encoded", 0) + b.get("anchor", {}).get("tokens_encoded", 0)
    return dict(ttft=t["ttft_ms"], frames=len(j.get("selected_frames", [])), tokens=tokens,
                energy=(st.mean(w) * t["ttft_ms"] / 1000) if w else None, oc3=oc, text=text, j=j, t=t)


def items(ds):
    f = f"{EVAL}/{ds}/items.json"
    return json.load(open(f)) if os.path.exists(f) else {}


def letter(text):
    m = re.match(r"\s*\(?([A-E])\b", text or "")
    return m[1] if m else None


def clips(ds, cfg):
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{R}/{ds}/{cfg}/*.json"))


def fmt(x, n=0):
    return "" if x is None else f"{x:,.{n}f}"


def e1(tag=""):
    out = ["| dataset | config | clips | mean TTFT ms | median TTFT ms | speed-up vs base | speed-up vs qwen2fps | frames | tokens through the ViT | energy J | oc3/run | MC accuracy |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in sorted(os.listdir(R)):
        base = {c: run(ds, "base" + tag, c) for c in clips(ds, "base" + tag)}
        qw = {c: run(ds, "qwen2fps", c) for c in clips(ds, "qwen2fps")} if not tag else {}
        it = items(ds)
        for cfg in ("qwen2fps", "base", "uni_arch", "uni4", "pw24", "pwd24") if not tag else ("base", "pw24"):
            cc = clips(ds, cfg + tag)
            rows = {c: run(ds, cfg + tag, c) for c in cc}
            rows = {c: r for c, r in rows.items() if r}
            if not rows:
                continue
            tt = [r["ttft"] for r in rows.values()]
            sp = [base[c]["ttft"] / r["ttft"] for c, r in rows.items() if c in base and base[c]]
            sq = [qw[c]["ttft"] / r["ttft"] for c, r in rows.items() if c in qw and qw[c]]
            en = [r["energy"] for r in rows.values() if r["energy"]]
            oc = [r["oc3"] for r in rows.values() if r["oc3"] is not None]
            acc = [float(letter(r["text"]) == it[c]["answer"]) for c, r in rows.items() if c in it and "answer" in it[c]]
            out.append(f"| {ds} | {cfg} | {len(rows)} | {fmt(st.mean(tt))} | {fmt(st.median(tt))} | "
                       f"{fmt(st.mean(sp), 2) + 'x' if sp else ''} | {fmt(st.mean(sq), 2) + 'x' if sq else ''} | {fmt(st.mean(r['frames'] for r in rows.values()), 1)} | "
                       f"{fmt(st.mean(r['tokens'] or 0 for r in rows.values()))} | {fmt(st.mean(en)) if en else ''} | "
                       f"{fmt(st.mean(oc), 1) if oc else ''} | {fmt(100 * st.mean(acc), 0) + '%' if acc else ''} |")
    return out


def e5():
    out = ["| dataset | config | clips | TTFT ms | flow ms | ViT ms | image prefill ms | text prefill ms | load + tokenize ms | "
           "anchor hidden ms | cache test ms | frames | motion complexity |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in sorted(os.listdir(R)):
        for cfg in ("base", "uni_arch", "uni4", "ours_nocache", "pw24", "pwd24"):
            rows = [r for r in (run(ds, cfg, c) for c in clips(ds, cfg)) if r]
            if not rows:
                continue
            def m(f):
                return st.mean(f(r["t"]["breakdown"]) for r in rows)
            stage = lambda b, k: b.get("rest", {}).get(k, 0) + b.get("anchor", {}).get(k, 0)
            hidden = m(lambda b: b["anchor"]["total_ms"] - b["wait_for_anchor_ms"] if "anchor" in b else 0)
            out.append(f"| {ds} | {cfg} | {len(rows)} | {fmt(st.mean(r['ttft'] for r in rows))} | "
                       f"{fmt(m(lambda b: b.get('analysis_ms', 0)))} | {fmt(m(lambda b: stage(b, 'vit_ms')))} | "
                       f"{fmt(m(lambda b: stage(b, 'prefill_img_ms')))} | {fmt(m(lambda b: stage(b, 'prefill_txt_ms')))} | "
                       f"{fmt(m(lambda b: b.get('rest_image_load_ms', 0) + b.get('rest_tokenize_ms', 0) + b.get('anchor', {}).get('load_ms', 0)))} | "
                       f"{fmt(hidden) if hidden else ''} | {fmt(m(lambda b: stage(b, 'wc_diff_ms') + stage(b, 'wc_prep_ms')), 1)} | "
                       f"{fmt(st.mean(r['frames'] for r in rows), 1)} | "
                       f"{fmt(st.mean(r['j'].get('global_complexity') or 0 for r in rows), 3)} |")
    return out


def e3():
    steps = ["base", "l1_sel", "l2_anchor", "l3_align", "l4_winbatch", "l5_f32bias", "l6_table", "l7_actfuse", "pw24", "pwd24"]
    vids = ["alley_1", "sleeping_1", "bandage_1", "temple_2", "ambush_5", "cave_2"]
    out = ["| step | clips | TTFT ms (mean over clips of the median of 3) | saving vs previous step | vs base |", "|---|---|---|---|---|"]
    prev = first = None
    for s in steps:
        per = []
        for v in vids:
            reps = [run("sintel", s + suf, v) for suf in ("", "_r2", "_r3")]
            reps = [r["ttft"] for r in reps if r]
            if reps:
                per.append(st.median(reps))
        if not per:
            continue
        m = st.mean(per)
        first = first or m
        out.append(f"| {s} | {len(per)} | {fmt(m)} | {fmt(prev - m) + ' ms' if prev else ''} | {fmt(first / m, 2)}x |")
        prev = m
    return out


def e4():
    out = ["| dataset | config | clips | mean TTFT ms | frames | MC accuracy |", "|---|---|---|---|---|---|"]
    for ds in ("nextqa", "videomme", "perception"):
        it = items(ds)
        for cfg in ("uni4", "uni_arch", "uni16", "pw12", "pw24", "pw36"):
            rows = {c: run(ds, cfg, c) for c in clips(ds, cfg)}
            if not rows:
                continue
            acc = [float(letter(r["text"]) == it[c]["answer"]) for c, r in rows.items() if c in it]
            out.append(f"| {ds} | {cfg} | {len(rows)} | {fmt(st.mean(r['ttft'] for r in rows.values()))} | "
                       f"{fmt(st.mean(r['frames'] for r in rows.values()), 1)} | {fmt(100 * st.mean(acc), 0) + '%' if acc else ''} |")
    return out


if __name__ == "__main__":
    exp = sys.argv[1]
    lines = {"e1": e1, "e3": e3, "e4": e4, "e5": e5, "e6": lambda: e1("_30w")}[exp]()
    text = "\n".join(lines) + "\n"
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text)
