#!/usr/bin/env python3
"""Power and energy of the main evaluation (scripts/run_eval.sh), every arm against qwen2fps, from the tegrastats trace
each run logs (100 ms; rails VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0, GPU load, junction temperature). Reads the runs
through eval_summary.py (EVAL_RUNS=/home/nvidia/data/accuracy/runs_eval_fa: ViT flash attention on).

    python3 power_analysis.py [out prefix=/home/nvidia/data/accuracy/power] [idle W]
        -> <prefix>.csv (per clip and arm), <prefix>.md (tables), <prefix>_trace_<dataset>_<clip>.csv (example traces)

The request window inside a trace: it ends at the last sample with the GPU >= 50% busy (the last decode step) and
lasts TTFT + decode (llama-jetson-vlm: ttft_ms + generation_ms; qwen2fps: total_ms, load and tokenize included), at the
trace's own sample period (its wall-clock span over its sample count). Energy = the sum of the three rails x period
over the window: to the first token, and to the whole answer. Energy above idle subtracts the board's idle power
(second argument, W; default 0 = not subtracted) over the same time. Peak power is the highest 3-rail sample in the
window; the junction rise is the window's highest junction temperature minus the first. The oc3 over-current events
come from the run's .oc3 file.

Energy savings by tier, qwen2fps -> pwd24 (as latency_per_clip.py, in J to the first token):
  input qwen2fps -> base, kernels -> uni_arch, frames -> uni4, adaptive -> ours_nocache, cache -> pw24, drop -> pwd24.
"""
import csv, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary as E

ARMS = ["qwen2fps", "base", "uni_arch", "uni4", "ours_nocache", "pw24", "pwd24"]
TIERS = [("input", "base"), ("kernels", "uni_arch"), ("frames", "uni4"), ("adaptive", "ours_nocache"), ("cache", "pw24"),
         ("drop", "pwd24")]
DATASETS = [("sintel", "Sintel"), ("msvd", "MSVD"), ("charades", "Charades-STA"), ("tacos", "TACoS"),
            ("nextqa", "NExT-QA"), ("videomme", "Video-MME"), ("perception", "Perception Test")]
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
EXAMPLES = [("videomme", "HmnKbBmYii8"), ("perception", "video_874_q1"), ("sintel", "alley_1")]


def trace(p):
    """Samples of a tegrastats log: (seconds since the first, {rail: W}, GPU %, junction C)."""
    L = [l for l in open(p) if "GR3D_FREQ" in l]
    secs = lambda l: sum(int(x) * m for x, m in zip(l[11:19].split(":"), (3600, 60, 1)))
    span = secs(L[-1]) - secs(L[0])
    dt = span / (len(L) - 1) if len(L) > 1 and span > 0 else 0.1
    out = []
    for i, l in enumerate(L):
        w = {r: int(re.search(rf"{r} (\d+)mW", l)[1]) / 1000 for r in RAILS}
        tj = re.search(r"tj@([\d.]+)C", l)
        out.append((i * dt, w, int(re.search(r"GR3D_FREQ (\d+)%", l)[1]), float(tj[1]) if tj else None))
    return out, dt


def row(ds, arm, clip, idle):
    r = E.run(ds, arm, clip)
    p = f"{E.R}/{ds}/{arm}/{clip}.tegrastats"
    if not r or not os.path.exists(p):
        return None
    T, dt = trace(p)
    ttft = r["ttft"] / 1000
    j = json.load(open(f"{E.R}/{ds}/{arm}/{clip}.json"))
    whole = (j["requests"][0]["total_ms"] if "requests" in j else j["timing"]["ttft_ms"] + j["timing"]["generation_ms"]) / 1000
    busy = [i for i, s in enumerate(T) if s[2] >= 50]
    if not busy:
        return None
    end = busy[-1] + 1
    start = max(0, end - round(whole / dt))
    first = min(end, start + round(ttft / dt))
    tot = lambda s: sum(s[1].values())
    e = lambda a, b, k=None: sum((s[1][k] if k else tot(s)) * dt for s in T[a:b])
    win = T[start:end]
    if first <= start or not win:
        return None   # no request window found in the trace
    tjs = [s[3] for s in win if s[3] is not None]
    d = dict(dataset=ds, clip=clip, arm=arm, ttft_s=ttft, answer_s=whole, frames=r["frames"],
             e_ttft=e(start, first), e_answer=e(start, end),
             e_ttft_gpu=e(start, first, "VDD_GPU_SOC"), e_ttft_cpu=e(start, first, "VDD_CPU_CV"),
             e_ttft_sys=e(start, first, "VIN_SYS_5V0"),
             p_avg_ttft=e(start, first) / max(dt, (first - start) * dt), p_avg_decode=e(first, end) / max(dt, (end - first) * dt),
             p_peak=max(tot(s) for s in win), tj_rise=(max(tjs) - tjs[0]) if tjs else None, oc3=r["oc3"],
             samples_ttft=first - start, text=r["text"])
    d["e_ttft_dyn"] = d["e_ttft"] - idle * ttft
    d["e_answer_dyn"] = d["e_answer"] - idle * whole
    return d, T, start, first, end


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/data/accuracy/power"
    idle = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    rows, traces = [], {}
    for ds, _ in DATASETS:
        for c in E.clips(ds, "pwd24"):
            for a in ARMS:
                x = row(ds, a, c, idle)
                if x:
                    rows.append(x[0])
                    if (ds, c) in EXAMPLES:
                        traces[(ds, c, a)] = x[1:]
    fields = [k for k in rows[0] if k != "text"] if rows else []
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # example traces: one CSV per clip, time from the start of the request, every arm that ran it
    for ds, c in EXAMPLES:
        arms = [a for a in ARMS if (ds, c, a) in traces]
        if not arms:
            continue
        with open(f"{prefix}_trace_{ds}_{c}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "t_s", "phase", "gpu_soc_w", "cpu_cv_w", "sys_5v_w", "total_w", "gpu_busy_pct", "tj_c"])
            for a in arms:
                T, s0, s1_, s2 = traces[(ds, c, a)]
                for i in range(s0, s2):
                    t, wr, g, tj = T[i]
                    w.writerow([a, round(t - T[s0][0], 2), "to first token" if i < s1_ else "decode",
                                *(round(wr[r], 2) for r in RAILS), round(sum(wr.values()), 2), g, tj])

    f0 = lambda x, n=0: "—" if x is None else f"{x:,.{n}f}"
    mean = lambda xs: st.mean(xs) if xs else None
    out = ["## Summary\n", f"Idle power subtracted for the energy above idle: {idle:.1f} W.\n",
           "Energy to the first token (J, mean per clip) and its ratio qwen2fps / arm:\n",
           "| Dataset | " + " | ".join(ARMS) + " | pwd24 × vs qwen2fps |", "|---|" + "---|" * (len(ARMS) + 1)]
    sec = []
    for ds, name in DATASETS:
        get = {(r["clip"], r["arm"]): r for r in rows if r["dataset"] == ds}
        cl = list(dict.fromkeys(r["clip"] for r in rows if r["dataset"] == ds))
        if not cl:
            continue
        g = lambda c, a, k: get[(c, a)][k] if (c, a) in get and get[(c, a)].get(k) is not None else None
        col = lambda a, k: mean([g(c, a, k) for c in cl if g(c, a, k) is not None])
        rq = mean([g(c, "qwen2fps", "e_ttft") / g(c, "pwd24", "e_ttft") for c in cl
                   if g(c, "qwen2fps", "e_ttft") and g(c, "pwd24", "e_ttft")])
        out.append(f"| {name} | " + " | ".join(f0(col(a, "e_ttft")) for a in ARMS) + f" | {f0(rq, 2)} |")
        it = E.items(ds)
        sec += [f"## {name} ({len(cl)} clips)\n"]
        # per arm: energy, power, rails, stress
        sec += ["**Per arm (means per clip):** energy to the first token and to the whole answer, above idle, average power "
                "to the first token and while decoding, peak power, the rail split of the energy to the first token, junction "
                "rise, oc3 events per run:\n",
                "| arm | J to first token | J whole answer | J above idle (first token) | W avg, to first token | W avg, decode | "
                "W peak | GPU/SoC | CPU/CV | SYS 5V | junction rise (C) | oc3/run |", "|---|" + "---|" * 11]
        for a in ARMS:
            if col(a, "e_ttft") is None:
                continue
            et = col(a, "e_ttft")
            sec.append(f"| {a} | {f0(et)} | {f0(col(a, 'e_answer'))} | {f0(col(a, 'e_ttft_dyn'))} | {f0(col(a, 'p_avg_ttft'), 1)} | "
                       f"{f0(col(a, 'p_avg_decode'), 1)} | {f0(col(a, 'p_peak'), 1)} | "
                       + " | ".join(f"{100 * col(a, k) / et:.0f}%" for k in ("e_ttft_gpu", "e_ttft_cpu", "e_ttft_sys")) +
                       f" | {f0(col(a, 'tj_rise'), 1)} | {f0(col(a, 'oc3'), 1)} |")
        # energy per correct answer (multiple choice)
        if it and "answer" in next(iter(it.values())):
            sec += ["", "**Energy per correct answer (multiple choice):**\n", "| arm | correct | J per correct answer (whole answer) |", "|---|---|---|"]
            for a in ARMS:
                rr = [get[(c, a)] for c in cl if (c, a) in get and c in it]
                if not rr:
                    continue
                ok = sum(E.letter(r["text"]) == it[r["clip"]]["answer"] for r in rr)
                sec.append(f"| {a} | {ok}/{len(rr)} | {f0(sum(r['e_answer'] for r in rr) / ok) if ok else '—'} |")
        # energy saved by tier, from qwen2fps
        sec += ["", "**Energy saved by tier, to the first token (J; negative = more energy).** The tiers add up to qwen2fps − pwd24; "
                "\\* this tier includes the one before it, whose arm has no run:\n",
                "| clip | qwen2fps J | input (→ base) | kernels | frame count | adaptive sampling | window cache | drop-clean | pwd24 J | saved vs qwen2fps |",
                "|---|" + "---|" * 9]
        arms = [a for a in ARMS if any((c, a) in get for c in cl)]
        complete = [c for c in cl if all((c, a) in get for a in arms)]
        tot = {k: [] for k in [k for k, _ in TIERS] + ["q", "d", "s"]}
        for c in cl:
            if (c, "qwen2fps") not in get or (c, "pwd24") not in get:
                continue
            cells, prev, merged = [], g(c, "qwen2fps", "e_ttft"), False
            for k, a in TIERS:
                if (c, a) not in get:
                    cells.append("—")
                    merged = True
                    continue
                v = prev - g(c, a, "e_ttft")
                cells.append(f0(v) + ("*" if merged else ""))
                if c in complete:
                    tot[k].append(v)
                prev, merged = g(c, a, "e_ttft"), False
            sv = g(c, "qwen2fps", "e_ttft") - g(c, "pwd24", "e_ttft")
            if c in complete:
                tot["q"].append(g(c, "qwen2fps", "e_ttft")); tot["d"].append(g(c, "pwd24", "e_ttft")); tot["s"].append(sv)
            sec.append(f"| {c} | {f0(g(c, 'qwen2fps', 'e_ttft'))} | " + " | ".join(cells) + f" | {f0(g(c, 'pwd24', 'e_ttft'))} | {f0(sv)} |")
        tm = {k: mean(v) for k, v in tot.items()}
        share = lambda k: f" ({100 * tm[k] / tm['s']:.0f}%)" if tm[k] is not None and tm["s"] and tm["s"] > 0 else ""
        sec += [f"| **mean** ({len(complete)} clips with every arm) | {f0(tm['q'])} | " +
                " | ".join(f0(tm[k]) + share(k) for k, _ in TIERS) + f" | {f0(tm['d'])} | {f0(tm['s'])} |", ""]
    open(f"{prefix}.md", "w").write("\n".join(out + [""] + sec) + "\n")
    print(f"{len(rows)} rows -> {prefix}.csv, {prefix}.md; traces for {sorted(set((d, c) for d, c, _ in traces))}")


if __name__ == "__main__":
    main()
