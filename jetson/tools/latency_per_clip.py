#!/usr/bin/env python3
"""Per-clip latency of the main evaluation (scripts/run_eval.sh), every arm against qwen2fps (Qwen2.5-VL's own 2 fps
video input). Reads the runs through eval_summary.py (EVAL_RUNS=... for runs_eval_fa, ViT flash attention on).

    python3 latency_per_clip.py [out prefix=/home/nvidia/data/accuracy/latency_per_clip]   -> <prefix>.csv, <prefix>.md

Per clip and arm: TTFT and its stages, summing to TTFT
  qwen2fps          ViT, prefill, other (load + tokenize, and the token dump where the run wrote one)
  llama-jetson-vlm  flow exposed (live optical flow not hidden behind the anchor frame), ViT, image prefill, text
                    prefill, other (frame load, tokenize, selection, the cache's change test); flow hidden is the
                    part of the flow that ran under the anchor frame's ViT and prefill
plus energy to the first token (eval_summary.py), peak board RAM (tegrastats; CSV only: it includes the OS), visual tokens encoded and prefilled, and the
window cache's frame actions. qwen2fps TTFT includes writing its token dump (the qwen2fps_nodump runs are at the older,
smaller frame size on Sintel and MSVD, so they are not used; on Charades-STA, at the same size, the dump costs nothing
measurable).

Savings by optimization tier (s), base -> pwd24 one tier at a time; the tiers sum to base TTFT minus pwd24 TTFT:
  kernels   base -> uni_arch       the kernel switches and window alignment (same 8 frames)
  frames    uni_arch -> uni4       4 uniform frames instead of 8
  adaptive  uni4 -> ours_nocache   motion-selected frames (3-8), live optical flow, anchor-first
  cache     ours_nocache -> pw24   the per-window ViT cache
  drop      pw24 -> pwd24          drop-clean: later frames prefill only their re-encoded windows
A missing arm merges its tier into the next one (marked *). The markdown opens with a summary across datasets.
"""
import csv, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary as E

ARMS = ["qwen2fps", "base", "uni_arch", "uni4", "ours_nocache", "pw24", "pwd24"]
TIERS = [("kernels", "uni_arch"), ("frames", "uni4"), ("adaptive", "ours_nocache"), ("cache", "pw24"), ("drop", "pwd24")]
DATASETS = [("sintel", "Sintel"), ("msvd", "MSVD"), ("charades", "Charades-STA"), ("tacos", "TACoS"),
            ("nextqa", "NExT-QA"), ("videomme", "Video-MME"), ("perception", "Perception Test")]
TOK_PER_WINDOW = 16   # a 112 px ViT window = 4 x 4 merged tokens


def peak_ram(ds, cfg, clip):
    p = f"{E.R}/{ds}/{cfg}/{clip}.tegrastats"
    v = [int(m[1]) for l in open(p) for m in [re.search(r"RAM (\d+)/", l)] if m] if os.path.exists(p) else []
    return max(v) if v else None


def row(ds, arm, clip):
    cfg = arm
    r = E.run(ds, cfg, clip)
    if not r:
        return None
    d = dict(dataset=ds, clip=clip, arm=arm, ttft=r["ttft"], energy_j=r["energy"], oc3=r["oc3"],
             ram_peak_mb=peak_ram(ds, cfg, clip), frames=r["frames"], tokens_vit=r["tokens"])
    if arm == "qwen2fps":
        q = json.load(open(f"{E.R}/{ds}/{cfg}/{clip}.json"))["requests"][0]
        d.update(frames=q["n_frames"], tokens_vit=q["n_visual_tokens"], tokens_llm=q["n_visual_tokens"],
                 vit=q["vit_ms"], prefill_img=q["prefill_ms"], prefill_txt=0.0, flow_exposed=0.0, flow_hidden=0.0)
    else:
        b = r["t"]["breakdown"]
        s = lambda k: b.get("rest", {}).get(k, 0) + b.get("anchor", {}).get(k, 0)
        flow = b.get("flow_ms", 0) or 0
        hidden = min(flow, b["anchor"]["done_at_ms"]) if "anchor" in b else 0
        wc = b.get("window_cache", {}).get("frames", [])
        d.update(vit=s("vit_ms"), prefill_img=s("prefill_img_ms"), prefill_txt=s("prefill_txt_ms"),
                 flow_exposed=flow - hidden, flow_hidden=hidden,
                 tokens_llm=sum(f["tokens_prefilled"] for f in wc) if wc else r["tokens"])
        if wc:
            full = wc[0]["tokens_encoded"]   # the key frame is encoded whole
            later = [f for f in wc[1:]]
            d.update(wc_key=sum(f["action"] == "key" for f in wc), wc_full=sum(f["action"] == "full" for f in wc),
                     wc_partial=sum(f["action"] == "pw" for f in wc), wc_skip=sum(f["action"] == "skip" for f in wc),
                     wc_windows_reenc=(sum(f["n_dirty"] for f in later) / (len(later) * full / TOK_PER_WINDOW))
                     if later else 0.0,
                     wc_vit_tokens_frac=sum(f["tokens_encoded"] for f in wc) / (full * len(wc)),
                     wc_llm_tokens_frac=sum(f["tokens_prefilled"] for f in wc) / (full * len(wc)),
                     wc_test_ms=sum(f["diff_ms"] + f["prep_ms"] for f in wc))
    d["other"] = d["ttft"] - d["flow_exposed"] - d["vit"] - d["prefill_img"] - d["prefill_txt"]
    return d


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/data/accuracy/latency_per_clip"
    rows = []
    for ds, _ in DATASETS:
        clips = E.clips(ds, "pw24")
        for c in clips:
            for a in ARMS:
                x = row(ds, a, c)
                if x:
                    rows.append(x)
    fields = ["dataset", "clip", "arm", "ttft", "flow_exposed", "flow_hidden", "vit", "prefill_img", "prefill_txt",
              "other", "frames", "tokens_vit", "tokens_llm", "energy_j", "ram_peak_mb", "oc3", "wc_key", "wc_full",
              "wc_partial", "wc_skip", "wc_windows_reenc", "wc_vit_tokens_frac", "wc_llm_tokens_frac", "wc_test_ms"]
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    s1 = lambda ms: "—" if ms is None else f"{ms / 1000:.2f}"
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}%"
    mean = lambda xs: st.mean(xs) if xs else None
    fr = lambda x: "—" if x is None else (f"{x:.0f}" if x == int(x) else f"{x:.1f}")
    head = {"qwen2fps": "qwen2fps (2 fps)", "base": "base (8)", "uni_arch": "uni_arch (8)", "uni4": "uni4 (4)",
            "ours_nocache": "ours_nocache (3–8)", "pw24": "pw24 (3–8)", "pwd24": "pwd24 (3–8)"}
    summ1 = ["| Dataset | clips | qwen2fps frames | our frames | qwen2fps TTFT | base TTFT | pwd24 TTFT | pwd24 × vs qwen2fps | pwd24 × vs base |",
             "|---|---|---|---|---|---|---|---|---|"]
    summ2 = ["| Dataset (clips) | base TTFT | kernels | frame count | adaptive sampling | window cache | drop-clean | pwd24 TTFT | saved vs base | saved vs qwen2fps |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for ds, name in DATASETS:
        get = {(r["clip"], r["arm"]): r for r in rows if r["dataset"] == ds}
        cl = list(dict.fromkeys(r["clip"] for r in rows if r["dataset"] == ds))
        if not cl:
            continue
        arms = [a for a in ARMS if any((c, a) in get for c in cl)]
        g = lambda c, a, k="ttft": get[(c, a)][k] if (c, a) in get else None
        out += [f"## {name} ({len(cl)} clips)\n"]

        # 1. TTFT, frames, speed-up
        out += ["**TTFT (s), frames sent, and speed-up of pwd24 (×, other TTFT / pwd24 TTFT):**\n",
                "| clip | qwen2fps frames | our frames | " + " | ".join(head[a] for a in arms) + " | pwd24 × vs qwen2fps | pwd24 × vs base |",
                "|---|---|---|" + "---|" * (len(arms) + 2)]
        sp = lambda c, a: g(c, a) / g(c, "pwd24") if g(c, a) and g(c, "pwd24") else None
        for c in cl:
            out.append(f"| {c} | {fr(g(c, 'qwen2fps', 'frames'))} | {fr(g(c, 'pwd24', 'frames'))} | " +
                       " | ".join(s1(g(c, a)) for a in arms) + " | " +
                       " | ".join(f"{sp(c, a):.2f}" if sp(c, a) else "—" for a in ("qwen2fps", "base")) + " |")
        col = lambda a, k="ttft", f=mean: f([g(c, a, k) for c in cl if g(c, a, k) is not None]) if any(g(c, a, k) is not None for c in cl) else None
        spm = {a: mean([x for x in (sp(c, a) for c in cl) if x]) for a in ("qwen2fps", "base")}
        out.append(f"| **mean** | {fr(col('qwen2fps', 'frames'))} | {fr(col('pwd24', 'frames'))} | " +
                   " | ".join(s1(col(a)) for a in arms) + " | " + " | ".join(f"**{spm[a]:.2f}**" if spm[a] else "—" for a in spm) + " |")
        out.append("| **median** | | | " + " | ".join(s1(col(a, f=st.median)) for a in arms) + " | | |\n")
        summ1.append(f"| {name} | {len(cl)} | {fr(col('qwen2fps', 'frames'))} | {fr(col('pwd24', 'frames'))} | {s1(col('qwen2fps'))} | "
                     f"{s1(col('base'))} | {s1(col('pwd24'))} | {spm['qwen2fps']:.2f} | {spm['base']:.2f} |")

        # 2. savings by optimization tier, from base
        out += ["**Savings by optimization tier (s saved; negative = slower).** kernels: base → uni_arch (same 8 frames); "
                "frame count: uni_arch → uni4 (8 → 4 uniform frames); adaptive sampling: uni4 → ours_nocache (motion-selected "
                "3–8 frames, live optical flow, anchor-first); window cache: ours_nocache → pw24; drop-clean: pw24 → pwd24. "
                "The tiers add up to base − pwd24. \\* this tier includes the one before it, whose arm has no run:\n",
                "| clip | base TTFT | kernels | frame count | adaptive sampling | window cache | drop-clean | pwd24 TTFT | saved vs base | saved vs qwen2fps |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        tot = {k: [] for k in [k for k, _ in TIERS] + ["base", "pwd24", "vs_base", "vs_q"]}
        complete = [c for c in cl if all((c, a) in get for a in arms)]
        for c in cl:
            if g(c, "base") is None or g(c, "pwd24") is None:
                continue
            cells, prev, merged = [], g(c, "base"), False
            for k, a in TIERS:
                if g(c, a) is None:
                    cells.append("—")
                    merged = True
                    continue
                v = prev - g(c, a)
                cells.append(f"{v / 1000:.2f}" + ("*" if merged else ""))
                if c in complete:
                    tot[k].append(v)
                prev, merged = g(c, a), False
            vb, vq = g(c, "base") - g(c, "pwd24"), (g(c, "qwen2fps") - g(c, "pwd24")) if g(c, "qwen2fps") else None
            if c in complete:
                tot["base"].append(g(c, "base")); tot["pwd24"].append(g(c, "pwd24")); tot["vs_base"].append(vb)
                tot["vs_q"] += [vq] if vq is not None else []
            out.append(f"| {c} | {s1(g(c, 'base'))} | " + " | ".join(cells) + f" | {s1(g(c, 'pwd24'))} | {s1(vb)} | {s1(vq)} |")
        star = lambda k: "*" if k != TIERS[0][0] and TIERS[[x for x, _ in TIERS].index(k) - 1][1] not in arms else ""
        tm = {k: mean(v) for k, v in tot.items()}
        share = lambda k: f" ({100 * tm[k] / tm['vs_base']:.0f}%)" if tm[k] is not None and tm["vs_base"] else ""
        out.append(f"| **mean** ({len(complete)} clips with every arm) | {s1(tm['base'])} | " +
                   " | ".join(s1(tm[k]) + (star(k) if tm[k] is not None else "") + share(k) for k, _ in TIERS) +
                   f" | {s1(tm['pwd24'])} | {s1(tm['vs_base'])} | {s1(tm['vs_q'])} |\n")
        summ2.append(f"| {name} ({len(complete)}) | {s1(tm['base'])} | " +
                     " | ".join(s1(tm[k]) + (star(k) if tm[k] is not None else "") + share(k) for k, _ in TIERS) +
                     f" | {s1(tm['pwd24'])} | {s1(tm['vs_base'])} | {s1(tm['vs_q'])} |")

        # 3. stages, qwen2fps against the design
        keys = [("qwen2fps", "vit"), ("qwen2fps", "prefill_img"), ("qwen2fps", "other"), ("pwd24", "flow_exposed"),
                ("pwd24", "flow_hidden"), ("pwd24", "vit"), ("pwd24", "prefill_img"), ("pwd24", "prefill_txt"), ("pwd24", "other")]
        hd = ["Q ViT", "Q prefill", "Q other", "D flow exposed", "D flow hidden", "D ViT", "D image prefill", "D text prefill", "D other"]
        out += ["**Stages (s), Q = qwen2fps, D = pwd24.** Q: ViT + prefill + other = TTFT. D: flow exposed + ViT + image prefill + "
                "text prefill + other = TTFT; flow hidden ran under the anchor frame and costs nothing:\n",
                "| clip | " + " | ".join(hd) + " |", "|---|" + "---|" * len(hd)]
        for c in cl:
            out.append(f"| {c} | " + " | ".join(s1(g(c, a, k)) for a, k in keys) + " |")
        out.append("| **mean** | " + " | ".join(s1(col(a, k)) for a, k in keys) + " |\n")

        # 4. window cache
        if any((c, "pw24") in get and "wc_key" in get[(c, "pw24")] for c in cl):
            out += ["**Window cache use (pw24 run).** Frame actions: key (encoded whole, the anchor), full (re-keyed: too many windows changed), "
                    "partial (only changed windows re-encoded). Windows re-encoded: share of the later frames' windows the change "
                    "test sent to the ViT. ViT saved: ours_nocache ViT − pw24 ViT (same frames). Prefill saved: pw24 image prefill − "
                    "pwd24 image prefill:\n",
                    "| clip | key / full / partial | windows re-encoded | ViT tokens encoded | ViT saved (s) | LLM tokens, pwd24 | "
                    "prefill saved (s) | change test (ms) |", "|---|---|---|---|---|---|---|---|"]
            vs, ps = [], []
            for c in cl:
                p, q, o = get.get((c, "pw24")), get.get((c, "pwd24")), get.get((c, "ours_nocache"))
                if not p or "wc_key" not in p:
                    continue
                v = o["vit"] - p["vit"] if o else None
                pr = p["prefill_img"] - q["prefill_img"] if q else None
                vs += [v] if v is not None else []
                ps += [pr] if pr is not None else []
                out.append(f"| {c} | {p['wc_key']} / {p['wc_full']} / {p['wc_partial']} | {pct(p['wc_windows_reenc'])} | "
                           f"{pct(p['wc_vit_tokens_frac'])} | {s1(v)} | {pct(q['wc_llm_tokens_frac']) if q else '—'} | {s1(pr)} | "
                           f"{p['wc_test_ms']:.1f} |")
            P = [get[(c, "pw24")] for c in cl if (c, "pw24") in get and "wc_key" in get[(c, "pw24")]]
            Q = [get[(c, "pwd24")] for c in cl if (c, "pwd24") in get and "wc_key" in get[(c, "pwd24")]]
            out.append(f"| **mean** | | {pct(mean([p['wc_windows_reenc'] for p in P]))} | {pct(mean([p['wc_vit_tokens_frac'] for p in P]))} | "
                       f"{s1(mean(vs))} | {pct(mean([q['wc_llm_tokens_frac'] for q in Q]))} | {s1(mean(ps))} | "
                       f"{mean([p['wc_test_ms'] for p in P]):.1f} |\n")

        # 5. cost of the answer: tokens, energy, memory
        ea = [a for a in ("qwen2fps", "base", "uni4", "pw24", "pwd24") if a in arms]
        out += ["**Visual tokens to the LLM and energy to the first token (J):**\n",
                "| clip | " + " | ".join(f"{a} tokens" for a in ("qwen2fps", "pwd24")) + " | " + " | ".join(f"{a} J" for a in ea) +
                " |", "|---|" + "---|" * (2 + len(ea))]
        g = lambda c, a, k: get[(c, a)][k] if (c, a) in get and get[(c, a)].get(k) is not None else None
        for c in cl:
            out.append(f"| {c} | " + " | ".join(f"{g(c, a, 'tokens_llm'):,}" if g(c, a, "tokens_llm") is not None else "—" for a in ("qwen2fps", "pwd24")) +
                       " | " + " | ".join(f"{g(c, a, 'energy_j'):.0f}" if g(c, a, "energy_j") else "—" for a in ea) + " |")
        mm = lambda a, k: mean([g(c, a, k) for c in cl if g(c, a, k) is not None])
        out.append("| **mean** | " + " | ".join(f"{mm(a, 'tokens_llm'):,.0f}" for a in ("qwen2fps", "pwd24")) + " | " +
                   " | ".join(f"{mm(a, 'energy_j'):.0f}" if mm(a, "energy_j") else "—" for a in ea) + " |\n")
    top = ["## Summary\n", "Frames sent, TTFT (mean s) and the speed-up of pwd24 (mean of per-clip ratios):\n", *summ1, "",
           "Mean seconds saved by each optimization tier, base → pwd24 (share of the saving over base in parentheses):\n", *summ2, ""]
    open(f"{prefix}.md", "w").write("\n".join(top + out) + "\n")
    print(f"{len(rows)} rows -> {prefix}.csv, {prefix}.md")


if __name__ == "__main__":
    main()
