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

Savings (s), qwen2fps -> pwd24 one change at a time; consecutive steps sum to qwen2fps TTFT minus pwd24 TTFT:
  input     qwen2fps -> base       8 uniform frames, image input, stock kernels, instead of Qwen's 2 fps frame pairs
  kernels   base -> uni_arch       the kernel switches and window alignment
  frames    uni_arch -> uni4       4 frames instead of 8
  select    uni4 -> ours_nocache   motion-selected frames (3-8), live optical flow, anchor-first
  cache     ours_nocache -> pw24   the per-window ViT cache
  drop      pw24 -> pwd24          drop-clean: later frames prefill only their re-encoded windows
A missing arm merges its step into the next one (marked *).
"""
import csv, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary as E

ARMS = ["qwen2fps", "base", "uni_arch", "uni4", "ours_nocache", "pw24", "pwd24"]
STEPS = [("input", "base"), ("kernels", "uni_arch"), ("frames", "uni4"), ("select", "ours_nocache"),
         ("cache", "pw24"), ("drop", "pwd24")]
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
    out = []
    for ds, name in DATASETS:
        get = {(r["clip"], r["arm"]): r for r in rows if r["dataset"] == ds}
        cl = list(dict.fromkeys(r["clip"] for r in rows if r["dataset"] == ds))
        if not cl:
            continue
        arms = [a for a in ARMS if any((c, a) in get for c in cl)]
        lab = lambda c: f"{c} ({get[(c, 'qwen2fps')]['frames']} / {get[(c, 'pw24')]['frames']})" if (c, "qwen2fps") in get else c
        mean = lambda xs: st.mean(xs) if xs else None
        out += [f"## {name} ({len(cl)} clips)\n", f"Clip label: (frames qwen2fps sends / frames ours sends).\n"]

        # 1. TTFT and speed-up over qwen2fps
        out += ["**TTFT (s) and speed-up over qwen2fps (×, qwen2fps TTFT / arm TTFT):**\n",
                "| clip | " + " | ".join(arms) + " | uni4 × | pw24 × | pwd24 × |", "|---|" + "---|" * (len(arms) + 3)]
        sp = lambda c, a: get[(c, "qwen2fps")]["ttft"] / get[(c, a)]["ttft"] if (c, a) in get and (c, "qwen2fps") in get else None
        for c in cl:
            out.append(f"| {lab(c)} | " + " | ".join(s1(get[(c, a)]["ttft"]) if (c, a) in get else "—" for a in arms) +
                       " | " + " | ".join(f"{sp(c, a):.2f}" if sp(c, a) else "—" for a in ("uni4", "pw24", "pwd24")) + " |")
        out.append("| **mean** | " + " | ".join(s1(mean([get[(c, a)]["ttft"] for c in cl if (c, a) in get])) for a in arms) +
                   " | " + " | ".join(f"**{mean([x for x in (sp(c, a) for c in cl) if x]):.2f}**" for a in ("uni4", "pw24", "pwd24")) + " |")
        out.append("| **median** | " + " | ".join(s1(st.median([get[(c, a)]["ttft"] for c in cl if (c, a) in get])) for a in arms) +
                   " | " + " | ".join(f"{st.median([x for x in (sp(c, a) for c in cl) if x]):.2f}" for a in ("uni4", "pw24", "pwd24")) + " |\n")

        # 2. stages, qwen2fps against the design
        keys = [("qwen2fps", "vit"), ("qwen2fps", "prefill_img"), ("qwen2fps", "other"), ("pwd24", "flow_exposed"),
                ("pwd24", "flow_hidden"), ("pwd24", "vit"), ("pwd24", "prefill_img"), ("pwd24", "prefill_txt"), ("pwd24", "other")]
        head = ["qwen2fps ViT", "qwen2fps prefill", "qwen2fps other", "pwd24 flow exposed", "pwd24 flow hidden",
                "pwd24 ViT", "pwd24 image prefill", "pwd24 text prefill", "pwd24 other"]
        out += ["**Stages (s).** qwen2fps: ViT + prefill + other = TTFT. pwd24: flow exposed + ViT + image prefill + text prefill + "
                "other = TTFT; flow hidden ran under the anchor frame and costs nothing:\n",
                "| clip | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for c in cl:
            out.append(f"| {lab(c)} | " + " | ".join(s1(get[(c, a)][k]) if (c, a) in get else "—" for a, k in keys) + " |")
        out.append("| **mean** | " + " | ".join(s1(mean([get[(c, a)][k] for c in cl if (c, a) in get])) for a, k in keys) + " |\n")

        # 3. where the savings come from
        out += ["**Where the savings come from (s saved by each step; negative = slower).** input: Qwen's 2 fps pairs → 8 uniform "
                "frames, stock kernels (base); kernels: the kernel switches (uni_arch); frames: 8 → 4 (uni4); select: motion "
                "selection + live flow + anchor-first (ours_nocache); cache: per-window ViT cache (pw24); drop: drop-clean (pwd24). "
                "\\* the step includes the one before it, whose arm has no run:\n",
                "| clip | input | kernels | frames | select | cache | drop | total |", "|---|---|---|---|---|---|---|---|"]
        tot = {k: [] for k, _ in STEPS + [("total", None)]}
        complete = [c for c in cl if all((c, a) in get for a in arms)]
        for c in cl:
            if (c, "qwen2fps") not in get or (c, "pwd24") not in get:
                continue
            cells, prev, merged = [], get[(c, "qwen2fps")]["ttft"], False
            for k, a in STEPS:
                if (c, a) not in get:
                    cells.append("—")
                    merged = True
                    continue
                v = prev - get[(c, a)]["ttft"]
                cells.append(f"{v / 1000:.2f}" + ("*" if merged else ""))
                if c in complete:
                    tot[k].append(v)
                prev, merged = get[(c, a)]["ttft"], False
            t = get[(c, "qwen2fps")]["ttft"] - get[(c, "pwd24")]["ttft"]
            if c in complete:
                tot["total"].append(t)
            out.append(f"| {lab(c)} | " + " | ".join(cells) + f" | {t / 1000:.2f} |")
        star = lambda k, a: "*" if a and a in arms and STEPS[[x for x, _ in STEPS].index(k) - 1][1] not in arms else ""
        out.append(f"| **mean** ({len(complete)} clips with every arm) | " +
                   " | ".join(s1(mean(tot[k])) + (star(k, a) if tot[k] else "") for k, a in STEPS + [("total", None)]) + " |\n")

        # 4. window cache
        if any((c, "pw24") in get and "wc_key" in get[(c, "pw24")] for c in cl):
            out += ["**Window cache use.** Frame actions: key (encoded whole, the anchor), full (re-keyed: too many windows changed), "
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
                out.append(f"| {lab(c)} | {p['wc_key']} / {p['wc_full']} / {p['wc_partial']} | {pct(p['wc_windows_reenc'])} | "
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
            out.append(f"| {lab(c)} | " + " | ".join(f"{g(c, a, 'tokens_llm'):,}" if g(c, a, "tokens_llm") is not None else "—" for a in ("qwen2fps", "pwd24")) +
                       " | " + " | ".join(f"{g(c, a, 'energy_j'):.0f}" if g(c, a, "energy_j") else "—" for a in ea) + " |")
        mm = lambda a, k: mean([g(c, a, k) for c in cl if g(c, a, k) is not None])
        out.append("| **mean** | " + " | ".join(f"{mm(a, 'tokens_llm'):,.0f}" for a in ("qwen2fps", "pwd24")) + " | " +
                   " | ".join(f"{mm(a, 'energy_j'):.0f}" if mm(a, "energy_j") else "—" for a in ea) + " |\n")
    open(f"{prefix}.md", "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} rows -> {prefix}.csv, {prefix}.md")


if __name__ == "__main__":
    main()
