#!/usr/bin/env python3
"""E14, equal frames: every arm gets the same N uniform frames (N = 4, 8, 16), so the differences come from the system
alone, not from sending fewer frames. One optimization tier is added per arm:
  stock      --baseline                                        (stock llama.cpp kernels)
  kernels    + window alignment and the kernel switches        (uni4 / uni_arch / uni16)
  cache      + per-window ViT cache, threshold 24              (ucwN)
  drop       + drop-clean                                      (ucwdN)
qwen2fps (Qwen's own 2 fps frame pairs, stock kernels) is listed beside them at its own frame and token count.

    EVAL_RUNS=/home/nvidia/data/accuracy/runs_eval_fa python3 eq_frames.py [out prefix=/home/nvidia/data/accuracy/eq_frames]
        -> <prefix>.csv (per dataset, arm and N: means over the clips every arm ran) and <prefix>.md

Per arm and N: mean TTFT, energy to the first token (power_analysis.py: rails integrated up to it), average power to
it, visual tokens through the ViT and into the LLM, and energy per frame and per 1,000 input visual tokens (the tokens
the frames make at full size, before the cache: with drop-clean the LLM sees fewer, but the ViT still reads every frame).
"""
import csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary as E
import power_analysis as P

DATASETS = [("tacos", "TACoS"), ("perception", "Perception Test"), ("videomme", "Video-MME"), ("sintel", "Sintel")]
TIERS = ["stock", "kernels", "cache", "drop"]
ARM = {(4, "stock"): "base4", (4, "kernels"): "uni4", (4, "cache"): "ucw4", (4, "drop"): "ucwd4",
       (8, "stock"): "base", (8, "kernels"): "uni_arch", (8, "cache"): "ucw8", (8, "drop"): "ucwd8",
       (16, "stock"): "base16", (16, "kernels"): "uni16", (16, "cache"): "ucw16", (16, "drop"): "ucwd16"}


def measure(ds, arm, clip):
    r = E.run(ds, arm, clip)
    pr = P.row(ds, arm, clip, 0.0)
    if not r or not pr:
        return None
    d = pr[0]
    b = r["t"]["breakdown"]
    wc = b.get("window_cache", {}).get("frames", [])
    llm = sum(f["tokens_prefilled"] for f in wc) if wc else r["tokens"]
    if "requests" in r["j"] or arm == "qwen2fps":
        llm = r["tokens"]
    return dict(ttft=r["ttft"] / 1000, j=d["e_ttft"], w=d["p_avg_ttft"], frames=r["frames"], vit_tok=r["tokens"],
                llm_tok=llm)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/data/accuracy/eq_frames"
    rows, md = [], []
    for ds, name in DATASETS:
        arms = list(ARM.values()) + ["qwen2fps"]
        clips = [c for c in E.clips(ds, "base") if all(os.path.exists(f"{E.R}/{ds}/{a}/{c}.json") for a in arms)]
        if not clips:
            continue
        M = {a: [m for m in (measure(ds, a, c) for c in clips) if m] for a in arms}
        mean = lambda a, k: st.mean(m[k] for m in M[a])
        md += [f"\n**{name}** ({len(clips)} clips with every arm)\n",
               "| frames | tier | TTFT s | J to first token | W avg | visual tokens, ViT | visual tokens, LLM | J per frame | "
               "J per 1k input tokens | × energy vs stock |", "|---|---|---|---|---|---|---|---|---|---|"]
        for n in (4, 8, 16):
            stock = mean(ARM[(n, "stock")], "j")
            for t in TIERS:
                a = ARM[(n, t)]
                if not M[a]:
                    continue
                v = {k: mean(a, k) for k in ("ttft", "j", "w", "frames", "vit_tok", "llm_tok")}
                v["in_tok"] = v["vit_tok"] if t == "stock" else mean(ARM[(n, "kernels")], "vit_tok")
                rows.append(dict(dataset=name, n=n, tier=t, arm=a, clips=len(M[a]), **{k: round(x, 3) for k, x in v.items()}))
                md.append(f"| {n} | {t} | {v['ttft']:.2f} | {v['j']:,.0f} | {v['w']:.1f} | {v['vit_tok']:,.0f} | {v['llm_tok']:,.0f} | "
                          f"{v['j'] / n:.1f} | {1000 * v['j'] / v['in_tok']:.1f} | {stock / v['j']:.2f} |")
        q = {k: mean("qwen2fps", k) for k in ("ttft", "j", "w", "frames", "vit_tok", "llm_tok")}
        q["in_tok"] = q["vit_tok"]
        rows.append(dict(dataset=name, n="2fps", tier="qwen2fps", arm="qwen2fps", clips=len(M["qwen2fps"]),
                         **{k: round(x, 3) for k, x in q.items()}))
        md.append(f"| {q['frames']:.1f} (2 fps) | qwen2fps | {q['ttft']:.2f} | {q['j']:,.0f} | {q['w']:.1f} | {q['vit_tok']:,.0f} | "
                  f"{q['llm_tok']:,.0f} | {q['j'] / q['frames']:.1f} | {1000 * q['j'] / q['in_tok']:.1f} | |")
    with open(prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    open(prefix + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
