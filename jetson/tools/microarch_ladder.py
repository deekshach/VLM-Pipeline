#!/usr/bin/env python3
"""Microarchitecture of the kernel switches against stock kernels, per transformer layer, from the Nsight Compute
ladder of scripts/run_microarch_fa.sh (ViT flash attention on).

    python3 microarch_ladder.py [dir=jetson/experiments/profiles/microarch_ladder] [out.md]

Each profile (<dir>/{vit,llm}_<cfg>_raw.csv) holds the kernels of one ViT full-frame encode (9 x 4 windows) or of the
last 512-token ubatch of an 8-frame image prefill. On this board ncu stops after 482 kernels (NVMAP error in replay),
so a stock profile holds only the first layers. Layers are cut at every second RMS norm (attention norm, FFN norm)
and only complete layers are compared, the same layer indices in every configuration:
  ViT  window-attention layers 1-6 (mean per layer) and full-attention layer 7 (Qwen2.5-VL fullatt_block_indexes)
  LLM  layers 1-12 (mean per layer)
Per layer and per kernel class: GPU time, kernels, DRAM bytes (L2 fills from system memory x 32 B: on Orin the DRAM
is system memory), L2 bytes (lts__t_sectors x 32 B), instructions, achieved DRAM bandwidth, and the time-weighted
tensor-pipe activity, SM and memory throughput (% of peak), achieved occupancy and warp stall reasons.
With an out path it also writes <out>_classes.csv: time and DRAM bytes per layer, per step and kernel class.
ncu flushes the caches before each kernel, so DRAM bytes are cold-cache figures.
"""
import csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsys_stage_kernels import classify

VIT = ["base", "winbatch", "f32bias", "table", "actfuse", "ropef16", "design"]
LLM = ["base", "f32bias", "table", "actfuse", "design"]
STEP = {"base": "stock kernels", "winbatch": "+ batched window attention", "f32bias": "+ F32 output, bias in epilogue",
        "table": "+ tuned GEMM table", "actfuse": "+ GEMM-input fusion", "ropef16": "+ F16 vision RoPE",
        "design": "+ residual in GEMM (every switch)"}
DRAM_PEAK = 204.8   # GB/s, LPDDR5 256-bit at 3,200 MHz (Orin AGX 64 GB); ~156-183 GB/s measured by streaming kernels
STALLS = ["long_scoreboard", "short_scoreboard", "math_pipe_throttle", "mio_throttle", "lg_throttle", "barrier",
          "wait", "not_selected", "selected", "dispatch_stall", "no_instruction", "branch_resolving", "membar", "drain",
          "imc_miss", "tex_throttle", "misc", "sleeping"]


def num(s):
    s = (s or "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def load(path):
    """Kernels of one profile, in launch order; the kernel ncu failed on (no duration) is dropped."""
    R = list(csv.reader(open(path)))
    h, out = R[0], []
    col = {c: i for i, c in enumerate(h)}
    g = lambda r, c: num(r[col[c]]) if c in col else None
    for r in R[2:]:
        t = g(r, "gpu__time_duration.sum")
        if not t or t != t:
            continue
        name = r[col["Kernel Name"]]
        out.append(dict(
            name=name, cls=classify(name, int(g(r, "launch__grid_dim_z") or 1)), t=t,
            dram=(g(r, "lts__d_sectors_fill_sysmem.sum") or 0) * 32, l2=(g(r, "lts__t_sectors.sum") or 0) * 32,
            inst=g(r, "sm__inst_executed.sum") or 0,
            tensor=g(r, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active") or 0,
            sm=g(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed") or 0,
            mem=g(r, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed") or 0,
            occ=g(r, "sm__warps_active.avg.pct_of_peak_sustained_active") or 0,
            waves=g(r, "launch__waves_per_multiprocessor") or 0,
            stall={s: g(r, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") or 0 for s in STALLS}))
    return out


def layers(K):
    """Complete layers: the kernels from one attention RMS norm to the next (every second norm starts a layer)."""
    norms = [i for i, k in enumerate(K) if k["cls"] == "norm"]
    starts = norms[0::2]
    return [K[a:b] for a, b in zip(starts, starts[1:])]


def agg(ks):
    t = sum(k["t"] for k in ks)
    w = lambda f: sum(k[f] * k["t"] for k in ks) / t if t else 0
    return dict(t=t, n=len(ks), dram=sum(k["dram"] for k in ks), l2=sum(k["l2"] for k in ks),
                inst=sum(k["inst"] for k in ks), tensor=w("tensor"), sm=w("sm"), mem=w("mem"), occ=w("occ"),
                stall={s: sum(k["stall"][s] * k["t"] for k in ks) / t if t else 0 for s in STALLS})


def mean_layer(L, idx):
    """Mean over the given layer indices of each layer's aggregate (and per class)."""
    sel = [L[i] for i in idx if i < len(L)]
    A = [agg(l) for l in sel]
    m = {f: st.mean(a[f] for a in A) for f in ("t", "n", "dram", "l2", "inst")}
    tt = sum(a["t"] for a in A)
    for f in ("tensor", "sm", "mem", "occ"):
        m[f] = sum(a[f] * a["t"] for a in A) / tt
    classes = sorted({k["cls"] for l in sel for k in l})
    m["cls"] = {}
    for c in classes:
        ks = [k for l in sel for k in l if k["cls"] == c]
        a = agg(ks)
        a.update(t=a["t"] / len(sel), n=a["n"] / len(sel), dram=a["dram"] / len(sel), l2=a["l2"] / len(sel),
                 inst=a["inst"] / len(sel))
        m["cls"][c] = a
    m["layers"] = len(sel)
    return m


def fmt_row(label, m, base=None):
    gbs = m["dram"] / m["t"] if m["t"] else 0   # bytes / ns = GB/s
    d = f" ({100 * (m['t'] / base['t'] - 1):+.0f}%)" if base else ""
    return (f"| {label} | {m['n']:.0f} | {m['t'] / 1e3:,.0f}{d} | {m['dram'] / 1e6:.1f} | {m['l2'] / 1e6:.1f} | "
            f"{m['inst'] / 1e6:.1f} | {gbs:.0f} | {m['tensor']:.0f}% | {m['sm']:.0f}% | {m['mem']:.0f}% | {m['occ']:.0f}% |")


HEAD = ("| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | "
        "SM throughput | memory throughput | achieved occupancy |")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "experiments", "profiles",
                                                           "microarch_ladder")
    out = []
    parts = [("vit", VIT, "ViT window-attention layer (mean of layers 1-6)", range(1, 7)),
             ("vit", VIT, "ViT full-attention layer (layer 7)", [7]),
             ("llm", LLM, "LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)", range(1, 13))]
    res, rows = {}, []
    for kind, cfgs, title, idx in parts:
        out += [f"\n**{title}**\n", HEAD, "|---|" + "---|" * 10]
        base = None
        for c in cfgs:
            p = f"{d}/{kind}_{c}_raw.csv"
            if not os.path.exists(p):
                continue
            L = layers(load(p))
            if max(idx) >= len(L):
                out.append(f"| {STEP[c]} | only {len(L)} complete layers | | | | | | | | | |")
                continue
            m = mean_layer(L, idx)
            res[(title, c)] = m
            for cl, a in m["cls"].items():
                rows.append(dict(part=kind if kind == "llm" else ("vit_window" if len(idx) > 1 else "vit_full"),
                                 step=STEP[c], cfg=c, cls=cl, t_us=round(a["t"] / 1e3, 1), dram_mb=round(a["dram"] / 1e6, 2),
                                 kernels=round(a["n"], 2), tensor_pct=round(a["tensor"], 1), occ_pct=round(a["occ"], 1)))
            out.append(fmt_row(STEP[c], m, base))
            base = base or m
        # per class, stock against every switch
        b, x = res.get((title, cfgs[0])), res.get((title, "design"))
        if b and x:
            out += [f"\n*{title}: per kernel class, stock → every switch*\n",
                    "| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | "
                    "memory throughput | occupancy | top stall (warps per issue) |", "|---|" + "---|" * 9]
            def top(a):
                s = max(a["stall"], key=a["stall"].get)
                return f"{s} {a['stall'][s]:.1f}"
            cols = [lambda a: f"{a['t'] / 1e3:,.0f}", lambda a: f"{a['n']:.0f}", lambda a: f"{a['dram'] / 1e6:.1f}",
                    lambda a: f"{a['dram'] / a['t']:.0f}", lambda a: f"{a['tensor']:.0f}%", lambda a: f"{a['sm']:.0f}%",
                    lambda a: f"{a['mem']:.0f}%", lambda a: f"{a['occ']:.0f}%", top]
            for cl in sorted(set(b["cls"]) | set(x["cls"]), key=lambda c: -(b["cls"].get(c, {}).get("t", 0))):
                pair = (b["cls"].get(cl), x["cls"].get(cl))
                out.append(f"| {cl} | " + " | ".join(" → ".join(fn(a) if a else "—" for a in pair) for fn in cols) + " |")
    text = "\n".join(out) + "\n"
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text)
        with open(os.path.splitext(sys.argv[2])[0] + "_classes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
