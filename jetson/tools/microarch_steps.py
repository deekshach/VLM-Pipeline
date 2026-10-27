#!/usr/bin/env python3
"""Bottleneck before and after each kernel switch, from the Nsight Compute ladder of scripts/run_microarch_fa.sh
(the same profiles and layers as microarch_ladder.py). For every step of the ladder, the kernel classes whose time
per layer changes by at least 0.5% of the stock layer are compared between the step before and the step itself:

    python3 microarch_steps.py [dir=jetson/experiments/profiles/microarch_ladder] [out.md]

Per class and side: kernels and GPU time per layer, kernel names and launch shape, speed of light (SM and memory
throughput, % of peak), issue slots used, tensor-pipe activity, achieved and theoretical occupancy with what limits
it, L1 and L2 hit rates, DRAM and L2 bytes and the achieved DRAM rate, instructions, and the top warp stalls
(warps per issue). Every figure is the time-weighted mean over the class's kernels, summed over the layer where it
is an amount. ncu flushes the caches before each kernel, so hit rates and DRAM bytes are cold-cache figures.
"""
import csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microarch_ladder as M

EXTRA = {"issue": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
         "l1hit": "l1tex__t_sector_hit_rate.pct", "l2hit": "lts__t_sector_hit_rate.pct",
         "l2tp": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
         "lsu": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
         "fma": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
         "alu": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
         "lat": "smsp__average_warp_latency_per_inst_issued.ratio",
         "theo": "sm__maximum_warps_per_active_cycle_pct",
         "lim_reg": "launch__occupancy_limit_registers", "lim_smem": "launch__occupancy_limit_shared_mem",
         "lim_warp": "launch__occupancy_limit_warps", "lim_blk": "launch__occupancy_limit_blocks",
         "gx": "launch__grid_dim_x", "gy": "launch__grid_dim_y", "gz": "launch__grid_dim_z", "blk": "launch__block_size"}
PARTS = [("vit", M.VIT, "ViT window-attention layer (mean of layers 1-6)", range(1, 7)),
         ("llm", M.LLM, "LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)", range(1, 13))]


def load(path):
    K = M.load(path)
    R = list(csv.reader(open(path)))
    col = {c: i for i, c in enumerate(R[0])}
    rows = [r for r in R[2:] if (lambda t: t and t == t)(M.num(r[col["gpu__time_duration.sum"]]))]
    for k, r in zip(K, rows):
        for f, c in EXTRA.items():
            k[f] = (M.num(r[col[c]]) if c in col else None) or 0
    return K


def short(name):
    n = name.split("(")[0].replace("void ", "")
    return n.split("<")[0][:48]


def sig(ks, nl):
    """Signature of one class over the selected layers (nl of them)."""
    t = sum(k["t"] for k in ks)
    w = lambda f: sum(k[f] * k["t"] for k in ks) / t
    lim = {}
    for k in ks:   # the binding limit (fewest blocks per SM) of the class's longest kernel shape
        L = {"registers": k["lim_reg"], "shared memory": k["lim_smem"], "warps": k["lim_warp"], "blocks": k["lim_blk"]}
        lim[min(L, key=L.get)] = lim.get(min(L, key=L.get), 0) + k["t"]
    stall = {s: sum(k["stall"][s] * k["t"] for k in ks) / t for s in M.STALLS}
    names = {}
    for k in ks:
        names[short(k["name"])] = names.get(short(k["name"]), 0) + k["t"]
    big = max(ks, key=lambda k: k["t"])
    return dict(n=len(ks) / nl, t=t / nl, sm=w("sm"), mem=w("mem"), issue=w("issue"), tensor=w("tensor"), occ=w("occ"),
                theo=w("theo"), l1hit=w("l1hit"), l2hit=w("l2hit"), l2tp=w("l2tp"), lsu=w("lsu"), fma=w("fma"),
                alu=w("alu"), lat=w("lat"), dram=sum(k["dram"] for k in ks) / nl, l2=sum(k["l2"] for k in ks) / nl,
                inst=sum(k["inst"] for k in ks) / nl, lim=max(lim, key=lim.get), stall=stall,
                names=sorted(names, key=lambda x: -names[x])[:2],
                shape=f'{big["gx"]:.0f}x{big["gy"]:.0f}x{big["gz"]:.0f} x {big["blk"]:.0f} thr, {big["waves"]:.1f} waves')


def fmt(s):
    if not s:
        return {}
    top = ", ".join(f"{k} {v:.1f}" for k, v in sorted(s["stall"].items(), key=lambda x: -x[1])[:3])
    return {"kernels / layer": f"{s['n']:.1f}", "GPU time µs / layer": f"{s['t'] / 1e3:,.0f}",
            "kernel (largest)": " / ".join(s["names"]), "launch (largest)": s["shape"],
            "SM throughput": f"{s['sm']:.0f}%", "memory throughput": f"{s['mem']:.0f}%", "L2 throughput": f"{s['l2tp']:.0f}%",
            "issue slots used": f"{s['issue']:.0f}%", "tensor pipe": f"{s['tensor']:.0f}%",
            "FMA / ALU / LSU pipe": f"{s['fma']:.0f}% / {s['alu']:.0f}% / {s['lsu']:.0f}%",
            "occupancy achieved / theoretical": f"{s['occ']:.0f}% / {s['theo']:.0f}% ({s['lim']}-limited)",
            "warp latency per issue (cycles)": f"{s['lat']:.1f}", "L1 / L2 hit": f"{s['l1hit']:.0f}% / {s['l2hit']:.0f}%",
            "DRAM MB / layer": f"{s['dram'] / 1e6:.1f}", "DRAM GB/s": f"{s['dram'] / s['t']:.0f}",
            "L2 MB / layer": f"{s['l2'] / 1e6:.1f}", "instructions M / layer": f"{s['inst'] / 1e6:.1f}",
            "top stalls (warps per issue)": top}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "experiments", "profiles",
                                                           "microarch_ladder")
    out = []
    for kind, cfgs, title, idx in PARTS:
        prof = {}
        for c in cfgs:
            p = f"{d}/{kind}_{c}_raw.csv"
            if os.path.exists(p):
                L = M.layers(load(p))
                prof[c] = [L[i] for i in idx]
        base_t = sum(k["t"] for l in prof[cfgs[0]] for k in l) / len(idx)
        out.append(f"\n## {title}\n\nStock layer: {base_t / 1e3:,.0f} µs.")
        for a, b in zip(cfgs, cfgs[1:]):
            if a not in prof or b not in prof:
                continue
            cls = lambda c, x: [k for l in prof[c] for k in l if k["cls"] == x]
            allc = sorted({k["cls"] for c in (a, b) for l in prof[c] for k in l})
            ta = sum(k["t"] for l in prof[a] for k in l) / len(idx)
            tb = sum(k["t"] for l in prof[b] for k in l) / len(idx)
            out.append(f"\n### {M.STEP[b]} (layer {ta / 1e3:,.0f} → {tb / 1e3:,.0f} µs, {100 * (tb / ta - 1):+.1f}%)\n")
            for x in allc:
                ka, kb = cls(a, x), cls(b, x)
                d_t = (sum(k["t"] for k in kb) - sum(k["t"] for k in ka)) / len(idx)
                if abs(d_t) < 0.005 * base_t:
                    continue
                fa, fb = fmt(sig(ka, len(idx)) if ka else None), fmt(sig(kb, len(idx)) if kb else None)
                out += [f"*{x}: {d_t / 1e3:+,.0f} µs per layer*\n", "| metric | before | after |", "|---|---|---|"]
                for m in (fa or fb):
                    out.append(f"| {m} | {fa.get(m, '—')} | {fb.get(m, '—')} |")
                out.append("")
    text = "\n".join(out) + "\n"
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text)


if __name__ == "__main__":
    main()
