#!/usr/bin/env python3
"""LLM prefill per projection GEMM and attention, from the Nsight Compute ladder of scripts/run_microarch_fa.sh
(the last 512-token ubatch of an 8-frame image prefill, Qwen2.5-VL-7B LLM, layers 1-12 as in microarch_ladder.py).

    python3 microarch_llm_detail.py [dir=jetson/experiments/profiles/microarch_ladder] [out.md]

GEMMs are named by their order in a layer (Q, K, V, O, gate, up, down: seven per layer in every step). Per GEMM and
step: tile, grid (N tiles x M tiles x split-K), waves, GPU time, tensor-pipe activity, L2 hit rate (lts__t_sector_hit_rate,
all L2 requests), L2 traffic, DRAM traffic and the top warp stalls. Attention (flash_attn_ext_f16): every stall reason in
warps per issue, issue activity, occupancy and its limiter.
ncu flushes the caches before each kernel, so hit rates and DRAM bytes are cold-cache figures: every operand's first
touch misses. A hit rate also falls when a kernel makes fewer L2 requests for the same misses (larger tiles reuse more
in shared memory), so it is read together with L2 and DRAM bytes.
"""
import csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microarch_ladder as M

ROLES = ["Q", "K", "V", "O", "gate", "up", "down"]
SHAPE = {"Q": "3584 x 3584", "K": "512 x 3584", "V": "512 x 3584", "O": "3584 x 3584", "gate": "18944 x 3584",
         "up": "18944 x 3584", "down": "3584 x 18944"}   # N x K of the weight; M = 512 tokens
STEPS = ["base", "table", "design"]
EXTRA = {"l2hit": "lts__t_sector_hit_rate.pct", "l1hit": "l1tex__t_sector_hit_rate.pct", "gx": "launch__grid_dim_x",
         "gy": "launch__grid_dim_y", "gz": "launch__grid_dim_z", "regs": "launch__registers_per_thread",
         "smem": "launch__shared_mem_per_block", "blk": "launch__block_size",
         "lim_reg": "launch__occupancy_limit_registers", "lim_smem": "launch__occupancy_limit_shared_mem",
         "lim_warp": "launch__occupancy_limit_warps", "lim_blk": "launch__occupancy_limit_blocks",
         "issue": "sm__issue_active.sum.pct_of_peak_sustained_elapsed",
         "lat": "smsp__average_warp_latency_per_inst_issued.ratio"}


def load(path):
    """microarch_ladder.load plus the extra columns, row for row."""
    K = M.load(path)
    R = list(csv.reader(open(path)))
    col = {c: i for i, c in enumerate(R[0])}
    rows = [r for r in R[2:] if (lambda t: t and t == t)(M.num(r[col["gpu__time_duration.sum"]]))]
    assert len(rows) == len(K)
    for k, r in zip(K, rows):
        for f, c in EXTRA.items():
            k[f] = M.num(r[col[c]]) if c in col else None
    return K


def tile(name):
    for t in ("256x128", "128x128", "128x64", "64x64"):
        if t in name:
            return t
    return "cutlass " + name.split("gemm_bf16_")[1].split("_")[0] if "gemm_bf16_" in name else "?"


def mean_k(ks):
    t = st.mean(k["t"] for k in ks)
    w = lambda f: sum(k[f] * k["t"] for k in ks) / sum(k["t"] for k in ks)
    stall = {s: w_s for s, w_s in ((s, sum(k["stall"][s] * k["t"] for k in ks) / sum(k["t"] for k in ks)) for s in M.STALLS)}
    return dict(t=t, tensor=w("tensor"), l2hit=w("l2hit"), l1hit=w("l1hit"), occ=w("occ"), issue=w("issue"),
                dram=st.mean(k["dram"] for k in ks), l2=st.mean(k["l2"] for k in ks), waves=ks[0]["waves"],
                grid=f'{ks[0]["gx"]:.0f} x {ks[0]["gy"]:.0f} x {ks[0]["gz"]:.0f}', tile=tile(ks[0]["name"]),
                stall=stall, k=ks[0])


def top(stall, n=3):
    return ", ".join(f"{s} {v:.1f}" for s, v in sorted(stall.items(), key=lambda x: -x[1])[:n])


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "experiments", "profiles",
                                                           "microarch_ladder")
    gem, att = {}, {}
    for s in STEPS:
        L = M.layers(load(f"{d}/llm_{s}_raw.csv"))[1:13]
        G = [[k for k in l if k["cls"].startswith("GEMM")] for l in L]
        assert all(len(g) == 7 for g in G), [len(g) for g in G]
        for i, r in enumerate(ROLES):
            gem[(s, r)] = mean_k([g[i] for g in G])
        att[s] = mean_k([k for l in L for k in l if k["cls"] == "attention"])
    out = ["**LLM prefill GEMMs per projection** (M = 512 tokens, mean of layers 1-12; split-K = grid z)\n",
           "| projection (N x K) | step | tile | grid N x M x split | waves | GPU time µs | tensor pipe | L2 hit | "
           "L2 MB | DRAM MB | top stalls (warps per issue) |", "|---|" + "---|" * 10]
    for r in ROLES:
        for s in STEPS:
            g = gem[(s, r)]
            out.append(f"| {r} ({SHAPE[r]}) | {s} | {g['tile']} | {g['grid']} | {g['waves']:.1f} | {g['t'] / 1e3:,.0f} | "
                       f"{g['tensor']:.0f}% | {g['l2hit']:.1f}% | {g['l2'] / 1e6:.0f} | {g['dram'] / 1e6:.0f} | {top(g['stall'])} |")
    out += ["\n**LLM prefill attention** (flash_attn_ext_f16, one kernel per layer, mean of layers 1-12)\n",
            "| step | GPU time µs | tensor pipe | issue active | achieved occupancy | regs/thread | smem/block KB | "
            "occupancy limit (blocks/SM: regs, smem, warps) | L1 hit | L2 hit | L2 MB | DRAM MB | DRAM GB/s |",
            "|---|" + "---|" * 12]
    for s in STEPS:
        a, k = att[s], att[s]["k"]
        out.append(f"| {s} | {a['t'] / 1e3:,.0f} | {a['tensor']:.0f}% | {a['issue']:.0f}% | {a['occ']:.0f}% | {k['regs']:.0f} | "
                   f"{k['smem'] / 1024:.0f} | {k['lim_reg']:.0f}, {k['lim_smem']:.0f}, {k['lim_warp']:.0f} | {a['l1hit']:.1f}% | "
                   f"{a['l2hit']:.1f}% | {a['l2'] / 1e6:.0f} | {a['dram'] / 1e6:.0f} | {a['dram'] / a['t']:.0f} |")
    out += ["\n*Attention warp stalls (warps per issue, base → design)*\n", "| stall reason | base | design |", "|---|---|---|"]
    for s_, v in sorted(att["base"]["stall"].items(), key=lambda x: -x[1]):
        if v >= 0.05 or att["design"]["stall"][s_] >= 0.05:
            out.append(f"| {s_} | {v:.2f} | {att['design']['stall'][s_]:.2f} |")
    out.append(f"| **sum** | {sum(att['base']['stall'].values()):.2f} | {sum(att['design']['stall'].values()):.2f} |")
    text = "\n".join(out) + "\n"
    print(text)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text)


if __name__ == "__main__":
    main()
