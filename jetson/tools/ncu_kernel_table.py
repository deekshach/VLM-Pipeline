#!/usr/bin/env python3
"""Per-kernel Nsight Compute summary (exported with `ncu --import X.ncu-rep --page raw --csv`):
duration, registers, occupancy and its limiter, SM / tensor / memory throughput, L1 and L2 hit
rates and the top warp-stall reasons, with launches of the same kernel and role grouped (median).

    python3 ncu_kernel_table.py raw.csv
"""
import csv, re, statistics as st, collections, sys
r = list(csv.reader(open(sys.argv[1])))
h = r[0]; ix = {c: i for i, c in enumerate(h)}
def g(row, c):
    v = row[ix[c]].replace(',', '')
    try: return float(v)
    except: return v
NAMES = [("flash_attn", "attention"), ("soft_max", "softmax"), ("Kernel2", "GEMM (cutlass)"), ("gemm", "GEMM (cuBLAS)"),
         ("rope_vision", "RoPE"), ("rms_norm", "RMSNorm"), ("cpy_scalar", "K/V -> F16 copy"),
         ("convert_unary", "cast F32->BF16"), ("unary_gated", "SwiGLU"), ("k_bin_bcast", "add")]
def role(name, grid):
    if "convert_unary<" in name:   # direction from the template arguments <src, dst>
        src = name.split("convert_unary<", 1)[1]
        return ("cast F32->BF16" if src.startswith("float") else "cast BF16->F32" if src.startswith("__nv_bfloat16") else "cast") + f" grid {grid}"
    for k, v in NAMES:
        if k in name:
            if v == "add":
                return f"add grid {grid}"
            if v.startswith("GEMM") or v.startswith("cast"):
                return f"{v} grid {grid}"
            return v
    return name[:30]
stalls = [c for c in h if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]
groups = collections.OrderedDict()
for row in r[2:]:
    k = role(row[ix["Kernel Name"]], row[ix["Grid Size"]])
    groups.setdefault(k, []).append(row)
M = lambda rows, c: st.median(g(x, c) for x in rows)
out = []
for k, rows in groups.items():
    d = dict(role=k, n=len(rows), us=M(rows, "gpu__time_duration.sum") * 1000,
             regs=M(rows, "launch__registers_per_thread"),
             occ_th=M(rows, "sm__maximum_warps_per_active_cycle_pct"),
             occ_ach=M(rows, "sm__warps_active.avg.pct_of_peak_sustained_active"),
             waves=M(rows, "launch__waves_per_multiprocessor"),
             sm=M(rows, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
             tensor=M(rows, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
             mem=M(rows, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
             l1=M(rows, "l1tex__t_sector_hit_rate.pct"), l2=M(rows, "lts__t_sector_hit_rate.pct"),
             fill=M(rows, "lts__d_sectors_fill_sysmem.avg.pct_of_peak_sustained_elapsed"),
             cpi=M(rows, "smsp__average_warp_latency_per_inst_issued.ratio") if "smsp__average_warp_latency_per_inst_issued.ratio" in ix else float('nan'))
    lim = {c: M(rows, c) for c in ("launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks")}
    lo = min(lim.values())
    d["limiter"] = "/".join(k.split("_limit_")[1] for k, v in lim.items() if v == lo)
    s = {c.split("stalled_")[1].split("_per_issue")[0]: M(rows, c) for c in stalls}
    s.pop("selected", None)
    tot = sum(s.values())
    d["stalls"] = ", ".join(f"{n} {v/tot*100:.0f}%" for n, v in sorted(s.items(), key=lambda x: -x[1])[:3])
    out.append(d)
print("| kernel | n | µs | regs/thr | occ theor→achieved | limiter | waves/SM | SM % | tensor % | mem % | L1 hit | L2 hit | top stalls |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in out:
    print(f"| {d['role']} | {d['n']} | {d['us']:.0f} | {d['regs']:.0f} | {d['occ_th']:.0f}→{d['occ_ach']:.0f}% | {d['limiter']} | {d['waves']:.2f} | {d['sm']:.0f} | {d['tensor']:.0f} | {d['mem']:.0f} | {d['l1']:.0f}% | {d['l2']:.0f}% | {d['stalls']} |")
