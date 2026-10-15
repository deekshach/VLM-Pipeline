#!/usr/bin/env python3
"""Compare the counters of jetson/scripts/run_microarch_ncu.sh across switch configurations.

For every report (<name>_raw.csv, exported with --print-units base) and kernel class: total kernel
time, DRAM read traffic (L2 sectors filled from system memory x 32 B), L2 requests and hit rate,
achieved occupancy, tensor-pipe utilisation and the main warp-stall reasons, each time-weighted.

    python3 ncu_compare.py <dir with *_raw.csv>
"""
import collections, csv, glob, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "."
TIME_SCALE = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6}
STALLS = ["long_scoreboard", "short_scoreboard", "wait", "mio_throttle", "lg_throttle", "math_pipe_throttle",
          "barrier", "not_selected", "dispatch_stall", "no_instruction", "branch_resolving", "drain", "tex_throttle"]


def classify(name, gz):
    n = name.lower()
    if "flash_attn" in n: return "attention"
    if "soft_max" in n: return "softmax"
    if any(k in n for k in ("gemm", "kernel2", "mul_mat", "cutlass", "xmma", "splitk")):
        if "splitk" in n or "reduce" in n: return "GEMM split-K reduce"
        if "bf16" not in n and "bfloat16" not in n: return "attn matmul"
        return "GEMM split-K" if gz > 1 else "GEMM"
    if "convert" in n: return "cast"
    if "bin_bcast" in n: return "add"
    if "norm" in n: return "norm"
    if "rope" in n: return "rope"
    if "cpy" in n or "set_rows" in n or "get_rows" in n: return "copy"
    if "glu" in n or "gated" in n or "gelu" in n: return "glu"
    return "other"


def load(path):
    rows = list(csv.reader(open(path)))
    if len(rows) < 3:
        return {}
    h, units = rows[0], rows[1]
    ix = {c: i for i, c in enumerate(h)}

    def num(r, c, default=0.0):
        if c not in ix: return default
        try: return float(r[ix[c]].replace(",", ""))
        except ValueError: return default

    out = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows[2:]:
        grid = r[ix["Grid Size"]] if "Grid Size" in ix else "(1, 1, 1)"
        gz = int(grid.strip("()").split(",")[2]) if grid.count(",") == 2 else 1
        cls = classify(r[ix["Kernel Name"]], gz)
        t = num(r, "gpu__time_duration.sum") * TIME_SCALE.get(units[ix["gpu__time_duration.sum"]], 1.0)
        o = out[cls]
        o["n"] += 1
        o["us"] += t
        o["dram_mb"] += num(r, "lts__d_sectors_fill_sysmem.sum") * 32 / 1e6
        o["l2_req_mb"] += num(r, "lts__t_sectors.sum") * 32 / 1e6
        for key, col in (("l2_hit", "lts__t_sector_hit_rate.pct"), ("occ", "sm__warps_active.avg.pct_of_peak_sustained_active"),
                         ("tensor", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                         ("mem", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")):
            o[key] += num(r, col) * t   # time-weighted, divided by o["us"] below
        stall = {s: num(r, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
        tot = sum(stall.values()) or 1.0
        for s, v in stall.items():
            o["stall_" + s] += v / tot * t
    return out


def main():
    reports = sorted(glob.glob(os.path.join(D, "*_raw.csv")))
    if not reports:
        print(__doc__); sys.exit(1)
    data = {os.path.basename(p)[:-8]: load(p) for p in reports}
    groups = collections.OrderedDict()
    for name in data:   # vit_<cfg>_<shape> -> group by shape; gemm_down_<algo> -> one group
        key = name.rsplit("_", 1)[1] if name.startswith("vit_") else "gemm_down"
        groups.setdefault(key, []).append(name)
    for g, names in groups.items():
        print(f"\n## {'ViT, ' + g + ' windows' if g != 'gemm_down' else 'prefill down projection 3584x18944, 512 tokens'}")
        print("| config | class | kernels | time us | DRAM read MB | L2 requests MB | L2 hit | occupancy | tensor % | mem % | top stalls |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name in names:
            cfg = name.split("_")[1] if name.startswith("vit_") else name.split("_")[-1]
            classes = sorted(data[name].items(), key=lambda kv: -kv[1]["us"])
            for cls, o in classes:
                w = o["us"] or 1.0
                st = sorted(((k[6:], v / w * 100) for k, v in o.items() if k.startswith("stall_")), key=lambda x: -x[1])[:2]
                print(f"| {cfg} | {cls} | {o['n']:.0f} | {o['us']:.0f} | {o['dram_mb']:.1f} | {o['l2_req_mb']:.1f} | "
                      f"{o['l2_hit'] / w:.0f}% | {o['occ'] / w:.0f}% | {o['tensor'] / w:.0f} | {o['mem'] / w:.0f} | "
                      + ", ".join(f"{s} {v:.0f}%" for s, v in st) + " |")
            tot = {k: sum(o[k] for o in data[name].values()) for k in ("us", "dram_mb", "l2_req_mb")}
            print(f"| {cfg} | **all** | | **{tot['us']:.0f}** | **{tot['dram_mb']:.1f}** | **{tot['l2_req_mb']:.1f}** | | | | | |")
            # A fixed launch window covers more layers when a switch removes kernels: without flash
            # attention every layer has exactly one softmax, so report per layer as well.
            n_layers = data[name].get("softmax", {}).get("n", 0)
            if n_layers:
                print(f"| {cfg} | **per layer** ({n_layers:.0f} layers) | | **{tot['us'] / n_layers:.0f}** | "
                      f"**{tot['dram_mb'] / n_layers:.1f}** | **{tot['l2_req_mb'] / n_layers:.1f}** | | | | | |")


if __name__ == "__main__":
    main()
