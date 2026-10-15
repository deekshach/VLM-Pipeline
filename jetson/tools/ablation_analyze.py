#!/usr/bin/env python3
"""Analyse jetson/scripts/run_ablation.sh: attribute each microarchitectural change to the switch causing it.

    python3 ablation_analyze.py [ablation dir]        # report (markdown) from nsys/, ncu/ and warm/
    python3 ablation_analyze.py --windows <nsys dir>  # ncu launch windows (used by run_ablation.sh nsys)

Composition. A ViT frame is 28 windowed-attention layers + 4 global layers (7, 15, 23, 31). Without window
batching both kinds run the same dense masked attention; with it, windowed layers run per-window attention
and global layers stay dense. ncu runs capture two whole layers around global layer 7, so per frame:
    frame = 32 x rest-of-layer + 28 x windowed attention + 4 x global attention
(the patch embedding and merger, < 2% of the ViT, are left out). nsys traces cover whole frames.
The LLM prefill: ncu captures two whole layers of the last 512-token ubatch; x 28 = one ubatch.

Attribution. With switches W, F, B (needs F), T and C (window-cache crop: 9x4 -> 5x2 windows), every valid
subset is a config; a switch's Shapley value is its marginal effect averaged over all orders in which the
switches can be turned on (B after F). Values add up to the total change from base to all-on, and a large
gap between a switch's "alone" effect and its Shapley value means it interacts with another switch.
"""
import collections, csv, glob, itertools, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_DIR = os.path.join(HERE, "..", "experiments", "profiles", "ablation")
TIME_SCALE = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3, "second": 1e6}
STALLS = ["long_scoreboard", "short_scoreboard", "wait", "mio_throttle", "lg_throttle", "math_pipe_throttle",
          "barrier", "not_selected", "dispatch_stall", "no_instruction", "branch_resolving", "drain", "tex_throttle"]
CLASSES = ["attention", "softmax", "attn matmul", "GEMM", "GEMM split-K", "cast", "add", "norm", "rope", "copy",
           "glu", "other"]
VIT_CFGS = "base T F FT FB FBT W WT WF WFT WFB WFBT".split()
LLM_CFGS = "base T F FT FB FBT".split()
N_LAYERS_VIT, N_GLOBAL, N_LAYERS_LLM = 32, 4, 28


def classify(name, gz):
    n = name.lower()
    if "flash_attn" in n: return "attention"
    if "soft_max" in n: return "softmax"
    if "batched_ptrs" in n: return "attn matmul"   # pointer setup of the batched attention products
    if any(k in n for k in ("gemm", "kernel2", "mul_mat", "cutlass", "xmma", "splitk")):
        if "bf16" not in n and "bfloat16" not in n: return "attn matmul"
        return "GEMM split-K" if gz > 1 else "GEMM"
    if "convert" in n: return "cast"
    if "bin_bcast" in n: return "add"
    if "norm" in n: return "norm"
    if "rope" in n: return "rope"
    if "cpy" in n or "set_rows" in n or "get_rows" in n or "concat" in n: return "copy"
    if "glu" in n or "gated" in n or "gelu" in n: return "glu"
    return "other"


def is_attn(k): return k["cls"] in ("softmax", "attn matmul")


def batched(k):
    """Per-window (batched) attention kernel vs dense: the batched products and softmax run one
    grid slice per (head, window), i.e. gridZ > 16 heads, or the softmax specialised for 64 columns."""
    return k["gz"] > 16 or "<1, 64, 64" in k["name"] or "batched_ptrs" in k["name"]


# ---------------------------------------------------------------- nsys
def nsys_instances(path):
    """{range name: [[kernel dicts in launch order] per range instance]} for the probe ranges (vit_*, prefill_*)."""
    db = sqlite3.connect(path)
    S = dict(db.execute("select id, value from StringIds"))
    db.execute("create temp table r as select rowid as rid, coalesce(text, (select value from StringIds where id = textId)) "
               "as name, start, end, globalTid from NVTX_EVENTS where end is not null")
    db.execute("delete from r where substr(name, 1, 4) != 'vit_' and substr(name, 1, 8) != 'prefill_'")
    out = collections.defaultdict(list)
    inst = {}
    for rid, name in db.execute("select rid, name from r order by start"):
        inst[rid] = []
        out[name].append(inst[rid])
    q = ("select r.rid, k.start, k.end, k.demangledName, k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join CUPTI_ACTIVITY_KIND_RUNTIME a on a.correlationId = k.correlationId "
         "join r on r.globalTid = a.globalTid and a.start between r.start and r.end order by k.start")
    for rid, ks, ke, sn, gz in db.execute(q):
        inst[rid].append({"name": S[sn], "gz": gz, "cls": classify(S[sn], gz), "us": (ke - ks) / 1e3})
    return out


def nsys_ranges(path):
    """{range name: kernels of all instances, 'us' divided by the number of instances (a mean instance)}."""
    out = {}
    for name, insts in nsys_instances(path).items():
        n = len(insts)
        out[name] = [dict(k, us=k["us"] / n) for ks in insts for k in ks]
    return out


def windows(nsys_dir):
    print("# run nvtx_range launch_skip launch_count  (two whole layers; ViT: layers 6-7, 7 = global)")
    for path in sorted(glob.glob(os.path.join(nsys_dir, "*.sqlite"))):
        kind, cfg = os.path.basename(path)[:-7].split("_", 1)
        R = nsys_instances(path)
        for rng, insts in sorted(R.items()):
            ks = insts[-1]   # the last repetition is the one marked without PROBE_MARK_ALL (ncu)
            if kind == "vit":
                s = [i for i, k in enumerate(ks) if k["cls"] == "softmax"]
                if len(s) < 10: continue
                skip = s[6] - 3   # the KQ product (and batched pointer setup) before layer 6's softmax
                count = s[8] + 3 - skip
                print(f"vit_{cfg}_{rng[4:]} {rng} {skip} {count}")
            elif rng.startswith("prefill_last_"):
                n = int(rng.rsplit("_", 1)[1])
                if cfg != "base" and n != 4608: continue
                f = [i for i, k in enumerate(ks) if k["cls"] == "attention" and "fixup" not in k["name"] and "combine" not in k["name"]]
                if len(f) < 13: continue
                print(f"llm_{cfg}_{n} {rng} {f[10]} {f[12] - f[10]}")


# ---------------------------------------------------------------- ncu
def ncu_kernels(path):
    rows = list(csv.reader(open(path)))
    if len(rows) < 3: return []
    h, units = rows[0], rows[1]
    ix = {c: i for i, c in enumerate(h)}

    def num(r, c):
        if c not in ix: return 0.0
        try: return float(r[ix[c]].replace(",", ""))
        except ValueError: return 0.0
    ks = []
    for r in rows[2:]:
        grid = r[ix["Grid Size"]] if "Grid Size" in ix else "(1, 1, 1)"
        gz = int(grid.strip("()").split(",")[2]) if grid.count(",") == 2 else 1
        name = r[ix["Kernel Name"]]
        k = {"name": name, "gz": gz, "cls": classify(name, gz),
             "us": num(r, "gpu__time_duration.sum") * TIME_SCALE.get(units[ix["gpu__time_duration.sum"]], 1.0),
             "dram": num(r, "lts__d_sectors_fill_sysmem.sum") * 32 / 1e6,
             "l2req": num(r, "lts__t_sectors.sum") * 32 / 1e6,
             "hit": num(r, "lts__t_sector_hit_rate.pct"),
             "occ": num(r, "sm__warps_active.avg.pct_of_peak_sustained_active"),
             "tensor": num(r, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")}
        st = {s: num(r, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
        tot = sum(st.values()) or 1.0
        k["stall"] = {s: v / tot for s, v in st.items()}
        ks.append(k)
    return ks


SUMS = ("us", "dram", "l2req")
WEIGHTED = ("hit", "occ", "tensor")


class Agg:
    """Sum of extensive metrics and time-weighted intensive metrics over a set of kernels, scaled."""
    def __init__(self): self.v = collections.defaultdict(float); self.stall = collections.defaultdict(float)

    def add(self, ks, scale=1.0):
        for k in ks:
            w = k["us"] * scale
            for m in SUMS: self.v[m] += k.get(m, 0.0) * scale
            for m in WEIGHTED: self.v[m + "_w"] += k.get(m, 0.0) * w
            for s, f in k.get("stall", {}).items(): self.stall[s] += f * w
        return self

    def get(self, m):
        if m in SUMS: return self.v[m]
        return self.v[m + "_w"] / self.v["us"] if self.v["us"] else 0.0

    def top_stall(self):
        if not self.stall: return ""
        s, v = max(self.stall.items(), key=lambda x: x[1])
        return f"{s} {v / self.v['us'] * 100:.0f}%" if self.v["us"] else ""


def vit_frame(ks):
    """{class: Agg per frame} from a two-layer ncu window (see module doc), plus 'all'."""
    s = [i for i, k in enumerate(ks) if k["cls"] == "softmax"]
    if len(s) < 3: return None
    lo, hi = s[0], s[2]                        # two complete layer periods
    per_layer = 2                              # layers in [lo, hi)
    rest = [k for k in ks[lo:hi] if not is_attn(k)]
    attn = [k for k in ks if is_attn(k)]
    dense = [k for k in attn if not batched(k)]
    bat = [k for k in attn if batched(k)]
    n_dense_sm = sum(1 for k in dense if k["cls"] == "softmax") or 1
    n_bat_sm = sum(1 for k in bat if k["cls"] == "softmax")
    out = collections.defaultdict(Agg)
    for k in rest: out[k["cls"]].add([k], N_LAYERS_VIT / per_layer)
    if n_bat_sm:     # windowed layers batched, global dense
        for k in bat: out[k["cls"]].add([k], (N_LAYERS_VIT - N_GLOBAL) / n_bat_sm)
        for k in dense: out[k["cls"]].add([k], N_GLOBAL / n_dense_sm)
    else:
        for k in dense: out[k["cls"]].add([k], N_LAYERS_VIT / n_dense_sm)
    tot = Agg()
    for k in rest: tot.add([k], N_LAYERS_VIT / per_layer)
    if n_bat_sm:
        tot.add(bat, (N_LAYERS_VIT - N_GLOBAL) / n_bat_sm); tot.add(dense, N_GLOBAL / n_dense_sm)
    else:
        tot.add(dense, N_LAYERS_VIT / n_dense_sm)
    out["all"] = tot
    return out


def llm_ubatch(ks):
    f = [i for i, k in enumerate(ks) if k["cls"] == "attention" and "fixup" not in k["name"] and "combine" not in k["name"]]
    lo, hi = (f[0], f[2]) if len(f) >= 3 else (0, len(ks))
    sel = ks[lo:hi]
    out = collections.defaultdict(Agg)
    for k in sel: out[k["cls"]].add([k], N_LAYERS_LLM / 2)
    out["all"] = Agg().add(sel, N_LAYERS_LLM / 2)
    return out


# ---------------------------------------------------------------- attribution
def shapley(value, factors, valid):
    """value(frozenset) -> float. Average marginal contribution over all valid orderings."""
    contrib = collections.defaultdict(float); n = 0
    for order in itertools.permutations(factors):
        cur = frozenset()
        if not all(valid(frozenset(order[:i + 1])) for i in range(len(order))): continue
        n += 1
        for f in order:
            nxt = cur | {f}
            contrib[f] += value(nxt) - value(cur)
            cur = nxt
    return {f: v / n for f, v in contrib.items()}


def cfg_set(c): return frozenset() if c == "base" else frozenset(c)


def set_cfg(s):
    c = "".join(x for x in "WFBT" if x in s)
    return c or "base"


def valid(s): return "B" not in s or "F" in s


def fmt(v, m):
    return f"{v / 1e3:.1f}" if m == "us" else f"{v:.0f}"


UNIT = {"us": "ms", "dram": "MB DRAM read", "l2req": "MB L2 requests"}


def memo(get):
    cache = {}
    def f(s):
        if s not in cache: cache[s] = get(s)
        return cache[s]
    return f


def attribution_table(get, factors, title, metrics=("us", "dram", "l2req")):
    """get(config set) -> {class: Agg} or None."""
    get = memo(get)
    print(f"\n### {title}\n")
    print("| metric | base | all on | " + " | ".join(f"{f} alone" for f in factors) + " | " +
          " | ".join(f"{f} Shapley" for f in factors) + " |")
    print("|---|---|---|" + "---|" * (2 * len(factors)))
    full = frozenset(factors)
    for m in metrics:
        val = lambda s: get(s)["all"].get(m)
        try:
            base, top = val(frozenset()), val(full)
            sh = shapley(val, factors, valid)
        except (TypeError, KeyError):
            continue
        alone = [val(frozenset({f})) - base if valid(frozenset({f})) else None for f in factors]
        print(f"| {UNIT[m]} | {fmt(base, m)} | {fmt(top, m)} ({(top / base - 1) * 100:+.0f}%) | " +
              " | ".join("-" if a is None else f"{fmt(a, m)}" for a in alone) + " | " +
              " | ".join(f"{fmt(sh[f], m)} ({sh[f] / (top - base) * 100:.0f}%)" if top != base else "-" for f in factors) + " |")


def class_shapley(get, factors, title, m):
    """Per kernel class: Shapley value of each switch on metric m. Shows which switch moves which class."""
    get = memo(get)
    print(f"\n### {title}: {UNIT[m]} per kernel class, Shapley value of each switch\n")
    print("| class | base | all on | " + " | ".join(factors) + " |")
    print("|---|---|---|" + "---|" * len(factors))
    for c in CLASSES:
        def val(s, c=c):
            d = get(s); return d[c].get(m) if c in d else 0.0
        try:
            base, top = val(frozenset()), val(frozenset(factors))
            sh = shapley(val, factors, valid)
        except (TypeError, KeyError):
            continue
        if max(abs(base), abs(top)) < (500 if m == "us" else 5): continue
        print(f"| {c} | {fmt(base, m)} | {fmt(top, m)} | " + " | ".join(fmt(sh[f], m) for f in factors) + " |")


def intensive_table(rows, title):
    """rows: [(label, {class: Agg})]; L2 hit, tensor pipe, occupancy, top stall of the main classes."""
    print(f"\n### {title}\n")
    print("| config | class | ms | DRAM MB | L2 hit | tensor % | occupancy | top stall |")
    print("|---|---|---|---|---|---|---|---|")
    for label, d in rows:
        if d is None: continue
        for c in ["softmax", "attn matmul", "attention", "GEMM", "GEMM split-K", "cast", "add", "all"]:
            if c not in d or d[c].get("us") < 200: continue
            a = d[c]
            print(f"| {label} | {c} | {a.get('us') / 1e3:.1f} | {a.get('dram'):.0f} | {a.get('hit'):.0f}% | "
                  f"{a.get('tensor'):.0f} | {a.get('occ'):.0f}% | {a.top_stall()} |")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--windows":
        windows(sys.argv[2]); return
    D = sys.argv[1] if len(sys.argv) > 1 else DEF_DIR
    print("# Switch ablation\n\nGenerated by `jetson/tools/ablation_analyze.py`; method in its docstring. "
          "W = batched windows, F = GEMM F32 output, B = bias fusion (needs F), T = tuned GEMM table, "
          "C = window-cache crop (full frame 9x4 -> 5x2 windows). ViT flash attention off.")

    # ---- nsys: real clocks, whole frames
    ns = {}
    for p in glob.glob(os.path.join(D, "nsys", "*.sqlite")):
        ns[os.path.basename(p)[:-7]] = nsys_ranges(p)
    if ns:
        print("\n## 1. Real-clock kernel time (nsys, whole frame / whole prefill)")

        def vit_ns(s):
            shape = "5x2" if "C" in s else "9x4"
            ks = ns.get("vit_" + set_cfg(s - {"C"}), {}).get("vit_" + shape)
            if not ks: return None
            d = collections.defaultdict(Agg)
            for k in ks: d[k["cls"]].add([k])
            d["all"] = Agg().add(ks)
            return d
        print("\nViT per frame, ms:\n\n| config | 9x4 | 5x2 | 1x2 |\n|---|---|---|---|")
        for c in VIT_CFGS:
            r = ns.get("vit_" + c, {})
            print(f"| {c} | " + " | ".join(f"{sum(k['us'] for k in r.get('vit_' + sh, [])) / 1e3:.1f}" for sh in ("9x4", "5x2", "1x2")) + " |")
        attribution_table(vit_ns, list("WFBTC"), "ViT kernel time per frame: attribution", ("us",))
        class_shapley(vit_ns, list("WFBTC"), "ViT (nsys)", "us")
        print("\nLLM prefill, ms (whole prefill / last 512-token ubatch):\n\n| config | " +
              " | ".join(f"N={n}" for n in (1152, 2304, 4608)) + " |\n|---|---|---|---|")
        for c in LLM_CFGS:
            r = ns.get("llm_" + c, {})
            cells = []
            for n in (1152, 2304, 4608):
                a = sum(k["us"] for k in r.get(f"prefill_{n}", [])) / 1e3
                l = sum(k["us"] for k in r.get(f"prefill_last_{n}", [])) / 1e3
                cells.append(f"{a:.0f} / {l:.0f}")
            print(f"| {c} | " + " | ".join(cells) + " |")
        r = ns.get("llm_base", {})
        if r:
            print("\nLast ubatch by class vs KV length (base), ms:\n\n| class | " + " | ".join(f"N={n}" for n in (1152, 2304, 4608)) + " |\n|---|---|---|---|")
            for c in CLASSES:
                v = [sum(k["us"] for k in r.get(f"prefill_last_{n}", []) if k["cls"] == c) / 1e3 for n in (1152, 2304, 4608)]
                if max(v) >= 1: print(f"| {c} | " + " | ".join(f"{x:.1f}" for x in v) + " |")

        def llm_ns(s):
            ks = ns.get("llm_" + set_cfg(s), {}).get("prefill_4608")
            if not ks: return None
            d = collections.defaultdict(Agg)
            for k in ks: d[k["cls"]].add([k])
            d["all"] = Agg().add(ks)
            return d
        attribution_table(llm_ns, list("FBT"), "LLM prefill (8 frames, 4608 tokens): attribution", ("us",))
        class_shapley(llm_ns, list("FBT"), "LLM prefill (nsys)", "us")

    # ---- ncu: counters
    nc = {os.path.basename(p)[:-8]: ncu_kernels(p) for p in glob.glob(os.path.join(D, "ncu", "*_raw.csv"))}
    if nc:
        print("\n## 2. Hardware counters (ncu, clocks at base, caches flushed per kernel), composed per frame")
        vf = {n: vit_frame(ks) for n, ks in nc.items() if n.startswith("vit_")}

        def vit_nc(s):
            return vf.get(f"vit_{set_cfg(s - {'C'})}_{'5x2' if 'C' in s else '9x4'}")
        print("\nViT per frame (ms / MB DRAM read / MB L2 requests):\n\n| config | 9x4 | 5x2 | 1x2 |\n|---|---|---|---|")
        for c in VIT_CFGS:
            cells = []
            for sh in ("9x4", "5x2", "1x2"):
                d = vf.get(f"vit_{c}_{sh}")
                cells.append("-" if not d else f"{d['all'].get('us') / 1e3:.0f} / {d['all'].get('dram'):.0f} / {d['all'].get('l2req'):.0f}")
            print(f"| {c} | " + " | ".join(cells) + " |")
        attribution_table(vit_nc, list("WFBTC"), "ViT per frame: attribution of time and memory traffic")
        for m in ("dram", "l2req", "us"):
            class_shapley(vit_nc, list("WFBTC"), "ViT (ncu)", m)
        intensive_table([(c, vf.get(f"vit_{c}_9x4")) for c in ("base", "W", "FB", "T", "FBT", "WFBT")],
                        "ViT full frame: efficiency of the main classes per config")
        intensive_table([(f"{c} {sh}", vf.get(f"vit_{c}_{sh}")) for c in ("base", "W") for sh in ("9x4", "5x2", "1x2")],
                        "ViT crops (window cache) with and without batched windows")

        lf = {n: llm_ubatch(ks) for n, ks in nc.items() if n.startswith("llm_")}
        if lf:
            print("\nLLM prefill, last ubatch (512 tokens), per ubatch (ms / MB DRAM / MB L2 req):\n")
            for n, d in sorted(lf.items()):
                print(f"- {n}: {d['all'].get('us') / 1e3:.0f} / {d['all'].get('dram'):.0f} / {d['all'].get('l2req'):.0f}")
            attribution_table(lambda s: lf.get(f"llm_{set_cfg(s)}_4608"), list("FBT"), "LLM prefill ubatch at N=4608: attribution")
            for m in ("dram", "us"):
                class_shapley(lambda s: lf.get(f"llm_{set_cfg(s)}_4608"), list("FBT"), "LLM prefill (ncu)", m)
            intensive_table([(f"base N={n}", lf.get(f"llm_base_{n}")) for n in (1152, 2304, 4608)],
                            "LLM prefill vs KV length (adaptive sampling: fewer frames = shorter KV)")
            intensive_table([(c, lf.get(f"llm_{c}_4608")) for c in LLM_CFGS], "LLM prefill per switch config (N=4608)")

    # ---- warm caches: cross-kernel reuse
    wa = {os.path.basename(p)[:-8]: ncu_kernels(p) for p in glob.glob(os.path.join(D, "warm", "*_raw.csv"))}
    if wa and nc:
        print("\n## 3. Producer -> consumer reuse in L2 (caches kept between kernels vs flushed)\n")
        print("| run | class | DRAM MB flushed | DRAM MB warm | saved by L2 reuse | L2 hit flushed -> warm |")
        print("|---|---|---|---|---|---|")
        for n in sorted(wa):
            if n not in nc: continue
            fn = vit_frame if n.startswith("vit_") else llm_ubatch
            a, b = fn(nc[n]), fn(wa[n])
            if not a or not b: continue
            for c in ["softmax", "attn matmul", "attention", "GEMM", "cast", "add", "norm", "rope", "glu", "copy", "all"]:
                if c not in a or c not in b or a[c].get("dram") < 5: continue
                x, y = a[c].get("dram"), b[c].get("dram")
                print(f"| {n} | {c} | {x:.0f} | {y:.0f} | {(1 - y / x) * 100:.0f}% | {a[c].get('hit'):.0f}% -> {b[c].get('hit'):.0f}% |")


if __name__ == "__main__":
    main()
