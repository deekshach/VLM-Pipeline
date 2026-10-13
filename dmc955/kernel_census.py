#!/usr/bin/env python3
"""
kernel_census.py -- every kernel in every stage, by launch shape, from an nsys trace.

For each of ViT / prefill / decode: the full list of distinct launch shapes with
grid, block, registers, shared memory, invocation count and share of stage time,
sorted by time and carrying a cumulative column so "what covers 80% of this stage"
is read straight off the table.

WHY NOT ncu_targets.py shapes
  That tool answers the same question but is welded to one trace: `phases()`
  hardcodes FRAME_KERNELS = 1270 and N_FRAMES = 13, so it silently mis-slices any
  run with a different frame count. It also defaults to --top 14 (truncating the
  tail) and never reports block size. This one takes phase boundaries from
  stage_attrib.phase_windows -- the single shared definition -- so it works on any
  trace the harness produces, and reports the complete census.

THE GROUPING KEY
  A kernel name is not a kernel, and neither is (name, grid): llama.cpp launches
  q_proj (K=3584) and down_proj (K=18944) at the SAME grid, registers and shared
  memory, differing only in the weight streamed -- 25.7 MB vs 136 MB, ~29 us vs
  ~144 us. nsys records no problem size, so duration is the only trace-visible
  discriminator. Shapes are therefore keyed on

      (name, grid, block, registers, shared memory, duration cluster)

  and a shape that splits into clusters is reported as separate rows, flagged.

Usage:
    python3 kernel_census.py <trace.nsys-rep|.sqlite>
    python3 kernel_census.py <trace> --csv results/kernels_f16.csv
    python3 kernel_census.py <trace> --cover 80     # only rows up to 80% cumulative
    python3 kernel_census.py <trace> --by-name      # collapse shapes, group by name
"""
import argparse, collections, csv as _csv, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_attrib import ensure_sqlite, connect, detect_streams, phase_windows


def clusters(durs, ratio=1.6, min_frac=0.04):
    """Split a shape's durations on a large relative gap, requiring both sides to
    be a meaningful share so one slow outlier cannot manufacture a cluster."""
    d = sorted(durs)
    if len(d) < 8:
        return [(d[0], d[-1])] if d else []
    cuts = []
    for i in range(1, len(d)):
        if d[i - 1] > 0 and d[i] / d[i - 1] >= ratio:
            if min(i, len(d) - i) >= max(2, int(min_frac * len(d))):
                cuts.append(i)
    out, lo = [], 0
    for cpt in cuts + [len(d)]:
        seg = d[lo:cpt]
        if seg:
            out.append((seg[0], seg[-1]))
        lo = cpt
    return out


def short(name, width=58):
    """Keep a DISCRIMINATING tail: several cutlass / mul_mat_f instantiations
    share everything up to the first '<'."""
    n = name.replace("void ", "")
    if "cutlass" in n:
        import re
        m = re.search(r"cutlass_\w+", n)
        if m:
            n = m.group(0)
    return n if len(n) <= width else n[:width - 1] + "…"


def census(db_path, by_name=False):
    db = connect(db_path)
    c = db.cursor()
    sid = {i: v for i, v in c.execute("SELECT id,value FROM StringIds")}
    vit_s, llm_s = detect_streams(c, sid)
    wins = phase_windows(c, sid, llm_s, vit_s)
    if not wins:
        sys.exit("!! no complete requests in this trace")

    Q = ("SELECT demangledName, gridX*gridY*gridZ, blockX*blockY*blockZ, "
         "registersPerThread, staticSharedMemory+dynamicSharedMemory, end-start "
         "FROM CUPTI_ACTIVITY_KIND_KERNEL "
         "WHERE streamId=? AND start>=? AND end<=?")
    raw = {p: collections.defaultdict(list) for p in ("vit", "prefill", "decode")}
    for (vs, ds, de, _) in wins:
        for phase, stream, a, b in (("vit", vit_s, vs, ds),
                                    ("prefill", llm_s, vs, ds),
                                    ("decode", llm_s, ds, de)):
            for nm, grid, blk, regs, smem, dur in c.execute(Q, (stream, a, b)):
                key = (sid[nm],) if by_name else (sid[nm], grid, blk, regs, smem)
                raw[phase][key].append(dur)

    nreq = len(wins)
    out = {}
    for phase, groups in raw.items():
        rows = []
        for key, durs in groups.items():
            segs = [(min(durs), max(durs))] if by_name else clusters([d / 1e3 for d in durs])
            multi = len(segs) > 1
            for lo, hi in segs:
                mem = durs if by_name else [d for d in durs if lo <= d / 1e3 <= hi]
                if not mem:
                    continue
                us = sorted(d / 1e3 for d in mem)
                rows.append(dict(
                    name=key[0],
                    grid=key[1] if not by_name else None,
                    block=key[2] if not by_name else None,
                    regs=key[3] if not by_name else None,
                    smem=key[4] if not by_name else None,
                    n=len(mem), n_per_req=len(mem) / nreq,
                    total_ms=sum(mem) / 1e6, ms_per_req=sum(mem) / 1e6 / nreq,
                    p50_us=statistics.median(us), p10_us=us[int(0.1 * (len(us) - 1))],
                    p90_us=us[int(0.9 * (len(us) - 1))], multimodal=multi))
        tot = sum(r["total_ms"] for r in rows) or 1.0
        rows.sort(key=lambda r: -r["total_ms"])
        cum = 0.0
        for r in rows:
            r["share"] = 100 * r["total_ms"] / tot
            cum += r["share"]
            r["cum"] = cum
        out[phase] = dict(rows=rows, total_ms=tot, ms_per_req=tot / nreq, n_req=nreq)
    return out, nreq


def fmt_smem(v):
    if v is None:
        return "-"
    return f"{v/1024:.0f}K" if v >= 1024 else str(v)


def report(res, nreq, cover=None, by_name=False):
    for phase in ("vit", "prefill", "decode"):
        p = res[phase]
        rows = p["rows"]
        shown = [r for r in rows if cover is None or r["cum"] - r["share"] < cover]
        print(f"\n{'='*132}")
        print(f"{phase.upper()}  —  {p['ms_per_req']:.1f} ms/request  "
              f"({p['total_ms']:.1f} ms over {nreq} requests)  ·  "
              f"{len(rows)} distinct {'kernels' if by_name else 'launch shapes'}"
              + (f"  ·  showing to {cover}% cumulative" if cover else ""))
        print("=" * 132)
        hdr = (f"{'share%':>7}{'cum%':>7}{'n/req':>8}{'ms/req':>9}{'p50_us':>9}"
               f"{'p10-p90 us':>15}{'grid':>8}{'blk':>5}{'reg':>5}{'smem':>6}  kernel")
        print(hdr); print("-" * 132)
        for r in shown:
            band = f"{r['p10_us']:.1f}-{r['p90_us']:.1f}"
            flag = " *" if r["multimodal"] else ""
            print(f"{r['share']:>7.2f}{r['cum']:>7.1f}{r['n_per_req']:>8.0f}"
                  f"{r['ms_per_req']:>9.2f}{r['p50_us']:>9.1f}{band:>15}"
                  f"{(r['grid'] if r['grid'] is not None else 0):>8}"
                  f"{(r['block'] if r['block'] is not None else 0):>5}"
                  f"{(r['regs'] if r['regs'] is not None else 0):>5}"
                  f"{fmt_smem(r['smem']):>6}  {short(r['name'])}{flag}")
        if cover and len(shown) < len(rows):
            rest = sum(r["total_ms"] for r in rows[len(shown):]) / nreq
            print(f"{'':>7}{'':>7}{'':>8}{rest:>9.2f}{'':>9}{'':>15}{'':>8}{'':>5}{'':>5}{'':>6}"
                  f"  ... {len(rows)-len(shown)} more shapes")
    print("\n* = this (name, grid, block, regs, smem) splits into >1 duration cluster:")
    print("  two different problem sizes share the launch shape and are reported separately.")
    print("  nsys records no problem size, so duration is the only discriminator.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--csv", help="write the full census here (all stages, all rows)")
    ap.add_argument("--cover", type=float, help="print only rows up to N%% cumulative")
    ap.add_argument("--by-name", action="store_true",
                    help="collapse launch shapes and group by kernel name only")
    a = ap.parse_args()

    res, nreq = census(ensure_sqlite(a.trace), by_name=a.by_name)
    report(res, nreq, a.cover, a.by_name)

    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        cols = ["stage", "share", "cum", "n", "n_per_req", "total_ms", "ms_per_req",
                "p10_us", "p50_us", "p90_us", "grid", "block", "regs", "smem",
                "multimodal", "name"]
        with open(a.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for phase in ("vit", "prefill", "decode"):
                for r in res[phase]["rows"]:
                    row = {k: r.get(k) for k in cols if k != "stage"}
                    row["stage"] = phase
                    for k in ("share", "cum", "n_per_req", "total_ms", "ms_per_req",
                              "p10_us", "p50_us", "p90_us"):
                        row[k] = round(row[k], 3)
                    w.writerow(row)
        print(f"-- wrote {a.csv}")


if __name__ == "__main__":
    sys.exit(main() or 0)
