#!/usr/bin/env python3
"""bench_serving_summarize.py — read an edge-serving run.

Reports PER-STAGE latency (ViT encode / LLM prefill / decode) per video,
averaged over repeated rounds with a confidence interval, plus the things a
per-process benchmark cannot show: the cold-start cost, the steady-state tail,
and drift over a sustained run.

Stage split: the serving JSON records encode_ms (ViT + prefill combined) and
generation_ms. mtmd-helper logs the two encode phases separately, so we slice
the run log at each "--- Request N (round R, video V) ---" marker and attribute
that request's "image slice encoded" / "image decoded (batch...)" lines to it.

Round 0 is treated as warmup and excluded: the very first request pays model
load, unramped clocks and uncaptured CUDA graphs. Run N+1 rounds to get N
measured ones.

Usage: bench_serving_summarize.py <stem>          # finds .json/.log/.gpu.csv
       bench_serving_summarize.py <json> [gpu.csv]
"""
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

RE_REQ     = re.compile(r"^--- Request (\d+) \(round (\d+), video (\S+)\) ---", re.M)
RE_VIT     = re.compile(r"image slice encoded in (\d+) ms")
RE_PREFILL = re.compile(r"image decoded \(batch \d+/\d+\) in (\d+) ms")

# Two-sided 95% t multipliers by degrees of freedom. With n=20 the normal
# approximation (1.96) understates the interval by ~7%, which matters when the
# whole point of repeating is to state an honest error bar.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001, 119: 1.980}


def t95(df):
    if df <= 0:
        return 0.0
    for k in sorted(T95):
        if df <= k:
            return T95[k]
    return 1.96


def ci95(vals):
    """Half-width of the 95% CI on the mean. 0 when there is nothing to say."""
    n = len(vals)
    if n < 2:
        return 0.0
    return t95(n - 1) * statistics.stdev(vals) / math.sqrt(n)


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def parse_stages(log_path):
    """{request_index: (vit_ms, prefill_ms)} from the run log."""
    out = {}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="ignore")
    marks = [(m.start(), int(m.group(1))) for m in RE_REQ.finditer(text)]
    for i, (pos, idx) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        block = text[pos:end]
        out[idx] = (sum(int(x) for x in RE_VIT.findall(block)),
                    sum(int(x) for x in RE_PREFILL.findall(block)))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    # NB: never use Path.with_suffix() here — stems contain dots (fps2.0@src8.0)
    # and it would eat everything after the last one.
    a = str(sys.argv[1])
    stem = Path(a[:-5] if a.endswith(".json") else a)
    jpath = Path(str(stem) + ".json")
    gpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(str(stem) + ".gpu.csv")

    j = json.load(open(jpath))
    reqs = j["requests"]
    if not reqs:
        print("No requests recorded.", file=sys.stderr)
        return 1

    stages = parse_stages(Path(str(stem) + ".log"))
    for r in reqs:
        vit, pre = stages.get(r["index"], (0, 0))
        r["vit_ms"], r["prefill_ms"] = vit, pre
        r["residual_ms"] = r["encode_ms"] - vit - pre
        r["ttft_ms"] = r["encode_ms"] + (r["generation_ms"] / r["n_tokens"]
                                         if r["n_tokens"] else 0.0)

    rounds = max(r["round"] for r in reqs) + 1
    measured = [r for r in reqs if r["round"] > 0] or reqs
    vids = sorted({r["video"] for r in reqs})

    print(f"\n=== Edge serving, per-stage latency ===")
    print(f"    {len(vids)} videos x {rounds} rounds = {len(reqs)} requests "
          f"({j['total_ms']/1000:.1f} s), one loaded model")
    print(f"    round 0 excluded as warmup -> {len(measured)} measured requests")
    print(f"    frames/request {j['base_frames']}, n_predict {j['n_predict']}")
    print(f"    values are mean +/- 95% CI over {rounds-1} rounds\n")

    # ---- per-video stage table -------------------------------------------
    hdr = (f'{"video":<13}{"ViT":>15}{"prefill":>15}{"resid":>8}'
           f'{"TTFT":>9}{"decode":>15}{"tok":>5}{"tok/s":>8}')
    print(hdr)
    print("-" * len(hdr))
    agg = {k: [] for k in ("vit_ms", "prefill_ms", "residual_ms", "ttft_ms",
                           "generation_ms", "request_ms", "tok_per_s")}
    for v in vids:
        rs = [r for r in measured if r["video"] == v]
        if not rs:
            continue
        for k in agg:
            agg[k].extend(r[k] for r in rs)
        vit = [r["vit_ms"] for r in rs]
        pre = [r["prefill_ms"] for r in rs]
        gen = [r["generation_ms"] for r in rs]
        print(f'{v:<13}'
              f'{statistics.mean(vit):>8.1f}±{ci95(vit):<6.1f}'
              f'{statistics.mean(pre):>8.1f}±{ci95(pre):<6.1f}'
              f'{statistics.mean([r["residual_ms"] for r in rs]):>8.1f}'
              f'{statistics.mean([r["ttft_ms"] for r in rs]):>9.1f}'
              f'{statistics.mean(gen):>8.1f}±{ci95(gen):<6.1f}'
              f'{statistics.mean([r["n_tokens"] for r in rs]):>5.0f}'
              f'{statistics.mean([r["tok_per_s"] for r in rs]):>8.1f}')

    # ---- pooled ----------------------------------------------------------
    print("-" * len(hdr))
    print(f'{"ALL":<13}'
          f'{statistics.mean(agg["vit_ms"]):>8.1f}±{ci95(agg["vit_ms"]):<6.1f}'
          f'{statistics.mean(agg["prefill_ms"]):>8.1f}±{ci95(agg["prefill_ms"]):<6.1f}'
          f'{statistics.mean(agg["residual_ms"]):>8.1f}'
          f'{statistics.mean(agg["ttft_ms"]):>9.1f}'
          f'{statistics.mean(agg["generation_ms"]):>8.1f}±{ci95(agg["generation_ms"]):<6.1f}'
          f'{"":>5}'
          f'{statistics.mean(agg["tok_per_s"]):>8.1f}')

    # Stage shares of the whole request. ViT/prefill are what visual-token
    # pruning attacks; decode is what KV work attacks.
    tot = statistics.mean(agg["request_ms"])
    print(f"\n--- Stage share of a {tot:.0f} ms request (pooled, n={len(agg['vit_ms'])}) ---")
    for name, key in (("ViT encode", "vit_ms"), ("LLM prefill", "prefill_ms"),
                      ("residual", "residual_ms"), ("LLM decode", "generation_ms")):
        m = statistics.mean(agg[key])
        print(f'  {name:<14}{m:>8.1f} ms  {m/tot*100:>5.1f}%')

    # ---- cold start ------------------------------------------------------
    v0 = reqs[0]["video"]
    later = [r for r in measured if r["video"] == v0]
    if later:
        c = reqs[0]
        w = statistics.mean([r["request_ms"] for r in later])
        we = statistics.mean([r["encode_ms"] for r in later])
        print(f"\n--- Cold start (request 0, '{v0}') vs its {len(later)} warm runs ---")
        print(f'  request  cold {c["request_ms"]:>7.0f} ms   warm {w:>7.0f} ms   '
              f'{(c["request_ms"]/w-1)*100:+.1f}%')
        print(f'  encode   cold {c["encode_ms"]:>7.0f} ms   warm {we:>7.0f} ms   '
              f'{(c["encode_ms"]/we-1)*100:+.1f}%')

    # ---- tail + drift ----------------------------------------------------
    lat = [r["request_ms"] for r in measured]
    print(f"\n--- Steady-state request latency (n={len(lat)}) ---")
    print(f'  mean {statistics.mean(lat):.0f} +/- {ci95(lat):.0f} ms   '
          f'median {statistics.median(lat):.0f}   p95 {pct(lat,95):.0f}   '
          f'p99 {pct(lat,99):.0f}   max {max(lat):.0f}')

    half = len(measured) // 2
    if half:
        a = statistics.mean([r["request_ms"] for r in measured[:half]])
        b = statistics.mean([r["request_ms"] for r in measured[half:]])
        print(f"\n--- Drift: first half vs second half of the measured run ---")
        print(f'  {a:.0f} ms -> {b:.0f} ms  ({(b/a-1)*100:+.1f}%)')
        if abs(b / a - 1) > 0.05:
            print("  !! >5% drift — check the GPU trace for thermal throttling.")

    # ---- GPU trace -------------------------------------------------------
    try:
        rows = [r for r in csv.DictReader(open(gpath))
                if r.get("sm_mhz", "").strip().isdigit()]
    except Exception:
        rows = []
    if rows:
        sm = [int(r["sm_mhz"]) for r in rows]
        tp = [int(r["temp_c"]) for r in rows]
        pw = [float(r["power_w"]) for r in rows
              if r["power_w"].replace(".", "", 1).isdigit()]
        n2 = len(sm) // 2
        print(f"\n--- GPU during the run ({len(rows)} samples) ---")
        print(f'  SM clock  median {statistics.median(sm):.0f} MHz   '
              f'first half {statistics.median(sm[:n2]):.0f}   '
              f'second half {statistics.median(sm[n2:]):.0f}')
        print(f'  temp      start {tp[0]} C   max {max(tp)} C   end {tp[-1]} C')
        if pw:
            print(f'  power     median {statistics.median(pw):.0f} W   max {max(pw):.0f} W')
        if statistics.median(sm[n2:]) < statistics.median(sm[:n2]) * 0.95:
            print("  !! SM clock fell >5% between halves — thermal throttling is real here,")
            print("     and any latency drift above is explained by it.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
