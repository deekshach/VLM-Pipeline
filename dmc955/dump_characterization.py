#!/usr/bin/env python3
"""dump_characterization.py — export a serving run to CSV, per video and per request.

Reuses bench_serving_summarize's log parsing and t-based CI so the CSVs agree
with what the terminal summary prints — one source of truth for both.

Usage: dump_characterization.py <stem> <outdir>
   e.g. dump_characterization.py serving_results/serve_faon_f8_r21 characterization
"""
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_serving_summarize as bss  # noqa: E402

STAGES = [("vit", "vit_ms"), ("prefill", "prefill_ms"), ("residual", "residual_ms"),
          ("encode", "encode_ms"), ("ttft", "ttft_ms"), ("decode", "generation_ms"),
          ("request", "request_ms"), ("tok_s", "tok_per_s")]


def stats(vals):
    return {
        "mean": statistics.mean(vals),
        "ci95": bss.ci95(vals),
        "sd":   statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min":  min(vals),
        "max":  max(vals),
        "p95":  bss.pct(vals, 95),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    a = str(sys.argv[1])
    stem = Path(a[:-5] if a.endswith(".json") else a)   # not with_suffix(): dots in stem
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    j = json.load(open(str(stem) + ".json"))
    reqs = j["requests"]

    # Same derivation the summariser uses: stage split from the run log, sliced
    # at the per-request markers.
    stages = bss.parse_stages(Path(str(stem) + ".log"))
    for r in reqs:
        vit, pre = stages.get(r["index"], (0, 0))
        r["vit_ms"], r["prefill_ms"] = vit, pre
        r["residual_ms"] = r["encode_ms"] - vit - pre
        r["ttft_ms"] = r["encode_ms"] + (r["generation_ms"] / r["n_tokens"]
                                         if r["n_tokens"] else 0.0)
        # Round 0 is the warmup pass; flagged, not dropped, so the raw file is complete.
        r["measured"] = 1 if r["round"] > 0 else 0

    # ---- per request -----------------------------------------------------
    cols = ["index", "round", "video", "measured", "t_offset_ms",
            "vit_ms", "prefill_ms", "residual_ms", "encode_ms",
            "ttft_ms", "generation_ms", "request_ms", "n_tokens", "tok_per_s"]
    p_req = outdir / "per_request.csv"
    with open(p_req, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in reqs:
            w.writerow([r[c] if isinstance(r[c], (int, str)) else f'{r[c]:.3f}'
                        for c in cols])

    # ---- per video (measured requests only) ------------------------------
    measured = [r for r in reqs if r["measured"]]
    vids = sorted({r["video"] for r in measured})
    head = ["video", "n"]
    for name, _ in STAGES:
        head += [f"{name}_{s}" for s in ("mean", "ci95", "sd", "min", "max", "p95")]
    head += ["n_tokens_mean", "n_tokens_min", "n_tokens_max"]

    p_vid = outdir / "per_video_summary.csv"
    with open(p_vid, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for v in vids:
            rs = [r for r in measured if r["video"] == v]
            row = [v, len(rs)]
            for _, key in STAGES:
                st = stats([r[key] for r in rs])
                row += [f'{st[s]:.3f}' for s in ("mean", "ci95", "sd", "min", "max", "p95")]
            nt = [r["n_tokens"] for r in rs]
            row += [f'{statistics.mean(nt):.2f}', min(nt), max(nt)]
            w.writerow(row)

        # Pooled row over every measured request, so the headline figure lives
        # in the same file as the per-video breakdown.
        row = ["ALL", len(measured)]
        for _, key in STAGES:
            st = stats([r[key] for r in measured])
            row += [f'{st[s]:.3f}' for s in ("mean", "ci95", "sd", "min", "max", "p95")]
        nt = [r["n_tokens"] for r in measured]
        row += [f'{statistics.mean(nt):.2f}', min(nt), max(nt)]
        w.writerow(row)

    # ---- provenance ------------------------------------------------------
    for suffix, dest in ((".env.txt", "run_env.txt"), (".gpu.csv", "gpu_trace.csv")):
        src = Path(str(stem) + suffix)
        if src.exists():
            shutil.copy(src, outdir / dest)

    print(f"wrote {p_vid}  ({len(vids)} videos + ALL)")
    print(f"wrote {p_req}  ({len(reqs)} requests, {len(measured)} measured)")
    print(f"copied run_env.txt, gpu_trace.csv into {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
