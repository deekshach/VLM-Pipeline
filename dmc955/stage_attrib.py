#!/usr/bin/env python3
"""
stage_attrib.py -- GPU-side ViT / prefill / decode attribution from an nsys trace.

WHY THIS EXISTS

`bench_serving_summarize.py` splits ViT from prefill by parsing mtmd's host-side
log lines:

    "image slice encoded in N ms"        -> ViT
    "image decoded (batch X/Y) in N ms"  -> prefill

Those are wall-clock spans around *synchronous* host calls. They are correct only
while the two phases are serial. The moment ViT encode and LLM prefill overlap --
which is the single largest optimisation available (578 ms/request, ~13% overlap
today) -- the spans overlap each other, stop summing to encode_ms, and
residual_ms goes negative. **The measurement that would evaluate the optimisation
is invalidated by the optimisation.**

This attributes stages from GPU activity instead, which stays correct under
overlap and additionally *reports* the overlap as a first-class number.

ROBUSTNESS NOTES (both learned the hard way)

  * Do NOT hardcode the ViT stream as 17. With CUDA graphs disabled the clip
    backend lands on stream 16. Streams are detected by kernel identity here:
    the ViT stream is whichever carries `rope_vision`; the LLM stream is
    whichever carries `mul_mat_vec_f`.
  * Phase boundaries follow PROFILING_HANDOFF.md section 6: prefill runs from the
    encode start until decode *begins*, NOT until the last ViT kernel -- the final
    frame's prefill runs after the vision tower finishes, and cutting there
    silently drops ~8% of prefill.

Usage:
    python3 stage_attrib.py <trace.nsys-rep | trace.sqlite> [--json out.json]
    python3 stage_attrib.py <trace> --csv results.csv --label ub640
"""
import argparse, collections, json, os, subprocess, sqlite3, statistics, sys

NSYS = os.environ.get("NSYS", "/usr/local/cuda-12.6/bin/nsys")

# nsys GPU_METRICS metricIds (see TARGET_INFO_GPU_METRICS in any export).
# NB: metricId 0 (GPC Clock) exports as garbage/negative on this box -- ignored.
METRICS = {14: "sm_active", 15: "sm_issue", 16: "tensor_active",
           29: "dram_rd", 30: "dram_wr", 27: "unalloc_warps", 24: "comp_warps"}


def ensure_sqlite(path):
    if path.endswith(".sqlite"):
        return path
    out = path.rsplit(".", 1)[0] + ".sqlite"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        print(f"-- exporting {os.path.basename(path)} -> sqlite ...", file=sys.stderr)
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite",
                        "true", "-o", out, path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def busy_and_span(rows):
    """Union of [start,end) intervals -> (busy_ns, span_ns). Handles overlap."""
    if not rows:
        return 0, 0
    rows = sorted(rows)
    busy = 0
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return busy + ce - cs, rows[-1][1] - rows[0][0]


def detect_streams(c, sid):
    """Identify ViT and LLM streams by kernel identity, never by stream number."""
    def stream_carrying(substr):
        ids = [i for i, v in sid.items() if substr in v]
        if not ids:
            return None
        q = (f"SELECT streamId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
             f"WHERE shortName IN ({','.join(map(str, ids))}) GROUP BY 1 ORDER BY 2 DESC")
        r = c.execute(q).fetchone()
        return r[0] if r else None

    vit = stream_carrying("rope_vision")
    llm = stream_carrying("mul_mat_vec_f")
    if vit is None or llm is None:
        sys.exit("!! could not identify streams: no rope_vision (ViT) or "
                 "mul_mat_vec_f (LLM) kernels found. Is this a VLM trace?")
    if vit == llm:
        sys.exit(f"!! ViT and LLM resolved to the same stream ({vit}); "
                 "attribution by stream is not possible for this trace.")
    return vit, llm


def phase_metrics(c, t0, t1):
    out = {}
    for mid, name in METRICS.items():
        vals = [v for (v,) in c.execute(
            "SELECT value FROM GPU_METRICS WHERE metricId=? AND timestamp BETWEEN ? AND ?",
            (mid, t0, t1))]
        if vals:
            out[name] = statistics.mean(vals)
    if "dram_rd" in out and "dram_wr" in out:
        out["dram_total"] = out["dram_rd"] + out["dram_wr"]
    if out.get("comp_warps") and out.get("unalloc_warps") is not None:
        tot = out["comp_warps"] + out["unalloc_warps"]
        if tot:
            out["unalloc_pct"] = 100.0 * out["unalloc_warps"] / tot
    return out


def connect(db_path):
    db = sqlite3.connect(db_path)
    # nsys's StringIds table captures process strings, which for this workload
    # includes the MODEL'S GENERATED TEXT. At a fixed -n the output is truncated
    # mid-token and can end in a partial multi-byte UTF-8 sequence, which makes
    # the default text_factory raise OperationalError on an unrelated query.
    # Decode leniently: we only ever match kernel names, which are pure ASCII.
    db.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return db


def phase_windows(c, sid, llm_s, vit_s, min_decode_kernels=1000):
    """-> [(encode_start, decode_start, decode_end, n_decode_kernels), ...], one per
    complete request. THE single definition of the phase boundaries; every tool
    that slices this trace must come through here so the definitions cannot drift.

    prefill runs from the encode start to the DECODE start, not to the last ViT
    kernel: the final frame's prefill executes after the vision tower finishes,
    and cutting there silently drops ~8% of the stage (handbook section 6)."""
    mmv = [i for i, v in sid.items() if v == "mul_mat_vec_f"]
    dec = c.execute(
        f"SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? "
        f"AND shortName IN ({','.join(map(str, mmv))}) ORDER BY start", (llm_s,)).fetchall()
    if not dec:
        sys.exit("!! no decode kernels found")
    cuts = [0]
    for i in range(1, len(dec)):
        if dec[i][0] - dec[i - 1][1] > 50_000_000:      # 50 ms gap = new request
            cuts.append(i)
    cuts.append(len(dec))
    dec_win = [(dec[a][0], dec[b - 1][1], b - a) for a, b in zip(cuts, cuts[1:])
               if b - a >= min_decode_kernels]         # drops the tiny warmup decode

    vit_all = c.execute(
        "SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? ORDER BY start",
        (vit_s,)).fetchall()
    out, prev_end = [], 0
    for (ds, de, nk) in dec_win:
        vs = next((s for (s,) in vit_all if prev_end < s < ds), None)
        prev_end = de
        if vs is not None:
            out.append((vs, ds, de, nk))
    return out


def analyse(db_path, min_decode_kernels=1000):
    db = connect(db_path)
    c = db.cursor()
    sid = {i: v for i, v in c.execute("SELECT id,value FROM StringIds")}
    vit_s, llm_s = detect_streams(c, sid)

    has_metrics = c.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='GPU_METRICS'"
    ).fetchone()[0] > 0

    vit_all = c.execute(
        "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? ORDER BY start",
        (vit_s,)).fetchall()

    reqs = []
    for (vs, ds, de, nk) in phase_windows(c, sid, llm_s, vit_s, min_decode_kernels):
        vit_rows = [(s, e) for s, e in vit_all if vs <= s and e <= ds]
        llm_rows = c.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? "
            "AND start>=? AND end<=? ORDER BY start", (llm_s, vs, ds)).fetchall()
        dec_rows = c.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? "
            "AND start>=? AND end<=? ORDER BY start", (llm_s, ds, de)).fetchall()

        vit_busy, _ = busy_and_span(vit_rows)
        pre_busy, _ = busy_and_span(llm_rows)
        enc_busy, enc_span = busy_and_span(vit_rows + llm_rows)
        dec_busy, dec_span = busy_and_span(dec_rows)
        overlap = vit_busy + pre_busy - enc_busy

        r = dict(vit_ms=vit_busy / 1e6, prefill_ms=pre_busy / 1e6,
                 encode_wall_ms=enc_span / 1e6, encode_busy_ms=enc_busy / 1e6,
                 overlap_ms=overlap / 1e6,
                 overlap_pct=100.0 * overlap / vit_busy if vit_busy else 0.0,
                 encode_idle_ms=(enc_span - enc_busy) / 1e6,
                 decode_ms=dec_span / 1e6, decode_busy_ms=dec_busy / 1e6,
                 decode_kernels=nk,
                 vit_kernels=len(vit_rows), prefill_kernels=len(llm_rows),
                 request_gpu_ms=(enc_span + dec_span) / 1e6)
        if has_metrics:
            r["encode_hw"] = phase_metrics(c, vs, ds)
            r["decode_hw"] = phase_metrics(c, ds, de)
        reqs.append(r)

    return dict(trace=os.path.basename(db_path), vit_stream=vit_s,
                llm_stream=llm_s, requests=reqs)


def report(res):
    print(f"\n=== GPU-side stage attribution: {res['trace']} ===")
    print(f"    ViT stream {res['vit_stream']}, LLM stream {res['llm_stream']} "
          f"(detected by kernel identity, not hardcoded)")
    rs = res["requests"]
    if not rs:
        print("    no complete requests found"); return
    print(f"    {len(rs)} complete request(s)\n")
    h = (f"{'req':>4}{'ViT ms':>9}{'prefill':>9}{'overlap':>9}{'ovl%':>7}"
         f"{'enc wall':>10}{'enc idle':>9}{'decode':>9}{'total':>9}")
    print(h); print("-" * len(h))
    for i, r in enumerate(rs):
        print(f"{i:>4}{r['vit_ms']:>9.1f}{r['prefill_ms']:>9.1f}{r['overlap_ms']:>9.1f}"
              f"{r['overlap_pct']:>6.1f}%{r['encode_wall_ms']:>10.1f}"
              f"{r['encode_idle_ms']:>9.1f}{r['decode_ms']:>9.1f}{r['request_gpu_ms']:>9.1f}")
    if len(rs) > 1:
        m = lambda k: statistics.mean(x[k] for x in rs)
        print("-" * len(h))
        print(f"{'mean':>4}{m('vit_ms'):>9.1f}{m('prefill_ms'):>9.1f}{m('overlap_ms'):>9.1f}"
              f"{m('overlap_pct'):>6.1f}%{m('encode_wall_ms'):>10.1f}"
              f"{m('encode_idle_ms'):>9.1f}{m('decode_ms'):>9.1f}{m('request_gpu_ms'):>9.1f}")

    print(f"\n--- pipelining headroom (the number T4 is about) ---")
    v, p, w = m('vit_ms'), m('prefill_ms'), m('encode_wall_ms')
    print(f"  ViT {v:.0f} + prefill {p:.0f} = {v+p:.0f} ms of work in {w:.0f} ms of wall")
    print(f"  overlap {m('overlap_ms'):.1f} ms = {m('overlap_pct'):.0f}% of ViT")
    print(f"  ceiling if ViT fully hidden: encode -> {max(v,p):.0f} ms "
          f"(saves {w-max(v,p):.0f} ms/request)")

    if "encode_hw" in rs[0]:
        print(f"\n--- hardware counters (nsys GPU_METRICS, whole phase) ---")
        print(f"{'':<16}{'encode':>10}{'decode':>10}")
        for k in ("sm_active", "sm_issue", "tensor_active", "dram_total", "unalloc_pct"):
            a = statistics.mean(x["encode_hw"].get(k, float('nan')) for x in rs)
            b = statistics.mean(x["decode_hw"].get(k, float('nan')) for x in rs)
            print(f"  {k:<14}{a:>10.1f}{b:>10.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", help="write full per-request results here")
    ap.add_argument("--csv", help="append one summary row here (for sweeps)")
    ap.add_argument("--label", default="", help="label for the --csv row")
    a = ap.parse_args()

    res = analyse(ensure_sqlite(a.trace))
    report(res)

    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
        print(f"-- wrote {a.json}")
    if a.csv:
        import csv as _csv
        rs = res["requests"]
        if rs:
            m = lambda k: statistics.mean(x[k] for x in rs)
            row = dict(label=a.label or res["trace"], n_requests=len(rs),
                       vit_ms=round(m('vit_ms'), 1), prefill_ms=round(m('prefill_ms'), 1),
                       overlap_ms=round(m('overlap_ms'), 1),
                       overlap_pct=round(m('overlap_pct'), 1),
                       encode_wall_ms=round(m('encode_wall_ms'), 1),
                       decode_ms=round(m('decode_ms'), 1),
                       request_gpu_ms=round(m('request_gpu_ms'), 1))
            if "encode_hw" in rs[0]:
                for k in ("sm_issue", "tensor_active", "dram_total"):
                    row[f"enc_{k}"] = round(statistics.mean(
                        x["encode_hw"].get(k, 0) for x in rs), 1)
                    row[f"dec_{k}"] = round(statistics.mean(
                        x["decode_hw"].get(k, 0) for x in rs), 1)
            new = not os.path.exists(a.csv)
            with open(a.csv, "a", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"-- appended row '{row['label']}' to {a.csv}")


if __name__ == "__main__":
    sys.exit(main() or 0)
