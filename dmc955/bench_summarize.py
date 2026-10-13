#!/usr/bin/env python3
"""bench_summarize.py — turn bench_vanilla.sh output into a CSV + a readable table.

Pulls two things per run:

  from <stem>.json   the harness's own timers
                     timing.encode_ms      = ViT encode + LLM prefill (conflated)
                     timing.generation_ms  = decode loop
                     inference.n_tokens
  from <stem>.log    mtmd-helper's per-chunk logs, which DO separate the phases
                     "image slice encoded in N ms"       -> ViT
                     "image decoded (batch i/n) in N ms" -> prefill

so vit_ms + prefill_ms should account for ~all of encode_ms. The residual is
tokenize + text-chunk decode; it's reported so you can see it stays small.

Runs are keyed by (video, frames, fa, rep). The fa axis exists because the
pruning variants force flash attention off — see the header of bench_vanilla.sh.

Usage: python3 bench_summarize.py bench_results/
"""
import json
import re
import statistics
import sys
from pathlib import Path

RE_VIT     = re.compile(r"image slice encoded in (\d+) ms")
RE_PREFILL = re.compile(r"image decoded \(batch \d+/\d+\) in (\d+) ms")
RE_STEM    = re.compile(r"(.+)_f(\d+)_fa(on|off|auto)_r(\d+)$")

# With --warmup-iters, the log holds one block of mtmd timing lines per
# iteration but the JSON's encode_ms covers only the LAST one. Summing the whole
# log would double-count the warmups and drive `residual` negative, so trim to
# the measured block. baseline_vlm.cpp prints this marker before it.
MEASURED_MARKER = "--- Measured iteration ---"


def measured_block(text: str) -> str:
    """The portion of a run log belonging to the measured iteration."""
    idx = text.rfind(MEASURED_MARKER)
    return text if idx < 0 else text[idx:]

COLS = ["video", "frames", "fa", "rep", "n_tokens",
        "vit_ms", "prefill_ms", "encode_ms", "residual_ms",
        "gen_ms", "ttft_ms", "tok_s", "vram_mib"]


def collect(outdir: Path):
    rows = []
    for jf in sorted(outdir.glob("*_r[0-9]*.json")):     # skip warmup runs
        stem = jf.with_suffix("")
        m = RE_STEM.match(stem.name)
        if not m:
            continue                                     # stale pre-fa-axis runs
        video, frames, fa, rep = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

        try:
            j = json.load(open(jf))
        except Exception as e:
            print(f"  skip {jf.name}: {e}", file=sys.stderr)
            continue

        t = j["timing"]
        encode_ms = float(t["encode_ms"])
        gen_ms    = float(t["generation_ms"])
        n_tokens  = int(j["inference"]["n_tokens"])

        log = stem.with_suffix(".log")
        text = measured_block(log.read_text(errors="ignore")) if log.exists() else ""
        vit_ms     = sum(int(x) for x in RE_VIT.findall(text))
        prefill_ms = sum(int(x) for x in RE_PREFILL.findall(text))

        vram = stem.with_suffix(".vram")
        try:
            vram_mib = int(vram.read_text().strip())
        except Exception:
            vram_mib = 0

        rows.append({
            "video": video, "frames": frames, "fa": fa, "rep": rep, "n_tokens": n_tokens,
            "vit_ms": vit_ms, "prefill_ms": prefill_ms, "encode_ms": encode_ms,
            "residual_ms": encode_ms - vit_ms - prefill_ms,
            "gen_ms": gen_ms,
            # first token is emitted right after prefill completes
            "ttft_ms": encode_ms + (gen_ms / n_tokens if n_tokens else 0.0),
            "tok_s": n_tokens / (gen_ms / 1000.0) if gen_ms else 0.0,
            "vram_mib": vram_mib,
        })
    return rows


def med(vals):
    return statistics.median(vals) if vals else 0.0


def spread(vals):
    """Half-range as a % of the median — the honest 'is this run stable' number."""
    if len(vals) < 2:
        return 0.0
    m = med(vals)
    return (max(vals) - min(vals)) / 2 / m * 100 if m else 0.0


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_results")
    rows = collect(outdir)
    if not rows:
        print(f"No completed runs found in {outdir}/", file=sys.stderr)
        return 1

    csv_path = outdir / "results.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in COLS) + "\n")

    # ---- per-config medians -------------------------------------------------
    configs = {}
    for r in rows:
        configs.setdefault((r["frames"], r["video"], r["fa"]), []).append(r)

    print(f"\n=== Vanilla VLM benchmark  ({len(rows)} runs) ===\n")
    hdr = (f'{"video":<12}{"nf":>4}{"fa":>5}{"ViT":>8}{"prefill":>9}{"resid":>7}'
           f'{"TTFT":>9}{"gen":>8}{"tok":>6}{"tok/s":>8}{"VRAM":>8}{"±%":>6}')
    print(hdr)
    print("-" * len(hdr))

    for key in sorted(configs, key=lambda k: (k[0], k[1], k[2])):
        frames, video, fa = key
        rs = configs[key]
        gens = [r["gen_ms"] for r in rs]
        print(f'{video:<12}{frames:>4}{fa:>5}'
              f'{med([r["vit_ms"] for r in rs]):>8.0f}'
              f'{med([r["prefill_ms"] for r in rs]):>9.0f}'
              f'{med([r["residual_ms"] for r in rs]):>7.0f}'
              f'{med([r["ttft_ms"] for r in rs]):>9.0f}'
              f'{med(gens):>8.0f}'
              f'{med([r["n_tokens"] for r in rs]):>6.0f}'
              f'{med([r["tok_s"] for r in rs]):>8.1f}'
              f'{med([r["vram_mib"] for r in rs]):>8.0f}'
              f'{spread(gens):>6.1f}')

    # ---- scaling with visual tokens ----------------------------------------
    # This is the headline: the vanilla cost curve your optimizations must beat.
    by_fk = {}
    for r in rows:
        by_fk.setdefault((r["frames"], r["fa"]), []).append(r)

    print(f"\n=== Scaling vs frame count (median across videos) ===\n")
    print(f'{"frames":>7}{"fa":>5}{"vis_tok~":>10}{"ViT ms":>9}{"prefill":>9}{"TTFT":>9}{"tok/s":>8}{"VRAM":>8}')
    print("-" * 65)
    for nf, fa in sorted(by_fk):
        rs = by_fk[(nf, fa)]
        print(f'{nf:>7}{fa:>5}{nf*592:>10}'
              f'{med([r["vit_ms"] for r in rs]):>9.0f}'
              f'{med([r["prefill_ms"] for r in rs]):>9.0f}'
              f'{med([r["ttft_ms"] for r in rs]):>9.0f}'
              f'{med([r["tok_s"] for r in rs]):>8.1f}'
              f'{med([r["vram_mib"] for r in rs]):>8.0f}')

    # ---- what flash attention alone costs -----------------------------------
    # Subtract this before crediting/blaming any pruning result that runs fa=off.
    frames_with_both = sorted({nf for nf, _ in by_fk}
                              & {nf for nf, fa in by_fk if fa == "off"}
                              & {nf for nf, fa in by_fk if fa == "on"})
    if frames_with_both:
        print(f"\n=== Flash attention penalty (off vs on, median across videos) ===")
        print("    This is the handicap a fa=off pruning run carries before it does\n"
              "    any pruning at all. Subtract it before crediting an optimization.\n")
        print(f'{"frames":>7}{"prefill on":>12}{"prefill off":>13}{"delta":>9}{"TTFT on":>10}{"TTFT off":>10}{"delta":>9}')
        print("-" * 70)
        for nf in frames_with_both:
            on, off = by_fk[(nf, "on")], by_fk[(nf, "off")]
            p_on, p_off = med([r["prefill_ms"] for r in on]), med([r["prefill_ms"] for r in off])
            t_on, t_off = med([r["ttft_ms"] for r in on]),    med([r["ttft_ms"] for r in off])
            dp = (p_off - p_on) / p_on * 100 if p_on else 0.0
            dt = (t_off - t_on) / t_on * 100 if t_on else 0.0
            print(f'{nf:>7}{p_on:>12.0f}{p_off:>13.0f}{dp:>8.0f}%{t_on:>10.0f}{t_off:>10.0f}{dt:>8.0f}%')

    # residual = encode_ms (JSON timer) - vit - prefill (parsed from the log).
    # Two independent sources; a large negative value means they disagree, which
    # in practice means the log parse picked up iterations the timer did not.
    bad = [r for r in rows if r["residual_ms"] < -20]
    if bad:
        print(f"\n!! {len(bad)} run(s) have strongly negative residual_ms "
              f"(worst {min(r['residual_ms'] for r in bad):.0f} ms).")
        print("   The log parse and the JSON timer disagree — check that the")
        print("   measured-iteration marker is present in those .log files.")

    hi = max(spread([r["gen_ms"] for r in rs]) for rs in configs.values())
    print(f"\nWorst-case run-to-run spread in gen_ms: {hi:.1f}%")
    if hi > 5:
        print("  ^ >5%: GPU clocks are drifting. Lock them (sudo nvidia-smi -lgc <MHz>)")
        print("    or raise --reps before trusting small deltas.")
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
