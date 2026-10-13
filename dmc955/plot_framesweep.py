#!/usr/bin/env python3
"""
plot_framesweep.py -- static figures for the frame sweep, no plotting library.

matplotlib is not installed here and the Python environment is PEP 668-managed
(installing needs --break-system-packages on a shared box), so these are emitted
as hand-built SVG. That is a feature for a paper: vector output, no dependency,
and the geometry is auditable.

Emits, into --outdir:
    fig_stage_latency.svg   grouped columns: ViT / prefill / decode ms vs frames
    fig_stage_share.svg     100% stacked:    share of GPU stage time

Rasterise to PNG with `--png` if google-chrome is present (headless screenshot
at 2x device scale).

Palette: slots 1-3 of the validated categorical palette. Verified with the
data-viz validator in BOTH modes on the all-pairs list --
  light: CVD dE 9.2, normal-vision dE 24.0, one contrast WARN (aqua 2.74:1)
  dark : CVD dE 9.4, normal-vision dE 20.9, all >= 3:1
The light-mode WARN obliges visible relief, so the decode series carries direct
labels at both ends and the full table ships beside the figure in FRAMESWEEP.md.

Usage:
    python3 plot_framesweep.py                      # SVG only
    python3 plot_framesweep.py --png                # SVG + PNG via headless Chrome
    python3 plot_framesweep.py --theme dark --png
"""
import argparse, csv, os, shutil, subprocess, sys, tempfile

THEME = {
    "light": dict(surface="#fbfbfc", rule="#e0e2e8", rule_strong="#c7cad3",
                  ink="#111318", ink2="#565b66", ink3="#878d99",
                  vit="#2a78d6", pre="#eb6834", dec="#1baf7a", on_fill="#ffffff"),
    "dark":  dict(surface="#16171a", rule="#2c2f36", rule_strong="#434852",
                  ink="#f2f3f5", ink2="#a2a7b3", ink3="#757b87",
                  vit="#3987e5", pre="#d95926", dec="#199e70", on_fill="#ffffff"),
}
SANS = "IBM Plex Sans, DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "IBM Plex Mono, DejaVu Sans Mono, Menlo, Consolas, monospace"

SERIES = [("vit_ms", "ViT encode", "vit"),
          ("prefill_ms", "LLM prefill", "pre"),
          ("decode_ms", "LLM decode", "dec")]

BAR_W, BAR_GAP, GAP_BEFORE_LAST = 24, 2, 0.55   # mark specs: <=24px, 2px surface gap


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def col_path(x, y, w, h, r=4.0):
    """Column with a 4px rounded data-end, square at the baseline."""
    if h <= 0.2:
        return ""
    r = max(0.0, min(r, h, w / 2))
    return (f"M{x:.2f},{y+h:.2f} L{x:.2f},{y+r:.2f} Q{x:.2f},{y:.2f} {x+r:.2f},{y:.2f} "
            f"L{x+w-r:.2f},{y:.2f} Q{x+w:.2f},{y:.2f} {x+w:.2f},{y+r:.2f} "
            f"L{x+w:.2f},{y+h:.2f} Z")


def bands(rows, x0, usable):
    """Band x-positions; an extra half-band before the last point marks the
    non-linear 8 -> 16 step, so the axis never implies even spacing."""
    units = len(rows) + GAP_BEFORE_LAST
    bw = usable / units
    return [(r, x0 + (i + (GAP_BEFORE_LAST if i == len(rows) - 1 else 0)) * bw, bw)
            for i, r in enumerate(rows)]


def txt(x, y, s, size=11, fill="#000", anchor="middle", family=MONO, weight="400"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="{family}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')


def legend_width():
    return sum(11 + 7 + len(lab) * 6.7 + 20 for _, lab, _ in SERIES) - 20


def legend(right_x, y, C):
    """Right-aligned to right_x so the last key never crowds the plot edge."""
    out, cx = [], right_x - legend_width()
    for _, label, key in SERIES:
        out.append(f'<rect x="{cx:.1f}" y="{y-9:.1f}" width="11" height="11" rx="2" fill="{C[key]}"/>')
        out.append(txt(cx + 18, y, label, 12, C["ink2"], "start", SANS))
        cx += 11 + 7 + len(label) * 6.7 + 20
    return "".join(out)


def fig_latency(rows, C, W=1060, H=430):
    L, R, T, B = 66, 18, 56, 52
    pw, ph = W - L - R, H - T - B
    ymax = 2400
    y = lambda v: T + ph * (1 - v / ymax)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="{C["surface"]}"/>',
         txt(L, 24, "Per-stage latency vs frame count", 16, C["ink"], "start", SANS, "600"),
         txt(L, 41, "GPU-busy time per stream · alley_1 · Qwen2.5-VL-7B-BF16 · n=128 · 3 requests/point",
             11.5, C["ink3"], "start", SANS),
         legend(W - R, 24, C)]

    for t in range(0, ymax + 1, 600):
        s.append(f'<line x1="{L}" x2="{L+pw}" y1="{y(t):.2f}" y2="{y(t):.2f}" '
                 f'stroke="{C["rule"]}" stroke-width="1"/>')
        s.append(txt(L - 10, y(t) + 4, f"{t:,}", 11, C["ink3"], "end"))
    s.append(txt(L - 10, T - 8, "ms", 10.5, C["ink3"], "end"))

    gw = BAR_W * 3 + BAR_GAP * 2
    bs = bands(rows, L, pw)
    for bi, (r, x, bw) in enumerate(bs):
        gx = x + (bw - gw) / 2
        for si, (col, _, key) in enumerate(SERIES):
            v = float(r[col])
            bx = gx + si * (BAR_W + BAR_GAP)
            s.append(f'<path d="{col_path(bx, y(v), BAR_W, ph * v / ymax)}" fill="{C[key]}"/>')
            # Selective direct labels: the endpoint group, plus the decode series at
            # N=1 -- decode is the low-contrast slot and needs relief at both ends.
            if bi == len(bs) - 1 or (bi == 0 and key == "dec"):
                # No thousands comma and 9.5px: "2230" is ~23px wide against a
                # 26px bar pitch, so adjacent labels cannot collide. A comma'd
                # label at 10.5px overflowed onto the next column.
                s.append(txt(bx + BAR_W / 2, y(v) - 7, f"{v:.0f}", 9.5, C["ink2"], weight="600"))
        s.append(txt(x + bw / 2, H - B + 20, r["frames"], 12, C["ink2"]))

    s.append(txt((bs[7][1] + bs[7][2] + bs[8][1]) / 2, H - B + 20, "⁄⁄", 12, C["ink3"]))
    s.append(f'<line x1="{L}" x2="{L+pw}" y1="{y(0):.2f}" y2="{y(0):.2f}" '
             f'stroke="{C["rule_strong"]}" stroke-width="1"/>')
    s.append(txt(L + pw / 2, H - 12, "frames per request", 11.5, C["ink3"], family=SANS))
    s.append('</svg>')
    return "\n".join(s)


def fig_share(rows, C, W=1060, H=350):
    L, R, T, B = 66, 18, 56, 52
    pw, ph = W - L - R, H - T - B
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="{C["surface"]}"/>',
         txt(L, 24, "Share of GPU stage time", 16, C["ink"], "start", SANS, "600"),
         txt(L, 41, "Decode dominates at one frame; the encode path is the majority by sixteen",
             11.5, C["ink3"], "start", SANS),
         legend(W - R, 24, C)]

    for t in range(0, 101, 25):
        yy = T + ph * (1 - t / 100)
        s.append(f'<line x1="{L}" x2="{L+pw}" y1="{yy:.2f}" y2="{yy:.2f}" '
                 f'stroke="{C["rule"]}" stroke-width="1"/>')
        s.append(txt(L - 10, yy + 4, f"{t}%", 11, C["ink3"], "end"))

    bs = bands(rows, L, pw)
    for bi, (r, x, bw) in enumerate(bs):
        vals = [float(r[c]) for c, _, _ in SERIES]
        tot = sum(vals)
        bx = x + (bw - BAR_W) / 2
        acc = 0.0
        for si, ((_, _, key), v) in enumerate(zip(SERIES, vals)):
            frac = v / tot
            top = T + ph * (1 - (acc + frac))
            gap = 0 if si == len(SERIES) - 1 else BAR_GAP      # 2px surface gap
            h = max(0.0, ph * frac - gap)
            p = col_path(bx, top + gap, BAR_W, h, 4 if si == 2 else 0)
            if p:
                s.append(f'<path d="{p}" fill="{C[key]}"/>')
            if key == "dec":
                dec_mid = acc + frac / 2      # midpoint of the TOP segment
            acc += frac
        if bi in (0, len(bs) - 1):        # label only the two ends of the inversion
            s.append(txt(x + bw / 2, T + ph * (1 - dec_mid) + 4,
                         f"{100 * vals[2] / tot:.0f}%", 10.5, C["on_fill"], weight="600"))
        s.append(txt(x + bw / 2, H - B + 20, r["frames"], 12, C["ink2"]))

    s.append(txt((bs[7][1] + bs[7][2] + bs[8][1]) / 2, H - B + 20, "⁄⁄", 12, C["ink3"]))
    s.append(f'<line x1="{L}" x2="{L+pw}" y1="{T+ph:.2f}" y2="{T+ph:.2f}" '
             f'stroke="{C["rule_strong"]}" stroke-width="1"/>')
    s.append(txt(L + pw / 2, H - 12, "frames per request", 11.5, C["ink3"], family=SANS))
    s.append('</svg>')
    return "\n".join(s)


def to_png(svg_path, png_path, w, h, scale=2):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        print("   (no chrome found -- skipping PNG)", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as td:
        html = os.path.join(td, "f.html")
        with open(html, "w") as f:
            f.write(f'<body style="margin:0">{open(svg_path).read()}</body>')
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", f"--force-device-scale-factor={scale}",
                        f"--window-size={w},{h}", f"--screenshot={png_path}",
                        f"--user-data-dir={td}/prof", f"file://{html}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return os.path.exists(png_path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here, "results", "framesweep.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "results"))
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    for r in rows:
        r["frames"] = int(r["label"].lstrip("f"))
    rows.sort(key=lambda r: r["frames"])
    C = THEME[a.theme]
    suffix = "" if a.theme == "light" else "_dark"
    os.makedirs(a.outdir, exist_ok=True)

    for name, builder, (w, h) in (("fig_stage_latency", fig_latency, (1060, 430)),
                                  ("fig_stage_share", fig_share, (1060, 350))):
        svg = os.path.join(a.outdir, f"{name}{suffix}.svg")
        with open(svg, "w") as f:
            f.write(builder(rows, C))
        print(f"-- {svg}")
        if a.png:
            png = os.path.join(a.outdir, f"{name}{suffix}.png")
            if to_png(svg, png, w, h):
                print(f"-- {png}")


if __name__ == "__main__":
    sys.exit(main() or 0)
