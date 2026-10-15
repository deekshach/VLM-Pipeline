#!/usr/bin/env python3
"""Compare adaptive frame selection driven by Sintel ground-truth flow vs OFA flow.

    python3 compare_selection.py <gt_dir> <ofa_dir>     (dirs of <video>_adaptive.json)
"""
import glob, json, os, statistics as st, sys

gt_dir, ofa_dir = sys.argv[1], sys.argv[2]


def load(d, v):
    p = f"{d}/{v}_adaptive.json"
    return json.load(open(p)) if os.path.exists(p) else None


def ttft(j):
    t = j["timing"]
    return t["flow_analysis_ms"] + t.get("frame_selection_ms", 0) + t["encode_ms"]


rows = []
for p in sorted(glob.glob(f"{gt_dir}/*_adaptive.json")):
    v = os.path.basename(p)[:-len("_adaptive.json")]
    g, o = load(gt_dir, v), load(ofa_dir, v)
    if not o:
        continue
    fg = [f["index"] for f in g["selected_frames"]]
    fo = [f["index"] for f in o["selected_frames"]]
    near = sum(any(abs(a - b) <= 2 for b in fg) for a in fo)
    rows.append(dict(v=v, cg=g["global_complexity"], co=o["global_complexity"], ng=len(fg), no=len(fo),
                     fg=fg, fo=fo, exact=len(set(fg) & set(fo)), near=near,
                     tg=ttft(g), to=ttft(o), ag=g["timing"]["flow_analysis_ms"], ao=o["timing"]["flow_analysis_ms"]))

print("| video | complexity GT → OFA | frames GT → OFA | GT frames | OFA frames | same | within ±2 | TTFT GT → OFA (ms) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['v']} | {r['cg']:.3f} → {r['co']:.3f} | {r['ng']} → {r['no']} | {r['fg']} | {r['fo']} | "
          f"{r['exact']}/{r['no']} | {r['near']}/{r['no']} | {r['tg']:.0f} → {r['to']:.0f} |")
n = len(rows)
same_n = sum(r["ng"] == r["no"] for r in rows)
print(f"\nvideos: {n}; same frame count: {same_n}/{n}")
print(f"selected frames identical: {sum(r['exact'] for r in rows)}/{sum(r['no'] for r in rows)}; "
      f"within 2 frames of a GT pick: {sum(r['near'] for r in rows)}/{sum(r['no'] for r in rows)}")
print(f"mean frames GT {st.mean(r['ng'] for r in rows):.2f}, OFA {st.mean(r['no'] for r in rows):.2f}")
print(f"mean flow analysis ms: GT {st.mean(r['ag'] for r in rows):.0f}, OFA {st.mean(r['ao'] for r in rows):.0f}")
print(f"mean TTFT (analysis + selection + encode) ms: GT {st.mean(r['tg'] for r in rows):.0f}, OFA {st.mean(r['to'] for r in rows):.0f}")
