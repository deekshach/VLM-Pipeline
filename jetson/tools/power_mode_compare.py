#!/usr/bin/env python3
"""One power mode against MAXN on the main evaluation (scripts/run_eval.sh with POWER_TAG=_<mode>, runs_eval_fa: ViT
flash attention on). Per dataset and arm, over the clips the mode ran that MAXN also has: mean TTFT and energy to the
first token (power_analysis.row, the three rails integrated up to the first token) in both modes, the mode's slow-down
and energy ratio, the speed-up over base inside each mode, and how often the answer matches MAXN's.

    python3 power_mode_compare.py [mode tag=_15w] [table tag=_15w_t15w] [out.md]

The table tag is the same mode with its own GEMM table (run_eval.sh GEMM_TABLE=gemm/table_orin_15w.txt): compared
against the mode's runs on the MAXN table, clip for clip.
"""
import os, statistics as st, sys

os.environ.setdefault("EVAL_RUNS", "/home/nvidia/data/accuracy/runs_eval_fa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_summary as E
import power_analysis as P

ARMS = ["qwen2fps", "base", "uni_arch", "ucw8", "ucwd8", "uni4", "ucwd4", "pw24", "pwd24"]
LADDER = ["base", "l1_sel", "l2_anchor", "l3_align", "l4_winbatch", "l5_f32bias", "l6_table", "l7_actfuse"]


def cell(ds, arm, clip):
    d = P.row(ds, arm, clip, 0)
    return d[0] if d else None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_15w"
    ttag = sys.argv[2] if len(sys.argv) > 2 else "_15w_t15w"
    out = [f"**{tag[1:].upper()} against MAXN** (mean over the clips both modes ran; energy to the first token, "
           "three rails)\n",
           "| dataset | arm | clips | TTFT MAXN s | TTFT mode s | slow-down | J MAXN | J mode | energy ratio | "
           "speed-up over base MAXN / mode | J saved over base MAXN / mode | same answer as MAXN |", "|---|" + "---|" * 11]
    summ = []
    for ds, name in P.DATASETS:
        rows = {}
        for arm in ARMS:
            cc = [c for c in E.clips(ds, arm + tag) if E.run(ds, arm, c)]
            pairs = [(cell(ds, arm, c), cell(ds, arm + tag, c)) for c in cc]
            pairs = [(a, b, c) for (a, b), c in zip(pairs, cc) if a and b]
            if pairs:
                rows[arm] = pairs
        if "base" not in rows:
            continue
        bc = {c: (a, b) for a, b, c in rows["base"]}
        for arm, pairs in rows.items():
            m = lambda f, i: st.mean(p[i][f] for p in pairs)
            t0, t1, e0, e1 = m("ttft_s", 0), m("ttft_s", 1), m("e_ttft", 0), m("e_ttft", 1)
            sp = [(bc[c][0]["ttft_s"] / a["ttft_s"], bc[c][1]["ttft_s"] / b["ttft_s"]) for a, b, c in pairs if c in bc]
            js = [(bc[c][0]["e_ttft"] / a["e_ttft"], bc[c][1]["e_ttft"] / b["e_ttft"]) for a, b, c in pairs if c in bc]
            same = sum(E.letter(a["text"]) == E.letter(b["text"]) if E.letter(a["text"]) else a["text"] == b["text"]
                       for a, b, _ in pairs)
            spx = f"{st.mean(s[0] for s in sp):.2f}× / {st.mean(s[1] for s in sp):.2f}×" if arm != "base" else ""
            jsx = f"{st.mean(s[0] for s in js):.2f}× / {st.mean(s[1] for s in js):.2f}×" if arm != "base" else ""
            out.append(f"| {name} | {arm} | {len(pairs)} | {t0:.1f} | {t1:.1f} | {t1 / t0:.2f}× | {e0:.0f} | {e1:.0f} | "
                       f"{e1 / e0:.2f}× | {spx} | {jsx} | {same}/{len(pairs)} |")
            summ.append((name, arm, t0, t1, e0, e1))

    out += ["\n**TACoS ladder** (cumulative: frame selection, then one switch at a time; mean of the clips)\n",
            "| step | TTFT MAXN s | TTFT mode s | slow-down | step saves MAXN / mode s | J MAXN | J mode |", "|---|" + "---|" * 6]
    prev = None
    for arm in LADDER:
        cc = E.clips("tacos", arm + tag)
        pr = [(cell("tacos", arm, c), cell("tacos", arm + tag, c)) for c in cc]
        pr = [p for p in pr if p[0] and p[1]]
        if not pr:
            continue
        a, b = st.mean(p[0]["ttft_s"] for p in pr), st.mean(p[1]["ttft_s"] for p in pr)
        ea, eb = st.mean(p[0]["e_ttft"] for p in pr), st.mean(p[1]["e_ttft"] for p in pr)
        dd = f"{prev[0] - a:.2f} / {prev[1] - b:.2f}" if prev else ""
        out.append(f"| {arm} | {a:.2f} | {b:.2f} | {b / a:.2f}× | {dd} | {ea:.0f} | {eb:.0f} |")
        prev = (a, b)

    out += [f"\n**{tag[1:].upper()}: MAXN GEMM table against the table retuned for the mode** (same clips)\n",
            "| dataset | arm | clips | TTFT MAXN table s | TTFT mode table s | change | J MAXN table | J mode table | "
            "same answer |", "|---|" + "---|" * 8]
    for ds, name in P.DATASETS:
        for arm in ARMS:
            cc = E.clips(ds, arm + ttag)
            pr = [(cell(ds, arm + tag, c), cell(ds, arm + ttag, c)) for c in cc]
            pr = [p for p in pr if p[0] and p[1]]
            if not pr:
                continue
            a, b = st.mean(p[0]["ttft_s"] for p in pr), st.mean(p[1]["ttft_s"] for p in pr)
            ea, eb = st.mean(p[0]["e_ttft"] for p in pr), st.mean(p[1]["e_ttft"] for p in pr)
            same = sum(p[0]["text"] == p[1]["text"] for p in pr)
            out.append(f"| {name} | {arm} | {len(pr)} | {a:.2f} | {b:.2f} | {100 * (b / a - 1):+.1f}% | {ea:.0f} | {eb:.0f} | "
                       f"{same}/{len(pr)} |")
    text = "\n".join(out) + "\n"
    print(text)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(text)


if __name__ == "__main__":
    main()
