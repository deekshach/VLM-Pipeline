#!/usr/bin/env python3
"""GPU kernel time of a llama-jetson-vlm nsys trace, by pipeline stage (NVTX range) and kernel class.
With two or more traces (e.g. a switch off and on), prints them side by side.

    python3 nsys_stage_kernels.py trace.sqlite [other.sqlite ...]

A kernel belongs to the innermost stage range (vit, prefill_img, prefill_txt, wc_diff, anchor, flow,
decode) that was open on the launching thread when it was launched.
"""
import collections, os, sqlite3, sys

STAGES = ["vit", "prefill_img", "prefill_txt", "decode", "flow"]
CLASSES = ["attention", "softmax", "attn matmul", "GEMM", "GEMM split-K", "cast", "add", "norm", "rope", "copy",
           "glu", "other"]


def classify(name, gz):
    n = name.lower()
    if "flash_attn" in n: return "attention"
    if "soft_max" in n: return "softmax"
    if any(k in n for k in ("gemm", "kernel2", "mul_mat", "cutlass", "xmma", "splitk")):
        # weight GEMMs take BF16 operands; the ViT's KQ / KQV products without flash attention are F32/TF32
        if "bf16" not in n and "bfloat16" not in n: return "attn matmul"
        return "GEMM split-K" if gz > 1 else "GEMM"   # gridZ > 1 on a weight GEMM = split-K slices
    if "convert" in n: return "cast"
    if "bin_bcast" in n or "add" in n: return "add"
    if "norm" in n: return "norm"
    if "rope" in n: return "rope"
    if "cpy" in n or "set_rows" in n or "get_rows" in n or "concat" in n: return "copy"
    if "glu" in n or "gated" in n or "gelu" in n: return "glu"
    return "other"


def load(path):
    db = sqlite3.connect(path)
    S = dict(db.execute("select id, value from StringIds"))
    ranges = collections.defaultdict(list)   # tid -> [(start, end, name)]
    for text, tid_, start, end, gtid in db.execute(
            "select text, textId, start, end, globalTid from NVTX_EVENTS where end is not null"):
        name = text if text else S.get(tid_)
        if name:
            ranges[gtid].append((start, end, name))
    launches = {cid: (t, gtid) for cid, t, gtid in db.execute(
        "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")}
    out = collections.defaultdict(lambda: collections.Counter())
    count = collections.Counter()
    for cid, ks, ke, sn, gz in db.execute(
            "select correlationId, start, end, demangledName, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        if cid not in launches:
            continue
        t, gtid = launches[cid]
        best = None
        for s, e, name in ranges.get(gtid, ()):
            if s <= t <= e and name in STAGES + ["wc_diff", "anchor"] and (best is None or s >= best[0]):
                best = (s, name)
        stage = best[1] if best else "unattributed"
        out[stage][classify(S[sn], gz)] += (ke - ks) / 1e6
        count[stage] += 1
    return out, count


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__); sys.exit(1)
    data = [load(p) for p in paths]
    names = [os.path.basename(p).rsplit(".", 1)[0] for p in paths]
    stages = [s for s in STAGES + ["unattributed"] if any(s in d[0] for d in data)]
    for st in stages:
        print(f"\n## {st}  (GPU kernel ms)")
        print("| class | " + " | ".join(names) + (" | change |" if len(paths) == 2 else " |"))
        print("|---|" + "---|" * len(names) + ("---|" if len(paths) == 2 else ""))
        for c in CLASSES + ["total"]:
            vals = [sum(d[0][st].values()) if c == "total" else d[0][st].get(c, 0.0) for d in data]
            if max(vals) < 0.05:
                continue
            row = f"| {'**total**' if c == 'total' else c} | " + " | ".join(f"{v:.1f}" for v in vals)
            if len(paths) == 2:
                row += f" | {vals[1] - vals[0]:+.1f} ({(vals[1] / vals[0] - 1) * 100:+.0f}%)" if vals[0] > 0 else " | new"
            print(row + " |")
        print("| kernels | " + " | ".join(str(d[1][st]) for d in data) + (" | |" if len(paths) == 2 else " |"))


if __name__ == "__main__":
    main()
