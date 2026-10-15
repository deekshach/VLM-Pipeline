# Per-shape kernel-time breakdown of the NVTX-marked ViT encodes in trace.sqlite.
import sqlite3, collections, sys
db = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else "trace.sqlite")
S = dict(db.execute("select id, value from StringIds"))
ranges = [(S.get(t, x) if t else x, s, e) for x, t, s, e in db.execute(
    "select text, textId, start, end from NVTX_EVENTS where end is not null")]
ranges = [r for r in ranges if r[0] and r[0].startswith("vit_")]
# kernels by launch time: runtime API start inside the range (correlation id)
rt = dict(db.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
ks = list(db.execute("select correlationId, start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"))
def cls(n, gz=1):
    n = n.lower()
    if "flash_attn" in n: return "attention"
    if "soft_max" in n: return "softmax"
    if gz == 16 and ("gemm" in n or "kernel2" in n): return "attn matmul"   # batched over the 16 heads (KQ, KQV)
    if "gemm" in n or "kernel2" in n or "mul_mat" in n or n.startswith(("ampere_", "sm80_", "sm87_")): return "GEMM"
    if "im2col" in n or "conv" in n and "convert" not in n: return "patch embed"
    return "elementwise"
print("shape | kernels | GPU busy ms | attention ms (%) [softmax, KQ/KQV matmuls] | GEMM ms (%) | elementwise ms (%)")
for name, s, e in ranges:
    sel = [k for k in ks if s <= rt.get(k[0], -1) <= e]
    by = collections.Counter(); grids = collections.Counter()
    for c, ks_, ke, sn, gx, gy, gz in sel:
        cl = cls(S[sn], gz); by[cl] += (ke - ks_) / 1e6
        if "flash_attn" in S[sn].lower(): grids[(gx, gy, gz)] += 1
    tot = sum(by.values())
    f = lambda k: f"{by[k]:.1f} ({by[k]/tot*100:.0f}%)"
    att = by["attention"] + by["softmax"] + by["attn matmul"]
    print(f"{name} | {len(sel)} | {tot:.1f} | {att:.1f} ({att/tot*100:.0f}%) [{by['softmax']:.1f}, {by['attn matmul']:.1f}] | {f('GEMM')} | {f('elementwise')}")
