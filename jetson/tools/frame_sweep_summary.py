#!/usr/bin/env python3
"""Summarise scripts/run_frame_sweep.sh: TTFT stages, power, energy and memory vs the number of frames.

    python frame_sweep_summary.py [sweep_dir=experiments/frame_sweep]

Per run (<video>__f<N>.json / .tegrastats / .oc3):
  stages        image load, tokenize, ViT, image prefill, text prefill (they sum to TTFT at baseline)
  power         mean of each rail over the samples with the GPU busy (GR3D_FREQ >= 50%): the encode phase, which is
                almost all of TTFT; module = VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0
  energy        module power x TTFT
  memory        peak RAM in use during the run minus RAM in use before the model loaded (tegrastats first sample)
  oc3           over-current events of the board's hardware throttle during the run
Writes summary.csv and summary.md next to the runs.
"""
import csv, glob, json, os, re, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/frame_sweep")
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def tegra(path):
    ram, busy = [], {r: [] for r in RAILS}
    for l in open(path):
        m = re.search(r"RAM (\d+)/", l)
        g = re.search(r"GR3D_FREQ (\d+)%", l)
        if not (m and g):
            continue
        ram.append(int(m[1]))
        if int(g[1]) >= 50:
            for r in RAILS:
                busy[r].append(int(re.search(rf"{r} (\d+)mW", l)[1]))
    return ram, {r: float(np.mean(v)) / 1000 if v else float("nan") for r, v in busy.items()}


rows = []
for j in glob.glob(f"{D}/*__f*.json"):
    if ".paused" in j:  # moved aside by run_power_mode.sh: the guard paused that request
        continue
    base = j[:-5]
    t = json.load(open(j))["timing"]
    b, r = t["breakdown"], t["breakdown"]["rest"]
    ram, p = tegra(base + ".tegrastats")
    oc = dict(l.split() for l in open(base + ".oc3"))
    row = dict(frames=r["n_images"], tokens=r["tokens_encoded"], ttft_ms=t["ttft_ms"], load_ms=b["rest_image_load_ms"],
               tokenize_ms=b["rest_tokenize_ms"], vit_ms=r["vit_ms"], prefill_img_ms=r["prefill_img_ms"],
               prefill_txt_ms=r["prefill_txt_ms"], gpu_w=p["VDD_GPU_SOC"], cpu_w=p["VDD_CPU_CV"], sys_w=p["VIN_SYS_5V0"],
               module_w=sum(p.values()), ram_peak_gb=max(ram) / 1024, ram_delta_gb=(max(ram) - ram[0]) / 1024,
               oc3_events=int(oc["after"]) - int(oc["before"]))
    row["energy_j"] = row["module_w"] * row["ttft_ms"] / 1000
    rows.append(row)
rows.sort(key=lambda r: r["frames"])
with open(f"{D}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
cols = ["frames", "tokens", "ttft_ms", "vit_ms", "prefill_img_ms", "prefill_txt_ms", "module_w", "gpu_w", "energy_j", "ram_delta_gb",
        "oc3_events"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for r in rows]
open(f"{D}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
