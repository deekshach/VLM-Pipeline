#!/usr/bin/env python3
"""Every run of the main evaluation, one row per clip and config, for checking the tables by hand.
Python env: /home/nvidia/data/envs/vlm-eval.

    python per_clip_results.py [out.csv=/home/nvidia/data/accuracy/per_clip_results.csv]

Configs and clips as qwen2fps_compare.py (runs_eval, else the earlier runs). Columns: dataset, clip, config, source
(runs_eval / runs / MISSING) and a note on runs not made under the paced, Qwen-resolution protocol (an earlier run of
the same frames: answer valid, timing not; or an earlier run of frames prepare_qres.py regenerated: stale), frames,
TTFT in s (qwen2fps: total minus decode), the answer, and for multiple choice the predicted and gold letter; for
captions the first reference.
"""
import csv, json, os, sys

import qwen2fps_compare as q

QRES = os.environ.get("QRES", "/mnt/external_ssd/VLM-datasets/qres")
FIELDS = ["dataset", "clip", "config", "source", "note", "frames", "ttft_s", "pred_letter", "gold", "correct", "answer",
          "reference"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{q.ACC}/per_clip_results.csv"
    regen = {tuple(l.split()) for l in open(f"{QRES}/affected.txt") if l.strip()}
    msvd_refs = json.load(open(f"{q.ACC}/msvd/captions.json"))
    rows = []
    for ds in q.DATASETS:
        p = f"{q.EVAL}/{ds}/items.json"
        items = json.load(open(p)) if os.path.exists(p) else {}
        for v in q.clips(ds):
            it = items.get(v, {})
            gold = it.get("answer", "")
            ref = msvd_refs[v][0] if ds == "msvd" else (it.get("refs") or [""])[0]
            for c in q.CFGS:
                src = next((s for s in q.SOURCES if os.path.exists(f"{s}/{ds}/{c}/{v}.json")), None)
                r = q.run(ds, c, v) if src else None
                if r is None:
                    if c in ("uni16", "pw12", "pw36") and ds not in ("nextqa", "videomme", "perception", "charades"):
                        continue   # E4 runs on four datasets only
                    rows.append(dict(dataset=ds, clip=v, config=c, source="MISSING"))
                    continue
                j = json.load(open(f"{src}/{ds}/{c}/{v}.json"))
                if "requests" in j:
                    t = j["requests"][0]
                    ttft = (t["total_ms"] - t["decode_ms"]) / 1000
                else:
                    ttft = j["timing"]["ttft_ms"] / 1000
                note = ""
                if src != q.SOURCES[0]:
                    stale = (ds, v, "qwen2fps" if c == "qwen2fps" else "pool") in regen
                    note = "stale: frames since regenerated" if stale else "earlier run, same frames: timing not paced"
                text = r[0].replace("\n", " ")
                rows.append(dict(dataset=ds, clip=v, config=c, source=os.path.basename(src), note=note, frames=r[1],
                                 ttft_s=round(ttft, 2), pred_letter=q.letter(text) if gold else "", gold=gold,
                                 correct=(q.letter(text) == gold) if gold else "", answer=text, reference=ref))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
