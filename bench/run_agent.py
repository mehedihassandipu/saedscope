"""Run the SAEDscope physics pipeline on SAED-Bench (same PNGs and information as the models get).

    python -m bench.run_agent --data bench/data/v0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from saedscope.crystal import builtin_library, fmt_index
from saedscope.io import Calibration, Pattern, load_pattern
from saedscope.pipeline import Options, analyse

from .common import grade, load_items, write_jsonl

LIB = {p.name: p for p in builtin_library()}


def predict(item: dict, data_dir: Path) -> str:
    pat = load_pattern(str(data_dir / item["image"]))
    m = item["meta"]
    cal = Calibration(inv_A_per_px=m["calibration_inv_A_per_px"], source="user")
    pat = Pattern(pat.image, pat.name, cal)
    task = item["task"]
    if task == "pattern_type":
        res = analyse(pat, Options(advise=False), phases=[])
        k = res.classification.kind
        if k == "mixed":
            k = res.classification.sub_kinds[0]
        return k
    if task == "ring_phase":
        phases = [LIB[n] for n in m["options"]]
        res = analyse(pat, Options(force_kind="polycrystalline", advise=False), phases=phases)
        if res.ring_result and res.ring_result.primary:
            return "ABCD"[m["options"].index(res.ring_result.primary.phase.name)]
        return ""
    if task == "ring_dspacing":
        res = analyse(pat, Options(force_kind="polycrystalline", advise=False), phases=[])
        sharp = [r for r in res.rings if r.kind == "sharp" and r.snr >= 6]
        if not sharp:
            return ""
        r0 = min(sharp, key=lambda r: r.radius)
        return f"{1 / (r0.radius * cal.inv_A_per_px):.4f}"
    if task == "zone_axis":
        res = analyse(pat, Options(force_kind="single crystal", advise=False), phases=[LIB[m["phase"]]])
        if res.spot_result and res.spot_result.solutions:
            u = res.spot_result.solutions[0].uvw_family
            return f"[{u[0]} {u[1]} {u[2]}]"
        return ""
    if task == "spot_phase":
        phases = [LIB[n] for n in m["options"]]
        res = analyse(pat, Options(force_kind="single crystal", advise=False), phases=phases)
        if res.spot_result and res.spot_result.solutions:
            return "ABCD"[m["options"].index(res.spot_result.solutions[0].phase.name)]
        return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data/v0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", default="", help="comma-separated subset of tasks")
    args = ap.parse_args()
    data = Path(args.data)
    items = load_items(data / "items.jsonl")
    if args.tasks:
        keep = set(args.tasks.split(","))
        items = [it for it in items if it["task"] in keep]
    rows = []
    t0 = time.time()
    for it in items:
        t1 = time.time()
        try:
            pred = predict(it, data)
            err = None
        except Exception as exc:  # keep going; count as wrong
            pred, err = "", f"{type(exc).__name__}: {exc}"
        ok = grade(it, pred, phase_lookup=LIB.get)
        rows.append({"id": it["id"], "task": it["task"], "model": "SAEDscope (physics pipeline)", "answer": pred,
                     "truth": it["answer"], "correct": ok, "seconds": round(time.time() - t1, 2), "error": err})
        print(f"{it['id']:16s} {str(pred):22s} truth={str(it['answer']):22s} {'✓' if ok else '✗'}")
    out = Path(args.out) if args.out else data / "results_saedscope.jsonl"
    write_jsonl(out, rows)
    print(f"done in {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
