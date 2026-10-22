"""Convert the real-data CSV template to SAED-Bench items.jsonl.

    python -m bench.real_to_jsonl bench/real/items_template.csv bench/real/items.jsonl
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from saedscope.crystal import builtin_library, canonical_uvw, lattice_summary_text

from .common import write_jsonl
from .generate import TYPE_Q, option_text

LIB = {p.name: p for p in builtin_library()}


def main(src: str, dst: str):
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    items = []
    for r in rows:
        c = float(r["calibration_inv_A_per_px"])
        cx, cy = float(r["centre_x"]), float(r["centre_y"])
        cal = (f"Calibration: 1 pixel = {c:.5f} Å⁻¹ in reciprocal space (d = 1/(r·{c:.5f}) Å for a ring or spot at r pixels from the centre). "
               f"The pattern centre (direct beam) is at approximately x = {cx:.0f}, y = {cy:.0f} px.")
        meta = {"phase": r["phase"] or None, "calibration_inv_A_per_px": c, "centre_xy_px": [cx, cy]}
        t = r["task"]
        if t == "pattern_type":
            prompt, ans = TYPE_Q, r["answer"].strip()
        elif t == "ring_phase":
            opts = [o.strip() for o in r["options"].split(";")]
            meta["options"] = opts
            lines = "\n".join(f"{'ABCD'[k]}) {option_text(LIB[o])}" for k, o in enumerate(opts))
            prompt = f"This is a ring (polycrystalline) electron diffraction pattern. {cal}\nWhich of these phases produced the pattern?\n{lines}\nAnswer with the letter only."
            ans = r["answer"].strip().upper()
        elif t == "ring_dspacing":
            prompt = f"This is a ring electron diffraction pattern. {cal}\nWhat is the d-spacing, in ångström, of the innermost (smallest-radius) diffraction ring? Answer with a number."
            ans = float(r["answer"])
        elif t == "zone_axis":
            p = LIB[r["phase"]]
            uvw = tuple(int(x) for x in r["answer"].replace("[", "").replace("]", "").split())
            prompt = (f"This is a single-crystal electron diffraction pattern of {option_text(p)}. {cal}\n"
                      "Along which zone axis [u v w] was this pattern recorded? Give the zone axis as three integers, e.g. [0 1 1]. "
                      "Any symmetry-equivalent direction is accepted.")
            ans = list(canonical_uvw(p, np.array(uvw)))
        else:
            raise ValueError(f"unknown task {t}")
        items.append({"id": r["id"], "task": t, "image": r["image"], "prompt": prompt, "answer": ans, "meta": meta})
    write_jsonl(dst, items)
    print(f"wrote {len(items)} items to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
