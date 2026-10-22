"""Generate SAED-Bench items: simulated patterns with exact, physics-checked answers.

    python -m bench.generate --n 40 --out bench/data/v0
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from saedscope import synth
from saedscope.crystal import builtin_library, canonical_uvw, fmt_index, lattice_summary_text, ring_lines
from saedscope.viz import display_image

from .common import write_jsonl

TYPE_Q = ("Is this electron diffraction pattern from a single crystal, a polycrystalline material, or an amorphous material? "
          "Answer with exactly one of: single crystal, polycrystalline, amorphous.")


def to_png(img: np.ndarray) -> np.ndarray:
    im, _ = display_image(img, max_px=10_000)
    return im


def option_text(p) -> str:
    return f"{p.name}, space group {p.spacegroup.split(' ')[0]}, {lattice_summary_text(p)}"


def too_similar(a, b, tol=0.02) -> bool:
    la = [l for l in ring_lines(a, 1.0) if l.intensity >= 20][:6]
    lb = [l.g for l in ring_lines(b, 1.0) if l.intensity >= 5]
    if not la or not lb:
        return False
    hits = sum(1 for l in la if min(abs(g - l.g) / l.g for g in lb) < tol)
    return hits / len(la) >= 0.8


def distractors(true, lib, rng, k=3):
    pool = [p for p in lib if p.name != true.name and not too_similar(true, p) and not too_similar(p, true)]

    def sim(p):
        s = 2 * len(p.elements & true.elements) + (1 if p.crystal_system == true.crystal_system else 0)
        return s + rng.random()

    pool.sort(key=sim, reverse=True)
    top = pool[: max(k + 3, 6)]
    rng.shuffle(top)
    return top[:k]


def zone_choices(phase, g_vis, min_spots=8):
    out = {}
    for uvw in itertools.product(range(-1, 3), repeat=3):
        if uvw == (0, 0, 0) or sum(abs(x) for x in uvw) > 3 or np.gcd.reduce(np.abs(uvw)) != 1:
            continue
        fam = canonical_uvw(phase, np.array(uvw))
        if fam in out:
            continue
        xy, inten, _, _ = synth.zone_spots(phase, uvw, g_vis)
        n = int(np.sum(inten > 0.02 * inten.max())) if len(inten) else 0
        if n >= min_spots:
            out[fam] = uvw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="items per task")
    ap.add_argument("--out", default="bench/data/v0")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    lib = list(builtin_library())
    items = []
    size = args.size

    def save(img, iid):
        path = out / "images" / f"{iid}.png"
        iio.imwrite(path, to_png(img))
        return f"images/{iid}.png"

    def common_meta(truth):
        return {"calibration_inv_A_per_px": truth["calibration"], "centre_xy_px": [round(truth["centre"][1], 1), round(truth["centre"][0], 1)]}

    def calib_text(truth):
        c = truth["calibration"]
        cx, cy = truth["centre"][1], truth["centre"][0]
        return (f"The image is {size}×{size} pixels. Calibration: 1 pixel = {c:.5f} Å⁻¹ in reciprocal space (d = 1/(r·{c:.5f}) Å for a ring "
                f"or spot at r pixels from the centre). The pattern centre (direct beam) is at approximately x = {cx:.0f}, y = {cy:.0f} px.")

    # ------------------------------------------------------------------ pattern type
    for i in range(args.n):
        seed = rng.randrange(10**6)
        kind = ["single crystal", "polycrystalline", "amorphous"][i % 3]
        counts = rng.choice([0.5, 1.0, 2.0])
        bs = rng.random() < 0.8
        if kind == "polycrystalline":
            p = rng.choice(lib)
            img, truth = synth.ring_pattern(p, size=size, seed=seed, spotty=rng.choice([0, 0, 0.3, 0.6, 0.8]), counts=counts, beam_stop=bs)
            meta = {"phase": p.name}
        elif kind == "single crystal":
            p = rng.choice(lib)
            g_vis = 0.5 * size * (1.1 / (0.46 * size))
            zs = zone_choices(p, g_vis)
            uvw = rng.choice(list(zs.values())) if zs else (0, 0, 1)
            img, truth = synth.spot_pattern(p, uvw, size=size, seed=seed, counts=counts, beam_stop=bs,
                                            double_diffraction=rng.random() < 0.3)
            meta = {"phase": p.name, "zone_axis": list(uvw)}
        else:
            img, truth = synth.amorphous_pattern(size=size, seed=seed, counts=counts, beam_stop=bs)
            meta = {"d_halo": truth["d_halo"]}
        iid = f"type_{i:03d}"
        items.append({"id": iid, "task": "pattern_type", "image": save(img, iid), "prompt": TYPE_Q, "answer": kind,
                      "meta": {**meta, **common_meta(truth)}})

    # ------------------------------------------------------------------ ring phase (4 options)
    for i in range(args.n):
        seed = rng.randrange(10**6)
        p = rng.choice(lib)
        opts = distractors(p, lib, rng) + [p]
        rng.shuffle(opts)
        letters = "ABCD"
        img, truth = synth.ring_pattern(p, size=size, seed=seed, spotty=rng.choice([0, 0, 0.3, 0.6]), counts=rng.choice([0.5, 1, 2]),
                                        beam_stop=rng.random() < 0.8)
        iid = f"ringphase_{i:03d}"
        lines = "\n".join(f"{letters[k]}) {option_text(o)}" for k, o in enumerate(opts))
        prompt = (f"This is a ring (polycrystalline) electron diffraction pattern. {calib_text(truth)}\n"
                  f"Which of these phases produced the pattern?\n{lines}\nAnswer with the letter only.")
        items.append({"id": iid, "task": "ring_phase", "image": save(img, iid), "prompt": prompt, "answer": letters[opts.index(p)],
                      "meta": {"phase": p.name, "options": [o.name for o in opts], **common_meta(truth)}})

    # ------------------------------------------------------------------ innermost ring d-spacing
    strong_first = [p for p in lib if ring_lines(p, 1.0) and ring_lines(p, 1.0)[0].intensity >= 25]
    for i in range(args.n):
        seed = rng.randrange(10**6)
        p = rng.choice(strong_first)
        img, truth = synth.ring_pattern(p, size=size, seed=seed, counts=rng.choice([0.5, 1, 2]), beam_stop=rng.random() < 0.8)
        g_vis = 0.5 * size * truth["calibration"] * 1.45
        first = [l for l in ring_lines(p, g_vis, min_rel_intensity=1.0)][0]
        iid = f"ringd_{i:03d}"
        prompt = (f"This is a ring electron diffraction pattern. {calib_text(truth)}\n"
                  "What is the d-spacing, in ångström, of the innermost (smallest-radius) diffraction ring? Answer with a number.")
        items.append({"id": iid, "task": "ring_dspacing", "image": save(img, iid), "prompt": prompt, "answer": round(first.d, 4),
                      "meta": {"phase": p.name, "hkl": first.label, **common_meta(truth)}})

    # ------------------------------------------------------------------ zone axis
    for i in range(args.n):
        seed = rng.randrange(10**6)
        for _ in range(20):
            p = rng.choice(lib)
            g_vis = 0.5 * size * (1.1 / (0.46 * size))
            zs = zone_choices(p, g_vis)
            if len(zs) >= 2:
                break
        fam, uvw = rng.choice(list(zs.items()))
        img, truth = synth.spot_pattern(p, uvw, size=size, seed=seed, counts=rng.choice([0.5, 1, 2]), beam_stop=rng.random() < 0.8,
                                        double_diffraction=rng.random() < 0.25)
        iid = f"zone_{i:03d}"
        prompt = (f"This is a single-crystal electron diffraction pattern of {option_text(p)}. {calib_text(truth)}\n"
                  "Along which zone axis [u v w] was this pattern recorded? Give the zone axis as three integers, e.g. [0 1 1]. "
                  "Any symmetry-equivalent direction is accepted.")
        items.append({"id": iid, "task": "zone_axis", "image": save(img, iid), "prompt": prompt, "answer": list(fam),
                      "meta": {"phase": p.name, "zone_axis_generated": list(uvw), "zone_family": fmt_index(fam, "[]"), **common_meta(truth)}})

    write_jsonl(out / "items.jsonl", items)
    with open(out / "info.json", "w") as fh:
        json.dump({"version": "0.1", "n_items": len(items), "per_task": args.n, "size": size, "seed": args.seed,
                   "phases": [p.name for p in lib]}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
