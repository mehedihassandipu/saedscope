"""SAED-Bench "hard" split: the same questions under realistic conditions.

    python -m bench.generate_hard --n 40 --out bench/data/v0-hard

What makes it harder (all things a microscopist meets every day):
  * single crystals slightly off the zone axis (0.5–2.5°): spot intensities follow the Laue circle, not the symmetry;
  * strong dynamical intensity scatter and double-diffraction spots at forbidden positions;
  * projector-lens ellipticity (up to 1.5%) and a nominal calibration that is only good to about ±1.5%;
  * textured or very coarse-grained (spotty) ring patterns, an amorphous carbon support-film halo, low counts;
  * a new task, spot_phase: name the phase of a single-crystal pattern from four options.
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
from saedscope.advisor import zone_geometry
from saedscope.crystal import builtin_library, canonical_uvw, fmt_index, ring_lines

from .common import write_jsonl
from .generate import TYPE_Q, distractors, option_text, to_png, zone_choices


def zone_lookalike(true_phase, true_uvw, other, tol_d=0.035, tol_a=3.0) -> bool:
    """Does any low-index zone of ``other`` give (nearly) the same 2D lattice as the true pattern?"""
    gt = zone_geometry(true_phase, true_uvw)
    if gt is None:
        return False
    for uvw in itertools.product(range(-3, 4), repeat=3):
        if uvw == (0, 0, 0) or sum(abs(x) for x in uvw) > 6 or np.gcd.reduce(np.abs(uvw)) != 1:
            continue
        g = zone_geometry(other, uvw)
        if g is None:
            continue
        for d1, d2 in ((g["d1"], g["d2"]), (g["d2"], g["d1"])):
            if abs(d1 / gt["d1"] - 1) < tol_d and abs(d2 / gt["d2"] - 1) < tol_d and abs(g["angle"] - gt["angle"]) < tol_a:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="items per task")
    ap.add_argument("--out", default="bench/data/v0-hard")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    lib = list(builtin_library())
    items = []
    size = args.size
    g_vis = 0.5 * size * (1.1 / (0.46 * size))

    def save(img, iid):
        iio.imwrite(out / "images" / f"{iid}.png", to_png(img))
        return f"images/{iid}.png"

    def nominal(truth, err=0.015):
        return truth["calibration"] * (1 + rng.uniform(-err, err))

    def meta_cal(truth, c_nom):
        return {"calibration_inv_A_per_px": c_nom, "calibration_true": truth["calibration"],
                "centre_xy_px": [round(truth["centre"][1], 1), round(truth["centre"][0], 1)]}

    def calib_text(truth, c_nom, exact=False):
        cx, cy = truth["centre"][1], truth["centre"][0]
        acc = "" if exact else " (nominal camera constant, accurate to about ±1.5%)"
        return (f"The image is {size}×{size} pixels. Calibration{acc}: 1 pixel = {c_nom:.5f} Å⁻¹ in reciprocal space "
                f"(d = 1/(r·{c_nom:.5f}) Å for a ring or spot at r pixels from the centre). The pattern centre (direct beam) is "
                f"at approximately x = {cx:.0f}, y = {cy:.0f} px.")

    def ring_kw():
        style = rng.choice(["texture", "spotty", "halo", "mixed"])
        kw = {"counts": rng.choice([0.3, 0.5, 1.0]), "beam_stop": rng.random() < 0.85,
              "ellipticity": (rng.uniform(0.3, 1.5), rng.uniform(0, 180))}
        if style in ("texture", "mixed"):
            kw["texture"] = rng.uniform(0.5, 0.9)
        if style in ("spotty", "mixed"):
            kw["spotty"] = rng.uniform(0.85, 0.96)
        if style in ("halo", "mixed"):
            kw["halo"] = rng.uniform(15, 35)
        return kw

    def spot_kw():
        return {"tilt_deg": rng.uniform(0.5, 2.5), "s0": rng.uniform(0.008, 0.014), "dynamical": rng.uniform(0.5, 0.8),
                "double_diffraction": rng.choice([0, 0, 0.04, 0.08]), "counts": rng.choice([0.3, 0.5, 1.0]),
                "beam_stop": rng.random() < 0.85, "ellipticity": (rng.uniform(0, 1.5), rng.uniform(0, 180))}

    # ------------------------------------------------------------------ pattern type
    for i in range(args.n):
        seed = rng.randrange(10**6)
        kind = ["single crystal", "polycrystalline", "amorphous"][i % 3]
        if kind == "polycrystalline":
            p = rng.choice(lib)
            img, truth = synth.ring_pattern(p, size=size, seed=seed, **ring_kw())
            meta = {"phase": p.name}
        elif kind == "single crystal":
            p = rng.choice(lib)
            zs = zone_choices(p, g_vis)
            uvw = rng.choice(list(zs.values())) if zs else (0, 0, 1)
            img, truth = synth.tilted_spot_pattern(p, uvw, size=size, seed=seed, **spot_kw())
            meta = {"phase": p.name, "zone_axis": list(uvw), "tilt_deg": round(truth["tilt_deg"], 2)}
        else:
            img, truth = synth.amorphous_pattern(size=size, seed=seed, counts=rng.choice([0.3, 0.5]), beam_stop=rng.random() < 0.85)
            meta = {"d_halo": truth["d_halo"]}
        iid = f"htype_{i:03d}"
        items.append({"id": iid, "task": "pattern_type", "image": save(img, iid), "prompt": TYPE_Q, "answer": kind,
                      "meta": {**meta, **meta_cal(truth, truth["calibration"])}})

    # ------------------------------------------------------------------ ring phase (4 options)
    for i in range(args.n):
        seed = rng.randrange(10**6)
        p = rng.choice(lib)
        opts = distractors(p, lib, rng) + [p]
        rng.shuffle(opts)
        img, truth = synth.ring_pattern(p, size=size, seed=seed, **ring_kw())
        c_nom = nominal(truth)
        iid = f"hringphase_{i:03d}"
        lines = "\n".join(f"{'ABCD'[k]}) {option_text(o)}" for k, o in enumerate(opts))
        prompt = (f"This is a ring (polycrystalline) electron diffraction pattern. {calib_text(truth, c_nom)}\n"
                  f"Which of these phases produced the pattern?\n{lines}\nAnswer with the letter only.")
        items.append({"id": iid, "task": "ring_phase", "image": save(img, iid), "prompt": prompt, "answer": "ABCD"[opts.index(p)],
                      "meta": {"phase": p.name, "options": [o.name for o in opts], **meta_cal(truth, c_nom)}})

    # ------------------------------------------------------------------ innermost ring d-spacing (exact calibration)
    first_ok = [p for p in lib if ring_lines(p, 1.0) and ring_lines(p, 1.0)[0].intensity >= 15]
    for i in range(args.n):
        seed = rng.randrange(10**6)
        p = rng.choice(first_ok)
        img, truth = synth.ring_pattern(p, size=size, seed=seed, **ring_kw())
        g_max = 0.5 * size * truth["calibration"] * 1.45
        first = ring_lines(p, g_max, min_rel_intensity=1.0)[0]
        iid = f"hringd_{i:03d}"
        prompt = (f"This is a ring electron diffraction pattern. {calib_text(truth, truth['calibration'], exact=True)}\n"
                  "What is the d-spacing, in ångström, of the innermost (smallest-radius) diffraction ring? Answer with a number.")
        items.append({"id": iid, "task": "ring_dspacing", "image": save(img, iid), "prompt": prompt, "answer": round(first.d, 4),
                      "meta": {"phase": p.name, "hkl": first.label, "first_ring_rel_intensity": round(first.intensity, 1),
                               **meta_cal(truth, truth["calibration"])}})

    # ------------------------------------------------------------------ zone axis (off-axis, distorted)
    for i in range(args.n):
        seed = rng.randrange(10**6)
        for _ in range(20):
            p = rng.choice(lib)
            zs = zone_choices(p, g_vis)
            if len(zs) >= 2:
                break
        fam, uvw = rng.choice(list(zs.items()))
        img, truth = synth.tilted_spot_pattern(p, uvw, size=size, seed=seed, **spot_kw())
        c_nom = nominal(truth)
        iid = f"hzone_{i:03d}"
        prompt = (f"This is a single-crystal electron diffraction pattern of {option_text(p)}, recorded close to (not exactly on) "
                  f"a zone axis. {calib_text(truth, c_nom)}\nWhich zone axis [u v w] is it closest to? Give three integers, e.g. [0 1 1]. "
                  "Any symmetry-equivalent direction is accepted.")
        items.append({"id": iid, "task": "zone_axis", "image": save(img, iid), "prompt": prompt, "answer": list(fam),
                      "meta": {"phase": p.name, "zone_axis_generated": list(uvw), "zone_family": fmt_index(fam, "[]"),
                               "tilt_deg": round(truth["tilt_deg"], 2), **meta_cal(truth, c_nom)}})

    # ------------------------------------------------------------------ spot phase (new task, 4 options)
    for i in range(args.n):
        seed = rng.randrange(10**6)
        for _ in range(20):
            p = rng.choice(lib)
            zs = zone_choices(p, g_vis)
            if zs:
                break
        fam, uvw = rng.choice(list(zs.items()))
        pool = [q for q in distractors(p, lib, rng, k=8) if not zone_lookalike(p, uvw, q)]
        if len(pool) < 3:
            pool += [q for q in lib if q.name != p.name and q not in pool and not zone_lookalike(p, uvw, q)]
        opts = pool[:3] + [p]
        rng.shuffle(opts)
        img, truth = synth.tilted_spot_pattern(p, uvw, size=size, seed=seed, **spot_kw())
        c_nom = nominal(truth)
        iid = f"hspotphase_{i:03d}"
        lines = "\n".join(f"{'ABCD'[k]}) {option_text(o)}" for k, o in enumerate(opts))
        prompt = (f"This is a single-crystal electron diffraction pattern recorded close to a zone axis. {calib_text(truth, c_nom)}\n"
                  f"Which of these phases produced the pattern?\n{lines}\nAnswer with the letter only.")
        items.append({"id": iid, "task": "spot_phase", "image": save(img, iid), "prompt": prompt, "answer": "ABCD"[opts.index(p)],
                      "meta": {"phase": p.name, "options": [o.name for o in opts], "zone_axis_generated": list(uvw),
                               "zone_family": fmt_index(canonical_uvw(p, np.array(uvw)), "[]"), "tilt_deg": round(truth["tilt_deg"], 2),
                               **meta_cal(truth, c_nom)}})

    write_jsonl(out / "items.jsonl", items)
    with open(out / "info.json", "w") as fh:
        json.dump({"version": "0.1-hard", "n_items": len(items), "per_task": args.n, "size": size, "seed": args.seed,
                   "phases": [p.name for p in lib]}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
