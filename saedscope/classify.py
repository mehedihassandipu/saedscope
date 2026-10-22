"""Training-free classification of a diffraction pattern into
single crystal / polycrystalline / amorphous / mixed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Classification:
    kind: str  # "single crystal" | "polycrystalline" | "amorphous" | "mixed" | "uncertain"
    confidence: float
    features: dict
    reasons: list[str] = field(default_factory=list)
    sub_kinds: list[str] = field(default_factory=list)  # for mixed patterns
    spotty_rings: bool = False
    has_halo: bool = False


def classify(rings, ring_cv, n_spots, lattice_inlier_frac, n_lattice_spots, spot_snr_median=None,
             friedel_frac: float | None = None) -> Classification:
    sharp = [r for r in rings if r.kind == "sharp" and r.snr >= 8]
    broad = [r for r in rings if r.kind == "broad" and r.snr >= 6]
    cvs = [c for c in ring_cv if np.isfinite(c)]
    med_cv = float(np.median(cvs)) if cvs else float("nan")
    feats = {
        "sharp_rings": len(sharp),
        "broad_halos": len(broad),
        "ring_uniformity_cv": None if not np.isfinite(med_cv) else round(med_cv, 3),
        "spots_detected": int(n_spots),
        "spots_on_one_lattice": int(n_lattice_spots),
        "lattice_inlier_fraction": None if lattice_inlier_frac is None else round(float(lattice_inlier_frac), 3),
        "friedel_paired_fraction": None if friedel_frac is None else round(float(friedel_frac), 3),
    }
    reasons = []
    frac = lattice_inlier_frac or 0
    # spots of one crystal come in Friedel pairs (g, -g); spots from many random grains rarely do
    paired = friedel_frac is not None and n_spots >= 6 and friedel_frac >= 0.5
    single = n_lattice_spots >= 6 and (frac >= 0.6 or (n_lattice_spots >= 10 and frac >= 0.45) or (paired and frac >= 0.4))
    poly = len(sharp) >= 2 and not single
    # many spots that are not one lattice = coarse grains (spotty rings); allowed on top of a halo (support film)
    # only when there are clearly too many spots for noise
    coarse_poly = ((not single) and n_spots >= 15 and len(sharp) < 2 and not paired
                   and (len(broad) == 0 or (n_spots >= 25 and frac < 0.4)))
    halo = len(broad) >= 1
    rings_continuous = len(sharp) >= 3 and np.isfinite(med_cv) and med_cv <= 0.3
    if single:
        reasons.append(f"{n_lattice_spots} sharp spots lie on one 2D lattice ({100 * lattice_inlier_frac:.0f}% of detected spots)"
                       + (f"; {100 * friedel_frac:.0f}% of spots have a Friedel partner." if friedel_frac is not None else "."))
    if poly:
        if np.isfinite(med_cv) and med_cv > 0.3:
            reasons.append(f"{len(sharp)} sharp rings with uneven ('spotty') intensity around the circle (uniformity CV {med_cv:.2f}): coarse grains or texture.")
        else:
            reasons.append(f"{len(sharp)} sharp rings with even intensity around the circle (uniformity CV {med_cv:.2f}).")
    if coarse_poly:
        reasons.append(f"{n_spots} spots that do not form a single lattice: likely a coarse-grained polycrystal (spotty rings).")
    if halo:
        if single or poly or coarse_poly:
            reasons.append(f"{len(broad)} broad diffuse halo(s) under the crystalline signal: an amorphous support film, "
                           "surface layer or diffuse background.")
        else:
            reasons.append(f"{len(broad)} broad diffuse halo(s), typical of amorphous material.")

    if single and rings_continuous:
        return Classification("mixed", 0.6, feats, reasons + ["Both a single-crystal spot lattice and continuous rings are present."],
                              sub_kinds=["single crystal", "polycrystalline"], has_halo=halo)
    if single:
        conf = min(0.99, 0.6 + 0.4 * (lattice_inlier_frac or 0)) * (0.85 if halo else 1.0)
        return Classification("single crystal", conf, feats, reasons, has_halo=halo)
    if poly or coarse_poly:
        spotty = coarse_poly or (np.isfinite(med_cv) and med_cv > 0.3)
        if halo and len(sharp) >= 1:
            return Classification("mixed", 0.6, feats, reasons + ["Crystalline rings on top of an amorphous halo: partially crystalline."],
                                  sub_kinds=["polycrystalline", "amorphous"], spotty_rings=spotty, has_halo=True)
        conf = 0.9 if poly and len(sharp) >= 3 else 0.7
        return Classification("polycrystalline", conf, feats, reasons, spotty_rings=spotty, has_halo=halo)
    if halo:
        return Classification("amorphous", 0.85 if len(sharp) == 0 else 0.6, feats, reasons, has_halo=True)
    if len(sharp) == 1:
        return Classification("uncertain", 0.4, feats, reasons + ["Only one sharp ring found; not enough to decide."])
    return Classification("uncertain", 0.3, feats, reasons + ["No clear rings, spots or halos were found."])
