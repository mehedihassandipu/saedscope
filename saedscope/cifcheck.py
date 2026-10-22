"""Does this pattern match this structure? A focused comparison of a diffraction pattern with one or more CIFs.

For each structure we report a verdict (consistent / consistent with a lattice change / partly / not consistent),
every predicted reflection next to the measured one, strong reflections that are missing, measured rings or spots the
structure cannot explain, and the lattice parameters refined from the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crystal import Phase, fmt_index
from .rings import score_phase
from .spots import analyse_spots

WINDOW = 0.15  # scale changes up to ±15% are searched: real lattice changes (≤ ~6%) and calibration errors
LATTICE_MAX = 6.0  # % beyond which a lattice change is implausible and the calibration is the likely cause


@dataclass
class CifCheck:
    phase: Phase
    mode: str  # "rings" | "spots" | "none"
    quality: float
    verdict: str  # "consistent" | "lattice change" | "partly" | "not consistent"
    headline: str
    explanation: str
    lattice_change_pct: float | None
    refined_cell: dict | None
    table: list[dict] = field(default_factory=list)
    unexplained: list[dict] = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"phase": self.phase.name, "source": self.phase.source, "mode": self.mode, "quality": round(self.quality, 3),
                "verdict": self.verdict, "headline": self.headline, "explanation": self.explanation,
                "lattice_change_pct": None if self.lattice_change_pct is None else round(self.lattice_change_pct, 2),
                "refined_cell": self.refined_cell, "metrics": self.metrics, "zones": self.zones[:3]}


def _cell(phase: Phase, scale: float) -> dict:
    lat = phase.structure.lattice
    out = {"a_cif": round(lat.a, 4), "b_cif": round(lat.b, 4), "c_cif": round(lat.c, 4),
           "a": round(lat.a * scale, 4), "b": round(lat.b * scale, 4), "c": round(lat.c * scale, 4),
           "alpha": round(lat.alpha, 2), "beta": round(lat.beta, 2), "gamma": round(lat.gamma, 2)}
    out["cubic"] = phase.crystal_system == "cubic"
    return out


def _verdict(quality: float, dev: float | None, calibrated: bool, cal_source: str, explained: float, strong_seen: float):
    """Turn the numbers into a verdict and plain-language reasons."""
    if quality >= 0.72 and explained >= 0.75 and strong_seen >= 0.7:
        if dev is None or abs(dev) <= 1.5:
            return "consistent", ""
        if abs(dev) <= LATTICE_MAX:
            why = ("the calibration (a scale bar can be several % off; check with a gold standard at the same camera length), "
                   if cal_source in ("scale bar", "camera length", "file metadata") else "")
            return "lattice change", (f"Possible causes: {why}a different composition (alloying shifts the lattice), "
                                      "strain, or temperature.")
        return "scale mismatch", (f"A {abs(dev):.0f}% lattice change is more than strain or alloying can produce, so the calibration is "
                                  "probably off. Calibrate with a gold (or aluminium) ring pattern at the same camera length.")
    if quality >= 0.55 and explained >= 0.5:
        return "partly", "Some measured features are not explained: a second phase, a different structure variant, or a wrong calibration."
    return "not consistent", ""


def _texture_zone(phase: Phase, seen_d: list[float], missing_d: list[float]):
    """A common zone axis containing every seen reflection but none of the missing strong ones (preferred orientation)."""
    if not seen_d or not missing_d:
        return None
    from .crystal import reflections

    ref = reflections(phase, 1.05 / min(seen_d + missing_d))
    groups = lambda d: ref.hkl[np.abs(1 / np.maximum(ref.g, 1e-9) - d) / d < 0.004]
    seen_g = [groups(d) for d in seen_d]
    miss_g = [groups(d) for d in missing_d]
    if any(len(g) == 0 for g in seen_g + miss_g):
        return None
    for uvw in [(0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 1, 2), (1, 1, 2), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)]:
        u = np.array(uvw)
        if all((g @ u == 0).any() for g in seen_g) and not any((g @ u == 0).any() for g in miss_g):
            return fmt_index(uvw, "[]")
    return None


def check_ring_pattern(res, phase: Phase) -> CifCheck | None:
    rings = [r for r in res.rings if r.kind == "sharp"]
    if len(rings) < 2:
        return None
    cal = res.calibration.inv_A_per_px if res.calibration.calibrated else None
    ps = score_phase(phase, res.rings, cal, tol=res.options.ring_tol, scale_window=WINDOW)
    if ps is None:
        return None
    c_fit = ps.calib
    if len(ps.matched) >= 2:  # least-squares lattice refinement on the matched rings (the search itself prefers the given scale)
        Rm = np.array([res.rings[i].radius for i, _, _ in ps.matched])
        gm = np.array([ps.lines[j].g for _, j, _ in ps.matched])
        c_ref = float(np.sum(Rm * gm) / np.sum(Rm * Rm))
        if not cal or abs(c_ref / cal - 1) <= WINDOW + 0.02:
            c_fit = c_ref
    c_use = cal or c_fit
    dev = (c_fit / cal - 1) * 100 if cal else None  # + means the measured lattice is larger than the CIF
    R = np.array([r.radius for r in rings])
    g_lo, g_hi = R.min() * c_fit * 0.95, R.max() * c_fit * 1.02
    idx_map = [k for k, r in enumerate(res.rings) if r.kind == "sharp"]
    by_line = {j: (i_full, e) for i_full, j, e in ps.matched}
    table = []
    for j, ln in enumerate(ps.lines):
        if not (g_lo <= ln.g <= g_hi) or ln.intensity < 2:
            continue
        row = {"hkl": ln.label, "d CIF (Å)": round(ln.d, 4), "relative intensity (CIF)": round(ln.intensity, 1)}
        if j in by_line:
            i_full = by_line[j][0]
            r = res.rings[i_full]
            d_meas = 1 / (r.radius * c_use)
            row.update({"seen": "yes", "ring": i_full + 1, "d measured (Å)": round(d_meas, 4),
                        "error (%)": round((d_meas / ln.d - 1) * 100, 2),
                        "error after lattice refinement (%)": round((1 / (r.radius * c_fit) / ln.d - 1) * 100, 2)})
        else:
            row.update({"seen": "no, strong" if ln.intensity >= 12 else "no, weak"})
        table.append(row)
    matched_rings = {i for i, _, _ in ps.matched}
    unexplained = [{"ring": i + 1, "d measured (Å)": round(1 / (res.rings[i].radius * c_use), 4), "S/N": round(res.rings[i].snr, 1)}
                   for i in idx_map if i not in matched_rings]
    strong = [t for t in table if t["relative intensity (CIF)"] >= 12]
    strong_seen = sum(1 for t in strong if t["seen"] == "yes") / max(len(strong), 1)
    explained = len(matched_rings) / len(rings)
    resid = [abs(t["error after lattice refinement (%)"]) for t in table if t.get("seen") == "yes"]
    texture = None
    if strong_seen < 0.7 and explained >= 0.75:
        texture = _texture_zone(phase, [t["d CIF (Å)"] for t in table if t["seen"] == "yes"],
                                [t["d CIF (Å)"] for t in table if t["seen"] == "no, strong"])
    verdict, why = _verdict(ps.score, dev, cal is not None, res.calibration.source, explained, 1.0 if texture else strong_seen)
    if texture and verdict in ("partly", "not consistent") and explained >= 0.75 and ps.score >= 0.55:
        verdict, why = _verdict(1.0, dev, cal is not None, res.calibration.source, 1.0, 1.0)
    # intensity sanity check: the brightest measured ring should not be a reflection this structure predicts to be weak
    weak_note = ""
    if ps.matched:
        i_top, j_top, _ = max(ps.matched, key=lambda m: res.rings[m[0]].height)
        ln_top = ps.lines[j_top]
        strongest_I = max((ps.lines[j].intensity for _, j, _ in ps.matched), default=100.0)
        if ln_top.intensity < 15 and strongest_I >= 30 and verdict in ("consistent", "lattice change", "scale mismatch", "partly"):
            verdict = "partly"
            weak_note = (f" The brightest measured ring (ring {i_top + 1}) would be {phase.name} {ln_top.label}, which this structure "
                         f"predicts to be weak ({ln_top.intensity:.0f}% of the strongest), so the intensities do not fit.")
            why = why or ""
    n_seen = sum(1 for t in table if t["seen"] == "yes")
    cell = _cell(phase, c_fit / cal) if cal else None
    tex_note = (f" The reflections that are seen all lie in the {texture} zone and the missing strong ones do not: the crystals share "
                f"a common {texture} orientation along the beam (preferred orientation, e.g. cubes lying on a face)." if texture else "")
    name = phase.name
    if verdict == "consistent":
        head = f"Consistent with {name}"
        expl = (f"{len(matched_rings)} of {len(rings)} measured rings are {name} reflections"
                + (f", d-spacings within {np.mean(resid):.2f}% on average" if resid else "")
                + (f", lattice within {abs(dev):.1f}% of the CIF." if dev is not None else " (ratios only: no calibration)."))
    elif verdict == "lattice change":
        head = f"Consistent with {name} if its lattice is {abs(dev):.1f}% {'larger' if dev > 0 else 'smaller'} than in the CIF"
        expl = (("The ring ratios fit (missing strong reflections are explained by orientation, below)" if texture else
                 "All strong reflections are present and the ring ratios fit") + f", but every d-spacing is offset by {dev:+.1f}%"
                + (f": refined a = {cell['a']:.4f} Å (CIF {cell['a_cif']:.4f} Å)." if cell and cell["cubic"] else ".") + " " + why)
    elif verdict == "scale mismatch":
        head = f"Matches {name} only with a {abs(dev):.0f}% scale change: check the calibration"
        expl = f"The ring ratios fit {name} ({len(matched_rings)} of {len(rings)} rings), but every d-spacing is offset by {dev:+.1f}%. {why}"
    elif verdict == "partly":
        head = f"Partly consistent with {name}" + (f" (lattice {abs(dev):.1f}% {'larger' if dev > 0 else 'smaller'})" if dev is not None and abs(dev) > 1.5 else "")
        n_miss = sum(1 for t in strong if t['seen'] != 'yes')
        expl = (f"{len(matched_rings)} of {len(rings)} rings fit" + (f", {n_miss} strong CIF reflection(s) are missing." if n_miss else ".")
                + (f" {why}" if why and not weak_note else ""))
    else:
        head = f"Not consistent with {name}"
        expl = f"Only {len(matched_rings)} of {len(rings)} rings fit and {sum(1 for t in strong if t['seen'] != 'yes')} strong CIF reflection(s) are missing."
    if verdict != "not consistent":
        expl = expl + tex_note
    expl = expl + weak_note
    return CifCheck(phase=phase, mode="rings", quality=float(ps.score), verdict=verdict, headline=head, explanation=expl.strip(),
                    lattice_change_pct=dev, refined_cell=cell, table=table, unexplained=unexplained,
                    metrics={"rings_explained": f"{len(matched_rings)}/{len(rings)}", "strong_reflections_seen": f"{sum(1 for t in strong if t['seen'] == 'yes')}/{len(strong)}",
                             "reflections_compared": n_seen, "mean_residual_pct": None if not resid else round(float(np.mean(resid)), 3),
                             **({"preferred orientation": texture} if texture else {})})


def check_spot_pattern(res, phase: Phase) -> CifCheck | None:
    sr = res.spot_result
    if sr is None or sr.fit is None or len(sr.spots.yx) < 4:
        return None
    cal = res.calibration.inv_A_per_px if res.calibration.calibrated else None
    out = analyse_spots(sr.spots, sr.centre, [phase], cal, res.image.shape, res.profile.r_min,
                        tol_d=res.options.spot_tol_d, tol_ang=res.options.spot_tol_angle, scale_window=WINDOW,
                        visible=getattr(sr, "visible", None))
    zones = [{"zone_axis": fmt_index(s.uvw_family, "[]"), "score": round(s.score, 3),
              "spots_explained_pct": round(100 * s.frac_explained, 1),
              "lattice_change_pct": None if s.scale_dev_pct is None else round(s.scale_dev_pct, 2)} for s in out.solutions[:5]]
    if not out.solutions:
        return CifCheck(phase=phase, mode="spots", quality=0.0, verdict="not consistent", headline=f"Not consistent with {phase.name}",
                        explanation="No zone axis of this structure reproduces the measured spot spacings and angle.",
                        lattice_change_pct=None, refined_cell=None, zones=[])
    s = out.solutions[0]
    # spots: calib fitted (Å⁻¹/px) vs nominal; d_meas/d_cif = c_fit/cal
    dev = (s.calib / cal - 1) * 100 if cal else None
    c_use = cal or s.calib
    table = []
    for rec in s.indexed:
        r_px = float(np.hypot(rec["dx"], rec["dy"]))
        d_meas = 1 / (r_px * c_use) if r_px > 0 else None
        if rec["hkl"]:
            table.append({"spot": rec["spot"] + 1, "hkl": rec["hkl"], "d CIF (Å)": round(rec["d_ref"], 4),
                          "d measured (Å)": None if d_meas is None else round(d_meas, 4),
                          "error (%)": None if d_meas is None else round((d_meas / rec["d_ref"] - 1) * 100, 2),
                          "allowed": "yes" if rec["allowed"] else "no (double diffraction?)"})
    unexplained = [{"spot": rec["spot"] + 1, "d measured (Å)": round(1 / (np.hypot(rec["dx"], rec["dy"]) * c_use), 4)}
                   for rec in s.indexed if not rec["hkl"]]
    missing = [p for p in s.predicted if p["allowed"] and not p["observed"] and p["rel_I"] >= 0.15]
    explained = s.frac_explained
    strong_seen = 1 - s.frac_missing
    verdict, why = _verdict(s.score, dev, cal is not None, res.calibration.source, explained, strong_seen)
    zone = fmt_index(s.uvw_family, "[]")
    name = phase.name
    if verdict == "consistent":
        head = f"Consistent with {name} along {zone}"
        expl = (f"{fmt_index(s.h1)} and {fmt_index(s.h2)}: d = {s.d1_meas:.3f} / {s.d2_meas:.3f} Å (CIF {s.d1_ref:.3f} / {s.d2_ref:.3f} Å), "
                f"angle {s.angle_meas:.1f}° (CIF {s.angle_ref:.1f}°); {100 * explained:.0f}% of spot intensity explained"
                + (f"; lattice within {abs(dev):.1f}% of the CIF." if dev is not None else "; ratios only (no calibration)."))
    elif verdict == "lattice change":
        head = f"Consistent with {name} along {zone} if its lattice is {abs(dev):.1f}% {'larger' if dev > 0 else 'smaller'}"
        expl = f"The spot geometry fits {zone} but all spacings are offset by {dev:+.1f}%. {why}"
    elif verdict == "scale mismatch":
        head = f"Matches {name} along {zone} only with a {abs(dev):.0f}% scale change: check the calibration"
        expl = f"The spot geometry (ratios and angle) fits {zone}, but all spacings are offset by {dev:+.1f}%. {why}"
    elif verdict == "partly":
        head = f"Partly consistent with {name} (best: {zone})"
        expl = f"{100 * explained:.0f}% of spot intensity explained; {len(missing)} strong predicted spot(s) missing. {why}"
    else:
        head = f"Not consistent with {name}"
        expl = (f"The best zone of this structure, {zone}, fits poorly (score {s.score:.2f}: {100 * explained:.0f}% of spot intensity on "
                f"predicted positions, {len(missing)} strong predicted spot(s) missing"
                + (f", {sum(1 for t in table if t['allowed'] != 'yes')} on forbidden positions" if any(t['allowed'] != 'yes' for t in table) else "")
                + (f", spacings off by {dev:+.1f}%" if dev is not None and abs(dev) > 1.5 else "") + ").")
    return CifCheck(phase=phase, mode="spots", quality=float(s.score), verdict=verdict, headline=head, explanation=expl.strip(),
                    lattice_change_pct=dev, refined_cell=_cell(phase, s.calib / cal) if cal else None, table=table,
                    unexplained=unexplained, zones=zones,
                    metrics={"zone_axis": zone, "spots_explained_pct": round(100 * explained, 1), "strong_spots_missing": len(missing),
                             "forbidden_spots": sum(1 for t in table if t["allowed"] != "yes")})


def check_row(res, phase: Phase) -> CifCheck | None:
    """One systematic row: does the structure have a reflection at the row's d-spacing?"""
    row = res.row
    if row is None or not res.calibration.calibrated:
        return None
    from .crystal import ring_lines

    d = 1.0 / (row.g_px * res.calibration.inv_A_per_px)
    lines = ring_lines(phase, g_max=1.2 / d)
    if not lines:
        return None
    best = min(lines, key=lambda ln: abs(d / ln.d - 1))
    dev = (d / best.d - 1) * 100
    if abs(dev) <= 1.5:
        verdict, head = "consistent", f"Consistent with {phase.name}: the row is {best.label} (d = {best.d:.3f} Å)"
    elif abs(dev) <= LATTICE_MAX:
        verdict, head = "lattice change", f"Consistent with {phase.name} ({best.label}) if its lattice is {abs(dev):.1f}% {'larger' if dev > 0 else 'smaller'}"
    else:
        verdict, head = "not consistent", f"Not consistent with {phase.name}: no reflection near d = {d:.3f} Å"
    expl = (f"One systematic row, orders {', '.join(str(o) for o in sorted(set(row.orders)))}, spacing d = {d:.3f} Å; "
            f"{phase.name} {best.label} has d = {best.d:.3f} Å ({dev:+.1f}%). A single row fits many structures, so this is a weak test: "
            "tilt to a zone axis for a decisive check.")
    return CifCheck(phase=phase, mode="row", quality=max(0.0, 1 - abs(dev) / 6), verdict=verdict, headline=head, explanation=expl,
                    lattice_change_pct=dev, refined_cell=None,
                    table=[{"hkl": best.label, "d CIF (Å)": round(best.d, 4), "d measured (Å)": round(d, 4), "error (%)": round(dev, 2)}],
                    metrics={"row orders seen": ", ".join(str(o) for o in sorted(set(row.orders)))})


ORDER = {"consistent": 0, "lattice change": 1, "scale mismatch": 2, "partly": 3, "not consistent": 4}


def check_cifs(res, phases: list[Phase]) -> list[CifCheck]:
    """Check every given structure against the analysed pattern; best first."""
    kind = res.classification.kind
    subs = set(res.classification.sub_kinds) if kind == "mixed" else {kind}
    out = []
    for ph in phases:
        c = None
        if getattr(res, "row", None) is not None:
            c = check_row(res, ph)
        elif "single crystal" in subs:
            c = check_spot_pattern(res, ph)
        if c is None and ("polycrystalline" in subs or kind in ("uncertain", "mixed")):
            c = check_ring_pattern(res, ph)
        if c is None:
            c = CifCheck(phase=ph, mode="none", quality=0.0, verdict="not consistent", headline=f"Cannot check {ph.name}",
                         explanation="The pattern has no sharp rings or spot lattice to compare (amorphous or too weak).",
                         lattice_change_pct=None, refined_cell=None)
        out.append(c)
    # among structures that all fit, the one needing the smallest lattice change comes first
    out.sort(key=lambda c: (ORDER[c.verdict], abs(c.lattice_change_pct) if c.verdict == "consistent" and c.lattice_change_pct is not None
                            else 0.0, -c.quality))
    same = [c for c in out if c.verdict == "consistent"]
    if res.calibration.calibrated and len(same) > 1:
        devs = ", ".join(f"{c.phase.name} {c.lattice_change_pct:+.1f}%" for c in same if c.lattice_change_pct is not None)
        for c in same:
            c.explanation += (f" Several structures fit within the accuracy of the calibration (lattice change: {devs}). "
                              f"{same[0].phase.name} fits best, but d-spacings alone cannot separate lattices this close "
                              "(about 1%): use the elements present (EDS) to decide.")
    if not res.calibration.calibrated and len(same) > 1:
        names = ", ".join(c.phase.name for c in same)
        for c in same:
            c.explanation += (f" Without a calibration only ring ratios are compared, and {names} give the same ratios, so this "
                              "check cannot tell them apart: calibrate (e.g. with a gold standard at the same camera length) or "
                              "use the elements present.")
    if out and out[0].verdict == "consistent" and out[0].lattice_change_pct is not None:
        for c in out[1:]:
            if c.verdict in ("lattice change", "scale mismatch"):
                c.explanation += (f" Note: {out[0].phase.name} fits the same pattern with no lattice change at this calibration, "
                                  "which is the simpler explanation.")
    return out
