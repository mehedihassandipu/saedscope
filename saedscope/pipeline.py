"""End-to-end analysis: pattern → type → phase / zone axis / halo, with notes."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import imageproc as ip
from .classify import Classification, classify
from .crystal import Phase, builtin_library, fetch_materials_project, filter_library, fmt_index, lattice_summary_text, parse_elements
from .io import Calibration, Pattern
from .physics import calibration_from_camera_length, electron_wavelength_A, nearest_neighbour_from_halo
from .rings import RingResult, identify_rings
from .rows import find_systematic_row, match_row, rings_from_spots
from .spots import SpotResult, analyse_spots, fit_lattice


@dataclass
class Options:
    elements: list[str] = field(default_factory=list)
    context: str = ""
    calib_inv_A_per_px: float | None = None  # user override
    voltage_kV: float | None = None
    camera_length_mm: float | None = None
    pixel_size_um: float | None = None
    use_builtin: bool = True
    mp_api_key: str | None = None
    mp_max_e_above_hull: float = 0.1
    user_phases: list[Phase] = field(default_factory=list)
    centre: tuple[float, float] | None = None
    r_min: float | None = None
    ring_tol: float = 0.015
    spot_tol_d: float = 0.03
    spot_tol_angle: float = 2.5
    scale_window: float = 0.035
    auto_invert: bool = True
    force_kind: str | None = None  # override classification
    auto_beamstop: bool = True
    hidden_beam_check: bool = True  # look for a direct beam hidden under the beam stop (checked against the usual centre)
    extra_mask: np.ndarray | None = None  # True = ignore (e.g. scale bar and its label)
    ellipticity: tuple[float, float] | None = None  # (ε %, axis °) from a standard, corrected before analysis
    advise: bool = True  # compute next-step advice (tilts / discriminating rings)


@dataclass
class AnalysisResult:
    name: str
    image: np.ndarray
    inverted: bool
    centre: tuple[float, float]
    valid: np.ndarray
    calibration: Calibration
    profile: ip.Profile
    rings: list[ip.Ring]
    spots: ip.Spots
    classification: Classification
    ring_result: RingResult | None
    spot_result: SpotResult | None
    amorphous: dict | None
    candidates: list[Phase]
    candidate_sources: dict
    notes: list[str]
    timings: dict
    options: Options
    advice: dict | None = None
    calibration_conflict: dict | None = None
    calibration_warning: dict | None = None  # geometry fits only at a very different scale (not used to change the answer)
    row: object | None = None  # rows.SystematicRow when only one row of spots is visible

    # ------------------------------------------------------------------
    @property
    def calib_used(self) -> float | None:
        if self.ring_result and self.ring_result.primary:
            return self.ring_result.primary.calib
        if self.spot_result and self.spot_result.solutions:
            return self.spot_result.solutions[0].calib
        return self.calibration.inv_A_per_px

    def best_phase(self) -> str | None:
        k = self.classification.kind
        if k in ("single crystal",) or (k == "mixed" and "single crystal" in self.classification.sub_kinds and self.spot_result and self.spot_result.solutions):
            if self.spot_result and self.spot_result.solutions:
                return self.spot_result.solutions[0].phase.name
        if self.ring_result and self.ring_result.primary:
            return self.ring_result.primary.phase.name
        return None

    def summary(self) -> dict:
        """JSON-serialisable summary (used by the report, export and LLM explanation)."""
        cal = self.calibration
        out = {
            "file": self.name,
            "pattern_type": self.classification.kind,
            "pattern_type_confidence": round(self.classification.confidence, 2),
            "pattern_type_reasons": self.classification.reasons,
            "pattern_features": self.classification.features,
            "spotty_rings": self.classification.spotty_rings,
            "calibrated": cal.calibrated,
            "calibration_source": cal.source,
            "calibration_inv_A_per_px": cal.inv_A_per_px,
            "voltage_kV": cal.voltage_kV,
            "camera_length_mm": cal.camera_length_mm,
            "centre_px": [round(self.centre[0], 2), round(self.centre[1], 2)],
            "image_inverted": self.inverted,
            "elements": self.options.elements,
            "context": self.options.context,
            "candidates_searched": len(self.candidates),
            "candidate_sources": self.candidate_sources,
            "notes": self.notes,
        }
        if self.ring_result:
            rr = self.ring_result
            out["rings"] = {
                "primary_phase": rr.primary.phase.summary() if rr.primary else None,
                "primary_score": round(rr.primary.score, 3) if rr.primary else None,
                "scale_deviation_pct": None if not rr.primary or rr.primary.scale_dev_pct is None else round(rr.primary.scale_dev_pct, 2),
                "secondary_phase": rr.secondary.phase.summary() if rr.secondary else None,
                "ambiguous": rr.ambiguous,
                "ranking": [s.to_row() for s in rr.ranking[:8]],
                "assignments": [
                    {"ring": a.ring_index + 1, "radius_px": round(a.radius_px, 2), "d_meas_A": None if not np.isfinite(a.d_meas) else round(a.d_meas, 4),
                     "phase": a.phase, "hkl": a.hkl, "d_ref_A": None if a.d_ref is None else round(a.d_ref, 4),
                     "error_pct": None if a.error_pct is None else round(a.error_pct, 2), "ambiguous_with": a.ambiguous_with}
                    for a in rr.assignments
                ],
                "unexplained_rings": [i + 1 for i in rr.unexplained],
                "notes": rr.notes,
            }
        if self.spot_result:
            sr = self.spot_result
            sol = sr.solutions[0] if sr.solutions else None
            out["spots"] = {
                "n_spots_used": sr.n_spots_used,
                "lattice": None if sr.fit is None else {
                    "g1_px": round(sr.fit.len_a, 2), "g2_px": round(sr.fit.len_b, 2), "angle_deg": round(sr.fit.angle_deg, 2),
                    "inlier_fraction": round(float(sr.fit.inliers.mean()), 3), "rms_px": round(sr.fit.rms_px, 3)},
                "best": None if sol is None else {
                    "phase": sol.phase.summary(), "zone_axis": fmt_index(sol.uvw_family, "[]"), "zone_axis_raw": list(sol.uvw_family),
                    "g1": {"hkl": fmt_index(sol.h1), "d_meas_A": round(sol.d1_meas, 4), "d_ref_A": round(sol.d1_ref, 4)},
                    "g2": {"hkl": fmt_index(sol.h2), "d_meas_A": round(sol.d2_meas, 4), "d_ref_A": round(sol.d2_ref, 4)},
                    "angle_meas_deg": round(sol.angle_meas, 2), "angle_ref_deg": round(sol.angle_ref, 2),
                    "score": round(sol.score, 3), "spots_explained_pct": round(100 * sol.frac_explained, 1),
                    "scale_deviation_pct": None if sol.scale_dev_pct is None else round(sol.scale_dev_pct, 2),
                    "forbidden_spots_observed": sum(1 for s in sol.indexed if s["allowed"] is False),
                },
                "ranking": [s.to_row() for s in sr.solutions[:8]],
                "ambiguous": len(sr.solutions) > 1 and sr.solutions[1].score > sr.solutions[0].score - 0.05
                and sr.solutions[1].phase.formula != sr.solutions[0].phase.formula,
                "notes": sr.notes,
            }
        if self.amorphous:
            out["amorphous"] = self.amorphous
        if self.calibration_conflict:
            out["calibration_conflict"] = self.calibration_conflict
        if self.calibration_warning:
            out["calibration_warning"] = self.calibration_warning
        if self.row is not None:
            out["systematic_row"] = self.row.summary()
        if self.advice:
            a = self.advice
            out["next_step"] = {"kind": a.get("kind"), "summary": a.get("summary"),
                                "options": [{k: v for k, v in it.items() if k not in ("rotation",)} for it in a.get("items", [])[:3]]}
        out["timings_s"] = {k: round(v, 2) for k, v in self.timings.items()}
        return out


def get_candidates(opts: Options) -> tuple[list[Phase], dict, list[str]]:
    notes: list[str] = []
    phases: list[Phase] = []
    sources = {"built-in": 0, "Materials Project": 0, "user CIF": 0}
    els = opts.elements
    if opts.use_builtin:
        lib = filter_library(builtin_library(), els) if els else list(builtin_library())
        phases += lib
        sources["built-in"] = len(lib)
    if opts.mp_api_key and els:
        try:
            mp = fetch_materials_project(els, opts.mp_api_key, opts.mp_max_e_above_hull)
            phases += mp
            sources["Materials Project"] = len(mp)
        except Exception as exc:
            notes.append(f"Materials Project lookup failed ({type(exc).__name__}: {exc}). Using the other candidate sources.")
    elif opts.mp_api_key and not els:
        notes.append("Materials Project search needs at least one element; skipped.")
    if opts.user_phases:
        phases += opts.user_phases
        sources["user CIF"] = len(opts.user_phases)
    if not els and not opts.user_phases:
        notes.append("No elements given: searched the whole built-in library. Results among look-alike phases are less reliable; add elements (e.g. from EDS).")
    if els and not phases:
        notes.append(f"No candidate phases found for elements {', '.join(els)}. Add a CIF or connect the Materials Project.")
    return phases, sources, notes


def resolve_calibration(pattern: Pattern, opts: Options) -> Calibration:
    cal = Calibration(**pattern.calibration.__dict__)
    if opts.voltage_kV:
        cal.voltage_kV = opts.voltage_kV
    if opts.camera_length_mm:
        cal.camera_length_mm = opts.camera_length_mm
    if opts.calib_inv_A_per_px:
        cal.inv_A_per_px = opts.calib_inv_A_per_px
        cal.source = "user"
    elif not cal.calibrated and cal.voltage_kV and cal.camera_length_mm and opts.pixel_size_um:
        cal.inv_A_per_px = calibration_from_camera_length(cal.voltage_kV, cal.camera_length_mm, opts.pixel_size_um)
        cal.source = "camera length"
    return cal


def _hidden_lattice_point(sp, guess, mask):
    """Lattice point of the spot lattice that lies under the beam stop, nearest to ``guess`` (None if not clear)."""
    order = np.argsort(-sp.intensity)[:60]
    P, I = sp.yx[order], sp.intensity[order]
    if len(P) < 6:
        return None
    p0 = P[0]
    fit = fit_lattice(P[1:] - p0, I[1:])
    if fit is None or fit.inliers.mean() < 0.6:
        return None
    A, B = np.asarray(fit.a, float), np.asarray(fit.b, float)
    M = np.stack([A, B], axis=1)
    try:
        n0 = np.linalg.solve(M, np.asarray(guess, float) - p0)
    except np.linalg.LinAlgError:
        return None
    step = min(np.linalg.norm(A), np.linalg.norm(B))
    h, w = mask.shape
    near_mask = ndimage.binary_dilation(mask, iterations=max(2, int(0.15 * step)))
    tip = ip.beamstop_tip(mask)
    g = np.asarray(guess, float)
    t = np.asarray(tip, float) if tip is not None else g
    # the direct beam sits at the free end of the beam stop; lattice points further along the rod are hidden too
    best = None
    lim = int(np.ceil(max(np.hypot(*(t - g)) / step, 1.0))) + 3
    for n1 in range(int(np.floor(n0[0])) - lim, int(np.floor(n0[0])) + lim + 1):
        for n2 in range(int(np.floor(n0[1])) - lim, int(np.floor(n0[1])) + lim + 1):
            q = p0 + n1 * A + n2 * B
            d_tip, d_glow = float(np.hypot(*(q - t))), float(np.hypot(*(q - g)))
            if min(d_tip, d_glow) > 1.6 * step:
                continue
            if not (0 <= q[0] < h and 0 <= q[1] < w):
                continue
            iy, ix = int(round(q[0])), int(round(q[1]))
            hidden = 0 <= iy < h and 0 <= ix < w and bool(near_mask[iy, ix])
            seen = float(np.min(np.linalg.norm(sp.yx - q, axis=1))) < 3.0
            # a visible spot cannot be the hidden direct beam; prefer points under the stop, near the glow centre and tip
            cost = d_glow + 0.5 * d_tip + (0.0 if hidden else 0.3 * step)
            if (hidden or not seen) and (best is None or cost < best[0]):
                best = (cost, (float(q[0]), float(q[1])))
    return None if best is None else best[1]


def _quality(res) -> float:
    """How well the analysis explains the pattern (to choose between two centre estimates)."""
    if res.spot_result is not None and res.spot_result.solutions:
        return float(res.spot_result.solutions[0].score)
    if res.ring_result is not None and res.ring_result.primary is not None:
        return float(res.ring_result.primary.score)
    if getattr(res, "row", None) is not None and res.row.matches:
        return 0.5
    return 0.0


def analyse(pattern: Pattern, opts: Options | None = None, phases: list[Phase] | None = None) -> AnalysisResult:
    opts = opts or Options()
    phases_in = phases
    t = {}
    t0 = time.time()
    notes = list(pattern.notes)
    img = ip.normalise(pattern.image)
    inverted = False
    if opts.auto_invert:
        img, inverted = ip.maybe_invert(img)
        if inverted:
            notes.append("Image looked inverted (dark spots on light background) and was inverted automatically.")
    cal = resolve_calibration(pattern, opts)
    if not cal.calibrated:
        notes.append("No calibration: phases are matched on d-spacing ratios only.")

    # 1. centre and profile
    user_valid = ~opts.extra_mask if opts.extra_mask is not None and opts.extra_mask.shape == img.shape else None
    centre = opts.centre or ip.find_centre(img, valid=user_valid, refine_rings=False, use_beamstop=opts.auto_beamstop)
    if opts.ellipticity and abs(opts.ellipticity[0]) >= 0.05:
        from .calibrate import correct_ellipticity

        img = correct_ellipticity(img, centre, opts.ellipticity[0], opts.ellipticity[1])
        notes.append(f"Ellipticity correction applied ({opts.ellipticity[0]:.2f}% at {opts.ellipticity[1]:.0f}°).")
    annot = ip.annotation_mask(img)
    hidden_beam = False
    if opts.centre is None and opts.auto_beamstop and opts.hidden_beam_check:
        bs_mask = ip.find_beamstop(img)
        if bs_mask.any():
            # the beam stop may hide the direct beam: check the centre against the symmetry of the diffuse glow
            c_g, cost_g, cost_0 = ip.glow_centre(img, bs_mask | annot, centre, search=0.18 * min(img.shape))
            move = np.isfinite(cost_g) and cost_0 > 2.0 * cost_g + 0.15 and np.hypot(c_g[0] - centre[0], c_g[1] - centre[1]) > 4
            if move:
                notes.append(f"Direct beam hidden by the beam stop: centre taken from the diffuse glow around it "
                             f"(moved {np.hypot(c_g[0] - centre[0], c_g[1] - centre[1]):.0f} px), then refined on Friedel pairs.")
                centre = c_g
                hidden_beam = True
    valid = ip.auto_mask(img, centre, detect_dark=opts.auto_beamstop)
    if annot.any():
        valid &= ~annot
    if user_valid is not None:
        valid &= user_valid
    prof = ip.build_profile(img, centre, valid, r_min=opts.r_min)
    rings = ip.detect_rings(prof)
    t["centre+profile"] = time.time() - t0

    # 2. spots and a quick lattice check
    t1 = time.time()
    sp = ip.detect_spots(img, centre, valid, r_min=prof.r_min, local_bg=hidden_beam)
    if hidden_beam and len(sp.yx) >= 6:
        # The glow gives the centre only roughly, and a spot lattice is symmetric about every lattice point and
        # half-lattice point, so Friedel pairs alone cannot pick the right one. The direct beam is a lattice point
        # that is hidden: take the lattice point under the beam stop nearest to the glow centre, then refine it.
        c_v = _hidden_lattice_point(sp, centre, bs_mask)
        if c_v is not None:
            c_r, n_r = ip.refine_centre_friedel(ip.detect_spots(img, c_v, valid, r_min=prof.r_min, local_bg=True), c_v, tol=3.0)
            if n_r >= 3 and np.hypot(c_r[0] - c_v[0], c_r[1] - c_v[1]) < 3:
                c_v = c_r
            centre = c_v
            valid = ip.auto_mask(img, centre, detect_dark=opts.auto_beamstop) & ~annot
            if user_valid is not None:
                valid &= user_valid
            sp = ip.detect_spots(img, centre, valid, r_min=prof.r_min, local_bg=True)
            prof = ip.build_profile(img, centre, valid, r_min=opts.r_min)
            rings = ip.detect_rings(prof)
    c_f, npairs = ip.refine_centre_friedel(sp, centre)
    if npairs >= 3 and np.hypot(c_f[0] - centre[0], c_f[1] - centre[1]) < 4 and opts.centre is None:
        centre_spots = c_f
    else:
        centre_spots = centre
    vecs = sp.yx - np.array(centre_spots)
    rr = np.linalg.norm(vecs, axis=1) if len(vecs) else np.zeros(0)
    keep = (rr > 1.2 * prof.r_min) & ~((rr < 2 * prof.r_min) & (sp.snr < 15)) if len(vecs) else np.zeros(0, bool)
    if keep.sum() < 8 and opts.centre is None and not any(r.kind == "sharp" and r.snr >= 8 for r in rings):
        # weak, noisy single-crystal patterns: look deeper, but keep only spots with a Friedel partner (g and −g),
        # which noise peaks almost never have; this also fixes the centre when the first estimate is off
        sp_lo = ip.detect_spots(img, centre, valid, r_min=prof.r_min, k_sigma=4.5)
        c_v, votes = ip.friedel_vote_centre(sp_lo.yx, centre)
        if votes >= 4:
            sp_lo = ip.detect_spots(img, c_v, valid, r_min=prof.r_min, k_sigma=4.5)
            c_r, n_r = ip.refine_centre_friedel(sp_lo, c_v, tol=3.0)
            if n_r >= 4:
                c_v = c_r
            paired = ip.friedel_paired(sp_lo.yx, c_v, tol=3.0)
            rr_lo = np.linalg.norm(sp_lo.yx - np.array(c_v), axis=1)
            paired &= rr_lo > 1.2 * prof.r_min
            if paired.sum() >= 6 and paired.sum() > keep.sum():
                sp = ip.spots_subset(sp_lo, paired)
                centre_spots = c_v
                vecs = sp.yx - np.array(centre_spots)
                keep = np.linalg.norm(vecs, axis=1) > 1.2 * prof.r_min
                notes.append(f"Weak spots: detected at a lower threshold and kept only the {int(paired.sum())} with a Friedel partner "
                             f"(g and −g); centre corrected by {np.hypot(c_v[0] - centre[0], c_v[1] - centre[1]):.1f} px.")
    fit = fit_lattice(vecs[keep][:80], sp.intensity[keep][:80]) if keep.sum() >= 4 else None
    inl_frac = float(fit.inliers.mean()) if fit is not None else None
    n_lat = int(fit.inliers.sum()) if fit is not None else 0
    t["spots"] = time.time() - t1

    # 3. classify
    ring_cv = [ip.ring_uniformity(img, centre, r, valid) for r in rings]
    for r, cv in zip(rings, ring_cv):
        r.uniformity_cv = cv
    fr_frac = None
    if keep.sum() >= 4:
        kv = sp.yx[keep]
        tol_f = max(3.0, 0.02 * float(np.median(np.linalg.norm(kv - np.array(centre_spots), axis=1))))
        fr_frac = float(ip.friedel_paired(kv[:120], centre_spots, tol=tol_f).mean())
    cls = classify(rings, ring_cv, int(keep.sum()), inl_frac, n_lat, friedel_frac=fr_frac)
    if opts.force_kind:
        cls = Classification(opts.force_kind, 1.0, cls.features, ["Pattern type set by the user."] + cls.reasons,
                             sub_kinds=cls.sub_kinds, spotty_rings=cls.spotty_rings, has_halo=cls.has_halo)
    # one systematic row only (crystal off a zone axis): no 2D lattice, no rings, but equally spaced spots through the centre
    row = None
    n_sharp = sum(1 for r in rings if r.kind == "sharp" and r.snr >= 8)
    coarse = n_sharp < 2 and (cls.kind == "polycrystalline" or (cls.kind == "mixed" and "single crystal" not in cls.sub_kinds))
    # "many spots" may just be counting noise around one row (a halo underneath does not change that)
    if not opts.force_kind and (cls.kind in ("amorphous", "uncertain") or coarse) and n_sharp < 2 and len(sp.yx) >= 2:
        row = find_systematic_row(sp, centre_spots, prof.r_min, prof.r_max)
        if row is not None and coarse:
            strong = np.argsort(-(sp.snr * sp.intensity))[:3]
            if sum(1 for k in strong if k in row.members) < 2 or np.median(sp.snr) > 0.5 * np.median(sp.snr[row.members]):
                row = None  # the strongest spots are not on the row, or all spots are equally strong: a real coarse polycrystal
        if row is not None:
            cls = Classification("single crystal", 0.6, {**cls.features, "systematic_row_spots": row.n_spots},
                                 [f"Only one row of {row.n_spots} equally spaced spots through the direct beam (orders "
                                  f"{', '.join(str(o) for o in sorted(set(row.orders)))}): a crystal slightly off a zone axis."]
                                 + [x for x in cls.reasons if "halo" in x], has_halo=cls.has_halo)
    kinds = set(cls.sub_kinds) if cls.kind == "mixed" else {cls.kind}
    if cls.kind == "uncertain":
        kinds = {"polycrystalline"} if any(r.kind == "sharp" for r in rings) else {"amorphous"}

    # 4. candidates
    t2 = time.time()
    if phases is None:
        phases, sources, cnotes = get_candidates(opts)
        notes += cnotes
    else:
        sources = {"provided": len(phases)}
    t["candidates"] = time.time() - t2

    ring_result = spot_result = amorphous = None
    conflict = None
    calib_warning = None
    # A calibration fitted to a standard is good to about 1%; a scale bar or nominal camera length can be several % off.
    # Let the allowed scale error follow the calibration's quality (only when the user kept the default).
    if cal.source == "standard" and opts.scale_window == Options.scale_window:
        opts = dataclasses.replace(opts, scale_window=0.015)
        notes.append("Calibration from a standard: d-spacings may deviate by at most about 1.5% overall.")
    elif cal.source in ("scale bar", "camera length") and opts.scale_window == Options.scale_window:
        # a burnt-in scale bar is a few pixels long and often rounded, and a nominal camera length is often 5-10% off:
        # allow up to ~10% (the report shows the deviation found)
        opts = dataclasses.replace(opts, scale_window=0.10)
        notes.append(f"Calibration from {'a scale bar' if cal.source == 'scale bar' else 'the nominal camera length'}: "
                     "d-spacings may deviate by up to about 10% overall; the deviation found is reported.")
    # 5a. rings
    if "polycrystalline" in kinds:
        t3 = time.time()
        if opts.centre is None and "single crystal" not in kinds:
            centre = ip.refine_centre_profile(img, centre, valid)
        stat = "mean" if cls.spotty_rings else "auto"
        prof = ip.build_profile(img, centre, valid, r_min=opts.r_min, stat=stat)
        rings = ip.detect_rings(prof)
        # projector-lens distortion makes rings elliptical; a few % smears each ring over a range of radii and can split it.
        # Measure it on this pattern's strongest ring and correct the pattern itself (no standard needed).
        sharp_now = [r for r in rings if r.kind == "sharp"]
        if sharp_now and not opts.ellipticity and "single crystal" not in kinds and not cls.spotty_rings:
            from .calibrate import correct_ellipticity, ring_ellipticity

            strongest = max(sharp_now, key=lambda r: r.height)
            eps, phi = ring_ellipticity(img, centre, strongest.radius, strongest.fwhm, valid)
            if eps >= 0.5:
                img = correct_ellipticity(img, centre, eps, phi)
                prof = ip.build_profile(img, centre, valid, r_min=opts.r_min, stat=stat)
                rings = ip.detect_rings(prof)
                notes.append(f"Rings were {eps:.1f}% elliptical (projector distortion, long axis at {phi:.0f}°); corrected before matching.")
        if sum(1 for r in rings if r.kind == "sharp") < 2 and len(sp.yx) >= 8:
            pseudo = rings_from_spots(sp, centre_spots, prof.r_min, prof.r_max)
            if len(pseudo) >= 2:
                rings = pseudo + [r for r in rings if r.kind == "broad"]
                notes.append(f"Too few spots per ring for a smooth profile: {len(pseudo)} rings were built from the radii of "
                             f"{int(sum(1 for _ in sp.yx))} individual spots (nanocrystalline or very coarse-grained sample).")
        for r in rings:
            r.uniformity_cv = ip.ring_uniformity(img, centre, r, valid)
        if phases:
            ring_result = identify_rings(rings, phases, cal.inv_A_per_px, tol=opts.ring_tol, scale_window=opts.scale_window)
        t["rings"] = time.time() - t3
    # 5b. spots
    if row is not None:
        match_row(row, phases or [], cal.inv_A_per_px)
        notes.append("Only one systematic row is visible, so the zone axis cannot be determined; its d-spacing is matched instead.")
    elif "single crystal" in kinds:
        t4 = time.time()
        if phases:
            visible = valid & ~ip.saturated_mask(img)
            spot_result = analyse_spots(sp, centre_spots, phases, cal.inv_A_per_px, img.shape, prof.r_min,
                                        tol_d=opts.spot_tol_d, tol_ang=opts.spot_tol_angle, scale_window=opts.scale_window,
                                        visible=visible)
            spot_result.visible = visible
            if "polycrystalline" not in kinds:
                centre = spot_result.centre
            # calibration conflict: nothing fits at the given calibration, but the spot geometry (ratios and angle) fits
            # perfectly at another scale. Either the calibration is off or the phase is something else.
            best = spot_result.solutions[0].score if spot_result.solutions else 0.0
            if cal.calibrated and best < 0.6 and spot_result.fit is not None:
                free = analyse_spots(sp, centre_spots, phases, None, img.shape, prof.r_min,
                                     tol_d=opts.spot_tol_d, tol_ang=opts.spot_tol_angle, scale_window=opts.scale_window,
                                     visible=visible)
                # several structures may fit the geometry; the one needing the smallest scale change is the most plausible.
                # A calibration more than ~25% off is not a plausible mistake (it would mean another basis assignment),
                # so such fits do not override the calibrated result.
                max_off = 0.25 if cal.source in ("scale bar", "camera length") else 0.15
                good = [q for q in free.solutions if q.score >= max(0.6, 0.85 * free.solutions[0].score, best + 0.2)
                        and abs(q.calib / cal.inv_A_per_px - 1) <= max_off] if free.solutions else []
                # if the phase that fits at the calibration also fits the geometry at that scale, the calibration is not in doubt
                best_phase = spot_result.solutions[0].phase.name if spot_result.solutions else None
                tol_scale = opts.scale_window + opts.spot_tol_d
                trusted = bool(best_phase) and any(q.phase.name == best_phase and abs(q.calib / cal.inv_A_per_px - 1) <= tol_scale
                                                   for q in free.solutions if q.score >= max(0.6, 0.85 * free.solutions[0].score))
                if trusted:
                    good = []
                if not good and not trusted and free.solutions:
                    # too far off to override the answer, but worth a warning: wrong camera, binning or camera length?
                    cands = [q for q in free.solutions if q.score >= max(0.6, 0.85 * free.solutions[0].score, best + 0.2)
                             and abs(q.calib / cal.inv_A_per_px - 1) <= 0.40]
                    if cands:
                        q = min(cands, key=lambda q: abs(q.calib / cal.inv_A_per_px - 1))
                        dev_w = (q.calib / cal.inv_A_per_px - 1) * 100
                        calib_warning = {"phase": q.phase.name, "zone_axis": fmt_index(q.uvw_family, "[]"),
                                         "score_ratio_only": round(q.score, 3), "calibration_off_by_pct": round(dev_w, 1)}
                        notes.append(f"Calibration check: {q.phase.name} {fmt_index(q.uvw_family, '[]')} matches the spot spacings and "
                                     f"angles exactly if the calibration were {dev_w:+.0f}% different. Check that the calibration was "
                                     "measured on this camera, at this camera length and binning.")
                if good:
                    f = min(good, key=lambda q: abs(q.calib / cal.inv_A_per_px - 1))
                    if f is not free.solutions[0]:
                        free.solutions.remove(f)
                        free.solutions.insert(0, f)
                    dev = (f.calib / cal.inv_A_per_px - 1) * 100
                    alts = []
                    if opts.use_builtin:
                        lib = [q for q in builtin_library() if q.name not in {x.name for x in phases}]
                        at_cal = analyse_spots(sp, centre_spots, lib, cal.inv_A_per_px, img.shape, prof.r_min,
                                               tol_d=opts.spot_tol_d, tol_ang=opts.spot_tol_angle, scale_window=opts.scale_window,
                                               visible=visible)
                        alts = [{"phase": a.phase.name, "zone_axis": fmt_index(a.uvw_family, "[]"), "score": round(a.score, 3),
                                 "scale_deviation_pct": None if a.scale_dev_pct is None else round(a.scale_dev_pct, 2)}
                                for a in at_cal.solutions[:6] if a.score >= 0.75][:4]
                    res_conflict = {"phase": f.phase.name, "zone_axis": fmt_index(f.uvw_family, "[]"), "score_ratio_only": round(f.score, 3),
                                    "score_at_calibration": round(best, 3), "calibration_off_by_pct": round(dev, 1),
                                    "fits_at_given_calibration": alts}
                    notes.append(f"Calibration conflict: {f.phase.name} {fmt_index(f.uvw_family, '[]')} fits the spot geometry exactly, "
                                 f"but only if the calibration is {dev:+.1f}% different from the one given."
                                 + (" At the given calibration these fit instead: " + ", ".join(f"{a['phase']} {a['zone_axis']}" for a in alts) + "."
                                    if alts else ""))
                    spot_result = free  # show the geometric indexing; the verdict and table report the conflict
                    spot_result.visible = visible
                    spot_result.notes.append("Indexed from spacing ratios and angles only, because of the calibration conflict.")
                    conflict = res_conflict
        t["zone axis"] = time.time() - t4
    # "rings" that sit exactly at the radii of the indexed spots are the spots themselves (broad spots on a strong diffuse
    # background give such profile peaks): a clean zone-axis pattern is a single crystal, not single crystal + rings
    if (cls.kind == "mixed" and "polycrystalline" in kinds and not opts.force_kind and spot_result is not None
            and spot_result.fit is not None and spot_result.fit.inliers.mean() >= 0.7 and spot_result.fit.inliers.sum() >= 8
            and (not spot_result.solutions or spot_result.solutions[0].score >= 0.6 or calib_warning is not None)):
        rs = np.linalg.norm(sp.yx - np.array(centre_spots), axis=1)
        own = [r for r in rings if r.kind == "sharp" and not np.any(np.abs(rs - r.radius) <= max(3.0, 0.03 * r.radius))]
        if len(own) < 2:
            cls = Classification("single crystal", cls.confidence, cls.features,
                                 [x for x in cls.reasons if "continuous rings" not in x and "Both a single-crystal" not in x]
                                 + ["The ring-like peaks in the radial profile are at the radii of the indexed spots, not separate rings."],
                                 has_halo=cls.has_halo)
            kinds = {"single crystal"}
            ring_result = None
    # no zone axis explains the spots, and there are many: treat them as rings from several small crystals
    good_lattice = (spot_result is not None and spot_result.fit is not None and spot_result.fit.inliers.mean() >= 0.7
                    and spot_result.fit.inliers.sum() >= 8)
    if (row is None and "single crystal" in kinds and ring_result is None and phases and len(sp.yx) >= 8 and not opts.force_kind
            and not good_lattice
            and (spot_result is None or not spot_result.solutions or spot_result.solutions[0].score < 0.5)):
        pseudo = rings_from_spots(sp, centre_spots, prof.r_min, prof.r_max)
        if len(pseudo) >= 2:
            rr = identify_rings(pseudo, phases, cal.inv_A_per_px, tol=max(opts.ring_tol, 0.02), scale_window=opts.scale_window)
            if rr.primary is not None and rr.primary.score >= 0.5:
                ring_result = rr
                rings = pseudo + [r for r in rings if r.kind == "broad"]
                spot_result = None
                cls = Classification("polycrystalline", 0.6, cls.features,
                                     cls.reasons + ["No single zone axis explains the spots; they lie on the rings of several small crystals."],
                                     spotty_rings=True, has_halo=cls.has_halo)
                kinds = {"polycrystalline"}
                notes.append(f"No single zone axis explains the spots; they were treated as rings from several crystals "
                             f"({len(pseudo)} rings from spot radii).")
    # 5c. amorphous halo
    halos = [r for r in rings if r.kind == "broad"]
    if "amorphous" in kinds or cls.has_halo:
        if halos:
            h = max(halos, key=lambda r: r.height)
            info = {"halo_radius_px": round(h.radius, 2), "halo_fwhm_px": round(h.fwhm, 2), "n_halos": len(halos)}
            if cal.calibrated:
                d_h = 1 / (h.radius * cal.inv_A_per_px)
                if 1.2 <= d_h <= 6.0:  # first halos of amorphous solids lie at about 1.5-5 Å
                    info.update({"halo_d_A": round(d_h, 3), "nearest_neighbour_A_approx": round(nearest_neighbour_from_halo(d_h), 3),
                                 "halo_width_inv_A": round(h.fwhm * cal.inv_A_per_px, 4)})
                else:
                    info["note"] = (f"Broad feature at d = {d_h:.2f} Å is outside the range of amorphous halos (1.2-6 Å); "
                                    "probably background or the illuminated area, not interpreted.")
            amorphous = info
        else:
            amorphous = {"halo_radius_px": None}

    # a best match that needs every spacing several % off points at the calibration, not at the phase
    if cal.calibrated and conflict is None:
        top = (spot_result.solutions[0] if spot_result is not None and spot_result.solutions else
               (ring_result.primary if ring_result is not None else None))
        dev = getattr(top, "scale_dev_pct", None) if top is not None else None
        if dev is not None and abs(dev) > 5:
            notes.append(f"Every d-spacing is {abs(dev):.1f}% {'smaller' if dev < 0 else 'larger'} than {top.phase.name} predicts. "
                         f"If the phase is right, the calibration ({cal.source}) is off by about that much: check it with a "
                         "standard at the same camera length.")
    res = AnalysisResult(name=pattern.name, image=img, inverted=inverted, centre=centre, valid=valid, calibration=cal,
                         profile=prof, rings=rings, spots=sp, classification=cls, ring_result=ring_result,
                         spot_result=spot_result, amorphous=amorphous, candidates=phases or [], candidate_sources=sources,
                         notes=notes, timings=t, options=opts, calibration_conflict=conflict, row=row,
                         calibration_warning=calib_warning)
    # 6. what to do next (tilts or rings that separate the remaining candidates)
    if opts.advise:
        t6 = time.time()
        try:
            from .advisor import advise

            res.advice = advise(res)
        except Exception as exc:  # advice is optional; never fail the analysis because of it
            res.notes.append(f"Next-step advice unavailable ({type(exc).__name__}).")
        t["next step"] = time.time() - t6
    t["total"] = time.time() - t0
    if hidden_beam:
        # safety net: the hidden-beam centre must explain the pattern better than the usual centre estimate
        alt = analyse(pattern, dataclasses.replace(opts, hidden_beam_check=False), phases=phases_in)
        zone = lambda r: float(r.spot_result.solutions[0].score) if r.spot_result is not None and r.spot_result.solutions else 0.0
        # keep the moved centre only if it gives a clearly better zone-axis solution than the usual centre
        if not (zone(res) >= 0.5 and zone(res) > zone(alt) + 0.02):
            alt.notes.append("A direct beam hidden under the beam stop was considered, but the usual centre explains the "
                             "pattern better and was kept.")
            alt.timings["total"] = time.time() - t0
            return alt
    return res
