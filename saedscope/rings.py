"""Phase identification for polycrystalline ring patterns.

Each candidate phase is scored by how well its simulated ring d-spacings
explain the measured rings, allowing a small uniform scale factor (DFT
lattice error / calibration error). Without calibration the scale is free,
so only d-spacing *ratios* are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crystal import Phase, RingLine, ring_lines
from .imageproc import Ring


@dataclass
class RingAssignment:
    ring_index: int
    radius_px: float
    d_meas: float
    phase: str | None
    hkl: str | None
    d_ref: float | None
    error_pct: float | None
    ambiguous_with: list[str] = field(default_factory=list)


@dataclass
class PhaseScore:
    phase: Phase
    score: float
    calib: float  # fitted Å⁻¹/px
    scale_dev_pct: float | None  # fitted vs nominal calibration (None if uncalibrated)
    matched: list[tuple[int, int, float]]  # (ring idx, line idx, rel err)
    n_rings: int
    missing_strong: list[RingLine]
    lines: list[RingLine]
    mean_err_pct: float

    def to_row(self) -> dict:
        return {
            "phase": self.phase.name,
            "formula": self.phase.formula,
            "space_group": self.phase.spacegroup,
            "score": round(self.score, 3),
            "rings_matched": f"{len(self.matched)}/{self.n_rings}",
            "mean_error_pct": round(self.mean_err_pct, 2),
            "scale_dev_pct": None if self.scale_dev_pct is None else round(self.scale_dev_pct, 2),
            "missing_strong_lines": ", ".join(f"{l.label} ({l.d:.2f} Å)" for l in self.missing_strong[:4]),
            "source": self.phase.source + (f" {self.phase.source_id}" if self.phase.source == "Materials Project" else ""),
            "e_above_hull": self.phase.e_above_hull,
        }


def _weights(rings: list[Ring]) -> np.ndarray:
    h = np.array([max(r.height, 0) for r in rings])
    w = np.sqrt(h / max(h.max(), 1e-12)) + 0.15
    return w


def _intensity_agreement(rings, matched, lines) -> float:
    """Rank correlation between measured ring heights and reference intensities (0..1)."""
    if len(matched) < 3:
        return 0.5
    hm = np.array([max(rings[i].height, 1e-9) for i, _, _ in matched])
    hr = np.array([max(lines[j].intensity, 1e-3) for _, j, _ in matched])
    a = np.argsort(np.argsort(hm))
    b = np.argsort(np.argsort(hr))
    if a.std() == 0 or b.std() == 0:
        return 0.5
    rho = float(np.corrcoef(a, b)[0, 1])
    return (rho + 1) / 2


def _evaluate(rings: list[Ring], R: np.ndarray, w: np.ndarray, lines: list[RingLine], c: float, tol: float,
              c_nominal: float | None = None, scale_window: float = 0.035):
    g_meas = R * c
    g_ref = np.array([l.g for l in lines])
    I_ref = np.array([l.intensity for l in lines])
    matched = []
    errs = []
    used = set()
    for i, gm in enumerate(g_meas):
        rel = np.abs(g_ref - gm) / g_ref
        j = int(np.argmin(rel))
        tol_i = max(tol, 0.35 * rings[i].rel_width)
        if rel[j] <= tol_i:
            matched.append((i, j, float(rel[j])))
            errs.append(rel[j] / tol_i)
            used.add(j)
    explained = sum(w[i] for i, _, _ in matched) / w.sum()
    g_lo, g_hi = g_meas.min() * 0.97, g_meas.max() * 1.01
    in_range = (g_ref >= g_lo) & (g_ref <= g_hi)
    strong = in_range & (I_ref >= 12)
    missing = [k for k in np.where(strong)[0] if k not in used]
    miss_frac = I_ref[missing].sum() / max(I_ref[strong].sum(), 1e-9) if strong.any() else 0.0
    mean_err = float(np.mean(errs)) if errs else 1.0
    agree = _intensity_agreement(rings, matched, lines)
    score = explained * (1 - 0.55 * miss_frac) * (1 - 0.25 * mean_err) * (0.8 + 0.2 * agree)
    if c_nominal:
        dev = abs(c / c_nominal - 1) / max(scale_window, 1e-6)
        score *= 1 - 0.15 * dev**2
    if len(matched) < 2:
        score *= 0.35
    return score, matched, missing


def _refine_c(R, lines, matched, c):
    if not matched:
        return c
    Ri = np.array([R[i] for i, _, _ in matched])
    gj = np.array([lines[j].g for _, j, _ in matched])
    return float(np.sum(Ri * gj) / np.sum(Ri * Ri))


def score_phase(phase: Phase, rings: list[Ring], calib: float | None, tol: float = 0.015,
                scale_window: float = 0.035, g_max: float | None = None) -> PhaseScore | None:
    sharp = [r for r in rings if r.kind == "sharp"]
    if len(sharp) == 0:
        return None
    R = np.array([r.radius for r in sharp])
    w = _weights(sharp)
    if g_max is None:
        g_max = (R.max() * (calib or 0.003)) * 1.15 if calib else 1.6
    lines = ring_lines(phase, g_max=max(g_max, 0.5))
    lines = [l for l in lines if l.intensity >= 1.0]
    if not lines:
        return None
    best = (-1.0, None, None, None)
    if calib:
        for dlt in np.arange(-scale_window, scale_window + 1e-9, 0.0015):
            c = calib * (1 + dlt)
            s, m, miss = _evaluate(sharp, R, w, lines, c, tol, calib, scale_window)
            if s > best[0]:
                best = (s, c, m, miss)
    else:
        strongest = np.argsort(-w)[:4]
        top_lines = sorted(range(len(lines)), key=lambda k: -lines[k].intensity)[:8]
        for i in strongest:
            for j in top_lines:
                c = lines[j].g / R[i]
                s, m, miss = _evaluate(sharp, R, w, lines, c, tol)
                if s > best[0]:
                    best = (s, c, m, miss)
    s, c, m, miss = best
    if c is None:
        return None
    # refine the scale on matched rings and re-evaluate once
    c2 = _refine_c(R, lines, m, c)
    if calib is None or abs(c2 / calib - 1) <= scale_window + 0.005:
        s2, m2, miss2 = _evaluate(sharp, R, w, lines, c2, tol, calib, scale_window)
        if s2 >= s - 1e-6:
            s, c, m, miss = s2, c2, m2, miss2
    # map matched ring indices back to the full ring list
    idx_map = [k for k, r in enumerate(rings) if r.kind == "sharp"]
    m_full = [(idx_map[i], j, e) for i, j, e in m]
    errs = [e * 100 for _, _, e in m]
    return PhaseScore(
        phase=phase, score=float(s), calib=float(c),
        scale_dev_pct=(c / calib - 1) * 100 if calib else None,
        matched=m_full, n_rings=len(sharp), missing_strong=[lines[k] for k in miss], lines=lines,
        mean_err_pct=float(np.mean(errs)) if errs else float("nan"),
    )


@dataclass
class RingResult:
    ranking: list[PhaseScore]
    primary: PhaseScore | None
    secondary: PhaseScore | None
    assignments: list[RingAssignment]
    unexplained: list[int]
    ambiguous: bool
    calibrated: bool
    notes: list[str]


def identify_rings(rings: list[Ring], phases: list[Phase], calib: float | None, tol: float = 0.015,
                   scale_window: float = 0.035) -> RingResult:
    notes: list[str] = []
    scores = []
    for p in phases:
        try:
            ps = score_phase(p, rings, calib, tol=tol, scale_window=scale_window)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"Could not score {p.name}: {exc}")
            continue
        if ps is not None:
            scores.append(ps)
    scores.sort(key=lambda s: -s.score)
    primary = scores[0] if scores and scores[0].score > 0.15 else None
    secondary = None
    sharp_idx = [k for k, r in enumerate(rings) if r.kind == "sharp"]
    assigned: dict[int, tuple[PhaseScore, int, float]] = {}
    if primary:
        for i, j, e in primary.matched:
            assigned[i] = (primary, j, e)
        unexplained = [k for k in sharp_idx if k not in assigned]
        if unexplained:
            # a second phase explaining the left-over rings, at the same scale
            best2 = None
            for ps in scores[1:]:
                if ps.phase.formula == primary.phase.formula and ps.phase.spacegroup == primary.phase.spacegroup:
                    continue
                lines = ps.lines
                g_ref = np.array([l.g for l in lines])
                expl = []
                for k in unexplained:
                    gm = rings[k].radius * primary.calib
                    rel = np.abs(g_ref - gm) / g_ref
                    j = int(np.argmin(rel))
                    if rel[j] <= max(tol, 0.35 * rings[k].rel_width) * 1.2:
                        expl.append((k, j, float(rel[j])))
                if not expl:
                    continue
                # penalise strong lines of the second phase that are absent
                g_meas_all = np.array([rings[k].radius * primary.calib for k in sharp_idx])
                strong = [l for l in lines if l.intensity >= 40 and g_meas_all.min() * 0.97 <= l.g <= g_meas_all.max()]
                absent = [l for l in strong if np.min(np.abs(g_meas_all - l.g) / l.g) > tol * 1.5]
                val = sum(rings[k].height for k, _, _ in expl) * (1 - 0.6 * len(absent) / max(len(strong), 1))
                if best2 is None or val > best2[0]:
                    best2 = (val, ps, expl)
            if best2 and best2[0] > 0:
                _, ps2, expl = best2
                secondary = ps2
                for k, j, e in expl:
                    assigned[k] = (ps2, j, e)
    unexplained = [k for k in sharp_idx if k not in assigned]

    assignments = []
    for k, r in enumerate(rings):
        if r.kind != "sharp":
            continue
        c = primary.calib if primary else calib
        d_meas = 1 / (r.radius * c) if c else float("nan")
        if k in assigned:
            ps, j, e = assigned[k]
            line = ps.lines[j]
            amb = []
            for other in scores[:6]:
                if other is ps or other.phase.formula == ps.phase.formula:
                    continue
                if any(abs(l.g - line.g) / line.g < tol and l.intensity > 5 for l in other.lines):
                    amb.append(other.phase.name)
            assignments.append(RingAssignment(k, r.radius, d_meas, ps.phase.name, line.label, line.d, (1 / d_meas - line.g) / line.g * 100 if c else None, amb[:2]))
        else:
            assignments.append(RingAssignment(k, r.radius, d_meas, None, None, None, None))

    ambiguous = len(scores) > 1 and primary is not None and (scores[0].score - scores[1].score) < 0.05 and scores[1].phase.formula != scores[0].phase.formula
    if calib is None:
        notes.append("Uncalibrated: phases were matched on d-spacing ratios only; phases with the same structure type but different lattice parameters cannot be distinguished.")
    if primary and primary.scale_dev_pct is not None and abs(primary.scale_dev_pct) > 2.0:
        notes.append(f"All d-spacings are offset by {primary.scale_dev_pct:+.1f}% from the reference. Check the camera-length calibration (or lattice expansion / DFT lattice error).")
    return RingResult(scores, primary, secondary, assignments, unexplained, ambiguous, calib is not None, notes)
