"""Patterns that are not a full zone axis: a single systematic row, and rings made of scattered spots.

* A crystal slightly off a zone axis often shows just one row of reflections through the direct beam (g, 2g, -g ...).
  That gives one d-spacing (very precisely, from all orders) but no zone axis. We measure it and say what to do next.
* Nanocrystalline material gives a few spots per ring rather than smooth rings. Clustering the radii of the spots
  recovers the rings so the usual ring matching can run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crystal import Phase, ring_lines
from .imageproc import Ring, Spots


@dataclass
class SystematicRow:
    g_px: float  # spacing of the row in pixels (first order)
    direction_deg: float  # in the image, from +x, clockwise (y down)
    orders: list[int]
    n_spots: int
    centre_shift_px: float  # along the row: correction of the centre from the Friedel pairs
    residual_px: float
    matches: list[dict] = field(default_factory=list)
    d_A: float | None = None
    members: list[int] = field(default_factory=list)  # indices into the Spots arrays

    def summary(self) -> dict:
        return {"d_A": None if self.d_A is None else round(self.d_A, 4), "spacing_px": round(self.g_px, 2),
                "direction_deg": round(self.direction_deg, 1), "orders_seen": sorted(set(self.orders)), "n_spots": self.n_spots,
                "matches": self.matches[:6]}


def find_systematic_row(spots: Spots, centre, r_min: float, r_max: float, ang_tol_deg: float = 4.0,
                        rel_tol: float = 0.05, min_snr: float = 6.0) -> SystematicRow | None:
    """Strongest row of spots through the centre with (near) equal spacing."""
    if spots is None or len(spots.yx) < 2:
        return None
    v = spots.yx - np.asarray(centre, float)
    r = np.hypot(v[:, 0], v[:, 1])
    use = (r > max(1.2 * r_min, 3.0)) & (r < 0.95 * r_max) & (spots.snr >= min_snr)
    if use.sum() < 2:
        return None
    idx_use = np.where(use)[0]
    v, r, inten = v[use], r[use], spots.intensity[use]
    best = None
    for k in np.argsort(-inten)[:10]:
        u = np.array([v[k, 1], v[k, 0]]) / max(r[k], 1e-9)  # (x, y) unit vector
        s = v[:, 1] * u[0] + v[:, 0] * u[1]  # along the row
        perp = np.abs(-v[:, 1] * u[1] + v[:, 0] * u[0])
        on = perp <= np.maximum(2.0, np.tan(np.radians(ang_tol_deg)) * np.abs(s))
        if on.sum() < 2:
            continue
        s_on, I_on = s[on], inten[on]
        # trial first-order spacing: each of the shortest few |s|, divided by small integers
        trials = sorted({abs(x) / n for x in sorted(np.abs(s_on))[:4] for n in (1, 2)})
        for g0 in trials:
            if g0 < max(1.2 * r_min, 3.0):
                continue
            n = np.rint(s_on / g0).astype(int)
            ok = (n != 0) & (np.abs(s_on - n * g0) <= rel_tol * np.abs(s_on) + 1.0)
            if ok.sum() < 2 or len(set(n[ok])) < 2:
                continue
            # with two or more different orders the centre offset along the row can be fitted too (s = n·g + δ)
            A = np.stack([n[ok].astype(float), np.ones(ok.sum())], axis=1) if len(set(n[ok])) >= 2 else n[ok][:, None].astype(float)
            sol, *_ = np.linalg.lstsq(A, s_on[ok], rcond=None)
            g_fit = float(abs(sol[0]))
            shift = float(sol[1]) if A.shape[1] == 2 else 0.0
            resid = float(np.sqrt(np.mean((s_on[ok] - (A @ sol)) ** 2)))
            missing_low = int(np.min(np.abs(n[ok]))) - 1  # a first order that is never seen means the spacing is a sub-multiple
            gaps = len(set(range(1, int(np.max(np.abs(n[ok]))) + 1)) - set(np.abs(n[ok]).tolist()))
            score = (float(I_on[ok].sum()) * (1 + 0.25 * (len(set(np.abs(n[ok]))) - 1)) / (1 + resid)
                     * 0.4 ** missing_low * 0.8 ** gaps)
            if best is None or score > best[0]:
                members = idx_use[np.where(on)[0][ok]].tolist()
                best = (score, g_fit, u, list(map(int, n[ok])), int(ok.sum()), shift, resid, members)
    if best is None:
        return None
    _, g_fit, u, orders, nsp, shift, resid, members = best
    if not ((max(orders) > 0 and min(orders) < 0) or len(set(abs(o) for o in orders)) >= 2):
        return None
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    return SystematicRow(g_px=g_fit, direction_deg=ang, orders=orders, n_spots=nsp, centre_shift_px=shift, residual_px=resid,
                         members=members)


def match_row(row: SystematicRow, phases: list[Phase], calib: float | None, tol: float = 0.02) -> None:
    """Fill row.d_A and the candidate reflections (hkl families) whose d matches the row's first order."""
    if not calib:
        return
    g = row.g_px * calib
    row.d_A = 1.0 / g
    out = []
    for ph in phases:
        for ln in ring_lines(ph, g_max=g * 1.1):
            err = (row.d_A / ln.d - 1) * 100
            if abs(err) <= tol * 100:
                out.append({"phase": ph.name, "hkl": ln.label, "d_ref_A": round(ln.d, 4), "error_pct": round(err, 2),
                            "rel_intensity": round(ln.intensity, 1)})
    out.sort(key=lambda m: (abs(m["error_pct"]), -m["rel_intensity"]))
    row.matches = out


def rings_from_spots(spots: Spots, centre, r_min: float, r_max: float, min_count: int = 2, rel_tol: float = 0.02,
                     min_snr: float = 6.0) -> list[Ring]:
    """Cluster spot radii into rings (nanocrystalline / very spotty patterns)."""
    if spots is None or len(spots.yx) < 6:
        return []
    v = spots.yx - np.asarray(centre, float)
    r = np.hypot(v[:, 0], v[:, 1])
    ok = (r > max(1.2 * r_min, 3.0)) & (r < 0.97 * r_max) & (spots.snr >= min_snr)
    r, I = r[ok], spots.intensity[ok]
    if len(r) < 4:
        return []
    o = np.argsort(r)
    r, I = r[o], I[o]
    groups, cur = [], [0]
    for k in range(1, len(r)):
        if r[k] - r[cur[-1]] <= max(1.2, rel_tol * r[k]):
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    rings = []
    noise = float(np.median(I)) / 10 + 1e-9
    for g in groups:
        if len(g) < min_count:
            continue
        w = I[g]
        rad = float(np.sum(r[g] * w) / np.sum(w))
        spread = float(np.sqrt(np.sum(w * (r[g] - rad) ** 2) / np.sum(w)))
        fwhm = max(1.0, 2.355 * spread)
        rings.append(Ring(radius=rad, height=float(w.sum()), fwhm=fwhm, snr=float(w.sum() / noise),
                          rel_width=fwhm / max(rad, 1e-9), kind="sharp"))
    return rings
