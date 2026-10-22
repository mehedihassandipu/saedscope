"""Single-crystal spot patterns: 2D lattice fit, zone-axis search and indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np

from .crystal import Phase, canonical_hkl, canonical_uvw, canonical_uvw_op, fmt_index, reflections
from .imageproc import Spots


@dataclass
class LatticeFit:
    a: np.ndarray  # 2D basis vector (dy, dx) in px
    b: np.ndarray
    inliers: np.ndarray  # bool per spot
    nm: np.ndarray  # (N, 2) integer indices per spot (nan if outlier)
    rms_px: float
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # centre correction (dy, dx) px

    @property
    def len_a(self):
        return float(np.linalg.norm(self.a))

    @property
    def len_b(self):
        return float(np.linalg.norm(self.b))

    @property
    def angle_deg(self):
        cosv = np.dot(self.a, self.b) / (self.len_a * self.len_b)
        return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))


def _reduce(a, b):
    """Lagrange–Gauss reduction of a 2D basis."""
    a, b = np.array(a, float), np.array(b, float)
    for _ in range(50):
        if np.dot(b, b) < np.dot(a, a):
            a, b = b, a
        mu = round(np.dot(a, b) / np.dot(a, a))
        if mu == 0:
            break
        b = b - mu * a
    if np.dot(b, b) < np.dot(a, a):
        a, b = b, a
    return a, b


def fit_lattice(vecs: np.ndarray, weights: np.ndarray | None = None, tol_frac: float = 0.08) -> LatticeFit | None:
    """Fit a 2D reciprocal lattice to spot vectors (relative to the centre)."""
    n = len(vecs)
    if n < 4:
        return None
    if weights is None:
        weights = np.ones(n)
    w = weights / weights.max()
    lens = np.linalg.norm(vecs, axis=1)
    cand_idx = np.argsort(lens)[: min(14, n)]
    cands = vecs[cand_idx]
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            cross = a[0] * b[1] - a[1] * b[0]
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if abs(cross) < 0.25 * la * lb:  # nearly collinear
                continue
            a, b = _reduce(a, b)
            M = np.stack([a, b], axis=1)
            try:
                coef = np.linalg.solve(M, vecs.T).T
            except np.linalg.LinAlgError:
                continue
            nm = np.round(coef)
            tol = max(2.0, tol_frac * min(np.linalg.norm(a), np.linalg.norm(b)))
            # one least-squares step including a centre offset (tolerates a few px of centre error)
            X = np.hstack([nm, np.ones((n, 1))])
            near = np.linalg.norm(vecs - nm @ np.stack([a, b]), axis=1) < 2.5 * tol
            if near.sum() >= 4:
                sol, *_ = np.linalg.lstsq(X[near], vecs[near], rcond=None)
                a2, b2, off = sol[0], sol[1], sol[2]
                ok_basis = abs(a2[0] * b2[1] - a2[1] * b2[0]) > 0.25 * np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-9
                if ok_basis and np.linalg.norm(off) < 0.5 * min(np.linalg.norm(a), np.linalg.norm(b)):
                    a, b = a2, b2
                    shifted = vecs - off
                    nm = np.round(np.linalg.solve(np.stack([a, b], axis=1), shifted.T).T)
                    err = np.linalg.norm(shifted - nm @ np.stack([a, b]), axis=1)
                else:
                    err = np.linalg.norm(vecs - nm @ np.stack([a, b]), axis=1)
            else:
                err = np.linalg.norm(vecs - nm @ np.stack([a, b]), axis=1)
            inl = err < tol
            # small indices preferred: penalise bases that are too fine
            score = float(np.sum(w[inl])) - 0.002 * float(np.sum(np.abs(nm[inl])))
            if best is None or score > best[0] + 1e-9:
                best = (score, a, b, inl, nm)
    if best is None:
        return None
    _, a, b, inl, nm = best
    # least-squares refinement of basis AND centre offset using inliers
    off = np.zeros(2)
    err = np.linalg.norm(vecs - nm @ np.stack([a, b]), axis=1)
    for _ in range(4):
        A = nm[inl]
        if len(A) < 4:
            break
        X = np.hstack([A, np.ones((len(A), 1))])
        sol, *_ = np.linalg.lstsq(X, vecs[inl], rcond=None)
        if abs(sol[0][0] * sol[1][1] - sol[0][1] * sol[1][0]) < 0.25 * np.linalg.norm(sol[0]) * np.linalg.norm(sol[1]) + 1e-9:
            break  # refinement collapsed the basis (degenerate inliers): keep the previous one
        a, b, off_step = sol[0], sol[1], sol[2]
        off = off + off_step
        vecs = vecs - off_step
        coef = np.linalg.solve(np.stack([a, b], axis=1), vecs.T).T
        nm = np.round(coef)
        err = np.linalg.norm(vecs - nm @ np.stack([a, b]), axis=1)
        tol = max(2.0, tol_frac * min(np.linalg.norm(a), np.linalg.norm(b)))
        inl = err < tol
    a, b = _reduce(a, b)
    try:
        coef = np.linalg.solve(np.stack([a, b], axis=1), vecs.T).T
    except np.linalg.LinAlgError:
        return None
    nm = np.round(coef)
    rms = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan")
    nm_out = np.where(inl[:, None], nm, np.nan)
    return LatticeFit(a=np.asarray(a), b=np.asarray(b), inliers=inl, nm=nm_out, rms_px=rms, offset=off)


@dataclass
class ZoneSolution:
    phase: Phase
    uvw: tuple[int, int, int]  # as found
    uvw_family: tuple[int, int, int]  # symmetry-canonical
    h1: tuple[int, int, int]
    h2: tuple[int, int, int]
    d1_meas: float
    d2_meas: float
    d1_ref: float
    d2_ref: float
    angle_meas: float
    angle_ref: float
    calib: float
    scale_dev_pct: float | None
    score: float
    frac_explained: float
    frac_missing: float
    indexed: list[dict] = field(default_factory=list)
    predicted: list[dict] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "phase": self.phase.name,
            "formula": self.phase.formula,
            "space_group": self.phase.spacegroup,
            "zone_axis": fmt_index(self.uvw_family, "[]"),
            "score": round(self.score, 3),
            "spots_explained_pct": round(100 * self.frac_explained, 1),
            "g1": f"{fmt_index(self.h1)} {self.d1_meas:.3f} / {self.d1_ref:.3f} Å",
            "g2": f"{fmt_index(self.h2)} {self.d2_meas:.3f} / {self.d2_ref:.3f} Å",
            "angle": f"{self.angle_meas:.1f}° / {self.angle_ref:.1f}°",
            "scale_dev_pct": None if self.scale_dev_pct is None else round(self.scale_dev_pct, 2),
            "source": self.phase.source + (f" {self.phase.source_id}" if self.phase.source == "Materials Project" else ""),
        }


def _int_reduce(v) -> tuple[int, int, int]:
    v = [int(round(x)) for x in v]
    g = 0
    for x in v:
        g = gcd(g, abs(x))
    g = g or 1
    v = [x // g for x in v]
    # sign convention: first non-zero positive
    for x in v:
        if x != 0:
            if x < 0:
                v = [-y for y in v]
            break
    return tuple(v)


def _angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def index_zone(fit: LatticeFit, spots_vecs: np.ndarray, spots_int: np.ndarray, phases: list[Phase], calib: float | None,
               img_shape, centre, r_min: float, tol_d: float = 0.03, tol_ang: float = 2.5, scale_window: float = 0.035,
               max_per_phase: int = 3, max_index_sum: int = 9, visible: np.ndarray | None = None) -> list[ZoneSolution]:
    la, lb, phi = fit.len_a, fit.len_b, fit.angle_deg
    ratio = lb / la
    h, w = img_shape
    r_vis = min(centre[0], centre[1], h - 1 - centre[0], w - 1 - centre[1])
    # an off-centre pattern (e.g. a beam stop near the edge, a cropped figure) has spots well beyond the nearest edge:
    # predict out to the farthest observed spot, and count predicted spots as missing only if they fall in the image
    if len(spots_vecs):
        r_far = min(float(np.max(np.linalg.norm(spots_vecs, axis=1))) * 1.03,
                    float(max(np.hypot(cy, cx) for cy in (centre[0], h - 1 - centre[0]) for cx in (centre[1], w - 1 - centre[1]))))
        r_vis = max(r_vis, r_far)
    def in_image(pos):
        yy, xx = centre[0] + pos[:, 0], centre[1] + pos[:, 1]
        ok = (yy > 2) & (yy < h - 3) & (xx > 2) & (xx < w - 3)
        if visible is not None and visible.shape == (h, w):
            iy, ix = np.clip(np.rint(yy).astype(int), 0, h - 1), np.clip(np.rint(xx).astype(int), 0, w - 1)
            ok &= visible[iy, ix]
        return ok
    solutions: list[ZoneSolution] = []
    for phase in phases:
        try:
            # uncalibrated: reach the outermost spots even if the basis turns out to be a fairly long g (up to ~0.6 Å⁻¹)
            g_cap = (r_vis * calib * 1.05) if calib else float(np.clip(0.6 * r_vis / max(la, 1e-9), 1.3, 2.5))
            ref = reflections(phase, max(g_cap, (lb * calib * 1.2) if calib else 1.3, 0.6))
        except Exception:
            continue
        # basis vectors are lattice vectors: they may be kinematically forbidden (e.g. (002) in anatase)
        hkl_a, gv_a, g_a = ref.hkl, ref.gvec, ref.g
        if len(g_a) == 0:
            continue
        cands = []
        if calib:
            for scale in (1.0,):
                ga_m, gb_m = la * calib, lb * calib
                A = np.where(np.abs(g_a - ga_m) / ga_m < tol_d + scale_window)[0]
                B = np.where(np.abs(g_a - gb_m) / gb_m < tol_d + scale_window)[0]
                for i in A:
                    for j in B:
                        cands.append((i, j))
        else:
            order = np.argsort(g_a)
            short = order[: min(60, len(order))]
            for i in short:
                target = g_a[i] * ratio
                B = np.where(np.abs(g_a - target) / target < tol_d)[0]
                for j in B:
                    cands.append((i, j))
        prelim = []
        for i, j in cands:
            if i == j:
                continue
            ang = _angle(gv_a[i], gv_a[j])
            if abs(ang - phi) > tol_ang:
                continue
            c_fit = 0.5 * (g_a[i] / la + g_a[j] / lb)
            if calib and abs(c_fit / calib - 1) > scale_window + tol_d:
                continue
            ratio_err = abs((g_a[j] / g_a[i]) - ratio) / ratio
            prelim.append((ratio_err / tol_d + abs(ang - phi) / tol_ang, i, j, ang, c_fit, ratio_err))
        prelim.sort(key=lambda t: t[0])
        best_by_fam: dict = {}
        for _, i, j, ang, c_fit, ratio_err in prelim[:400]:
            h1, h2 = hkl_a[i], hkl_a[j]
            uvw = _int_reduce(np.cross(h1, h2))
            if uvw == (0, 0, 0) or sum(abs(x) for x in uvw) > max_index_sum:
                continue
            fam = canonical_uvw(phase, np.array(uvw))
            if fam in best_by_fam and best_by_fam[fam].score > 0.97:
                continue
            sol = _verify(phase, ref, h1, h2, uvw, fam, fit, spots_vecs, spots_int, c_fit, calib, la, lb, phi,
                          g_a[i], g_a[j], ang, r_vis, r_min, ratio_err, tol_d, scale_window, in_image=in_image)
            if sol and (fam not in best_by_fam or sol.score > best_by_fam[fam].score):
                best_by_fam[fam] = sol
        local: list[ZoneSolution] = list(best_by_fam.values())
        local.sort(key=lambda s: -s.score)
        solutions.extend(local[:max_per_phase])
    solutions.sort(key=lambda s: -s.score)
    return solutions


def _verify(phase, ref, h1, h2, uvw, fam, fit, vecs, inten, c_fit, calib, la, lb, phi, g1, g2, ang, r_vis, r_min, ratio_err, tol_d,
            scale_window: float = 0.035, in_image=None):
    """Simulate the zero-order Laue zone for [uvw] and compare with the observed spots."""
    inz = (ref.hkl @ np.array(uvw)) == 0
    hz, F2z, allowed_z = ref.hkl[inz], ref.F2[inz], ref.allowed[inz]
    M = np.stack([h1, h2], axis=1).astype(float)  # 3x2
    coef, *_ = np.linalg.lstsq(M, hz.T.astype(float), rcond=None)
    coef = coef.T  # (n, 2) fractional n, m
    pos = coef @ np.stack([fit.a, fit.b])  # predicted (dy, dx) px
    rpos = np.linalg.norm(pos, axis=1)
    vis = (rpos < r_vis * 0.98) & (rpos > r_min)
    pos, hz, F2z, allowed_z, coef = pos[vis], hz[vis], F2z[vis], allowed_z[vis], coef[vis]
    if len(pos) == 0:
        return None
    tol_px = max(2.5, 0.08 * min(la, lb))
    # observed → predicted
    expl_w, tot_w = 0.0, float(np.sum(inten))
    indexed = []
    matched_pred = set()
    for k, v in enumerate(vecs):
        dist = np.linalg.norm(pos - v, axis=1)
        p = int(np.argmin(dist))
        if dist[p] < tol_px:
            expl_w += inten[k]
            matched_pred.add(p)
            gm = np.linalg.norm(v) * c_fit
            hk = tuple(int(x) for x in hz[p])
            indexed.append({"spot": k, "hkl": fmt_index(hk), "hkl_raw": hk, "d_meas": 1 / gm if gm > 0 else None,
                            "d_ref": 1 / np.linalg.norm(hz[p] @ phase.structure.lattice.reciprocal_lattice_crystallographic.matrix),
                            "allowed": bool(allowed_z[p]), "dy": float(v[0]), "dx": float(v[1])})
        else:
            indexed.append({"spot": k, "hkl": None, "hkl_raw": None, "d_meas": (1 / (np.linalg.norm(v) * c_fit)), "d_ref": None, "allowed": None, "dy": float(v[0]), "dx": float(v[1])})
    frac_expl = expl_w / max(tot_w, 1e-12)
    # intensity on kinematically forbidden positions: a little is double diffraction, a lot means a wrong (doubled) cell
    forb_w = sum(inten[s["spot"]] for s in indexed if s["allowed"] is False)
    frac_forb = forb_w / max(tot_w, 1e-12)
    # predicted strong allowed spots that are not observed
    strong = allowed_z & (F2z >= 0.15 * F2z[allowed_z].max()) if allowed_z.any() else allowed_z
    near = np.linalg.norm(pos, axis=1) < max(np.linalg.norm(vecs, axis=1).max(), 1) * 1.02
    strong = strong & near
    if in_image is not None:
        strong = strong & in_image(pos)
    missing = [p for p in np.where(strong)[0] if p not in matched_pred]
    frac_miss = len(missing) / max(strong.sum(), 1)
    ang_err = abs(ang - phi)
    score = frac_expl * (1 - 0.5 * frac_miss) * (1 - 0.3 * min(ratio_err / tol_d, 1)) * (1 - 0.2 * min(ang_err / 2.5, 1))
    score *= 1 - 0.7 * max(0.0, frac_forb - 0.12)
    if calib:
        dev = abs(c_fit / calib - 1)
        score *= 1 - 0.15 * (dev / max(scale_window, 1e-6)) ** 2
    score *= 1 - 0.012 * (sum(abs(x) for x in uvw) - 1)  # prefer low-index zones when patterns are equivalent
    predicted = [{"hkl": fmt_index(tuple(int(x) for x in hz[p])), "hkl_raw": tuple(int(x) for x in hz[p]), "dy": float(pos[p][0]), "dx": float(pos[p][1]),
                  "allowed": bool(allowed_z[p]), "rel_I": float(F2z[p] / max(F2z.max(), 1e-12)), "observed": p in matched_pred}
                 for p in range(len(pos)) if allowed_z[p] or p in matched_pred]
    # For reporting, use the two shortest *observed, allowed* spots (the fitted basis may be an extinct lattice vector)
    h1r, h2r = tuple(int(x) for x in h1), tuple(int(x) for x in h2)
    d1m, d2m, d1r, d2r, angm, angr = 1 / (la * c_fit), 1 / (lb * c_fit), 1 / g1, 1 / g2, phi, ang
    obs = sorted([s for s in indexed if s["hkl"] and s["allowed"]], key=lambda s: s["dy"] ** 2 + s["dx"] ** 2)
    if obs and (not allowed_z[np.argmin(np.linalg.norm(pos - fit.a, axis=1))] or not allowed_z[np.argmin(np.linalg.norm(pos - fit.b, axis=1))]):
        s1 = obs[0]
        v1 = np.array([s1["dy"], s1["dx"]])
        s2 = next((s for s in obs[1:] if abs(v1[0] * s["dx"] - v1[1] * s["dy"]) > 0.3 * np.linalg.norm(v1) * np.hypot(s["dy"], s["dx"])), None)
        if s2 is not None:
            recip = phase.structure.lattice.reciprocal_lattice_crystallographic.matrix
            v2 = np.array([s2["dy"], s2["dx"]])
            h1r, h2r = s1["hkl_raw"], s2["hkl_raw"]
            d1m, d2m = s1["d_meas"], s2["d_meas"]
            d1r, d2r = s1["d_ref"], s2["d_ref"]
            angm = _angle(v1, v2)
            angr = _angle(np.array(h1r) @ recip, np.array(h2r) @ recip)
    if angm > 90.0 and h2r is not None:  # report the acute angle (use -g2)
        h2r = tuple(-x for x in h2r)
        angm, angr = 180.0 - angm, 180.0 - angr
    # express all indices in the setting of the canonical zone axis (so labels match the reported [uvw])
    _, W = canonical_uvw_op(phase, np.array(uvw))
    Winv = np.rint(np.linalg.inv(W)).astype(int)
    tr = lambda h: tuple(int(x) for x in (np.asarray(h) @ Winv))
    for srec in indexed:
        if srec["hkl_raw"] is not None:
            srec["hkl_raw"] = tr(srec["hkl_raw"])
            srec["hkl"] = fmt_index(srec["hkl_raw"])
    for prec in predicted:
        prec["hkl"] = fmt_index(tr(prec["hkl_raw"]))
    h1r, h2r = tr(h1r), tr(h2r)
    return ZoneSolution(phase=phase, uvw=uvw, uvw_family=fam, h1=h1r, h2=h2r,
                        d1_meas=d1m, d2_meas=d2m, d1_ref=d1r, d2_ref=d2r,
                        angle_meas=angm, angle_ref=angr, calib=c_fit,
                        scale_dev_pct=(c_fit / calib - 1) * 100 if calib else None,
                        score=float(score), frac_explained=float(frac_expl), frac_missing=float(frac_miss),
                        indexed=indexed, predicted=predicted)


@dataclass
class SpotResult:
    fit: LatticeFit | None
    solutions: list[ZoneSolution]
    spots: Spots
    centre: tuple[float, float]
    n_spots_used: int
    notes: list[str]


def analyse_spots(spots: Spots, centre, phases: list[Phase], calib: float | None, img_shape, r_min: float,
                  tol_d: float = 0.03, tol_ang: float = 2.5, scale_window: float = 0.035, n_use: int = 80,
                  visible: np.ndarray | None = None) -> SpotResult:
    """``visible``: pixels where a spot could be seen (False under a beam stop or in a saturated glow); predicted
    spots elsewhere are not counted as missing."""
    notes: list[str] = []
    yx = spots.yx[:n_use]
    inten = spots.intensity[:n_use]
    snr = spots.snr[:n_use]
    vecs = yx - np.array(centre)
    rr = np.linalg.norm(vecs, axis=1)
    keep = (rr > 1.2 * r_min) & ~((rr < 2 * r_min) & (snr < 15))
    yx, inten, vecs = yx[keep], inten[keep], vecs[keep]
    fit = fit_lattice(vecs, inten)
    if fit is None:
        return SpotResult(None, [], spots, centre, 0, ["Not enough spots for a lattice fit."])
    centre = (float(centre[0] + fit.offset[0]), float(centre[1] + fit.offset[1]))
    vecs = yx - np.array(centre)
    if fit.inliers.mean() < 0.5:
        notes.append(f"Only {100 * fit.inliers.mean():.0f}% of spots lie on one 2D lattice: possibly several crystals, twins or a textured polycrystal.")
    sols = index_zone(fit, vecs[fit.inliers], inten[fit.inliers], phases, calib, img_shape, centre, r_min,
                      tol_d=tol_d, tol_ang=tol_ang, scale_window=scale_window, visible=visible)
    if calib is None:
        notes.append("Uncalibrated: the zone axis was found from the spot-spacing ratio and angle only.")
    return SpotResult(fit, sols, spots, centre, int(keep.sum()), notes)
