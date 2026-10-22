""""What should I do next?" — experiment advice that separates competing answers.

Spot patterns: every competing (phase, zone) hypothesis implies a crystal orientation. For a physical
specimen tilt, each hypothesis predicts a different new pattern. We search low-index target zones within
the holder's tilt range and rank tilts by how well they separate the hypotheses.

Ring patterns: list the reflections that one candidate has and the other lacks (or shifts), so the user
knows which ring to look for (or which zone to record from a single grain).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .crystal import Phase, canonical_uvw, fmt_index, reflections, ring_lines


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation U (det +1) minimising Σ w |U p_i − q_i|²."""
    if w is None:
        w = np.ones(len(P))
    H = (P * w[:, None]).T @ Q
    Uu, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T))
    D = np.diag([1, 1, d])
    return Vt.T @ D @ Uu.T


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _angle(u, v) -> float:
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def low_index_directions(max_sum: int = 4, max_comp: int = 3):
    out = []
    for uvw in itertools.product(range(-max_comp, max_comp + 1), repeat=3):
        if uvw == (0, 0, 0) or sum(abs(x) for x in uvw) > max_sum:
            continue
        if np.gcd.reduce(np.abs(uvw)) != 1:
            continue
        out.append(uvw)
    return out


def zone_geometry(phase: Phase, uvw, g_max: float = 1.0) -> dict | None:
    """Two shortest independent allowed reflections of a zone: d1, d2, angle; and number of spots."""
    ref = reflections(phase, g_max)
    m = ((ref.hkl @ np.array(uvw)) == 0) & ref.allowed
    if m.sum() < 4:
        return None
    hkl, gv, g, F2 = ref.hkl[m], ref.gvec[m], ref.g[m], ref.F2[m]
    strong = F2 >= 0.02 * F2.max()
    hkl, gv, g = hkl[strong], gv[strong], g[strong]
    o = np.argsort(g)
    h1, v1, g1 = hkl[o[0]], gv[o[0]], g[o[0]]
    for k in o[1:]:
        if np.linalg.norm(np.cross(v1, gv[k])) > 0.2 * g1 * g[k]:
            ang = _angle(v1, gv[k])
            if ang > 90:
                ang = 180 - ang
            return {"d1": 1 / g1, "d2": 1 / g[k], "angle": ang, "h1": fmt_index(h1), "h2": fmt_index(hkl[k]), "n_spots": int(len(g))}
    return None


# ----------------------------------------------------------------------------
# Orientation of a spot-pattern hypothesis
# ----------------------------------------------------------------------------

def orientation_from_solution(sol, calib: float | None) -> tuple[np.ndarray, float]:
    """Rotation U mapping crystal Cartesian (reciprocal/direct) vectors to the lab frame
    (x right, y down in the image, z along the beam). Returns (U, rms angular misfit in degrees)."""
    phase = sol.phase
    B = phase.structure.lattice.reciprocal_lattice_crystallographic.matrix
    A = phase.structure.lattice.matrix
    c = calib or sol.calib
    P, Q, W = [], [], []
    for s in sol.indexed:
        if s["hkl_raw"] is None or not s.get("allowed", True):
            continue
        gc = np.asarray(s["hkl_raw"]) @ B
        gl = np.array([s["dx"], s["dy"], 0.0]) * c
        n = np.linalg.norm(gc)
        if n == 0:
            continue
        P.append(gc / n)
        Q.append(gl / max(np.linalg.norm(gl), 1e-12))
        W.append(1.0)
    if len(P) < 2:
        raise ValueError("not enough indexed spots")
    zc = np.asarray(sol.uvw_family) @ A
    zc = zc / np.linalg.norm(zc)
    best = None
    for zsign in (1, -1):
        Pz = np.vstack(P + [zc])
        Qz = np.vstack(Q + [np.array([0, 0, zsign])])
        Wz = np.array(W + [len(P)])
        U = _kabsch(Pz, Qz, Wz)
        mis = np.degrees(np.arccos(np.clip(np.sum((Pz @ U.T) * Qz, axis=1), -1, 1)))
        rms = float(np.sqrt(np.mean(mis[:-1] ** 2)))
        if best is None or rms < best[1]:
            best = (U, rms)
    return best


@dataclass
class TiltAdvice:
    target_phase: str
    target_zone: str
    tilt_deg: float
    axis_img: tuple[float, float]  # unit tilt-axis direction in the image (x right, y down)
    R: np.ndarray  # specimen rotation (lab frame)
    predictions: list[dict] = field(default_factory=list)  # one per hypothesis
    separates: int = 0
    of: int = 0
    n_on_zone: int = 0
    text: str = ""
    axis_note: str = ""

    def to_dict(self):
        return {"target": f"{self.target_phase} {self.target_zone}", "tilt_deg": round(self.tilt_deg, 1),
                "axis_image_xy": [round(self.axis_img[0], 4), round(self.axis_img[1], 4)], "axis_note": self.axis_note,
                "separates": f"{self.separates}/{self.of}", "predictions": self.predictions, "text": self.text,
                "rotation": np.round(self.R, 6).tolist()}


def _describe(pred: dict) -> str:
    if pred.get("zone") is None:
        return f"{pred['phase']}: no low-index zone at that tilt (off-axis, few spots)"
    g = pred.get("geom")
    if not g:
        return f"{pred['phase']}: high-index zone {pred['zone']} (few spots)"
    return f"{pred['phase']}: {pred['zone']}, spots {g['h1']} {g['d1']:.2f} Å and {g['h2']} {g['d2']:.2f} Å at {g['angle']:.0f}°"


def _different(a: dict, b: dict, tol_d=0.03, tol_a=3.0) -> bool:
    if (a.get("zone") is None) != (b.get("zone") is None):
        return True
    if a.get("zone") is None:
        return False
    ga, gb = a.get("geom"), b.get("geom")
    if not ga or not gb:
        return bool(ga) != bool(gb)
    return (abs(ga["d1"] - gb["d1"]) / ga["d1"] > tol_d or abs(ga["d2"] - gb["d2"]) / ga["d2"] > tol_d
            or abs(ga["angle"] - gb["angle"]) > tol_a)


def _axis_note(axis_xy: np.ndarray, sol) -> str:
    """Describe the tilt axis relative to the indexed spots of the best solution (what the user sees on screen)."""
    spots = [s for s in sol.indexed if s["hkl"] and s.get("allowed")]
    if not spots:
        return ""
    ax_ang = np.degrees(np.arctan2(axis_xy[1], axis_xy[0]))

    def rel(s):
        a = np.degrees(np.arctan2(s["dy"], s["dx"]))
        d = (ax_ang - a + 90) % 180 - 90  # signed, in (-90, 90]
        return d

    g1 = next((s for s in spots if s["hkl"] == fmt_index(sol.h1)), None)
    near = min(spots, key=lambda s: (abs(rel(s)), s["dx"] ** 2 + s["dy"] ** 2))
    if abs(rel(near)) <= 4:
        return f"along the {near['hkl']} spot row; keep that row excited while tilting"
    if g1 is not None:
        d = rel(g1)
        return f"{abs(d):.0f}° {'clockwise' if d > 0 else 'anticlockwise'} from g1 {g1['hkl']}"
    return ""


def advise_tilts(solutions, calib: float | None, max_tilt: float = 35.0, score_margin: float = 0.15, max_hyp: int = 4,
                 n_out: int = 5) -> dict:
    """Rank specimen tilts that best separate the competing spot-pattern hypotheses."""
    if not solutions:
        return {"hypotheses": [], "advice": []}
    best = solutions[0].score
    hyps = []
    seen = set()
    for s in solutions:
        if s.score < best - score_margin:
            break
        key = (s.phase.name, s.uvw_family)
        if key in seen:
            continue
        seen.add(key)
        try:
            U, rms = orientation_from_solution(s, calib)
        except ValueError:
            continue
        hyps.append({"sol": s, "U": U, "rms": rms})
        if len(hyps) >= max_hyp:
            break
    if not hyps:
        return {"hypotheses": [], "advice": []}
    z = np.array([0.0, 0.0, 1.0])
    dirs = low_index_directions()
    cand_dirs = low_index_directions(max_sum=6, max_comp=4)
    advice: list[TiltAdvice] = []
    for hi, H in enumerate(hyps):
        ph = H["sol"].phase
        A = ph.structure.lattice.matrix
        cur = np.asarray(H["sol"].uvw_family) @ A
        for uvw in dirs:
            v_c = np.asarray(uvw) @ A
            if _angle(v_c, cur) < 2 or _angle(v_c, -cur) < 2:
                continue
            v = H["U"] @ (v_c / np.linalg.norm(v_c))
            if v[2] < 0:
                v = -v
            theta = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
            if theta > max_tilt or theta < 3:
                continue
            geom_i = zone_geometry(ph, uvw)
            if geom_i is None or geom_i["n_spots"] < 6:
                continue
            axis = np.cross(v, z)
            axis = axis / np.linalg.norm(axis)
            R = _rot(axis, np.radians(theta))
            preds = []
            for hj, Hj in enumerate(hyps):
                phj = Hj["sol"].phase
                Aj = phj.structure.lattice.matrix
                if hj == hi:
                    preds.append({"phase": phj.name, "zone": fmt_index(canonical_uvw(phj, np.array(uvw)), "[]"), "uvw": list(uvw),
                                  "geom": geom_i, "hyp": hj})
                    continue
                beam_c = Hj["U"].T @ R.T @ z  # beam direction in hypothesis j's crystal frame
                bestz = None
                for u2 in cand_dirs:
                    a = _angle(np.asarray(u2) @ Aj, beam_c)
                    a = min(a, 180 - a)
                    if bestz is None or a < bestz[0]:
                        bestz = (a, u2)
                if bestz[0] <= 1.5:
                    g = zone_geometry(phj, bestz[1])
                    preds.append({"phase": phj.name, "zone": fmt_index(canonical_uvw(phj, np.array(bestz[1])), "[]"), "uvw": list(bestz[1]),
                                  "geom": g, "hyp": hj})
                else:
                    preds.append({"phase": phj.name, "zone": None, "uvw": None, "geom": None, "hyp": hj, "off_by_deg": round(bestz[0], 1)})
            others = [p for p in preds if p["hyp"] != hi]
            sep = sum(1 for p in others if _different(preds[hi], p))
            n_on = sum(1 for p in preds if p.get("geom"))
            adv = TiltAdvice(target_phase=ph.name, target_zone=fmt_index(canonical_uvw(ph, np.array(uvw)), "[]"), tilt_deg=float(theta),
                             axis_img=(float(axis[0]), float(axis[1])), R=R,
                             predictions=[{k: v for k, v in p.items() if k != "hyp"} for p in preds],
                             separates=sep, of=len(others), n_on_zone=n_on)
            advice.append(adv)
    if not advice:
        return {"hypotheses": [_hyp_row(H) for H in hyps], "advice": []}
    multi = len(hyps) > 1

    def rank(a: TiltAdvice):
        frac = a.separates / a.of if a.of else 0
        idx = sum(abs(int(ch)) for ch in a.target_zone if ch.isdigit())
        if multi:  # separate everything first, then prefer tilts where several candidates give a recognisable zone
            return (-frac, -min(a.n_on_zone, 3), round(a.tilt_deg / 10), idx, a.tilt_deg)
        return (round(a.tilt_deg / 5), idx, a.tilt_deg)

    advice.sort(key=rank)
    top_sol = hyps[0]["sol"]
    out, seen_t = [], set()
    for a in advice:
        k = round(a.tilt_deg / 3)  # symmetry-equivalent tilts of a symmetric pattern look alike: keep the options diverse
        if k in seen_t:
            continue
        seen_t.add(k)
        a.axis_note = _axis_note(np.array(a.axis_img), top_sol)
        lines = [f"Tilt {a.tilt_deg:.0f}° about the dashed axis" + (f", {a.axis_note}." if a.axis_note else ".")]
        for p in a.predictions:
            lines.append("If " + _describe(p) + ".")
        if multi:
            lines.append("This separates all candidates." if a.separates == a.of else
                         f"This separates {a.separates} of {a.of} alternatives." if a.separates else "This does not separate the candidates.")
        a.text = " ".join(lines)
        out.append(a)
        if len(out) >= n_out:
            break
    return {"hypotheses": [_hyp_row(H) for H in hyps], "advice": [a.to_dict() for a in out]}


def _hyp_row(H) -> dict:
    s = H["sol"]
    return {"phase": s.phase.name, "zone": fmt_index(s.uvw_family, "[]"), "score": round(s.score, 3),
            "orientation_fit_deg": round(H["rms"], 2), "orientation": np.round(H["U"], 6).tolist()}


# ----------------------------------------------------------------------------
# Ring patterns
# ----------------------------------------------------------------------------

def discriminating_rings(phases: list[Phase], g_lo: float, g_hi: float, tol: float = 0.015, min_I: float = 2.0, n: int = 4) -> list[dict]:
    """Rings that one candidate shows and the others don't (within the visible range)."""
    out = []
    lines = {p.name: [l for l in ring_lines(p, g_max=g_hi * 1.02) if g_lo <= l.g <= g_hi] for p in phases}
    for p in phases:
        others = [q for q in phases if q.name != p.name]
        for l in lines[p.name]:
            if l.intensity < min_I:
                continue
            clash = [q.name for q in others if any(abs(m.g - l.g) / l.g < tol and m.intensity >= 1 for m in lines[q.name])]
            if len(clash) == 0:
                out.append({"phase": p.name, "hkl": l.label, "d_A": round(l.d, 4), "rel_intensity": round(l.intensity, 1),
                            "absent_in": [q.name for q in others]})
    out.sort(key=lambda r: -r["rel_intensity"])
    by_phase = {}
    for r in out:
        by_phase.setdefault(r["phase"], [])
        if len(by_phase[r["phase"]]) < n:
            by_phase[r["phase"]].append(r)
    return [r for v in by_phase.values() for r in v]


def advise(res) -> dict:
    """Top-level advice for an AnalysisResult."""
    out = {"kind": None, "summary": "", "items": []}
    kind = res.classification.kind
    row = getattr(res, "row", None)
    if row is not None:
        out["kind"] = "row"
        out["summary"] = (f"Only one systematic row is visible (along {row.direction_deg:.0f}° in the image). Tilt the specimen about this row, "
                          "keeping its spots bright (follow the matching Kikuchi band), until a second row of spots appears: that is a "
                          "zone axis, and SAEDscope can then index it fully.")
        return out
    conf = getattr(res, "calibration_conflict", None)
    if conf:
        alts = ", ".join(f"{a['phase']} {a['zone_axis']}" for a in conf.get("fits_at_given_calibration", [])[:3])
        out["kind"] = "calibration"
        out["summary"] = (f"Resolve the calibration first. {conf['phase']} {conf['zone_axis']} needs the calibration to be "
                          f"{conf['calibration_off_by_pct']:+.1f}% off" + (f"; at the given calibration {alts} fit instead" if alts else "")
                          + ". Record a gold (or aluminium) ring pattern at the same camera length and use it under "
                          "Calibration → From a standard pattern; if your phase is right, this image itself gives the corrected "
                          "calibration.")
        return out
    if res.spot_result and res.spot_result.solutions and kind in ("single crystal", "mixed"):
        adv = advise_tilts(res.spot_result.solutions, res.calibration.inv_A_per_px)
        out["kind"] = "tilt"
        out["hypotheses"] = adv["hypotheses"]
        out["items"] = adv["advice"]
        if len(adv["hypotheses"]) > 1:
            names = ", ".join(f"{h['phase']} {h['zone']}" for h in adv["hypotheses"])
            first = adv["advice"][0] if adv["advice"] else None
            out["summary"] = (f"{len(adv['hypotheses'])} answers fit this pattern ({names}). "
                              + (f"Best next step: {first['text']}" if first else "No tilt within ±35° separates them; record a pattern at another region or add elements."))
        elif adv["advice"]:
            out["summary"] = "The answer is unambiguous. To confirm it, record a second zone axis: " + adv["advice"][0]["text"]
        return out
    if res.ring_result and res.ring_result.primary:
        rr = res.ring_result
        top = [rr.primary.phase] + [s.phase for s in rr.ranking[1:4] if s.score >= rr.primary.score - 0.12 and s.phase.formula != rr.primary.phase.formula]
        c = rr.primary.calib
        radii = [r.radius for r in res.rings if r.kind == "sharp"]
        if radii and c:
            g_lo, g_hi = min(radii) * c * 0.8, max(radii) * c * 1.05
        else:
            g_lo, g_hi = 0.2, 1.2
        out["kind"] = "rings"
        if len(top) > 1:
            items = discriminating_rings(top, g_lo, g_hi)
            out["items"] = items
            names = " vs ".join(p.name for p in top)
            if items:
                tips = []
                for p in top:
                    mine = [i for i in items if i["phase"] == p.name][:2]
                    if mine:
                        tips.append(f"{p.name} would also show " + " and ".join(f"a ring at {i['d_A']:.3f} Å ({i['hkl']}, relative intensity {i['rel_intensity']:.0f})" for i in mine))
                out["summary"] = (f"{names} fit almost equally well. " + "; ".join(tips)
                                  + ". Check for those rings (weak rings may need a longer exposure), add elements from EDS, or record a spot pattern from a single grain.")
            else:
                out["summary"] = (f"{names} have rings at the same positions within the measurement precision, so a ring pattern cannot separate them. "
                                  "Add elements from EDS or record a spot pattern from a single grain.")
        else:
            out["summary"] = "The ring match is unambiguous among the candidates searched."
        return out
    out["summary"] = "No crystalline phase to refine." if kind == "amorphous" else "Add elements, a CIF or a calibration and analyse again."
    return out
