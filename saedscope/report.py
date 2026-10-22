"""Caption, methods text, tables and exports."""

from __future__ import annotations

import io
import json

import numpy as np

from .crystal import fmt_index
from .physics import electron_wavelength_A


def _fmt(x, nd=3):
    return "?" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def caption(res) -> str:
    s = res.summary()
    k = s["pattern_type"]
    cal = "calibrated" if s["calibrated"] else "uncalibrated (d-spacing ratios only)"
    if k == "single crystal" and s.get("spots", {}).get("best"):
        b = s["spots"]["best"]
        return (f"Selected-area electron diffraction pattern indexed as {b['phase']['name']} ({b['phase']['space_group']}) "
                f"viewed along the {b['zone_axis']} zone axis. Spots {b['g1']['hkl']} (d = {b['g1']['d_meas_A']:.3f} Å) and "
                f"{b['g2']['hkl']} (d = {b['g2']['d_meas_A']:.3f} Å) are marked; {b['spots_explained_pct']:.0f}% of detected spots are indexed. Pattern {cal}.")
    if s.get("rings", {}).get("primary_phase"):
        r = s["rings"]
        p = r["primary_phase"]
        idx = [a for a in r["assignments"] if a["phase"] == p["name"]]
        labels = ", ".join(f"({a['hkl']})" for a in idx[:6])
        sec = f" Additional ring(s) are consistent with {r['secondary_phase']['name']}." if r.get("secondary_phase") else ""
        return (f"Ring electron diffraction pattern of a polycrystalline sample indexed as {p['name']} ({p['space_group']}); "
                f"rings {labels} are labelled.{sec} Pattern {cal}.")
    if k == "amorphous":
        a = s.get("amorphous") or {}
        d = a.get("halo_d_A")
        return ("Electron diffraction pattern showing diffuse halo(s) characteristic of an amorphous structure"
                + (f"; the first halo corresponds to d ≈ {d:.2f} Å." if d else "."))
    return f"Electron diffraction pattern ({k})."


def methods(res, software_version: str = "SAEDscope 0.1") -> str:
    s = res.summary()
    cal = res.calibration
    parts = []
    volt = f" at {cal.voltage_kV:.0f} kV (λ = {electron_wavelength_A(cal.voltage_kV) * 100:.3f} pm)" if cal.voltage_kV else ""
    cl = f", camera length {cal.camera_length_mm:.0f} mm" if cal.camera_length_mm else ""
    parts.append(f"Selected-area electron diffraction patterns were recorded{volt}{cl}.")
    if cal.calibrated:
        src = {"file metadata": "the calibration stored in the image file", "user": "a user-supplied reciprocal-space calibration",
               "camera length": "the nominal camera length and detector pixel size", "standard": "a calibration standard"}.get(cal.source, cal.source)
        parts.append(f"Patterns were calibrated using {src} ({cal.inv_A_per_px:.5f} Å⁻¹ per pixel).")
    else:
        parts.append("No absolute calibration was available; phases were compared using ratios of d-spacings.")
    srcs = ", ".join(f"{k} ({v})" for k, v in s["candidate_sources"].items() if v)
    parts.append(f"Patterns were analysed with {software_version}: the pattern centre was located from Friedel symmetry, "
                 "the pattern type was classified from ring, spot and halo features, and candidate phases"
                 + (f" ({srcs})" if srcs else "")
                 + " were compared with measured d-spacings and interplanar angles using kinematically simulated reflections "
                 "(Doyle–Turner electron scattering factors).")
    if s.get("rings", {}).get("primary_phase"):
        dev = s["rings"].get("scale_deviation_pct")
        parts.append(f"Ring positions were matched within ±{100 * res.options.ring_tol:.1f}% allowing a uniform scale factor"
                     + (f" (fitted deviation {dev:+.2f}%)." if dev is not None else "."))
    if s.get("spots", {}).get("best"):
        parts.append(f"Zone axes were determined from the two shortest reciprocal-lattice vectors (d-spacing tolerance ±{100 * res.options.spot_tol_d:.0f}%, "
                     f"angle tolerance ±{res.options.spot_tol_angle:.1f}°) and verified against the simulated zero-order Laue zone.")
    parts.append("Diffraction intensities were treated as qualitative because of dynamical scattering.")
    return " ".join(parts)


def rings_table(res):
    import pandas as pd

    if not res.ring_result:
        return pd.DataFrame()
    rows = []
    for a in res.ring_result.assignments:
        r = res.rings[a.ring_index]
        rows.append({"ring": a.ring_index + 1, "radius (px)": round(a.radius_px, 2),
                     "d measured (Å)": None if not np.isfinite(a.d_meas) else round(a.d_meas, 4),
                     "phase": a.phase or "", "hkl": a.hkl or "", "d reference (Å)": None if a.d_ref is None else round(a.d_ref, 4),
                     "error (%)": None if a.error_pct is None else round(a.error_pct, 2),
                     "uniformity CV": None if not np.isfinite(r.uniformity_cv) else round(r.uniformity_cv, 2),
                     "S/N": round(r.snr, 1), "also fits": ", ".join(a.ambiguous_with)})
    return pd.DataFrame(rows)


def spots_table(res):
    """One row per detected spot. d is measured with the given calibration when there is one, so a calibration
    problem shows up as a uniform error instead of being hidden by the fitted scale."""
    import pandas as pd

    if not res.spot_result or not res.spot_result.solutions:
        return pd.DataFrame()
    sol = res.spot_result.solutions[0]
    cal = res.calibration.inv_A_per_px if res.calibration.calibrated else None
    c_use = cal or sol.calib
    zone = fmt_index(sol.uvw_family, "[]")
    rows = []
    for s in sol.indexed:
        r_px = float(np.hypot(s["dx"], s["dy"]))
        d_meas = 1 / (r_px * c_use) if r_px > 0 and c_use else None
        hk = s.get("hkl_raw")
        rows.append({"spot": s["spot"] + 1,
                     "h k l": " ".join(str(int(v)) for v in hk) if hk is not None else "",
                     "hkl": s["hkl"] or "",
                     "distance from centre (px)": round(r_px, 2),
                     "d measured (Å)": None if d_meas is None else round(d_meas, 4),
                     "d reference (Å)": None if s["d_ref"] is None else round(s["d_ref"], 4),
                     "error (%)": None if not s["d_ref"] or d_meas is None else round((d_meas / s["d_ref"] - 1) * 100, 2),
                     "allowed": {True: "yes", False: "no (double diffraction?)", None: ""}[s["allowed"]],
                     "phase": sol.phase.name if s["hkl"] else "", "zone axis": zone if s["hkl"] else ""})
    df = pd.DataFrame(rows)
    df["_unindexed"] = df["h k l"] == ""
    df = df.sort_values(["_unindexed", "distance from centre (px)"]).drop(columns="_unindexed")
    return df


def csv_bytes(df) -> bytes:
    """CSV that opens cleanly in Excel (UTF-8 with BOM, so Å and overlined indices survive)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def ranking_table(res):
    import pandas as pd

    if res.spot_result and res.spot_result.solutions and res.classification.kind in ("single crystal", "mixed"):
        return pd.DataFrame([s.to_row() for s in res.spot_result.solutions[:10]])
    if res.ring_result:
        return pd.DataFrame([s.to_row() for s in res.ring_result.ranking[:10]])
    return pd.DataFrame()


def summary_json(res) -> str:
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(res.summary(), indent=2, default=default, ensure_ascii=False)


def best_phase_cif(res) -> tuple[str, str] | None:
    from pymatgen.io.cif import CifWriter

    phase = None
    if res.spot_result and res.spot_result.solutions and res.classification.kind in ("single crystal", "mixed"):
        phase = res.spot_result.solutions[0].phase
    elif res.ring_result and res.ring_result.primary:
        phase = res.ring_result.primary.phase
    if phase is None:
        return None
    text = str(CifWriter(phase.structure, symprec=0.01))
    safe = "".join(ch if ch.isalnum() else "_" for ch in phase.formula)
    return f"{safe}.cif", text
