"""Real-data helpers: calibration from a standard, ellipticity, scale bars."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import imageproc as ip
from .crystal import Phase, builtin_library
from .physics import electron_wavelength_A
from .rings import score_phase

STANDARDS = ["Au", "Al", "Ag", "Pt", "Si", "Ni", "Cu", "MgO", "TiN"]


@dataclass
class StandardCalibration:
    standard: str
    calib_inv_A_per_px: float
    camera_constant_A_px: float  # λL in Å·px  (d = λL / R)
    rings_matched: int
    rings_detected: int
    rms_error_pct: float
    centre: tuple[float, float]
    ellipticity_pct: float
    ellipticity_axis_deg: float
    camera_length_mm: float | None = None
    table: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = (f"{self.standard}: {self.rings_matched} of {self.rings_detected} rings matched, rms error {self.rms_error_pct:.2f}%. "
             f"Calibration {self.calib_inv_A_per_px:.6f} Å⁻¹/px ({10 * self.calib_inv_A_per_px:.5f} nm⁻¹/px), "
             f"camera constant λL = {self.camera_constant_A_px:.1f} Å·px.")
        if self.camera_length_mm:
            s += f" Effective camera length ≈ {self.camera_length_mm:.0f} mm."
        if abs(self.ellipticity_pct) >= 0.3:
            s += (f" Ring ellipticity {self.ellipticity_pct:.2f}% (long axis at {self.ellipticity_axis_deg:.0f}°); the correction can be "
                  "applied to patterns from the same session and camera length (Advanced).")
        else:
            s += " Rings are circular to within 0.3%."
        return s


def ring_ellipticity(img: np.ndarray, centre, radius: float, width: float, valid=None, n_sectors: int = 72) -> tuple[float, float]:
    """Fit r(θ) = R0 [1 + ε cos 2(θ − φ)] to one ring. Returns (ε in %, φ in degrees, image coordinates)."""
    r = ip.radius_map(img.shape, centre)
    th = ip.angle_map(img.shape, centre)
    band = np.abs(r - radius) <= max(3.0, 2.0 * width)
    if valid is not None:
        band &= valid
    thetas, radii = [], []
    edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = band & (th >= a) & (th < b)
        if m.sum() < 10:
            continue
        rr, vv = r[m], img[m]
        bins = np.round(rr * 2) / 2
        ub = np.unique(bins)
        prof = np.array([vv[bins == u].mean() for u in ub])
        if len(prof) < 5:
            continue
        prof = prof - np.percentile(prof, 20)
        prof = np.clip(prof, 0, None)
        if prof.sum() <= 0:
            continue
        top = prof >= 0.5 * prof.max()
        radii.append(float(np.sum(ub[top] * prof[top]) / np.sum(prof[top])))
        thetas.append(0.5 * (a + b))
    if len(radii) < n_sectors // 3:
        return 0.0, 0.0
    t, R = np.array(thetas), np.array(radii)
    X = np.stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t)], axis=1)
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    # robust second pass
    res = R - X @ coef
    keep = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    if keep.sum() >= 8:
        coef, *_ = np.linalg.lstsq(X[keep], R[keep], rcond=None)
    r0, c2, s2 = coef
    eps = float(np.hypot(c2, s2) / r0 * 100)
    phi = float(np.degrees(0.5 * np.arctan2(s2, c2)))
    return eps, phi


def correct_ellipticity(img: np.ndarray, centre, eps_pct: float, phi_deg: float) -> np.ndarray:
    """Resample so that rings become circular (inverse of r(θ) = R0 [1 + ε cos 2(θ − φ)])."""
    if abs(eps_pct) < 1e-3:
        return img
    e = eps_pct / 100.0
    phi = np.radians(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    Rm = np.array([[c, -s], [s, c]])  # (x, y) rotation
    S = np.diag([1 + e, 1 - e])  # stretch along the long axis in the distorted image
    M_xy = Rm @ S @ Rm.T  # maps corrected → distorted (x, y)
    # ndimage works in (row=y, col=x)
    P = np.array([[0, 1], [1, 0]])
    M = P @ M_xy @ P
    ctr = np.array(centre)
    offset = ctr - M @ ctr
    return ndimage.affine_transform(img, M, offset=offset, order=1, mode="nearest")


def calibrate_from_standard(image: np.ndarray, standard: str | Phase = "Au", voltage_kV: float | None = None,
                            pixel_size_um: float | None = None, correct: bool = True) -> StandardCalibration:
    phase = standard if isinstance(standard, Phase) else {p.name: p for p in builtin_library()}[standard]
    img = ip.normalise(image)
    centre = ip.find_centre(img, refine_rings=True)
    valid = ip.auto_mask(img, centre)
    prof = ip.build_profile(img, centre, valid)
    rings = [r for r in ip.detect_rings(prof) if r.kind == "sharp"]
    notes = []
    if len(rings) < 2:
        raise ValueError("Fewer than two sharp rings found: is this a polycrystalline standard (e.g. evaporated gold)?")
    strongest = max(rings, key=lambda r: r.height)
    eps, phi = ring_ellipticity(img, centre, strongest.radius, strongest.fwhm, valid)
    if correct and eps >= 0.3:
        img = correct_ellipticity(img, centre, eps, phi)
        prof = ip.build_profile(img, centre, valid)
        rings = [r for r in ip.detect_rings(prof) if r.kind == "sharp"]
        notes.append(f"Ellipticity {eps:.2f}% corrected before fitting the camera constant.")
    ps = score_phase(phase, rings, None, tol=0.015)
    if ps is None or len(ps.matched) < 2:
        raise ValueError(f"Could not match the rings to {phase.name}. Check the standard material.")
    c = ps.calib
    table, errs = [], []
    for i, j, e in ps.matched:
        ln = ps.lines[j]
        d_meas = 1 / (rings[i].radius * c)
        err = (d_meas / ln.d - 1) * 100
        errs.append(err)
        table.append({"ring": i + 1, "radius_px": round(rings[i].radius, 2), "hkl": ln.label, "d_ref_A": round(ln.d, 4),
                      "d_meas_A": round(d_meas, 4), "error_pct": round(err, 3)})
    L = None
    if voltage_kV and pixel_size_um:
        L = (pixel_size_um * 1e-3) / (electron_wavelength_A(voltage_kV) * c)
    return StandardCalibration(standard=phase.name, calib_inv_A_per_px=float(c), camera_constant_A_px=float(1 / c),
                               rings_matched=len(ps.matched), rings_detected=len(rings),
                               rms_error_pct=float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"),
                               centre=centre, ellipticity_pct=eps, ellipticity_axis_deg=phi, camera_length_mm=L,
                               table=table, notes=notes)


# ----------------------------------------------------------------------------
# Scale bars (patterns cropped from papers or exported with a burnt-in bar)
# ----------------------------------------------------------------------------

@dataclass
class ScaleBar:
    length_px: float
    bbox: tuple[int, int, int, int]  # y0, y1, x0, x1
    polarity: str  # "light" or "dark"
    score: float


def detect_scale_bars(image: np.ndarray, max_candidates: int = 3) -> list[ScaleBar]:
    """Find straight horizontal bars of uniform extreme intensity (typical burnt-in scale bars)."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    h, w = img.shape
    lo, hi = np.percentile(img, [1, 99.5])
    out = []
    for pol, mask in (("light", img >= hi - 0.02 * (hi - lo)), ("dark", img <= lo + 0.02 * (hi - lo))):
        mask = ndimage.binary_opening(mask, structure=np.ones((1, max(3, w // 60))))
        lab, n = ndimage.label(mask)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            hh = sl[0].stop - sl[0].start
            ww = sl[1].stop - sl[1].start
            if ww < 0.05 * w or ww > 0.6 * w or hh < 1 or hh > max(4, 0.06 * h) or ww < 6 * hh:
                continue
            region = lab[sl] == k
            fill = region.mean()
            if fill < 0.85:
                continue
            yc = 0.5 * (sl[0].start + sl[0].stop)
            border = min(yc, h - yc) / h  # bars sit near the top/bottom
            score = fill * (1.0 if border < 0.25 else 0.5) * min(1.0, ww / (0.1 * w))
            out.append(ScaleBar(length_px=float(ww), bbox=(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop), polarity=pol, score=float(score)))
    out.sort(key=lambda b: -b.score)
    return out[:max_candidates]


def calib_from_scale_bar(bar_length_px: float, value: float, unit: str) -> float:
    """Å⁻¹ per pixel from a reciprocal-space scale bar, e.g. 5 nm⁻¹ over 120 px."""
    u = unit.replace(" ", "").lower()
    if u in ("nm⁻¹", "1/nm", "nm-1"):
        inv_A = value / 10.0
    elif u in ("å⁻¹", "1/å", "a-1", "1/a", "å-1"):
        inv_A = value
    else:
        raise ValueError("Scale bar must be in reciprocal units (nm⁻¹ or Å⁻¹) for a diffraction pattern.")
    return inv_A / bar_length_px


def scale_bar_mask(shape, bar: ScaleBar, pad_text: float = 3.0) -> np.ndarray:
    """Mask the bar and the label text usually printed next to it."""
    y0, y1, x0, x1 = bar.bbox
    hgt = max(y1 - y0, 4)
    m = np.zeros(shape, bool)
    yy0 = max(0, int(y0 - pad_text * 8 - hgt * 4))
    yy1 = min(shape[0], int(y1 + hgt * 4 + 8))
    xx0 = max(0, int(x0 - 0.15 * (x1 - x0)))
    xx1 = min(shape[1], int(x1 + 0.15 * (x1 - x0)))
    m[yy0:yy1, xx0:xx1] = True
    return m
