"""Synthetic diffraction patterns with known ground truth (for demos & testing)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .crystal import Phase, reflections, ring_lines


def _canvas(size: int):
    y, x = np.indices((size, size), dtype=np.float64)
    return y, x


def _background(size, centre, rng, strength=1.0):
    y, x = _canvas(size)
    r = np.hypot(y - centre[0], x - centre[1])
    r0 = size * rng.uniform(0.03, 0.06)
    p = rng.uniform(1.6, 2.4)
    diffuse = strength * 60.0 / (1 + (r / r0) ** p)
    beam = 4000.0 * np.exp(-(r**2) / (2 * (size * 0.006) ** 2))
    return diffuse + beam + 2.0


def _beam_stop(img, centre, rng, enable=True):
    if not enable:
        return img, np.ones(img.shape, bool)
    size = img.shape[0]
    y, x = _canvas(size)
    ang = rng.uniform(0, 2 * np.pi)
    ux, uy = np.cos(ang), np.sin(ang)
    dx, dy = x - centre[1], y - centre[0]
    along = dx * ux + dy * uy
    perp = np.abs(-dx * uy + dy * ux)
    width = size * rng.uniform(0.018, 0.03)
    rod = (along > -width) & (perp < width)
    disc = np.hypot(dx, dy) < width * 1.6
    stop = rod | disc
    stop = ndimage.gaussian_filter(stop.astype(float), 1.5)
    out = img * (1 - 0.97 * stop)
    return out, stop < 0.5


def _noise(img, rng, counts=1.0):
    lam = np.clip(img * counts, 0, None)
    return rng.poisson(lam).astype(np.float64) / counts + rng.normal(0, 0.6, img.shape)


def ring_pattern(phase: Phase, size: int = 768, calib: float | None = None, centre=None, seed: int = 0,
                 spotty: float = 0.0, ring_width_px: float = 1.6, beam_stop: bool = True, counts: float = 1.0,
                 amplitude: float = 80.0, halo: float = 0.0, ellipticity: tuple[float, float] = (0.0, 0.0),
                 texture: float = 0.0, scale_bar: bool = False):
    """Polycrystalline ring pattern. ``spotty`` 0 (smooth) … 1 (few grains)."""
    rng = np.random.default_rng(seed)
    if centre is None:
        centre = (size / 2 + rng.uniform(-25, 25), size / 2 + rng.uniform(-25, 25))
    if calib is None:
        calib = rng.uniform(0.95, 1.25) / (0.46 * size)
    y, x = _canvas(size)
    th = np.arctan2(y - centre[0], x - centre[1])
    r = np.hypot(y - centre[0], x - centre[1])
    if ellipticity[0]:
        e, ph = ellipticity[0] / 100.0, np.radians(ellipticity[1])
        r = r / (1 + e * np.cos(2 * (th - ph)))  # rings of radius R appear at R(1 + e cos 2(θ-φ))
    img = _background(size, centre, rng)
    g_max = 0.5 * size * calib * 1.45
    lines = ring_lines(phase, g_max=g_max, min_rel_intensity=1.0)
    for ln in lines:
        R = ln.g / calib
        if R > size:
            continue
        radial = np.exp(-((r - R) ** 2) / (2 * ring_width_px**2))
        if spotty > 0:
            n_grains = int(np.clip(400 * (1 - spotty) + 15, 15, 600))
            grid = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
            mod1 = np.zeros_like(grid)
            angs = rng.uniform(-np.pi, np.pi, n_grains)
            amps = rng.lognormal(0, 0.5, n_grains)
            sig = max(0.6 * ring_width_px / max(R, 1), 2 * np.pi / 4096)
            for a, am in zip(angs, amps):
                d = np.angle(np.exp(1j * (grid - a)))
                mod1 += am * np.exp(-(d**2) / (2 * sig**2))
            mod1 = mod1 / (mod1.mean() + 1e-9)
            mod = np.interp(th, grid, mod1, period=2 * np.pi)
            radial = radial * (spotty * mod + (1 - spotty))
        if texture > 0:  # preferred orientation: intensity concentrated in arcs
            ang0 = rng.uniform(0, np.pi)
            arcs = np.exp(-(np.angle(np.exp(2j * (th - ang0))) ** 2) / (2 * (1.2 - texture) ** 2))
            radial = radial * (texture * arcs * 2 + (1 - texture))
        img += amplitude * (ln.intensity / 100.0) * radial * (1 + 0.15 * rng.normal())
    if halo > 0:
        img += halo * np.exp(-((r * calib - 0.45) ** 2) / (2 * 0.05**2))
    img, valid = _beam_stop(img, centre, rng, beam_stop)
    img = _noise(img, rng, counts)
    truth = {"type": "polycrystalline", "phase": phase.name, "calibration": calib, "centre": centre,
             "ellipticity": list(ellipticity)}
    if scale_bar:
        img, bar = burn_scale_bar(img, calib, rng)
        truth["scale_bar"] = bar
    return img, truth


def burn_scale_bar(img, calib, rng, value_nm_inv: float = 5.0):
    """Burn a white reciprocal-space scale bar (and a text-like block) into the lower-left corner."""
    h, w = img.shape
    length = value_nm_inv / 10.0 / calib
    x0, y0 = int(0.06 * w), int(0.92 * h)
    hi = float(np.percentile(img, 99.9)) * 1.5
    out = img.copy()
    out[y0:y0 + 5, x0:x0 + int(round(length))] = hi
    out[y0 - 22:y0 - 8, x0 + 4:x0 + 4 + int(0.4 * length)] = hi * 0.9  # stand-in for the "5 1/nm" label
    return out, {"value": value_nm_inv, "unit": "nm⁻¹", "length_px": float(round(length))}


def zone_spots(phase: Phase, uvw, g_max: float, double_diffraction: bool = False):
    """In-plane 2D coordinates (Å⁻¹) and intensities of ZOLZ reflections for zone [uvw]."""
    ref = reflections(phase, g_max)
    uvw = np.asarray(uvw, int)
    inz = (ref.hkl @ uvw) == 0
    if double_diffraction:
        m = inz
    else:
        m = inz & ref.allowed
    gv, hkl, F2, allowed = ref.gvec[m], ref.hkl[m], ref.F2[m], ref.allowed[m]
    zdir = uvw @ phase.structure.lattice.matrix
    zdir = zdir / np.linalg.norm(zdir)
    al = np.where(allowed)[0]
    if len(al) == 0:
        return np.zeros((0, 2)), np.zeros(0), hkl, allowed
    e1 = gv[al[np.argmin(np.linalg.norm(gv[al], axis=1))]]
    e1 = e1 - np.dot(e1, zdir) * zdir
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(zdir, e1)
    xy = np.stack([gv @ e1, gv @ e2], axis=1)
    inten = F2.copy()
    if double_diffraction:
        inten[~allowed] = 0.08 * F2[allowed].max() if allowed.any() else 0
    return xy, inten, hkl, allowed


def oriented_spots(phase: Phase, U: np.ndarray, g_max: float, voltage_kV: float = 200.0, s0: float = 0.012,
                   min_rel: float = 0.01):
    """Spots for an arbitrary crystal orientation (not just an exact zone axis).

    ``U`` rotates crystal Cartesian vectors into the lab frame (x right, y down, z along the beam). The intensity of
    each reflection is |F|² times a Gaussian in the excitation error s (distance of g from the Ewald sphere), which
    stands in for thickness, beam convergence and crystal bending. Returns lab xy (Å⁻¹), relative intensity, hkl, s.
    """
    from .physics import electron_wavelength_A

    lam = electron_wavelength_A(voltage_kV)
    ref = reflections(phase, g_max)
    m = ref.allowed & (ref.g > 0)
    gl = ref.gvec[m] @ U.T
    g2 = np.sum(gl**2, axis=1)
    s = gl[:, 2] + 0.5 * lam * g2
    w = ref.F2[m] * np.exp(-((s / s0) ** 2))
    if w.size == 0 or w.max() <= 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3), int), np.zeros(0)
    w = w / w.max()
    keep = w >= min_rel
    return gl[keep, :2], w[keep], ref.hkl[m][keep], s[keep]


def spot_pattern(phase: Phase, uvw=(0, 0, 1), size: int = 768, calib: float | None = None, centre=None, seed: int = 0,
                 rotation_deg: float | None = None, spot_sigma: float = 1.6, beam_stop: bool = True, counts: float = 1.0,
                 dynamical: float = 0.35, double_diffraction: bool = False, amplitude: float = 900.0):
    rng = np.random.default_rng(seed)
    if centre is None:
        centre = (size / 2 + rng.uniform(-25, 25), size / 2 + rng.uniform(-25, 25))
    if calib is None:
        calib = rng.uniform(0.95, 1.25) / (0.46 * size)
    if rotation_deg is None:
        rotation_deg = rng.uniform(0, 360)
    g_max = 0.5 * size * calib * 1.45
    xy, inten, hkl, allowed = zone_spots(phase, uvw, g_max, double_diffraction)
    t = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = xy @ R.T
    img = _background(size, centre, rng, strength=0.7)
    y, x = _canvas(size)
    if len(inten):
        g = np.linalg.norm(xy, axis=1)
        env = np.exp(-((g / 0.9) ** 2))
        dyn = rng.lognormal(0, dynamical, len(inten))
        amps = inten / inten.max() * env * dyn
        for (gx, gy), a in zip(xy, amps):
            px, py = centre[1] + gx / calib, centre[0] + gy / calib
            if not (0 <= px < size and 0 <= py < size):
                continue
            y0, y1 = int(max(py - 6 * spot_sigma, 0)), int(min(py + 6 * spot_sigma + 1, size))
            x0, x1 = int(max(px - 6 * spot_sigma, 0)), int(min(px + 6 * spot_sigma + 1, size))
            yy, xx = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
            img[y0:y1, x0:x1] += amplitude * a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * spot_sigma**2))
    img, valid = _beam_stop(img, centre, rng, beam_stop)
    img = _noise(img, rng, counts)
    truth = {"type": "single crystal", "phase": phase.name, "zone_axis": tuple(int(v) for v in uvw),
             "calibration": calib, "centre": centre, "rotation_deg": rotation_deg}
    return img, truth


def zone_orientation(phase: Phase, uvw, rotation_deg: float = 0.0, tilt_deg: float = 0.0, tilt_axis_deg: float = 0.0) -> np.ndarray:
    """Rotation U (crystal Cartesian → lab) that puts [uvw] along the beam (+z), rotated in-plane, then tilted off-axis."""
    zc = np.asarray(uvw, float) @ phase.structure.lattice.matrix
    zc /= np.linalg.norm(zc)
    tmp = np.array([1.0, 0, 0]) if abs(zc[0]) < 0.9 else np.array([0, 1.0, 0])
    xc = tmp - tmp.dot(zc) * zc
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    U0 = np.stack([xc, yc, zc])  # rows: crystal → lab axes
    t = np.radians(rotation_deg)
    Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    a = np.radians(tilt_axis_deg)
    ax = np.array([np.cos(a), np.sin(a), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    th = np.radians(tilt_deg)
    Rt = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return Rt @ Rz @ U0


def tilted_spot_pattern(phase: Phase, uvw=(0, 0, 1), size: int = 768, calib: float | None = None, centre=None, seed: int = 0,
                        tilt_deg: float = 1.0, s0: float = 0.010, spot_sigma: float = 1.8, beam_stop: bool = True,
                        counts: float = 1.0, dynamical: float = 0.6, double_diffraction: float = 0.0,
                        ellipticity: tuple[float, float] = (0.0, 0.0), amplitude: float = 900.0, voltage_kV: float = 200.0):
    """Realistic single-crystal pattern: slightly off the zone axis (Laue-circle intensity asymmetry), strong
    dynamical intensity scatter, optional double-diffraction spots and projector ellipticity."""
    from .physics import electron_wavelength_A

    rng = np.random.default_rng(seed)
    if centre is None:
        centre = (size / 2 + rng.uniform(-25, 25), size / 2 + rng.uniform(-25, 25))
    if calib is None:
        calib = rng.uniform(0.95, 1.25) / (0.46 * size)
    rot = rng.uniform(0, 360)
    tilt_axis = rng.uniform(0, 360)
    U = zone_orientation(phase, uvw, rot, tilt_deg, tilt_axis)
    g_max = 0.5 * size * calib * 1.45
    lam = electron_wavelength_A(voltage_kV)
    ref = reflections(phase, g_max)
    m = ref.g > 0
    gl = ref.gvec[m] @ U.T
    s = gl[:, 2] + 0.5 * lam * np.sum(gl**2, axis=1)
    F2 = ref.F2[m].copy()
    allowed = ref.allowed[m]
    fmax = F2[allowed].max()
    if double_diffraction > 0:
        F2[~allowed] = double_diffraction * fmax
    else:
        F2[~allowed] = 0
    w = F2 / fmax * np.exp(-((s / s0) ** 2))
    g = np.linalg.norm(gl[:, :2], axis=1)
    w *= np.exp(-((g / 0.9) ** 2)) * rng.lognormal(0, dynamical, len(w))
    keep = w > 0.004
    xy, w = gl[keep, :2], w[keep]
    if ellipticity[0]:
        e, ph = ellipticity[0] / 100.0, np.radians(ellipticity[1])
        th = np.arctan2(xy[:, 1], xy[:, 0])
        xy = xy * (1 + e * np.cos(2 * (th - ph)))[:, None]
    img = _background(size, centre, rng, strength=0.7)
    y, x = _canvas(size)
    if len(w):
        amps = w / w.max()
        for (gx, gy), a in zip(xy, amps):
            px, py = centre[1] + gx / calib, centre[0] + gy / calib
            if not (0 <= px < size and 0 <= py < size):
                continue
            y0, y1 = int(max(py - 6 * spot_sigma, 0)), int(min(py + 6 * spot_sigma + 1, size))
            x0, x1 = int(max(px - 6 * spot_sigma, 0)), int(min(px + 6 * spot_sigma + 1, size))
            yy, xx = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
            img[y0:y1, x0:x1] += amplitude * a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * spot_sigma**2))
    img, valid = _beam_stop(img, centre, rng, beam_stop)
    img = _noise(img, rng, counts)
    truth = {"type": "single crystal", "phase": phase.name, "zone_axis": tuple(int(v) for v in uvw), "calibration": calib,
             "centre": centre, "rotation_deg": rot, "tilt_deg": tilt_deg, "ellipticity": list(ellipticity)}
    return img, truth


def amorphous_pattern(size: int = 768, calib: float | None = None, centre=None, seed: int = 0, beam_stop: bool = True,
                      d_halo: float | None = None, counts: float = 1.0):
    rng = np.random.default_rng(seed)
    if centre is None:
        centre = (size / 2 + rng.uniform(-25, 25), size / 2 + rng.uniform(-25, 25))
    if calib is None:
        calib = rng.uniform(0.95, 1.25) / (0.46 * size)
    if d_halo is None:
        d_halo = rng.uniform(1.9, 3.2)
    y, x = _canvas(size)
    r = np.hypot(y - centre[0], x - centre[1])
    g = r * calib
    g1 = 1 / d_halo
    img = _background(size, centre, rng)
    img += 70 * np.exp(-((g - g1) ** 2) / (2 * (0.07 * g1 / 0.47) ** 2))
    img += 25 * np.exp(-((g - 1.8 * g1) ** 2) / (2 * (0.10 * g1 / 0.47) ** 2))
    img, valid = _beam_stop(img, centre, rng, beam_stop)
    img = _noise(img, rng, counts)
    return img, {"type": "amorphous", "d_halo": d_halo, "calibration": calib, "centre": centre}
