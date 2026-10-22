"""Image processing for diffraction patterns: centre finding, radial profiles,
background removal, ring detection and spot detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize, signal, sparse
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------

def normalise(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:  # RGB(A) → grey
        img = img[..., :3].mean(axis=2)
    lo = np.percentile(img, 0.5)
    img = img - lo
    img[img < 0] = 0
    return img


def maybe_invert(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Invert dark-on-light figures (common in papers): the border should be dark."""
    h, w = img.shape
    b = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([img[:b].ravel(), img[-b:].ravel(), img[:, :b].ravel(), img[:, -b:].ravel()])
    centre = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if np.median(border) > np.median(centre) * 1.15 and np.median(border) > np.percentile(img, 60):
        return img.max() - img, True
    return img, False


def auto_mask(img: np.ndarray, centre: tuple[float, float] | None = None, detect_dark: bool = True) -> np.ndarray:
    """Boolean mask of usable pixels: excludes saturated pixels and very dark
    beam-stop-like regions (dark pixels much darker than their annulus)."""
    valid = np.ones(img.shape, bool)
    mx = img.max()
    if mx > 0:
        sat = img >= mx * 0.999
        if 0 < sat.mean() < 0.05:
            lab, n = ndimage.label(sat)
            if n:
                sizes = ndimage.sum(sat, lab, index=np.arange(1, n + 1))
                big = np.zeros(n + 1, bool)
                lim = 0.12 * max(img.shape)
                for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                    # a compact saturated blob is the direct beam; saturated rings/spots are kept
                    ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
                    big[k] = sizes[k - 1] > max(60, 0.0005 * img.size) and ext < lim
                valid &= ~ndimage.binary_dilation(big[lab], iterations=2)
    if centre is not None and detect_dark:
        r = radius_map(img.shape, centre)
        rb = np.clip(r.astype(int), 0, None)
        prof = _binned_percentile(img, rb, valid, 50)
        ref = prof[np.clip(rb, 0, len(prof) - 1)]
        dark = img < 0.25 * ref
        dark = ndimage.binary_opening(dark, iterations=2)
        lab, n = ndimage.label(dark)
        if n:
            sizes = ndimage.sum(dark, lab, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes > 0.002 * img.size
            valid &= ~ndimage.binary_dilation(keep[lab], iterations=3)
    return valid


def radius_map(shape, centre) -> np.ndarray:
    cy, cx = centre
    y, x = np.indices(shape, dtype=np.float64)
    return np.hypot(y - cy, x - cx)


def angle_map(shape, centre) -> np.ndarray:
    cy, cx = centre
    y, x = np.indices(shape, dtype=np.float64)
    return np.arctan2(y - cy, x - cx)


# ----------------------------------------------------------------------------
# Centre finding
# ----------------------------------------------------------------------------

def find_beamstop(img: np.ndarray) -> np.ndarray:
    """Large, very dark connected regions (beam stop / its holder), found without knowing the centre."""
    if max(img.shape) > 1024:  # work on a half/quarter-size copy of big detector images, then scale the mask back
        f = int(np.ceil(max(img.shape) / 1024))
        small = find_beamstop(_block_mean(normalise(img), f))
        big = np.repeat(np.repeat(small, f, axis=0), f, axis=1)
        out = np.zeros(img.shape, bool)
        out[: big.shape[0], : big.shape[1]] = big[: img.shape[0], : img.shape[1]]
        return out
    im = normalise(img)
    med = np.median(im)
    if med <= 0:
        return np.zeros(im.shape, bool)
    sm = ndimage.gaussian_filter(im, 2)
    dark = sm < 0.3 * ndimage.gaussian_filter(sm, max(8, im.shape[0] / 16))
    dark &= sm < 0.5 * med
    dark = ndimage.binary_opening(dark, iterations=2)
    lab, n = ndimage.label(dark)
    if not n:
        return np.zeros(im.shape, bool)
    sizes = ndimage.sum(dark, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes > 0.003 * im.size
    return ndimage.binary_dilation(keep[lab], iterations=4)


def inpaint(im: np.ndarray, mask: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Fill masked pixels by normalised Gaussian convolution of the surroundings."""
    if not mask.any():
        return im
    w = (~mask).astype(float)
    num = ndimage.gaussian_filter(im * w, sigma)
    den = ndimage.gaussian_filter(w, sigma)
    filled = num / np.maximum(den, 1e-6)
    big = ndimage.gaussian_filter(im * w, sigma * 6) / np.maximum(ndimage.gaussian_filter(w, sigma * 6), 1e-6)
    filled = np.where(den > 0.05, filled, big)
    return np.where(mask, filled, im)


def _block_mean(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape
    return img[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _block_any(mask: np.ndarray, f: int) -> np.ndarray:
    h, w = mask.shape
    return mask[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).any(axis=(1, 3))


def fast_median(img: np.ndarray, size: int) -> np.ndarray:
    """Median filter that stays fast for big windows on big images (median of a block-averaged copy, then upsampled)."""
    if size <= 25:
        return ndimage.median_filter(img, size=size)
    f = int(np.ceil(size / 9))
    small = ndimage.median_filter(_block_mean(img, f), size=max(3, int(round(size / f)) | 1))
    out = ndimage.zoom(small, (img.shape[0] / small.shape[0], img.shape[1] / small.shape[1]), order=1)
    return out[: img.shape[0], : img.shape[1]]


def annotation_mask(img: np.ndarray) -> np.ndarray:
    """Burnt-in scale bars and the label above them (paper figures, screenshots), so they are not taken for spots."""
    from .calibrate import detect_scale_bars

    h, w = img.shape
    m = np.zeros((h, w), bool)
    for b in detect_scale_bars(img):
        y0, y1, x0, x1 = b.bbox
        yc = 0.5 * (y0 + y1)
        if b.score < 0.8 or min(yc, h - yc) > 0.2 * h:
            continue
        pad = max(14, int(0.07 * h))
        if yc > h / 2:  # label above a bar at the bottom, below a bar at the top
            m[max(0, y0 - pad):min(h, y1 + 4), max(0, x0 - 4):min(w, x1 + 4)] = True
        else:
            m[max(0, y0 - 4):min(h, y1 + pad), max(0, x0 - 4):min(w, x1 + 4)] = True
    return m


def beamstop_tip(mask: np.ndarray):
    """Free end of the beam stop (where the direct beam sits): the point of the largest masked region farthest from
    where it enters the image; the centroid if it does not touch the border."""
    lab, n = ndimage.label(mask)
    if not n:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    comp = lab == k
    ys, xs = np.nonzero(comp)
    h, w = mask.shape
    edge = (ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)
    if not edge.any():
        return float(ys.mean()), float(xs.mean())
    ey, ex = ys[edge].mean(), xs[edge].mean()
    i = int(np.argmax((ys - ey) ** 2 + (xs - ex) ** 2))
    return float(ys[i]), float(xs[i])


def saturated_mask(img: np.ndarray, min_area: int | None = None) -> np.ndarray:
    """Large clipped regions (e.g. an over-exposed glow in an 8-bit figure): spots inside them cannot be seen."""
    mx = float(np.max(img))
    if mx <= 0:
        return np.zeros(img.shape, bool)
    sat = img >= 0.995 * mx
    if sat.mean() > 0.2:
        return np.zeros(img.shape, bool)
    min_area = min_area or max(40, int(0.0008 * img.size))
    lab, n = ndimage.label(sat)
    if not n:
        return sat
    sizes = ndimage.sum(sat, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_area
    return ndimage.binary_dilation(keep[lab], iterations=2)


def glow_centre(img: np.ndarray, mask: np.ndarray, start, search: float = 40.0):
    """Centre of the diffuse glow around the direct beam, from its 180° symmetry with masked pixels (beam stop) ignored.

    When a beam stop hides the direct beam, a spot lattice alone is symmetric about every half-lattice point, so the
    usual Friedel test can lock onto the wrong one; the smooth glow (inelastic and thermal diffuse scattering) is
    centred on the direct beam only. Returns (centre, cost at centre, cost at ``start``); costs are normalised."""
    if max(img.shape) > 1024:  # the glow is smooth: a smaller copy is enough, and much faster
        f = int(np.ceil(max(img.shape) / 1024))
        c_s, cost_s, cost0_s = glow_centre(_block_mean(normalise(img), f), _block_any(mask, f),
                                           ((start[0] - (f - 1) / 2) / f, (start[1] - (f - 1) / 2) / f), search / f)
        return (c_s[0] * f + (f - 1) / 2, c_s[1] * f + (f - 1) / 2), cost_s, cost0_s
    h, w = img.shape
    size = max(7, int(0.04 * min(h, w)) | 1)
    bg = fast_median(normalise(img), size)
    bg = np.log1p(bg / max(float(np.percentile(bg, 50)), 1e-9))
    valid = (~ndimage.binary_dilation(mask, iterations=3)).astype(float)
    rmin, rmax = 0.05 * min(h, w), 0.33 * min(h, w)
    st = max(2.0, rmax / 40)
    yy, xx = np.mgrid[-rmax:rmax + 1e-9:st, -rmax:rmax + 1e-9:st]
    rr = np.hypot(yy, xx)
    sel = (rr >= rmin) & (rr <= rmax) & ((yy > 0) | ((yy == 0) & (xx > 0)))
    yy, xx = yy[sel], xx[sel]

    def cost(c):
        ay, ax, by, bx = c[0] + yy, c[1] + xx, c[0] - yy, c[1] - xx
        ins = (ay >= 0) & (ay < h - 1) & (ax >= 0) & (ax < w - 1) & (by >= 0) & (by < h - 1) & (bx >= 0) & (bx < w - 1)
        ay, ax, by, bx = ay[ins], ax[ins], by[ins], bx[ins]
        ok = (ndimage.map_coordinates(valid, [ay, ax], order=0) > 0.5) & (ndimage.map_coordinates(valid, [by, bx], order=0) > 0.5)
        if ok.sum() < 150:
            return np.inf
        a = ndimage.map_coordinates(bg, [ay[ok], ax[ok]], order=1)
        b = ndimage.map_coordinates(bg, [by[ok], bx[ok]], order=1)
        return float(np.mean((a - b) ** 2) / (np.var(np.r_[a, b]) + 1e-12))

    step = max(1.0, search / 13)
    best = (cost(start), tuple(start))
    for cy in np.arange(start[0] - search, start[0] + search + 1e-9, step):
        for cx in np.arange(start[1] - search, start[1] + search + 1e-9, step):
            v = cost((cy, cx))
            if v < best[0]:
                best = (v, (float(cy), float(cx)))
    c1 = best[1]
    for cy in np.arange(c1[0] - step, c1[0] + step + 1e-9, max(0.5, step / 4)):
        for cx in np.arange(c1[1] - step, c1[1] + step + 1e-9, max(0.5, step / 4)):
            v = cost((cy, cx))
            if v < best[0]:
                best = (v, (float(cy), float(cx)))
    return best[1], best[0], cost(tuple(start))


def _autoconv_peak(a: np.ndarray, prior=None, window: float | None = None):
    a = a - a.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    s = a * win
    conv = signal.fftconvolve(s, s, mode="full")
    if prior is not None and window is not None:
        yy, xx = np.indices(conv.shape)
        conv = np.where((np.abs(yy - 2 * prior[0]) < 2 * window) & (np.abs(xx - 2 * prior[1]) < 2 * window), conv, -np.inf)
    iy, ix = np.unravel_index(np.argmax(conv), conv.shape)
    dy = _parabolic(conv[max(iy - 1, 0) : iy + 2, ix]) if 0 < iy < conv.shape[0] - 1 and np.isfinite(conv[iy - 1:iy + 2, ix]).all() else 0.0
    dx = _parabolic(conv[iy, max(ix - 1, 0) : ix + 2]) if 0 < ix < conv.shape[1] - 1 and np.isfinite(conv[iy, ix - 1:ix + 2]).all() else 0.0
    return (iy + dy) / 2.0, (ix + dx) / 2.0


def find_centre(img: np.ndarray, valid: np.ndarray | None = None, refine_rings: bool = True, use_beamstop: bool = True) -> tuple[float, float]:
    """Find the pattern centre.

    1. Mask and inpaint the beam stop (it breaks the symmetry).
    2. Coarse estimate from the bright, centrosymmetric diffuse glow around the direct beam.
    3. Friedel symmetry: the auto-convolution of the high-pass (spots/rings) image peaks at twice the centre.
    4. Refine by minimising the image − 180°-rotated-image difference (and optionally ring sharpness).
    """
    base = normalise(img)
    im = np.log1p(base / max(np.percentile(base, 50), 1e-9))
    bs = find_beamstop(base) if use_beamstop else np.zeros(base.shape, bool)
    if valid is not None:
        bs |= ~valid
    im = inpaint(im, bs)
    h, w = im.shape
    f = max(1, int(np.ceil(max(h, w) / 512)))
    small = im[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3)) if f > 1 else im
    # coarse: weighted centroid of the brightest smooth region
    lp = ndimage.gaussian_filter(small, max(3, small.shape[0] * 0.03))
    thr = np.percentile(lp, 99.0)
    yy, xx = np.indices(lp.shape)
    wts = np.clip(lp - thr, 0, None)
    c0 = (float(np.sum(wts * yy) / wts.sum()), float(np.sum(wts * xx) / wts.sum())) if wts.sum() > 0 else (small.shape[0] / 2, small.shape[1] / 2)
    # Friedel symmetry on the high-pass image, searched near the coarse estimate
    hp = np.clip(small - ndimage.gaussian_filter(small, 6), 0, None)
    hp = ndimage.gaussian_filter(hp, 1.0)
    cy, cx = _autoconv_peak(hp, prior=c0, window=0.15 * max(small.shape))
    # if the high-pass image has too little structure, fall back to the full image
    if hp.std() < 1e-6:
        cy, cx = _autoconv_peak(small, prior=c0, window=0.2 * max(small.shape))
    centre = ((cy + 0.5) * f - 0.5, (cx + 0.5) * f - 0.5)
    hp_full = ndimage.gaussian_filter(np.clip(im - ndimage.gaussian_filter(im, 6), 0, None), 1.0)
    centre = refine_centre_symmetry(hp_full + 0.3 * im, centre)
    if refine_rings:
        centre = refine_centre_profile(base, centre, ~bs if valid is None else (valid & ~bs))
    return float(centre[0]), float(centre[1])


def _parabolic(v: np.ndarray) -> float:
    if len(v) != 3:
        return 0.0
    denom = v[0] - 2 * v[1] + v[2]
    if denom == 0:
        return 0.0
    return float(0.5 * (v[0] - v[2]) / denom)


def refine_centre_symmetry(im: np.ndarray, centre, search: float = 4.0) -> tuple[float, float]:
    """Refine by minimising the difference between the image and its 180° rotation."""
    h, w = im.shape
    r = int(min(h, w) * 0.42)
    cy0, cx0 = centre
    yy, xx = np.mgrid[-r:r:3, -r:r:3].astype(float)
    m = (yy**2 + xx**2) < r**2
    yy, xx = yy[m], xx[m]

    def cost(p):
        cy, cx = p
        a = ndimage.map_coordinates(im, [cy + yy, cx + xx], order=1, mode="nearest")
        b = ndimage.map_coordinates(im, [cy - yy, cx - xx], order=1, mode="nearest")
        return float(np.mean((a - b) ** 2))

    res = optimize.minimize(cost, [cy0, cx0], method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-9, "maxiter": 200})
    cy, cx = res.x
    if abs(cy - cy0) > search * 3 or abs(cx - cx0) > search * 3:
        return centre
    return float(cy), float(cx)


def refine_centre_profile(img: np.ndarray, centre, valid=None, search: float = 2.5) -> tuple[float, float]:
    h, w = img.shape
    step = max(1, int(max(h, w) / 700))
    sub = img[::step, ::step]
    vsub = valid[::step, ::step] if valid is not None else np.ones(sub.shape, bool)
    cy0, cx0 = centre[0] / step, centre[1] / step
    y, x = np.indices(sub.shape, dtype=np.float64)
    wts = np.log1p(sub) * vsub

    def cost(p):
        r = np.hypot(y - p[0], x - p[1])
        rb = (r * 2).astype(int)
        num = np.bincount(rb.ravel(), weights=wts.ravel())
        den = np.bincount(rb.ravel(), weights=vsub.ravel().astype(float))
        prof = num / np.maximum(den, 1)
        n = int(min(sub.shape) * 0.9)
        prof = prof[4:n]
        hp = prof - ndimage.uniform_filter1d(prof, 15)
        return -float(np.sum(hp**2))

    res = optimize.minimize(cost, [cy0, cx0], method="Nelder-Mead", options={"xatol": 0.05, "maxiter": 120})
    cy, cx = res.x
    if np.hypot(cy - cy0, cx - cx0) > search:
        return centre
    return float(cy * step), float(cx * step)


# ----------------------------------------------------------------------------
# Radial profile
# ----------------------------------------------------------------------------

def _binned_percentile(img, rbin, valid, q):
    rb = rbin[valid].ravel()
    v = img[valid].ravel()
    order = np.lexsort((v, rb))
    rb, v = rb[order], v[order]
    nb = int(rbin.max()) + 1
    counts = np.bincount(rb, minlength=nb)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.zeros(nb)
    for i in range(nb):
        c = counts[i]
        if c:
            out[i] = v[starts[i] + min(c - 1, int(q / 100 * c))]
    return out


@dataclass
class Profile:
    r: np.ndarray  # radius (px) bin centres
    median: np.ndarray  # robust (median) azimuthal profile
    mean: np.ndarray
    background: np.ndarray
    signal: np.ndarray  # median - background
    noise: float
    r_min: float
    r_max: float


def radial_profile(img: np.ndarray, centre, valid=None, bin_px: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if valid is None:
        valid = np.ones(img.shape, bool)
    r = radius_map(img.shape, centre)
    rbin = (r / bin_px).astype(int)
    nb = rbin.max() + 1
    cnt = np.bincount(rbin[valid].ravel(), minlength=nb).astype(float)
    mean = np.bincount(rbin[valid].ravel(), weights=img[valid].ravel(), minlength=nb) / np.maximum(cnt, 1)
    med = _binned_percentile(img, rbin, valid, 50)
    rc = (np.arange(nb) + 0.5) * bin_px
    return rc, med, mean


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 12) -> np.ndarray:
    """Asymmetric least-squares baseline (Eilers & Boelens)."""
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2), dtype=float)
    DD = lam * D.dot(D.T)
    w = np.ones(n)
    z = y
    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def is_quantised(img: np.ndarray) -> bool:
    """8-bit-like images (PNG/JPG, paper figures): medians become staircases, so use means."""
    sample = img[:: max(1, img.shape[0] // 256), :: max(1, img.shape[1] // 256)]
    return len(np.unique(sample)) <= 300


def build_profile(img: np.ndarray, centre, valid=None, r_min: float | None = None, bin_px: float = 0.5, stat: str = "auto") -> Profile:
    rc, med, mean = radial_profile(img, centre, valid, bin_px)
    if stat == "mean" or (stat == "auto" and is_quantised(img)):
        med = mean
    h, w = img.shape
    cy, cx = centre
    r_edge = min(cy, cx, h - 1 - cy, w - 1 - cx)
    r_max = max(10.0, r_edge * 0.98)
    if r_min is None:
        r_min = estimate_r_min(rc, med, r_max)
    m = (rc >= r_min) & (rc <= r_max)
    r, med_c, mean_c = rc[m], med[m], mean[m]
    logp = np.log(np.maximum(med_c, 1e-6) + 1e-3 * max(med_c.max(), 1e-9))
    lam = (len(r) ** 2) * 2.0
    base_log = als_baseline(logp, lam=lam, p=0.02)
    bg = np.exp(base_log) - 1e-3 * max(med_c.max(), 1e-9)
    sig = med_c - bg
    hp = sig - ndimage.median_filter(sig, size=9)
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp)))) or float(np.std(sig) * 0.05 + 1e-9)
    # On small detectors (e.g. 256 px) rings are only a few bins narrower than the 9-bin filter, so the residual above
    # contains ring curvature and overestimates the noise. Bin-to-bin differences see only the noise; trust them up to 4x.
    if len(sig) > 8:
        d2 = sig[1:-1] - 0.5 * (sig[:-2] + sig[2:])
        noise_d2 = float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(1.5))
        if noise_d2 > 0:
            noise = min(noise, max(noise_d2, 0.25 * noise))
    return Profile(r=r, median=med_c, mean=mean_c, background=bg, signal=sig, noise=noise, r_min=float(r_min), r_max=float(r_max))


def estimate_r_min(rc: np.ndarray, prof: np.ndarray, r_max: float) -> float:
    """Skip the direct beam / beam stop: first clear minimum after the central maximum."""
    lim = rc < r_max * 0.5
    p = ndimage.gaussian_filter1d(prof[lim], 2)
    if len(p) < 10:
        return 5.0
    start = 3
    i_max = start + int(np.argmax(p[start : max(start + 1, len(p) // 3)]))
    mins = signal.argrelmin(p[i_max:], order=4)[0]
    base = rc[i_max + mins[0]] if len(mins) else rc[i_max] + 0.04 * r_max
    return float(min(max(base, 0.04 * r_max, 4.0), 0.13 * r_max))


# ----------------------------------------------------------------------------
# Ring detection
# ----------------------------------------------------------------------------

@dataclass
class Ring:
    radius: float  # px
    height: float
    fwhm: float  # px
    snr: float
    uniformity_cv: float = float("nan")  # robust angular coefficient of variation
    rel_width: float = 0.0  # fwhm / radius
    kind: str = "sharp"  # "sharp" | "broad"

    def to_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in self.__dict__.items()}


# A sharp ring found on top of a halo must reach this fraction of the halo's height. The running median leaves a bump
# of at most ~2.6% at a smooth halo's apex (measured on simulated halos); real rings on halos measured 6-19%.
MIN_RING_ON_HALO = 0.045


def detect_rings(prof: Profile, min_snr: float = 4.0, max_rings: int = 25) -> list[Ring]:
    sig = ndimage.gaussian_filter1d(prof.signal, 1.0)
    step = prof.r[1] - prof.r[0] if len(prof.r) > 1 else 1.0
    prom = max(min_snr * prof.noise, 0.02 * max(sig.max(), 1e-9))
    peaks, props = signal.find_peaks(sig, prominence=prom, distance=max(2, int(2 / step)))
    if len(peaks) == 0:
        return []
    widths, _, _, _ = signal.peak_widths(sig, peaks, rel_height=0.5)
    rings = []
    for pk, wd, pr in zip(peaks, widths, props["prominences"]):
        # sub-bin refinement: centroid of the top part of the peak
        lo, hi = max(pk - 2, 0), min(pk + 3, len(sig))
        seg = sig[lo:hi] - sig[lo:hi].min()
        rr = prof.r[lo:hi]
        rad = float(np.sum(seg * rr) / np.sum(seg)) if seg.sum() > 0 else float(prof.r[pk])
        fwhm = float(wd * step)
        rings.append(Ring(radius=rad, height=float(sig[pk]), fwhm=fwhm, snr=float(pr / max(prof.noise, 1e-12)), rel_width=fwhm / max(rad, 1e-9)))
    # suppress sub-peaks on the shoulders / saturated plateau of a stronger ring
    rings.sort(key=lambda r: -r.height)
    kept: list[Ring] = []
    for r in rings:
        if any(abs(r.radius - k.radius) < max(2.0, 0.6 * k.fwhm) for k in kept):
            continue
        kept.append(r)
    rings = kept[:max_rings]
    rings.sort(key=lambda r: r.radius)
    for r in rings:
        r.kind = "broad" if r.rel_width > 0.09 else "sharp"
    rings = _instrument_limited(prof, sig, rings)
    # sharp rings riding on a broad feature are merged into it above: a support-film halo, or the glow of the direct
    # beam around a bright inner ring (e.g. Au 111 on a calibration pattern). Keep the broad hosts for the search below.
    broad = [r for r in rings if r.kind == "broad"]
    # drop the tail of the direct beam that survives background subtraction
    rings = [r for r in rings if not (r.kind == "broad" and r.radius < 2.2 * prof.r_min)]
    if broad:
        win = max(9, int(round(8.0 / step)) | 1)
        hp2 = sig - ndimage.median_filter(sig, size=win)
        pk2, pp2 = signal.find_peaks(hp2, prominence=max(6.0 * prof.noise, 1e-12), distance=max(2, int(2 / step)))
        if len(pk2):
            w2, _, _, _ = signal.peak_widths(hp2, pk2, rel_height=0.5)
            sharp_now = [r for r in rings if r.kind == "sharp"]
            for pk, wd, pr in zip(pk2, w2, pp2["prominences"]):
                rad, fwhm = float(prof.r[pk]), float(wd * step)
                if fwhm < 1.0 or fwhm / max(rad, 1e-9) > 0.05:
                    continue
                host = [b for b in broad if abs(rad - b.radius) < 0.8 * b.fwhm]
                if not host:
                    continue
                # the running median leaves a small bump at a halo's apex; a real ring stands out from it
                if hp2[pk] < MIN_RING_ON_HALO * max(b.height for b in host):
                    continue
                if any(abs(rad - q.radius) < max(2.0, q.fwhm) for q in sharp_now):
                    continue
                lo, hi = max(pk - 2, 0), min(pk + 3, len(hp2))
                seg = hp2[lo:hi] - hp2[lo:hi].min()
                if seg.sum() > 0:
                    rad = float(np.sum(seg * prof.r[lo:hi]) / np.sum(seg))
                rings.append(Ring(radius=rad, height=float(hp2[pk]), fwhm=fwhm, snr=float(pr / max(prof.noise, 1e-12)),
                                  rel_width=fwhm / max(rad, 1e-9), kind="sharp"))
            rings.sort(key=lambda r: r.radius)
    return rings


def _gauss1(x, a, m, s, b):
    return a * np.exp(-0.5 * ((x - m) / s) ** 2) + b


def _gauss2(x, a1, m1, a2, m2, s, b):
    return a1 * np.exp(-0.5 * ((x - m1) / s) ** 2) + a2 * np.exp(-0.5 * ((x - m2) / s) ** 2) + b


def _instrument_limited(prof: Profile, sig: np.ndarray, rings: list[Ring]) -> list[Ring]:
    """Small detectors: a ring only a few pixels wide can exceed the relative-width limit for 'sharp'. If the pattern has
    clearly sharp rings, peaks at most twice their typical width are crystalline rings too (limited by the detector and
    beam), and one that is clearly wider may be two unresolved rings (e.g. fcc 111 + 200): try splitting it."""
    from scipy.optimize import curve_fit

    ref = [r for r in rings if r.kind == "sharp" and r.snr >= 10 and r.rel_width <= 0.08 and r.radius > 1.5 * prof.r_min]
    if not ref:
        return rings
    w_ref = float(np.median([r.fwhm for r in ref]))
    if not 2.5 <= w_ref <= 8.0:  # only rings a few pixels wide: a well-sampled pattern is handled by the relative rule
        return rings
    s_ref = w_ref / 2.355
    radii = sorted(r.radius for r in rings)
    out = []
    for r in rings:
        if r.kind != "broad" or r.rel_width > 0.2 or r.fwhm > 2.0 * w_ref or r.radius < 2.2 * prof.r_min or r.snr < 6:
            out.append(r)
            continue
        r.kind = "sharp"
        split = None
        # fit window: around this peak, but not reaching the neighbouring detected rings
        lo_r = max([q for q in radii if q < r.radius - 1e-6], default=-np.inf)
        hi_r = min([q for q in radii if q > r.radius + 1e-6], default=np.inf)
        a_lo = max(r.radius - 1.3 * r.fwhm, 0.5 * (r.radius + lo_r) if np.isfinite(lo_r) else -np.inf)
        a_hi = min(r.radius + 1.3 * r.fwhm, 0.5 * (r.radius + hi_r) if np.isfinite(hi_r) else np.inf)
        m = (prof.r >= a_lo) & (prof.r <= a_hi)
        x, y = prof.r[m], sig[m]
        if len(x) >= 10:
            try:
                p1, _ = curve_fit(_gauss1, x, y, p0=[y.max(), r.radius, r.fwhm / 2.355, 0.0], maxfev=5000)
                sse1 = float(np.sum((y - _gauss1(x, *p1)) ** 2))
                if abs(p1[2]) >= 1.3 * s_ref:
                    best = None
                    for side in (-1, 1):
                        p0 = [y.max(), r.radius - 0.2 * side * r.fwhm, 0.4 * y.max(), r.radius + 0.5 * side * r.fwhm, s_ref, 0.0]
                        lo = [0, a_lo, 0, a_lo, 0.6 * s_ref, -np.inf]
                        hi = [np.inf, a_hi, np.inf, a_hi, 1.1 * s_ref, np.inf]
                        try:
                            p2, _ = curve_fit(_gauss2, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
                        except Exception:
                            continue
                        sse2 = float(np.sum((y - _gauss2(x, *p2)) ** 2))
                        if best is None or sse2 < best[0]:
                            best = (sse2, p2)
                    if best is not None and best[0] < 0.25 * sse1:
                        a1, m1, a2, m2, sg, _ = best[1]
                        if (min(a1, a2) >= 0.15 * max(a1, a2) and abs(m2 - m1) >= 1.5 * sg
                                and all(abs(mm - r.radius) <= r.fwhm for mm in (m1, m2))):
                            split = [(m1, a1), (m2, a2)]
                            fw = 2.355 * sg
            except Exception:
                split = None
        if split:
            for mm, aa in sorted(split):
                out.append(Ring(radius=float(mm), height=float(aa), fwhm=float(fw), snr=float(aa / max(prof.noise, 1e-12)),
                                rel_width=float(fw / max(mm, 1e-9)), kind="sharp"))
        else:
            out.append(r)
    out.sort(key=lambda q: q.radius)
    return out


def ring_uniformity(img: np.ndarray, centre, ring: Ring, valid=None, n_sectors: int = 72) -> float:
    """Robust coefficient of variation of intensity around the ring (low = continuous ring)."""
    r = radius_map(img.shape, centre)
    half = max(1.0, ring.fwhm * 0.5)
    m = np.abs(r - ring.radius) <= half
    if valid is not None:
        m &= valid
    if m.sum() < n_sectors:
        return float("nan")
    th = angle_map(img.shape, centre)[m]
    v = img[m]
    sec = ((th + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    cnt = np.bincount(sec, minlength=n_sectors)
    s = np.bincount(sec, weights=v, minlength=n_sectors) / np.maximum(cnt, 1)
    s = s[cnt > 3]
    if len(s) < n_sectors // 3:
        return float("nan")
    med = np.median(s)
    if med <= 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(s - med)) / med + (np.percentile(s, 95) - med) / (4 * med))


# ----------------------------------------------------------------------------
# Spot detection
# ----------------------------------------------------------------------------

@dataclass
class Spots:
    yx: np.ndarray  # (N, 2) sub-pixel positions
    intensity: np.ndarray  # (N,)
    snr: np.ndarray  # (N,)
    noise: float
    residual: np.ndarray = field(repr=False, default=None)


def radial_background_image(img: np.ndarray, centre, valid=None, bin_px: float = 1.0) -> np.ndarray:
    rc, med, _ = radial_profile(img, centre, valid, bin_px)
    r = radius_map(img.shape, centre)
    return np.interp(r, rc, med)


def detect_spots(img: np.ndarray, centre, valid=None, r_min: float = 0.0, min_distance: int | None = None, k_sigma: float = 7.0,
                 max_spots: int = 300, local_bg: bool = False) -> Spots:
    """``local_bg``: also subtract a local median background (for a glow that is not round, e.g. around a beam stop)."""
    from skimage.feature import peak_local_max

    h, w = img.shape
    bg = radial_background_image(img, centre, valid)
    if local_bg:
        md = min_distance if min_distance is not None else max(3, int(min(h, w) * 0.012))
        bg = np.maximum(bg, fast_median(img, 2 * md + 3))
    res = ndimage.gaussian_filter(img - bg, 1.0)
    if valid is not None:
        res = np.where(valid, res, 0)
    # radius-dependent noise level (bright rings/halos are noisier: Poisson statistics)
    r = radius_map(img.shape, centre)
    rbin = (r / 2.0).astype(int)
    vmask = valid if valid is not None else np.ones(img.shape, bool)
    # centre the residual on its per-radius median first: low-count, log-scaled images have skewed noise, and the
    # MAD around zero would then overestimate the noise and hide weak spots
    med_r = _binned_percentile(res, rbin, vmask, 50)
    med_map = med_r[np.clip(rbin, 0, len(med_r) - 1)]
    mad = _binned_percentile(np.abs(res - med_map), rbin, vmask, 50)
    sig_r = ndimage.median_filter(1.4826 * mad, size=9)
    sig_r = np.maximum(sig_r, 1e-9)
    sigma_map = sig_r[np.clip(rbin, 0, len(sig_r) - 1)]
    noise = float(np.median(sig_r[sig_r > 1e-9])) if np.any(sig_r > 1e-9) else 1e-9
    z = (res - med_map) / sigma_map
    if min_distance is None:
        min_distance = max(3, int(min(h, w) * 0.012))
    pk = peak_local_max(z, min_distance=min_distance, threshold_abs=k_sigma, exclude_border=2, num_peaks=max_spots * 2)
    if len(pk) == 0:
        return Spots(yx=np.zeros((0, 2)), intensity=np.zeros(0), snr=np.zeros(0), noise=noise, residual=res)
    rp = np.hypot(pk[:, 0] - centre[0], pk[:, 1] - centre[1])
    pk = pk[rp > max(r_min, 2.0)]
    yx, inten = [], []
    half = max(2, min_distance // 2)
    for y, x in pk:
        y0, y1 = max(y - half, 0), min(y + half + 1, h)
        x0, x1 = max(x - half, 0), min(x + half + 1, w)
        win = np.clip(res[y0:y1, x0:x1], 0, None)
        if win.sum() <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        yx.append([np.sum(win * yy) / win.sum(), np.sum(win * xx) / win.sum()])
        inten.append(win.sum())
    yx, inten = np.array(yx), np.array(inten)
    order = np.argsort(-inten)[:max_spots]
    yx, inten = yx[order], inten[order]
    if len(yx) == 0:
        return Spots(yx=np.zeros((0, 2)), intensity=np.zeros(0), snr=np.zeros(0), noise=noise, residual=res)
    snr = ndimage.map_coordinates(z, yx.T, order=1)
    return Spots(yx=yx, intensity=inten, snr=snr, noise=noise, residual=res)


def friedel_vote_centre(yx: np.ndarray, centre, max_shift: float = 25.0, bin_px: float = 1.0, n_use: int = 150) -> tuple[tuple[float, float], int]:
    """Centre from Friedel-pair midpoints: every pair (g, −g) votes for the same point, noise pairs scatter.

    Works when the first centre estimate is off by up to ``max_shift`` pixels (noisy, few-spot patterns)."""
    pts = np.asarray(yx, float)[:n_use]
    if len(pts) < 4:
        return centre, 0
    c = np.array(centre, float)
    r = np.linalg.norm(pts - c, axis=1)
    i, j = np.triu_indices(len(pts), 1)
    mid = 0.5 * (pts[i] + pts[j])
    ok = (np.linalg.norm(mid - c, axis=1) < max_shift) & (np.abs(r[i] - r[j]) < 2 * max_shift) & (r[i] > max_shift) & (r[j] > max_shift)
    mid = mid[ok]
    if len(mid) < 2:
        return centre, 0
    n = int(2 * max_shift / bin_px) + 1
    H, ye, xe = np.histogram2d(mid[:, 0], mid[:, 1], bins=n, range=[[c[0] - max_shift, c[0] + max_shift], [c[1] - max_shift, c[1] + max_shift]])
    H = ndimage.uniform_filter(H, 3, mode="constant") * 9  # votes within ±1.5 bins
    k = np.unravel_index(int(np.argmax(H)), H.shape)
    votes = int(round(H[k]))
    cy, cx = 0.5 * (ye[k[0]] + ye[k[0] + 1]), 0.5 * (xe[k[1]] + xe[k[1] + 1])
    near = np.linalg.norm(mid - np.array([cy, cx]), axis=1) < 2 * bin_px
    if near.sum() >= 2:
        cy, cx = np.median(mid[near], axis=0)
    return (float(cy), float(cx)), votes


def friedel_paired(yx: np.ndarray, centre, tol: float = 3.0) -> np.ndarray:
    """Boolean mask of spots whose Friedel partner (2c − p) is also detected."""
    pts = np.asarray(yx, float)
    if len(pts) < 2:
        return np.zeros(len(pts), bool)
    c = np.array(centre, float)
    out = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - (2 * c - p), axis=1)
        d[i] = np.inf
        out[i] = d.min() < tol
    return out


def spots_subset(sp: Spots, mask: np.ndarray) -> Spots:
    return Spots(yx=sp.yx[mask], intensity=sp.intensity[mask], snr=sp.snr[mask], noise=sp.noise, residual=sp.residual)


def refine_centre_friedel(spots: Spots, centre, tol: float | None = None, n_use: int = 60) -> tuple[tuple[float, float], int]:
    """Refine the centre from the *positions* of Friedel pairs (g, -g).

    Uses geometry only, so it is robust to Friedel-law violations in intensity
    (dynamical scattering, non-centrosymmetric crystals)."""
    if len(spots.yx) < 4:
        return centre, 0
    pts = spots.yx[:n_use]
    c = np.array(centre, float)
    rr = np.linalg.norm(pts - c, axis=1)
    if tol is None:
        tol = max(3.0, 0.03 * float(np.median(rr)))
    mids: list = []
    for _ in range(4):
        mids = []
        for i, p in enumerate(pts):
            q_expected = 2 * c - p
            d = np.linalg.norm(pts - q_expected, axis=1)
            j = int(np.argmin(d))
            if j != i and d[j] < tol:
                mids.append((p + pts[j]) / 2)
        if len(mids) < 3:
            return centre, len(mids)
        c_new = np.median(np.array(mids), axis=0)
        if np.linalg.norm(c_new - c) < 0.05:
            c = c_new
            break
        c = c_new
        tol = max(2.0, tol * 0.7)
    return (float(c[0]), float(c[1])), len(mids)
