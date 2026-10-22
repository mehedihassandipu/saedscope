"""Loading diffraction patterns (.dm3/.dm4, TIFF, PNG/JPG, NPY) with calibration metadata."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .physics import scale_units_to_inv_A


@dataclass
class Calibration:
    inv_A_per_px: float | None = None  # Å⁻¹ per pixel
    voltage_kV: float | None = None
    camera_length_mm: float | None = None
    source: str = "none"  # "file metadata" | "user" | "camera length" | "standard" | "none"

    @property
    def calibrated(self) -> bool:
        return self.inv_A_per_px is not None and self.inv_A_per_px > 0


@dataclass
class Pattern:
    image: np.ndarray
    name: str
    calibration: Calibration = field(default_factory=Calibration)
    metadata: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _first_number(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return None


def _load_dm(path: str, name: str) -> Pattern:
    from rsciio.digitalmicrograph import file_reader

    dicts = file_reader(path)
    d = next((x for x in dicts if np.asarray(x["data"]).ndim == 2), dicts[0])
    img = np.asarray(d["data"], dtype=np.float64)
    if img.ndim > 2:
        img = img.reshape(-1, *img.shape[-2:]).sum(axis=0)
    cal = Calibration()
    notes = []
    axes = d.get("axes", [])
    if len(axes) >= 2:
        scale = axes[-1].get("scale")
        units = axes[-1].get("units")
        c = scale_units_to_inv_A(scale, units)
        if c:
            cal.inv_A_per_px = c
            cal.source = "file metadata"
        else:
            notes.append(f"File scale is {scale} {units!s} (not reciprocal-space); calibration needed.")
    md = d.get("metadata", {}) or {}
    cal.voltage_kV = _first_number(md, "Acquisition_instrument", "TEM", "beam_energy")
    cl = _first_number(md, "Acquisition_instrument", "TEM", "camera_length")
    if cl:
        # HyperSpy stores camera length in mm
        cal.camera_length_mm = cl
    return Pattern(image=img, name=name, calibration=cal, metadata={"axes": axes, "metadata": md}, notes=notes)


FOUR_D_EXT = (".zspy", ".hspy", ".hdf5", ".h5", ".mib")
FOUR_D_MODES = {"max": "maximum over all scan positions (spots of every grain)", "mean": "average of all scan positions",
                "brightest": "the single scan position with the strongest diffraction"}


def _load_4d(path: str, name: str, mode: str) -> Pattern:
    try:
        from .extract import extract
        info: dict = {}
        img, cal, how = extract(path, mode=mode, info=info)
    except ImportError as exc:
        raise ValueError(f"reading 4D-STEM files needs extra packages: pip install \"zarr<3\" h5py rosettasciio ({exc})") from exc
    notes = [f"4D-STEM scan {' × '.join(map(str, info.get('scan_shape', ())))}: using the {how}."]
    calib = Calibration(inv_A_per_px=cal, voltage_kV=info.get("voltage_kV") or 200.0,
                        source="file metadata" if cal else "none")
    if cal:
        notes.append(f"Calibration from the file: {cal:.5f} Å⁻¹ per pixel"
                     + (f", {info['voltage_kV']:.0f} kV." if info.get("voltage_kV") else "."))
    return Pattern(image=img, name=name, calibration=calib, notes=notes, metadata={"four_d_mode": mode, **info})


def load_pattern(source, name: str | None = None, mode: str = "max") -> Pattern:
    """Load from a path or from bytes (uploaded file). ``name`` must carry the extension for bytes.
    ``mode`` only matters for 4D-STEM files (max, mean or brightest)."""
    if isinstance(source, (bytes, bytearray)):
        if not name:
            raise ValueError("name with extension required when loading from bytes")
        ext = os.path.splitext(name)[1].lower()
        if ext in FOUR_D_EXT:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                fh.write(source)
                tmp = fh.name
            try:
                return _load_4d(tmp, name, mode)
            finally:
                os.unlink(tmp)
        if ext in (".dm3", ".dm4"):
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                fh.write(source)
                tmp = fh.name
            try:
                return _load_dm(tmp, name)
            finally:
                os.unlink(tmp)
        if ext == ".npy":
            return Pattern(image=np.load(io.BytesIO(source)).astype(np.float64), name=name)
        return _load_image(io.BytesIO(source), name)
    path = str(source)
    name = name or os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in FOUR_D_EXT:
        return _load_4d(path, name, mode)
    if ext in (".dm3", ".dm4"):
        return _load_dm(path, name)
    if ext == ".npy":
        return Pattern(image=np.load(path).astype(np.float64), name=name)
    return _load_image(path, name)


def _velox_metadata(fp) -> dict:
    """Calibration stored by Thermo Fisher Velox/TIA in TIFF exports (FEI_TITAN XML tag): reciprocal pixel size,
    voltage, camera length and detector. Empty if absent."""
    try:
        import re

        import tifffile

        if hasattr(fp, "seek"):
            fp.seek(0)
        with tifffile.TiffFile(fp) as t:
            tag = t.pages[0].tags.get("FEI_TITAN")
            xml = tag.value if tag is not None else None
        if hasattr(fp, "seek"):
            fp.seek(0)
    except Exception:
        return {}
    if not isinstance(xml, str) or "<Metadata" not in xml:
        return {}
    out = {}
    num = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, xml, re.S))
    m = re.search(r'<PixelSize>\s*<X unit="([^"]+)">\s*([-+0-9.eE]+)\s*</X>', xml)
    if m:
        unit, v = m.group(1).strip().lower(), float(m.group(2))
        per_A = {"1/m": 1e-10, "1/nm": 0.1, "1/a": 1.0, "1/å": 1.0}.get(unit)
        if per_A and v > 0:
            out["inv_A_per_px"] = v * per_A
    kv = num(r"<AccelerationVoltage>\s*([-+0-9.eE]+)\s*<")
    if kv:
        out["voltage_kV"] = kv / 1000.0 if kv > 1000 else kv
    L = num(r"<CameraLength>\s*([-+0-9.eE]+)\s*<")
    if L:
        out["camera_length_mm"] = L * 1000.0 if L < 10 else L
    d = re.search(r"<BinaryResult>.*?<Detector>([^<]+)</Detector>", xml, re.S)
    if d:
        out["detector"] = d.group(1).strip()
    mode = re.search(r"<ProjectorMode>([^<]+)<", xml)
    if mode and mode.group(1).strip().lower() != "diffraction":
        out.pop("inv_A_per_px", None)  # an image, not a diffraction pattern: the pixel size is in real space
    return out


def _load_image(fp, name: str) -> Pattern:
    import imageio.v3 as iio

    vmeta = _velox_metadata(fp) if name.lower().endswith((".tif", ".tiff")) else {}
    img = np.asarray(iio.imread(fp), dtype=np.float64)
    notes = []
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2) if img.shape[-1] in (3, 4) else img.sum(axis=0)
        notes.append("Colour image converted to greyscale.")
    if img.max() <= 255 and name.lower().endswith((".png", ".jpg", ".jpeg")):
        notes.append("8-bit image (e.g. from a paper or screenshot): intensities are compressed, positions are still usable.")
    raw = img
    img, n_cut = strip_annotation_panel(img)
    meta = {}
    if n_cut:
        notes.append(f"Removed a {n_cut}-row annotation strip (scale bar and text exported below the pattern).")
        meta["image_with_annotation"] = raw  # the scale bar lives in the strip
    calib = Calibration()
    if vmeta:
        meta.update({k: v for k, v in vmeta.items() if k in ("detector", "camera_length_mm", "voltage_kV")})
        if vmeta.get("inv_A_per_px"):
            calib = Calibration(inv_A_per_px=vmeta["inv_A_per_px"], voltage_kV=vmeta.get("voltage_kV"),
                                camera_length_mm=vmeta.get("camera_length_mm"), source="camera length")  # nominal value
            notes.append(f"Calibration from the file (Velox): {vmeta['inv_A_per_px']:.5f} Å⁻¹ per pixel"
                         + (f", {vmeta['detector']}" if vmeta.get("detector") else "")
                         + (f", camera length {vmeta['camera_length_mm']:.0f} mm" if vmeta.get("camera_length_mm") else "")
                         + (f", {vmeta['voltage_kV']:.0f} kV" if vmeta.get("voltage_kV") else "")
                         + ". This is the nominal value; a gold standard on the same camera is more accurate.")
    meta["image_shape"] = tuple(img.shape)
    return Pattern(image=img, name=name, calibration=calib, notes=notes, metadata=meta)


def strip_annotation_panel(img: np.ndarray, max_frac: float = 0.15) -> tuple[np.ndarray, int]:
    """Software such as Gatan DigitalMicrograph can export the scale bar on a flat strip below (or above) the pattern.
    Rows of that strip are almost all one value, unlike the noisy detector rows, so they are easy to find and remove."""
    h = img.shape[0]
    if h < 64:
        return img, 0

    def mode(row):
        vals, counts = np.unique(row, return_counts=True)
        k = int(np.argmax(counts))
        return vals[k], counts[k] / row.size

    def run(rows) -> int:
        bg, share = mode(rows[0])
        if share < 0.85:
            return 0
        n = 0
        for row in rows:  # the strip may hold a bar and text, so allow rows that are mostly (not all) background
            v, sh = mode(row)
            if v != bg or sh < 0.6:
                break
            n += 1
        return n

    lim = int(max_frac * h)
    bottom = run(img[::-1][:lim])
    top = run(img[:lim])
    bottom = bottom if bottom >= 16 else 0
    top = top if top >= 16 else 0
    if not (bottom or top):
        return img, 0
    return img[top:h - bottom], top + bottom
