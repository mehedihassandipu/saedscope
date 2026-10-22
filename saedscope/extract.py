"""Turn a 4D-STEM / scanning-diffraction file into one diffraction pattern SAEDscope can read.

    python -m saedscope.extract mgo_nano.zspy                  # mean pattern -> mgo_nano_mean.npy (+ .png preview)
    python -m saedscope.extract mgo_nano.zspy --mode max       # max projection: spots of every grain
    python -m saedscope.extract mgo_nano.zspy --mode brightest # the scan position with the strongest diffraction
    python -m saedscope.extract twinned_nanowire.hdf5 --mode pos --x 20 --y 35

Reads .zspy, .hspy, .hdf5/.h5 (HyperSpy format), .mib and anything else RosettaSciIO reads. Needs `pip install zarr h5py`
for .zspy/.hspy. The calibration of the diffraction axes (if stored) is printed so you can type it into the app.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _read(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".zspy":
        from rsciio.zspy import file_reader
    elif ext in (".hspy", ".hdf5", ".h5"):
        from rsciio.hspy import file_reader
    elif ext == ".mib":
        from rsciio.quantumdetector import file_reader
    else:
        from rsciio.utils.tools import get_file_handle  # noqa: F401  (validate rsciio is present)
        import rsciio

        for plugin in rsciio.IO_PLUGINS:
            if ext.lstrip(".") in [e.lower() for e in plugin.get("file_extensions", [])]:
                mod = __import__(plugin["api"], fromlist=["file_reader"])
                file_reader = mod.file_reader
                break
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    items = file_reader(path, lazy=True)
    return items[0]


def _signal_calibration(axes) -> float | None:
    """Å⁻¹ per pixel from the last two (signal) axes, if their units are reciprocal lengths."""
    sig = [a for a in axes if a.get("navigate") is False] or axes[-2:]
    a = sig[-1]
    units = str(a.get("units") or "").replace(" ", "").replace("$", "").replace("{", "").replace("}", "").replace("\\", "").lower()
    units = units.replace("^-1", "-1").replace("^−1", "-1").replace("⁻¹", "-1").replace("angstrom", "a").replace("å", "a")
    scale = a.get("scale")
    if not scale or scale == 1.0:
        return None
    if units in ("1/nm", "nm-1"):
        return float(scale) / 10.0
    if units in ("1/a", "a-1"):
        return float(scale)
    return None


def _beam_energy_kV(meta) -> float | None:
    try:
        v = float(meta["Acquisition_instrument"]["TEM"]["beam_energy"])
        return v if 20 <= v <= 1500 else None
    except (KeyError, TypeError, ValueError):
        return None


def extract(path: str, mode: str = "mean", x: int | None = None, y: int | None = None,
            info: dict | None = None) -> tuple[np.ndarray, float | None, str]:
    """One 2D pattern from a 4D dataset. Pass a dict as ``info`` to receive the beam energy and scan shape."""
    item = _read(path)
    data = item["data"]
    axes = item["axes"]
    if info is not None:
        info["voltage_kV"] = _beam_energy_kV(item.get("metadata") or {})
        info["scan_shape"] = tuple(int(n) for n in data.shape[:-2])
        info["title"] = str(((item.get("metadata") or {}).get("General") or {}).get("title") or "")
    if data.ndim < 2:
        raise ValueError("Not an image or diffraction dataset.")
    if data.ndim == 2:
        return np.asarray(data, dtype=np.float64), _signal_calibration(axes), "single pattern"
    nav_shape = data.shape[:-2]
    flat = data.reshape((-1,) + data.shape[-2:])
    if mode == "mean":
        out = flat.mean(axis=0)
        how = f"mean of {flat.shape[0]} patterns"
    elif mode == "max":
        out = flat.max(axis=0)
        how = f"maximum over {flat.shape[0]} patterns"
    elif mode == "pos":
        if x is None or y is None:
            raise ValueError("--mode pos needs --x and --y (scan position)")
        out = data[(y, x) if len(nav_shape) == 2 else (x,)]
        how = f"scan position x={x}, y={y}"
    elif mode == "brightest":
        # strongest diffraction away from the direct beam: high-pass energy of each pattern
        h, w = data.shape[-2:]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        ring = ((yy - cy) ** 2 + (xx - cx) ** 2) > (0.08 * min(h, w)) ** 2
        energy = (flat * ring).std(axis=(1, 2))
        try:
            energy = energy.compute()
        except AttributeError:
            pass
        k = int(np.argmax(energy))
        out = flat[k]
        pos = np.unravel_index(k, nav_shape)
        how = f"brightest scan position {tuple(int(v) for v in pos)[::-1]} (x, y)"
    else:
        raise ValueError("mode must be mean, max, pos or brightest")
    try:
        out = out.compute()
    except AttributeError:
        pass
    return np.asarray(out, dtype=np.float64), _signal_calibration(axes), how


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--mode", default="mean", choices=["mean", "max", "pos", "brightest"])
    ap.add_argument("--x", type=int)
    ap.add_argument("--y", type=int)
    ap.add_argument("--out", default=None, help="output .npy (default: <file>_<mode>.npy)")
    args = ap.parse_args()
    img, cal, how = extract(args.file, args.mode, args.x, args.y)
    out = args.out or f"{Path(args.file).with_suffix('')}_{args.mode}.npy"
    np.save(out, img)
    try:
        import imageio.v3 as iio

        from .viz import display_image

        iio.imwrite(os.path.splitext(out)[0] + "_preview.png", display_image(img, max_px=1024)[0])
    except Exception:
        pass
    print(f"Saved {out} ({img.shape[1]} × {img.shape[0]} px, {how}).")
    if cal:
        print(f"Calibration stored in the file: {cal:.6f} Å⁻¹ per pixel ({cal * 10:.5f} nm⁻¹ per pixel).")
        print("In the app choose Calibration -> 'Å⁻¹ per pixel' and type this number (check it against a gold standard if you can).")
    else:
        print("No reciprocal-space calibration found in the file: calibrate in the app (standard, scale bar or camera length).")


if __name__ == "__main__":
    main()
