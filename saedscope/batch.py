"""Batch analysis of many patterns → one CSV row per file (+ optional figures and JSON).

    python -m saedscope.batch path/to/folder --elements "Ni, Al, O" --calib-nm 0.0346 --out results/
    python -m saedscope.batch folder --standard gold_standard.dm4 --standard-material Au --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .io import Calibration, Pattern, load_pattern
from .pipeline import Options, analyse

EXTS = {".dm3", ".dm4", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}


def summary_row(res) -> dict:
    s = res.summary()
    row = {"file": s["file"], "pattern_type": s["pattern_type"], "confidence": s["pattern_type_confidence"],
           "calibrated": s["calibrated"], "phase": "", "space_group": "", "zone_axis": "", "score": "", "evidence": "",
           "ambiguous": "", "notes": " | ".join(s["notes"])[:300]}
    if s.get("spots", {}).get("best") and s["pattern_type"] in ("single crystal", "mixed"):
        b = s["spots"]["best"]
        row.update(phase=b["phase"]["name"], space_group=b["phase"]["space_group"], zone_axis=b["zone_axis"], score=b["score"],
                   evidence=f"{b['g1']['hkl']} {b['g1']['d_meas_A']:.3f} Å, {b['g2']['hkl']} {b['g2']['d_meas_A']:.3f} Å, {b['angle_meas_deg']:.1f}°",
                   ambiguous=s["spots"]["ambiguous"])
    elif s.get("rings", {}).get("primary_phase"):
        r = s["rings"]
        p = r["primary_phase"]
        idx = [a for a in r["assignments"] if a["phase"] == p["name"]]
        row.update(phase=p["name"] + (f" + {r['secondary_phase']['name']}" if r.get("secondary_phase") else ""),
                   space_group=p["space_group"], score=r["primary_score"],
                   evidence=", ".join(f"{a['d_meas_A']:.3f} Å ({a['hkl']})" for a in idx[:5]), ambiguous=r["ambiguous"])
    elif s.get("amorphous", {}).get("halo_d_A"):
        a = s["amorphous"]
        row.update(evidence=f"halo d = {a['halo_d_A']:.2f} Å, r_nn ≈ {a['nearest_neighbour_A_approx']:.2f} Å")
    return row


def analyse_many(paths, opts: Options, calib: Calibration | None = None, out_dir: str | None = None, figures: bool = False,
                 progress=None) -> list[dict]:
    rows = []
    for k, p in enumerate(paths):
        try:
            pat = load_pattern(p) if isinstance(p, (str, os.PathLike)) else load_pattern(p[1], p[0])
            if calib is not None and calib.calibrated and not pat.calibration.calibrated:
                pat = Pattern(pat.image, pat.name, calib, pat.metadata, pat.notes)
            res = analyse(pat, opts)
            row = summary_row(res)
            if out_dir:
                od = Path(out_dir)
                od.mkdir(parents=True, exist_ok=True)
                stem = Path(pat.name).stem
                from .report import summary_json

                (od / f"{stem}.json").write_text(summary_json(res))
                if figures:
                    from .viz import export_png

                    (od / f"{stem}.png").write_bytes(export_png(res))
        except Exception as exc:
            name = p if isinstance(p, (str, os.PathLike)) else p[0]
            row = {"file": os.path.basename(str(name)), "pattern_type": "error", "notes": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if progress:
            progress(k + 1, len(paths), row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--elements", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--calib-nm", type=float, default=None, help="calibration in nm⁻¹ per pixel")
    ap.add_argument("--calib-A", type=float, default=None, help="calibration in Å⁻¹ per pixel")
    ap.add_argument("--standard", default=None, help="pattern of a calibration standard taken at the same settings")
    ap.add_argument("--standard-material", default="Au")
    ap.add_argument("--correct-ellipticity", action="store_true",
                    help="apply the ellipticity measured on the standard (same session and camera length only)")
    ap.add_argument("--mp-key", default=os.environ.get("MP_API_KEY"))
    ap.add_argument("--out", default="saedscope_results")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    from .llm import merge_elements

    els, _ = merge_elements(args.elements, args.context)
    cal = None
    ell = None
    if args.standard:
        from .calibrate import calibrate_from_standard

        std = calibrate_from_standard(load_pattern(args.standard).image, args.standard_material)
        print(std.summary())
        cal = Calibration(inv_A_per_px=std.calib_inv_A_per_px, source="standard")
        ell = (std.ellipticity_pct, std.ellipticity_axis_deg) if (args.correct_ellipticity and std.ellipticity_pct >= 0.3) else None
    elif args.calib_nm or args.calib_A:
        cal = Calibration(inv_A_per_px=args.calib_A or args.calib_nm / 10, source="user")
    paths = sorted(str(p) for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    opts = Options(elements=els, context=args.context, mp_api_key=args.mp_key, ellipticity=ell)
    rows = analyse_many(paths, opts, cal, args.out, args.figures,
                        progress=lambda k, n, r: print(f"[{k}/{n}] {r['file']}: {r.get('pattern_type')} {r.get('phase', '')} {r.get('zone_axis', '')}"))
    out = Path(args.out) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
