"""Figures for the write-up and slides.   python docs/make_figures.py   →  docs/figures/*.png"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saedscope import synth  # noqa: E402
from saedscope.crystal import builtin_library, fmt_index  # noqa: E402
from saedscope.io import Calibration, Pattern  # noqa: E402
from saedscope.pipeline import Options, analyse  # noqa: E402
from saedscope.viz import display_image  # noqa: E402

OUT = ROOT / "docs" / "figures"
GREEN, INK, MUTED, LINE, PALE = "#3F6212", "#1F2A37", "#6B7280", "#CBD5C0", "#F3F5F2"
ACCENT = ["#0F766E", "#C2410C", "#7C3AED", "#2563EB", "#BE185D"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "savefig.bbox": "tight"})
LIB = {p.name: p for p in builtin_library()}


def box(ax, x, y, w, h, title, body="", fc="white", ec=LINE, tc=INK, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=10.5, color=tc, weight="bold" if bold else "normal")
    if body:
        ax.text(x + w / 2, y + h - 0.062, body, ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.35)


def arrow(ax, p, q, color=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color, shrinkA=2, shrinkB=2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.03, 1)
    ax.set_axis_off()
    box(ax, 0.005, 0.54, 0.15, 0.40, "Inputs", "pattern: .dm4, .tif, .png\nelements from EDS\n\"annealed in air\"\ncalibration: file, gold\nstandard or scale bar")
    box(ax, 0.005, 0.06, 0.15, 0.36, "Candidates", "built-in library (31)\nMaterials Project\nyour CIF files\nfiltered by elements")
    box(ax, 0.195, 0.54, 0.15, 0.40, "Image", "beam stop masked\ncentre from Friedel\nsymmetry\nellipticity corrected\nprofile and spots")
    box(ax, 0.385, 0.54, 0.15, 0.40, "Pattern type", "even sharp rings:\npolycrystal\nspots on a 2D lattice:\nsingle crystal\nbroad halo: amorphous")
    box(ax, 0.595, 0.71, 0.18, 0.23, "Rings", "d-spacings matched to\nkinematical ring lists")
    box(ax, 0.595, 0.385, 0.18, 0.28, "Spots", "plane pairs fit d₁, d₂, angle\nzone axis = h₁ × h₂\nchecked by simulation")
    box(ax, 0.595, 0.06, 0.18, 0.28, "Halo", "halo d → nearest-\nneighbour distance\n(Ehrenfest)")
    box(ax, 0.82, 0.54, 0.175, 0.40, "Answer", "phase and zone axis\nevery reflection indexed\nerrors, look-alikes\nflagged\nfigure, caption, CIF", fc=PALE, ec=GREEN)
    box(ax, 0.82, 0.06, 0.175, 0.40, "Next step", "the tilt that separates\nlook-alikes, with the\npattern each predicts\nexplanation: a language\nmodel only rewords\nthe numbers", fc=PALE, ec=GREEN)
    arrow(ax, (0.155, 0.74), (0.195, 0.74))
    arrow(ax, (0.345, 0.74), (0.385, 0.74))
    for y in (0.86, 0.60, 0.28):
        arrow(ax, (0.535, 0.74), (0.595, y))
    ax.plot([0.155, 0.565], [0.20, 0.20], color=LINE, lw=1.4)
    ax.plot([0.565, 0.565], [0.12, 0.76], color=LINE, lw=1.4)
    for y in (0.76, 0.45, 0.12):
        arrow(ax, (0.565, y), (0.595, y), color="#A3B18A")
    for y in (0.825, 0.525, 0.20):
        arrow(ax, (0.775, y), (0.82, 0.74 if y > 0.3 else 0.62))
    arrow(ax, (0.9075, 0.54), (0.9075, 0.46), color=GREEN)
    ax.text(0.005, 0.0, "No training data: every number comes from physics (electron wavelength, structure factors, Weiss zone law).",
            ha="left", fontsize=9, color=MUTED)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=200)
    plt.close(fig)


def _overlay(ax, res, title):
    im, f = display_image(res.image, max_px=900)
    ax.imshow(im, cmap="gray")
    ax.set_axis_off()
    cy, cx = res.centre[0] / f, res.centre[1] / f
    k = res.classification.kind
    if k in ("single crystal", "mixed") and res.spot_result and res.spot_result.solutions:
        sol = res.spot_result.solutions[0]
        c0 = res.spot_result.centre
        for s in sol.indexed:
            if s["hkl"]:
                x, y = c0[1] / f + s["dx"] / f, c0[0] / f + s["dy"] / f
                ax.add_patch(plt.Circle((x, y), 7, fill=False, color="#2DD4BF", lw=1))
                if np.hypot(s["dx"], s["dy"]) / f < 160:
                    ax.text(x, y - 10, s["hkl"], color="#99F6E4", fontsize=6.5, ha="center")
    elif res.ring_result and res.ring_result.primary:
        for a in res.ring_result.assignments[:9]:
            col = "#2DD4BF" if a.phase else "#9CA3AF"
            ax.add_patch(plt.Circle((cx, cy), a.radius_px / f, fill=False, color=col, lw=0.9))
            if a.hkl:
                ang = -np.pi / 3.2 + 0.2 * (a.ring_index % 5)
                ax.text(cx + a.radius_px / f * np.cos(ang) + 3, cy + a.radius_px / f * np.sin(ang), a.hkl, color="#99F6E4", fontsize=7)
    elif res.amorphous and res.amorphous.get("halo_radius_px"):
        ax.add_patch(plt.Circle((cx, cy), res.amorphous["halo_radius_px"] / f, fill=False, color="#FBBF24", lw=1.2, ls="--"))
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def fig_examples():
    cases = []
    img, t = synth.ring_pattern(LIB["NiAl (B2)"], seed=21, size=900, spotty=0.3)
    cases.append((img, t, ["Ni", "Al", "O"]))
    img, t = synth.spot_pattern(LIB["Fe₃O₄ (magnetite)"], (1, 1, 2), seed=22, size=900)
    cases.append((img, t, ["Fe", "O"]))
    img, t = synth.amorphous_pattern(seed=23, size=900, d_halo=2.35)
    cases.append((img, t, ["Zr", "Cu"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (img, t, els) in zip(axes, cases):
        res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=t["calibration"])), Options(elements=els, advise=False))
        s = res.summary()
        if res.classification.kind == "single crystal":
            b = s["spots"]["best"]
            title = f"Single crystal: {b['phase']['name']} {b['zone_axis']}\n{b['g1']['hkl']} {b['g1']['d_meas_A']:.3f} Å, {b['g2']['hkl']} {b['g2']['d_meas_A']:.3f} Å, {b['angle_meas_deg']:.1f}°"
        elif res.classification.kind == "polycrystalline":
            r = s["rings"]
            n = sum(1 for a in r["assignments"] if a["phase"] == r["primary_phase"]["name"])
            title = f"Polycrystal: {r['primary_phase']['name']}\n{n} of {len(r['assignments'])} rings indexed"
        else:
            a = s["amorphous"]
            title = f"Amorphous: halo at d = {a['halo_d_A']:.2f} Å\nnearest neighbours ≈ {a['nearest_neighbour_A_approx']:.2f} Å"
        _overlay(ax, res, title)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_examples.png", dpi=200)
    plt.close(fig)


def fig_advisor():
    from saedscope.synth import oriented_spots

    img, t = synth.spot_pattern(LIB["α-Ti"], (0, 0, 1), seed=7)
    res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=t["calibration"] * 1.01)), Options())
    adv = res.advice
    it = adv["items"][0]
    hyps = adv["hypotheses"]
    sol = res.spot_result.solutions[0]
    c = sol.calib
    gx = np.array([s["dx"] * c for s in sol.indexed])
    gy = np.array([s["dy"] * c for s in sol.indexed])
    g_vis = float(np.max(np.hypot(gx, gy))) * 1.08
    R = np.array(it["rotation"])
    ax_v = np.array(it["axis_image_xy"])
    n = 1 + len(hyps)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.4))
    by = {p.name: p for p in res.candidates}
    for k, ax in enumerate(axes):
        ax.plot([-g_vis * ax_v[0], g_vis * ax_v[0]], [-g_vis * ax_v[1], g_vis * ax_v[1]], ls="--", color="#B45309", lw=1)
        ax.plot([0], [0], marker="+", color="#F59E0B", ms=10, mew=2)
        if k == 0:
            ax.scatter(gx, gy, s=14, color=INK)
            ax.set_title("Measured now\n(4 answers fit)", fontsize=9.5)
        else:
            h, p = hyps[k - 1], it["predictions"][k - 1]
            U = R @ np.array(h["orientation"])
            xy, w, _, _ = oriented_spots(by[h["phase"]], U, g_max=g_vis)
            m = np.hypot(xy[:, 0], xy[:, 1]) <= g_vis
            ax.scatter(xy[m, 0], xy[m, 1], s=8 + 90 * np.sqrt(w[m]), color=ACCENT[(k - 1) % 5])
            ax.set_title(f"If {p['phase']}\n{p['zone'] or 'off-axis'} after {it['tilt_deg']:.0f}° tilt", fontsize=9.5)
        ax.set_xlim(-g_vis, g_vis)
        ax.set_ylim(g_vis, -g_vis)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_advisor.png", dpi=200)
    plt.close(fig)
    return it["text"]


def fig_benchmark():
    from bench.common import TASKS
    from bench.summarize import SHORT

    splits = [("v0", "standard", "#A3B18A"), ("v0-hard", "hard (used to fix the pipeline)", "#6B8E23"),
              ("v0-hard-heldout", "hard, held out (new seed)", GREEN)]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    tasks = list(TASKS)
    width = 0.26
    for k, (sp, label, col) in enumerate(splits):
        f = ROOT / "bench" / "data" / sp / "results_saedscope.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        acc = []
        for tsk in tasks:
            r = [x["correct"] for x in rows if x["task"] == tsk]
            acc.append(100 * sum(r) / len(r) if r else np.nan)
        tot = 100 * sum(x["correct"] for x in rows) / len(rows)
        x = np.arange(len(tasks)) + (k - 1) * width
        ax.bar(x, np.nan_to_num(acc), width, color=col, label=f"{label}: {tot:.0f}% overall")
        for xi, a in zip(x, acc):
            if not np.isnan(a):
                ax.text(xi, a + 1.2, f"{a:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(tasks)), [SHORT[t] for t in tasks], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title("SAED-Bench: SAEDscope baseline, 40 items per task (spot phase exists only in the hard splits)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_benchmark.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["pipeline", "examples", "advisor", "benchmark"]
    for w in which:
        r = globals()[f"fig_{w}"]()
        print(w, "done", r or "")
