"""Fast sanity tests on simulated patterns with known answers.  Run: pytest -q"""

import numpy as np
import pytest

from saedscope import synth
from saedscope.crystal import builtin_library, canonical_uvw, ring_lines
from saedscope.io import Calibration, Pattern
from saedscope.physics import electron_wavelength_A
from saedscope.pipeline import Options, analyse

LIB = {p.name: p for p in builtin_library()}


def test_wavelength_200kV():
    assert electron_wavelength_A(200) == pytest.approx(0.025079, rel=1e-3)


def test_gold_ring_dspacings():
    lines = ring_lines(LIB["Au"], 0.6)
    assert lines[0].label == "111" and lines[0].d == pytest.approx(2.3546, rel=1e-3)
    assert lines[1].label == "200"


def test_structures_have_right_formulae():
    assert LIB["Cu₂O (cuprite)"].formula == "Cu2O"
    assert len(LIB["Fe₃O₄ (magnetite)"].structure) == 56
    assert len(LIB["TiO₂ (anatase)"].structure) == 12


def _pat(img, truth):
    return Pattern(img, "t", Calibration(inv_A_per_px=truth["calibration"], source="test"))


def test_ring_pattern_identified():
    img, truth = synth.ring_pattern(LIB["NiAl (B2)"], seed=11, size=640)
    res = analyse(_pat(img, truth), Options(elements=["Ni", "Al", "O"]))
    assert res.classification.kind == "polycrystalline"
    assert res.ring_result.primary.phase.name == "NiAl (B2)"


def test_spot_pattern_zone_axis():
    img, truth = synth.spot_pattern(LIB["Si"], (0, 1, 1), seed=12, size=640)
    res = analyse(_pat(img, truth), Options(elements=["Si"]))
    assert res.classification.kind == "single crystal"
    sol = res.spot_result.solutions[0]
    assert sol.phase.name == "Si"
    assert sol.uvw_family == canonical_uvw(LIB["Si"], np.array((0, 1, 1)))


def test_amorphous_detected():
    img, truth = synth.amorphous_pattern(seed=13, size=640, d_halo=2.4)
    res = analyse(_pat(img, truth), Options())
    assert res.classification.kind == "amorphous"
    assert res.amorphous["halo_d_A"] == pytest.approx(2.4, rel=0.04)


def test_calibration_from_gold_standard_and_ellipticity():
    from saedscope.calibrate import calibrate_from_standard

    img, t = synth.ring_pattern(LIB["Au"], seed=11, ellipticity=(1.2, 30.0))
    std = calibrate_from_standard(img, "Au")
    assert std.calib_inv_A_per_px == pytest.approx(t["calibration"], rel=0.003)
    assert std.ellipticity_pct == pytest.approx(1.2, abs=0.3)


def test_scale_bar_calibration():
    from saedscope.calibrate import calib_from_scale_bar, detect_scale_bars

    img, t = synth.ring_pattern(LIB["Au"], seed=12, scale_bar=True)
    bar = detect_scale_bars(img)[0]
    c = calib_from_scale_bar(bar.length_px, t["scale_bar"]["value"], "1/nm")
    assert c == pytest.approx(t["calibration"], rel=0.01)


def test_advisor_separates_hexagonal_lookalikes():
    img, t = synth.spot_pattern(LIB["α-Ti"], (0, 0, 1), seed=7)
    res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=t["calibration"] * 1.01)), Options())
    adv = res.advice
    assert adv["kind"] == "tilt" and len(adv["hypotheses"]) >= 2
    best = adv["items"][0]
    assert best["tilt_deg"] <= 35 and best["separates"].split("/")[0] == best["separates"].split("/")[1]
    assert "next_step" in res.summary()


def test_annotation_strip_removed():
    from saedscope.io import strip_annotation_panel

    rng = np.random.default_rng(0)
    img = rng.poisson(20, (300, 256)).astype(float)
    img[256:, :] = 0.0
    img[270:274, 40:120] = 255.0  # a scale bar in the strip
    out, n = strip_annotation_panel(img)
    assert n == 44 and out.shape == (256, 256)


def test_calibration_conflict_reported():
    img, t = synth.spot_pattern(LIB["Cu"], (0, 0, 1), seed=21)
    # tell it the sample is Au: Cu [001] fits Au [001] geometrically but needs a ~11% different calibration
    res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=t["calibration"])), Options(elements=["Au"], advise=False))
    c = res.calibration_conflict
    assert c is not None and c["phase"] == "Au" and abs(c["calibration_off_by_pct"]) > 8
    assert any(a["phase"] == "Cu" for a in c["fits_at_given_calibration"])


def test_cif_check_verdicts():
    from pymatgen.io.cif import CifWriter

    from saedscope.cifcheck import check_cifs
    from saedscope.crystal import phase_from_cif

    def cif(name, scale=1.0):
        st = LIB[name].structure.copy()
        st.scale_lattice(st.volume * scale**3)
        return phase_from_cif(str(CifWriter(st)), f"{name} x{scale}")

    img, t = synth.ring_pattern(LIB["NiAl (B2)"], seed=3)
    cifs = [cif("NiAl (B2)"), cif("NiAl (B2)", 1.03), cif("Au")]
    res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=t["calibration"])), Options(use_builtin=False, user_phases=cifs, advise=False))
    v = {c.phase.name: c for c in check_cifs(res, cifs)}
    assert v["NiAl (B2) x1.0"].verdict == "consistent"
    assert v["NiAl (B2) x1.03"].verdict == "lattice change" and v["NiAl (B2) x1.03"].lattice_change_pct == pytest.approx(-2.9, abs=0.5)
    assert v["Au x1.0"].verdict in ("not consistent", "partly")


def test_single_systematic_row():
    # a crystal off the zone axis: only g, 2g, -g, -2g along one line; spacing chosen as Au (200)
    rng = np.random.default_rng(1)
    n = 512
    yy, xx = np.indices((n, n), dtype=float)
    cy, cx = 251.3, 258.7
    img = 4000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2)) + 40 / (1 + (np.hypot(yy - cy, xx - cx) / 20) ** 2) + 2
    u = np.array([np.cos(0.4), np.sin(0.4)])
    calib = 0.004
    g_px = (1 / 2.0391) / calib
    for k, a in ((1, 300), (-1, 280), (2, 120), (-2, 110)):
        px, py = cx + k * g_px * u[0], cy + k * g_px * u[1]
        img += a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * 1.8**2))
    img = rng.poisson(img).astype(float)
    res = analyse(Pattern(img, "row", Calibration(inv_A_per_px=calib)), Options(elements=["Au"], advise=True))
    s = res.summary()
    assert res.classification.kind == "single crystal" and "systematic_row" in s
    assert s["systematic_row"]["d_A"] == pytest.approx(2.0391, rel=0.01)
    assert s["systematic_row"]["matches"][0]["hkl"] == "200"
    assert res.advice["kind"] == "row"


def test_texture_zone():
    # Au rings 200, 220, 400 seen but 111 and 311 missing: all seen planes contain [001], the missing ones do not
    from saedscope.cifcheck import _texture_zone

    a = 4.0782
    d = lambda h, k, l: a / np.sqrt(h * h + k * k + l * l)
    assert _texture_zone(LIB["Au"], [d(2, 0, 0), d(2, 2, 0), d(4, 0, 0)], [d(1, 1, 1), d(3, 1, 1)]) == "[001]"
    assert _texture_zone(LIB["Au"], [d(1, 1, 1), d(2, 0, 0), d(2, 2, 0)], [d(3, 1, 1)]) is None


def test_small_detector_rings():
    # a 256 px detector: Au rings a few pixels wide (relative width above the usual 'sharp' limit), 111/200 nearly merged
    img, t = synth.ring_pattern(LIB["Au"], size=256, calib=0.0100, centre=(128.3, 127.6), seed=5, ring_width_px=2.0,
                                beam_stop=False, counts=50.0)
    res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=0.0100)), Options(elements=["Au"], advise=False))
    assert res.classification.kind == "polycrystalline"
    assert res.ring_result.primary is not None and res.ring_result.primary.phase.name == "Au"


def test_hidden_direct_beam_off_centre():
    # pattern centre far from the image centre, direct beam under the beam stop (as in many cropped paper figures):
    # the spot lattice alone is symmetric about every lattice and half-lattice point, the glow and the stop pick the right one
    for seed in (2, 5):
        img, t = synth.spot_pattern(LIB["BaTiO₃ (perovskite)"], (0, 0, 1), size=400, centre=(252.3, 281.7), seed=seed, calib=0.0042)
        res = analyse(Pattern(img, "x", Calibration(inv_A_per_px=0.0042)), Options(elements=["Ba", "Ti", "O"], advise=False))
        assert np.hypot(res.centre[0] - 252.3, res.centre[1] - 281.7) < 2
        best = res.spot_result.solutions[0]
        assert best.phase.name == "BaTiO₃ (perovskite)" and canonical_uvw(best.phase, np.array([0, 0, 1])) == best.uvw_family


def test_builtin_count():
    from saedscope.crystal import N_BUILTIN

    assert len(builtin_library()) == N_BUILTIN
