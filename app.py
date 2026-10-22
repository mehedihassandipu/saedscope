"""SAEDscope: identify the phase and zone axis in an electron diffraction pattern.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import streamlit as st

from saedscope import report, synth, viz
from saedscope.crystal import N_BUILTIN, builtin_library, canonical_uvw, fmt_index, phase_from_cif
from saedscope.io import FOUR_D_EXT, Calibration, Pattern, load_pattern
from saedscope.llm import explain, merge_elements
from saedscope.physics import calibration_from_camera_length
from saedscope.pipeline import Options, analyse
from saedscope.calibrate import STANDARDS, calib_from_scale_bar, calibrate_from_standard, detect_scale_bars, scale_bar_mask
from saedscope.batch import analyse_many
from saedscope.cifcheck import check_cifs

st.set_page_config(page_title="SAEDscope", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
.block-container { padding-top: 3.2rem; max-width: 1500px; }
.app-title { font-size: 1.15rem !important; font-weight: 700; color: #1F2A37; margin: 0; letter-spacing: 0.01em; }
.app-sub { color: #4B5563; margin: 0.1rem 0 1.1rem 0; font-size: 0.95rem; }
div.verdict { font-size: 2.1rem !important; line-height: 1.2; font-weight: 700; color: #111827; margin: 0.1rem 0 0.4rem 0; max-width: 40ch; }
.verdict .kind { color: #3F6212; }
div.evidence { color: #374151; font-size: 1.05rem !important; margin-bottom: 0.9rem; font-variant-numeric: tabular-nums; max-width: 95ch; }
.flag { display: inline-block; background: #FEF3C7; color: #78350F; border-radius: 4px; padding: 0.15rem 0.5rem; margin: 0 0.4rem 0.4rem 0; font-size: 0.88rem; }
.ok { display: inline-block; background: #ECFCCB; color: #365314; border-radius: 4px; padding: 0.15rem 0.5rem; margin: 0 0.4rem 0.4rem 0; font-size: 0.88rem; }
.truth { color: #1D4ED8; font-size: 0.92rem; }
.empty h3 { margin-top: 0.4rem; }
div[data-testid="stSidebar"] h3 { font-size: 0.98rem; margin-top: 0.6rem; }
.step { display: flex; align-items: center; gap: 0.5rem; margin: 1.0rem 0 0.35rem 0; font-weight: 700; font-size: 0.98rem; color: #1F2A37; }
.step .num { display: inline-flex; width: 1.45rem; height: 1.45rem; border-radius: 50%; align-items: center; justify-content: center;
             font-size: 0.8rem; background: #E5E7EB; color: #374151; }
.step.done .num { background: #3F6212; color: white; }
.step .st { font-weight: 400; font-size: 0.8rem; margin-left: auto; color: #6B7280; }
.step.done .st { color: #3F6212; } .step.todo .st { color: #B45309; } .step.need .st { color: #B91C1C; }
.check { display: grid; grid-template-columns: 1.6rem 1fr; gap: 0.15rem 0.4rem; margin: 0.35rem 0; line-height: 1.45; }
.check .ic { font-weight: 700; text-align: center; } .check.done .ic { color: #3F6212; } .check.todo .ic { color: #B45309; } .check.need .ic { color: #B91C1C; }
.check .tx b { color: #111827; } .check .tx span { color: #4B5563; font-size: 0.93rem; }
.guide { background: #F3F5F2; border-left: 4px solid #3F6212; padding: 0.6rem 0.9rem; border-radius: 4px; margin: 0.2rem 0 0.9rem 0; max-width: 110ch; color: #1F2A37; }
.guide b { color: #3F6212; }
</style>
""",
    unsafe_allow_html=True,
)

LIB = {p.name: p for p in builtin_library()}

EXAMPLES = {
    "Polycrystalline gold rings (calibration standard)": ("ring", "Au", {}, "Au"),
    "NiAl alloy rings": ("ring", "NiAl (B2)", {}, "Ni, Al, O"),
    "Anatase TiO₂, coarse grains (spotty rings)": ("ring", "TiO₂ (anatase)", {"spotty": 0.85}, "Ti, O"),
    "Silicon single crystal, [011] zone": ("spot", "Si", {"uvw": (0, 1, 1)}, "Si"),
    "Magnetite single crystal, [112] zone": ("spot", "Fe₃O₄ (magnetite)", {"uvw": (1, 1, 2)}, "Fe, O"),
    "ZnO single crystal, [110] zone": ("spot", "ZnO (wurtzite)", {"uvw": (1, 1, 0)}, "Zn, O"),
    "Amorphous film (metallic glass)": ("amorphous", None, {"d_halo": 2.35}, "Zr, Cu"),
    "Hexagonal spot pattern, sample unknown (look-alikes)": ("spot", "α-Ti", {"uvw": (0, 0, 1)}, ""),
}


@st.cache_data(show_spinner=False)
def make_example(label: str):
    kind, phase, kw, _ = EXAMPLES[label]
    kw = dict(kw)
    seed = int(hashlib.md5(label.encode()).hexdigest(), 16) % 1000  # stable across restarts
    if kind == "ring":
        img, truth = synth.ring_pattern(LIB[phase], seed=seed, size=900, **kw)
    elif kind == "spot":
        uvw = kw.pop("uvw") if "uvw" in kw else (0, 0, 1)
        img, truth = synth.spot_pattern(LIB[phase], uvw, seed=seed, size=900, **kw)
    else:
        img, truth = synth.amorphous_pattern(seed=seed, size=900, **kw)
    return img.astype(np.float32), truth


FOUR_D_CHOICES = {"All grains overlaid (max), recommended": "max", "Average of the scan (mean)": "mean",
                  "Single strongest scan position": "brightest"}


@st.cache_data(show_spinner=False, max_entries=6)
def load_4d_cached(file_id: str, fname: str, mode: str, _data: bytes):
    return load_pattern(_data, fname, mode=mode)


@st.cache_data(show_spinner=False)
def fit_standard(data: bytes, fname: str, material: str, volt, pix):
    pat = load_pattern(data, fname)
    cal = calibrate_from_standard(pat.image, material, voltage_kV=volt, pixel_size_um=pix)
    meta = {"det": pat.metadata.get("detector"), "shape": tuple(pat.image.shape), "file_cal": pat.calibration.inv_A_per_px,
            "L": pat.metadata.get("camera_length_mm")}
    return cal, meta


def camera_mismatch(std_cal, std_meta, file_meta):
    """Reason to think the standard was not recorded on the same camera / camera length / binning as the pattern."""
    if std_cal is None or not file_meta:
        return None
    if std_meta.get("det") and file_meta.get("det") and std_meta["det"] != file_meta["det"]:
        return (f'The standard was recorded on {std_meta["det"]}, this pattern on {file_meta["det"]}. A camera constant does '
                'not transfer between cameras.')
    if file_meta.get("cal"):
        dev = (std_cal.calib_inv_A_per_px / file_meta["cal"] - 1) * 100
        if abs(dev) > 10:
            where = ", ".join(x for x in (file_meta.get("det"), f'{file_meta["L"]:.0f} mm' if file_meta.get("L") else None) if x)
            return (f'The standard gives {std_cal.calib_inv_A_per_px:.5f} Å⁻¹/px, but this file records '
                    f'{file_meta["cal"]:.5f} Å⁻¹/px{f" ({where})" if where else ""}: {abs(dev):.0f}% apart, much more than a '
                    'nominal camera length is usually off. The standard was probably recorded on another camera, camera length '
                    'or binning. Use a standard from the same camera, or choose "From the file / example".')
    if std_meta.get("shape") and file_meta.get("shape") and std_meta["shape"] != file_meta["shape"]:
        return (f'The standard is {std_meta["shape"][1]} × {std_meta["shape"][0]} px, this pattern {file_meta["shape"][1]} × '
                f'{file_meta["shape"][0]} px: different binning or camera, so the camera constant may not transfer.')
    return None


def run_analysis(img, name, cal_kwargs, opts_kwargs, cif_texts):
    user_phases = []
    notes = []
    for fname, text in cif_texts:
        try:
            user_phases.append(phase_from_cif(text, name=os.path.splitext(fname)[0]))
        except Exception as exc:
            notes.append(f"Could not read {fname}: {exc}")
    pat = Pattern(image=np.asarray(img, dtype=np.float64), name=name, calibration=Calibration(**cal_kwargs))
    res = analyse(pat, Options(user_phases=user_phases, **opts_kwargs))
    res.notes = notes + res.notes
    return res


# ----------------------------------------------------------------------------
# Sidebar: inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    goal = st.radio("What do you want to do?", ["Identify the phase", "Check against my CIF files"], key="goal",
                    help="Identify: search the built-in library (and Materials Project) for the phase. "
                         "Check: test whether your pattern matches the structure(s) in your own CIF files.")
    check_mode = goal == "Check against my CIF files"
    h_pattern = st.empty()
    qp = st.query_params
    qp_example = None
    if "example" in qp:
        try:
            qp_example = list(EXAMPLES)[int(qp["example"])]
        except (ValueError, IndexError):
            qp_example = None
    source = st.radio("Source", ["Upload a file", "Try an example", "Several files"], horizontal=True, label_visibility="collapsed",
                      index=1 if qp_example else 0)
    upload = None
    example = None
    batch_files = None
    four_d_mode = None
    if source == "Several files":
        batch_files = st.file_uploader("Diffraction patterns", type=["dm3", "dm4", "tif", "tiff", "png", "jpg", "jpeg", "npy"],
                                       accept_multiple_files=True, help="Each file is analysed with the settings below; you get one table.")
    elif source == "Upload a file":
        upload = st.file_uploader("Diffraction pattern", type=["dm3", "dm4", "tif", "tiff", "png", "jpg", "jpeg", "npy",
                                                               "zspy", "hspy", "hdf5", "h5", "mib"],
                                  help="Original .dm3/.dm4 files keep their calibration. Screenshots and paper figures also work (uncalibrated). "
                                       "4D-STEM scans (.zspy, .hspy, .hdf5, .mib) are turned into one pattern; you choose how below.")
        if upload is not None and os.path.splitext(upload.name)[1].lower() in FOUR_D_EXT:
            four_d_mode = st.radio("This is a 4D-STEM scan. Which pattern should be analysed?", list(FOUR_D_CHOICES), key="four_d_mode",
                                   help="Max: every grain's spots on one pattern, like a ring pattern (best for nanocrystals). "
                                        "Mean: the average, smoother but weaker spots. Brightest: one scan position, usually one grain.")
    else:
        example = st.selectbox("Example pattern (simulated, answer known)", list(EXAMPLES),
                               index=list(EXAMPLES).index(qp_example) if qp_example else 0)

    h_sample = st.empty()
    default_els = EXAMPLES[example][3] if example else ""
    elements_text = st.text_input("Elements present", value=default_els, placeholder="e.g. Ni, Al, O (from EDS)",
                                  help="Restricts the search to phases made of these elements. Leave empty to search everything (less reliable).")
    context = st.text_area("Anything else about the sample?", placeholder="e.g. NiAl alloy annealed in air at 800 °C", height=70)

    h_cal = st.empty()
    cal_mode = st.selectbox("How is the pattern calibrated?",
                            ["From the file / example", "From a standard pattern", "From a scale bar in the image", "nm⁻¹ per pixel",
                             "Å⁻¹ per pixel", "Voltage and camera length", "Not calibrated (ratios only)"], key="cal_mode",
                            help="Best: a ring pattern of a standard (e.g. evaporated gold) taken at the same camera length. "
                                 "Scale bars and nominal camera lengths can be several % off.")
    cal_value = None
    volt = cam = pix = None
    std_cal = None
    std_meta = {}
    std_name = None
    bar_value, bar_unit, bar_len_override = 5.0, "nm⁻¹", 0.0
    if cal_mode == "From a standard pattern":
        std_file = st.file_uploader("Standard pattern (same camera length)", type=["dm3", "dm4", "tif", "tiff", "png", "jpg", "jpeg", "npy"],
                                    key="std", help="A ring pattern of a known polycrystalline standard, e.g. evaporated gold.")
        std_mat = st.selectbox("Standard material", STANDARDS)
        if std_file is not None:
            std_name = std_file.name
            try:
                std_cal, std_meta = fit_standard(std_file.getvalue(), std_file.name, std_mat, None, None)
                st.success(std_cal.summary())
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")
    elif cal_mode == "From a scale bar in the image":
        c1, c2 = st.columns([1, 1])
        bar_value = c1.number_input("Bar value", value=5.0, min_value=0.0, step=0.5, help="The number printed next to the bar.")
        bar_unit = c2.selectbox("Unit", ["nm⁻¹", "Å⁻¹"])
        bar_len_override = st.number_input("Bar length in pixels (0 = detect automatically)", value=0.0, min_value=0.0, step=1.0)
    elif cal_mode in ("nm⁻¹ per pixel", "Å⁻¹ per pixel"):
        cal_value = st.number_input(cal_mode, min_value=0.0, value=0.0, format="%.6f")
    elif cal_mode == "Voltage and camera length":
        c1, c2 = st.columns(2)
        volt = c1.number_input("Voltage (kV)", value=200.0, step=10.0)
        cam = c2.number_input("Camera length (mm)", value=400.0, step=10.0)
        pix = st.number_input("Detector pixel size incl. binning (µm)", value=14.0, step=1.0)

    h_cand = st.empty()
    if check_mode:
        cif_files = st.file_uploader("CIF files to check", type=["cif"], accept_multiple_files=True, key="cifs",
                                     help="One or more structures. Each is compared with the pattern; several are ranked.")
        use_builtin, mp_key = False, ""
    else:
        use_builtin = st.checkbox(f"Built-in library ({N_BUILTIN} common phases)", value=True)
        mp_key = st.text_input("Materials Project API key (optional)", type="password", value=os.environ.get("MP_API_KEY", ""),
                               help="Searches every known phase in your elements' chemical systems. Free key at materialsproject.org.")
        cif_files = st.file_uploader("Your own CIF files (optional)", type=["cif"], accept_multiple_files=True, key="cifs",
                                     help="Added to the candidates; you also get a CIF check for each.")

    with st.expander("AI explanation"):
        provider = st.selectbox("Language model", ["Built-in explanation (no key needed)", "Anthropic", "OpenAI-compatible"])
        prov = {"Anthropic": "anthropic", "OpenAI-compatible": "openai"}.get(provider, "none")
        model = api_key = base_url = None
        if prov == "anthropic":
            model = st.text_input("Model", value=os.environ.get("SAED_ANTHROPIC_MODEL", "claude-sonnet-5-5"))
            api_key = st.text_input("API key", type="password",  # never pre-filled: the server's key must not reach the browser
                                    placeholder="using the server's key" if os.environ.get("ANTHROPIC_API_KEY") else "sk-ant-…")
        elif prov == "openai":
            model = st.text_input("Model", value=os.environ.get("SAED_OPENAI_MODEL", ""), placeholder="model name")
            base_url = st.text_input("Base URL (blank for OpenAI)", value=os.environ.get("OPENAI_BASE_URL", ""),
                                     help="Any OpenAI-compatible endpoint, e.g. a local Ollama or vLLM server for open models.")
            api_key = st.text_input("API key", type="password",
                                    placeholder="using the server's key" if os.environ.get("OPENAI_API_KEY") else "key (blank for local servers)")

    with st.expander("Advanced"):
        ring_tol = st.slider("Ring d-spacing tolerance (%)", 0.5, 4.0, 1.5, 0.1) / 100
        spot_tol = st.slider("Spot d-spacing tolerance (%)", 1.0, 6.0, 3.0, 0.5) / 100
        ang_tol = st.slider("Spot angle tolerance (°)", 0.5, 6.0, 2.5, 0.5)
        window = st.slider("Allowed uniform scale error (%)", 0.0, 8.0, 3.5, 0.5) / 100
        force = st.selectbox("Pattern type", ["Detect automatically", "single crystal", "polycrystalline", "amorphous"])
        overlay = st.checkbox("Show overlay on the pattern", value=True)
        auto_bs = st.checkbox("Detect and ignore the beam stop automatically", value=True)
        use_ell = st.checkbox("Correct ring ellipticity measured on the standard", value=False,
                              help="Projector-lens distortion makes rings slightly elliptical. The distortion measured on the standard "
                                   "only transfers if the sample was recorded in the same session and at the same camera length; "
                                   "turn it on in that case and keep it only if the fit improves.")
        manual_c = st.checkbox("Set the pattern centre myself")
        cx_in = cy_in = None
        if manual_c:
            c1, c2 = st.columns(2)
            prev = st.session_state.get("res")
            cx0 = float(prev.centre[1]) if prev is not None else 0.0
            cy0 = float(prev.centre[0]) if prev is not None else 0.0
            cx_in = c1.number_input("Centre x (px)", value=cx0, step=0.5)
            cy_in = c2.number_input("Centre y (px)", value=cy0, step=0.5)
        rmin_in = st.number_input("Ignore everything inside this radius (px, 0 = automatic)", value=0.0, min_value=0.0, step=1.0)

    run = st.button("Analyse pattern", type="primary", width="stretch")
    run = run or bool(st.session_state.get("run_main", False))

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown('<p class="app-title">SAEDscope</p><p class="app-sub">Identify the phase and zone axis in an electron diffraction pattern. '
            "Physics does the matching; the AI explains the evidence.</p>", unsafe_allow_html=True)

img = name = truth = None
file_meta = None  # calibration, detector and size recorded in the uploaded file
bar_source = None  # image to search for a burnt-in scale bar (may include an exported annotation strip)
cal_kwargs = {}
if example:
    img, truth = make_example(example)
    name = example
    cal_kwargs = {"inv_A_per_px": float(truth["calibration"]), "voltage_kV": 200.0, "source": "file metadata"}
elif upload is not None:
    data = upload.getvalue()
    try:
        if four_d_mode:
            with st.spinner("Reading the 4D-STEM scan (can take a minute for a large file)..."):
                pat = load_4d_cached(getattr(upload, "file_id", upload.name), upload.name, FOUR_D_CHOICES[four_d_mode], data)
        else:
            pat = load_pattern(data, upload.name)
        img, name = pat.image, upload.name
        bar_source = pat.metadata.get("image_with_annotation")
        file_meta = {"cal": pat.calibration.inv_A_per_px, "det": pat.metadata.get("detector"),
                     "L": pat.metadata.get("camera_length_mm"), "shape": tuple(pat.image.shape)}
        cal_kwargs = dict(pat.calibration.__dict__)
        for n in pat.notes:
            st.caption(n)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}. Supported: .dm3, .dm4, .tif, .png, .jpg, .npy, and 4D-STEM .zspy/.hspy/.hdf5/.mib.")

if cal_mode == "nm⁻¹ per pixel" and cal_value:
    cal_kwargs.update({"inv_A_per_px": cal_value / 10, "source": "user"})
elif cal_mode == "Å⁻¹ per pixel" and cal_value:
    cal_kwargs.update({"inv_A_per_px": cal_value, "source": "user"})
elif cal_mode == "Voltage and camera length" and volt and cam and pix:
    cal_kwargs.update({"inv_A_per_px": calibration_from_camera_length(volt, cam, pix), "voltage_kV": volt, "camera_length_mm": cam, "source": "camera length"})
elif cal_mode == "Not calibrated (ratios only)":
    cal_kwargs.update({"inv_A_per_px": None, "source": "none"})
elif cal_mode == "From a standard pattern" and std_cal is not None:
    cal_kwargs.update({"inv_A_per_px": std_cal.calib_inv_A_per_px, "source": "standard"})

bar_mask = None
bar_note = None
if cal_mode == "From a scale bar in the image" and img is not None:
    in_strip = bar_source is not None
    bars = detect_scale_bars(bar_source if in_strip else img)
    length = bar_len_override or (bars[0].length_px if bars else 0)
    if length and bar_value:
        try:
            cal_kwargs.update({"inv_A_per_px": calib_from_scale_bar(length, bar_value, bar_unit), "source": "scale bar"})
            bar_note = f"Scale bar: {length:.0f} px = {bar_value:g} {bar_unit}" + (" (detected automatically)" if not bar_len_override else "")
            if bars and not bar_len_override and not in_strip:  # a bar in the removed strip needs no mask
                bar_mask = scale_bar_mask(np.asarray(img).shape[:2], bars[0])
        except ValueError as exc:
            st.error(str(exc))
    else:
        bar_note = "No scale bar found automatically: enter its length in pixels in the sidebar."
ellipticity = (std_cal.ellipticity_pct, std_cal.ellipticity_axis_deg) if (std_cal is not None and use_ell and std_cal.ellipticity_pct >= 0.3) else None


def _step(ph, n, title, state, note):
    label = {"done": "✓ " + note, "todo": "⚠ " + note, "need": "✗ " + note, "opt": note}[state]
    ph.markdown(f'<div class="step {state}"><span class="num">{n}</span>{title}<span class="st">{label}</span></div>', unsafe_allow_html=True)


def _set_cal_mode(mode):
    st.session_state["cal_mode"] = mode


have_pattern = img is not None or bool(batch_files)
have_elements = bool(elements_text.strip()) or bool(context.strip())
calibrated_now = bool(cal_kwargs.get("inv_A_per_px"))
n_cifs = len(cif_files or [])
_step(h_pattern, 1, "Pattern", "done" if have_pattern else "need", "loaded" if have_pattern else "required")
_step(h_sample, 2, "Sample", "done" if have_elements else "todo", "elements given" if have_elements else "recommended")
_step(h_cal, 3, "Calibration", "done" if calibrated_now else "todo", (cal_kwargs.get("source") or "set") if calibrated_now else "recommended")
if check_mode:
    _step(h_cand, 4, "Structures to check", "done" if n_cifs else "need", f"{n_cifs} CIF" + ("s" if n_cifs != 1 else "") if n_cifs else "required")
else:
    _step(h_cand, 4, "Candidate phases", "done", ("library" if use_builtin else "") + (f" + {n_cifs} CIF" if n_cifs else ""))

# Batch mode -------------------------------------------------------------------
if batch_files:
    st.markdown(f"#### {len(batch_files)} files ready")
    if run:
        prog = st.progress(0.0, text="Analysing…")
        cal_obj = Calibration(**cal_kwargs) if cal_kwargs.get("inv_A_per_px") else None
        bopts = Options(elements=merge_elements(elements_text, context)[0], context=context, use_builtin=use_builtin,
                        mp_api_key=mp_key or None, ellipticity=ellipticity, auto_beamstop=auto_bs)
        rows = analyse_many([(f.name, f.getvalue()) for f in batch_files], bopts, cal_obj,
                            progress=lambda k, n, r: prog.progress(k / n, text=f"{k}/{n}: {r['file']}"))
        import pandas as pd
        df = pd.DataFrame(rows).fillna("").astype(str)
        st.session_state["batch_df"] = df
    df = st.session_state.get("batch_df")
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download table (CSV)", report.csv_bytes(df), file_name="saedscope_batch.csv", mime="text/csv")
    else:
        st.caption("Press Analyse pattern to run all files with the current settings.")
    st.stop()

elements, ctx_notes = merge_elements(elements_text, context)

if img is None:
    st.markdown(
        '<div class="empty"><h3>Add a diffraction pattern to start</h3>'
        "<p>Upload a .dm3/.dm4 file (best: calibration is read automatically), a TIFF, or a PNG/JPG cropped from a paper. "
        "Or pick <b>Try an example</b> in the sidebar to see a simulated pattern whose answer is known.</p>"
        "<p>SAEDscope works out whether the pattern is from a single crystal, a polycrystal or an amorphous material, "
        "then finds the phase and zone axis that best explain the spots or rings, and shows you the evidence.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

key = hashlib.md5(np.ascontiguousarray(img[::7, ::7]).tobytes()).hexdigest() + str(
    (sorted(cal_kwargs.items(), key=lambda kv: kv[0]), elements, context, use_builtin, bool(mp_key), ring_tol, spot_tol, ang_tol, window, force,
     [f.name for f in cif_files or []], auto_bs, ellipticity, cx_in, cy_in, rmin_in, bar_mask is not None))

auto_run = "example" in st.query_params and "run" in st.query_params
if run or auto_run and st.session_state.get("key") != key or st.session_state.get("key") != key and st.session_state.get("auto", False):
    with st.spinner("Finding the centre, classifying the pattern and matching candidate phases…"):
        opts = dict(elements=elements, context=context, use_builtin=use_builtin, mp_api_key=mp_key or None, ring_tol=ring_tol,
                    spot_tol_d=spot_tol, spot_tol_angle=ang_tol, scale_window=window,
                    force_kind=None if force == "Detect automatically" else force, auto_beamstop=auto_bs,
                    ellipticity=ellipticity, extra_mask=bar_mask,
                    centre=(cy_in, cx_in) if (manual_c and cx_in) else None, r_min=rmin_in or None)
        cifs = [(f.name, f.getvalue().decode("utf-8", errors="ignore")) for f in cif_files or []]
        res = run_analysis(img, name, cal_kwargs, opts, cifs)
        checks = check_cifs(res, res.options.user_phases) if res.options.user_phases else []
        if check_mode and checks and res.spot_result and res.spot_result.solutions:
            top_name = checks[0].phase.name  # show the structure the check ranks first in the overlay and zone view
            res.spot_result.solutions.sort(key=lambda z: z.phase.name != top_name)
        expl, expl_src = explain(res.summary(), prov, model or "", api_key or None, base_url or None)
        st.session_state.update({"res": res, "key": key, "expl": expl, "expl_src": expl_src, "auto": True, "checks": checks})

res = st.session_state.get("res")
if res is None or st.session_state.get("key") != key:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        im, _ = viz.display_image(np.asarray(img, dtype=np.float64))
        st.image(im, caption=f"{name}: press Analyse pattern", width="stretch")
    with c2:
        st.markdown("#### Before you analyse")
        items = []

        def item(state, title, text):
            ic = {"done": "✓", "todo": "!", "need": "✗"}[state]
            items.append(f'<div class="check {state}"><div class="ic">{ic}</div><div class="tx"><b>{title}</b><br><span>{text}</span></div></div>')

        shp = np.asarray(img).shape
        item("done", "Pattern", f"{name} ({shp[1]} × {shp[0]} px)")
        if elements:
            item("done", "Elements", ", ".join(elements) + (f" (from your description: {'; '.join(ctx_notes)})" if ctx_notes else ""))
        else:
            item("todo", "Elements: none given",
                 "Phases with the same structure (Au/Ag/Al, Cu/Ni/austenite) cannot be told apart without them. Type the elements from EDS in step 2.")
        found_bar = None
        if not calibrated_now and cal_mode != "From a scale bar in the image":
            bars_now = detect_scale_bars(bar_source if bar_source is not None else img)
            found_bar = bars_now[0] if bars_now else None
        if calibrated_now:
            src = cal_kwargs.get("source")
            txt = f"{cal_kwargs['inv_A_per_px']:.6f} Å⁻¹/px ({src})"
            if src == "scale bar":
                item("todo", "Calibration from the scale bar", txt + ". " + (bar_note or "") +
                     ". Scale bars can be several % off (on our microscope about 10%); if nothing fits, SAEDscope will say so. "
                     "A gold ring pattern at the same camera length is more reliable.")
            elif src == "standard" and camera_mismatch(std_cal, std_meta, file_meta):
                item("todo", "Calibration: the standard may be from another camera", camera_mismatch(std_cal, std_meta, file_meta))
            elif src == "standard" and std_name and upload is not None and std_name == upload.name:
                item("todo", "Calibration: the standard is the sample itself",
                     "This only checks the standard against itself. Upload the sample you want to identify as the pattern in step 1.")
            else:
                item("done", "Calibration", txt + (f". {bar_note}" if bar_note else ""))
        else:
            item("todo", "Calibration: none",
                 "Only ratios between rings or spots can be used, so phases of the same structure type cannot be separated. "
                 + ("A scale bar was found in the image (use the button below; type its value in step 3). " if found_bar is not None else "")
                 + "Best: a gold ring pattern at the same camera length (step 3, From a standard pattern).")
        if check_mode:
            if n_cifs:
                item("done", "Structures to check", ", ".join(f.name for f in cif_files))
            else:
                item("need", "Structures to check: none", "Upload at least one CIF file in step 4.")
        else:
            item("done", "Candidates", (f"built-in library ({N_BUILTIN} phases)" if use_builtin else "no built-in library")
                 + (f" + {n_cifs} CIF file(s), each also checked separately" if n_cifs else "") + (" + Materials Project" if mp_key else ""))
        if ellipticity:
            item("done", "Ellipticity correction", f"{ellipticity[0]:.2f}% at {ellipticity[1]:.0f}° from the standard")
        st.markdown("".join(items), unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if found_bar is not None:
            b1.button(f"Use the scale bar ({found_bar.length_px:.0f} px)", on_click=_set_cal_mode, args=("From a scale bar in the image",),
                      width="stretch")
        if not calibrated_now:
            b2.button("Calibrate with a standard", on_click=_set_cal_mode, args=("From a standard pattern",), width="stretch")
        st.button("Analyse pattern", type="primary", key="run_main", width="stretch", disabled=check_mode and not n_cifs)
        if check_mode and not n_cifs:
            st.caption("Add a CIF in step 4 to enable the check.")
    st.stop()

s = res.summary()
kind = res.classification.kind

# Verdict ---------------------------------------------------------------------
conf = s.get("calibration_conflict")
if conf and kind in ("single crystal", "mixed"):
    alts = conf.get("fits_at_given_calibration") or []
    verdict = f'<span class="kind">{kind.capitalize()}</span>, but no candidate fits at this calibration'
    evidence = (f'{conf["phase"]} {conf["zone_axis"]} matches the spot spacings and angles exactly, but only if the calibration is '
                f'{conf["calibration_off_by_pct"]:+.1f}% off. Check the scale bar or camera length'
                + (f'; at the given calibration the spots fit {", ".join(a["phase"] + " " + a["zone_axis"] for a in alts[:3])}.' if alts else '.'))
elif s.get("calibration_warning") and kind in ("single crystal", "mixed") and not (s.get("spots", {}).get("best") or {}).get("score", 0) >= 0.4:
    w = s["calibration_warning"]
    verdict = f'<span class="kind">{kind.capitalize()}</span>, but nothing fits at this calibration'
    evidence = (f'{w["phase"]} {w["zone_axis"]} matches the spot spacings and angles exactly, but only if the calibration were '
                f'{w["calibration_off_by_pct"]:+.0f}% different. That is more than a scale bar or camera-length error: check that the '
                'calibration was measured on this camera, at this camera length and binning (a standard from another camera does not transfer).')
elif s.get("systematic_row") and kind == "single crystal":
    rw = s["systematic_row"]
    ms = rw.get("matches") or []
    if rw.get("d_A"):
        verdict = f'<span class="kind">Single crystal</span>, one systematic row: d = {rw["d_A"]:.3f} Å'
        evidence = ((f'Matches {", ".join(m["phase"] + " (" + m["hkl"] + ", " + format(m["error_pct"], "+.1f") + "%)" for m in ms[:4])}. '
                     if ms else "No candidate has a reflection at this spacing. ")
                    + "A single row gives one d-spacing but no zone axis: tilt about this row (keep it excited) until a second row appears, then analyse again.")
    else:
        verdict = '<span class="kind">Single crystal</span>, one systematic row'
        evidence = f'Row spacing {rw["spacing_px"]:.1f} px; calibrate to convert it to a d-spacing. Tilt about this row to reach a zone axis.'
elif s.get("spots", {}).get("best") and kind in ("single crystal", "mixed") and s["spots"]["best"]["score"] < 0.4:
    b = s["spots"]["best"]
    verdict = f'<span class="kind">{kind.capitalize()}</span>, no confident match'
    evidence = (f'The best candidate, {b["phase"]["name"]} along {b["zone_axis"]}, explains the spots poorly (score {b["score"]:.2f}). '
                'Add the elements present or a CIF, check the calibration, or record a pattern closer to a zone axis.')
elif s.get("spots", {}).get("best") and kind in ("single crystal", "mixed"):
    b = s["spots"]["best"]
    verdict = f'<span class="kind">{kind.capitalize()}</span> {b["phase"]["name"]} along {b["zone_axis"]}'
    evidence = (f'{b["g1"]["hkl"]} and {b["g2"]["hkl"]} at {b["angle_meas_deg"]:.1f}° (expected {b["angle_ref_deg"]:.1f}°), '
                f'd = {b["g1"]["d_meas_A"]:.3f} / {b["g2"]["d_meas_A"]:.3f} Å; {b["spots_explained_pct"]:.0f}% of spot intensity explained; '
                f'{b["phase"]["space_group"]}.')
    sp_look = [row["phase"] for row in s["spots"].get("ranking", []) if row.get("space_group") == b["phase"]["space_group"]
               and row.get("zone_axis") == b["zone_axis"] and isinstance(row.get("score"), (int, float)) and row["score"] >= b["score"] - 0.06]
    sp_look = list(dict.fromkeys(sp_look))
    if (not s["calibrated"] or s["spots"].get("ambiguous")) and len(sp_look) > 1:
        verdict = f'<span class="kind">{kind.capitalize()}</span>, {b["phase"]["space_group"].split(" ")[0]} structure along {b["zone_axis"]} ({", ".join(sp_look[:5])})'
        evidence = (f'{b["g1"]["hkl"]} and {b["g2"]["hkl"]} at {b["angle_meas_deg"]:.1f}°. '
                    + ('Without a calibration these phases give the same spot pattern, so they cannot be told apart: add the elements present, or calibrate with a standard.'
                       if not s["calibrated"] else 'These phases have nearly the same lattice, so the spot pattern cannot separate them: add the elements present (e.g. from EDS).'))
elif s.get("rings", {}).get("primary_phase"):
    r = s["rings"]
    p = r["primary_phase"]
    n_idx = sum(1 for a in r["assignments"] if a["phase"] == p["name"])
    errs = [abs(a["error_pct"]) for a in r["assignments"] if a["phase"] == p["name"] and a["error_pct"] is not None]
    verdict = f'<span class="kind">{kind.capitalize()}</span> {p["name"]}' + (f' with {r["secondary_phase"]["name"]}' if r.get("secondary_phase") else "")
    evidence = f'{n_idx} of {len(r["assignments"])} rings indexed' + (f", mean d-spacing error {np.mean(errs):.2f}%" if errs else "") + f'; {p["space_group"]}, {p["source"]}.'
    lookalikes = [row["phase"] for row in r.get("ranking", []) if row.get("space_group") == p["space_group"]
                  and isinstance(row.get("score"), (int, float)) and row["score"] >= (r.get("primary_score") or 0) - 0.06]
    if (not s["calibrated"] or r.get("ambiguous")) and len(lookalikes) > 1:
        verdict = f'<span class="kind">{kind.capitalize()}</span>, {p["space_group"].split(" ")[0]} structure ({", ".join(lookalikes[:5])}{", …" if len(lookalikes) > 5 else ""})'
        if not s["calibrated"]:
            evidence = (f'{n_idx} of {len(r["assignments"])} rings fit this structure. Without a calibration these phases give identical ring ratios, '
                        'so they cannot be told apart: add the elements present, or calibrate with a standard.')
        else:
            evidence = (f'{n_idx} of {len(r["assignments"])} rings fit this structure. These phases have nearly the same lattice parameter, '
                        'so ring positions cannot separate them even with a calibration: add the elements present (e.g. from EDS).')
elif kind == "amorphous":
    a = s.get("amorphous") or {}
    verdict = '<span class="kind">Amorphous</span>, no crystalline phase to index'
    evidence = (f'Diffuse halo at d ≈ {a["halo_d_A"]:.2f} Å, mean nearest-neighbour distance ≈ {a["nearest_neighbour_A_approx"]:.2f} Å (Ehrenfest).'
                if a.get("halo_d_A") else "Diffuse halo(s) only; calibrate to convert the halo radius to a distance.")
else:
    verdict = f'<span class="kind">{kind.capitalize()}</span>, no confident match'
    evidence = "Try adding elements, a CIF, or a calibration."
checks = st.session_state.get("checks") or []
if check_mode and checks:
    top = checks[0]
    verdict = f'<span class="kind">{kind.capitalize()}</span>: {top.headline}'
    evidence = top.explanation
    n_ok = sum(1 for c in checks if c.verdict == "consistent")
    if n_ok > 1:
        evidence += (f" Note: {n_ok} of your structures are consistent with this pattern "
                     f"({', '.join(c.phase.name for c in checks if c.verdict == 'consistent')}); see the CIF check tab and Next step.")
st.markdown(f'<div class="verdict">{verdict}</div><div class="evidence">{evidence}</div>', unsafe_allow_html=True)

flags = []
if not s["calibrated"]:
    flags.append(("flag", "Uncalibrated: ratios only"))
else:
    flags.append(("ok", f"Calibrated ({s['calibration_source']})"))
if s.get("calibration_conflict"):
    flags.append(("flag", f"Calibration conflict ({s['calibration_conflict']['calibration_off_by_pct']:+.0f}%)"))
if s.get("calibration_warning"):
    flags.append(("flag", f"Calibration probably from another camera ({s['calibration_warning']['calibration_off_by_pct']:+.0f}%)"))
if s.get("rings", {}).get("ambiguous") or s.get("spots", {}).get("ambiguous"):
    flags.append(("flag", "Ambiguous: a look-alike phase fits almost as well"))
dev = (s.get("rings") or {}).get("scale_deviation_pct") or ((s.get("spots") or {}).get("best") or {}).get("scale_deviation_pct")
if dev is not None and abs(dev) > 1.5:
    flags.append(("flag", f"d-spacings offset {dev:+.1f}% from reference"))
if s.get("rings", {}).get("unexplained_rings"):
    flags.append(("flag", f"{len(s['rings']['unexplained_rings'])} unexplained ring(s)"))
if not elements and not check_mode:
    flags.append(("flag", "No elements given"))
n_c = s["candidates_searched"]
flags.append(("ok", f"{n_c} candidate{'s' if n_c != 1 else ''} searched in {s['timings_s'].get('total', 0):.1f} s"))
st.markdown("".join(f'<span class="{c}">{t}</span>' for c, t in flags), unsafe_allow_html=True)
if truth:
    ans = truth.get("phase") or f"diffuse halo at d = {truth.get('d_halo', 0):.2f} Å"
    if truth.get("zone_axis"):
        z = canonical_uvw(LIB[truth["phase"]], np.array(truth["zone_axis"])) if truth.get("phase") in LIB else truth["zone_axis"]
        ans += f" along {fmt_index(z, '[]')}"
    st.markdown(f'<p class="truth">Simulated example. True answer: {truth["type"]}, {ans}.</p>', unsafe_allow_html=True)

# What to do next, in one place ------------------------------------------------
guide = []
if not s["calibrated"]:
    guide.append("Calibrate (step 3) to compare absolute d-spacings; without it, look-alike phases of one structure type are indistinguishable.")
if not elements and not check_mode:
    guide.append("Add the elements from EDS (step 2) to narrow the candidates.")
adv_sum = (res.advice or {}).get("summary")
if adv_sum and (s.get("spots", {}).get("ambiguous") or s.get("rings", {}).get("ambiguous") or s.get("calibration_conflict")):
    guide.append(adv_sum)
elif adv_sum and kind == "single crystal":
    guide.append(adv_sum)
if checks and not check_mode:
    guide.append("CIF check: " + "; ".join(f"{c.phase.name}: {c.verdict}" for c in checks) + " (see the CIF check tab).")
if guide:
    st.markdown('<div class="guide"><b>What to do next.</b> ' + " ".join(guide[:3]) + "</div>", unsafe_allow_html=True)

# Views -----------------------------------------------------------------------
left, right = st.columns([1.1, 1])
with left:
    st.plotly_chart(viz.pattern_figure(res, show_overlay=overlay), width="stretch", config={"displaylogo": False, "scrollZoom": True})
with right:
    if res.spot_result and res.spot_result.solutions and kind in ("single crystal", "mixed"):
        st.plotly_chart(viz.zone_figure(res), width="stretch", config={"displaylogo": False})
    else:
        st.plotly_chart(viz.profile_figure(res), width="stretch", config={"displaylogo": False})

tab_names = ["Explanation", "Next step", "Evidence", "Candidates", "Report", "Diagnostics"]
if checks:
    tab_names = (["CIF check"] + tab_names) if check_mode else (tab_names[:1] + ["CIF check"] + tab_names[1:])
tabs = dict(zip(tab_names, st.tabs(tab_names)))
tab_expl, tab_next, tab_evidence, tab_cands, tab_report, tab_diag = (tabs[n] for n in ["Explanation", "Next step", "Evidence", "Candidates", "Report", "Diagnostics"])
if checks:
    with tabs["CIF check"]:
        import pandas as pd
        badge = {"consistent": ("ok", "Consistent"), "lattice change": ("flag", "Consistent with a lattice change"),
                 "scale mismatch": ("flag", "Scale mismatch: check calibration"), "partly": ("flag", "Partly consistent"),
                 "not consistent": ("flag", "Not consistent")}
        st.markdown("Each of your structures compared with the pattern, best first. *Lattice change* is the uniform scale that best "
                    "fits (+ means the measured lattice is larger than in the CIF).")
        for i, c in enumerate(checks):
            cls, lab = badge[c.verdict]
            with st.expander(f"{c.phase.name}: {lab}", expanded=(i == 0)):
                st.markdown(f'<span class="{cls}">{lab}</span> <b>{c.headline}</b>', unsafe_allow_html=True)
                st.write(c.explanation)
                nice = {"rings_explained": "measured rings explained", "strong_reflections_seen": "strong CIF reflections seen",
                        "reflections_compared": "reflections compared", "mean_residual_pct": "mean d error after lattice refinement (%)",
                        "zone_axis": "zone axis", "spots_explained_pct": "spot intensity explained (%)",
                        "strong_spots_missing": "strong predicted spots missing", "forbidden_spots": "spots on forbidden positions"}
                m = {nice.get(k, k): v for k, v in c.metrics.items()}
                if c.lattice_change_pct is not None:
                    m["lattice change vs CIF (%)"] = round(c.lattice_change_pct, 2)
                if c.refined_cell:
                    rc = c.refined_cell
                    m["refined a, b, c (Å)"] = f"{rc['a']:.4f}, {rc['b']:.4f}, {rc['c']:.4f}  (CIF {rc['a_cif']:.4f}, {rc['b_cif']:.4f}, {rc['c_cif']:.4f})"
                if m:
                    st.dataframe(pd.DataFrame([{"quantity": k, "value": str(v)} for k, v in m.items()]), hide_index=True, width="stretch")
                if c.table:
                    st.markdown("**Reflections: CIF vs measured**")
                    tdf = pd.DataFrame(c.table)
                    for col in ("ring", "spot"):
                        if col in tdf:
                            tdf[col] = tdf[col].astype("Int64")
                    st.dataframe(tdf.astype(object).where(tdf.notna(), ""), hide_index=True, width="stretch")
                if c.unexplained:
                    st.markdown("**Measured " + ("rings" if c.mode == "rings" else "spots") + " this structure does not explain**")
                    st.dataframe(pd.DataFrame(c.unexplained), hide_index=True, width="stretch")
                if c.zones and len(c.zones) > 1:
                    st.markdown("**Zone axes of this structure, best first**")
                    st.dataframe(pd.DataFrame(c.zones).rename(columns={"zone_axis": "zone axis", "spots_explained_pct": "spot intensity explained (%)",
                                                                       "lattice_change_pct": "lattice change (%)"}), hide_index=True, width="stretch")
                if c.table:
                    st.download_button("Download this comparison (CSV)", report.csv_bytes(pd.DataFrame(c.table)),
                                       file_name=f"cif_check_{c.phase.name}.csv", mime="text/csv", key=f"dl_cif_{i}")
with tab_expl:
    st.markdown(st.session_state.get("expl", ""))
    src = st.session_state.get("expl_src", "template")
    st.caption("Written by " + ("the built-in explainer from the measured numbers." if src == "template" else f"{src} from the measured numbers (it does not compute anything)."))
with tab_next:
    adv = res.advice or {}
    items = adv.get("items", [])
    if adv.get("kind") == "tilt" and items:
        hyps = adv.get("hypotheses", [])
        if len(hyps) > 1:
            st.markdown(f"**{len(hyps)} answers fit this pattern.** Each one implies a different crystal orientation, so the same specimen tilt "
                        "gives a different new pattern for each. These tilts (within ±35°) tell them apart.")
        else:
            st.markdown("**The answer is unambiguous.** To confirm it, record a second zone axis. These are the closest ones.")
        labels = [f"Tilt {it['tilt_deg']:.0f}°: " + ", ".join(f"{p['phase']} → {p['zone'] or 'off-axis'}" for p in it["predictions"]) for it in items]
        pick = st.radio("Suggested tilts", range(len(items)), format_func=lambda i: labels[i], label_visibility="collapsed")
        it = items[pick]
        st.write(it["text"])
        st.plotly_chart(viz.tilt_figure(res, it), width="stretch", config={"displaylogo": False})
        st.caption("Left: the spots measured now, with the tilt axis dashed. Then the pattern each candidate predicts after the tilt, "
                   "in the same frame (kinematical intensities, excitation error included). Tilt axis directions assume the "
                   "diffraction pattern is not rotated relative to the specimen holder; check the image/diffraction rotation on your microscope.")
        if hyps:
            import pandas as pd
            st.dataframe(pd.DataFrame([{"candidate": h["phase"], "zone now": h["zone"], "score": h["score"],
                                        "orientation fit (°)": h["orientation_fit_deg"]} for h in hyps]), hide_index=True, width="stretch")
    elif adv.get("kind") == "rings":
        st.write(adv.get("summary", ""))
        if items:
            import pandas as pd
            st.dataframe(pd.DataFrame([{"ring of": i["phase"], "hkl": i["hkl"], "d (Å)": i["d_A"], "relative intensity": i["rel_intensity"],
                                        "absent in": ", ".join(i["absent_in"])} for i in items]), hide_index=True, width="stretch")
    else:
        st.write(adv.get("summary") or "No advice for this pattern.")
with tab_evidence:
    if res.ring_result:
        st.markdown("Each measured ring, the plane it was assigned to, and the d-spacing error.")
        st.dataframe(report.rings_table(res), hide_index=True, width="stretch")
    if res.spot_result and res.spot_result.solutions:
        st.markdown("Each detected spot and its (hkl) in the best zone-axis solution.")
        st.dataframe(report.spots_table(res), hide_index=True, width="stretch")
    if res.amorphous:
        st.json(res.amorphous)
with tab_cands:
    st.markdown("All candidate phases, best first. Score combines rings/spots explained, missing strong reflections, d-spacing error and scale deviation.")
    st.dataframe(report.ranking_table(res), hide_index=True, width="stretch")
    st.caption("Sources: " + ", ".join(f"{k}: {v}" for k, v in s["candidate_sources"].items() if v))
with tab_report:
    st.markdown("**Figure caption**")
    st.write(report.caption(res))
    st.markdown("**Methods paragraph**")
    st.write(report.methods(res))
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Download figure (PNG)", viz.export_png(res, dpi=220), file_name="saedscope_figure.png", mime="image/png", width="stretch")
    c2.download_button("Download results (JSON)", report.summary_json(res), file_name="saedscope_results.json", mime="application/json", width="stretch")
    tbl = report.spots_table(res) if (res.spot_result and res.spot_result.solutions and kind in ("single crystal", "mixed")) else report.rings_table(res)
    c3.download_button("Download table (CSV)", report.csv_bytes(tbl), file_name="saedscope_table.csv", mime="text/csv", width="stretch")
    cif = report.best_phase_cif(res)
    if cif:
        c4.download_button("Download best phase (CIF)", cif[1], file_name=cif[0], mime="chemical/x-cif", width="stretch")
with tab_diag:
    st.markdown("**Why this pattern type**")
    for r in res.classification.reasons:
        st.write("• " + r)
    st.json(res.classification.features)
    st.markdown("**Notes**")
    for n in res.notes + (s.get("rings", {}) or {}).get("notes", []) + (s.get("spots", {}) or {}).get("notes", []):
        st.write("• " + n)
    if ctx_notes:
        st.write("• From your description: " + "; ".join(ctx_notes))
    st.markdown("**Timing (s)**")
    st.json(s["timings_s"])
