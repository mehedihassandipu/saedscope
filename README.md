# SAEDscope + SAED-Bench

**Identify the phase and zone axis in an electron diffraction pattern, and test whether AI models can do it too.**

* **SAEDscope** is an assistant for selected-area electron diffraction (SAED). Give it a pattern and a sentence about
  the sample. It works out whether the pattern comes from a single crystal, a polycrystal or an amorphous material,
  finds the phase and zone axis that explain the spots or rings, labels every reflection, and explains the evidence.
  Physics does the matching; a language model only explains the numbers. No training data is needed.
* **SAED-Bench** is a benchmark of electron-diffraction questions with exact answers, for comparing vision-language models
  with SAEDscope: a standard split (160 items), a realistic hard split (200) and a held-out hard split (200). SAEDscope
  scores 99%, 97.5% and 95.5%. See [`bench/README.md`](bench/README.md). The benchmark images are not shipped in the zip;
  `python -m bench.generate --n 40 --out bench/data/v0` and `python -m bench.generate_hard ...` recreate them identically.
* A one-page write-up is in [`docs/WRITEUP.md`](docs/WRITEUP.md); slide figures in `docs/figures/`
  (`python docs/make_figures.py`); a timed demo script in [`docs/PITCH_NOTES.md`](docs/PITCH_NOTES.md); hosting in
  [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Quick start

```bash
git clone https://github.com/mehedihassandipu/saedscope.gitcd saedscope
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest -q tests` (about a minute).

Open the app, pick **Try an example** in the sidebar (simulated patterns with known answers) or upload your own
pattern. Links such as `http://localhost:8501/?example=3&run=1` open an example and run it directly (handy for demos).

## What goes in

| Input | Notes |
|---|---|
| Pattern | `.dm3`/`.dm4` (calibration read from the file), `.tif`, `.png`/`.jpg` (e.g. cropped from a paper), `.npy` |
| Elements | e.g. `Ni, Al, O` from EDS. Restricts the search; strongly recommended |
| Description | free text, e.g. "NiAl alloy annealed in air". Formulas and material names are turned into elements; "in air" adds oxides |
| Calibration | from the file; **from a standard pattern** (e.g. evaporated gold at the same camera length: fits the camera constant and ring ellipticity); **from a scale bar** burnt into the image (detected automatically); Å⁻¹ or nm⁻¹ per pixel; voltage + camera length + pixel size; or none (ratio-only mode) |
| Candidates | built-in library of 33 phases, Materials Project (free API key, `pip install mp-api`), and/or your own CIFs |
| AI explanation | optional: Anthropic or any OpenAI-compatible endpoint (including local open models). Without a key a built-in explainer is used |

## What comes out

* Pattern type with the reasons (rings / spots / halo features).
* **Rings:** ranked phases, each ring's d-spacing, (hkl) and error, a possible second phase, unexplained rings, a uniform
  scale offset (calibration check).
* **Spots:** zone axis [uvw], every spot indexed, measured vs expected d-spacings and angle, simulated zone-axis pattern,
  forbidden spots flagged as possible double diffraction.
* **Amorphous:** halo d-spacing and an Ehrenfest estimate of the nearest-neighbour distance.
* **Next step:** when several answers fit, the tilt (within ±35°) that tells them apart, with the pattern each candidate
  predicts after that tilt; for look-alike ring patterns, the rings one candidate has and the other lacks. When the answer
  is unambiguous, the nearest second zone axis to confirm it.
* Figure (PNG), caption, methods paragraph, tables (CSV), full results (JSON), best-matching phase (CIF).

## Real data

**Known-answer tests on public data** (pyxem demo data, Zenodo 14113591, CC BY 4.0; details in `docs/WRITEUP.md`):

| Dataset | Result |
| --- | --- |
| `mgo_nano.zspy` (4D-STEM, MgO nanocrystals) | MgO; CIF check: MgO consistent (refined lattice within 0.4%), [001] preferred orientation found, Au/Al/Cu rejected or flagged |
| `au_xgrating_20cm.tif` (gold grating, 256 px) | Au rings (111/200 doublet resolved); as a standard 0.01002 Å⁻¹/px, within 0.4% of pyxem's tutorial value on the same footing |

* **Calibrate once per camera length.** Choose *From a standard pattern*, upload a gold (or Al, Ag, Pt, Si, Ni, Cu, MgO,
  TiN) ring pattern taken at the same settings. SAEDscope fits the camera constant, reports the error per ring and measures
  ring ellipticity (projector-lens distortion); the ellipticity correction is optional (*Advanced*), because the distortion
  only transfers to patterns from the same session and camera length.
* **Calibration conflicts are reported.** If no candidate fits at the given calibration but one fits the spot geometry at
  another scale, SAEDscope says by how much the calibration would have to be off, and which phases fit at the given scale.
  On our microscope this caught scale bars that were about 10% off (see `docs/WRITEUP.md`).
* **Figures cropped from papers** (PNG/JPG, 8-bit, often saturated): burnt-in scale bars and their labels are masked;
  a direct beam hidden under the beam stop is found from the diffuse glow and the end of the stop, then fixed to the spot
  lattice (a lattice alone is symmetric about every half-lattice point, so Friedel pairs cannot decide); predicted spots
  under the stop or in a saturated glow are not counted as missing; scale-bar calibrations may be off by up to ~10%
  and the deviation is reported. Inverted (dark-spot) figures are flipped automatically. Still not handled: index labels or
  arrows drawn on the pattern, false-colour maps.
* **Small detectors** (e.g. 256 px): rings a few pixels wide are recognised as crystalline, and unresolved pairs such as fcc
  111/200 are split.
* **Ring ellipticity is corrected automatically** on ring patterns: it is measured on the pattern's strongest ring and
  removed before matching (2–3% on our microscope, enough to split rings).
* **Look-alike phases are named as a group.** Without a calibration, all phases of one structure type give the same
  ring ratios; with one, phases whose lattices differ by under ~1% (Au, Ag, Al) still cannot be separated. The verdict
  then lists the group and asks for the elements present.
* **4D-STEM files** (`.zspy`, `.hspy`, `.hdf5`, `.mib`): upload them in the app and choose which pattern to analyse
  (all grains overlaid = max, recommended for nanocrystals; mean; or the single strongest scan position). The stored
  calibration and beam energy are used. From the command line: `python -m saedscope.extract file.zspy --mode
  mean|max|brightest|pos`. Needs `pip install "zarr<3" h5py` (in `requirements.txt`).
* **One systematic row** (a crystal slightly off a zone axis: g, 2g, −g through the direct beam) is recognised: its
  d-spacing is measured from all orders and matched, and the verdict says the zone axis cannot be determined from one row.
* **Spotty nanocrystalline patterns:** when there are no smooth rings, the radii of the spots are clustered into rings.
* **Check against my CIF files:** choose it at the top of the sidebar, upload your CIF(s). For each structure you get a
  verdict (consistent / consistent with an X% lattice change / scale mismatch: check calibration / partly / not
  consistent), every predicted reflection next to the measured one, missing strong reflections, unexplained rings or
  spots, the refined lattice parameters and, for spot patterns, the best zone axes. Several CIFs are ranked. When strong
  reflections are missing because the crystals share an orientation (e.g. cubes lying on a face), it names the zone.
* **Guided steps:** the sidebar shows numbered steps with their status; before analysis a checklist flags missing
  elements or calibration, offers one-click fixes (use the detected scale bar, calibrate with a standard) and warns when
  the standard is the sample itself. After analysis, *What to do next* sits right under the verdict.
* **Exported images with an annotation strip** (scale bar and text below the pattern) are handled: the strip is removed
  for the analysis and searched for the scale bar.
* **Patterns from papers:** *From a scale bar in the image* finds a burnt-in reciprocal-space bar (you type its value,
  e.g. 5 nm⁻¹) and masks the bar and its label.
* **Many patterns:** *Several files* in the app, or the command line:

  ```bash
  python -m saedscope.batch my_patterns/ --elements "Ni, Al, O" --standard gold.dm4 --standard-material Au --out results/ --figures
  ```

  writes one row per pattern to `results/summary.csv`, plus a JSON (and optionally a figure) per pattern.
* **Manual control** (*Advanced*): pattern centre, inner radius, tolerances, forcing the pattern type, beam-stop masking.

## How it works

1. **Centre:** beam stop masked and inpainted; coarse centre from the diffuse glow; refined with Friedel symmetry
   (auto-convolution of the high-pass image) and, for rings, profile sharpness; for spots, jointly with the lattice fit.
2. **Pattern type (no training):** sharp rings with even intensity → polycrystalline; ≥6 spots on one 2D lattice, most
   with a Friedel partner (g and −g) → single crystal; many spots without partners → coarse-grained polycrystal; broad
   halos only → amorphous; combinations → mixed. Weak spot patterns are searched again at a lower threshold, keeping only
   spots whose Friedel partner is also found; sharp rings sitting on a support-film halo are separated from it.
3. **Candidates:** phases whose elements are a subset of yours (built-in, Materials Project, your CIFs).
4. **Rings:** robust azimuthal profile → background (asymmetric least squares) → ring positions → each candidate's
   kinematical ring list is matched allowing a small uniform scale factor; score = rings explained × penalty for missing
   strong lines × precision × intensity agreement × scale plausibility. Without calibration, d-spacing ratios are used.
5. **Spots:** spot detection with radius-dependent noise → 2D lattice fit (with centre offset) → plane pairs matching
   both d-spacings and the angle → zone axis = cross product → verified against the simulated zero-order Laue zone.
6. **Next step:** each competing (phase, zone) answer fixes a crystal orientation (Kabsch fit of the indexed spots plus the
   beam direction). For low-index zones reachable within ±35° we apply the same specimen tilt to every answer, find the zone
   each would show and simulate it (kinematical intensities with excitation error), and rank tilts by how many answers they
   separate. The tilt axis is given relative to the indexed spots ("along the (010) row").
7. **Explanation:** the structured result is handed to a language model (or the built-in explainer) with the rule
   "use only numbers that appear in the results".

## Tests

```bash
pytest -q
```

## Project layout

```
app.py                 Streamlit app
saedscope/             analysis engine (io, imageproc, crystal, rings, spots, classify, pipeline, advisor, calibrate,
                       batch, viz, report, llm, synth)
bench/                 SAED-Bench: generate.py, run_agent.py, run_vlm.py, summarize.py, real/ (template for real data)
tests/                 sanity tests on simulated patterns
```

## Prior work and positioning

Classical indexing tools (e.g. CrysTBox) match spacings and angles once the user supplies candidate phases. Machine-learning
approaches classify patterns into a fixed set of trained phases (Mika et al., *J. Materiomics* 2024: 66–73% on 9 Pu–Zr phases)
or predict crystal symmetry from millions of simulated patterns (Nathani et al., *npj Comput. Mater.* 2026: 89% crystal system),
without identifying the specific material or using sample context. SAEDscope adds automatic, context-aware candidate search,
open-set identification from any database phase, pattern-type detection and explanations; SAED-Bench measures whether general
vision-language models can do this at all.

## Limitations

* Kinematical simulation: intensities are used only as a weak hint (dynamical scattering).
* Phases whose lattices differ by less than ~1–2% (e.g. Au and Al) cannot be separated by SAED d-spacings alone; the app flags them as ambiguous.
* Materials Project structures are DFT-relaxed (lattice ~1–2% large); the matcher allows a uniform scale factor and reports it.
* Built for single patterns and small batches; it is not designed for million-pattern 4D-STEM maps.
* Tilt directions assume the diffraction pattern is not rotated relative to the holder axes; check the rotation calibration
  of your microscope before following a suggested tilt axis.
* Without calibration (ratio-only), phases of the same structure type are indistinguishable (e.g. NiAl [001] vs Al [100]).
* `.dm3`/`.dm4` reading uses RosettaSciIO and has been tested only on simulated data in this build.
* The Materials Project lookup and the language-model calls need your own API keys and have not been exercised offline in this build.

## Data credits

The MgO nanocrystal 4D-STEM scan and the gold cross-grating pattern used as known-answer tests are from the pyxem
4D-STEM demo data (C. Francis, [Zenodo 14113591](https://zenodo.org/records/14113591), CC BY 4.0). They are not
included in this repository. Our own microscope images are not included either.

## Provenance

Built for the OSI Hackathon (October 2026) by Mehedi Hassan Dipu. The first version (the core pipeline and the
standard and hard benchmark splits) was written on 21 October. During the event I added the real-data evaluation (our microscope
and public data), the calibration checks, the CIF check and tilt advice, 4D-STEM, small-detector and paper-figure
support, Velox calibration reading, and the write-up and video.

## Licence

MIT, see [`LICENSE`](LICENSE).
