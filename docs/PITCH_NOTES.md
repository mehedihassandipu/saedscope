# Pitch notes (edit freely)

## One line
Classical SAED tools are accurate but need an expert to drive them; ML models are fast but closed-set black boxes;
general AI models are flexible but untested. SAEDscope combines physics (does the matching) with AI (does the reasoning),
and SAED-Bench measures whether general vision-language models can read electron diffraction at all.

## Two-minute video script

Record the app full screen at 1500 px wide. Each example link opens and runs a pattern directly:
`http://localhost:8501/?example=N&run=1` (N counts from 0 in the example list).

| Time | Show | Say |
|---|---|---|
| 0:00 | A pattern in ImageJ with a ruler | "Indexing one diffraction pattern by hand takes us 20 to 30 minutes, and look-alike phases are easy to mix up." (time your own manual indexing and use that number) |
| 0:15 | Upload your own ring pattern, type "NiAl annealed in air", choose *From a standard pattern* and upload the gold pattern | "We calibrate once with gold. SAEDscope fits the camera constant and corrects lens ellipticity." |
| 0:35 | Verdict, ring overlay, Evidence tab | "Polycrystalline NiAl: every ring indexed, d-spacing error under a tenth of a percent. Nothing here is trained; it is diffraction physics." |
| 0:55 | `?example=4&run=1` (magnetite [112]) | "Single crystal: zone axis, every spot labelled, and the simulated pattern it should match." |
| 1:10 | `?example=7&run=1`, then the **Next step** tab | "With no elements given, four phases fit this hexagonal pattern. SAEDscope says which tilt tells them apart and what each answer would look like after it." |
| 1:35 | `docs/figures/fig4_benchmark.png`, then the VLM leaderboard | "SAED-Bench: 360 graded questions, including realistic off-axis patterns. Here is how general AI models do against physics." |
| 1:50 | Limitations slide | "Next: our own real patterns as a test split, and dynamical intensities." |

## Slide figures (`python docs/make_figures.py`)

* `fig1_pipeline.png`: how SAEDscope works.
* `fig2_examples.png`: polycrystal, single crystal and amorphous results side by side.
* `fig3_advisor.png`: the tilt that separates four look-alike answers.
* `fig4_benchmark.png`: SAEDscope accuracy on the standard and hard splits.
* `docs/screenshots/`: app screenshots (rings, spots, amorphous, calibration, batch, next step).

## Related-work paragraph (for the 1-page write-up)
Automated SAED analysis has followed three routes. Classical indexing tools (e.g., CrysTBox) match measured spacings
and angles against simulated patterns but require the user to supply candidate phases. Machine-learning approaches
classify patterns into a fixed set of trained phases (66–73% on 9 Pu–Zr phases; Mika et al., 2024) or predict crystal
symmetry from millions of simulated patterns (89% crystal system; Nathani et al., 2026), without identifying the specific
material or using sample context. Meanwhile, general multimodal AI models are increasingly used to read characterization
data, but have been benchmarked only on multiple-choice questions or X-ray diffraction (MatQnA; CrystalXRD-Bench).
We introduce SAED-Bench, a quantitative benchmark of vision-language models on electron diffraction, together with a
physics-grounded agent that performs open-set, context-aware phase and zone-axis identification without training data.

## Claims to avoid
* "First automated SAED indexing" (CrysTBox and others exist).
* "Outperforms ML methods" (different data and tasks; cite their numbers as context only).
* Anything about real-data accuracy until the real split is run.

## Before submitting
* Run the vision-language models on SAED-Bench, both splits (`bench/README.md`), and regenerate the leaderboards.
* Add 5–10 real patterns per task to `bench/real/` and run both the agent and at least one model on them.
* Time a manual indexing of one pattern for the "minutes vs seconds" slide.
* Fill in the Provenance section of README.md.
