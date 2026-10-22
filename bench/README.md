# SAED-Bench v0.1 (standard and hard splits)

**Can AI models read electron diffraction patterns?** SAED-Bench asks vision-language models the questions a TEM user
answers every day, on patterns whose answers are known exactly, and compares them with SAEDscope's physics pipeline.

## Tasks (40 items each, 160 total)

| Task | Question | Graded as |
|---|---|---|
| `pattern_type` | Single crystal, polycrystalline or amorphous? | exact (chance 33%) |
| `ring_phase` | Which of four phases produced this ring pattern? (calibration and centre given) | exact letter (chance 25%) |
| `ring_dspacing` | d-spacing of the innermost ring (calibration and centre given) | within ±3% |
| `zone_axis` | Zone axis of this single-crystal pattern of a named phase (calibration and centre given) | any symmetry-equivalent [uvw] |

## Hard split (`v0-hard`, 200 items; held-out copy `v0-hard-heldout`)

The same four questions plus `spot_phase` (which of four phases made this single-crystal pattern?), under the conditions a
microscopist actually meets:

* single crystals 0.5–2.5° off the zone axis, so spot intensities follow the Laue circle rather than the symmetry
  (excitation error from the Ewald sphere, thickness-like falloff);
* strong dynamical intensity scatter; double-diffraction spots at forbidden positions;
* projector ellipticity up to 1.5% and a nominal calibration that is only good to ±1.5% (stated in the prompt);
* textured or very coarse-grained ring patterns, an amorphous carbon-film halo, low counts;
* `spot_phase` distractors are rejected if any of their low-index zones gives the same 2D lattice (within 3.5% and 3°).

```bash
python -m bench.generate_hard --n 40 --out bench/data/v0-hard                       # seed 4242
python -m bench.generate_hard --n 40 --seed 777 --out bench/data/v0-hard-heldout    # held-out copy
```

## How items are made

* Structures: 31 common phases (metals, oxides, nitrides; cubic, hexagonal, tetragonal, trigonal).
* Patterns: kinematical simulation with Doyle–Turner electron scattering factors, a direct beam, diffuse background,
  a beam stop, Poisson noise, random centre offset and rotation, random "dynamical" intensity perturbations, spotty rings
  (coarse grains) and occasional double-diffraction spots.
* Images are shipped as 8-bit log-scaled PNGs (what a person sees on screen). The models and SAEDscope get the same PNG
  and the same text.
* Distractor phases in `ring_phase` are chosen to be plausible (shared elements / crystal system) but physically
  distinguishable: a distractor is rejected if ≥80% of the true phase's strong rings fall within 2% of its rings.
* Answers are generated, never hand-labelled, so there is no labelling error and no chance the items were seen in training.

## Run it

```bash
python -m bench.generate --n 40 --out bench/data/v0      # run first: recreates the 160 images (≈2 min, byte-identical, seed 2026)
python -m bench.run_agent  --data bench/data/v0          # SAEDscope physics pipeline (no API key)

export ANTHROPIC_API_KEY=...                              # vision-language models
python -m bench.run_vlm --provider anthropic --model claude-sonnet-5-5 --limit 8   # cheap smoke test first
python -m bench.run_vlm --provider anthropic --model claude-sonnet-5-5

export OPENAI_API_KEY=...                                 # any OpenAI-compatible endpoint
python -m bench.run_vlm --provider openai --model <model> [--base-url https://openrouter.ai/api/v1]

python -m bench.summarize --data bench/data/v0           # leaderboard.md + leaderboard.png
# the same commands with --data bench/data/v0-hard for the hard split
```

`run_vlm` resumes if interrupted, retries on errors, and stores each raw answer for inspection.
Open-weight vision models can be tested through any OpenAI-compatible server (vLLM, Ollama) with `--base-url`.

## Limitations (say these in the write-up)

* Kinematical simulation: real patterns have stronger dynamical effects, thickness variation and imperfect tilts.
  Add a real split (`bench/real/`) to check the conclusions transfer.
* **SAEDscope's score here is optimistic:** the benchmark patterns are generated with the same kinematical physics that
  SAEDscope uses for matching, and its parameters were tuned on these simulations. Treat it as an upper bound; the real-data
  split is the fair test for SAEDscope. The models, by contrast, have never seen this simulator.
* The built-in phase set is small (31 phases).

## Current results

| Split | Model | Pattern type | Ring phase | Innermost ring d | Zone axis | Spot phase | Overall |
|---|---|---|---|---|---|---|---|
| any | Random guessing | 33% | 25% | — | — | 25% | — |
| standard (160) | SAEDscope | 100% | 100% | 100% | 98% | — | 99% |
| hard (200) | SAEDscope, first version | 88% | 92% | 78% | 70% | 80% | 82% |
| hard (200) | SAEDscope | 98% | 100% | 95% | 100% | 95% | 97.5% |
| hard, held out (200) | SAEDscope | 100% | 100% | 92% | 90% | 95% | 95.5% |
| all | Vision-language models | *run `bench.run_vlm` with your API keys* | | | | | |

* The hard split exposed three weaknesses (weak off-axis spots, rings hidden under a support-film halo, coarse grains
  mistaken for single crystals). They were fixed with physics (Friedel-pair spot validation, a halo-aware ring search, a
  Friedel-pair fraction for the pattern type), and a held-out hard split was generated afterwards with a new seed to check
  that the fixes generalise. Report the held-out number. Later changes made for real data (small detectors, figures
  cropped from papers, a hidden direct beam) were checked on all three splits only for regressions: 4 items were gained
  and 1 lost (`hspotphase_003`, Al [100] with strong double-diffraction spots, now ranked behind magnetite).
* Calibration and centre are given to the models to keep the comparison fair; this makes the tasks easier than real use.

## Related work

CrystalXRD-Bench (2026) evaluates vision-language models on powder X-ray diffraction indexing; MatQnA and MatCha (2025)
use multiple-choice questions from paper figures. SAED-Bench targets electron diffraction with quantitative,
physics-graded answers.
