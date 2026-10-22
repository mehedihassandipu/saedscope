# Real-data split (add your own patterns)

The simulated split tests reading skill with exact answers. A small real split shows the results hold on
experimental data. Add 5–10 patterns per task with the answer your expert gives.

1. Put images in `bench/real/images/` (PNG/TIFF; for the models, export what a person would see).
2. Fill `items_template.csv` (one row per question) and convert it:

```bash
python -m bench.real_to_jsonl bench/real/items_template.csv bench/real/items.jsonl
```

Columns:

| column | meaning |
|---|---|
| `id` | unique id, e.g. `real_zone_01` |
| `task` | `pattern_type`, `ring_phase`, `ring_dspacing` or `zone_axis` |
| `image` | path relative to `bench/real/`, e.g. `images/nial_01.png` |
| `calibration_inv_A_per_px` | reciprocal-space calibration (Å⁻¹ per pixel) |
| `centre_x`, `centre_y` | approximate pattern centre in pixels |
| `phase` | true phase name (must match a built-in phase name, or leave blank for amorphous) |
| `options` | for `ring_phase`: four phase names separated by `;` in the order A;B;C;D |
| `answer` | `single crystal`/`polycrystalline`/`amorphous`, a letter, a d-spacing in Å, or `u v w` |

Then run the agent and models with `--data bench/real` exactly as for the simulated split.
Do not publish unpublished lab images without your advisor's permission.
