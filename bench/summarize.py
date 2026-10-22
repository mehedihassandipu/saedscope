"""Summarise SAED-Bench results: accuracy per task and model, with a chance baseline and a chart.

    python -m bench.summarize --data bench/data/v0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import TASKS, load_items

CHANCE = {"pattern_type": 1 / 3, "ring_phase": 1 / 4, "spot_phase": 1 / 4}
SHORT = {"pattern_type": "Pattern type", "ring_phase": "Ring phase (4 options)", "ring_dspacing": "Innermost ring d (±3%)", "zone_axis": "Zone axis",
         "spot_phase": "Spot phase (4 options)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data/v0")
    args = ap.parse_args()
    data = Path(args.data)
    items = load_items(data / "items.jsonl")
    n_task = defaultdict(int)
    for it in items:
        n_task[it["task"]] += 1
    acc = {}
    for f in sorted(data.glob("results_*.jsonl")):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        model = rows[0].get("model", f.stem)
        per = defaultdict(lambda: [0, 0])
        for r in rows:
            per[r["task"]][0] += int(bool(r["correct"]))
            per[r["task"]][1] += 1
        acc[model] = {t: (c / n if n else None, n) for t, (c, n) in per.items()}
    present = {t for per in acc.values() for t in per}
    tasks = [t for t in TASKS if t in present] or list(TASKS)
    lines = ["| Model | " + " | ".join(SHORT[t] for t in tasks) + " | Overall |", "|---" * (len(tasks) + 2) + "|"]
    lines.append("| Random guessing | " + " | ".join(f"{100 * CHANCE[t]:.0f}%" if t in CHANCE else "—" for t in tasks) + " | — |")
    for model, per in acc.items():
        cells = []
        tot_c = tot_n = 0
        for t in tasks:
            if t in per and per[t][0] is not None:
                a, n = per[t]
                cells.append(f"{100 * a:.0f}% (n={n})")
                tot_c += a * n
                tot_n += n
            else:
                cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {100 * tot_c / tot_n:.0f}% |" if tot_n else " | — |")
    table = "\n".join(lines)
    print(table)
    (data / "leaderboard.md").write_text(table + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(acc)
    if models:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        w = 0.8 / (len(models) + 1)
        x = np.arange(len(tasks))
        cols = ["#3F6212", "#1D4ED8", "#B45309", "#7C3AED", "#BE185D", "#0F766E"]
        for k, m in enumerate(models):
            vals = [100 * acc[m][t][0] if t in acc[m] and acc[m][t][0] is not None else 0 for t in tasks]
            ax.bar(x + k * w, vals, w, label=m, color=cols[k % len(cols)])
        ch = [100 * CHANCE.get(t, 0) for t in tasks]
        ax.bar(x + len(models) * w, ch, w, label="random guessing", color="#D1D5DB")
        ax.set_xticks(x + w * len(models) / 2, [SHORT[t] for t in tasks], fontsize=9)
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.spines[["top", "right"]].set_visible(False)
        info = data / "info.json"
        ver = json.loads(info.read_text()).get("version", "0.1") if info.exists() else "0.1"
        ax.set_title(f"SAED-Bench v{ver}", fontsize=11)
        fig.tight_layout()
        fig.savefig(data / "leaderboard.png", dpi=170)
        print(f"chart → {data / 'leaderboard.png'}")


if __name__ == "__main__":
    main()
