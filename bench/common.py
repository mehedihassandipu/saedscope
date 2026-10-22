"""Shared helpers for SAED-Bench: prompts, answer parsing and grading."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TASKS = {
    "pattern_type": "Pattern type (single crystal / polycrystalline / amorphous)",
    "ring_phase": "Phase identification from a ring pattern (4 options)",
    "ring_dspacing": "d-spacing of the innermost ring (±3%)",
    "zone_axis": "Zone axis of a single-crystal spot pattern",
    "spot_phase": "Phase identification from a single-crystal spot pattern (4 options; hard split)",
}

SYSTEM = ("You are an expert transmission electron microscopist. You will be shown a selected-area electron diffraction (SAED) "
          "pattern and asked one question. Reason carefully, then give your final answer on the last line in the form "
          "'ANSWER: <answer>'.")


def load_items(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = re.findall(r"ANSWER\s*[:：]\s*(.+)", text, flags=re.I)
    ans = m[-1] if m else text.strip().splitlines()[-1]
    return ans.strip().strip("*").strip()


def parse_uvw(s: str):
    s = s.replace("−", "-").replace("–", "-")
    m = re.search(r"\[?\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*\]?", s)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.search(r"\[\s*(-?\d)(-?\d)(-?\d)\s*\]", s)  # compact [011] / [1-10]
    if m:
        return tuple(int(x) for x in m.groups())
    digits = re.findall(r"-?\d", s)
    if len(digits) == 3:
        return tuple(int(x) for x in digits)
    return None


def parse_float(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", "."))
    return float(m.group()) if m else None


def grade(item: dict, answer: str, phase_lookup=None) -> bool:
    task = item["task"]
    a = (answer or "").strip().lower()
    if task == "pattern_type":
        truth = item["answer"]
        if "single" in a:
            pred = "single crystal"
        elif "poly" in a:
            pred = "polycrystalline"
        elif "amorph" in a:
            pred = "amorphous"
        else:
            return False
        return pred == truth
    if task in ("ring_phase", "spot_phase"):
        m = re.search(r"\b([A-D])\b", answer.upper()) if answer else None
        return bool(m) and m.group(1) == item["answer"]
    if task == "ring_dspacing":
        v = parse_float(a)
        return v is not None and abs(v - item["answer"]) / item["answer"] <= 0.03
    if task == "zone_axis":
        uvw = parse_uvw(answer or "")
        if not uvw or uvw == (0, 0, 0):
            return False
        if phase_lookup is None:
            return False
        from saedscope.crystal import canonical_uvw
        from math import gcd

        g = 0
        for x in uvw:
            g = gcd(g, abs(x))
        uvw = tuple(x // (g or 1) for x in uvw)
        ph = phase_lookup(item["meta"]["phase"])
        t = np.array(item["answer"])
        t = t // max(1, int(np.gcd.reduce(np.abs(t))))
        return canonical_uvw(ph, np.array(uvw)) == canonical_uvw(ph, t)
    return False


def build_prompt(item: dict) -> str:
    return item["prompt"]
