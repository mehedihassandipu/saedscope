"""Language layer: plain-English context → elements, and results → explanation.

Principle: numbers come from the physics code; the language model only reads
the structured results and explains them. Without an API key a deterministic
template explanation is used, so the app always works.
"""

from __future__ import annotations

import json
import os
import re

from pymatgen.core import Composition, Element

from .crystal import parse_elements

MATERIAL_WORDS = {
    "stainless steel": ["Fe", "Cr", "Ni", "Mo", "Mn"], "316": ["Fe", "Cr", "Ni", "Mo", "Mn"], "304": ["Fe", "Cr", "Ni", "Mn"],
    "steel": ["Fe", "C", "Mn", "Cr"], "ferrite": ["Fe"], "austenite": ["Fe", "Ni", "Cr"], "inconel": ["Ni", "Cr", "Fe", "Mo", "Nb"],
    "superalloy": ["Ni", "Al", "Cr", "Co"], "brass": ["Cu", "Zn"], "bronze": ["Cu", "Sn"], "alumina": ["Al", "O"], "sapphire": ["Al", "O"],
    "titania": ["Ti", "O"], "rutile": ["Ti", "O"], "anatase": ["Ti", "O"], "zirconia": ["Zr", "O"], "silica": ["Si", "O"], "ceria": ["Ce", "O"],
    "magnetite": ["Fe", "O"], "hematite": ["Fe", "O"], "rust": ["Fe", "O"], "titanium": ["Ti"], "aluminium": ["Al"], "aluminum": ["Al"],
    "gold": ["Au"], "silver": ["Ag"], "copper": ["Cu"], "nickel": ["Ni"], "iron": ["Fe"], "silicon": ["Si"], "germanium": ["Ge"],
    "tungsten": ["W"], "platinum": ["Pt"], "palladium": ["Pd"], "zinc oxide": ["Zn", "O"], "magnesium": ["Mg"], "chromium": ["Cr"],
}
OXIDISING = ("in air", "oxidi", "oxidat", "annealed in o", "oxide", "corrosion", "rust")


def parse_context(text: str) -> tuple[list[str], list[str]]:
    """Deterministic parsing of a free-text sample description. Returns (elements, notes)."""
    if not text:
        return [], []
    low = text.lower()
    els: list[str] = []
    notes = []
    for word, e in MATERIAL_WORDS.items():
        if word in low:
            for x in e:
                if x not in els:
                    els.append(x)
            notes.append(f"'{word}' → {', '.join(e)}")
    # formulas such as NiAl, TiO2, LiCoO2, Ni3Al (need ≥2 elements or a digit to avoid words like 'In', 'As')
    for tok in re.findall(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?){2,}\b|\b[A-Z][a-z]?\d+\b", text):
        try:
            comp = Composition(tok)
            syms = [el.symbol for el in comp.elements]
            if len(syms) >= 1 and all(Element(s) for s in syms):
                for s in syms:
                    if s not in els:
                        els.append(s)
                notes.append(f"formula {tok} → {', '.join(syms)}")
        except Exception:
            continue
    if any(k in low for k in OXIDISING) and els and "O" not in els:
        els.append("O")
        notes.append("oxidising conditions → added O (oxides)")
    return els, notes


def merge_elements(elements_text: str, context: str) -> tuple[list[str], list[str]]:
    els = parse_elements(elements_text)
    ctx_els, notes = parse_context(context)
    for e in ctx_els:
        if e not in els:
            els.append(e)
    return els, notes


# ----------------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert transmission electron microscopist and crystallographer. You receive the JSON output of a "
    "physics-based selected-area electron diffraction (SAED) analysis. Explain the result to a materials-science "
    "graduate student in at most 170 words of plain prose (no headings, no bullet lists). Rules: use ONLY numbers "
    "that appear in the JSON; never invent d-spacings, phases or indices. State the pattern type and the identified "
    "phase/zone axis with the key evidence. If the result is ambiguous, uncalibrated, has unexplained rings, forbidden "
    "spots, a large scale deviation, or relies on DFT lattice parameters, say so plainly. Remind that SAED intensities "
    "are affected by dynamical scattering only if relevant. End with ONE concrete next step; if the JSON has a next_step entry, use its tilt or ring advice (e.g. tilt to a specific "
    "zone axis, calibrate with a gold standard, add EDS elements, or confirm with an experimental CIF)."
)


def template_explanation(s: dict) -> str:
    k = s["pattern_type"]
    out = []
    reasons = " ".join(s.get("pattern_type_reasons", [])[:2])
    out.append(f"**Pattern type: {k}** (confidence {s['pattern_type_confidence']:.0%}). {reasons}")
    if k in ("single crystal", "mixed") and s.get("spots", {}).get("best") and not s.get("calibration_conflict"):
        b = s["spots"]["best"]
        p = b["phase"]
        out.append(
            f"The spot pattern is indexed as **{p['name']}** ({p['space_group']}) along the **{b['zone_axis']}** zone axis. "
            f"The two shortest reflections are {b['g1']['hkl']} (measured d = {b['g1']['d_meas_A']:.3f} Å vs {b['g1']['d_ref_A']:.3f} Å) and "
            f"{b['g2']['hkl']} ({b['g2']['d_meas_A']:.3f} Å vs {b['g2']['d_ref_A']:.3f} Å) at {b['angle_meas_deg']:.1f}° (expected {b['angle_ref_deg']:.1f}°); "
            f"{b['spots_explained_pct']:.0f}% of the spot intensity is explained by the simulated {b['zone_axis']} pattern."
        )
        if b.get("forbidden_spots_observed"):
            out.append(f"{b['forbidden_spots_observed']} observed spot(s) are kinematically forbidden; this is common in SAED and usually caused by double diffraction.")
        if s["spots"].get("ambiguous") and len(s["spots"]["ranking"]) > 1:
            alt = s["spots"]["ranking"][1]
            out.append(f"An alternative fits almost as well: {alt['phase']} {alt['zone_axis']} (score {alt['score']} vs {s['spots']['ranking'][0]['score']}).")
    if s.get("systematic_row"):
        rw = s["systematic_row"]
        ms = rw.get("matches") or []
        out.append(("Only **one systematic row** of spots is visible" + (f", with d = {rw['d_A']:.3f} Å" if rw.get("d_A") else "")
                    + f" (orders {', '.join(str(o) for o in rw['orders_seen'])}). "
                    + (("It matches " + ", ".join(f"{m['phase']} {m['hkl']} ({m['error_pct']:+.1f}%)" for m in ms[:4]) + ". ") if ms else "")
                    + "One row fixes a single d-spacing but not the zone axis, so the phase is only narrowed down."))
    if s.get("calibration_conflict"):
        c = s["calibration_conflict"]
        alts = c.get("fits_at_given_calibration") or []
        out.append(f"**Calibration conflict.** {c['phase']} {c['zone_axis']} explains the spot spacings and angles, but only if the calibration "
                   f"is {c['calibration_off_by_pct']:+.1f}% different from the one given. Either the scale bar or camera length is off, or the "
                   "sample is a different phase"
                   + (": at the given calibration the spots fit " + ", ".join(f"{a['phase']} {a['zone_axis']}" for a in alts[:3]) + "." if alts else ".")
                   + " Calibrate with a standard (e.g. evaporated gold) at the same camera length to decide.")
    if s.get("rings", {}).get("primary_phase"):
        r = s["rings"]
        p = r["primary_phase"]
        idx = [a for a in r["assignments"] if a["phase"] == p["name"]]
        total = len(r["assignments"])
        errs = [abs(a["error_pct"]) for a in idx if a["error_pct"] is not None]
        lines = ", ".join(f"{a['d_meas_A']:.3f} Å ({a['hkl']})" for a in idx[:4] if a["d_meas_A"])
        txt = f"The rings are best matched by **{p['name']}** ({p['space_group']}): {len(idx)} of {total} rings indexed"
        if errs:
            txt += f" with a mean d-spacing error of {sum(errs) / len(errs):.2f}%"
        txt += f", e.g. {lines}."
        out.append(txt)
        if r.get("scale_deviation_pct") is not None and abs(r["scale_deviation_pct"]) > 1.5:
            out.append(f"All d-spacings are offset by {r['scale_deviation_pct']:+.1f}% from the reference, which points to a calibration error, lattice strain, or (for Materials Project structures) DFT lattice expansion.")
        if r.get("secondary_phase"):
            sec = r["secondary_phase"]["name"]
            secl = [a for a in r["assignments"] if a["phase"] == sec]
            out.append(f"Additional ring(s) at {', '.join(str(a['d_meas_A']) + ' Å' for a in secl)} match **{sec}**, suggesting a second phase.")
        if r.get("unexplained_rings"):
            out.append(f"Ring(s) {', '.join(map(str, r['unexplained_rings']))} are not explained by the candidates.")
        if r.get("ambiguous") and len(r["ranking"]) > 1:
            alt = r["ranking"][1]
            out.append(f"**Ambiguous:** {alt['phase']} fits almost as well (score {alt['score']} vs {r['ranking'][0]['score']}); their lattices differ by only a few percent.")
    if s.get("amorphous") and s["amorphous"].get("halo_d_A"):
        a = s["amorphous"]
        out.append(f"A diffuse halo at d ≈ {a['halo_d_A']:.2f} Å indicates short-range order only; by the Ehrenfest relation the mean nearest-neighbour distance is roughly {a['nearest_neighbour_A_approx']:.2f} Å.")
    elif k == "amorphous":
        out.append("No crystalline phase is indexed.")
    if not s["calibrated"]:
        out.append("The pattern is uncalibrated, so only d-spacing ratios were used: phases with the same structure type but different lattice parameters cannot be told apart.")
    if not s.get("elements"):
        out.append("No elements were given, so the whole built-in library was searched.")
    # next step
    nxt = None
    if s.get("rings", {}).get("ambiguous") or s.get("spots", {}).get("ambiguous"):
        nxt = "add the elements from EDS or record a second pattern at another zone axis to separate the look-alike candidates"
    elif not s["calibrated"]:
        nxt = "calibrate with a gold or aluminium standard at the same camera length"
    elif s.get("rings", {}).get("unexplained_rings"):
        nxt = "add candidate phases (e.g. oxides, carbides) or an experimental CIF for the unexplained rings"
    elif k == "single crystal":
        nxt = "confirm with a second zone axis (tilt) or an experimental CIF"
    ns = (s.get("next_step") or {}).get("summary")
    if ns and (s.get("rings", {}).get("ambiguous") or s.get("spots", {}).get("ambiguous") or k == "single crystal"):
        out.append(f"*Next step:* {ns}")
    elif nxt:
        out.append(f"*Next step:* {nxt}.")
    return "\n\n".join(out)


def llm_explanation(summary: dict, provider: str, model: str, api_key: str | None = None, base_url: str | None = None,
                    timeout: float = 60.0) -> str:
    """Explanation from a language model. ``provider``: 'anthropic' or 'openai' (any OpenAI-compatible endpoint)."""
    payload = json.dumps(_trim(summary), ensure_ascii=False)
    user = f"Analysis results (JSON):\n{payload}\n\nWrite the explanation."
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout)
        msg = client.messages.create(model=model, max_tokens=600, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in msg.content).strip()
    if provider == "openai":
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)  # local servers ignore the key
        client = OpenAI(api_key=key, base_url=base_url or None, timeout=timeout)
        r = client.chat.completions.create(model=model, max_tokens=600,
                                           messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()
    raise ValueError(f"unknown provider {provider}")


def _trim(s: dict) -> dict:
    s = json.loads(json.dumps(s, default=str))
    if "rings" in s:
        s["rings"]["ranking"] = s["rings"]["ranking"][:4]
    if "spots" in s:
        s["spots"]["ranking"] = s["spots"]["ranking"][:4]
    s.pop("timings_s", None)
    return s


def explain(summary: dict, provider: str = "none", model: str = "", api_key: str | None = None, base_url: str | None = None) -> tuple[str, str]:
    """Returns (text, source). Falls back to the template on any error."""
    if provider in ("anthropic", "openai") and model:
        env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if not (api_key or os.environ.get(env) or (provider == "openai" and base_url)):
            name = "Anthropic" if provider == "anthropic" else "the OpenAI-compatible endpoint"
            return (template_explanation(summary) + f"\n\n_(No API key for {name}: paste one under AI explanation in the sidebar. "
                    "Showing the built-in explanation.)_"), "template"
        try:
            return llm_explanation(summary, provider, model, api_key, base_url), f"{provider}:{model}"
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:180]
            return (template_explanation(summary) + f"\n\n_(Language model unavailable: {type(exc).__name__}: {msg} "
                    "Showing the built-in explanation.)_"), "template"
    return template_explanation(summary), "template"
