"""Crystal phases and kinematical electron-diffraction reflections.

* A small built-in library of common phases (experimental lattice parameters)
  so the app works offline and without an API key.
* Optional Materials Project lookup (needs an API key and ``mp-api``).
* Optional user CIF files (e.g. exported from COD or ICSD).
* A vectorised kinematical calculator: d-spacings, structure factors with
  electron scattering factors (Doyle–Turner / Mott–Bethe parameters shipped
  with pymatgen), ring grouping and symmetry-aware labels.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_PARAMS_PATH = None


def _scattering_params() -> dict:
    global _PARAMS_PATH
    import pymatgen.analysis.diffraction.xrd as _xrd

    _PARAMS_PATH = os.path.join(os.path.dirname(_xrd.__file__), "atomic_scattering_params.json")
    with open(_PARAMS_PATH) as fh:
        return json.load(fh)


_SCAT = _scattering_params()

GASES = {"H", "He", "N", "O", "F", "Ne", "Cl", "Ar", "Kr", "Xe", "Rn"}


@dataclass
class Phase:
    name: str
    structure: Structure
    source: str = "built-in"
    source_id: str = ""
    e_above_hull: float | None = None
    note: str = ""
    _sga: SpacegroupAnalyzer | None = field(default=None, repr=False, compare=False)

    @property
    def sga(self) -> SpacegroupAnalyzer:
        if self._sga is None:
            self._sga = SpacegroupAnalyzer(self.structure, symprec=0.01)
        return self._sga

    @property
    def formula(self) -> str:
        return self.structure.composition.reduced_formula

    @property
    def elements(self) -> set[str]:
        return {el.symbol for el in self.structure.composition.elements}

    @property
    def spacegroup(self) -> str:
        try:
            return f"{self.sga.get_space_group_symbol()} ({self.sga.get_space_group_number()})"
        except Exception:
            return "?"

    @property
    def crystal_system(self) -> str:
        try:
            return self.sga.get_crystal_system()
        except Exception:
            return "?"

    @property
    def key(self) -> str:
        return f"{self.name}|{self.source}|{self.source_id}"

    def summary(self) -> dict:
        lat = self.structure.lattice
        return {
            "name": self.name,
            "formula": self.formula,
            "space_group": self.spacegroup,
            "crystal_system": self.crystal_system,
            "a": round(lat.a, 4),
            "b": round(lat.b, 4),
            "c": round(lat.c, 4),
            "alpha": round(lat.alpha, 2),
            "beta": round(lat.beta, 2),
            "gamma": round(lat.gamma, 2),
            "source": self.source,
            "source_id": self.source_id,
            "e_above_hull_eV_atom": self.e_above_hull,
            "note": self.note,
        }


# ----------------------------------------------------------------------------
# Built-in library (room-temperature experimental lattice parameters, rounded)
# ----------------------------------------------------------------------------

def _cubic(sg: str, a: float, species: list[str], coords: list[list[float]]) -> Structure:
    return Structure.from_spacegroup(sg, Lattice.cubic(a), species, coords)


def _hex(sg: str, a: float, c: float, species, coords) -> Structure:
    return Structure.from_spacegroup(sg, Lattice.hexagonal(a, c), species, coords)


def _tet(sg: str, a: float, c: float, species, coords) -> Structure:
    return Structure.from_spacegroup(sg, Lattice.tetragonal(a, c), species, coords)


def _builtin_specs():
    fcc = lambda el, a: (_cubic("Fm-3m", a, [el], [[0, 0, 0]]), "fcc")
    bcc = lambda el, a: (_cubic("Im-3m", a, [el], [[0, 0, 0]]), "bcc")
    rocksalt = lambda m, x, a: (_cubic("Fm-3m", a, [m, x], [[0, 0, 0], [0.5, 0.5, 0.5]]), "rock salt")
    diamond = lambda el, a: (_cubic("F-43m", a, [el, el], [[0, 0, 0], [0.25, 0.25, 0.25]]), "diamond")
    hcp = lambda el, a, c: (_hex("P6_3/mmc", a, c, [el], [[1 / 3, 2 / 3, 0.25]]), "hcp")
    corundum = lambda m, a, c, z, x: (
        _hex("R-3c", a, c, [m, "O"], [[0, 0, z], [x, 0, 0.25]]),
        "corundum",
    )
    specs = [
        ("Au", *fcc("Au", 4.0782)),
        ("Al", *fcc("Al", 4.0495)),
        ("Cu", *fcc("Cu", 3.6149)),
        ("Ni", *fcc("Ni", 3.5240)),
        ("Ag", *fcc("Ag", 4.0857)),
        ("Pt", *fcc("Pt", 3.9242)),
        ("Pd", *fcc("Pd", 3.8907)),
        ("γ-Fe (austenite)", *fcc("Fe", 3.5910)),
        ("α-Fe (ferrite)", *bcc("Fe", 2.8665)),
        ("W", *bcc("W", 3.1652)),
        ("Mo", *bcc("Mo", 3.1470)),
        ("Cr", *bcc("Cr", 2.8846)),
        ("Si", *diamond("Si", 5.4310)),
        ("Ge", *diamond("Ge", 5.6579)),
        ("α-Ti", *hcp("Ti", 2.9508, 4.6855)),
        ("Mg", *hcp("Mg", 3.2094, 5.2108)),
        ("Zn", *hcp("Zn", 2.6649, 4.9468)),
        ("NiO", *rocksalt("Ni", "O", 4.1771)),
        ("MgO", *rocksalt("Mg", "O", 4.2117)),
        ("TiN", *rocksalt("Ti", "N", 4.2417)),
        ("SrTiO₃ (perovskite)", _cubic("Pm-3m", 3.9050, ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]), "perovskite"),
        # cubic above ~120 °C; at room temperature tetragonal (a 3.99, c 4.04 Å), indistinguishable here from the cubic cell
        ("BaTiO₃ (perovskite)", _cubic("Pm-3m", 4.0060, ["Ba", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]), "perovskite"),
        ("CeO₂ (fluorite)", _cubic("Fm-3m", 5.4110, ["Ce", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]]), "fluorite"),
        ("NiAl (B2)", _cubic("Pm-3m", 2.8870, ["Ni", "Al"], [[0, 0, 0], [0.5, 0.5, 0.5]]), "B2 / CsCl"),
        ("Ni₃Al (L1₂)", _cubic("Pm-3m", 3.5670, ["Al", "Ni"], [[0, 0, 0], [0, 0.5, 0.5]]), "L1₂"),
        ("Cu₂O (cuprite)", _cubic("Pn-3m", 4.2696, ["Cu", "O"], [[0.25, 0.25, 0.25], [0, 0, 0]]), "cuprite"),  # pymatgen uses origin choice 1
        (
            "Fe₃O₄ (magnetite)",
            _cubic("Fd-3m", 8.3960, ["Fe", "Fe", "O"], [[0, 0, 0], [0.625, 0.625, 0.625], [0.3799, 0.3799, 0.3799]]),  # origin choice 1
            "spinel",
        ),
        (
            "TiO₂ (rutile)",
            _tet("P4_2/mnm", 4.5937, 2.9587, ["Ti", "O"], [[0, 0, 0], [0.3048, 0.3048, 0]]),
            "rutile",
        ),
        (
            "TiO₂ (anatase)",
            _tet("I4_1/amd", 3.7845, 9.5143, ["Ti", "O"], [[0, 0, 0], [0, 0, 0.2083]]),  # origin choice 1
            "anatase",
        ),
        (
            "ZnO (wurtzite)",
            _hex("P6_3mc", 3.2498, 5.2066, ["Zn", "O"], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.3819]]),
            "wurtzite",
        ),
        ("α-Al₂O₃ (corundum)", *corundum("Al", 4.7589, 12.9910, 0.35216, 0.30624)),
        ("Cr₂O₃ (eskolaite)", *corundum("Cr", 4.9590, 13.5940, 0.3475, 0.3058)),
        ("α-Fe₂O₃ (hematite)", *corundum("Fe", 5.0356, 13.7489, 0.35530, 0.3059)),
    ]
    return specs


N_BUILTIN = 33  # number of built-in phases (checked by a test); lets the app show it without building the library


@lru_cache(maxsize=1)
def builtin_library() -> tuple[Phase, ...]:
    phases = []
    for name, struct, proto in _builtin_specs():
        phases.append(Phase(name=name, structure=struct, source="built-in", source_id=proto, note="experimental lattice parameters"))
    return tuple(phases)


# ----------------------------------------------------------------------------
# Candidate selection
# ----------------------------------------------------------------------------

def parse_elements(text: str) -> list[str]:
    """Parse 'Ni, Al, O' / 'Ni Al O' / 'NiAlO' into a sorted list of valid element symbols."""
    if not text:
        return []
    import re

    tokens = re.findall(r"[A-Z][a-z]?", text)
    out = []
    for t in tokens:
        try:
            Element(t)
            if t not in out:
                out.append(t)
        except Exception:
            continue
    return out


def chemical_subsystems(elements: Iterable[str]) -> list[str]:
    els = sorted(set(elements))
    systems = []
    for n in range(1, len(els) + 1):
        for combo in itertools.combinations(els, n):
            if n == 1 and combo[0] in GASES:
                continue
            systems.append("-".join(combo))
    return systems


def filter_library(phases: Iterable[Phase], elements: Iterable[str] | None) -> list[Phase]:
    phases = list(phases)
    if not elements:
        return phases
    allowed = set(elements)
    return [p for p in phases if p.elements <= allowed]


def fetch_materials_project(elements: Iterable[str], api_key: str, max_e_above_hull: float = 0.1, max_results: int = 80) -> list[Phase]:
    """Fetch all phases in the chemical (sub)systems from the Materials Project.

    Requires ``pip install mp-api`` and a free API key from materialsproject.org.
    """
    from mp_api.client import MPRester  # imported lazily (heavy, optional)

    systems = chemical_subsystems(elements)
    if not systems:
        return []
    phases: list[Phase] = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=systems,
            energy_above_hull=(0, max_e_above_hull),
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull", "symmetry", "theoretical"],
        )
    docs = sorted(docs, key=lambda d: (getattr(d, "energy_above_hull", 0) or 0))[:max_results]
    for d in docs:
        sym = getattr(d, "symmetry", None)
        sg = getattr(sym, "symbol", "") if sym else ""
        theo = getattr(d, "theoretical", None)
        phases.append(
            Phase(
                name=f"{d.formula_pretty} ({sg})" if sg else d.formula_pretty,
                structure=d.structure,
                source="Materials Project",
                source_id=str(d.material_id),
                e_above_hull=float(d.energy_above_hull) if d.energy_above_hull is not None else None,
                note=("DFT structure (lattice ~1-2% large); " + ("theoretical" if theo else "has experimental ICSD entry")),
            )
        )
    return phases


def phase_from_cif(text: str, name: str | None = None) -> Phase:
    s = Structure.from_str(text, fmt="cif")
    return Phase(name=name or s.composition.reduced_formula, structure=s, source="user CIF", source_id=name or "", note="user-supplied CIF")


# ----------------------------------------------------------------------------
# Kinematical reflections
# ----------------------------------------------------------------------------

@dataclass
class Reflections:
    hkl: np.ndarray  # (N, 3) int
    gvec: np.ndarray  # (N, 3) cartesian reciprocal vectors, Å⁻¹ (no 2π)
    g: np.ndarray  # (N,) |g|, Å⁻¹
    F2: np.ndarray  # (N,) |F|² (Å²)
    allowed: np.ndarray  # (N,) bool, kinematically allowed (|F|² > threshold)


@dataclass
class RingLine:
    g: float
    d: float
    intensity: float  # relative 0-100 (ring/azimuthal-average weighting)
    label: str  # e.g. "111" or "221/300"
    multiplicity: int


def electron_f(symbol: str, s2: np.ndarray) -> np.ndarray:
    """Electron scattering factor f_e(s) in Å with s = sinθ/λ = g/2 (s2 in Å⁻²)."""
    coeffs = np.asarray(_SCAT.get(symbol) or _SCAT.get("Si"))
    return np.sum(coeffs[:, 0][None, :] * np.exp(-coeffs[:, 1][None, :] * s2[:, None]), axis=1)


_REFL_CACHE: dict[tuple, Reflections] = {}


def reflections(phase: Phase, g_max: float = 1.4, debye_waller_B: float = 0.5) -> Reflections:
    key = (phase.key, round(g_max, 3))
    if key in _REFL_CACHE:
        return _REFL_CACHE[key]
    s = phase.structure
    recip = s.lattice.reciprocal_lattice_crystallographic.matrix  # rows a*, b*, c*
    lens = np.linalg.norm(recip, axis=1)
    # bound on indices from |g| <= g_max (use reciprocal of direct-lattice spacing bound)
    direct = s.lattice.matrix
    hmax = [int(math.ceil(g_max * np.linalg.norm(direct[i]))) + 1 for i in range(3)]
    rng = [np.arange(-m, m + 1) for m in hmax]
    H, K, L = np.meshgrid(*rng, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    gvec = hkl @ recip
    g = np.linalg.norm(gvec, axis=1)
    keep = g <= g_max + 1e-9
    hkl, gvec, g = hkl[keep], gvec[keep], g[keep]
    s2 = (g / 2.0) ** 2
    F = np.zeros(len(hkl), dtype=complex)
    for site in s:
        x = site.frac_coords
        phase_fac = np.exp(2j * np.pi * (hkl @ x))
        for sp, occ in site.species.items():
            sym = sp.symbol if hasattr(sp, "symbol") else str(sp)
            F += occ * electron_f(sym, s2) * phase_fac
    F2 = np.abs(F) ** 2 * np.exp(-2 * debye_waller_B * s2)
    thr = 1e-4 * F2.max() if len(F2) else 0
    allowed = F2 > thr
    out = Reflections(hkl=hkl.astype(int), gvec=gvec, g=g, F2=F2, allowed=allowed)
    _REFL_CACHE[key] = out
    _ = lens
    return out


_OPS_CACHE: dict[str, list] = {}


def _point_ops_frac(phase: Phase) -> list[np.ndarray]:
    if phase.key in _OPS_CACHE:
        return _OPS_CACHE[phase.key]
    try:
        ops = phase.sga.get_point_group_operations(cartesian=False)
        mats = [np.round(op.rotation_matrix).astype(int) for op in ops]
    except Exception:
        mats = [np.eye(3, dtype=int)]
    # include inversion (Friedel's law) for diffraction symmetry
    mats = mats + [-m for m in mats]
    uniq = {m.tobytes(): m for m in mats}
    mats = list(uniq.values())
    _OPS_CACHE[phase.key] = mats
    return mats


def canonical_hkl(phase: Phase, hkl: np.ndarray) -> tuple[int, int, int]:
    """Symmetry-canonical representative of a Miller index (Laue class)."""
    best = None
    for W in _point_ops_frac(phase):
        h = tuple(int(v) for v in (np.asarray(hkl) @ W))
        key = (sum(1 for v in h if v >= 0), h)
        if best is None or key > best[0]:
            best = (key, h)
    return best[1]


def canonical_uvw(phase: Phase, uvw: np.ndarray) -> tuple[int, int, int]:
    return canonical_uvw_op(phase, uvw)[0]


def canonical_uvw_op(phase: Phase, uvw: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray]:
    """Canonical zone-axis representative and the point operation W with W @ uvw = canonical.
    Miller indices transform as h' = h @ inv(W) so that the zone law h·u = 0 is preserved."""
    best = None
    for W in _point_ops_frac(phase):
        u = tuple(int(v) for v in (W @ np.asarray(uvw)))
        key = (sum(1 for v in u if v >= 0), u)
        if best is None or key > best[0]:
            best = (key, u, W)
    return best[1], best[2]


def fmt_index(t: Iterable[int], bracket: str = "()") -> str:
    parts = []
    for v in t:
        v = int(v)
        parts.append(f"{abs(v)}̅" if v < 0 else str(v))
    s = "".join(parts) if all(abs(int(v)) < 10 for v in t) else " ".join(parts)
    return f"{bracket[0]}{s}{bracket[1]}"


def fmt_index_plain(t: Iterable[int]) -> str:
    return "".join(f"-{abs(int(v))}" if int(v) < 0 else str(int(v)) for v in t)


_RING_CACHE: dict[tuple, list] = {}


def _families(phase: Phase, hkls: np.ndarray) -> list[tuple[int, int, int]]:
    """Distinct symmetry families among a set of Miller indices (canonical labels)."""
    ops = np.array(_point_ops_frac(phase))  # (n_ops, 3, 3)
    remaining = {tuple(int(v) for v in h) for h in hkls}
    fams = []
    while remaining:
        h = np.array(next(iter(remaining)))
        orbit = np.einsum("j,ojk->ok", h, ops)
        orbit_set = {tuple(int(v) for v in o) for o in orbit}
        remaining -= orbit_set
        # canonical: most non-negative entries, then lexicographically largest
        best = max(orbit_set, key=lambda t: (sum(1 for v in t if v >= 0), t))
        fams.append(best)
    fams.sort(key=lambda t: (-sum(abs(v) for v in t), t), reverse=True)
    return fams


def ring_lines(phase: Phase, g_max: float = 1.4, min_rel_intensity: float = 0.5) -> list[RingLine]:
    """Powder-ring lines (grouped by d-spacing) with relative intensities."""
    key = (phase.key, round(g_max, 3), min_rel_intensity)
    if key in _RING_CACHE:
        return _RING_CACHE[key]
    ref = reflections(phase, g_max)
    m = ref.allowed
    hkl, g, F2 = ref.hkl[m], ref.g[m], ref.F2[m]
    if len(g) == 0:
        return []
    order = np.argsort(g)
    hkl, g, F2 = hkl[order], g[order], F2[order]
    groups = []
    start = 0
    for i in range(1, len(g) + 1):
        if i == len(g) or (g[i] - g[start]) / g[start] > 2e-4:
            groups.append((start, i))
            start = i
    lines = []
    for a, b in groups:
        gg = float(np.mean(g[a:b]))
        inten = float(np.sum(F2[a:b]) / max(gg, 1e-6))  # azimuthal average ~ m|F|²/g
        fams = _families(phase, hkl[a:b])
        label = "/".join(fmt_index_plain(f) for f in fams[:3])
        lines.append(RingLine(g=gg, d=1 / gg, intensity=inten, label=label, multiplicity=b - a))
    mx = max(l.intensity for l in lines)
    for l in lines:
        l.intensity = 100 * l.intensity / mx
    out = [l for l in lines if l.intensity >= min_rel_intensity]
    _RING_CACHE[key] = out
    return out


def lattice_summary_text(phase: Phase) -> str:
    lat = phase.structure.lattice
    cs = phase.crystal_system
    if cs == "cubic":
        return f"a = {lat.a:.4f} Å"
    if cs in ("hexagonal", "trigonal", "tetragonal"):
        return f"a = {lat.a:.4f} Å, c = {lat.c:.4f} Å"
    return f"a = {lat.a:.3f}, b = {lat.b:.3f}, c = {lat.c:.3f} Å"


def composition_ok(formula: str) -> bool:
    try:
        Composition(formula)
        return True
    except Exception:
        return False
