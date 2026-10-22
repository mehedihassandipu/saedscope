"""Basic electron-diffraction physics helpers.

Conventions used throughout SAEDscope
-------------------------------------
* Reciprocal-space lengths are ``g = 1/d`` in inverse angstroms (no 2*pi).
* Image calibration ``c`` is "inverse angstroms per pixel" (Å⁻¹/px), so a ring
  of radius ``R`` pixels has ``g = R * c`` and ``d = 1 / (R * c)``.
"""

from __future__ import annotations

import math

# Physical constants (CODATA 2018)
H = 6.62607015e-34  # Planck constant, J s
M_E = 9.1093837015e-31  # electron rest mass, kg
E_CHARGE = 1.602176634e-19  # elementary charge, C
C_LIGHT = 299792458.0  # speed of light, m/s

# Ehrenfest relation for amorphous materials: r_nn ≈ 1.23 * d_halo
EHRENFEST_FACTOR = 1.23


def electron_wavelength_A(voltage_kV: float) -> float:
    """Relativistic electron wavelength in angstroms for an accelerating voltage in kV."""
    if voltage_kV <= 0:
        raise ValueError("voltage must be positive")
    v = voltage_kV * 1e3
    p = math.sqrt(2 * M_E * E_CHARGE * v * (1 + E_CHARGE * v / (2 * M_E * C_LIGHT**2)))
    return H / p * 1e10


def calibration_from_camera_length(voltage_kV: float, camera_length_mm: float, pixel_size_um: float) -> float:
    """Å⁻¹ per pixel from voltage, camera length and (binned) detector pixel size.

    R = λ L / d  →  g = 1/d = R / (λ L)  →  c = pixel_size / (λ L).
    """
    lam = electron_wavelength_A(voltage_kV)
    pixel_mm = pixel_size_um * 1e-3
    return pixel_mm / (lam * camera_length_mm)


def camera_constant_A_px(calibration_inv_A_per_px: float) -> float:
    """Camera constant λL expressed in Å·px (d = λL / R)."""
    return 1.0 / calibration_inv_A_per_px


def nearest_neighbour_from_halo(d_halo_A: float) -> float:
    """Approximate mean nearest-neighbour distance from an amorphous halo (Ehrenfest)."""
    return EHRENFEST_FACTOR * d_halo_A


def scale_units_to_inv_A(scale: float, units: str | None) -> float | None:
    """Convert a reciprocal-space pixel scale with units to Å⁻¹/px.

    Returns None when the units are not reciprocal-space (e.g. real-space nm),
    which means the file is not calibrated in reciprocal space.
    """
    if scale is None or scale <= 0 or units is None:
        return None
    u = units.strip().replace(" ", "").replace("^-1", "⁻¹").replace("-1", "⁻¹").lower()
    nm_inv = {"1/nm", "nm⁻¹", "1nm", "nm-1", "1/nanometer"}
    a_inv = {"1/å", "å⁻¹", "1/a", "a⁻¹", "1/angstrom", "1/ang"}
    m_inv = {"1/m", "m⁻¹"}
    if u in nm_inv:
        return scale / 10.0
    if u in a_inv:
        return scale
    if u in m_inv:
        return scale * 1e-10
    if u in {"1/pm", "pm⁻¹"}:
        return scale * 100.0
    return None
