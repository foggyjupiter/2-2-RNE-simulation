"""Hertz contact time for the acoustic pulse width (spec 6.2) and a particle-mode check.

The acoustic contact time ``tau_c`` is set from the *real* material properties,
independently of the (relaxed) stiffness used in the dynamics: the dynamics gives
when and how hard each impact is (impulse), the acoustics gives over how long
that impulse is delivered.
"""

from __future__ import annotations

import math

import numpy as np

from gransim.config import Material, SimConfig

HERTZ_TIME_COEFF = 2.868  # τ = 2.868 (m_eff² / (R_eff E*² v))^(1/5), undamped Hertz impact


def effective_modulus(a: Material, b: Material) -> float:
    """E* [Pa] with 1/E* = (1-ν_a²)/E_a + (1-ν_b²)/E_b."""
    return 1.0 / ((1 - a.poisson_ratio ** 2) / a.youngs_modulus
                  + (1 - b.poisson_ratio ** 2) / b.youngs_modulus)


def hertz_contact_time(m_eff, R_eff, E_eff, v, v_floor: float = 1e-3, tau_max: float = 5e-3):
    """Hertz contact duration [s] (vectorised).

    m_eff [kg], R_eff [m], E_eff [Pa] (E*), v [m/s] normal impact speed. Speeds below
    ``v_floor`` are clipped (τ diverges as v → 0) and the result is capped at ``tau_max``.
    """
    v = np.maximum(np.abs(np.asarray(v, dtype=float)), v_floor)
    tau = HERTZ_TIME_COEFF * (np.asarray(m_eff) ** 2 / (np.asarray(R_eff) * E_eff ** 2 * v)) ** 0.2
    return np.minimum(tau, tau_max)


def event_contact_times(events: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Acoustic τ_c [s] per event from real moduli (particle–particle or particle–wall)."""
    p, c = cfg.particles.material, cfg.container.material
    E_pp = effective_modulus(p, p)
    E_pw = effective_modulus(p, c)
    E = np.where(events["kind"] == 1, E_pw, E_pp)
    return hertz_contact_time(events["m_eff"], events["R_eff"], E, events["v_rel_n"])


def sphere_lowest_mode_hz(diameter: float, material: Material, density: float) -> float:
    """Approximate lowest free-vibration frequency of a solid sphere [Hz] ≈ 0.84 c_t / d.

    Used only to confirm that particle self-vibration lies above the output band
    (for 6 mm plastic spheres it is ~100 kHz), so it is not modelled.
    """
    c_t = math.sqrt(material.youngs_modulus / (2 * (1 + material.poisson_ratio) * density))
    return 0.84 * c_t / diameter
