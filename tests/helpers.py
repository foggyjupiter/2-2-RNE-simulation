"""Shared helpers for the validation tests."""

from __future__ import annotations

import numpy as np

from gransim.config import SimConfig
from gransim.dynamics.engine import A_DCORR, A_DN, A_DT, A_EN, A_ET, A_WK, Simulation


def still_config(**kw) -> SimConfig:
    """Config with the container at rest (free mode, A = 0). Keyword args are top-level sections."""
    base = {"excitation": {"mode": "free", "amplitude": 0.0, "frequency": 10.0}}
    base.update(kw)
    return SimConfig.model_validate(base)


def total_energy(sim: Simulation) -> float:
    """KE (trans + rot) + gravitational PE + stored elastic energy [J]."""
    ke_t, ke_r, pe = sim.energies()
    return ke_t + ke_r + pe + sim.acc[A_EN] + sim.acc[A_ET]


def dissipated(sim: Simulation) -> float:
    """Net energy removed by contacts (dissipation minus k_t-variation work) [J]."""
    return sim.acc[A_DN] + sim.acc[A_DT] + sim.acc[A_DCORR] - sim.acc[A_WK]


def energy_residual(diag: dict) -> np.ndarray:
    """ΔE_total - (W_in + W_k - D) along a diagnostics series [J]."""
    E = diag["ke_trans"] + diag["ke_rot"] + diag["pe"] + diag["e_n"] + diag["e_t"]
    D = diag["d_n"] + diag["d_t"] + diag["d_corr"]
    return (E - E[0]) - (diag["w_in"] + diag["w_k"] - D)


def run_until_separated(sim: Simulation, in_contact, max_steps: int = 2_000_000):
    """Step one at a time; return (velocities just before first contact, just after it ends).

    ``in_contact(sim) -> bool`` decides whether the bodies currently overlap.
    """
    v_before = None
    started = False
    for _ in range(max_steps):
        if not started:
            v_prev = (sim.v.copy(), sim.w.copy())
        sim.step(1)
        c = in_contact(sim)
        if c and not started:
            started = True
            v_before = v_prev
        elif started and not c:
            return v_before, (sim.v.copy(), sim.w.copy())
    raise RuntimeError("contact did not finish")
