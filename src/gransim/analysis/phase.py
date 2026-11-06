"""Ground-truth phase indicators from the simulated state (spec 9.1).

In the experiment the phase was judged by eye; here positions, velocities and
every contact are known, so the phase can be measured directly. All
indicators are averages over the recording window.

| indicator              | solid-like (bed moves as a block) | fluid-like            |
|------------------------|-----------------------------------|-----------------------|
| ``collectivity``       | → 1 (KE is centre-of-mass motion) | small                 |
| ``phase_concentration``| → 1 (impacts at one phase)        | small                 |
| ``free_flight``        | small (always touching)           | larger                |
| ``sustained_fraction`` | large                             | small                 |
| ``height_mean/std``    | low, compact                      | high, spread out      |
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from gransim.events.tracker import IMPACT, SUSTAINED


@njit(cache=True)
def _union_length(pid, ts, te, n_particles):
    """Total length of the union of [ts, te) intervals per particle (input sorted by pid, ts)."""
    out = np.zeros(n_particles)
    cur = -1
    a = 0.0
    b = 0.0
    for k in range(pid.size):
        p = pid[k]
        if p != cur:
            if cur >= 0:
                out[cur] += b - a
            cur = p
            a = ts[k]
            b = te[k]
        elif ts[k] > b:
            out[cur] += b - a
            a = ts[k]
            b = te[k]
        elif te[k] > b:
            b = te[k]
    if cur >= 0:
        out[cur] += b - a
    return out


def contact_time_fraction(events: np.ndarray, n_particles: int, duration: float) -> np.ndarray:
    """Per-particle fraction of the window spent touching anything (particle or wall)."""
    pp = events["j"] >= 0
    pid = np.concatenate([events["i"], events["j"][pp]]).astype(np.int64)
    ts = np.concatenate([events["t_start"], events["t_start"][pp]])
    te = np.concatenate([events["t_end"], events["t_end"][pp]])
    o = np.lexsort((ts, pid))
    return _union_length(pid[o], ts[o], te[o], n_particles) / duration


def phase_indicators(events: np.ndarray, diagnostics: dict, n_particles: int,
                     record_window: tuple[float, float]) -> dict:
    """Compute the ground-truth indicators over the recording window."""
    t0, t1 = record_window
    T = t1 - t0
    d = diagnostics
    m = (d["t"] >= t0) & (d["t"] <= t1)
    ke_rel = d["ke_rel"][m].mean()
    out: dict = {
        "height_mean": float(d["mean_height"][m].mean()),
        "height_std": float(d["height_std"][m].mean()),
        "height_temporal_std": float(d["mean_height"][m].std()),
        "collectivity": float(d["ke_com"][m].mean() / ke_rel) if ke_rel > 0 else 1.0,
        "ke_rel_per_particle": float(ke_rel / max(n_particles, 1)),
    }
    if n_particles:
        frac = contact_time_fraction(events, n_particles, T)
        out["free_flight"] = float(1.0 - frac.mean())
    else:
        out["free_flight"] = 1.0
    dur = events["duration"]
    tot = dur.sum()
    out["sustained_fraction"] = float(dur[events["contact_type"] == SUSTAINED].sum() / tot) if tot > 0 else 0.0
    imp = events[events["contact_type"] == IMPACT]
    if imp.size:
        z = np.exp(2j * np.pi * imp["phase"])
        out["phase_concentration"] = float(abs(z.mean()))
        w = imp["J_n"]
        out["phase_concentration_impulse"] = float(abs((w * z).sum() / w.sum()))
        out["mean_impact_phase"] = float(np.mod(np.angle(z.mean()) / (2 * np.pi), 1.0))
        out["impact_rate"] = float(imp.size / T)
        out["impact_rate_per_particle"] = float(imp.size / T / max(n_particles, 1))
    else:
        out.update(phase_concentration=0.0, phase_concentration_impulse=0.0,
                   mean_impact_phase=math.nan, impact_rate=0.0, impact_rate_per_particle=0.0)
    return out
