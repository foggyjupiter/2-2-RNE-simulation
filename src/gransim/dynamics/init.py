"""Particle property arrays and overlap-free initial placement."""

from __future__ import annotations

import math

import numpy as np

from gransim.config import SimConfig


def particle_properties(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle radius [m] and mass [kg] arrays (supports polydispersity).

    Density is held at the value implied by (mass, diameter); diameters are
    drawn from a normal distribution truncated at ±3σ when diameter_rel_std > 0.
    """
    p = cfg.particles
    n = p.count
    d = np.full(n, p.diameter)
    if p.diameter_rel_std > 0 and n > 0:
        z = np.clip(rng.standard_normal(n), -3.0, 3.0)
        d = p.diameter * (1.0 + p.diameter_rel_std * z)
    mass = p.density * math.pi / 6.0 * d ** 3
    return 0.5 * d, mass


def lattice_positions(cfg: SimConfig, radius: np.ndarray, rng: np.random.Generator,
                      gap_frac: float = 0.05, jitter_frac: float = 0.02) -> np.ndarray:
    """Place particles on a jittered simple-cubic lattice from the bottom up.

    Spacing is (1 + gap_frac) × max diameter; random jitter keeps the packing
    from staying perfectly crystalline. Raises if the box cannot hold N.
    """
    n = radius.size
    Lx, Ly, Lz = cfg.container.size
    dmax = 2.0 * (radius.max() if n else cfg.particles.diameter / 2)
    s = dmax * (1.0 + gap_frac)
    nx, ny, nz = (int(math.floor((L - dmax) / s)) + 1 for L in (Lx, Ly, Lz))
    if nx * ny * nz < n:
        raise ValueError(f"cannot place {n} particles on a lattice in the container "
                         f"(capacity {nx * ny * nz})")
    pts = []
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                pts.append((dmax / 2 + i * s, dmax / 2 + j * s, dmax / 2 + k * s))
                if len(pts) == n:
                    break
            if len(pts) == n:
                break
        if len(pts) == n:
            break
    x = np.array(pts, dtype=np.float64).reshape(n, 3)
    if n:
        # centre the lattice footprint horizontally
        x[:, 0] += (Lx - ((nx - 1) * s + dmax)) / 2
        x[:, 1] += (Ly - ((ny - 1) * s + dmax)) / 2
        jitter = jitter_frac * dmax * (rng.random((n, 3)) - 0.5) * 2
        jitter[:, 2] = np.abs(jitter[:, 2])
        x += jitter
        x[:, 0] = np.clip(x[:, 0], radius, Lx - radius)
        x[:, 1] = np.clip(x[:, 1], radius, Ly - radius)
        x[:, 2] = np.maximum(x[:, 2], radius)
    return x
