"""Acoustic order parameters of the first study (spec 9.1), computed from audio.

ψ1  energy ratio of the first to the second half of each shaking period
    (needs the excitation phase; with no phase reference the best rotation is used)
ψ2  1 − normalised spectral Shannon entropy
ψ3  excess kurtosis of the waveform (K − 3) — the one the first study relied on
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def psi1(y: np.ndarray, fs: float, f_shake: float, phase0: float | None = None,
         n_bins: int = 64) -> float:
    """First-half / second-half energy ratio over the shaking period.

    phase0: excitation phase of sample 0 (0..1). None = unknown → the rotation that
    maximises the ratio is used (as a phase-free variant).
    """
    prof = folded_energy(y, fs, f_shake, n_bins, phase0 or 0.0)
    h = n_bins // 2
    if phase0 is not None:
        return float(prof[:h].sum() / max(prof[h:].sum(), 1e-30))
    best = 0.0
    for r in range(n_bins):
        p = np.roll(prof, -r)
        best = max(best, p[:h].sum() / max(p[h:].sum(), 1e-30))
    return float(best)


def psi2(y: np.ndarray) -> float:
    """1 − H/H_max of the normalised power spectrum."""
    P = np.abs(np.fft.rfft(y)) ** 2
    P = P / P.sum()
    nz = P[P > 0]
    H = -np.sum(nz * np.log(nz))
    return float(1.0 - H / np.log(len(P)))


def psi3(y: np.ndarray) -> float:
    """Excess kurtosis K − 3 of the waveform."""
    return float(stats.kurtosis(y, fisher=True))


def folded_energy(y: np.ndarray, fs: float, f_shake: float, n_bins: int = 64,
                  phase0: float = 0.0) -> np.ndarray:
    """Mean energy per excitation-phase bin (phase of each sample = phase0 + f t, mod 1)."""
    ph = (phase0 + f_shake * np.arange(len(y)) / fs) % 1.0
    b = np.minimum((ph * n_bins).astype(int), n_bins - 1)
    e = np.bincount(b, weights=y * y, minlength=n_bins)
    c = np.bincount(b, minlength=n_bins)
    return e / np.maximum(c, 1)
