"""Clip-level analysis: the shaking frequency is estimated per clip (the real shaker drifts
6.04-6.64 Hz around its nominal 6.2 Hz), so the period-folded profile is not smeared."""

import numpy as np
import pytest

from gransim.analysis.order_params import estimate_shake_frequency, folded_energy
from gransim.analysis.sim2real import features

FS = 48000


def clicks(f, dur=5.0, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(dur * FS))
    for t in np.arange(0.01, dur, 1 / f):
        i = int(t * FS)
        x[i:i + 300] += rng.standard_normal(300) * 0.5
    return x


def test_shake_frequency_estimate():
    for f in (6.04, 6.2, 6.5, 6.64):
        assert abs(estimate_shake_frequency(clicks(f), FS, 6.2) - f) < 0.02


def test_fold_uses_the_clip_frequency():
    x = clicks(6.5)
    smeared = folded_energy(x, FS, 6.2)
    d = features(x, FS, 6.2)
    assert abs(d["f_shake_est"] - 6.5) < 0.02
    assert d["fold_peak_to_mean"] > 2 * smeared.max() / smeared.mean()


def test_crank_slider_motion():
    """λ = 0 is the pure sine; λ > 0 gives (1+λ)Aω² downward at the top and (1−λ)Aω² upward
    at the bottom, with u = dz/dt."""
    import math

    from gransim.dynamics.walls import container_motion
    A, w, lam = 0.0134, 2 * math.pi * 6.2, 0.2
    t = np.linspace(1.0, 1.0 + 2 * math.pi / w, 20001)
    z0 = np.array([container_motion(x, A, w, 0.0)[0] for x in t])
    assert np.allclose(z0, A * np.sin(w * t), atol=1e-12)
    z = np.array([container_motion(x, A, w, 0.0, lam)[0] for x in t])
    u = np.array([container_motion(x, A, w, 0.0, lam)[1] for x in t])
    dt = t[1] - t[0]
    assert np.allclose(np.gradient(z, dt)[5:-5], u[5:-5], rtol=0, atol=1e-4 * A * w)
    acc = np.gradient(u, dt)[5:-5]
    assert acc.min() == pytest.approx(-(1 + lam) * A * w ** 2, rel=2e-3)
    assert acc.max() == pytest.approx((1 - lam) * A * w ** 2, rel=2e-2)
