"""Clip-level analysis: the shaking frequency is estimated per clip (the real shaker drifts
6.04-6.64 Hz around its nominal 6.2 Hz), so the period-folded profile is not smeared."""

import numpy as np

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
