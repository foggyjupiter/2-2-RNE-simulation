"""Impact events -> force pulse train at the output rate (spec 6.1, step 1 (ii)).

Each impact deposits a pulse of area J_n (its normal impulse) and width τ_c.
Pulses are evaluated *analytically* at their exact (fractional) start time on a
fine grid (``oversample`` × output rate), then decimated to the output rate with
an anti-aliasing FIR. So a pulse narrower than one output sample is still
represented correctly, and its start time is never rounded to the output grid.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import signal

SHAPES = {"half_sine": 0, "hertz": 1, "delta": 2}
# ∫_0^1 sin(πu)^1.5 du  (normalises the Hertz-like profile to unit area)
_HERTZ_AREA = 0.5 * math.gamma(1.25) * math.gamma(0.5) / math.gamma(1.75)


@njit(cache=True)
def _deposit(buf, t, J, tau, t0, fs_fine, shape, area_hertz):
    n = buf.shape[0]
    dtf = 1.0 / fs_fine
    for k in range(t.shape[0]):
        ts = t[k] - t0
        w = tau[k]
        if shape == 2 or w * fs_fine < 4.0:
            # delta (or a pulse shorter than 4 fine samples): linear split between two samples
            u = ts * fs_fine
            i0 = int(math.floor(u))
            fr = u - i0
            if 0 <= i0 < n:
                buf[i0] += J[k] * (1.0 - fr) * fs_fine
            if 0 <= i0 + 1 < n:
                buf[i0 + 1] += J[k] * fr * fs_fine
            continue
        i_start = int(math.ceil(ts * fs_fine))
        i_end = int(math.floor((ts + w) * fs_fine))
        if i_end < 0 or i_start >= n:
            continue
        # amplitude so the continuous pulse has area J
        if shape == 0:
            amp = J[k] * math.pi / (2.0 * w)
        else:
            amp = J[k] / (area_hertz * w)
        # discrete-sum correction so Σ samples · dt equals J exactly
        s = 0.0
        for i in range(i_start, i_end + 1):
            u = (i * dtf - ts) / w
            if shape == 0:
                s += math.sin(math.pi * u)
            else:
                s += math.sin(math.pi * u) ** 1.5
        if s <= 0.0:
            continue
        scale = J[k] * fs_fine / s
        for i in range(max(i_start, 0), min(i_end + 1, n)):
            u = (i * dtf - ts) / w
            if shape == 0:
                buf[i] += scale * math.sin(math.pi * u)
            else:
                buf[i] += scale * math.sin(math.pi * u) ** 1.5


def render_pulses(t: np.ndarray, J: np.ndarray, tau: np.ndarray, t0: float, n_out: int,
                  fs_out: float, shape: str = "half_sine", oversample: int = 16) -> np.ndarray:
    """Force signal [N] at fs_out made of one pulse per event.

    t: pulse start times [s]; J: impulses [N s]; tau: widths [s]; t0: time of output
    sample 0 [s]. The sum of output samples / fs_out equals ΣJ (pulses fully inside).
    """
    if shape not in SHAPES:
        raise ValueError(f"pulse shape must be one of {list(SHAPES)}, got {shape!r}")
    fs_fine = fs_out * oversample
    pad = 64 * oversample
    buf = np.zeros(n_out * oversample + 2 * pad)
    if len(t):
        _deposit(buf, np.asarray(t, float), np.asarray(J, float), np.asarray(tau, float),
                 t0 - pad / fs_fine, fs_fine, SHAPES[shape], _HERTZ_AREA)
    out = signal.resample_poly(buf, 1, oversample, window=("kaiser", 10.0))
    return out[pad // oversample: pad // oversample + n_out]
