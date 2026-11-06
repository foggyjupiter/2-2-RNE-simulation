"""Impulse responses from source force to microphone (spec 6.1, step 2).

Default ("measured"): a random realisation of the statistical response model
measured from the first-study recordings (see :mod:`gransim.acoustics.assets`):

    h(t) = sqrt(drr/(1+drr)) · d(t) + sqrt(1/(1+drr)) · r(t)

with d a short click (random phase, measured direct spectrum, 0.3 ms decay) and r a
ring/reverb tail (noise shaped by the measured tail spectrum × exp(−t/τ)). Each has
unit energy, so h has unit energy. Every channel (bottom / top / side / pp) gets its
own realisation, as different walls and positions do not ring identically.

Fallback ("synthetic"): sum of damped sinusoids (frequency, Q, amplitude) from the
config, used when no measured model is available.

No N-dependent change of the response is applied: none has been measured yet
(spec 6.1 asks for N-wise measured responses; until then the response is N-independent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from gransim.config import AcousticConfig

ASSETS = Path(__file__).resolve().parents[3] / "assets"
DIRECT_DECAY_S = 3.0e-4  # 1/e decay of the click envelope measured on isolated impacts


def load_response_model(cfg: AcousticConfig) -> dict | None:
    p = Path(cfg.response_file) if cfg.response_file else ASSETS / "impulse_response" / "response_model.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files} | {"path": str(p)}


def _shaped_noise(psd_f: np.ndarray, psd: np.ndarray, n: int, fs: float,
                  rng: np.random.Generator) -> np.ndarray:
    """White noise of length n filtered (zero phase) to the power spectrum psd(psd_f)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    mag = np.sqrt(np.interp(f, psd_f, psd, right=0.0))
    return np.fft.irfft(X * mag, n)


def _unit(x: np.ndarray) -> np.ndarray:
    e = np.sqrt(np.sum(x * x))
    return x / e if e > 0 else x


def measured_ir(model: dict, cfg: AcousticConfig, rng: np.random.Generator) -> np.ndarray:
    fs = cfg.fs
    n = int(cfg.ir_length_s * fs)
    t = np.arange(n) / fs
    f = model["f"]
    tau = float(cfg.ring_tau or model["tau"])
    drr = float(cfg.drr or model["drr"])
    d = _shaped_noise(f, model["direct_psd"], n, fs, rng) * np.exp(-t / DIRECT_DECAY_S)
    r = _shaped_noise(f, model["tail_psd"], n, fs, rng) * np.exp(-t / tau)
    ramp = int(0.0005 * fs)
    r[:ramp] *= np.linspace(0.0, 1.0, ramp)
    return np.sqrt(drr / (1 + drr)) * _unit(d) + np.sqrt(1 / (1 + drr)) * _unit(r)


def synthetic_ir(cfg: AcousticConfig, rng: np.random.Generator) -> np.ndarray:
    fs = cfg.fs
    n = int(cfg.ir_length_s * fs)
    t = np.arange(n) / fs
    h = np.zeros(n)
    for f0, Q, a in cfg.synthetic_modes:
        if f0 >= fs / 2:
            continue
        h += a * np.exp(-np.pi * f0 / Q * t) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    return _unit(h)


def draw_responses(cfg: AcousticConfig, channels: list[str], rng: np.random.Generator):
    """One impulse response per channel. Returns (dict name → ir, info dict)."""
    model = load_response_model(cfg) if cfg.response == "measured" else None
    if cfg.response == "measured" and model is None:
        raise FileNotFoundError("measured response model not found; build it with "
                                "`gransim build-assets` or set acoustics.response=synthetic")
    out = {}
    for c in channels:
        out[c] = measured_ir(model, cfg, rng) if model is not None else synthetic_ir(cfg, rng)
    info = {"response": cfg.response,
            "response_file": model["path"] if model is not None else None,
            "drr": float(cfg.drr or model["drr"]) if model is not None else None,
            "ring_tau_s": float(cfg.ring_tau or model["tau"]) if model is not None else None}
    return out, info


def apply_response(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution, truncated to len(x)."""
    return signal.fftconvolve(x, h)[: len(x)]
