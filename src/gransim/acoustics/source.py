"""Source signals (spec 6.1, step 1).

Channels: ``bottom``, ``top``, ``side`` (four side walls), ``pp`` (particle-particle).
Each wall channel is the sum of

  (i)  the wall's normal force time series, low-passed below ``band_split_hz``
       (sustained contacts, collective hits, load fluctuations), and
  (ii) one pulse per *impact* event on that wall (area J_n, width τ_c), high-passed
       above ``band_split_hz`` (restores the high frequencies the relaxed stiffness
       smears out).

The low-pass and high-pass are exactly complementary zero-phase filters
(L(f) + H(f) = 1). For an isolated impact the wall force integral equals J_n, so the
summed source carries the impulse exactly once (no double counting): (i) holds its
DC/low band, (ii) only the band above the split. Particle-particle impacts are not
wall forces; they enter as pulses (high band) scaled by ``pp_gain``.
"""

from __future__ import annotations

import numpy as np
from gransim.acoustics.contact_time import event_contact_times
from gransim.acoustics.pulse import render_pulses
from gransim.config import SimConfig

CHANNELS = ("bottom", "top", "side", "pp")
WALL_TO_CHANNEL = {0: "bottom", 1: "top", 2: "side", 3: "side", 4: "side", 5: "side"}


def lowpass_response(f: np.ndarray, fc: float) -> np.ndarray:
    """Zero-phase raised-cosine low-pass: 1 below fc/√2, 0 above fc·√2."""
    lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
    x = np.clip((np.log(np.maximum(f, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def split(x: np.ndarray, fs: float, fc: float, part: str) -> np.ndarray:
    """Low ('low') or complementary high ('high') band of x, zero phase (padded FFT)."""
    n = len(x)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, m)
    L = lowpass_response(np.fft.rfftfreq(m, 1 / fs), fc)
    return np.fft.irfft(X * (L if part == "low" else 1.0 - L), m)[:n]


PLATE_AXES = {0: (0, 1), 1: (0, 1), 2: (1, 2), 3: (1, 2), 4: (0, 2), 5: (0, 2)}   # in-plane axes


def radiation_weights(imp: np.ndarray, cfg: SimConfig, seed: int) -> np.ndarray:
    """Per-impact radiation efficiency (mean ≈ 1): plate-mode position weighting for wall
    impacts and a log-normal impact-to-impact factor (both off by default)."""
    ac = cfg.acoustics
    w = np.ones(len(imp))
    if ac.position_weighting > 0:
        L = cfg.container.size
        pos = np.stack([imp["x"], imp["y"], imp["z"]], 1)
        for wid, (a, b) in PLATE_AXES.items():
            m = (imp["kind"] == 1) & (imp["wall_id"] == wid)
            if m.any():
                u = np.clip(pos[m, a] / L[a], 0, 1)
                v = np.clip(pos[m, b] / L[b], 0, 1)
                shape = np.sin(np.pi * u) * np.sin(np.pi * v)
                # normalise to mean 1 for impacts spread uniformly over the wall
                p = ac.position_weighting
                from math import gamma, pi, sqrt
                mean_sin_p = gamma((p + 1) / 2) / (sqrt(pi) * gamma(p / 2 + 1))
                w[m] = shape ** p / mean_sin_p ** 2
    if ac.impact_gain_sigma > 0:
        s = ac.impact_gain_sigma
        w *= np.exp(np.random.default_rng([seed, 991]).standard_normal(len(imp)) * s - 0.5 * s * s)
    return w


def build_sources(events: np.ndarray, wall_force, cfg: SimConfig, t_start: float,
                  n_out: int, seed: int = 0) -> tuple[dict[str, np.ndarray], dict]:
    """Source force signals [N] per channel at cfg.acoustics.fs, sample 0 at t_start [s].

    events: processed event table; wall_force: :class:`WallForce`.
    """
    ac = cfg.acoustics
    fs = ac.fs
    fc = ac.band_split_hz
    out = {c: np.zeros(n_out) for c in CHANNELS}

    if ac.wall_force_component:
        # wall force bins are impulse-conserving averages; resample to fs then low-pass
        wf = wall_force.force
        for w in range(wf.shape[0]):
            # cumulative impulse interpolation keeps ∫F dt exactly
            cum = np.concatenate([[0.0], np.cumsum(wf[w]) / wall_force.fs])
            tb = wall_force.t0 + np.arange(wf.shape[1] + 1) / wall_force.fs
            edges = t_start + np.arange(n_out + 1) / fs
            imp = np.diff(np.interp(edges, tb, cum, left=0.0, right=cum[-1]))
            out[WALL_TO_CHANNEL[w]] += split(imp * fs, fs, fc, "low")

    tau = np.zeros(0)
    if ac.impact_component and len(events):
        imp = events[events["contact_type"] == 0]
        if ac.tau_mode == "hertz_real":
            tau = event_contact_times(imp, cfg)
        else:
            tau = np.full(len(imp), ac.tau_fixed)
        rw = radiation_weights(imp, cfg, seed)
        for ch in CHANNELS:
            if ch == "pp":
                m = imp["kind"] == 0
                gain = ac.pp_gain
            else:
                walls = [w for w, c in WALL_TO_CHANNEL.items() if c == ch]
                m = (imp["kind"] == 1) & np.isin(imp["wall_id"], walls)
                gain = 1.0
            if not m.any() or gain == 0:
                continue
            # pulse centred on the event's force peak; width τ_c
            t_ev = imp["t_peak"][m] - 0.5 * tau[m]
            p = render_pulses(t_ev, imp["J_n"][m] * rw[m], tau[m], t_start, n_out, fs,
                              ac.pulse_shape, ac.oversample)
            out[ch] += gain * split(p, fs, fc, "high")
    if ac.highpass_hz > 0:
        for ch in CHANNELS:
            out[ch] = split(out[ch], fs, ac.highpass_hz, "high")
    info = {"band_split_hz": fc, "highpass_hz": ac.highpass_hz,
            "position_weighting": ac.position_weighting, "impact_gain_sigma": ac.impact_gain_sigma, "n_impacts": int((events["contact_type"] == 0).sum()) if len(events) else 0,
            "tau_c_median_s": float(np.median(tau)) if len(tau) else None,
            "tau_mode": ac.tau_mode, "pulse_shape": ac.pulse_shape, "pp_gain": ac.pp_gain}
    return out, info
