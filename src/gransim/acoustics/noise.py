"""Background (motor) noise (spec 6.1, step 3).

The library holds recordings of the empty box shaken at a given rpm, taken at the
microphone position. The file with the nearest rpm is used; if it is more than 3 % off,
it is resampled by rpm_file / rpm so the rotation harmonics follow the requested rpm. There is no phase
reference in the recordings, so the start point is random (recorded in the metadata).
Without a library, a synthetic noise (pink-ish broadband + rotation harmonics) is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy import signal

from gransim.acoustics.assets import read_wav
from gransim.acoustics.transfer import ASSETS
from gransim.config import AcousticConfig


def load_index(cfg: AcousticConfig) -> tuple[Path, list[dict]]:
    d = Path(cfg.noise_dir) if cfg.noise_dir else ASSETS / "noise"
    idx = d / "index.yaml"
    if not idx.exists():
        return d, []
    return d, (yaml.safe_load(idx.read_text()) or {}).get("files", [])


def draw_noise(cfg: AcousticConfig, rpm: float, n: int, rng: np.random.Generator):
    """Noise signal of n samples at cfg.fs (recorded level) and its metadata."""
    d, files = load_index(cfg)
    if files:
        best = min(abs(f["rpm"] - rpm) for f in files)
        cands = [f for f in files if abs(f["rpm"] - rpm) <= best + 1e-9]
        f = cands[rng.integers(len(cands))]
        fs, x = read_wav(d / f["file"])
        if fs != cfg.fs:
            x = signal.resample_poly(x, cfg.fs, fs)
        scaled = abs(f["rpm"] - rpm) / rpm > 0.03
        if scaled:
            # no recording at this rpm: play the nearest one faster/slower so the rotation
            # harmonics move with the rpm (approximation; record the real one when possible)
            from fractions import Fraction
            r = Fraction(f["rpm"] / rpm).limit_denominator(200)
            x = signal.resample_poly(x, r.numerator, r.denominator)
        start = int(rng.integers(len(x)))
        reps = int(np.ceil((start + n) / len(x)))
        y = np.tile(x, reps + 1)[start:start + n]
        return y, {"noise_file": f["file"], "noise_rpm": f["rpm"], "noise_rpm_mismatch":
                   abs(f["rpm"] - rpm) / rpm, "noise_rpm_scaled": scaled, "noise_start": start,
                   "noise_phase_aligned": False}
    # synthetic fallback: −3 dB/oct noise at the measured empty-box level + rotation harmonics
    w = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / cfg.fs)
    w[1:] /= np.sqrt(fr[1:] / 100.0)
    y = np.fft.irfft(w, n)
    t = np.arange(n) / cfg.fs
    f_rot = rpm / 60.0
    for k in range(1, 20):
        y += 0.3 * np.std(y) / k * np.sin(2 * np.pi * k * f_rot * t + rng.uniform(0, 2 * np.pi))
    y *= 0.016 / np.std(y)
    return y, {"noise_file": None, "noise_rpm": rpm, "noise_rpm_mismatch": 0.0,
               "noise_start": 0, "noise_phase_aligned": False}
