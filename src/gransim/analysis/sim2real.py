"""Synthesised vs recorded audio statistics (spec 9.1, sim2real).

The same feature set is computed on recordings and on synthesised clips of the same
N and rpm. Tolerance per feature = spread of the repeated recordings (mean ± k·std,
at least ``min_tol`` relative). Features:

  rms, clip_fraction                level and saturation of the device
  band_db_*                         octave-band levels relative to total [dB]
  centroid_hz                       spectral centroid
  onset_rate                        detected impact onsets per second (> 12 kHz band)
  kurtosis (ψ3), entropy (ψ2)       first-study order parameters
  fold_peak_to_mean                 peakiness of the period-folded energy (phase-free)
  fold_shape_corr                   correlation of the folded profile with the mean
                                    recorded profile (best circular shift)
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from gransim.acoustics.assets import detect_onsets
from gransim.analysis.order_params import folded_energy, psi2, psi3

BANDS = [(125, 500), (500, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000),
         (16000, 24000)]


def features(y: np.ndarray, fs: int, f_shake: float, clip_level: float = 0.99) -> dict:
    f, P = signal.welch(y, fs, nperseg=4096)
    tot = P.sum()
    out = {"rms": float(np.sqrt(np.mean(y ** 2))),
           "clip_fraction": float(np.mean(np.abs(y) >= clip_level)),
           "centroid_hz": float((f * P).sum() / tot)}
    for lo, hi in BANDS:
        if lo < fs / 2:
            out[f"band_db_{lo}_{hi}"] = float(10 * np.log10(P[(f >= lo) & (f < hi)].sum() / tot + 1e-20))
    ons, _ = detect_onsets(y, fs)
    out["onset_rate"] = len(ons) / (len(y) / fs)
    out["kurtosis"] = psi3(y)
    out["entropy_psi2"] = psi2(y)
    prof = folded_energy(y, fs, f_shake, 64)
    out["fold_peak_to_mean"] = float(prof.max() / prof.mean())
    out["_fold"] = prof / prof.mean()
    return out


def best_shift_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(np.corrcoef(np.roll(a, r), b)[0, 1] for r in range(len(a))))


def compare(real: list[dict], sim: list[dict], k: float = 2.0, min_tol: float = 0.1) -> list[dict]:
    """Rows: feature, real mean/std, sim mean, difference, tolerance, pass."""
    ref_fold = np.mean([r["_fold"] for r in real], axis=0)
    for d in real + sim:
        d["fold_shape_corr"] = best_shift_corr(d["_fold"], ref_fold)
    keys = [k_ for k_ in real[0] if not k_.startswith("_")]
    rows = []
    for key in keys:
        rv = np.array([r[key] for r in real])
        sv = np.array([s[key] for s in sim])
        m, sd = rv.mean(), rv.std(ddof=1) if len(rv) > 1 else 0.0
        if key.startswith("band_db"):
            tol = max(k * sd, 2.0)            # dB
        elif key == "clip_fraction":
            tol = max(k * sd, 0.01)
        else:
            tol = max(k * sd, min_tol * abs(m))
        diff = sv.mean() - m
        rows.append({"feature": key, "real_mean": float(m), "real_std": float(sd),
                     "sim_mean": float(sv.mean()), "sim_std": float(sv.std()),
                     "diff": float(diff), "tol": float(tol), "pass": bool(abs(diff) <= tol)})
    return rows


# --------------------------------------------------------------------------
# Driver: recordings folder + simulated runs → calibrated gain and comparison table
# --------------------------------------------------------------------------

def _compose(clean: np.ndarray, noise: np.ndarray, gain: float, clip_level: float) -> np.ndarray:
    return np.clip(gain * clean + noise, -clip_level, clip_level)


def run_sim2real(rec_dir, run_dirs: dict, f_shake: float, n_render: int = 2,
                 gains=None, acoustics_overrides: dict | None = None, seed: int = 0) -> dict:
    """Compare recordings ``rec_dir/{N}-*.wav`` with runs ``run_dirs[N] = [paths]``.

    The audio of each run is rendered once without gain/noise/clipping; the microphone
    gain is then calibrated (one scalar for all N) by least squares on log RMS, and
    the final clips are composed as clip(gain · clean + recorded noise).
    """
    import glob
    from pathlib import Path

    from gransim.acoustics.assets import read_wav
    from gransim.acoustics.noise import draw_noise
    from gransim.acoustics.render import load_run, render
    from gransim.config import AcousticConfig

    rng = np.random.default_rng(seed)
    real, clean = {}, {}
    for N, runs in sorted(run_dirs.items()):
        files = sorted(glob.glob(str(Path(rec_dir) / f"{N}-*.wav")))
        real[N] = []
        for f in files:
            fs, x = read_wav(f)
            real[N].append(features(x, fs, f_shake))
        clean[N] = []
        for rd in runs:
            cfg, ev, wf, win = load_run(rd)
            ac = cfg.acoustics.model_dump()
            ac.update({"mic_gain": 1.0, "noise": False, "clip": False, **(acoustics_overrides or {})})
            ac = AcousticConfig.model_validate(ac)
            for k in range(n_render):
                r = render(cfg, ev, wf, win, seed=int(rng.integers(1 << 30)), acoustics=ac)
                nz, _ = draw_noise(ac, 60 * f_shake, len(r.clean), rng)
                clean[N].append((r.clean, nz, r.fs))
    # calibrate gain on log RMS (clean + noise, before clipping matters little at low N)
    if gains is None:
        gains = np.geomspace(0.05, 5.0, 61)
    best, best_err = None, np.inf
    for g in gains:
        err = 0.0
        for N in clean:
            rr = np.mean([d["rms"] for d in real[N]])
            ss = np.mean([np.sqrt(np.mean(_compose(c, n, g, 1.0) ** 2)) for c, n, _ in clean[N]])
            err += (np.log(ss) - np.log(rr)) ** 2
        if err < best_err:
            best, best_err = float(g), err
    table = {}
    for N in clean:
        sim = [features(_compose(c, n, best, 1.0), fs, f_shake) for c, n, fs in clean[N]]
        table[N] = compare(real[N], sim)
    return {"mic_gain": best, "log_rms_rmse": float(np.sqrt(best_err / len(clean))),
            "table": table}
