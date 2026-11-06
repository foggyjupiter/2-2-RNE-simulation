"""Build measured acoustic assets from first-study recordings.

The first-study clips (48 kHz, phone at ≈2.7 m, same box and shaker) give:

* **Motor-noise library** (``assets/noise``): the N = 0 clips (empty box shaken at the
  working rpm) are the real background, incl. the shaker's own clicks.
* **Response model** (``assets/impulse_response/response_model.npz``): what the
  microphone hears for one impact on the box, as measured statistics:

  - ``direct_psd``: spectrum of the first ``DIRECT_S`` after impact onsets (the click:
    box + direct path), background-subtracted;
  - ``tail_psd``: spectrum of the decaying ring between shaking cycles (box modes
    around 9-10 kHz + room reverberation);
  - ``tau``: decay time constant of the ring [s] (≈ 0.1 s, i.e. RT60 ≈ 0.7 s);
  - ``drr``: direct-to-reverberant energy ratio (a starting value; refined by the
    sim2real comparison).

  The renderer draws a random impulse response with these statistics
  (:mod:`gransim.acoustics.transfer`). The real contact pulse (half-sine, τ_ref) is
  divided out of ``direct_psd`` so the renderer can apply its own per-event τ_c.

Nothing N-dependent is introduced here (spec 14).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import yaml
from scipy import signal
from scipy.io import wavfile

DIRECT_S = 0.0015    # length of the direct (click) window after an onset [s]
NFFT = 1024          # spectral resolution of the stored PSDs (47 Hz at 48 kHz)


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """(fs, mono float signal in [-1, 1])."""
    fs, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


def write_wav(path: str | Path, fs: int, x: np.ndarray, bits: int = 16) -> None:
    """Write mono wav; 16-bit PCM or 32-bit float."""
    if bits == 16:
        x = np.clip(np.asarray(x, float), -1.0, 1.0 - 2.0 ** -15)
        wavfile.write(path, fs, np.round(x * 32768).astype(np.int16))
    elif bits == 32:
        wavfile.write(path, fs, np.asarray(x, np.float32))
    else:
        raise ValueError("bits must be 16 or 32 (float)")


def onset_envelope(x: np.ndarray, fs: int, band_hz: float = 12000.0) -> np.ndarray:
    """RMS envelope (1 kHz smoothing) of the band above ``band_hz``.

    Impacts are broadband clicks, while the box/room ring is concentrated around
    9-10 kHz and beats strongly; detecting onsets above 12 kHz avoids triggering on
    the beating of the ring.
    """
    hp = min(band_hz, 0.4 * fs)
    b, a = signal.butter(4, hp / (fs / 2), "high")
    y = signal.filtfilt(b, a, x)
    p = signal.filtfilt(*signal.butter(2, 1000 / (fs / 2)), y * y)
    return np.sqrt(np.maximum(p, 0.0))


def detect_onsets(x: np.ndarray, fs: int, rise_db: float = 15.0, min_gap: float = 0.003,
                  abs_factor: float = 3.0):
    """Impact onsets: the high-band envelope rises by ``rise_db`` within 1 ms and exceeds
    ``abs_factor`` × its median (rejects small motor clicks)."""
    env = onset_envelope(x, fs)
    lag = int(0.001 * fs)
    ratio = 20 * np.log10((env[lag:] + 1e-5) / (env[:-lag] + 1e-5))
    cand = np.flatnonzero((ratio[1:] > rise_db) & (ratio[:-1] <= rise_db)) + 1 + lag
    floor = abs_factor * np.median(env)
    out = []
    for c in cand:
        peak = env[c:c + lag].max() if c + lag <= len(env) else env[c]
        if peak < floor:
            continue
        if not out or c - out[-1] > min_gap * fs:
            out.append(int(c))
    return np.array(out, dtype=int), env


def half_sine_spectrum(f: np.ndarray, tau: float) -> np.ndarray:
    """|Fourier transform| of a unit-area half-sine pulse of width tau (1 at f = 0)."""
    u = 2 * np.asarray(f, float) * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.cos(np.pi * f * tau) / (1 - u ** 2)
    P = np.where(np.isclose(u, 1.0), np.pi / 4, P)
    return np.abs(P)


def _psd(segs: list[np.ndarray], fs: int) -> np.ndarray:
    w = np.hanning(NFFT)
    acc = np.zeros(NFFT // 2 + 1)
    for s in segs:
        z = np.zeros(NFFT)
        z[: min(len(s), NFFT)] = s[:NFFT]
        z[: len(s)] *= w[: len(s)] if len(s) < NFFT else w
        acc += np.abs(np.fft.rfft(z)) ** 2
    return acc / max(len(segs), 1)


def measure_response(clips: list[str], tau_ref: float, clip_level: float = 0.98,
                     noise_clips: list[str] | None = None) -> dict:
    """Direct/tail spectra, ring decay and direct-to-reverberant ratio from sparse clips.

    ``noise_clips`` (empty box) give the background spectrum, subtracted from the tail.
    """
    direct_segs, tail_segs, taus, drrs, noise_segs = [], [], [], [], []
    fs0 = None
    for c in clips:
        fs, x = read_wav(c)
        fs0 = fs0 or fs
        ons, env = detect_onsets(x, fs)
        nd = int(DIRECT_S * fs)
        pre = int(0.0002 * fs)
        # direct clicks: onsets not clipped, whose following 1.5 ms is not clipped
        for o in ons:
            if o - pre - nd < 0 or o + nd >= len(x):
                continue
            seg = x[o - pre:o + nd]
            if np.max(np.abs(seg)) > clip_level:
                continue
            direct_segs.append(seg - seg.mean())
            # level just before the onset = ongoing ring (subtracted below)
            noise_segs.append(x[o - pre - nd:o - pre] - x[o - pre - nd:o - pre].mean())
        # tails: longest gaps between onsets (≥ 60 ms) after a strong onset
        gaps = np.diff(ons)
        for k in np.argsort(gaps)[::-1][:6]:
            if gaps[k] < 0.06 * fs:
                break
            a = ons[k] + int(0.005 * fs)
            b = ons[k + 1] - int(0.002 * fs)
            seg = x[a:b]
            if np.max(np.abs(seg)) > clip_level:
                continue
            tail_segs.append(seg)
            e = env[a:b]
            t = np.arange(len(e)) / fs
            p = np.polyfit(t, np.log(e + 1e-9), 1)
            if p[0] < 0:
                taus.append(-1.0 / p[0])
            # DRR of the onset that started this gap: direct energy vs ring energy
            o = ons[k]
            if o + nd < len(x) and np.max(np.abs(x[o:o + nd])) <= clip_level:
                e_dir = np.sum(x[o:o + nd] ** 2) - nd * np.mean(x[o - nd:o] ** 2)
                tau_k = -1.0 / p[0] if p[0] < 0 else 0.1
                e_tail = (np.mean(seg[: int(0.005 * fs)] ** 2)) * fs * tau_k / 2
                if e_dir > 0 and e_tail > 0:
                    drrs.append(e_dir / e_tail)
    fs = int(fs0)
    f = np.fft.rfftfreq(NFFT, 1 / fs)
    P_dir = _psd(direct_segs, fs) - _psd(noise_segs, fs)
    P_dir = np.maximum(P_dir, 1e-3 * P_dir.max())
    # divide out the real contact pulse (regularised)
    H = half_sine_spectrum(f, tau_ref)
    P_dir = P_dir / np.maximum(H, 0.25) ** 2
    P_tail = _psd([s[i:i + NFFT] for s in tail_segs for i in range(0, len(s) - NFFT, NFFT // 2)], fs)
    if noise_clips:
        nsegs = []
        for c in noise_clips:
            _, xn = read_wav(c)
            nsegs += [xn[i:i + NFFT] for i in range(0, len(xn) - NFFT, NFFT // 2)]
        P_tail = np.maximum(P_tail - _psd(nsegs, fs), 1e-3 * P_tail.max())
    return {"fs": fs, "f": f, "direct_psd": P_dir / P_dir.sum(), "tail_psd": P_tail / P_tail.sum(),
            "tau": float(np.median(taus)), "tau_spread": [float(np.percentile(taus, 25)),
                                                         float(np.percentile(taus, 75))],
            "drr": float(np.median(drrs)) if drrs else 0.05,
            "n_direct": len(direct_segs), "n_tail": len(tail_segs), "tau_ref": tau_ref}


def build_assets(recordings: str | Path, out_root: str | Path, rpm: float, tau_ref: float,
                 sparse_count: int = 5, noise_count: int = 0, clip_s: float = 5.0) -> dict:
    """Write assets/noise/*.wav + index.yaml and assets/impulse_response/response_model.npz.

    ``recordings`` is a folder with files named ``{N}-{k}.wav`` (first-study layout).
    """
    recordings, out_root = Path(recordings), Path(out_root)
    noise_dir = out_root / "noise"
    ir_dir = out_root / "impulse_response"
    noise_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in sorted(glob.glob(str(recordings / f"{noise_count}-*.wav"))):
        fs, x = read_wav(f)
        n = int(clip_s * fs)
        mid = (len(x) - n) // 2
        name = f"motor_rpm{int(round(rpm))}_{Path(f).stem}.wav"
        write_wav(noise_dir / name, fs, x[mid:mid + n])
        entries.append({"file": name, "rpm": float(rpm), "fs": int(fs),
                        "source": f"first study, N={noise_count} clip {Path(f).name}"})
    (noise_dir / "index.yaml").write_text(yaml.safe_dump(
        {"description": "Motor/shaker background recorded with an empty box (N=0) at the "
                        "microphone position (phone, 2.7 m). rpm estimated from the envelope "
                        "periodicity of the first-study clips.", "files": entries},
        sort_keys=False, allow_unicode=True))

    clips = sorted(glob.glob(str(recordings / f"{sparse_count}-*.wav")))
    noise_clips = sorted(glob.glob(str(recordings / f"{noise_count}-*.wav")))
    m = measure_response(clips, tau_ref, noise_clips=noise_clips)
    np.savez_compressed(ir_dir / "response_model.npz", f=m["f"], direct_psd=m["direct_psd"],
                        tail_psd=m["tail_psd"], fs=m["fs"], tau=m["tau"], drr=m["drr"],
                        tau_ref=tau_ref)
    meta = {k: m[k] for k in ("fs", "tau", "tau_spread", "drr", "n_direct", "n_tail", "tau_ref")}
    (ir_dir / "index.yaml").write_text(yaml.safe_dump(
        {"description": "Statistical response model of one impact at the microphone, measured "
                        f"from N={sparse_count} first-study clips: direct-click spectrum (contact "
                        "pulse divided out), ring/reverb tail spectrum, decay time, direct-to-"
                        "reverberant ratio. Mixed walls; no N dependence measured yet.",
         "response_model": {"file": "response_model.npz", **meta}},
        sort_keys=False, allow_unicode=True))
    return {"noise_files": len(entries), "response_model": meta}
