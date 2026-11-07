"""Black-box acoustic rendering: every acoustic setting from physics and the apparatus,
none fitted to the first-study recordings.

The earlier renderer (:mod:`gransim.acoustics.render`) takes its impulse response, its
equaliser, its direct-to-reverberant ratio and its microphone gain from fits to the
first-study recordings, and half its clips use real BB clicks cut from them. A model
trained on that audio has partly seen the test material. Here the only measured input is
the background noise of the *empty* box (N = 0, clips 1-5 only: clips 6-15 of every N stay
for evaluation). Everything else is drawn per render from a physically reasonable range
(moderate: a normal room and a phone a few metres away, no extreme set-ups).

Chain (per channel bottom / top / side / pp, same sources as the earlier renderer)::

    source (wall-force low band + Hertz pulses, real tau_c)          [build_sources, reused]
      ⊛ h_channel = mic ⊛ [ sqrt(D)·direct + sqrt(R)·room ]          [this module]
    audio = clip( mic_gain · Σ channels  +  noise_gain · empty-box noise )

direct  the box radiating the impact
        * plate modes of each 3 mm acrylic wall (thin-plate theory: f_mn from the
          bending stiffness, boundary condition between simply supported and clamped →
          frequency factor 1-1.8, loss factor 0.02-0.08 → each mode rings
          exp(-π f η t)); amplitude = mode shape at a random point × radiation
          efficiency (∝ f / f_coincidence below coincidence, 1 above); ±4 % frequency
          scatter per mode (thickness, mounting)
        * rigid-body radiation of the whole box (broadband, follows the force pulse):
          10-50 % of the direct energy
        * attack: modes and early reflections build up over 0.05-0.5 ms
room    diffuse reverberation: noise with the direct sound's smoothed spectrum (the
        diffuse field carries the radiated power), decaying with RT60(f) from Sabine with
        air absorption (high frequencies die faster); room 30-150 m³, RT60(1 kHz)
        0.3-0.8 s; direct-to-reverberant ratio from the critical distance
        r_c² = Q V / (100 π RT60) with directivity Q 1-4 and microphone distance 1.5-3.5 m
mic     smooth random frequency response (±1.5 dB s.d. below 8 kHz, ±3 dB above,
        knots one octave apart) and the 48 kHz recorder's anti-alias roll-off (20.5-22.5 kHz)

pp      particle-particle impacts reach the microphone through the air inside the box and
        all six walls: their own response (modes of every wall), level 0.15-0.8 of a wall
        impact of the same impulse

Not used here (each was fitted to the recordings or is a per-impact amplitude spread):
the measured response model, the sim2real equaliser, recorded BB clicks, and the
per-impact radiation spread σ (impact_gain_sigma = 0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from gransim.acoustics.assets import read_wav
from gransim.acoustics.source import CHANNELS, build_sources
from gransim.acoustics.transfer import ASSETS, apply_response
from gransim.config import SimConfig

SALT = 0xB1AC  # separates the black-box draws from every other use of a render seed

# acrylic (PMMA) walls; E and ν come from the container config
ACRYLIC_DENSITY = 1190.0      # kg/m³
C_AIR = 343.0                 # m/s
FS = 48000

# air absorption at 20 °C, 50 % RH (ISO 9613-1), dB per km
_AIR_F = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000], float)
_AIR_DB_KM = np.array([0.4, 1.1, 2.7, 4.7, 9.9, 29.0, 105.0, 375.0, 750.0])

# noise: empty-box recordings, clips 1-5 only (6-15 of every N are evaluation clips)
NOISE_CLIPS = (1, 2, 3, 4, 5)
NOISE_RPM = 372.0

# ---------------------------------------------------------------------------------
# parameter ranges (moderate, physically reasonable)
# ---------------------------------------------------------------------------------
RANGES = {
    # recording chain
    # digital units per newton: not derivable from physics. Set like an operator who keeps
    # the loudest material in range (median clipping a few %), ±4.5 dB unknown phone/app gain
    "mic_gain":        ("loguniform", 0.28, 0.8),
    "noise_gain":      ("uniform", 0.7, 1.4),        # ≈ ±3 dB around the recorded level
    # particles
    "particle_E":      ("loguniform", 1.5e9, 7.0e10),  # plastics (ABS/PLA) .. glass
    "pp_gain":         ("loguniform", 0.15, 0.8),    # particle-particle vs wall impact level
    "position_weighting": ("uniform", 0.0, 1.5),     # impacts mid-wall radiate more
    # box
    "plate_bc":        ("uniform", 1.0, 1.8),        # simply supported (1) .. clamped (~1.8)
    "plate_eta":       ("loguniform", 0.02, 0.08),   # acrylic loss factor incl. mounting
    "rigid_frac":      ("uniform", 0.1, 0.5),        # rigid-body share of the direct energy
    "attack_s":        ("loguniform", 5.0e-5, 5.0e-4),
    # room and microphone position
    "room_volume":     ("loguniform", 30.0, 150.0),  # m³: small room .. classroom
    "rt60":            ("uniform", 0.3, 0.8),        # s at 1 kHz
    "directivity_q":   ("uniform", 1.0, 4.0),        # box on a table: half-space .. quarter
    "mic_distance":    ("uniform", 1.5, 3.5),        # m
    "mic_low_sd_db":   ("fixed", 1.5, 1.5),
    "mic_high_sd_db":  ("fixed", 3.0, 3.0),
    "aa_cutoff_hz":    ("uniform", 20500.0, 22500.0),
}
IR_LENGTH_S = 0.6        # long enough for RT60 ≤ 0.8 s at the band level that matters (−45 dB)
HIGHPASS_HZ = 500.0      # 5 cm box radiates little below; same value as the earlier renderer


@dataclass
class BBParams:
    mic_gain: float
    noise_gain: float
    particle_E: float
    pp_gain: float
    position_weighting: float
    plate_bc: float
    plate_eta: float
    rigid_frac: float
    attack_s: float
    room_volume: float
    rt60: float
    directivity_q: float
    mic_distance: float
    mic_low_sd_db: float
    mic_high_sd_db: float
    aa_cutoff_hz: float
    drr: float = 0.0          # derived: direct/reverberant energy ratio (linear)
    noise_clip: int = 1
    noise_start: int = 0

    def as_row(self) -> dict:
        d = asdict(self)
        d["drr_db"] = 10 * math.log10(self.drr)
        return {f"bb_{k}": v for k, v in d.items()}


def draw_params(render_seed: int) -> BBParams:
    """Acoustic set-up of one render, from its seed (rerunning gives identical audio)."""
    rng = np.random.default_rng([int(render_seed), SALT])
    vals = {}
    for k, (dist, lo, hi) in RANGES.items():
        u = float(rng.random())
        if dist == "uniform":
            vals[k] = lo + u * (hi - lo)
        elif dist == "loguniform":
            vals[k] = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            vals[k] = lo
    p = BBParams(**vals)
    rc2 = p.directivity_q * p.room_volume / (100 * math.pi * p.rt60)    # critical distance²
    p.drr = rc2 / p.mic_distance ** 2
    p.noise_clip = int(NOISE_CLIPS[rng.integers(len(NOISE_CLIPS))])
    return p


# ---------------------------------------------------------------------------------
# plate modes of the box walls
# ---------------------------------------------------------------------------------
def plate_constants(cfg: SimConfig) -> tuple[float, float]:
    """(sqrt(D/ρh) [m²/s], coincidence frequency [Hz]) of the acrylic walls."""
    E, nu = cfg.container.material.youngs_modulus, cfg.container.material.poisson_ratio
    h = cfg.container.wall_thickness
    D = E * h ** 3 / (12 * (1 - nu ** 2))
    k = math.sqrt(D / (ACRYLIC_DENSITY * h))
    f_c = C_AIR ** 2 / (2 * math.pi * k)
    return k, f_c


def walls(cfg: SimConfig) -> dict[str, list[tuple[float, float]]]:
    """Plate dimensions (a, b) [m] behind each channel."""
    Lx, Ly, Lz = cfg.container.size
    bottom, top = [(Lx, Ly)], [(Lx, Ly)]
    side = [(Lx, Lz), (Lx, Lz), (Ly, Lz), (Ly, Lz)]
    return {"bottom": bottom, "top": top, "side": side, "pp": bottom + top + side}


def plate_modes(a: float, b: float, k: float, bc: float, f_max: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and modal amplitudes (shape at a random point) of one plate."""
    fs_, amp = [], []
    u, v = rng.random(), rng.random()          # point of excitation / observation
    for m in range(1, 40):
        for n in range(1, 40):
            f = bc * 0.5 * math.pi * k * ((m / a) ** 2 + (n / b) ** 2)
            if f > f_max:
                if n == 1:
                    return np.array(fs_), np.array(amp)
                break
            fs_.append(f * (1 + 0.04 * rng.standard_normal()))
            amp.append(math.sin(m * math.pi * u) * math.sin(n * math.pi * v))
    return np.array(fs_), np.array(amp)                                  # pragma: no cover


def _unit(x: np.ndarray) -> np.ndarray:
    e = math.sqrt(float(np.sum(x * x)))
    return x / e if e > 0 else x


def direct_ir(cfg: SimConfig, p: BBParams, channel: str, n: int,
              rng: np.random.Generator) -> np.ndarray:
    """Direct sound of the box for one channel (unit energy)."""
    k, f_c = plate_constants(cfg)
    t = np.arange(n) / FS
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(m, 1 / FS)
    H = np.zeros(len(f), complex)
    f_first = np.inf
    for a, b in walls(cfg)[channel]:
        fk, amp = plate_modes(a, b, k, p.plate_bc, 0.5 * FS, rng)
        f_first = min(f_first, float(fk.min()))
        for fi, ai in zip(fk, amp):
            # surface acceleration per unit force of one mode (∝ its shape at the impact
            # point): stiffness-controlled (∝ f²) below the mode, resonant at it, mass-
            # controlled above; loss factor η sets the ring time 1/(π f η)
            H += ai * f ** 2 / (fi ** 2 - f ** 2 + 1j * p.plate_eta * fi * f)
    # broadband share: higher-order wall/cavity coupling not resolved as single modes,
    # stiffness-controlled below the first wall mode like the modes themselves
    x = f / f_first
    B = (x ** 2 / (1 + x ** 2)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f)))
    B[0] = 0
    Hm = H / math.sqrt(np.mean(np.abs(H) ** 2))
    Bm = B / math.sqrt(np.mean(np.abs(B) ** 2))
    H = math.sqrt(1 - p.rigid_frac) * Hm + math.sqrt(p.rigid_frac) * Bm
    # radiation efficiency of a wall below coincidence: radiated power ∝ f / f_c
    H *= np.sqrt(np.minimum(1.0, f / f_c))
    h = np.fft.irfft(H, m)[:n]
    h *= 1.0 - np.exp(-t / p.attack_s)
    return _unit(h)


# ---------------------------------------------------------------------------------
# room
# ---------------------------------------------------------------------------------
def rt60_of_f(f: np.ndarray, p: BBParams) -> np.ndarray:
    """RT60 [s] vs frequency: Sabine with the room's absorption plus air absorption."""
    A = 0.161 * p.room_volume / p.rt60                                     # m² at 1 kHz
    m = np.interp(np.log(np.maximum(f, 1.0)), np.log(_AIR_F), _AIR_DB_KM) / 1000 / 4.343
    m1k = np.interp(np.log(1000.0), np.log(_AIR_F), _AIR_DB_KM) / 1000 / 4.343
    A_room = A - 4 * m1k * p.room_volume
    return 0.161 * p.room_volume / (A_room + 4 * m * p.room_volume)


def room_ir(direct: np.ndarray, p: BBParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """Diffuse reverberation (unit energy): noise with the direct sound's smoothed power
    spectrum, each band decaying with its own RT60."""
    t = np.arange(n) / FS
    f = np.fft.rfftfreq(n, 1 / FS)
    P = np.abs(np.fft.rfft(direct, n)) ** 2
    # smooth over ±1/6 octave
    logf = np.log2(np.maximum(f, 1.0))
    cs = np.concatenate([[0.0], np.cumsum(P)])
    lo = np.searchsorted(logf, logf - 1 / 6)
    hi = np.searchsorted(logf, logf + 1 / 6, side="right")
    Ps = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    X = np.fft.rfft(rng.standard_normal(n)) * np.sqrt(Ps)
    edges = np.geomspace(250.0, 0.5 * FS, 13)
    y = np.zeros(n)
    for f0, f1 in zip(edges[:-1], edges[1:]):
        band = np.where((f >= f0) & (f < f1), X, 0)
        if f0 == edges[0]:
            band = np.where(f < f1, X, 0)
        T = float(rt60_of_f(np.array([math.sqrt(f0 * f1)]), p)[0])
        y += np.fft.irfft(band, n) * np.exp(-6.91 * t / T)
    gap = int(0.002 * FS)                     # first reflections arrive after a few ms
    y[:gap] *= np.linspace(0.0, 1.0, gap)
    return _unit(y)


# ---------------------------------------------------------------------------------
# microphone / recorder
# ---------------------------------------------------------------------------------
def mic_curve(p: BBParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(knot frequencies, gain dB): smooth random response + anti-alias roll-off."""
    knots = np.array([250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000], float)
    sd = np.where(knots < 8000, p.mic_low_sd_db, p.mic_high_sd_db)
    g = np.clip(rng.standard_normal(len(knots)), -2, 2) * sd
    g -= np.interp(1000.0, knots, g)          # level is set by mic_gain (1 kHz reference)
    return knots, g


def apply_mic(h: np.ndarray, p: BBParams, knots: np.ndarray, gdb: np.ndarray) -> np.ndarray:
    m = 1 << int(np.ceil(np.log2(2 * len(h))))
    f = np.fft.rfftfreq(m, 1 / FS)
    g = 10 ** (np.interp(np.log(np.maximum(f, 1.0)), np.log(knots), gdb) / 20)
    # anti-alias: raised-cosine from 0.92·fc to fc, then −60 dB
    x = np.clip((f - 0.92 * p.aa_cutoff_hz) / (0.08 * p.aa_cutoff_hz), 0, 1)
    g *= 0.001 + 0.999 * 0.5 * (1 + np.cos(np.pi * x))
    return np.fft.irfft(np.fft.rfft(h, m) * g, m)[: len(h)]


def channel_irs(cfg: SimConfig, p: BBParams, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng([int(seed), SALT, 1])
    n = int(IR_LENGTH_S * FS)
    knots, gdb = mic_curve(p, rng)            # one microphone for all channels
    out = {}
    for c in CHANNELS:
        d = direct_ir(cfg, p, c, n, rng)
        r = room_ir(d, p, n, rng)
        h = math.sqrt(p.drr / (1 + p.drr)) * d + math.sqrt(1 / (1 + p.drr)) * r
        out[c] = apply_mic(h, p, knots, gdb)
    return out


# ---------------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _noise_clip(k: int) -> np.ndarray:
    fs, x = read_wav(ASSETS / "noise" / f"motor_rpm{int(NOISE_RPM)}_0-{k}.wav")
    if fs != FS:
        x = signal.resample_poly(x, FS, fs)
    return x


def draw_noise(p: BBParams, rpm: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = _noise_clip(p.noise_clip)
    if abs(NOISE_RPM - rpm) / rpm > 0.03:
        from fractions import Fraction
        r = Fraction(NOISE_RPM / rpm).limit_denominator(200)
        x = signal.resample_poly(x, r.numerator, r.denominator)
    start = int(rng.integers(len(x)))
    p.noise_start = start
    reps = int(np.ceil((start + n) / len(x)))
    return np.tile(x, reps + 1)[start:start + n]


# ---------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------
def render_bb(cfg: SimConfig, events: np.ndarray, wall_force, record_window: tuple,
              render_seed: int, warmup_s: float = 0.3, duration_s: float = 5.0):
    """Audio (float, clipped at ±1) and metadata of one black-box render."""
    p = draw_params(render_seed)
    ac = cfg.acoustics.model_copy(update={
        "fs": FS, "impact_source": "synthetic", "impact_gain_sigma": 0.0,
        "pp_gain": p.pp_gain, "position_weighting": p.position_weighting,
        "particle_youngs_modulus": p.particle_E, "highpass_hz": HIGHPASS_HZ,
        "tau_mode": "hertz_real", "pulse_shape": "half_sine", "noise": False, "eq": False})
    cfg = cfg.model_copy(update={"acoustics": ac})
    t0, t1 = record_window
    if t1 - t0 - warmup_s < duration_s - 1e-9:
        raise ValueError(f"record window {t1 - t0:.3f} s too short")
    n_tot = int(round((warmup_s + duration_s) * FS))
    n_out = int(round(duration_s * FS))
    src, src_info = build_sources(events, wall_force, cfg, t0, n_tot, int(render_seed))
    irs = channel_irs(cfg, p, render_seed)
    clean = p.mic_gain * sum(apply_response(src[c], irs[c]) for c in CHANNELS)[n_tot - n_out:]
    A, f, gam = cfg.excitation.resolved(cfg.gravity)
    rng = np.random.default_rng([int(render_seed), SALT, 2])
    y = clean + p.noise_gain * draw_noise(p, 60.0 * f, n_out, rng)
    clip_frac = float(np.mean(np.abs(y) >= 1.0))
    y = np.clip(y, -1.0, 1.0)
    meta = {"renderer": "blackbox", **p.as_row(),
            "rms": float(np.sqrt(np.mean(y ** 2))), "rms_clean": float(np.sqrt(np.mean(clean ** 2))),
            "clip_fraction": clip_frac, "tau_c_median_s": src_info.get("tau_c_median_s"),
            "n_impacts_render": src_info.get("n_impacts"),
            "noise_file": f"motor_rpm{int(NOISE_RPM)}_0-{p.noise_clip}.wav",
            "particle_E_acoustic_pa": p.particle_E}
    return y, meta
