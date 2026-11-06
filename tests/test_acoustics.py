"""Spec 10.1 acoustic tests: pulse rendering, fractional delay, linearity, τ_c direction,
no double counting between the wall-force and impact-pulse components."""

import numpy as np
import pytest

from gransim.acoustics.contact_time import (effective_modulus, hertz_contact_time,
                                            sphere_lowest_mode_hz)
from gransim.acoustics.pulse import render_pulses
from gransim.acoustics.render import render
from gransim.acoustics.source import CHANNELS, build_sources
from gransim.config import SimConfig
from gransim.dynamics.engine import WallForce
from gransim.events.tracker import EVENT_DTYPE

FS = 48000


def one_event(t_peak=0.1, J=1e-3, v=1.0, wall=0, tau_dyn=1e-4):
    ev = np.zeros(1, EVENT_DTYPE)
    ev["t_start"] = t_peak - tau_dyn / 2
    ev["t_peak"] = t_peak
    ev["t_end"] = t_peak + tau_dyn / 2
    ev["kind"] = 1
    ev["contact_type"] = 0
    ev["wall_id"] = wall
    ev["v_rel_n"] = v
    ev["J_n"] = J
    ev["m_eff"] = 1.2e-4
    ev["R_eff"] = 3e-3
    return ev


def wall_force_for(ev, fs_wf=22050.0, T=0.3):
    """Wall force bins holding each event's impulse spread over its dynamic contact time."""
    n = int(T * fs_wf)
    F = np.zeros((6, n))
    for e in ev:
        if e["kind"] != 1:
            continue
        a, b = e["t_start"] * fs_wf, e["t_end"] * fs_wf
        for k in range(int(a), int(np.ceil(b))):
            ov = min(b, k + 1) - max(a, k)
            F[e["wall_id"], k] += e["J_n"] * ov / (b - a) * fs_wf
    return WallForce(force=F, fs=fs_wf, t0=0.0)


def test_hertz_contact_time_values():
    cfg = SimConfig()
    E = effective_modulus(cfg.particles.material, cfg.container.material)
    tau1 = hertz_contact_time(1.2e-4, 3e-3, E, 1.0)
    assert 3e-5 < tau1 < 8e-5                       # ≈ 50 μs (spec 2)
    assert hertz_contact_time(1.2e-4, 3e-3, E, 0.1) > tau1   # slower → longer
    assert sphere_lowest_mode_hz(0.006, cfg.particles.material, 1060.0) > 50e3


@pytest.mark.parametrize("shape", ["half_sine", "hertz"])
def test_single_pulse_area_and_position(shape):
    J, tau, t = 2e-3, 2e-4, 0.0101
    y = render_pulses(np.array([t]), np.array([J]), np.array([tau]), 0.0, 2000, FS, shape)
    assert y.sum() / FS == pytest.approx(J, rel=1e-3)
    centre = (np.arange(len(y)) * y).sum() / y.sum() / FS
    assert centre == pytest.approx(t + tau / 2, abs=0.2 / FS)
    # a wide pulse (10 samples) keeps its width: energy-weighted spread ~ τ/√(12)-ish
    spread = np.sqrt(((np.arange(len(y)) / FS - centre) ** 2 * np.abs(y)).sum() / np.abs(y).sum())
    assert 0.1 * tau < spread < 0.4 * tau


def test_fractional_shift_preserves_energy():
    tau = 5e-5  # shorter than ~2.5 output samples
    e = []
    for shift in (0.0, 0.3, 0.5, 0.7):
        y = render_pulses(np.array([0.01 + shift / FS]), np.array([1e-3]), np.array([tau]),
                          0.0, 1500, FS, "half_sine")
        e.append(np.sum(y ** 2))
    assert np.ptp(e) / np.mean(e) < 0.01


def _cfg(**ac):
    base = {"noise": False, "clip": False, "warmup_s": 0.0, "response": "synthetic", "highpass_hz": 0.0}
    base.update(ac)
    return SimConfig(acoustics=base)


def test_linearity_double_impulse_doubles_audio():
    ev = one_event()
    wf = wall_force_for(ev)
    cfg = _cfg()
    a = render(cfg, ev, wf, (0.0, 0.3), seed=3).audio
    ev2 = ev.copy()
    ev2["J_n"] *= 2
    b = render(cfg, ev2, wall_force_for(ev2), (0.0, 0.3), seed=3).audio
    assert np.allclose(b, 2 * a, atol=1e-12 * np.abs(a).max())


def test_centroid_moves_with_contact_time():
    """Shorter τ_c → brighter (spectral centroid up), at 48 kHz output."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.01, 0.49, 200))
    J = np.full(200, 1e-3)
    cents = []
    for tau in (2e-5, 1e-4, 4e-4):
        y = render_pulses(t, J, np.full(200, tau), 0.0, int(0.5 * FS), FS, "half_sine")
        P = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / FS)
        cents.append((f * P).sum() / P.sum())
    assert cents[0] > cents[1] > cents[2]


def test_no_double_counting_isolated_impact():
    """Wall-force (low band) + impact pulse (high band): total impulse = J exactly once."""
    ev = one_event(J=1.5e-3)
    wf = wall_force_for(ev)
    cfg = _cfg()
    n = int(0.3 * FS)
    src, _ = build_sources(ev, wf, cfg, 0.0, n)
    total = sum(src[c].sum() for c in CHANNELS) / FS
    assert total == pytest.approx(1.5e-3, rel=1e-3)
    # each component alone also carries it in its own band: low part holds the DC
    only_low, _ = build_sources(ev, wf, _cfg(impact_component=False), 0.0, n)
    assert sum(only_low[c].sum() for c in CHANNELS) / FS == pytest.approx(1.5e-3, rel=1e-3)
    only_high, _ = build_sources(ev, wf, _cfg(wall_force_component=False), 0.0, n)
    assert abs(sum(only_high[c].sum() for c in CHANNELS) / FS) < 1e-3 * 1.5e-3


def test_measured_response_model_renders():
    ev = one_event()
    cfg = SimConfig(acoustics={"noise": True, "warmup_s": 0.0})
    r = render(cfg, ev, wall_force_for(ev), (0.0, 0.3), seed=1)
    assert np.isfinite(r.audio).all() and r.meta["response"] == "measured"
    assert r.meta["noise_file"] is not None


def test_click_library_and_recorded_impacts(tmp_path):
    """Isolated clicks are cut from a recording; impact_source=recorded puts them at the
    impact times with the synthetic direct energy, and leaves the rest of the sound alone."""
    from gransim.acoustics.assets import build_click_library, write_wav

    rng = np.random.default_rng(0)
    x = 0.002 * rng.standard_normal(3 * FS)
    for t in np.arange(0.1, 2.9, 0.1):                  # isolated clicks every 100 ms
        i = int(t * FS)
        x[i:i + 100] += 0.4 * rng.standard_normal(100) * np.exp(-np.arange(100) / 20)
    write_wav(tmp_path / "drops.wav", FS, x)
    info = build_click_library([str(tmp_path / "drops.wav")], tmp_path / "clicks.npz")
    assert info["n_clicks"] >= 10
    with np.load(tmp_path / "clicks.npz") as z:
        assert np.allclose((z["clips"] ** 2).sum(1), 1.0, rtol=1e-5)

    ev = np.concatenate([one_event(t_peak=0.05 + 0.04 * k, J=1e-3 * (1 + k % 3)) for k in range(5)])
    wf = wall_force_for(ev)
    syn = _cfg(response="measured", eq=False)
    rec = _cfg(response="measured", eq=False, impact_source="recorded",
               click_library=str(tmp_path / "clicks.npz"))
    a = render(syn, ev, wf, (0.0, 0.3), seed=5, keep_components=True)
    b = render(rec, ev, wf, (0.0, 0.3), seed=5, keep_components=True)
    assert b.meta["impact_source"] == "recorded" and b.meta["click_scale"] > 0
    assert not np.allclose(a.clean, b.clean)
    assert np.sum(b.clean ** 2) == pytest.approx(np.sum(a.clean ** 2), rel=0.5)
    # channels without impacts are untouched
    for c in CHANNELS:
        if c != "bottom":
            assert np.allclose(a.components[c], b.components[c])
