"""Black-box acoustics: draws are reproducible and physical, nothing fitted is used."""

import math

import numpy as np

from gransim.acoustics import blackbox as bb
from gransim.config import SimConfig


def test_draws_reproducible_and_in_range():
    a, b = bb.draw_params(123), bb.draw_params(123)
    assert a == b
    for s in range(300):
        p = bb.draw_params(s)
        for k, (dist, lo, hi) in bb.RANGES.items():
            assert lo - 1e-9 <= getattr(p, k) <= hi + 1e-9, k
        assert p.noise_clip in bb.NOISE_CLIPS            # evaluation clips 6-15 never used
        rc2 = p.directivity_q * p.room_volume / (100 * math.pi * p.rt60)
        assert math.isclose(p.drr, rc2 / p.mic_distance ** 2)


def test_drr_is_room_physics():
    drr_db = [10 * math.log10(bb.draw_params(s).drr) for s in range(2000)]
    # a phone 1.5-3.5 m away in a 30-150 m³ room: beyond the critical distance
    assert -20 < min(drr_db) and max(drr_db) < 5
    assert -12 < float(np.median(drr_db)) < -4


def test_plate_modes_match_thin_plate_theory():
    cfg = SimConfig()
    k, f_c = bb.plate_constants(cfg)
    f, _ = bb.plate_modes(0.05, 0.05, k, 1.0, 24000.0, np.random.default_rng(0))
    # simply supported 5x5 cm, 3 mm acrylic: f_11 ≈ 1.9 kHz, 13 modes below 24 kHz
    assert len(f) == 13
    assert 1700 < f.min() < 2200
    assert 11000 < f_c < 13500


def test_irs_unit_energy_and_hf_decays_faster():
    cfg = SimConfig()
    p = bb.draw_params(7)
    irs = bb.channel_irs(cfg, p, 7)
    assert set(irs) == {"bottom", "top", "side", "pp"}
    T = bb.rt60_of_f(np.array([1000.0, 16000.0]), p)
    assert math.isclose(T[0], p.rt60, rel_tol=1e-6) and T[1] < T[0]
    for h in irs.values():
        assert np.all(np.isfinite(h))
        # unit energy before the microphone curve (±few dB)
        assert 0.3 < float(np.sum(h * h)) < 3.0
