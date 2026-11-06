"""Spec 10.1 event consistency: one event per collision, merging, classification,
impulse ↔ momentum ↔ wall-force consistency, plausible event counts."""

import math

import numpy as np
import pytest

from gransim.config import SimConfig
from gransim.dynamics.engine import (E_FLAG, E_I, E_J, E_JN, E_KIND, E_TEND, E_TSTART, E_VN0,
                                     E_WALL, N_EV, KIND_PP, KIND_PW, Simulation)
from gransim.events.tracker import (IMPACT, SUSTAINED, classification_sensitivity,
                                    collision_interval_check, merge_and_classify,
                                    process_events, resolve_event_params)

from helpers import still_config


def record(sim, duration):
    sim.begin_recording(duration)
    sim.run(duration)
    raw, wf = sim.finish_recording()
    ep = resolve_event_params(sim.cfg, sim.cp)
    return process_events(raw, sim.radius, sim.mass, sim.f, ep), raw, wf, ep


def test_isolated_binary_collision_is_one_event():
    cfg = still_config(gravity=0.0, particles={"count": 2},
                       contact={"friction_pp": 0.0, "restitution_pp": 0.7})
    d = cfg.particles.diameter
    sim = Simulation(cfg, positions=np.array([[0.02, 0.025, 0.03],
                                              [0.02 + 1.05 * d, 0.025, 0.03]]), warn=False)
    sim.v[0] = [0.4, 0, 0]
    sim.v[1] = [-0.4, 0, 0]
    p0 = sim.mass[0] * sim.v[0, 0]
    ev, raw, wf, _ = record(sim, 0.01)
    assert ev.size == 1
    e = ev[0]
    assert e["kind"] == KIND_PP and e["contact_type"] == IMPACT and e["n_merged"] == 1
    assert e["v_rel_n"] == pytest.approx(0.8, rel=1e-6)
    # impulse equals the momentum change of particle 0
    assert e["J_n"] == pytest.approx(p0 - sim.mass[0] * sim.v[0, 0], rel=1e-9)
    assert e["duration"] == pytest.approx(cfg.contact.contact_time, rel=0.15)
    assert e["m_eff"] == pytest.approx(sim.mass[0] / 2)
    assert e["flags"] == 0


def test_bouncing_ball_impacts_then_rest():
    """Drop onto the floor: bounce times follow ballistics; final resting contact is sustained."""
    e_rest, h = 0.7, 0.02
    cfg = still_config(particles={"count": 1},
                       contact={"restitution_pw": e_rest, "friction_pw": 0.0})
    R = cfg.particles.diameter / 2
    sim = Simulation(cfg, positions=np.array([[0.025, 0.025, R + h]]), warn=False)
    ev, raw, wf, ep = record(sim, 1.0)
    floor = ev[ev["wall_id"] == 0]
    imp = floor[floor["contact_type"] == IMPACT]
    g = cfg.gravity
    v0 = math.sqrt(2 * g * h)
    assert imp.size >= 4
    # first touch after free fall from h
    assert imp["t_start"][0] == pytest.approx(math.sqrt(2 * h / g), rel=1e-3)
    assert imp["v_rel_n"][0] == pytest.approx(v0, rel=1e-3)
    # each flight between bounces is ballistic: gap = 2 v / g with v the next impact speed
    flights = imp["t_start"][1:4] - imp["t_end"][:3]
    assert np.allclose(flights, 2 * imp["v_rel_n"][1:4] / g, rtol=0.01)
    # successive impact speeds shrink by e (5 %, per-collision numerical scatter included)
    assert np.allclose(imp["v_rel_n"][1:4] / imp["v_rel_n"][:3], e_rest, rtol=0.05)
    # first impulse = m (v_in + v_out)
    m = sim.mass[0]
    assert imp["J_n"][0] == pytest.approx(m * v0 * (1 + e_rest), rel=0.02)
    # the ball ends at rest: last floor contact is sustained and still open at the end
    last = floor[-1]
    assert last["contact_type"] == SUSTAINED and last["flags"] & 2


def test_resting_particle_contact_is_sustained():
    cfg = still_config(particles={"count": 1})
    sim = Simulation(cfg, positions=np.array([[0.025, 0.025, 0.003]]), warn=False)
    sim.settle(0.2)
    ev, *_ = record(sim, 0.5)
    assert ev.size == 1
    assert ev[0]["contact_type"] == SUSTAINED and ev[0]["wall_id"] == 0
    # the whole weight × time goes through this contact
    assert ev[0]["J_n"] == pytest.approx(sim.mass[0] * cfg.gravity * 0.5, rel=1e-3)


def _raw(segments, kind=KIND_PW, wall=0, i=0, j=-1):
    rows = np.zeros((len(segments), N_EV))
    for r, (ts, te, vn, jn) in enumerate(segments):
        rows[r, E_TSTART], rows[r, E_TEND], rows[r, E_VN0], rows[r, E_JN] = ts, te, vn, jn
        rows[r, E_KIND], rows[r, E_WALL], rows[r, E_I], rows[r, E_J] = kind, wall, i, j
    return rows


def test_chatter_is_merged():
    gap, v_imp, dmax = 2e-4, 0.03, 6e-4
    # one real impact followed by on/off chatter at near-zero speed with tiny gaps
    segs = [(0.0, 2e-4, 0.5, 1e-4)] + [(2e-4 + k * 1e-4, 2.5e-4 + k * 1e-4, 1e-3, 1e-7)
                                       for k in range(1, 8)]
    rows, n_merged, ctype = merge_and_classify(_raw(segs), gap, v_imp, dmax)
    assert rows.shape[0] == 1 and n_merged[0] == 8
    assert rows[0, E_JN] == pytest.approx(1e-4 + 7e-7)
    assert rows[0, E_TEND] == pytest.approx(segs[-1][1])


def test_separate_collisions_not_merged():
    gap, v_imp, dmax = 2e-4, 0.03, 6e-4
    # (a) far apart in time; (b) close in time but a genuine (fast) re-impact
    far = _raw([(0.0, 2e-4, 0.5, 1e-4), (0.05, 0.0502, 0.4, 1e-4)])
    close_fast = _raw([(0.0, 2e-4, 0.5, 1e-4), (2.5e-4, 4.5e-4, 0.4, 1e-4)])
    other_pair = np.vstack([_raw([(0.0, 2e-4, 0.5, 1e-4)], i=0),
                            _raw([(2.1e-4, 4e-4, 0.001, 1e-7)], i=1)])
    for raw in (far, close_fast):
        rows, n_merged, ctype = merge_and_classify(raw, gap, v_imp, dmax)
        assert rows.shape[0] == 2 and np.all(n_merged == 1) and np.all(ctype == IMPACT)
    rows, n_merged, _ = merge_and_classify(other_pair, gap, v_imp, dmax)
    assert rows.shape[0] == 2


def shaken(n=60, mu=0.3, freq=9.0):
    return SimConfig(particles={"count": n},
                     excitation={"mode": "fixed_amplitude", "amplitude": 0.0134, "frequency": freq},
                     contact={"model": "linear", "friction_pp": mu, "friction_pw": mu},
                     time={"settle_time": 0.3})


@pytest.fixture(scope="module")
def shaken_frictionless():
    sim = Simulation(shaken(mu=0.0), warn=False)
    sim.settle()
    sim.run(0.3)
    P0 = (sim.mass[:, None] * sim.v).sum(0)
    t0 = sim.t
    ev, raw, wf, ep = record(sim, 0.4)
    P1 = (sim.mass[:, None] * sim.v).sum(0)
    return sim, ev, raw, wf, ep, P0, P1, sim.t - t0


def test_wall_force_integral_matches_event_impulse(shaken_frictionless):
    sim, ev, raw, wf, *_ = shaken_frictionless
    for w in range(6):
        J_force = wf.force[w].sum() / wf.fs
        J_events = ev["J_n"][(ev["kind"] == KIND_PW) & (ev["wall_id"] == w)].sum()
        assert J_force == pytest.approx(J_events, rel=1e-9, abs=1e-15)
    assert wf.force[0].sum() > 0  # the bottom is hit


def test_event_impulses_match_momentum_change(shaken_frictionless):
    """μ = 0: wall impulses (normal only) + gravity = change of total momentum."""
    sim, ev, raw, wf, ep, P0, P1, T = shaken_frictionless
    pw = ev[ev["kind"] == KIND_PW]
    J = {w: pw["J_n"][pw["wall_id"] == w].sum() for w in range(6)}
    M = sim.mass.sum()
    assert P1[2] - P0[2] == pytest.approx(J[0] - J[1] - M * sim.cfg.gravity * T, abs=1e-9)
    assert P1[0] - P0[0] == pytest.approx(J[2] - J[3], abs=1e-9)
    assert P1[1] - P0[1] == pytest.approx(J[4] - J[5], abs=1e-9)
    # every pp impulse is internal; just check they are recorded and positive
    assert np.all(ev["J_n"] >= 0) and (ev["kind"] == KIND_PP).sum() > 0


def test_event_counts_plausible_and_stiffness_condition(shaken_frictionless):
    sim, ev, raw, wf, ep, *_ = shaken_frictionless
    T = 0.4
    cycles = T * sim.f
    imp = ev[ev["contact_type"] == IMPACT]
    per_particle_cycle = 2 * imp.size / sim.n / cycles  # each impact involves ≤ 2 particles
    assert 0.5 < per_particle_cycle < 500
    assert np.all(ev["duration"] > 0)
    assert np.all((ev["phase"] >= 0) & (ev["phase"] < 1))
    chk = collision_interval_check(ev, ep["binary_contact_time"])
    assert chk["ok"], chk


def test_classification_sensitivity_monotone(shaken_frictionless):
    sim, ev, raw, wf, ep, *_ = shaken_frictionless
    s = classification_sensitivity(raw, ep, factors=(0.5, 1.0, 2.0))
    n = s["n_impacts"]
    # higher v_impact -> fewer impacts; longer duration_max -> more impacts
    assert np.all(np.diff(n, axis=0) <= 0)
    assert np.all(np.diff(n, axis=1) >= 0)
