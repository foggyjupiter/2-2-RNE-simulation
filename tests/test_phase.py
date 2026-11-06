"""Ground-truth phase indicators and critical-Γ extraction (unit level)."""

import numpy as np
import pytest

from gransim.analysis.critical import fit_transition, midpoint_crossing, variance_peak
from gransim.analysis.phase import contact_time_fraction, phase_indicators
from gransim.dynamics.engine import Simulation
from gransim.events.tracker import EVENT_DTYPE, process_events, resolve_event_params

from helpers import still_config


def test_logistic_fit_recovers_gamma_c():
    g = np.linspace(1, 8, 15)
    y = 0.1 + 0.8 / (1 + np.exp(-(g - 3.3) / 0.4))
    y += np.random.default_rng(0).normal(0, 0.01, g.size)
    r = fit_transition(g, y)
    assert r["method"] == "logistic"
    assert r["gamma_c"] == pytest.approx(3.3, abs=0.1)
    assert r["width"] == pytest.approx(0.4, rel=0.3)


def test_midpoint_crossing_and_variance_peak():
    assert midpoint_crossing([1, 2, 3, 4], [1.0, 0.9, 0.1, 0.0]) == pytest.approx(2.5)
    runs = np.array([[1, 1, 1], [0.9, 0.2, 0.6], [0, 0, 0.1]])
    assert variance_peak([1, 2, 3], runs) == 2


def test_contact_union_fraction():
    ev = np.zeros(4, EVENT_DTYPE)
    # particle 0: [0,1) and [0.5,2) overlap -> 2; particle 1 (as j): [3,4) -> 1, and own [0,1) -> 2
    ev["i"] = [0, 0, 2, 1]
    ev["j"] = [-1, -1, 1, -1]
    ev["t_start"] = [0.0, 0.5, 3.0, 0.0]
    ev["t_end"] = [1.0, 2.0, 4.0, 1.0]
    frac = contact_time_fraction(ev, 3, 10.0)
    assert frac == pytest.approx([0.2, 0.2, 0.1])


def _indicators(sim, duration):
    sim.begin_recording(duration)
    t0 = sim.t
    sim.run(duration)
    raw, _ = sim.finish_recording()
    ev = process_events(raw, sim.radius, sim.mass, sim.f, resolve_event_params(sim.cfg, sim.cp))
    return phase_indicators(ev, sim.diag.as_arrays(), sim.n, (t0, sim.t))


def test_resting_bed_is_solid_like():
    sim = Simulation(still_config(particles={"count": 30}), warn=False)
    sim.settle()
    ind = _indicators(sim, 0.3)
    assert ind["free_flight"] < 0.01
    assert ind["sustained_fraction"] > 0.99


def test_granular_gas_is_flying():
    cfg = still_config(gravity=0.0, particles={"count": 20},
                       contact={"restitution_pp": 1.0, "restitution_pw": 1.0,
                                "friction_pp": 0.0, "friction_pw": 0.0})
    sim = Simulation(cfg, warn=False)
    sim.v[:] = np.random.default_rng(3).normal(0, 0.5, sim.v.shape)
    ind = _indicators(sim, 0.3)
    assert ind["free_flight"] > 0.95
    assert ind["collectivity"] < 0.3
    assert ind["phase_concentration"] < 0.3  # no shaking -> no preferred phase
