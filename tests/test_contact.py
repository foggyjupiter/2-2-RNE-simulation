"""Spec 10.1 basic mechanics: restitution, momentum, friction/rotation, elastic drift, resting."""

import numpy as np
import pytest

from gransim.dynamics.contact import (MODEL_HERTZ, MODEL_LINEAR, _single_collision_e,
                                      damping_for_restitution, hertz_contact_time)
from gransim.dynamics.engine import Simulation

from helpers import dissipated, run_until_separated, still_config, total_energy

E_VALUES = [0.3, 0.6, 0.9]


@pytest.mark.parametrize("model", [MODEL_LINEAR, MODEL_HERTZ])
@pytest.mark.parametrize("e", [0.1, 0.3, 0.6, 0.9, 0.99])
def test_restitution_table_roundtrip(model, e):
    ratio = 0.05 if model == MODEL_LINEAR else 1e-3
    damp = damping_for_restitution(e, model, ratio)
    offsets = (np.arange(8) + 0.5) / 8
    t_c = np.pi if model == MODEL_LINEAR else hertz_contact_time(1, 1, 0.75, 1)
    got = np.mean([_single_collision_e(model, damp, ratio * t_c, o) for o in offsets])
    assert got == pytest.approx(e, rel=2e-3)


def test_clipping_matters_at_low_e():
    """The unclipped textbook formula would give a different damping at low e (spec 7.1)."""
    e = 0.3
    zeta_textbook = -np.log(e) / np.sqrt(np.pi ** 2 + np.log(e) ** 2)
    zeta_table = damping_for_restitution(e, MODEL_LINEAR, 1e-3)
    assert abs(zeta_table - zeta_textbook) / zeta_textbook > 0.05


def _drop_config(model, e, mu=0.3):
    contact = {"model": model, "restitution_pw": e, "restitution_pp": e,
               "friction_pw": mu, "friction_pp": mu}
    return still_config(particles={"count": 1}, contact=contact)


@pytest.mark.parametrize("model", ["linear", "hertz"])
@pytest.mark.parametrize("e", E_VALUES)
def test_floor_bounce_restitution(model, e):
    """Free fall onto the bottom wall: rebound/impact speed ratio = e within 5%."""
    cfg = _drop_config(model, e)
    x0 = np.array([[0.025, 0.025, 0.02]])
    sim = Simulation(cfg, positions=x0, warn=False)
    R = sim.radius[0]
    (v0, _), (v1, _) = run_until_separated(sim, lambda s: s.x[0, 2] < R)
    ratio = -v1[0, 2] / v0[0, 2]
    assert ratio == pytest.approx(e, rel=0.05)


@pytest.mark.parametrize("model", ["linear", "hertz"])
@pytest.mark.parametrize("e", E_VALUES)
def test_head_on_collision(model, e):
    """Two equal spheres, no gravity: momentum conserved, separation speed = e × approach speed."""
    cfg = still_config(gravity=0.0, particles={"count": 2},
                       contact={"model": model, "restitution_pp": e})
    d = cfg.particles.diameter
    x0 = np.array([[0.02, 0.025, 0.03], [0.02 + 1.02 * d, 0.025, 0.03]])
    sim = Simulation(cfg, positions=x0, warn=False)
    sim.v[0] = [0.7, 0.0, 0.0]
    sim.v[1] = [-0.3, 0.0, 0.0]
    p0 = (sim.mass[:, None] * sim.v).sum(0)
    ke0 = total_energy(sim)
    contact = lambda s: np.linalg.norm(s.x[0] - s.x[1]) < s.radius.sum()
    (va, _), (vb, _) = run_until_separated(sim, contact)
    sim.step(1)  # flush the stored elastic energy of the last contact step
    p1 = (sim.mass[:, None] * sim.v).sum(0)
    assert np.allclose(p1, p0, rtol=0, atol=1e-12)
    ratio = (vb[1, 0] - vb[0, 0]) / (va[0, 0] - va[1, 0])
    assert ratio == pytest.approx(e, rel=0.05)
    # energy lost equals the logged dissipation (before/after: no stored elastic energy)
    assert ke0 - total_energy(sim) == pytest.approx(dissipated(sim), rel=0.02)


@pytest.mark.parametrize("offset", [0.2, 0.6])
@pytest.mark.parametrize("mu", [0.05, 0.5])
def test_oblique_collision_with_friction_and_rotation(offset, mu):
    """Linear and angular momentum (incl. spin) conserved; friction makes spin; energy not created."""
    cfg = still_config(gravity=0.0, particles={"count": 2},
                       contact={"restitution_pp": 0.8, "friction_pp": mu})
    d = cfg.particles.diameter
    x0 = np.array([[0.02, 0.025, 0.03], [0.02 + 1.01 * d, 0.025 + offset * d, 0.03]])
    sim = Simulation(cfg, positions=x0, warn=False)
    sim.v[0] = [0.5, 0.0, 0.1]
    sim.v[1] = [-0.5, 0.0, 0.0]
    sim.w[1] = [0.0, 0.0, 20.0]
    inertia = 0.4 * sim.mass * sim.radius ** 2

    def momenta(s):
        p = (s.mass[:, None] * s.v).sum(0)
        L = (s.mass[:, None] * np.cross(s.x, s.v)).sum(0) + (inertia[:, None] * s.w).sum(0)
        return p, L

    p0, L0 = momenta(sim)
    E0 = total_energy(sim)
    contact = lambda s: np.linalg.norm(s.x[0] - s.x[1]) < s.radius.sum()
    run_until_separated(sim, contact)
    sim.step(1)
    p1, L1 = momenta(sim)
    assert np.allclose(p1, p0, atol=1e-12)
    assert np.allclose(L1, L0, atol=1e-12 + 1e-9 * np.abs(L0).max())
    assert np.abs(sim.w[0]).max() > 1.0  # tangential force spun up particle 0
    E1 = total_energy(sim)
    assert E1 <= E0 * (1 + 1e-6)
    assert E0 - E1 == pytest.approx(dissipated(sim), rel=0.02, abs=1e-9 * E0)


def test_elastic_many_body_energy_conserved():
    """e = 1, μ = 0, gravity on, container still: total energy drift over 2 s < 0.5 %."""
    cfg = still_config(particles={"count": 60},
                       contact={"restitution_pp": 1.0, "restitution_pw": 1.0,
                                "friction_pp": 0.0, "friction_pw": 0.0})
    sim = Simulation(cfg, warn=False)
    sim.v[:] = np.random.default_rng(1).normal(0.0, 0.4, sim.v.shape)
    sim.step(1)
    E0 = total_energy(sim)
    for _ in range(20):
        sim.step(int(0.1 / sim.dt))
        assert total_energy(sim) == pytest.approx(E0, rel=5e-3)
    assert dissipated(sim) == pytest.approx(0.0, abs=1e-6 * E0)


@pytest.mark.parametrize("model", ["linear", "hertz"])
def test_resting_particle_stays_put(model):
    """A particle placed at rest on the bottom neither sinks nor bounces."""
    cfg = _drop_config(model, 0.8)
    sim = Simulation(cfg, positions=np.array([[0.025, 0.025, 0.003]]), warn=False)
    sim.settle(0.2)
    z0 = sim.x[0, 2]
    R = sim.radius[0]
    assert R - z0 < 0.005 * 2 * R  # static overlap < 0.5 % of d
    zs = []
    for _ in range(50):
        sim.step(int(0.02 / sim.dt))
        zs.append(sim.x[0, 2])
    assert np.ptp(zs) < 1e-7
    assert abs(sim.v[0, 2]) < 1e-5
