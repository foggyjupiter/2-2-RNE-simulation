"""Spec 10.1 energy budget and numerical soundness under shaking.

The energy balance  Δ(KE + PE + E_elastic) = W_in − D  is checked with the
dissipation D computed from the same relative velocities that enter the
force law. Omitting the wall velocity from that relative velocity (the
classic bug) breaks the balance by >100 % and does not improve with smaller
dt; correct code closes it to ~0.1 % and converges with dt.
"""

import numpy as np
import pytest

from gransim.dynamics.engine import Simulation

from helpers import energy_residual

from gransim.config import SimConfig


def shaken(n=100, freq=8.0, dt=None, e=0.8, mu=0.3, **contact):
    return SimConfig(particles={"count": n},
                     excitation={"mode": "fixed_amplitude", "amplitude": 0.0134, "frequency": freq},
                     contact={"restitution_pp": e, "restitution_pw": e,
                              "friction_pp": mu, "friction_pw": mu, **contact},
                     time={"settle_time": 0.3, "dt": dt})


def run(cfg, duration):
    sim = Simulation(cfg, warn=False)
    sim.settle()
    sim.run(duration)
    return sim, sim.diag.as_arrays()


ENERGY_TOL = 0.01  # |residual| / W_in at the default dt (contact_time / 20)


@pytest.mark.parametrize("mu,rotation", [(0.3, True), (0.3, False), (0.0, True)])
def test_energy_balance_closes(mu, rotation):
    sim, d = run(shaken(mu=mu, rotation=rotation), 0.6)
    W = d["w_in"][-1]
    assert W > 1e-3  # the walls really inject energy (Γ ≈ 3.5)
    res = energy_residual(d)
    assert np.max(np.abs(res)) / W < ENERGY_TOL


def test_energy_balance_hertz():
    sim, d = run(shaken(n=60, model="hertz", modulus_scale=0.05), 0.4)
    W = d["w_in"][-1]
    assert W > 1e-3
    assert np.max(np.abs(energy_residual(d))) / W < ENERGY_TOL


def test_energy_residual_converges_with_dt():
    """Residual shrinks when dt is halved (discretisation, not a modelling error)."""
    r = []
    for dt in (1e-5, 5e-6):
        _, d = run(shaken(mu=0.0, dt=dt), 0.4)
        r.append(abs(energy_residual(d)[-1]) / d["w_in"][-1])
    assert r[1] < 0.6 * r[0]


def test_overlap_small():
    """Max overlap stays a few % of the diameter at the default relaxed stiffness (Γ ≈ 6.5, N = 300)."""
    sim, _ = run(shaken(n=300, freq=11.0), 0.5)
    assert sim.acc[5] < 0.05


def test_dt_convergence_of_bulk_state():
    """Halving dt leaves time-averaged kinetic energy and bed height unchanged.

    A single run fluctuates by ~5 % between seeds, so means over 3 seeds are compared.
    """
    out = {}
    for dt in (1e-5, 5e-6):
        ke, h = [], []
        for seed in range(3):
            cfg = shaken(n=100, dt=dt)
            cfg.seed = seed
            _, d = run(cfg, 1.5)
            m = d["t"] > 0.5
            ke.append(d["ke_trans"][m].mean())
            h.append(d["mean_height"][m].mean())
        out[dt] = (np.mean(ke), np.mean(h))
    (k1, h1), (k2, h2) = out[1e-5], out[5e-6]
    assert k2 == pytest.approx(k1, rel=0.10)
    assert h2 == pytest.approx(h1, rel=0.03)
