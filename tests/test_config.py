import math

import pytest
from pydantic import ValidationError

from gransim.config import (SimConfig, amplitude_from_gamma, derived_quantities,
                            frequency_from_gamma, gamma_from_amplitude)


def test_gamma_conversions_roundtrip():
    A, f = 0.0134, 16.7
    gam = gamma_from_amplitude(A, f)
    assert gam == pytest.approx(15.0, rel=0.01)  # spec 2: Γ=15 @ 16.7 Hz ⇔ A≈1.34 cm
    assert amplitude_from_gamma(gam, f) == pytest.approx(A)
    assert frequency_from_gamma(gam, A) == pytest.approx(f)


def test_fixed_amplitude_rejects_gamma():
    with pytest.raises(ValidationError, match="fixed_amplitude"):
        SimConfig(excitation={"mode": "fixed_amplitude", "amplitude": 0.01,
                              "frequency": 10, "gamma": 3.0})


def test_fixed_amplitude_derives_gamma_from_rpm():
    cfg = SimConfig(excitation={"mode": "fixed_amplitude", "amplitude": 0.0134, "rpm": 600})
    A, f, gam = cfg.excitation.resolved()
    assert f == pytest.approx(10.0)
    assert gam == pytest.approx(0.0134 * (2 * math.pi * 10) ** 2 / 9.81)


def test_free_mode_needs_exactly_two():
    SimConfig(excitation={"mode": "free", "amplitude": None, "frequency": 8, "gamma": 3})
    with pytest.raises(ValidationError, match="exactly two"):
        SimConfig(excitation={"mode": "free", "amplitude": 0.01, "frequency": 8, "gamma": 3})
    cfg = SimConfig(excitation={"mode": "free", "amplitude": None, "frequency": 8, "gamma": 3})
    A, f, gam = cfg.excitation.resolved()
    assert gamma_from_amplitude(A, f) == pytest.approx(3.0)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        SimConfig(particles={"count": 10, "colour": "red"})


def test_derived_quantities():
    cfg = SimConfig(particles={"count": 300}, contact={"model": "linear"})
    dq = derived_quantities(cfg)
    assert dq["layers_F"] == pytest.approx(300 * 0.006 ** 2 / 0.05 ** 2)  # ≈ 4.3 (spec 2)
    assert dq["particle_to_container_mass"] == pytest.approx(300 * 1.2e-4 / 0.07)
    assert dq["steps_per_contact"] == pytest.approx(20.0)  # e = 0.8 needs no refinement
    low_e = derived_quantities(SimConfig(contact={"model": "linear", "restitution_pp": 0.3}))
    assert low_e["steps_per_contact"] >= 40 and low_e["restitution_scatter"] <= 0.05


def test_default_is_converged_hertz():
    """Production default = softest stiffness that passed the convergence test (spec 10.1)."""
    cfg = SimConfig()
    assert cfg.contact.model == "hertz" and cfg.contact.modulus_scale == 0.25
    dq = derived_quantities(cfg)
    assert 1.5 < dq["contact_time_pp_s"] / derived_quantities(
        SimConfig(contact={"modulus_scale": 1.0}))["contact_time_pp_s"] < 2.0


def test_softer_than_allowed_warns():
    from gransim.analysis.stiffness import stiffness_warning
    allowed = {"linear": None, "hertz": {"min_modulus_scale": 0.25}}
    assert stiffness_warning(SimConfig(), allowed) is None
    assert "softer" in stiffness_warning(SimConfig(contact={"modulus_scale": 0.1}), allowed)
    assert "linear" in stiffness_warning(SimConfig(contact={"model": "linear"}), allowed)
