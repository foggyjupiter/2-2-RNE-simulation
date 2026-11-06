"""Spec 10.1 stiffness convergence.

Fast: the recorded allowed stiffness exists and the default config is not softer.
Slow (``pytest -m slow``): re-run the hardest condition (N = 300, Γ = 2.3, dense) for the
default relaxed Hertz stiffness and for real stiffness and require agreement.
The full matrix is produced by ``gransim stiffness-test`` (results/stiffness/).
"""

import pytest

from gransim.analysis.stiffness import build_cases, evaluate, load_allowed, stiffness_warning
from gransim.analysis.validation import run_many
from gransim.config import SimConfig


def test_allowed_stiffness_recorded_and_default_complies():
    allowed = load_allowed()
    assert allowed is not None, "run `gransim stiffness-test --record`"
    assert allowed["hertz"]["min_modulus_scale"] <= SimConfig().contact.modulus_scale
    assert stiffness_warning(SimConfig(), allowed) is None


@pytest.mark.slow
def test_default_stiffness_matches_real_in_dense_state():
    base = SimConfig(time={"settle_time": 0.5, "discard_time": 1.0, "record_time": 2.0})
    s = base.contact.modulus_scale
    cases = build_cases(base, [(300, 2.3)], seeds=(0,), factors=(), include_real=True,
                        hertz_scales=(s,))
    res = run_many([c["config"] for c in cases], workers=2)
    summary = evaluate(cases, res, reference="real")
    assert summary["level_pass"][f"h{s:g}"], summary["rows"]
