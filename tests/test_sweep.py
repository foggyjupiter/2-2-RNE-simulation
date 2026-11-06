"""Sweep sampling: multi-condition systems share split, render settings and seeds;
single-condition specs are unchanged; rpm-scaled background noise."""

import numpy as np

from gransim.config import AcousticConfig
from gransim.sweep.runner import _split_for
from gransim.sweep.sampling import build_jobs

AXES = {"particles.count": {"dist": "int_uniform", "low": 0, "high": 100},
        "contact.restitution_pw": {"dist": "uniform", "low": 0.55, "high": 0.92},
        "acoustics.mic_gain": {"dist": "loguniform", "low": 0.65, "high": 1.3}}


def spec(**kw):
    return {"seed": 1, "n_dynamics": 6, "renders_per_run": 2, "axes": AXES,
            "contrast_pairs": {"same_N": 1}, **kw}


def test_single_condition_jobs_unchanged_by_condition_support():
    a = build_jobs(spec())
    assert len(a) == 8 and all(j.system_id == "" for j in a)
    b = build_jobs(spec())
    assert [(j.run_id, j.params, j.seed) for j in a] == [(j.run_id, j.params, j.seed) for j in b]


def test_multi_condition_systems_share_everything_but_the_condition():
    vals = [300, 372, 450]
    jobs = build_jobs(spec(conditions={"key": "excitation.rpm", "values": vals}))
    assert len(jobs) == 8 * 3
    single = build_jobs(spec())
    by_sys = {}
    for j in jobs:
        by_sys.setdefault(j.system_id, []).append(j)
    for s0 in single:
        group = by_sys[s0.run_id]
        assert [j.params["excitation.rpm"] for j in group] == vals
        for j in group:
            rest = {k: v for k, v in j.params.items() if k != "excitation.rpm"}
            assert rest == s0.params
            assert j.render_params == s0.render_params and j.render_seeds == group[0].render_seeds
            assert j.pair_id == s0.pair_id
        sp = {_split_for(j, {"holdout_box": {}}, np.random.default_rng(j.split_seed ^ 0x5EED))
              for j in group}
        assert len(sp) == 1                              # whole system in one split
        assert len({j.seed for j in group}) == len(vals)  # independent dynamics


def test_noise_is_rescaled_to_other_rpm():
    from gransim.acoustics.noise import draw_noise
    ac = AcousticConfig()
    rng = np.random.default_rng(0)
    y372, m372 = draw_noise(ac, 372, 48000 * 2, rng)
    y558, m558 = draw_noise(ac, 558, 48000 * 2, np.random.default_rng(0))
    assert not m372["noise_rpm_scaled"] and m558["noise_rpm_scaled"]
    assert y558.size == 48000 * 2 and np.isfinite(y558).all()
