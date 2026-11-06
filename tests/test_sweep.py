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


def test_condition_jitter_keeps_nominal_value():
    vals = [300, 372, 450]
    plain = build_jobs(spec(conditions={"key": "excitation.rpm", "values": vals}))
    jit = build_jobs(spec(conditions={"key": "excitation.rpm", "values": vals, "jitter": 0.04}))
    assert [j.seed for j in plain] == [j.seed for j in jit]           # nothing else changes
    for a, b in zip(plain, jit):
        assert b.condition_value == a.params["excitation.rpm"]
        assert abs(b.params["excitation.rpm"] / b.condition_value - 1) <= 0.04
        assert {k: v for k, v in a.params.items() if k != "excitation.rpm"} == \
               {k: v for k, v in b.params.items() if k != "excitation.rpm"}
    assert len({round(j.params["excitation.rpm"] / j.condition_value, 6) for j in jit}) > 5


def test_noise_is_rescaled_to_other_rpm():
    from gransim.acoustics.noise import draw_noise
    ac = AcousticConfig()
    rng = np.random.default_rng(0)
    y372, m372 = draw_noise(ac, 372, 48000 * 2, rng)
    y558, m558 = draw_noise(ac, 558, 48000 * 2, np.random.default_rng(0))
    assert not m372["noise_rpm_scaled"] and m558["noise_rpm_scaled"]
    assert y558.size == 48000 * 2 and np.isfinite(y558).all()


def test_rerender_reproduces_and_adds_axes(tmp_path):
    """Re-rendering with no change gives the same audio (to 1 LSB); a new per-render axis
    is drawn from the render seed, so a system's conditions share it."""
    import json

    import yaml

    from gransim.acoustics.assets import read_wav
    from gransim.sweep.rerender import rerender_dataset
    from gransim.sweep.runner import run_sweep

    sp = {"base": "configs/experiment_replica.yaml",
          "overrides": {"contact": {"model": "hertz"}, "time": {"settle_time": 0.2, "record_time": 0.45}},
          "seed": 3, "n_dynamics": 1, "renders_per_run": 2, "sampling": "random",
          "conditions": {"key": "excitation.rpm", "values": [372, 450]},
          "axes": {"particles.count": {"dist": "int_uniform", "low": 3, "high": 4},
                   "acoustics.mic_gain": {"dist": "loguniform", "low": 0.65, "high": 1.3}}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(sp))
    src = tmp_path / "src"
    assert run_sweep(tmp_path / "spec.yaml", src, workers=1, progress=lambda m: None)["ok"] == 2

    same = tmp_path / "same"
    assert rerender_dataset(src, same, workers=1, progress=lambda m: None)["rows"] == 4
    for line in (same / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        _, a = read_wav(src / r["file_name"])
        _, b = read_wav(same / r["file_name"])
        assert np.abs(a - b).max() <= 2 / 32768

    new = tmp_path / "new"
    rerender_dataset(src, new, axes={"acoustics.impact_gain_sigma": {"dist": "uniform", "low": 0.6, "high": 1.5}},
                     fixed={"direct_attack_s": 2e-4}, workers=1, progress=lambda m: None)
    rows = [json.loads(x) for x in (new / "manifest.jsonl").read_text().splitlines()]
    by_render = {}
    for r in rows:
        assert 0.6 <= r["impact_gain_sigma"] <= 1.5 and r["direct_attack_s"] == 2e-4
        by_render.setdefault(r["render"], set()).add(r["impact_gain_sigma"])
    assert all(len(v) == 1 for v in by_render.values())       # shared by both conditions
    assert len({next(iter(v)) for v in by_render.values()}) == 2  # independent per render
