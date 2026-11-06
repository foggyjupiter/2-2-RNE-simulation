"""Parameter sampling for dataset sweeps (spec 8.1, 8.2).

A sweep spec (YAML) lists *axes* as dotted config keys with a distribution:

    axes:
      particles.count:       {dist: int_uniform, low: 0, high: 100}
      particles.mass:        {dist: loguniform, low: 1.2e-4, high: 4.0e-4}
      contact.restitution_pw: {dist: uniform, low: 0.55, high: 0.9}

Physical axes are sampled per dynamics run with a Latin hypercube (scipy.stats.qmc);
acoustic axes (keys starting with ``acoustics.``) are drawn independently per render,
except acoustic keys listed in ``blind_axes`` (material properties such as the true
modulus), which belong to the run.
Contrast pairs (spec 8.1): (a) same N, different material; (b) same material,
different N. A pair's two runs share a ``pair_id``.

Multi-condition sweeps (``conditions: {key: excitation.rpm, values: [...]}``): every
sampled system is simulated under each condition (same N, material, nuisance and
recording setup); its jobs share a ``system_id`` and always land in the same split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

DISTS = ("uniform", "loguniform", "int_uniform", "choice", "normal")


def _from_unit(u: float, spec: dict):
    d = spec["dist"]
    # PyYAML reads "1.5e9" (no sign in the exponent) as a string
    spec = {k: (float(v) if k in ("low", "high", "mean", "std") else v) for k, v in spec.items()}
    if d == "uniform":
        return spec["low"] + u * (spec["high"] - spec["low"])
    if d == "loguniform":
        return math.exp(math.log(spec["low"]) + u * (math.log(spec["high"]) - math.log(spec["low"])))
    if d == "int_uniform":
        return int(min(spec["low"] + math.floor(u * (spec["high"] - spec["low"] + 1)), spec["high"]))
    if d == "choice":
        vals = spec["values"]
        return vals[min(int(u * len(vals)), len(vals) - 1)]
    if d == "normal":
        from scipy.stats import norm
        return float(spec["mean"] + spec["std"] * norm.ppf(min(max(u, 1e-9), 1 - 1e-9)))
    raise ValueError(f"unknown dist {d!r}; use one of {DISTS}")


def sample_axes(axes: dict, n: int, rng: np.random.Generator, method: str = "lhs") -> list[dict]:
    keys = list(axes)
    if not keys or n == 0:
        return [{} for _ in range(n)]
    if method == "lhs":
        U = qmc.LatinHypercube(d=len(keys), seed=rng).random(n)
    elif method == "random":
        U = rng.random((n, len(keys)))
    else:
        raise ValueError("sampling must be 'lhs' or 'random'")
    return [{k: _from_unit(float(U[i, j]), axes[k]) for j, k in enumerate(keys)} for i in range(n)]


@dataclass
class Job:
    run_id: str
    params: dict                      # dotted key → value (dynamics side)
    seed: int
    pair_id: str = ""
    pair_type: str = ""               # "", "same_N", "same_material"
    render_params: list = field(default_factory=list)  # one dict per render (acoustics.* keys)
    # multi-condition sweeps: one *system* (N, material, recording setup) observed under
    # several excitation conditions; all its jobs share system_id, render settings, render
    # seeds (same responses / noise choice) and split
    system_id: str = ""
    condition_index: int = 0
    render_seeds: list = field(default_factory=list)
    split_seed: int | None = None


def build_jobs(spec: dict) -> list[Job]:
    """All dynamics jobs of a sweep (deterministic given spec['seed'])."""
    rng = np.random.default_rng(spec.get("seed", 0))
    axes = spec["axes"]
    blind = set(spec.get("blind_axes", []))
    # per-run axes: everything physical, incl. material properties that only the acoustics
    # use (e.g. the true modulus); per-render axes: the remaining acoustic conditions
    per_render = {k for k in axes if k.startswith("acoustics.") and k not in blind}
    dyn_axes = {k: v for k, v in axes.items() if k not in per_render}
    ac_axes = {k: v for k, v in axes.items() if k in per_render}
    count_key = spec.get("label_key", "particles.count")
    n_main = int(spec["n_dynamics"])
    K = int(spec.get("renders_per_run", 1))
    jobs: list[Job] = []

    def renders():
        return sample_axes(ac_axes, K, rng, "random")

    for i, p in enumerate(sample_axes(dyn_axes, n_main, rng, spec.get("sampling", "lhs"))):
        jobs.append(Job(f"r{i:05d}", p, int(rng.integers(1 << 31)), render_params=renders()))

    pairs = spec.get("contrast_pairs", {}) or {}
    # (a) same N, different material
    for k in range(int(pairs.get("same_N", 0))):
        a, b = sample_axes(dyn_axes, 2, rng, "random")
        b[count_key] = a[count_key]
        for side, p in (("a", a), ("b", b)):
            jobs.append(Job(f"pN{k:04d}{side}", p, int(rng.integers(1 << 31)), f"pN{k:04d}",
                            "same_N", renders()))
    # (b) same material (and nuisance), different N
    for k in range(int(pairs.get("same_material", 0))):
        a, b = sample_axes(dyn_axes, 2, rng, "random")
        b = {**a, count_key: b[count_key]}
        for side, p in (("a", a), ("b", b)):
            jobs.append(Job(f"pM{k:04d}{side}", p, int(rng.integers(1 << 31)), f"pM{k:04d}",
                            "same_material", renders()))
    return expand_conditions(jobs, spec)


def expand_conditions(jobs: list[Job], spec: dict) -> list[Job]:
    """Multi-condition sweeps (spec 'conditions': {key, values}): each sampled system is
    run once per value. Draws nothing from the sampling RNG, so specs without
    'conditions' produce exactly the same jobs as before."""
    cond = spec.get("conditions")
    if not cond:
        return jobs
    key, values = cond["key"], list(cond["values"])
    out = []
    for j in jobs:
        rs = [int(np.random.default_rng([j.seed, 7, k]).integers(1 << 31))
              for k in range(len(j.render_params))]
        for c, v in enumerate(values):
            out.append(Job(f"{j.run_id}_c{c}", {**j.params, key: v},
                           int(np.random.default_rng([j.seed, 11, c]).integers(1 << 31)),
                           j.pair_id, j.pair_type, j.render_params, system_id=j.run_id,
                           condition_index=c, render_seeds=rs, split_seed=j.seed))
    return out


def nested(dotted: dict) -> dict:
    """{'a.b': 1, 'a.c': 2} → {'a': {'b': 1, 'c': 2}}."""
    out: dict = {}
    for k, v in dotted.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def in_box(params: dict, box: dict) -> bool:
    """True if every key's value lies in its [low, high] (used for held-out regions)."""
    for k, (lo, hi) in box.items():
        v = params.get(k)
        if v is None or not (lo <= v <= hi):
            return False
    return True
