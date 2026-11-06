"""Stiffness convergence test (spec 10.1, v2 addition).

The relaxed-stiffness / acoustic-τc separation (spec 6.2) holds only if the
dynamics statistics that feed the acoustics do not depend on the contact
stiffness. We run representative conditions with the relaxed stiffness ×1,
×4, ×16 and with real (Hertz) stiffness, and compare against the real-
stiffness reference, relative to seed-to-seed scatter.

The smallest stiffness factor that passes every condition is the
"allowed stiffness" for data generation; it is stored in
``assets/reference/allowed_stiffness.yaml`` and configs using a softer
contact trigger a warning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from gransim.config import SimConfig, deep_merge

ALLOWED_PATH = Path(__file__).resolve().parents[3] / "assets" / "reference" / "allowed_stiffness.yaml"

# metric -> (kind, tolerance). "rel": |a-b|/|b|; "abs": |a-b|; "tv": total variation of histograms.
DEFAULT_CRITERIA = {
    "impulse_sq_rate": ("rel", 0.20),      # energy of the impact-pulse sound component
    "impulse_sq_rate_pw": ("rel", 0.20),   # same, wall hits only
    "J_q50": ("rel", 0.25),                # typical impact impulse
    "J_q90": ("rel", 0.20),                # strong impacts
    "phase_hist_J2": ("tv", 0.15),         # when in the cycle the sound energy arrives
    "collectivity": ("abs", 0.10),         # ground-truth phase indicators
    "phase_concentration_impulse": ("abs", 0.10),
    "height_mean": ("rel", 0.10),
}
# reported but not used for pass/fail (expected to depend on stiffness; see README)
INFO_METRICS = ("impact_rate", "impact_rate_pw", "impact_rate_pp", "free_flight",
                "sustained_fraction", "interval_ratio")


def stiffness_variants(base: SimConfig, factors=(1.0, 4.0, 16.0), include_real: bool = True,
                       hertz_scales=()) -> list[tuple[str, dict]]:
    """(label, config overrides) for each stiffness level.

    Linear: factor f scales k_n by f (contact time by 1/√f), label ``x{f}``.
    Hertz:  modulus scale s multiplies the real Young's moduli (contact time ∝ s^-0.4),
            label ``h{s}``; ``real`` is Hertz with s = 1.
    """
    tc = base.contact.contact_time
    # modulus_scale is irrelevant for the linear model; pin it so case hashes are stable
    out = [(f"x{f:g}", {"contact": {"model": "linear", "contact_time": tc / math.sqrt(f),
                                    "modulus_scale": 1.0}})
           for f in factors]
    out += [(f"h{s:g}", {"contact": {"model": "hertz", "modulus_scale": float(s)}})
            for s in hertz_scales]
    if include_real:
        out.append(("real", {"contact": {"model": "hertz", "modulus_scale": 1.0}}))
    return out


def build_cases(base: SimConfig, conditions: list[tuple[int, float]], seeds=(0, 1),
                factors=(1.0, 4.0, 16.0), include_real: bool = True,
                hertz_scales=()) -> list[dict]:
    """Expand (N, Γ) conditions × stiffness variants × seeds into config dicts with labels."""
    from gransim.config import frequency_from_gamma

    A = base.excitation.amplitude
    cases = []
    for N, gam in conditions:
        for label, ov in stiffness_variants(base, factors, include_real, hertz_scales):
            for seed in seeds:
                d = deep_merge(base.model_dump(mode="json"), ov)
                d = deep_merge(d, {"seed": seed, "particles": {"count": N},
                                   "excitation": {"mode": "fixed_amplitude", "amplitude": A,
                                                  "frequency": frequency_from_gamma(gam, A),
                                                  "rpm": None, "gamma": None}})
                cases.append({"N": N, "gamma_target": gam, "stiffness": label, "seed": seed,
                              "config": d})
    return cases


def _compare(kind: str, a, b) -> float:
    if kind == "rel":
        return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else math.inf)
    if kind == "abs":
        return abs(a - b)
    if kind == "tv":
        return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())
    raise ValueError(kind)


def _mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows]
    if isinstance(vals[0], list):
        return np.mean(np.asarray(vals), axis=0)
    return float(np.mean(vals))


def evaluate(cases: list[dict], results: list[dict], criteria: Optional[dict] = None,
             reference: str = "real") -> dict:
    """Compare every stiffness level with the reference, per (N, Γ) condition.

    A metric passes when its deviation from the reference (seed means) is within
    the tolerance *or* within 2× the seed-to-seed deviation of the reference
    itself (so pure noise never fails a level).
    """
    crit = criteria or DEFAULT_CRITERIA
    groups: dict = {}
    for c, r in zip(cases, results):
        groups.setdefault((c["N"], c["gamma_target"]), {}).setdefault(c["stiffness"], []).append(r)
    rows = []
    level_pass: dict = {}
    for (N, gam), levels in sorted(groups.items()):
        ref = levels[reference]
        for label, rs in levels.items():
            row = {"N": N, "gamma": gam, "stiffness": label, "pass": True, "fails": []}
            for m, (kind, tol) in crit.items():
                a, b = _mean(rs, m), _mean(ref, m)
                dev = _compare(kind, a, b)
                noise = 2.0 * _compare(kind, ref[0][m], ref[-1][m]) if len(ref) > 1 else 0.0
                row[m] = a if not isinstance(a, np.ndarray) else None
                row[f"{m}_dev"] = dev
                if dev > max(tol, noise):
                    row["pass"] = False
                    row["fails"].append(m)
            for m in INFO_METRICS:
                if m in rs[0]:
                    row[m] = _mean(rs, m)
            row["elapsed_s"] = float(np.mean([r["elapsed_s"] for r in rs]))
            rows.append(row)
            level_pass[label] = level_pass.get(label, True) and row["pass"]
    return {"rows": rows, "level_pass": level_pass, "criteria": {k: list(v) for k, v in crit.items()}}


def allowed_levels(summary: dict, base: SimConfig, factors=(1.0, 4.0, 16.0),
                   hertz_scales=()) -> dict:
    """Softest passing level per contact model (None when no level of that model passed).

    linear: largest passing contact_time; hertz: smallest passing modulus_scale
    (``real`` always counts as the stiffest Hertz level).
    """
    lp = summary["level_pass"]
    lin = None
    for f in sorted(factors):
        if lp.get(f"x{f:g}", False):
            lin = {"factor": f, "max_contact_time": base.contact.contact_time / math.sqrt(f)}
            break
    hz = None
    for s_ in sorted(hertz_scales) + [1.0]:
        label = "real" if s_ == 1.0 else f"h{s_:g}"
        if lp.get(label, False):
            hz = {"min_modulus_scale": float(s_)}
            break
    return {"linear": lin, "hertz": hz}


def save_allowed(allowed: dict, summary: dict, path: Path = ALLOWED_PATH, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "Softest contact stiffness per model that passed the stiffness convergence "
                       "test (gransim stiffness-test) against real Hertz stiffness. Softer contacts "
                       "must not be used for dataset generation (spec 10.1). null = no level of "
                       "that model passed.",
        "linear": allowed["linear"],
        "hertz": allowed["hertz"],
        "level_pass": {k: bool(v) for k, v in summary["level_pass"].items()},
        "note": note,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def load_allowed(path: Path = ALLOWED_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def stiffness_warning(cfg: SimConfig, allowed: Optional[dict]) -> Optional[str]:
    """Warning text if cfg uses a contact softer than the recorded allowed stiffness."""
    if not allowed:
        return None
    c = cfg.contact
    if c.model == "linear":
        lin = allowed.get("linear")
        if lin is None:
            return ("contact.model = linear did not pass the stiffness convergence test at any "
                    "stiffness (dense states deviate from real Hertz contacts); use model = hertz "
                    "for dataset generation")
        if c.contact_time > lin["max_contact_time"] * 1.0001:
            return (f"contact.contact_time = {c.contact_time:.3g} s is softer than the allowed "
                    f"{lin['max_contact_time']:.3g} s (stiffness convergence test)")
    else:
        hz = allowed.get("hertz")
        if hz and c.modulus_scale < hz["min_modulus_scale"] * 0.9999:
            return (f"contact.modulus_scale = {c.modulus_scale:g} is softer than the allowed "
                    f"{hz['min_modulus_scale']:g} (stiffness convergence test)")
    return None


def write_report(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stiffness_convergence.json").write_text(json.dumps(summary, indent=2, default=float))
