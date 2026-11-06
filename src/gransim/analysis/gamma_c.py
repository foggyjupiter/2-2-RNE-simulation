"""Γc(N) reproduction (spec 10.1): sweep Γ at fixed amplitude, extract Γc per N, compare.

Order parameters used for Γc (each gives its own Γc):

* ``kurtosis_total_hp`` — excess kurtosis of the high-passed total wall force, the
  pre-acoustic proxy of the first study's ψ3 (waveform kurtosis). Fitted on log scale.
* ``sustained_fraction`` — share of contact time in sustained contacts (the contact
  network fluidises).
* ``phase_concentration_impulse`` — how strongly impact impulse is locked to one
  excitation phase (loss of the collective "bouncing bed").
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gransim.analysis.critical import fit_transition
from gransim.config import SimConfig, deep_merge, frequency_from_gamma

ORDER_PARAMS = ("kurtosis_total_hp", "sustained_fraction", "phase_concentration_impulse")
LOG_PARAMS = ("kurtosis_total_hp", "kurtosis_bottom")
REFERENCE_CSV = Path(__file__).resolve().parents[3] / "assets" / "reference" / "gamma_c_experiment.csv"


def gamma_c_cases(base: SimConfig, Ns, gammas, seeds=(0,)) -> list[dict]:
    A = base.excitation.amplitude
    out = []
    for N in Ns:
        for g in gammas:
            for s in seeds:
                d = deep_merge(base.model_dump(mode="json"),
                               {"seed": s, "particles": {"count": int(N)},
                                "excitation": {"mode": "fixed_amplitude", "amplitude": A,
                                               "frequency": frequency_from_gamma(g, A),
                                               "rpm": None, "gamma": None}})
                out.append(d)
    return out


def extract_gamma_c(results: list[dict], order_params=ORDER_PARAMS) -> dict:
    """{N: {order_param: fit dict}} from sweep results (seed-averaged per Γ)."""
    out: dict = {}
    for N in sorted({r["N"] for r in results}):
        rs = [r for r in results if r["N"] == N]
        gam = np.array(sorted({round(r["gamma"], 6) for r in rs}))
        out[N] = {}
        for op in order_params:
            y = np.array([np.mean([r[op] for r in rs if round(r["gamma"], 6) == g]) for g in gam])
            if op in LOG_PARAMS:
                y = np.log(np.maximum(y, 1e-3))
            out[N][op] = fit_transition(gam, y)
    return out


def load_reference(path: Path = REFERENCE_CSV) -> dict:
    """{N: (Γc, err)} from the experiment CSV (comment lines start with #)."""
    with open(path, encoding="utf-8") as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    out = {}
    for row in csv.DictReader(rows):
        out[int(row["N"])] = (float(row["gamma_c"]), float(row.get("gamma_c_err") or "nan"))
    return out


def comparison_table(gc: dict, reference: dict, order_params=ORDER_PARAMS) -> list[dict]:
    rows = []
    for N in sorted(gc):
        row = {"N": N}
        ref = reference.get(N)
        if ref is None and reference:  # interpolate the experiment curve
            ks = np.array(sorted(reference))
            if ks[0] <= N <= ks[-1]:
                ref = (float(np.interp(N, ks, [reference[k][0] for k in ks])), float("nan"))
        row["gamma_c_experiment"] = ref[0] if ref else float("nan")
        for op in order_params:
            row[f"gamma_c_{op}"] = gc[N][op]["gamma_c"]
            row[f"diff_{op}"] = gc[N][op]["gamma_c"] - row["gamma_c_experiment"]
        rows.append(row)
    return rows


def trend_increasing(gc: dict, op: str) -> bool:
    """Spec pass criterion: Γc increases with N (allowing one non-monotone step of < 0.1)."""
    Ns = sorted(gc)
    v = [gc[N][op]["gamma_c"] for N in Ns]
    return v[-1] > v[0] and all(b - a > -0.1 for a, b in zip(v, v[1:]))
