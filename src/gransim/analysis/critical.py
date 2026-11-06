"""Critical Γ extraction (simple version, spec 9.2 / 10.1).

An order parameter y(Γ) that switches between two plateaus is fitted with a
logistic step

    y(Γ) = lo + (hi − lo) / (1 + exp(−(Γ − Γc) / w))

and Γc is its midpoint, w the transition width. If the fit fails, the first
crossing of the mid level (linear interpolation) is returned instead.

With repeated runs per Γ, :func:`variance_peak` gives the more principled
estimate (Γ where the run-to-run variance of the order parameter peaks).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def _logistic(g, lo, hi, gc, w):
    return lo + (hi - lo) / (1.0 + np.exp(-(g - gc) / w))


def midpoint_crossing(gamma: np.ndarray, y: np.ndarray) -> float:
    """First Γ where y crosses (min+max)/2, linearly interpolated (data sorted by Γ)."""
    o = np.argsort(gamma)
    g, v = np.asarray(gamma, float)[o], np.asarray(y, float)[o]
    mid = 0.5 * (v[0] + v[-1])
    s = np.sign(v - mid)
    for k in range(len(g) - 1):
        if s[k] == 0:
            return float(g[k])
        if s[k] != s[k + 1]:
            return float(g[k] + (mid - v[k]) * (g[k + 1] - g[k]) / (v[k + 1] - v[k]))
    return math.nan


def fit_transition(gamma, y) -> dict:
    """Fit a logistic step to y(Γ); returns gamma_c, width, plateaus and the method used."""
    g = np.asarray(gamma, float)
    v = np.asarray(y, float)
    o = np.argsort(g)
    g, v = g[o], v[o]
    gc0 = midpoint_crossing(g, v)
    if not np.isfinite(gc0):
        gc0 = float(np.median(g))
    p0 = [v[0], v[-1], gc0, max(0.05 * (g[-1] - g[0]), 1e-3)]
    try:
        popt, pcov = curve_fit(_logistic, g, v, p0=p0, maxfev=20000)
        lo, hi, gc, w = popt
        if not (g[0] <= gc <= g[-1]) or not np.all(np.isfinite(popt)):
            raise RuntimeError("fit outside range")
        err = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else math.nan
        return {"gamma_c": float(gc), "gamma_c_err": err, "width": float(abs(w)),
                "lo": float(lo), "hi": float(hi), "method": "logistic"}
    except (RuntimeError, ValueError):
        return {"gamma_c": gc0, "gamma_c_err": math.nan, "width": math.nan,
                "lo": float(v[0]), "hi": float(v[-1]), "method": "midpoint"}


def variance_peak(gamma, y_runs) -> float:
    """Γ at which the variance over repeated runs of the order parameter is largest.

    ``y_runs`` has shape (len(gamma), n_repeats).
    """
    var = np.var(np.asarray(y_runs, float), axis=1)
    return float(np.asarray(gamma, float)[int(np.argmax(var))])
