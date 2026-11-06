"""Turn raw contact records into the event table: chatter merging + classification.

The Numba kernel emits one *raw* event per uninterrupted contact (see
``dynamics.engine``). Here we

1. merge chatter: a re-contact of the same pair (same particles, or same
   particle and wall) that starts within ``merge_gap`` of the previous
   contact's end, with approach speed below ``v_impact``, continues the
   previous event (``n_merged`` counts the segments);
2. classify: ``impact`` if the approach speed at first touch is
   >= ``v_impact`` and the (merged) duration is <= ``duration_max``,
   otherwise ``sustained``.

Sustained contacts are never rendered as pulses by the acoustics; their
force fluctuations are carried by the wall-force time series instead.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from gransim.config import SimConfig
from gransim.dynamics.contact import MODEL_LINEAR, hertz_contact_time
from gransim.dynamics.engine import (E_FLAG, E_FPEAK, E_I, E_J, E_JN, E_JT, E_KIND, E_PX, E_PY,
                                     E_PZ, E_TEND, E_TPEAK, E_TSTART, E_VN0, E_WALL, KIND_PP)

IMPACT = 0
SUSTAINED = 1

EVENT_DTYPE = np.dtype([
    ("t_start", "f8"),      # contact start, shaking time axis [s]
    ("t_peak", "f8"),       # time of peak normal force [s]
    ("t_end", "f8"),        # first time the contact is gone [s]
    ("kind", "i1"),         # 0 = particle-particle (pp), 1 = particle-wall (pw)
    ("contact_type", "i1"),  # 0 = impact, 1 = sustained
    ("n_merged", "i4"),     # number of raw contact segments merged (1 = not merged)
    ("wall_id", "i1"),      # 0 bottom, 1 top, 2 x-, 3 x+, 4 y-, 5 y+; -1 for pp
    ("i", "i4"),            # particle index
    ("j", "i4"),            # partner particle index, -1 for wall
    ("v_rel_n", "f8"),      # normal approach speed at first touch [m/s]
    ("J_n", "f8"),          # normal impulse ∫F_n dt [N s]
    ("J_t", "f8"),          # tangential impulse ∫|F_t| dt [N s]
    ("F_peak", "f8"),       # peak normal force [N]
    ("duration", "f8"),     # t_end - t_start [s]
    ("x", "f8"), ("y", "f8"), ("z", "f8"),  # contact point at peak force, container frame [m]
    ("m_eff", "f8"),        # effective mass [kg]
    ("R_eff", "f8"),        # effective radius [m]
    ("phase", "f8"),        # excitation phase at t_start, in [0, 1)
    ("flags", "i1"),        # bit 1: started before recording, bit 2: still open at end
])


def resolve_event_params(cfg: SimConfig, cp: dict) -> dict:
    """Fill in automatic defaults for v_impact, duration_max, merge_gap (SI units)."""
    ec = cfg.events
    d = cfg.particles.diameter
    v_imp = ec.v_impact if ec.v_impact is not None else 0.1 * math.sqrt(2 * cfg.gravity * d)
    if cp["model"] == MODEL_LINEAR:
        tc = max(cp["contact_time_pp"], cp["contact_time_pw"])
    else:
        m, R = cfg.particles.mass, d / 2
        tc = max(hertz_contact_time(m / 2, R / 2, cp["E_pp"], v_imp),
                 hertz_contact_time(m, R, cp["E_pw"], v_imp))
    return {
        "v_impact": v_imp,
        "duration_max": ec.duration_max if ec.duration_max is not None else 3.0 * tc,
        "merge_gap": ec.merge_gap if ec.merge_gap is not None else tc,
        "binary_contact_time": tc,
        "wall_force_rate": ec.wall_force_rate,
    }


@njit(cache=True)
def _group_ids(order, key, t_start, t_end, vn0, merge_gap, v_impact):
    """Assign merge-group ids along ``order`` (sorted by key, then t_start)."""
    gid = np.empty(order.size, np.int64)
    g = -1
    last_key = -1
    last_end = -1e300
    for n in range(order.size):
        k = order[n]
        chatter = (g >= 0 and key[k] == last_key
                   and t_start[k] - last_end <= merge_gap and vn0[k] < v_impact)
        if chatter:
            last_end = max(last_end, t_end[k])
        else:
            g += 1
            last_key = key[k]
            last_end = t_end[k]
        gid[k] = g
    return gid, g + 1


def merge_and_classify(raw: np.ndarray, merge_gap: float, v_impact: float,
                       duration_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (merged rows in raw layout, n_merged, contact_type)."""
    n = raw.shape[0]
    if n == 0:
        return raw.copy(), np.zeros(0, np.int64), np.zeros(0, np.int64)
    kind = raw[:, E_KIND].astype(np.int64)
    i = raw[:, E_I].astype(np.int64)
    j = raw[:, E_J].astype(np.int64)
    wall = raw[:, E_WALL].astype(np.int64)
    # unique pair key: pp -> (i, j); pw -> (i, wall)
    partner = np.where(kind == KIND_PP, j, wall + 1_000_000_000)
    key = i * 2_000_000_000 + partner
    order = np.lexsort((raw[:, E_TSTART], key))
    gid, ng = _group_ids(order, key, raw[:, E_TSTART], raw[:, E_TEND], raw[:, E_VN0],
                         merge_gap, v_impact)
    out = np.zeros((ng, raw.shape[1]))
    first = np.full(ng, -1, np.int64)
    # first segment (by start time) of each group
    for k in order:
        if first[gid[k]] < 0:
            first[gid[k]] = k
    out[:] = raw[first]
    out[:, E_TEND] = np.maximum.reduceat(raw[order, E_TEND], _starts(gid[order]))
    out[:, E_JN] = np.add.reduceat(raw[order, E_JN], _starts(gid[order]))
    out[:, E_JT] = np.add.reduceat(raw[order, E_JT], _starts(gid[order]))
    flags = np.zeros(ng, np.int64)
    np.bitwise_or.at(flags, gid, raw[:, E_FLAG].astype(np.int64))
    out[:, E_FLAG] = flags
    # peak force (and its time/position) over all segments
    peak_idx = first.copy()
    best = raw[first, E_FPEAK].copy()
    for k in range(n):
        g = gid[k]
        if raw[k, E_FPEAK] > best[g]:
            best[g] = raw[k, E_FPEAK]
            peak_idx[g] = k
    for c in (E_FPEAK, E_TPEAK, E_PX, E_PY, E_PZ):
        out[:, c] = raw[peak_idx, c]
    n_merged = np.bincount(gid, minlength=ng)
    duration = out[:, E_TEND] - out[:, E_TSTART]
    ctype = np.where((out[:, E_VN0] >= v_impact) & (duration <= duration_max),
                     IMPACT, SUSTAINED)
    # sort by start time
    o = np.argsort(out[:, E_TSTART], kind="stable")
    return out[o], n_merged[o], ctype[o]


def _starts(sorted_gid: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, sorted_gid[1:] != sorted_gid[:-1]])


def process_events(raw: np.ndarray, radius: np.ndarray, mass: np.ndarray, frequency: float,
                   params: dict) -> np.ndarray:
    """Raw kernel rows → structured event table (:data:`EVENT_DTYPE`)."""
    rows, n_merged, ctype = merge_and_classify(raw, params["merge_gap"], params["v_impact"],
                                               params["duration_max"])
    ev = np.zeros(rows.shape[0], EVENT_DTYPE)
    if rows.shape[0] == 0:
        return ev
    i = rows[:, E_I].astype(np.int64)
    j = rows[:, E_J].astype(np.int64)
    pp = rows[:, E_KIND] == KIND_PP
    jj = np.where(pp, j, 0)
    ev["t_start"] = rows[:, E_TSTART]
    ev["t_peak"] = rows[:, E_TPEAK]
    ev["t_end"] = rows[:, E_TEND]
    ev["kind"] = rows[:, E_KIND]
    ev["contact_type"] = ctype
    ev["n_merged"] = n_merged
    ev["wall_id"] = rows[:, E_WALL]
    ev["i"] = i
    ev["j"] = j
    ev["v_rel_n"] = rows[:, E_VN0]
    ev["J_n"] = rows[:, E_JN]
    ev["J_t"] = rows[:, E_JT]
    ev["F_peak"] = rows[:, E_FPEAK]
    ev["duration"] = rows[:, E_TEND] - rows[:, E_TSTART]
    ev["x"], ev["y"], ev["z"] = rows[:, E_PX], rows[:, E_PY], rows[:, E_PZ]
    ev["m_eff"] = np.where(pp, mass[i] * mass[jj] / (mass[i] + mass[jj]), mass[i])
    ev["R_eff"] = np.where(pp, radius[i] * radius[jj] / (radius[i] + radius[jj]), radius[i])
    ev["phase"] = np.mod(frequency * rows[:, E_TSTART], 1.0)
    ev["flags"] = rows[:, E_FLAG]
    return ev


def classification_sensitivity(raw: np.ndarray, params: dict,
                               factors=(0.25, 0.5, 1.0, 2.0, 4.0)) -> dict:
    """How the impact/sustained split responds to v_impact and duration_max (spec 5.1).

    Scales each threshold by ``factors`` around its configured value and returns
    the impact count and the fraction of normal impulse carried by impacts.
    """
    f = np.asarray(factors, dtype=float)
    n_imp = np.zeros((f.size, f.size), np.int64)
    j_frac = np.zeros((f.size, f.size))
    for a, fv in enumerate(f):
        for b, fd in enumerate(f):
            rows, _, ct = merge_and_classify(raw, params["merge_gap"], params["v_impact"] * fv,
                                             params["duration_max"] * fd)
            imp = ct == IMPACT
            n_imp[a, b] = imp.sum()
            tot = rows[:, E_JN].sum()
            j_frac[a, b] = rows[imp, E_JN].sum() / tot if tot > 0 else 0.0
    return {"v_impact_factors": f, "duration_max_factors": f,
            "n_impacts": n_imp, "impact_impulse_fraction": j_frac}


def collision_interval_check(events: np.ndarray, contact_time: float,
                             min_ratio: float = 10.0) -> dict:
    """Spec 4.1 condition (b): contact time ≪ typical interval between a particle's impacts.

    Returns the median per-particle interval between successive impact starts and
    its ratio to the binary contact time; ``ok`` is False when the ratio < min_ratio.
    """
    imp = events[events["contact_type"] == IMPACT]
    if imp.size < 2:
        return {"median_interval": math.inf, "ratio": math.inf, "ok": True}
    pid = np.concatenate([imp["i"], imp["j"][imp["j"] >= 0]])
    ts = np.concatenate([imp["t_start"], imp["t_start"][imp["j"] >= 0]])
    o = np.lexsort((ts, pid))
    pid, ts = pid[o], ts[o]
    same = pid[1:] == pid[:-1]
    gaps = np.diff(ts)[same]
    if gaps.size == 0:
        return {"median_interval": math.inf, "ratio": math.inf, "ok": True}
    med = float(np.median(gaps))
    ratio = med / contact_time
    return {"median_interval": med, "ratio": ratio, "ok": bool(ratio >= min_ratio)}
