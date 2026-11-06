"""Contact force laws, restitution ↔ damping lookup tables, stiffness derivation.

Normal force (force magnitude along the contact normal, never tensile)::

    linear : F_n = k_n δ + c_n δ̇,              c_n = 2 ζ sqrt(m_eff k_n)
    hertz  : F_n = k_h δ^1.5 + c_h δ^0.25 δ̇,     c_h = α sqrt(m_eff k_h),
             k_h = (4/3) E* sqrt(R_eff)

``δ`` is the overlap [m] and ``δ̇`` the approach rate [m/s]. Because the force
is clipped at zero, the textbook formula relating ζ to the restitution
coefficient is wrong (most visibly at low e). We therefore obtain ζ (linear)
or α (Hertz/Tsuji) from a lookup table built by integrating a single
collision with the *same* integrator the engine uses (spec 7.1 option (b)).
Both tables are dimensionless, so one table serves every mass/stiffness.

Tangential force: Coulomb-capped spring-dashpot (Cundall–Strack) with the
spring elongation ξ kept per contact and rotated onto the current tangent
plane every step.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from gransim.config import SimConfig

MODEL_LINEAR = 0
MODEL_HERTZ = 1


# --------------------------------------------------------------------------
# Effective quantities and analytic helpers
# --------------------------------------------------------------------------

def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Hertz effective modulus E* = 1 / ((1-ν1²)/E1 + (1-ν2²)/E2) [Pa]."""
    return 1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2)


def hertz_contact_time(m_eff: float, R_eff: float, E_eff: float, v: float) -> float:
    """Elastic Hertz contact duration [s].

    tau_c = 2.87 (m_eff² / (R_eff E_eff² v))^(1/5); m_eff [kg], R_eff [m],
    E_eff = E* [Pa], v = normal impact speed [m/s].
    """
    return 2.868 * (m_eff ** 2 / (R_eff * E_eff ** 2 * v)) ** 0.2


def linear_stiffness_for_contact_time(m_eff: float, t_c: float) -> float:
    """k_n [N/m] such that the undamped half period π sqrt(m_eff/k_n) equals t_c [s]."""
    return m_eff * math.pi ** 2 / t_c ** 2


# --------------------------------------------------------------------------
# Single-collision integrator used to build the restitution tables
# --------------------------------------------------------------------------

@njit(cache=True)
def _single_collision_e(model: int, damp: float, dt: float, offset: float) -> float:
    """Restitution of one normal collision, dimensionless units m_eff=k=v0=1.

    Integrates with the engine's scheme (force from current state, then
    v += F dt, then x += v dt). ``offset`` in [0,1) sets the initial overlap
    to offset*v0*dt, mimicking the random sub-step phase at which contacts
    are first detected.
    """
    d = offset * dt
    dd = 1.0  # approach rate
    for _ in range(10_000_000):
        if d <= 0.0:
            return -dd
        if model == MODEL_LINEAR:
            f = d + 2.0 * damp * dd
        else:
            f = d * math.sqrt(d) + damp * math.sqrt(math.sqrt(d)) * dd
        if f < 0.0:
            f = 0.0
        dd -= f * dt
        d += dd * dt
    return np.nan


def _table_grid() -> np.ndarray:
    return np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), np.geomspace(1.0, 50.0, 200)]))


@lru_cache(maxsize=64)
def restitution_table(model: int, dt_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """(damping_grid, e_grid) for a model.

    dt_ratio = dt / t_c with t_c = π sqrt(m_eff/k) (linear) or the Hertz elastic
    contact time (hertz). Values are averaged over sub-step detection offsets.
    """
    grid = _table_grid()
    if model == MODEL_LINEAR:
        t_c = math.pi
    else:
        # k_h = 1 ⇔ (4/3) E* sqrt(R) = 1 ⇔ E* = 3/4 with R = 1;  tau ∝ E*^(-2/5)
        t_c = hertz_contact_time(1.0, 1.0, 0.75, 1.0)
    dt = dt_ratio * t_c
    if model == MODEL_LINEAR:
        # explicit integration of the dashpot is stable only for c·dt/m < 2 (c = 2ζ here)
        grid = grid[2.0 * grid * dt < 1.5]
    offsets = (np.arange(8) + 0.5) / 8.0
    e = np.array([np.mean([_single_collision_e(model, z, dt, o) for o in offsets]) for z in grid])
    # Near the explicit-stability limit e rises again (numerical artefact): keep the monotone part.
    k = int(np.argmin(e)) + 1
    return grid[:k], e[:k]


def damping_for_restitution(e: float, model: int, dt_ratio: float = 1e-3) -> float:
    """Damping parameter (ζ for linear, α for Hertz) that yields restitution e.

    Accounts for the tensile-force clipping and for the integrator at the given
    dt_ratio (see :func:`restitution_table`).
    """
    if not 0.0 < e <= 1.0:
        raise ValueError(f"restitution must be in (0, 1], got {e}")
    grid, eg = restitution_table(model, round(float(dt_ratio), 5))
    if e >= eg[0]:
        return 0.0
    if e < eg[-1]:
        raise ValueError(f"restitution {e} below table range (min {eg[-1]:.3g})")
    if np.any(np.diff(eg) > 1e-9):
        raise RuntimeError("restitution table is not monotonic; refine grid")
    return float(np.interp(e, eg[::-1], grid[::-1]))


def restitution_scatter(e: float, model: int, damp: float, dt_ratio: float) -> float:
    """Max relative deviation of a single collision's restitution from e over detection phases.

    Contacts are detected at a random sub-step phase, which makes each collision's
    restitution scatter around the mean (zero-mean numerical noise). Grows with
    dt/t_c and with damping (low e).
    """
    t_c = math.pi if model == MODEL_LINEAR else hertz_contact_time(1.0, 1.0, 0.75, 1.0)
    es = np.array([_single_collision_e(model, damp, dt_ratio * t_c, o)
                   for o in np.linspace(0.02, 0.98, 25)])
    return float(np.max(np.abs(es - e)) / e)


# --------------------------------------------------------------------------
# Parameter derivation from the config
# --------------------------------------------------------------------------

def _v_ref(cfg: "SimConfig") -> float:
    """Conservative (high) impact speed used to bound the shortest Hertz contact [m/s]."""
    A, f, _ = cfg.excitation.resolved(cfg.gravity)
    v = 2.0 * A * 2.0 * math.pi * f + math.sqrt(2.0 * cfg.gravity * cfg.container.size[2])
    return max(v, 1.0) + 0.1  # floor: never assume impacts slower than ~1 m/s


def derive_contact_params(cfg: "SimConfig") -> dict:
    """Resolve stiffness, damping, contact times and dt from the config.

    Returns a plain dict (SI units). For the linear model, k is constant per
    pair type and derived from ``contact.contact_time`` using the mean
    particle's m_eff. For Hertz, E* per pair type is returned; k_h is formed
    per contact from R_eff.
    """
    c = cfg.contact
    p = cfg.particles
    m, R = p.mass, 0.5 * p.diameter
    m_pp, m_pw = 0.5 * m, m
    R_pp, R_pw = 0.5 * R, R
    Ep, nup = p.material.youngs_modulus, p.material.poisson_ratio
    Ew, nuw = cfg.container.material.youngs_modulus, cfg.container.material.poisson_ratio
    out: dict = {"model": MODEL_LINEAR if c.model == "linear" else MODEL_HERTZ}

    if c.model == "linear":
        tc_pp = tc_pw = c.contact_time
        out["k_pp"] = linear_stiffness_for_contact_time(m_pp, tc_pp)
        out["k_pw"] = linear_stiffness_for_contact_time(m_pw, tc_pw)
        out["E_pp"] = out["E_pw"] = 0.0
    else:
        out["E_pp"] = c.modulus_scale * effective_modulus(Ep, nup, Ep, nup)
        out["E_pw"] = c.modulus_scale * effective_modulus(Ep, nup, Ew, nuw)
        v = _v_ref(cfg)
        tc_pp = hertz_contact_time(m_pp, R_pp, out["E_pp"], v)
        tc_pw = hertz_contact_time(m_pw, R_pw, out["E_pw"], v)
        out["k_pp"] = out["k_pw"] = 0.0

    # dt: contact_time / dt_safety, refined automatically until the per-collision
    # restitution scatter (from the sub-step phase of contact detection) is below
    # time.max_restitution_scatter. Only matters for low e (strong damping).
    tc_min = min(tc_pp, tc_pw)
    safety = cfg.time.dt_safety
    while True:
        dt_rec = tc_min / safety
        dt = cfg.time.dt if cfg.time.dt is not None else dt_rec
        damp, scat = {}, {}
        for key, e, tc in (("pp", c.restitution_pp, tc_pp), ("pw", c.restitution_pw, tc_pw)):
            model = out["model"]
            ratio = dt / tc
            if model == MODEL_LINEAR:
                damp[key] = damping_for_restitution(e, model, ratio)
            else:
                # Hertz restitution is velocity independent in the continuum limit
                damp[key] = damping_for_restitution(e, model, 1e-3)
            scat[key] = restitution_scatter(e, model, damp[key], ratio)
        worst = max(scat.values())
        if cfg.time.dt is not None or worst <= cfg.time.max_restitution_scatter or safety > 640:
            break
        safety *= 2.0
    out.update(contact_time_pp=tc_pp, contact_time_pw=tc_pw, dt=dt, dt_recommended=dt_rec,
               damp_pp=damp["pp"], damp_pw=damp["pw"], restitution_scatter=worst)
    return out


# --------------------------------------------------------------------------
# Force kernel (shared by particle-particle and particle-wall contacts)
# --------------------------------------------------------------------------

@njit(cache=True, inline="always")
def contact_force(nx, ny, nz, delta, vx, vy, vz, m_eff, R_eff,
                  model, k_lin, e_star, damp, mu, kt_ratio, ct_ratio, xi, dt):
    """Force on body "i" at one contact.

    Inputs: unit normal n pointing from the partner (particle j or wall) to i;
    overlap delta > 0 [m]; relative velocity of i's contact point w.r.t. the
    partner's contact point (vx, vy, vz) [m/s]; ``xi`` (4,) history record
    (tangential spring elongation ξ and previous k_t), updated in place.

    Returns (Fn, Ftx, Fty, Ftz, Fs, Fdx, Fdy, Fdz, diss_n, diss_t, E_n, E_t, W_k):
      Fn       normal force magnitude (>= 0) [N], acts along +n on i
      Ft*      tangential force on i [N]
      Fs       elastic part of the normal force [N] (dissipative normal part = Fn - Fs)
      Fd*      dissipative part of the tangential force [N]
               (= Ft + k_t ξ_trial: viscous term when sticking, slip excess when sliding)
      diss_*   energy dissipated this step in normal / tangential [J], evaluated with the
               start-of-step relative velocity (the engine adds a mid-step correction)
      E_n, E_t elastic energy stored in the contact after the step [J]
      W_k      energy added to the tangential spring because k_t changed with the
               overlap (Hertz only; zero for the linear model) [J]
    """
    vn = vx * nx + vy * ny + vz * nz
    ddot = -vn  # approach rate
    vtx = vx - vn * nx
    vty = vy - vn * ny
    vtz = vz - vn * nz

    if model == 0:
        k = k_lin
        fs = k * delta
        c = 2.0 * damp * math.sqrt(m_eff * k)
        fn = fs + c * ddot
        kt = kt_ratio * k
        en = 0.5 * k * delta * delta
    else:
        kh = (4.0 / 3.0) * e_star * math.sqrt(R_eff)
        sd = math.sqrt(delta)
        fs = kh * delta * sd
        c = damp * math.sqrt(m_eff * kh) * math.sqrt(sd)  # includes δ^0.25
        fn = fs + c * ddot
        kt = kt_ratio * 1.5 * kh * sd
        en = 0.4 * kh * delta * delta * sd
    if fn < 0.0:
        fn = 0.0
    diss_n = (fn - fs) * ddot * dt

    # --- tangential: rotate stored elongation onto the current tangent plane
    old2 = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    xn = xi[0] * nx + xi[1] * ny + xi[2] * nz
    xi[0] -= xn * nx
    xi[1] -= xn * ny
    xi[2] -= xn * nz
    new2 = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    if new2 > 0.0:
        s = math.sqrt(old2 / new2)
        xi[0] *= s
        xi[1] *= s
        xi[2] *= s
    et_old = 0.5 * kt * old2
    w_k = 0.5 * (kt - xi[3]) * old2  # stiffness changed at fixed elongation
    xi[3] = kt
    xi[0] += vtx * dt
    xi[1] += vty * dt
    xi[2] += vtz * dt
    ct = ct_ratio * c
    ftx = -kt * xi[0] - ct * vtx
    fty = -kt * xi[1] - ct * vty
    ftz = -kt * xi[2] - ct * vtz
    ftm = math.sqrt(ftx * ftx + fty * fty + ftz * ftz)
    fmax = mu * fn
    if ftm > fmax:
        # sliding: Coulomb-capped force, elongation cut back to match it
        if ftm > 0.0:
            r = fmax / ftm
            ftx *= r
            fty *= r
            ftz *= r
        if kt > 0.0:
            xi[0] = -ftx / kt
            xi[1] = -fty / kt
            xi[2] = -ftz / kt
        else:
            xi[0] = 0.0
            xi[1] = 0.0
            xi[2] = 0.0
        et = 0.5 * kt * (xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2])
        # whole friction force is dissipative; energy = work on the slip minus storage change
        fdx = ftx
        fdy = fty
        fdz = ftz
        diss_t = -(ftx * vtx + fty * vty + ftz * vtz) * dt - (et - et_old)
    else:
        # sticking: spring is elastic (consistent with the symplectic update, like the
        # normal spring); only the viscous part dissipates
        et = 0.5 * kt * (xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2])
        fdx = -ct * vtx
        fdy = -ct * vty
        fdz = -ct * vtz
        diss_t = ct * (vtx * vtx + vty * vty + vtz * vtz) * dt
    return fn, ftx, fty, ftz, fs, fdx, fdy, fdz, diss_n, diss_t, en, et, w_k
