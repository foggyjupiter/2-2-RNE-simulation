"""DEM integration loop (Numba) and the Python-side simulation driver.

Integrator: semi-implicit (symplectic) Euler. Forces are evaluated from
(x_n, v_n, ω_n, wall state at t_n), then v_{n+1} = v_n + a dt,
ω_{n+1} = ω_n + α dt, x_{n+1} = x_n + v_{n+1} dt.

Energy bookkeeping (lab frame, used by the energy-balance test):
    W_in   work done by the moving walls = Σ F_wall→particle · U_wall dt
    D_n    normal dissipation       = Σ (F_n − F_elastic) δ̇ dt
    D_t    tangential dissipation   = Σ (−F_t·v_t) dt − ΔE_t
    E_n, E_t  elastic energy stored in contacts at the current step
so that  Δ(KE_trans + KE_rot + PE + E_n + E_t) = W_in − D_n − D_t
up to O(dt) discretisation error. Crucially the dissipation is computed
from the same relative velocity used in the force law; if the wall velocity
were missing from that relative velocity the balance would not close.

Contact history: each particle i stores its particle-particle contacts with
j > i in up to MAX_CONTACTS slots (partner index + per-contact float record
of length N_HIST). Wall contacts use a fixed slot per wall. Records are
double-buffered and matched by partner index every step.

Contact events (spec 5.1 A): the history record also carries the event
accumulators (start time, approach speed at first touch, normal/tangential
impulse, peak force with its time and position). When a contact present in
the previous step is gone, its record is emitted to the event buffer as one
raw event. Chatter merging and impact/sustained classification happen
afterwards in :mod:`gransim.events.tracker`.

Wall force (spec 5.1 B): the normal force each wall receives is accumulated
as impulse into bins of width 1/fs (default 22050 Hz), so the stored
bin-average force conserves impulse exactly.

Only steps at t >= t_rec0 (the recording window) emit events and fill wall
force bins.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from numba import njit

from gransim.config import SimConfig, check_config, derived_quantities
from gransim.dynamics.contact import contact_force, derive_contact_params
from gransim.dynamics.init import lattice_positions, particle_properties, random_positions
from gransim.dynamics.neighbors import build_cells, cell_coord, make_grid
from gransim.dynamics.walls import N_WALLS, container_motion, wall_plane

MAX_CONTACTS = 16     # pp contacts stored per particle (partners j > i)

# per-contact history record layout
H_XI = 0              # [0:3] tangential spring elongation ξ [m]
H_KT = 3              # k_t at the previous step [N/m]
H_TSTART = 4          # contact start time [s]
H_VN0 = 5             # normal approach speed at first touch [m/s]
H_JN = 6              # ∫ F_n dt [N s]
H_JT = 7              # ∫ |F_t| dt [N s]
H_FPEAK = 8           # max F_n [N]
H_TPEAK = 9           # time of max F_n [s]
H_PX, H_PY, H_PZ = 10, 11, 12  # contact point at peak force, container frame [m]
H_FLAG = 13           # bit 1: started before the recording window
N_HIST = 14

# raw event row layout
(E_TSTART, E_TEND, E_TPEAK, E_KIND, E_WALL, E_I, E_J, E_VN0, E_JN, E_JT, E_FPEAK,
 E_PX, E_PY, E_PZ, E_FLAG) = range(15)
N_EV = 15
KIND_PP = 0
KIND_PW = 1
FLAG_START_CLIPPED = 1
FLAG_END_CLIPPED = 2

# accumulator indices
A_WIN, A_DN, A_DT, A_EN, A_ET, A_MAXOV, A_NPP, A_NPW, A_DCORR, A_WK = range(10)
N_ACC = 10

ERR_OK = 0
ERR_CONTACT_OVERFLOW = 1
ERR_EVENT_BUFFER_FULL = 2


@njit(cache=True, inline="always")
def _emit(ev, ev_n, rec, kind, wall, i, j, t_end, extra_flag):
    k = ev_n[0]
    ev[k, E_TSTART] = rec[H_TSTART]
    ev[k, E_TEND] = t_end
    ev[k, E_TPEAK] = rec[H_TPEAK]
    ev[k, E_KIND] = kind
    ev[k, E_WALL] = wall
    ev[k, E_I] = i
    ev[k, E_J] = j
    ev[k, E_VN0] = rec[H_VN0]
    ev[k, E_JN] = rec[H_JN]
    ev[k, E_JT] = rec[H_JT]
    ev[k, E_FPEAK] = rec[H_FPEAK]
    ev[k, E_PX] = rec[H_PX]
    ev[k, E_PY] = rec[H_PY]
    ev[k, E_PZ] = rec[H_PZ]
    ev[k, E_FLAG] = float(int(rec[H_FLAG]) | extra_flag)
    ev_n[0] = k + 1


@njit(cache=True, inline="always")
def _accumulate(rec, is_new, t, dt, ddot, fn, ftx, fty, ftz, px, py, pz):
    """Update the event accumulators of one contact record for this step."""
    if is_new:
        rec[H_TSTART] = t
        rec[H_VN0] = ddot
    rec[H_JN] += fn * dt
    rec[H_JT] += math.sqrt(ftx * ftx + fty * fty + ftz * ftz) * dt
    if fn > rec[H_FPEAK] or is_new:
        rec[H_FPEAK] = fn
        rec[H_TPEAK] = t
        rec[H_PX] = px
        rec[H_PY] = py
        rec[H_PZ] = pz


class StepParams(NamedTuple):
    """Scalar parameters passed to the Numba kernel (SI units)."""
    dt: float
    g: float
    Lx: float
    Ly: float
    Lz: float
    amp: float
    omega: float
    ramp: float
    model: int
    k_pp: float
    k_pw: float
    E_pp: float
    E_pw: float
    damp_pp: float
    damp_pw: float
    mu_pp: float
    mu_pw: float
    kt_ratio: float
    ct_ratio: float
    rotation: int
    global_damping: float
    cell: float


@njit(cache=True)
def advance(x, v, w, rad, mass, inv_I,
            hj, hc, hf, nj, nc, nf, hfound,
            wf, won, nwf, nwon,
            head, nxt, F, T, Fd, Td, dims, P, t0, nsteps, acc,
            ev, ev_n, wforce, t_rec0, fs_force):
    """Advance the system by up to ``nsteps`` steps starting at shaking time t0 [s].

    History buffers are swapped internally every step. Returns
    (steps_done, number_of_swaps_mod_2, error_code); the caller must swap its
    own buffer references when the parity is 1. Stops early with
    ERR_EVENT_BUFFER_FULL when the event buffer may overflow in the next step.
    """
    N = x.shape[0]
    dt = P.dt
    parity = 0
    err = ERR_OK
    nyc = dims[1]
    nzc = dims[2]
    nbins = wforce.shape[1]
    reserve = N * (MAX_CONTACTS + N_WALLS)
    for s in range(nsteps):
        if ev_n[0] + reserve > ev.shape[0]:
            return s, parity, ERR_EVENT_BUFFER_FULL
        t = t0 + s * dt
        recording = t >= t_rec0
        zc, uc = container_motion(t, P.amp, P.omega, P.ramp)
        F[:, :] = 0.0
        T[:, :] = 0.0
        Fd[:, :] = 0.0
        Td[:, :] = 0.0
        acc[A_EN] = 0.0
        acc[A_ET] = 0.0
        acc[A_NPP] = 0.0
        acc[A_NPW] = 0.0
        build_cells(x, zc, P.cell, dims, head, nxt)
        for i in range(N):
            for k in range(hc[i]):
                hfound[i, k] = 0

        # ---------------- particle-particle ----------------
        for i in range(N):
            nc[i] = 0
            ci = cell_coord(x[i, 0], P.cell, dims[0])
            cj = cell_coord(x[i, 1], P.cell, nyc)
            ck = cell_coord(x[i, 2] - zc, P.cell, nzc)
            for a in range(max(ci - 1, 0), min(ci + 2, dims[0])):
                for b in range(max(cj - 1, 0), min(cj + 2, nyc)):
                    for c in range(max(ck - 1, 0), min(ck + 2, nzc)):
                        j = head[(a * nyc + b) * nzc + c]
                        while j >= 0:
                            if j > i:
                                dx = x[i, 0] - x[j, 0]
                                dy = x[i, 1] - x[j, 1]
                                dz = x[i, 2] - x[j, 2]
                                rs = rad[i] + rad[j]
                                r2 = dx * dx + dy * dy + dz * dz
                                if r2 < rs * rs and r2 > 0.0:
                                    q = nc[i]
                                    if q >= MAX_CONTACTS:
                                        err = ERR_CONTACT_OVERFLOW
                                    else:
                                        dist = math.sqrt(r2)
                                        n0 = dx / dist
                                        n1 = dy / dist
                                        n2 = dz / dist
                                        delta = rs - dist
                                        # carry history
                                        slot = -1
                                        for k in range(hc[i]):
                                            if hj[i, k] == j:
                                                slot = k
                                                hfound[i, k] = 1
                                                break
                                        nj[i, q] = j
                                        if slot >= 0:
                                            for m in range(hf.shape[2]):
                                                nf[i, q, m] = hf[i, slot, m]
                                        else:
                                            for m in range(hf.shape[2]):
                                                nf[i, q, m] = 0.0
                                        nc[i] = q + 1
                                        # lever arms to the contact point (sum = centre
                                        # distance, so angular momentum is conserved exactly)
                                        ri = rad[i] - 0.5 * delta
                                        rj = rad[j] - 0.5 * delta
                                        # w × n
                                        win0 = w[i, 1] * n2 - w[i, 2] * n1
                                        win1 = w[i, 2] * n0 - w[i, 0] * n2
                                        win2 = w[i, 0] * n1 - w[i, 1] * n0
                                        wjn0 = w[j, 1] * n2 - w[j, 2] * n1
                                        wjn1 = w[j, 2] * n0 - w[j, 0] * n2
                                        wjn2 = w[j, 0] * n1 - w[j, 1] * n0
                                        vx = v[i, 0] - ri * win0 - v[j, 0] - rj * wjn0
                                        vy = v[i, 1] - ri * win1 - v[j, 1] - rj * wjn1
                                        vz = v[i, 2] - ri * win2 - v[j, 2] - rj * wjn2
                                        meff = mass[i] * mass[j] / (mass[i] + mass[j])
                                        reff = rad[i] * rad[j] / rs
                                        fn, ftx, fty, ftz, fs, fdx, fdy, fdz, dn, dtt, en, et, wk = contact_force(
                                            n0, n1, n2, delta, vx, vy, vz, meff, reff,
                                            P.model, P.k_pp, P.E_pp, P.damp_pp, P.mu_pp,
                                            P.kt_ratio, P.ct_ratio, nf[i, q], dt)
                                        _accumulate(nf[i, q], slot < 0, t, dt,
                                                    -(vx * n0 + vy * n1 + vz * n2),
                                                    fn, ftx, fty, ftz,
                                                    x[i, 0] - ri * n0, x[i, 1] - ri * n1,
                                                    x[i, 2] - ri * n2 - zc)
                                        fx = fn * n0 + ftx
                                        fy = fn * n1 + fty
                                        fz = fn * n2 + ftz
                                        F[i, 0] += fx
                                        F[i, 1] += fy
                                        F[i, 2] += fz
                                        F[j, 0] -= fx
                                        F[j, 1] -= fy
                                        F[j, 2] -= fz
                                        # torque: r × F_t with r_i = -l_i n, r_j = +l_j n (F_t on j is -F_t)
                                        nt0 = n1 * ftz - n2 * fty
                                        nt1 = n2 * ftx - n0 * ftz
                                        nt2 = n0 * fty - n1 * ftx
                                        T[i, 0] -= ri * nt0
                                        T[i, 1] -= ri * nt1
                                        T[i, 2] -= ri * nt2
                                        T[j, 0] -= rj * nt0
                                        T[j, 1] -= rj * nt1
                                        T[j, 2] -= rj * nt2
                                        # dissipative parts (for the mid-step energy correction)
                                        fdn = fn - fs
                                        gx = fdn * n0 + fdx
                                        gy = fdn * n1 + fdy
                                        gz = fdn * n2 + fdz
                                        Fd[i, 0] += gx
                                        Fd[i, 1] += gy
                                        Fd[i, 2] += gz
                                        Fd[j, 0] -= gx
                                        Fd[j, 1] -= gy
                                        Fd[j, 2] -= gz
                                        nd0 = n1 * fdz - n2 * fdy
                                        nd1 = n2 * fdx - n0 * fdz
                                        nd2 = n0 * fdy - n1 * fdx
                                        Td[i, 0] -= ri * nd0
                                        Td[i, 1] -= ri * nd1
                                        Td[i, 2] -= ri * nd2
                                        Td[j, 0] -= rj * nd0
                                        Td[j, 1] -= rj * nd1
                                        Td[j, 2] -= rj * nd2
                                        acc[A_DN] += dn
                                        acc[A_DT] += dtt
                                        acc[A_WK] += wk
                                        acc[A_EN] += en
                                        acc[A_ET] += et
                                        acc[A_NPP] += 1.0
                                        ov = delta / rs
                                        if ov > acc[A_MAXOV]:
                                            acc[A_MAXOV] = ov
                            j = nxt[j]

        # contacts that existed in the previous step but are gone now -> events
        if recording:
            for i in range(N):
                for k in range(hc[i]):
                    if hfound[i, k] == 0:
                        _emit(ev, ev_n, hf[i, k], KIND_PP, -1, i, hj[i, k], t, 0)

        # ---------------- particle-wall ----------------
        for i in range(N):
            for wi in range(N_WALLS):
                pos, ax, sgn = wall_plane(wi, P.Lx, P.Ly, P.Lz, zc)
                sd = (x[i, ax] - pos) * sgn
                delta = rad[i] - sd
                if delta > 0.0:
                    ri = sd  # lever arm: centre to contact point on the wall plane
                    n0 = 0.0
                    n1 = 0.0
                    n2 = 0.0
                    if ax == 0:
                        n0 = sgn
                    elif ax == 1:
                        n1 = sgn
                    else:
                        n2 = sgn
                    was_on = won[i, wi] == 1
                    if was_on:
                        for m in range(wf.shape[2]):
                            nwf[i, wi, m] = wf[i, wi, m]
                    else:
                        for m in range(wf.shape[2]):
                            nwf[i, wi, m] = 0.0
                    nwon[i, wi] = 1
                    win0 = w[i, 1] * n2 - w[i, 2] * n1
                    win1 = w[i, 2] * n0 - w[i, 0] * n2
                    win2 = w[i, 0] * n1 - w[i, 1] * n0
                    vx = v[i, 0] - ri * win0
                    vy = v[i, 1] - ri * win1
                    vz = v[i, 2] - ri * win2 - uc  # wall velocity (0, 0, uc)
                    fn, ftx, fty, ftz, fs, fdx, fdy, fdz, dn, dtt, en, et, wk = contact_force(
                        n0, n1, n2, delta, vx, vy, vz, mass[i], rad[i],
                        P.model, P.k_pw, P.E_pw, P.damp_pw, P.mu_pw,
                        P.kt_ratio, P.ct_ratio, nwf[i, wi], dt)
                    _accumulate(nwf[i, wi], not was_on, t, dt,
                                -(vx * n0 + vy * n1 + vz * n2), fn, ftx, fty, ftz,
                                x[i, 0] - ri * n0, x[i, 1] - ri * n1, x[i, 2] - ri * n2 - zc)
                    if recording:
                        b = int((t - t_rec0) * fs_force)
                        if b < nbins:
                            wforce[wi, b] += fn * dt
                    fx = fn * n0 + ftx
                    fy = fn * n1 + fty
                    fz = fn * n2 + ftz
                    F[i, 0] += fx
                    F[i, 1] += fy
                    F[i, 2] += fz
                    nt0 = n1 * ftz - n2 * fty
                    nt1 = n2 * ftx - n0 * ftz
                    nt2 = n0 * fty - n1 * ftx
                    T[i, 0] -= ri * nt0
                    T[i, 1] -= ri * nt1
                    T[i, 2] -= ri * nt2
                    fdn = fn - fs
                    Fd[i, 0] += fdn * n0 + fdx
                    Fd[i, 1] += fdn * n1 + fdy
                    Fd[i, 2] += fdn * n2 + fdz
                    Td[i, 0] -= ri * (n1 * fdz - n2 * fdy)
                    Td[i, 1] -= ri * (n2 * fdx - n0 * fdz)
                    Td[i, 2] -= ri * (n0 * fdy - n1 * fdx)
                    acc[A_WIN] += fz * uc * dt
                    acc[A_DN] += dn
                    acc[A_DT] += dtt
                    acc[A_WK] += wk
                    acc[A_EN] += en
                    acc[A_ET] += et
                    acc[A_NPW] += 1.0
                    ov = delta / (2.0 * rad[i])
                    if ov > acc[A_MAXOV]:
                        acc[A_MAXOV] = ov
                else:
                    nwon[i, wi] = 0
                    if recording and won[i, wi] == 1:
                        _emit(ev, ev_n, wf[i, wi], KIND_PW, wi, i, -1, t, 0)

        # ---------------- integrate ----------------
        # Dissipation above is evaluated with start-of-step velocities, while the
        # kinetic-energy change corresponds to the mid-step velocity. A_DCORR adds
        # the difference for the dissipative forces only, -Σ_i (Fd_i·Δv_i + Td_i·Δω_i)·dt/2.
        # (Elastic parts need no correction: they are consistent with the positions
        # under the symplectic scheme.)
        gd = P.global_damping
        corr = 0.0
        for i in range(N):
            im = 1.0 / mass[i]
            for k in range(3):
                fk = F[i, k] - gd * mass[i] * v[i, k]
                if k == 2:
                    fk -= mass[i] * P.g
                dv = fk * im * dt
                corr += Fd[i, k] * dv
                v[i, k] += dv
            if P.rotation == 1:
                for k in range(3):
                    dw = (T[i, k] * inv_I[i] - gd * w[i, k]) * dt
                    corr += Td[i, k] * dw
                    w[i, k] += dw
            for k in range(3):
                x[i, k] += v[i, k] * dt
        acc[A_DCORR] -= 0.5 * corr * dt

        # swap history buffers
        hj, nj = nj, hj
        hc, nc = nc, hc
        hf, nf = nf, hf
        wf, nwf = nwf, wf
        won, nwon = nwon, won
        parity ^= 1
        if err != ERR_OK:
            return s + 1, parity, err
    return nsteps, parity, err


# --------------------------------------------------------------------------
# Python driver
# --------------------------------------------------------------------------

@dataclass
class Diagnostics:
    """Time series sampled every ``output.diagnostics_interval`` (shaking time axis)."""
    t: list = field(default_factory=list)            # [s] since shaking start
    zc: list = field(default_factory=list)           # container displacement [m]
    ke_trans: list = field(default_factory=list)     # [J]
    ke_rot: list = field(default_factory=list)       # [J]
    pe: list = field(default_factory=list)           # gravitational, lab frame [J]
    e_n: list = field(default_factory=list)          # stored normal elastic [J]
    e_t: list = field(default_factory=list)          # stored tangential elastic [J]
    w_in: list = field(default_factory=list)         # cumulative wall work [J]
    d_n: list = field(default_factory=list)          # cumulative normal dissipation [J]
    d_t: list = field(default_factory=list)          # cumulative tangential dissipation [J]
    d_corr: list = field(default_factory=list)       # cumulative mid-step work correction [J]
    w_k: list = field(default_factory=list)          # cumulative k_t-variation work (Hertz) [J]
    mean_height: list = field(default_factory=list)  # mean particle z in container frame [m]
    n_pp: list = field(default_factory=list)         # current pp contacts
    n_pw: list = field(default_factory=list)         # current wall contacts

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.__dict__.items()}


@dataclass
class WallForce:
    """Normal force received by each wall, bin-averaged (impulse / bin width).

    force[w, b] is the mean force [N] on wall w over [t0 + b/fs, t0 + (b+1)/fs),
    t0 being the start of the recording window on the shaking time axis.
    Wall order: bottom, top, x-, x+, y-, y+.
    """
    force: np.ndarray
    fs: float
    t0: float

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(self.force.shape[1]) / self.fs


class Simulation:
    """Holds state and buffers; thin wrapper over :func:`advance`.

    Typical use::

        sim = Simulation(cfg)
        sim.settle()
        sim.run(cfg.time.discard_time)
        sim.begin_recording(cfg.time.record_time)
        sim.run(cfg.time.record_time)
        raw_events, wall_force = sim.finish_recording()

    Tests may overwrite ``sim.x``, ``sim.v``, ``sim.w`` before stepping.
    """

    def __init__(self, cfg: SimConfig, positions: Optional[np.ndarray] = None,
                 warn: bool = True, event_capacity: int = 100_000):
        self.cfg = cfg
        if warn:
            check_config(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.radius, self.mass = particle_properties(cfg, self.rng)
        n = self.radius.size
        self.n = n
        self.inv_I = 1.0 / (0.4 * self.mass * self.radius ** 2) if n else np.zeros(0)
        if positions is not None:
            self.x = positions.copy()
        elif cfg.particles.init == "random":
            self.x = random_positions(cfg, self.radius, self.rng)
        else:
            self.x = lattice_positions(cfg, self.radius, self.rng)
        self.v = np.zeros((n, 3))
        self.w = np.zeros((n, 3))
        self.cp = derive_contact_params(cfg)
        self.A, self.f, self.gamma = cfg.excitation.resolved(cfg.gravity)
        dmax = 2.0 * (self.radius.max() if n else cfg.particles.diameter / 2)
        self.dims, self.cell = make_grid(cfg.container.size, dmax)
        self._alloc(event_capacity)
        self.t = 0.0  # shaking time [s]
        self.acc = np.zeros(N_ACC)
        self.diag = Diagnostics()
        self.wall_time = 0.0

    def _alloc(self, event_capacity: int) -> None:
        n = self.n
        self.hj = np.full((n, MAX_CONTACTS), -1, np.int32)
        self.nj = self.hj.copy()
        self.hc = np.zeros(n, np.int32)
        self.nc = self.hc.copy()
        self.hf = np.zeros((n, MAX_CONTACTS, N_HIST))
        self.nf = self.hf.copy()
        self.hfound = np.zeros((n, MAX_CONTACTS), np.uint8)
        self.wf = np.zeros((n, N_WALLS, N_HIST))
        self.nwf = self.wf.copy()
        self.won = np.zeros((n, N_WALLS), np.uint8)
        self.nwon = self.won.copy()
        self.head = np.full(int(np.prod(self.dims)), -1, np.int32)
        self.nxt = np.full(max(n, 1), -1, np.int32)
        self.F = np.zeros((n, 3))
        self.T = np.zeros((n, 3))
        self.Fd = np.zeros((n, 3))
        self.Td = np.zeros((n, 3))
        cap = max(event_capacity, 2 * n * (MAX_CONTACTS + N_WALLS) + 16)
        self.ev = np.zeros((cap, N_EV))
        self.ev_n = np.zeros(1, np.int64)
        self.t_rec0 = math.inf
        self.fs_force = float(self.cfg.events.wall_force_rate)
        self.wforce = np.zeros((N_WALLS, 0))

    def params(self, shaking: bool = True, global_damping: float = 0.0) -> StepParams:
        c = self.cfg.contact
        Lx, Ly, Lz = self.cfg.container.size
        cp = self.cp
        return StepParams(
            dt=cp["dt"], g=self.cfg.gravity, Lx=Lx, Ly=Ly, Lz=Lz,
            amp=self.A if shaking else 0.0, omega=2.0 * math.pi * self.f,
            ramp=self.cfg.excitation.ramp_time, model=cp["model"],
            k_pp=cp["k_pp"], k_pw=cp["k_pw"], E_pp=cp["E_pp"], E_pw=cp["E_pw"],
            damp_pp=cp["damp_pp"], damp_pw=cp["damp_pw"],
            mu_pp=c.friction_pp, mu_pw=c.friction_pw,
            kt_ratio=c.tangential_stiffness_ratio, ct_ratio=c.tangential_damping_ratio,
            rotation=1 if c.rotation else 0, global_damping=global_damping, cell=self.cell)

    @property
    def dt(self) -> float:
        return self.cp["dt"]

    def _swap(self) -> None:
        self.hj, self.nj = self.nj, self.hj
        self.hc, self.nc = self.nc, self.hc
        self.hf, self.nf = self.nf, self.hf
        self.wf, self.nwf = self.nwf, self.wf
        self.won, self.nwon = self.nwon, self.won

    def _kernel(self, nsteps: int, P: StepParams, t0: float, t_rec0: float) -> None:
        done = 0
        while done < nsteps:
            tic = time.perf_counter()
            k, parity, err = advance(
                self.x, self.v, self.w, self.radius, self.mass, self.inv_I,
                self.hj, self.hc, self.hf, self.nj, self.nc, self.nf, self.hfound,
                self.wf, self.won, self.nwf, self.nwon,
                self.head, self.nxt, self.F, self.T, self.Fd, self.Td,
                self.dims, P, t0 + done * P.dt, nsteps - done, self.acc,
                self.ev, self.ev_n, self.wforce, t_rec0, self.fs_force)
            self.wall_time += time.perf_counter() - tic
            done += k
            if parity:
                self._swap()
            if err == ERR_CONTACT_OVERFLOW:
                raise RuntimeError(f"more than {MAX_CONTACTS} contacts on one particle "
                                   "(overlaps too large: stiffness too low or dt too big)")
            if err == ERR_EVENT_BUFFER_FULL:
                grown = np.zeros((2 * self.ev.shape[0], N_EV))
                grown[: self.ev_n[0]] = self.ev[: self.ev_n[0]]
                self.ev = grown

    def step(self, nsteps: int) -> None:
        """Advance nsteps with the shaking parameters, advancing self.t (test helper)."""
        if nsteps <= 0:
            return
        if self.n:
            self._kernel(nsteps, self.params(), self.t, self.t_rec0)
        self.t += nsteps * self.dt

    def settle(self, duration: Optional[float] = None) -> None:
        """Let particles settle under gravity with the container at rest (not recorded)."""
        dur = self.cfg.time.settle_time if duration is None else duration
        P = self.params(shaking=False, global_damping=self.cfg.time.settle_damping)
        n = int(round(dur / self.dt))
        if self.n and n > 0:
            self._kernel(n, P, 0.0, math.inf)
        self.v[:] = 0.0
        self.w[:] = 0.0
        self.acc[:] = 0.0
        self._restart_open_contacts(self.t, flag=0.0)

    def _restart_open_contacts(self, t: float, flag: float) -> None:
        """Reset event accumulators of currently open contacts to start at time t."""
        for rec in (self.hf, self.wf):
            rec[..., H_TSTART] = t
            rec[..., H_TPEAK] = t
            rec[..., H_JN] = 0.0
            rec[..., H_JT] = 0.0
            rec[..., H_FPEAK] = 0.0
            rec[..., H_FLAG] = flag

    def begin_recording(self, duration: float) -> None:
        """Start the recording window at the current time (events + wall force).

        Contacts already open are restarted (impulse counted from now) and flagged
        FLAG_START_CLIPPED, so event impulses and the wall-force integral agree exactly.
        """
        self.t_rec0 = self.t
        nb = int(math.ceil(duration * self.fs_force)) + 1
        self.wforce = np.zeros((N_WALLS, nb))
        self.ev_n[0] = 0
        self._restart_open_contacts(self.t, float(FLAG_START_CLIPPED))

    def finish_recording(self) -> tuple[np.ndarray, WallForce]:
        """Close still-open contacts (FLAG_END_CLIPPED) and return (raw events, wall force)."""
        rows = [self.ev[: self.ev_n[0]].copy()]
        extra = []
        for i in range(self.n):
            for k in range(self.hc[i]):
                extra.append(_raw_row(self.hf[i, k], KIND_PP, -1, i, int(self.hj[i, k]), self.t))
            for wi in range(N_WALLS):
                if self.won[i, wi]:
                    extra.append(_raw_row(self.wf[i, wi], KIND_PW, wi, i, -1, self.t))
        if extra:
            rows.append(np.array(extra))
        raw = np.concatenate(rows, axis=0) if rows else np.zeros((0, N_EV))
        wf = WallForce(force=self.wforce * self.fs_force, fs=self.fs_force, t0=self.t_rec0)
        self.t_rec0 = math.inf
        return raw, wf

    def energies(self) -> tuple[float, float, float]:
        """(KE_trans, KE_rot, PE_grav) [J] of the current state."""
        ke_t = 0.5 * float(np.sum(self.mass[:, None] * self.v ** 2))
        ke_r = 0.5 * float(np.sum((1.0 / self.inv_I)[:, None] * self.w ** 2)) if self.n else 0.0
        pe = self.cfg.gravity * float(np.sum(self.mass * self.x[:, 2]))
        return ke_t, ke_r, pe

    def sample(self) -> None:
        d = self.diag
        ke_t, ke_r, pe = self.energies()
        zc, _ = container_motion(self.t, self.A, 2 * math.pi * self.f, self.cfg.excitation.ramp_time)
        d.t.append(self.t)
        d.zc.append(zc)
        d.ke_trans.append(ke_t)
        d.ke_rot.append(ke_r)
        d.pe.append(pe)
        d.e_n.append(self.acc[A_EN])
        d.e_t.append(self.acc[A_ET])
        d.w_in.append(self.acc[A_WIN])
        d.d_n.append(self.acc[A_DN])
        d.d_t.append(self.acc[A_DT])
        d.d_corr.append(self.acc[A_DCORR])
        d.w_k.append(self.acc[A_WK])
        d.mean_height.append(float(np.mean(self.x[:, 2]) - zc) if self.n else 0.0)
        d.n_pp.append(self.acc[A_NPP])
        d.n_pw.append(self.acc[A_NPW])

    def run(self, duration: float, trajectory: Optional[list] = None) -> None:
        """Shake for ``duration`` seconds, sampling diagnostics (and optionally trajectory)."""
        out = self.cfg.output
        n_diag = max(1, int(round(out.diagnostics_interval / self.dt)))
        n_traj = max(1, int(round(out.trajectory_interval / self.dt)))
        n_total = int(round(duration / self.dt))
        if not self.diag.t:
            self.sample()
        done = 0
        while done < n_total:
            k = min(n_diag - (done % n_diag), n_total - done)
            if trajectory is not None:
                k = min(k, n_traj - (done % n_traj))
            self.step(k)
            done += k
            if done % n_diag == 0 or done == n_total:
                self.sample()
            if trajectory is not None and done % n_traj == 0:
                zc, _ = container_motion(self.t, self.A, 2 * math.pi * self.f,
                                         self.cfg.excitation.ramp_time)
                trajectory.append((self.t, zc, self.x.copy()))


def _raw_row(rec: np.ndarray, kind: int, wall: int, i: int, j: int, t_end: float) -> np.ndarray:
    row = np.zeros(N_EV)
    row[E_TSTART] = rec[H_TSTART]
    row[E_TEND] = t_end
    row[E_TPEAK] = rec[H_TPEAK]
    row[E_KIND] = kind
    row[E_WALL] = wall
    row[E_I] = i
    row[E_J] = j
    row[E_VN0] = rec[H_VN0]
    row[E_JN] = rec[H_JN]
    row[E_JT] = rec[H_JT]
    row[E_FPEAK] = rec[H_FPEAK]
    row[E_PX], row[E_PY], row[E_PZ] = rec[H_PX], rec[H_PY], rec[H_PZ]
    row[E_FLAG] = int(rec[H_FLAG]) | FLAG_END_CLIPPED
    return row


@dataclass
class DynamicsResult:
    cfg: SimConfig
    derived: dict
    diagnostics: dict
    radius: np.ndarray
    mass: np.ndarray
    final_x: np.ndarray
    final_v: np.ndarray
    final_w: np.ndarray
    wall_time_s: float
    events: np.ndarray            # processed event table (structured array), see events.tracker
    wall_force: WallForce
    event_params: dict
    trajectory: Optional[dict] = None


def run_dynamics(cfg: SimConfig) -> DynamicsResult:
    """Full single run: settle → shake (discard) → record. Returns events, wall force, diagnostics."""
    from gransim.events.tracker import process_events, resolve_event_params

    sim = Simulation(cfg)
    sim.settle()
    traj: Optional[list] = [] if cfg.output.save_trajectory else None
    sim.run(cfg.time.discard_time, trajectory=traj)
    sim.begin_recording(cfg.time.record_time)
    sim.run(cfg.time.record_time, trajectory=traj)
    raw, wall_force = sim.finish_recording()
    ep = resolve_event_params(cfg, sim.cp)
    events = process_events(raw, sim.radius, sim.mass, sim.f, ep)
    traj_d = None
    if traj:
        traj_d = {"t": np.array([a for a, _, _ in traj]),
                  "zc": np.array([b for _, b, _ in traj]),
                  "x": np.stack([c for _, _, c in traj])}
    return DynamicsResult(cfg=cfg, derived=derived_quantities(cfg),
                          diagnostics=sim.diag.as_arrays(), radius=sim.radius, mass=sim.mass,
                          final_x=sim.x.copy(), final_v=sim.v.copy(), final_w=sim.w.copy(),
                          wall_time_s=sim.wall_time, events=events, wall_force=wall_force,
                          event_params=ep, trajectory=traj_d)
