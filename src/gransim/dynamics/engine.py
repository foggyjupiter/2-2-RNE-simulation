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
from gransim.dynamics.init import lattice_positions, particle_properties
from gransim.dynamics.neighbors import build_cells, cell_coord, make_grid
from gransim.dynamics.walls import N_WALLS, container_motion, wall_plane

MAX_CONTACTS = 16     # pp contacts stored per particle (partners j > i)
N_HIST = 4            # per-contact float record: [xi_x, xi_y, xi_z, k_t of previous step]

# accumulator indices
A_WIN, A_DN, A_DT, A_EN, A_ET, A_MAXOV, A_NPP, A_NPW, A_DCORR, A_WK = range(10)
N_ACC = 10

ERR_OK = 0
ERR_CONTACT_OVERFLOW = 1


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
            hj, hc, hf, nj, nc, nf,
            wf, won, nwf, nwon,
            head, nxt, F, T, Fd, Td, dims, P, t0, nsteps, acc):
    """Advance the system by ``nsteps`` steps starting at shaking time t0 [s].

    History buffers are swapped internally every step; returns
    (number_of_swaps_mod_2, error_code). The caller must swap its own
    references when the parity is 1.
    """
    N = x.shape[0]
    dt = P.dt
    parity = 0
    err = ERR_OK
    nyc = dims[1]
    nzc = dims[2]
    for s in range(nsteps):
        t = t0 + s * dt
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
                    if won[i, wi] == 1:
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
    return parity, err


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


class Simulation:
    """Holds state and buffers; thin wrapper over :func:`advance`.

    Typical use::

        sim = Simulation(cfg)
        sim.settle()
        sim.run(cfg.time.discard_time + cfg.time.record_time)

    Tests may overwrite ``sim.x``, ``sim.v``, ``sim.w`` before stepping.
    """

    def __init__(self, cfg: SimConfig, positions: Optional[np.ndarray] = None,
                 warn: bool = True):
        self.cfg = cfg
        if warn:
            check_config(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.radius, self.mass = particle_properties(cfg, self.rng)
        n = self.radius.size
        self.n = n
        self.inv_I = 1.0 / (0.4 * self.mass * self.radius ** 2) if n else np.zeros(0)
        self.x = positions.copy() if positions is not None else lattice_positions(
            cfg, self.radius, self.rng)
        self.v = np.zeros((n, 3))
        self.w = np.zeros((n, 3))
        self.cp = derive_contact_params(cfg)
        self.A, self.f, self.gamma = cfg.excitation.resolved(cfg.gravity)
        dmax = 2.0 * (self.radius.max() if n else cfg.particles.diameter / 2)
        self.dims, self.cell = make_grid(cfg.container.size, dmax)
        self._alloc()
        self.t = 0.0  # shaking time [s]
        self.acc = np.zeros(N_ACC)
        self.diag = Diagnostics()
        self.wall_time = 0.0

    def _alloc(self) -> None:
        n = self.n
        self.hj = np.full((n, MAX_CONTACTS), -1, np.int32)
        self.nj = self.hj.copy()
        self.hc = np.zeros(n, np.int32)
        self.nc = self.hc.copy()
        self.hf = np.zeros((n, MAX_CONTACTS, N_HIST))
        self.nf = self.hf.copy()
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

    def step(self, nsteps: int, params: Optional[StepParams] = None, t0: Optional[float] = None) -> None:
        """Advance nsteps; advances self.t only when params is the shaking parameter set."""
        if self.n == 0 or nsteps <= 0:
            if params is None:
                self.t += nsteps * self.dt
            return
        P = params if params is not None else self.params()
        start = self.t if t0 is None else t0
        tic = time.perf_counter()
        parity, err = advance(self.x, self.v, self.w, self.radius, self.mass, self.inv_I,
                              self.hj, self.hc, self.hf, self.nj, self.nc, self.nf,
                              self.wf, self.won, self.nwf, self.nwon,
                              self.head, self.nxt, self.F, self.T, self.Fd, self.Td,
                              self.dims, P,
                              start, nsteps, self.acc)
        self.wall_time += time.perf_counter() - tic
        if parity:
            self.hj, self.nj = self.nj, self.hj
            self.hc, self.nc = self.nc, self.hc
            self.hf, self.nf = self.nf, self.hf
            self.wf, self.nwf = self.nwf, self.wf
            self.won, self.nwon = self.nwon, self.won
        if err == ERR_CONTACT_OVERFLOW:
            raise RuntimeError(f"more than {MAX_CONTACTS} contacts on one particle "
                               "(overlaps too large: stiffness too low or dt too big)")
        if params is None:
            self.t += nsteps * self.dt

    def settle(self, duration: Optional[float] = None) -> None:
        """Let particles settle under gravity with the container at rest (not recorded)."""
        dur = self.cfg.time.settle_time if duration is None else duration
        P = self.params(shaking=False, global_damping=self.cfg.time.settle_damping)
        n = int(round(dur / self.dt))
        chunk = 10000
        while n > 0:
            k = min(chunk, n)
            self.step(k, params=P, t0=0.0)
            n -= k
        # free decay of the remaining motion without extra damping, then reset accounting
        self.v[:] = 0.0
        self.w[:] = 0.0
        self.acc[:] = 0.0

    def energies(self) -> tuple[float, float, float]:
        """(KE_trans, KE_rot, PE_grav) [J] of the current state."""
        ke_t = 0.5 * float(np.sum(self.mass[:, None] * self.v ** 2))
        ke_r = 0.5 * float(np.sum((1.0 / self.inv_I)[:, None] * self.w ** 2)) if self.n else 0.0
        pe = self.cfg.gravity * float(np.sum(self.mass * self.x[:, 2]))
        return ke_t, ke_r, pe

    def sample(self) -> None:
        d = self.diag
        ke_t, ke_r, pe = self.energies()
        zc, _ = _motion_py(self.t, self.A, 2 * math.pi * self.f, self.cfg.excitation.ramp_time)
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
        P = self.params(shaking=True)
        if not self.diag.t:
            self.sample()
        done = 0
        while done < n_total:
            k = min(n_diag, n_total - done)
            if trajectory is not None:
                k = min(k, n_traj - (done % n_traj))
            self.step(k, params=P, t0=self.t)
            self.t += k * self.dt
            done += k
            if done % n_diag == 0 or done == n_total:
                self.sample()
            if trajectory is not None and done % n_traj == 0:
                zc, _ = _motion_py(self.t, self.A, 2 * math.pi * self.f,
                                   self.cfg.excitation.ramp_time)
                trajectory.append((self.t, zc, self.x.copy()))


def _motion_py(t: float, amp: float, omega: float, ramp: float) -> tuple[float, float]:
    return container_motion(t, amp, omega, ramp)


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
    trajectory: Optional[dict] = None


def run_dynamics(cfg: SimConfig) -> DynamicsResult:
    """Full single run: settle → shake (discard + record). Returns diagnostics and final state."""
    sim = Simulation(cfg)
    sim.settle()
    traj: Optional[list] = [] if cfg.output.save_trajectory else None
    sim.run(cfg.time.discard_time + cfg.time.record_time, trajectory=traj)
    traj_d = None
    if traj:
        traj_d = {"t": np.array([a for a, _, _ in traj]),
                  "zc": np.array([b for _, b, _ in traj]),
                  "x": np.stack([c for _, _, c in traj])}
    return DynamicsResult(cfg=cfg, derived=derived_quantities(cfg),
                          diagnostics=sim.diag.as_arrays(), radius=sim.radius, mass=sim.mass,
                          final_x=sim.x.copy(), final_v=sim.v.copy(), final_w=sim.w.copy(),
                          wall_time_s=sim.wall_time, trajectory=traj_d)
