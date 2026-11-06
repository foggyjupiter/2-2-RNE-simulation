"""Configuration schema, validation, unit conversion and dimensionless numbers.

All physical quantities are SI (m, kg, s, N, Pa). The YAML file mirrors the
pydantic models below one-to-one; every field can also be set from Python::

    cfg = SimConfig(particles={"count": 150}, excitation={"frequency": 8.0})

Coordinates: container (inner box) frame has its origin at the inner bottom
corner, x/y horizontal, z up. The whole container moves vertically as
``z_c(t) = A * s(t) * sin(2 pi f t)`` in the lab frame, where ``s(t)`` is a
smooth start-up ramp (see :class:`ExcitationConfig`).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

G_DEFAULT = 9.81


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Unit helpers
# --------------------------------------------------------------------------

def gamma_from_amplitude(amplitude: float, frequency: float, g: float = G_DEFAULT) -> float:
    """Dimensionless acceleration Γ = A (2πf)² / g. amplitude [m], frequency [Hz]."""
    return amplitude * (2.0 * math.pi * frequency) ** 2 / g if g > 0 else math.inf


def amplitude_from_gamma(gamma: float, frequency: float, g: float = G_DEFAULT) -> float:
    """Amplitude A [m] giving Γ at frequency f [Hz]."""
    return gamma * g / (2.0 * math.pi * frequency) ** 2


def frequency_from_gamma(gamma: float, amplitude: float, g: float = G_DEFAULT) -> float:
    """Frequency f [Hz] giving Γ at amplitude A [m]."""
    return math.sqrt(gamma * g / amplitude) / (2.0 * math.pi)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

class Material(_Strict):
    youngs_modulus: float = Field(gt=0, description="Young's modulus E [Pa]")
    poisson_ratio: float = Field(ge=0, lt=0.5, description="Poisson ratio ν [-]")


class ContainerConfig(_Strict):
    size: tuple[float, float, float] = Field(
        (0.05, 0.05, 0.07), description="Inner dimensions (Lx, Ly, Lz) [m]")
    wall_thickness: float = Field(0.003, gt=0, description="Wall thickness [m] (metadata)")
    mass: float = Field(0.070, gt=0, description="Container mass [kg] (metadata, mass ratio)")
    material: Material = Material(youngs_modulus=3.2e9, poisson_ratio=0.37)  # acrylic (PMMA)

    @model_validator(mode="after")
    def _positive(self) -> "ContainerConfig":
        if min(self.size) <= 0:
            raise ValueError(f"container.size must be positive, got {self.size}")
        return self


class ParticleConfig(_Strict):
    count: int = Field(100, ge=0, description="Number of particles N (the label)")
    diameter: float = Field(0.006, gt=0, description="Mean diameter d [m]")
    mass: float = Field(1.2e-4, gt=0, description="Mass of a particle of diameter d [kg]")
    diameter_rel_std: float = Field(
        0.0, ge=0, lt=0.3,
        description="Relative std of diameter for polydispersity (0 = monodisperse). "
                    "Density is held fixed, so mass scales with d^3.")
    material: Material = Material(youngs_modulus=2.3e9, poisson_ratio=0.35)  # ABS-like BB
    init: Literal["random", "lattice"] = Field(
        "random", description="Initial placement before settling: random sequential addition "
                              "(disordered packing) or jittered lattice")

    @property
    def density(self) -> float:
        """Material density [kg/m^3] implied by mass and diameter."""
        return self.mass / (math.pi / 6.0 * self.diameter ** 3)


class ContactConfig(_Strict):
    model: Literal["linear", "hertz"] = Field(
        "linear", description="Normal force law: linear spring-dashpot or Hertz (Tsuji damping)")
    restitution_pp: float = Field(0.8, gt=0, le=1, description="Particle-particle restitution e")
    restitution_pw: float = Field(0.8, gt=0, le=1, description="Particle-wall restitution e")
    friction_pp: float = Field(0.3, ge=0, description="Particle-particle Coulomb friction μ")
    friction_pw: float = Field(0.3, ge=0, description="Particle-wall Coulomb friction μ")
    rotation: bool = Field(True, description="Integrate angular velocity / apply torques")
    contact_time: float = Field(
        2.0e-4, gt=0,
        description="linear model only: undamped binary contact duration π·sqrt(m_eff/k_n) [s]. "
                    "k_n is derived from it separately for particle-particle and particle-wall "
                    "pairs (relaxed stiffness).")
    modulus_scale: float = Field(
        1.0, gt=0, le=1.0,
        description="hertz model only: multiply real Young's moduli by this factor "
                    "(1 = real stiffness).")
    tangential_stiffness_ratio: float = Field(
        2.0 / 7.0, gt=0, description="k_t / k_n (for Hertz: k_t / (dF_n/dδ))")
    tangential_damping_ratio: float = Field(0.5, ge=0, description="c_t / c_n")

    @property
    def stiffness_mode(self) -> str:
        """'real' for Hertz with real moduli, otherwise 'relaxed'."""
        return "real" if (self.model == "hertz" and self.modulus_scale == 1.0) else "relaxed"


class ExcitationConfig(_Strict):
    """Vertical sinusoidal motion of the whole container.

    ``fixed_amplitude`` (experiment-replica mode): A is mechanically fixed
    (crank drive), only f (or rpm) is a variable and Γ follows from it.
    Specifying Γ in this mode is an error. ``free``: give any two of
    (A, f|rpm, Γ).
    """

    mode: Literal["fixed_amplitude", "free"] = "fixed_amplitude"
    amplitude: Optional[float] = Field(0.0134, ge=0, description="A [m]")
    frequency: Optional[float] = Field(None, gt=0, description="f [Hz]")
    rpm: Optional[float] = Field(None, gt=0, description="Motor speed [rev/min] = 60 f")
    gamma: Optional[float] = Field(None, ge=0, description="Γ = A(2πf)²/g [-]")
    ramp_time: float = Field(
        0.2, ge=0, description="Smooth start-up of the amplitude envelope [s]")

    @model_validator(mode="after")
    def _check(self) -> "ExcitationConfig":
        if self.frequency is not None and self.rpm is not None:
            if not math.isclose(self.frequency * 60.0, self.rpm, rel_tol=1e-9):
                raise ValueError("excitation: give frequency OR rpm, not both "
                                 f"(frequency={self.frequency}, rpm={self.rpm})")
        f_given = self.frequency is not None or self.rpm is not None
        if self.mode == "fixed_amplitude":
            if self.gamma is not None:
                raise ValueError(
                    "excitation.mode=fixed_amplitude: Γ is derived from f because A is fixed; "
                    "do not set excitation.gamma (use mode=free to choose Γ and f independently)")
            if self.amplitude is None:
                raise ValueError("excitation.mode=fixed_amplitude requires excitation.amplitude")
            if not f_given:
                raise ValueError("excitation.mode=fixed_amplitude requires frequency or rpm")
        else:
            n_given = sum([self.amplitude is not None, f_given, self.gamma is not None])
            if n_given != 2:
                raise ValueError(
                    "excitation.mode=free: give exactly two of amplitude, frequency|rpm, gamma "
                    f"(got amplitude={self.amplitude}, frequency={self.frequency}, "
                    f"rpm={self.rpm}, gamma={self.gamma})")
        return self

    def resolved(self, g: float = G_DEFAULT) -> tuple[float, float, float]:
        """Return (A [m], f [Hz], Γ [-])."""
        f = self.frequency if self.frequency is not None else (
            self.rpm / 60.0 if self.rpm is not None else None)
        A, gam = self.amplitude, self.gamma
        if self.mode == "fixed_amplitude":
            return A, f, gamma_from_amplitude(A, f, g)
        if f is None:
            if A == 0:
                raise ValueError("excitation: cannot derive frequency from Γ when amplitude is 0")
            f = frequency_from_gamma(gam, A, g)
        elif A is None:
            A = amplitude_from_gamma(gam, f, g)
        else:
            gam = gamma_from_amplitude(A, f, g)
        return A, f, gam


class TimeConfig(_Strict):
    dt: Optional[float] = Field(
        None, gt=0, description="Integration step [s]; None = automatic (contact_time/dt_safety)")
    dt_safety: float = Field(20.0, ge=5, description="Minimum steps per contact duration")
    max_restitution_scatter: float = Field(
        0.05, gt=0,
        description="Automatic dt is halved until the per-collision restitution scatter "
                    "(numerical, from the detection sub-step phase) is below this fraction")
    settle_time: float = Field(0.5, ge=0, description="Settling under gravity before shaking [s]")
    settle_damping: float = Field(
        20.0, ge=0, description="Extra viscous damping rate during settling only [1/s]")
    discard_time: float = Field(
        1.0, ge=0, description="Shaking time discarded before recording (incl. ramp) [s]")
    record_time: float = Field(5.0, gt=0, description="Recorded duration [s]")


class EventConfig(_Strict):
    """Contact-event classification (spec 5.1). ``None`` = automatic default."""

    v_impact: Optional[float] = Field(
        None, gt=0,
        description="Approach speed above which a contact can be an impact [m/s]; "
                    "default 0.1·sqrt(2 g d)")
    duration_max: Optional[float] = Field(
        None, gt=0,
        description="Longest contact still counted as an impact [s]; "
                    "default 3 × binary contact time at v_impact")
    merge_gap: Optional[float] = Field(
        None, ge=0,
        description="Re-contacts of the same pair within this gap (and slower than v_impact) "
                    "are merged into one event [s]; default = binary contact time")
    wall_force_rate: float = Field(
        22050.0, gt=0, description="Sampling rate of the per-wall force time series [Hz]")


class OutputConfig(_Strict):
    diagnostics_interval: float = Field(1.0e-3, gt=0, description="Energy/state sampling [s]")
    save_trajectory: bool = False
    trajectory_interval: float = Field(2.0e-3, gt=0, description="Trajectory frame interval [s]")


class SimConfig(_Strict):
    seed: int = 0
    gravity: float = Field(G_DEFAULT, ge=0, description="g [m/s^2], acts along -z")
    container: ContainerConfig = ContainerConfig()
    particles: ParticleConfig = ParticleConfig()
    contact: ContactConfig = ContactConfig()
    excitation: ExcitationConfig = ExcitationConfig(frequency=8.0)
    time: TimeConfig = TimeConfig()
    events: EventConfig = EventConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _cross(self) -> "SimConfig":
        d = self.particles.diameter * (1 + 3 * self.particles.diameter_rel_std)
        if d >= min(self.container.size):
            raise ValueError(f"particle diameter {d} m does not fit container {self.container.size}")
        return self


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_config(path: str | Path, overrides: Optional[dict] = None) -> SimConfig:
    """Load and validate a YAML config. ``overrides`` is a nested dict merged on top."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = deep_merge(data, overrides)
    return SimConfig.model_validate(data)


def deep_merge(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_snapshot(cfg: SimConfig, out_dir: str | Path, extra: Optional[dict] = None) -> Path:
    """Write the exact config plus derived dimensionless numbers to out_dir/config_snapshot.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"config": cfg.model_dump(mode="json"), "derived": derived_quantities(cfg)}
    if extra:
        doc.update(extra)
    p = out_dir / "config_snapshot.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    return p


# --------------------------------------------------------------------------
# Derived numbers and checks
# --------------------------------------------------------------------------

def derived_quantities(cfg: SimConfig) -> dict:
    """Dimensionless parameters and resolved values (also used for warnings)."""
    from gransim.dynamics.contact import derive_contact_params  # avoid import cycle

    g = cfg.gravity
    A, f, gam = cfg.excitation.resolved(g)
    Lx, Ly, Lz = cfg.container.size
    p = cfg.particles
    N, d = p.count, p.diameter
    cp = derive_contact_params(cfg)
    T = 1.0 / f
    out = {
        "amplitude_m": A,
        "frequency_hz": f,
        "rpm": 60.0 * f,
        "gamma": gam,
        "shaking_strength_S": (A * 2 * math.pi * f) ** 2 / (g * d) if g > 0 else None,
        "layers_F": N * d ** 2 / (Lx * Ly),
        "particle_density_kg_m3": p.density,
        "particle_to_container_size": d / min(Lx, Ly, Lz),
        "particle_to_container_mass": N * p.mass / cfg.container.mass,
        "stiffness_mode": cfg.contact.stiffness_mode,
        "contact_time_pp_s": cp["contact_time_pp"],
        "contact_time_pw_s": cp["contact_time_pw"],
        "contact_time_over_period": max(cp["contact_time_pp"], cp["contact_time_pw"]) / T,
        "dt_s": cp["dt"],
        "dt_recommended_s": cp["dt_recommended"],
        "steps_per_contact": min(cp["contact_time_pp"], cp["contact_time_pw"]) / cp["dt"],
        "restitution_scatter": cp["restitution_scatter"],
    }
    return out


def check_config(cfg: SimConfig) -> list[str]:
    """Return a list of human-readable warnings (and emit them via ``warnings``)."""
    dq = derived_quantities(cfg)
    msgs = []
    if dq["contact_time_over_period"] > 0.01:
        msgs.append(
            f"contact time / excitation period = {dq['contact_time_over_period']:.3g} > 0.01: "
            "stiffness too low relative to shaking period (spec 4.1 condition (a))")
    if dq["dt_s"] > dq["dt_recommended_s"] * 1.0001:
        msgs.append(
            f"dt = {dq['dt_s']:.3g} s exceeds recommended {dq['dt_recommended_s']:.3g} s "
            f"(contact_time / {cfg.time.dt_safety:g})")
    if dq["layers_F"] > 0 and cfg.particles.count * cfg.particles.diameter ** 3 > 0.5 * math.prod(cfg.container.size):
        msgs.append("particle volume exceeds half the container volume")
    for m in msgs:
        warnings.warn(m, stacklevel=2)
    return msgs
