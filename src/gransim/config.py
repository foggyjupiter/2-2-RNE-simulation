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
    density_kg_m3: Optional[float] = Field(
        None, gt=0, description="If set, the particle mass is derived from it and the diameter "
                                "(mass = ρ π d³ / 6), overriding `mass`. Used by sweeps that "
                                "vary diameter and material density independently")

    @model_validator(mode="after")
    def _mass_from_density(self) -> "ParticleConfig":
        if self.density_kg_m3 is not None:
            m = self.density_kg_m3 * math.pi / 6.0 * self.diameter ** 3
            if not math.isclose(self.mass, m, rel_tol=1e-12):
                self.__dict__["mass"] = m  # bypass validate_assignment recursion
        return self

    @property
    def density(self) -> float:
        """Material density [kg/m^3] implied by mass and diameter."""
        return self.mass / (math.pi / 6.0 * self.diameter ** 3)


class ContactConfig(_Strict):
    model: Literal["linear", "hertz"] = Field(
        "hertz", description="Normal force law: Hertz (Tsuji damping, default) or linear "
                             "spring-dashpot")
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
        0.25, gt=0, le=1.0,
        description="hertz model only: multiply real Young's moduli by this factor "
                    "(1 = real stiffness). Default 0.25 = softest value that passed the "
                    "stiffness convergence test (contact time ≈ 1.7× real).")
    restitution_law: Literal["constant", "viscoelastic"] = Field(
        "constant",
        description="hertz model only. constant: Tsuji damping, e independent of the impact "
                    "speed. viscoelastic: Kuwabara-Kono / Brilliantov damping "
                    "F_d = (3/2) A k_h sqrt(δ) δ̇, so e rises towards 1 for slow impacts "
                    "(e = f(γ_ref·(v/v_ref)^(1/5))); restitution_pp/pw then give e at "
                    "restitution_ref_speed. Dense beds collide mostly slowly, sparse ones hit "
                    "the walls fast, so this changes how collective a dense bed is.")
    restitution_ref_speed: float = Field(
        0.5, gt=0, description="viscoelastic law: impact speed [m/s] at which restitution_pp/pw "
                               "hold (≈ the shaker's peak speed A·ω at 372 rpm)")
    tangential_stiffness_ratio: float = Field(
        2.0 / 7.0, gt=0, description="k_t / k_n (for Hertz: k_t / (dF_n/dδ))")
    tangential_damping_ratio: float = Field(0.5, ge=0, description="c_t / c_n")
    wall_roughness_deg: float = Field(
        2.0, ge=0, le=15,
        description="RMS surface slope [deg] of a frozen random micro-relief on the bottom "
                    "and top walls (wavelengths 0.5-2 d). Stands for BB asphericity, wall "
                    "unevenness and shaker wobble; breaks the unphysical lockstep bouncing "
                    "of sparse layers. 0 = perfectly flat walls.")

    @model_validator(mode="after")
    def _law_needs_hertz(self):
        if self.restitution_law == "viscoelastic" and self.model != "hertz":
            raise ValueError("contact.restitution_law=viscoelastic needs contact.model=hertz")
        return self

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


class AcousticConfig(_Strict):
    """Acoustic synthesis (spec 6). Re-rendering from saved events never reruns the dynamics."""

    fs: int = Field(48000, ge=8000, description="Output sampling rate [Hz]. The first-study "
                    "recordings are 48 kHz and most impact energy lies at 8-24 kHz.")
    warmup_s: float = Field(
        0.3, ge=0, description="Initial part of the record window rendered but not output, so "
                               "the ring of earlier impacts is present from the first sample [s]")
    duration_s: Optional[float] = Field(
        None, gt=0, description="Output length [s]; None = record window minus warm-up")
    pulse_shape: Literal["half_sine", "hertz", "delta"] = "half_sine"
    oversample: int = Field(16, ge=2, description="Fine-grid factor for pulse rendering")
    tau_mode: Literal["hertz_real", "fixed"] = Field(
        "hertz_real", description="Acoustic contact time: Hertz with the real moduli and each "
                                  "event's impact speed, or a fixed value")
    tau_fixed: float = Field(5.0e-5, gt=0, description="τ_c for tau_mode=fixed [s]")
    particle_youngs_modulus: Optional[float] = Field(
        None, gt=0, description="Real particle E [Pa] for the acoustic τ_c only; None = "
                                "particles.material. Lets a sweep vary the true modulus "
                                "without changing the (relaxed, validated) dynamic stiffness")
    band_split_hz: float = Field(
        1500.0, gt=0, description="Crossover between the wall-force component (below) and the "
                                  "impact-pulse component (above) [Hz]; complementary "
                                  "zero-phase filters, so no impulse is counted twice")
    highpass_hz: float = Field(
        500.0, ge=0, description="Radiation + microphone high-pass applied to every source "
                                 "[Hz]: a 5 cm box radiates little at low frequency and phone "
                                 "microphones roll off; without it the slowly varying wall load "
                                 "dominates below 4 kHz, unlike the recordings")
    wall_force_component: bool = Field(True, description="Source (i): low-passed wall force")
    impact_component: bool = Field(True, description="Source (ii): impact pulses (high band)")
    pp_gain: float = Field(1.0, ge=0, description="Relative gain of particle-particle impacts "
                                                  "(they reach the box only indirectly)")
    position_weighting: float = Field(
        0.0, ge=0, description="Exponent of the wall's fundamental plate-mode shape "
                               "sin(πu/L_u)·sin(πv/L_v) at each wall impact (normalised to mean 1): "
                               "impacts near the middle of a wall radiate more than near its edges. "
                               "0 = off")
    impact_gain_sigma: float = Field(
        0.0, ge=0, description="Log-normal σ of a per-impact radiation efficiency: stands for "
                               "unmodelled impact-to-impact variation (angle, spin, local wall "
                               "stiffness). 0 = off")
    impact_gain_norm: Literal["mean", "energy"] = Field(
        "mean", description="Normalisation of the per-impact factor: mean = unit mean amplitude "
                            "(radiated energy grows by e^(σ²)); energy = unit mean square, so σ "
                            "changes only how the impact energy is distributed, not the level "
                            "(keeps σ and the microphone gain independent when both are randomised)")
    response: Literal["measured", "synthetic"] = Field(
        "measured", description="measured = statistical model from first-study recordings "
                                "(assets/impulse_response/response_model.npz); synthetic = modes")
    response_file: Optional[str] = Field(None, description="Override path of the response model")
    drr: Optional[float] = Field(
        6.0, gt=0, description="Direct-to-reverberant energy ratio of the response. Default "
                               "calibrated against the first-study recordings (sim2real: "
                               "kurtosis, clipping and high-band level); None = the value "
                               "estimated directly from isolated onsets (0.19, biased low "
                               "because clipped onsets are excluded)")
    ring_tau: Optional[float] = Field(None, gt=0, description="Ring/reverb decay time constant "
                                                              "[s]; None = measured value")
    ir_length_s: float = Field(0.35, gt=0, description="Length of the drawn impulse responses [s]")
    direct_attack_s: float = Field(
        0.0, ge=0, description="Attack time constant of the direct (click) part of the response "
                               "[s]: envelope (1 − e^(−t/τa))·e^(−t/τd). Real clicks rise over "
                               "≈ 0.5 ms (modes building up, early reflections); 0 = instant")
    eq: bool = Field(True, description="Apply the sim2real spectral equaliser "
                                       "(assets/impulse_response/eq.npz) if present; one "
                                       "N-independent correction for all clips")
    synthetic_modes: list[tuple[float, float, float]] = Field(
        [(5900.0, 60.0, 0.3), (7400.0, 60.0, 0.4), (9400.0, 80.0, 1.0)],
        description="synthetic response: (frequency [Hz], Q, relative amplitude) per mode")
    mic_gain: float = Field(1.0, gt=0, description="Digital full-scale units per newton of "
                                                   "source force (calibrated by sim2real)")
    noise: bool = Field(True, description="Add background (motor) noise")
    noise_dir: Optional[str] = Field(None, description="Noise library folder (index.yaml); "
                                                       "None = assets/noise")
    noise_gain: float = Field(1.0, ge=0, description="1 = recorded level (same mic gain)")
    clip: bool = Field(True, description="Hard-clip at ±clip_level like the recording device")
    clip_level: float = Field(1.0, gt=0)
    bits: Literal[16, 32] = Field(16, description="16-bit PCM or 32-bit float wav")


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
    acoustics: AcousticConfig = AcousticConfig()

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
    try:
        from gransim.analysis.stiffness import load_allowed, stiffness_warning
        w = stiffness_warning(cfg, load_allowed())
    except Exception:  # pragma: no cover - a missing/corrupt file must not block runs
        w = None
    if w:
        msgs.append(w)
    for m in msgs:
        warnings.warn(m, stacklevel=2)
    return msgs
