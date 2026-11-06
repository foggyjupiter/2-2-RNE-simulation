"""Full acoustic rendering of one dynamics run (spec 6): sources → responses → microphone.

    audio = clip( mic_gain · Σ_channels (source_c ⊛ h_c) + noise_gain · noise )

Works only from saved run outputs (events.npz, wall_force.npz, config_snapshot.yaml),
so the audio can be re-rendered with other acoustic settings in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from gransim.acoustics.assets import write_wav
from gransim.acoustics.noise import draw_noise
from gransim.acoustics.source import CHANNELS, build_sources
from gransim.acoustics.transfer import apply_response, draw_responses
from gransim.config import AcousticConfig, SimConfig


@dataclass
class Rendered:
    audio: np.ndarray            # final signal (gain, noise, clipping applied) at fs
    fs: int
    clean: np.ndarray            # mic_gain · Σ wet channels, no noise, no clipping
    components: dict             # per channel wet signals and wall/impact split (diagnostics)
    meta: dict


def load_run(run_dir: str | Path):
    from gransim.events.io import load_events, load_wall_force

    run_dir = Path(run_dir)
    snap = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    cfg = SimConfig.model_validate(snap["config"])
    ev, params, win = load_events(run_dir / "events.npz")
    wf = load_wall_force(run_dir / "wall_force.npz")
    return cfg, ev, wf, win


@lru_cache(maxsize=4)
def load_click_library(path: str | None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parents[3] / "assets" / "impacts" / "clicks_bb.npz"
    if not p.exists():
        raise FileNotFoundError(f"click library {p} not found; build it with `gransim build-clicks`")
    with np.load(p) as z:
        return {"clips": z["clips"].astype(np.float64), "fs": int(z["fs"]), "pre": int(z["pre"]),
                "path": str(p)}


def replace_direct_with_clicks(wet: dict, impacts: dict, direct: dict, ac: AcousticConfig,
                               seed: int, t0: float, n_tot: int) -> dict:
    """impact_source = recorded: in each channel, swap the impacts' synthetic direct sound
    (pulse ⊛ direct part of the response) for real clicks at the same times, each scaled by
    the impact's weight; one factor per run gives the clicks the synthetic direct energy
    (level-neutral). The reverberant tail of every impact stays as synthesised."""
    if direct is None or not direct:
        raise ValueError("impact_source=recorded needs the measured response model")
    lib = load_click_library(ac.click_library)
    if lib["fs"] != ac.fs:
        raise ValueError(f"click library at {lib['fs']} Hz, rendering at {ac.fs} Hz")
    C, pre = lib["clips"], lib["pre"]
    L = C.shape[1]
    crng = np.random.default_rng([seed, 4242])
    syn, rec = {}, {}
    for c, d in impacts.items():
        syn[c] = apply_response(d["signal"], direct[c])
        y = np.zeros(n_tot)
        if len(d["t"]):
            k = crng.integers(len(C), size=len(d["t"]))
            start = np.round((np.asarray(d["t"]) - t0) * ac.fs).astype(np.int64) - pre
            pos = start[:, None] + np.arange(L)[None, :]
            val = np.asarray(d["w"])[:, None] * C[k]
            ok = (pos >= 0) & (pos < n_tot)
            y = np.bincount(pos[ok], weights=val[ok], minlength=n_tot)[:n_tot]
        rec[c] = y
    e_syn = sum(float(np.sum(v * v)) for v in syn.values())
    e_rec = sum(float(np.sum(v * v)) for v in rec.values())
    kappa = float(np.sqrt(e_syn / e_rec)) if e_rec > 0 else 0.0
    for c in impacts:
        wet[c] = wet[c] - syn[c] + kappa * rec[c]
    return {"impact_source": "recorded", "click_library": lib["path"], "click_scale": kappa,
            "n_clicks_library": int(len(C))}


def render(cfg: SimConfig, events: np.ndarray, wall_force, record_window: tuple,
           seed: int = 0, acoustics: Optional[AcousticConfig] = None,
           keep_components: bool = False) -> Rendered:
    """Render audio for a run. ``acoustics`` overrides cfg.acoustics (re-rendering)."""
    ac = acoustics or cfg.acoustics
    if acoustics is not None:
        cfg = cfg.model_copy(update={"acoustics": acoustics})
    rng = np.random.default_rng(seed)
    fs = ac.fs
    t0_rec, t1_rec = record_window
    avail = t1_rec - t0_rec - ac.warmup_s
    if avail <= 0:
        raise ValueError(f"record window {t1_rec - t0_rec:.3f} s shorter than warm-up {ac.warmup_s} s")
    dur = ac.duration_s or avail
    if dur > avail + 1e-9:
        raise ValueError(f"requested {dur} s but only {avail:.3f} s after warm-up")
    n_tot = int(round((ac.warmup_s + dur) * fs))
    n_out = int(round(dur * fs))

    impacts = {} if ac.impact_source == "recorded" else None
    src, src_info = build_sources(events, wall_force, cfg, t0_rec, n_tot, seed, impacts=impacts)
    direct = {} if impacts is not None else None
    irs, ir_info = draw_responses(ac, list(CHANNELS), rng, direct_parts=direct)
    wet = {c: apply_response(src[c], irs[c]) for c in CHANNELS}
    if impacts is not None:
        src_info.update(replace_direct_with_clicks(wet, impacts, direct, ac, seed, t0_rec, n_tot))
    clean = ac.mic_gain * sum(wet[c] for c in CHANNELS)[n_tot - n_out:]

    A, f, gam = cfg.excitation.resolved(cfg.gravity)
    rpm = 60.0 * f
    noise_meta = {"noise_file": None}
    y = clean.copy()
    if ac.noise and ac.noise_gain > 0:
        nz, noise_meta = draw_noise(ac, rpm, n_out, rng)
        y += ac.noise_gain * nz
    clipped = float(np.mean(np.abs(y) >= ac.clip_level)) if ac.clip else 0.0
    if ac.clip:
        y = np.clip(y, -ac.clip_level, ac.clip_level)
    meta = {"fs": fs, "duration_s": dur, "audio_t0_s": t0_rec + ac.warmup_s,
            "render_seed": seed, "mic_gain": ac.mic_gain, "noise_gain": ac.noise_gain if ac.noise else 0.0,
            "clip_fraction": clipped, "rms": float(np.sqrt(np.mean(y ** 2))),
            "rms_clean": float(np.sqrt(np.mean(clean ** 2))),
            **src_info, **ir_info, **noise_meta,
            "acoustics": ac.model_dump(mode="json")}
    comps = {}
    if keep_components:
        comps = {c: ac.mic_gain * wet[c][n_tot - n_out:] for c in CHANNELS}
        comps["sources"] = {c: src[c][n_tot - n_out:] for c in CHANNELS}
    return Rendered(audio=y, fs=fs, clean=clean, components=comps, meta=meta)


def render_run(run_dir: str | Path, out_wav: str | Path, seed: int = 0,
               acoustics: Optional[AcousticConfig] = None) -> dict:
    """Render a saved run to a wav file; returns the metadata (also written as .json)."""
    import json

    cfg, ev, wf, win = load_run(run_dir)
    r = render(cfg, ev, wf, win, seed=seed, acoustics=acoustics)
    ac = acoustics or cfg.acoustics
    write_wav(out_wav, r.fs, r.audio, bits=ac.bits)
    Path(out_wav).with_suffix(".json").write_text(json.dumps(r.meta, indent=2))
    return r.meta
