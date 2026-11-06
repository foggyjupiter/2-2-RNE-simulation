"""Parallel, resumable dataset generation (spec 8.1).

For each job: run the dynamics → save the run (events, wall force, snapshot, summary,
ground-truth phase indicators) → render ``renders_per_run`` clips with independently
drawn acoustic conditions → write wavs and one metadata row per clip. A ``done.json``
marker makes the job skippable on restart; failures are logged and do not stop the sweep.

Layout under the output folder::

    runs/<run_id>/{config_snapshot.yaml, events.npz, wall_force.npz, summary.json, done.json}
    audio/<shard>/<run_id>_k<k>.wav
    manifest.csv, manifest.jsonl, metadata.csv   (rebuilt from the done markers)
    failures.jsonl
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from gransim.sweep.sampling import Job, build_jobs, in_box, nested

SHARD_SIZE = 500  # runs per audio sub-folder


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _base_config(spec: dict):
    from gransim.config import deep_merge, load_config

    base = load_config(spec["base"], spec.get("overrides") or {})
    return base, deep_merge


def _split_for(job: Job, spec: dict, rng: np.random.Generator) -> str:
    """Default split: held-out material box → 'test_ood'; else by group hash."""
    box = spec.get("holdout_box") or {}
    if box and in_box(job.params, {k: tuple(v) for k, v in box.items()}):
        return "test_ood"
    if job.pair_id:
        return "pairs"
    u = rng.random()
    return "train" if u < 0.8 else ("val" if u < 0.9 else "test")


def _nominal(job: Job, spec: dict, cfg) -> dict:
    """Nominal condition of a multi-condition job (what the experimenter sets); for an
    excitation-speed condition also the nominal rpm and Γ (the simulated ones carry the
    jitter). Empty for single-condition sweeps."""
    if job.condition_value is None:
        return {}
    key = (spec.get("conditions") or {}).get("key", "")
    out = {"condition_value": job.condition_value}
    if key in ("excitation.rpm", "excitation.frequency"):
        rpm = job.condition_value if key == "excitation.rpm" else 60.0 * job.condition_value
        ex = cfg.excitation.model_copy(update={"rpm": rpm, "frequency": None})
        _, _, g_nom = ex.resolved(cfg.gravity)
        out.update({"rpm_nominal": rpm, "gamma_nominal": g_nom})
    return out


def run_job(job: Job, spec: dict, out: str) -> dict:
    """Run one dynamics job and its renders. Returns {'run_id', 'rows'} or raises."""
    from gransim.acoustics.assets import write_wav
    from gransim.acoustics.render import render
    from gransim.analysis.phase import phase_indicators
    from gransim.config import AcousticConfig, SimConfig, save_snapshot
    from gransim.dynamics.engine import run_dynamics
    from gransim.events.io import save_events, save_wall_force

    out = Path(out)
    rd = out / "runs" / job.run_id
    done = rd / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    rd.mkdir(parents=True, exist_ok=True)
    base, deep_merge = _base_config(spec)
    data = deep_merge(base.model_dump(), nested(job.params))
    data["seed"] = job.seed
    cfg = SimConfig.model_validate(data)
    tic = time.perf_counter()
    res = run_dynamics(cfg)
    el = time.perf_counter() - tic
    t0 = res.wall_force.t0
    win = (t0, t0 + cfg.time.record_time)
    save_snapshot(cfg, rd)
    save_events(rd / "events.npz", res.events, res.event_params, win)
    save_wall_force(rd / "wall_force.npz", res.wall_force)
    phase = phase_indicators(res.events, res.diagnostics, cfg.particles.count, win)
    A, f, gam = cfg.excitation.resolved(cfg.gravity)
    summary = {"elapsed_s": el, "n_events": int(res.events.size),
               "n_impacts": int((res.events["contact_type"] == 0).sum()), **res.derived}
    (rd / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    rows = []
    shard = f"shard{int(job.run_id[1:6]) // SHARD_SIZE if job.run_id[1:6].isdigit() else 'P'}"
    adir = out / "audio" / shard
    adir.mkdir(parents=True, exist_ok=True)
    if job.system_id:   # multi-condition: split and render seeds shared by the system
        split = _split_for(job, spec, np.random.default_rng(job.split_seed ^ 0x5EED))
        seeds = list(job.render_seeds)
    else:               # single condition (unchanged behaviour of earlier sweeps)
        rng = np.random.default_rng(job.seed ^ 0x5EED)
        split = _split_for(job, spec, rng)
        seeds = [int(rng.integers(1 << 31)) for _ in job.render_params]
    for k, rp in enumerate(job.render_params):
        ac_data = deep_merge(cfg.acoustics.model_dump(), nested(rp).get("acoustics", {}))
        ac = AcousticConfig.model_validate(ac_data)
        r = render(cfg, res.events, res.wall_force, win, seed=seeds[k], acoustics=ac)
        name = f"{job.run_id}_k{k}.wav"
        write_wav(adir / name, r.fs, r.audio, bits=ac.bits)
        m = r.meta
        rows.append({
            "file_name": f"audio/{shard}/{name}", "run_id": job.run_id, "render": k,
            "system_id": job.system_id or job.run_id, "condition_index": job.condition_index,
            "N": cfg.particles.count, "split": split,
            "pair_id": job.pair_id, "pair_type": job.pair_type,
            # material (blind axes) and all physical parameters
            "particle_mass_kg": cfg.particles.mass, "particle_diameter_m": cfg.particles.diameter,
            "particle_density": cfg.particles.density,
            "restitution_pp": cfg.contact.restitution_pp, "restitution_pw": cfg.contact.restitution_pw,
            "friction_pp": cfg.contact.friction_pp, "friction_pw": cfg.contact.friction_pw,
            "particle_E_acoustic_pa": ac.particle_youngs_modulus or cfg.particles.material.youngs_modulus,
            "particle_E_dynamics_pa": cfg.particles.material.youngs_modulus,
            "poisson_ratio": cfg.particles.material.poisson_ratio,
            "rotation": cfg.contact.rotation, "wall_roughness_deg": cfg.contact.wall_roughness_deg,
            # excitation
            "amplitude_m": A, "frequency_hz": f, "rpm": 60 * f, "gamma": gam,
            **_nominal(job, spec, cfg),
            "excitation_mode": cfg.excitation.mode,
            # dynamics
            "contact_model": cfg.contact.model, "stiffness_mode": cfg.contact.stiffness_mode,
            "modulus_scale": cfg.contact.modulus_scale, "dt_s": summary["dt_s"],
            "stiffness_validated": cfg.contact.model == "hertz" and cfg.contact.modulus_scale >= 0.25,
            "v_impact": res.event_params["v_impact"], "duration_max": res.event_params["duration_max"],
            "merge_gap": res.event_params["merge_gap"],
            "n_impacts": summary["n_impacts"], "layers_F": summary["layers_F"],
            # acoustics
            "fs": m["fs"], "duration_s": m["duration_s"], "render_seed": m["render_seed"],
            "mic_gain": m["mic_gain"], "noise_gain": m["noise_gain"], "noise_file": m["noise_file"],
            "noise_start": m.get("noise_start"), "noise_phase_aligned": m.get("noise_phase_aligned"),
            "noise_rpm_scaled": m.get("noise_rpm_scaled", False),
            "drr": m["drr"], "ring_tau_s": m["ring_tau_s"], "eq": m["eq"],
            "pulse_shape": m["pulse_shape"], "tau_mode": m["tau_mode"],
            "tau_c_median_s": m["tau_c_median_s"], "band_split_hz": m["band_split_hz"],
            "highpass_hz": m["highpass_hz"], "pp_gain": m["pp_gain"],
            "impact_gain_sigma": m.get("impact_gain_sigma", 0.0),
            "impact_gain_norm": m.get("impact_gain_norm", "mean"),
            "position_weighting": m.get("position_weighting", 0.0),
            "direct_attack_s": ac.direct_attack_s,
            "impact_source": ac.impact_source,
            "clip_fraction": m["clip_fraction"], "rms": m["rms"],
            # ground-truth phase (spec 9.1)
            **{f"gt_{k_}": v for k_, v in phase.items()},
            "seed": job.seed,
        })
    rec = {"run_id": job.run_id, "elapsed_s": el, "rows": rows}
    done.write_text(json.dumps(rec, default=float))
    return rec


def run_sweep(spec_path: str | Path, out: str | Path, workers: int | None = None,
              limit: int | None = None, time_budget: float | None = None,
              progress=print, shard: tuple[int, int] | None = None) -> dict:
    """Run (or resume) a sweep. ``limit`` caps the number of jobs started this call.

    ``shard`` = (k, n): only jobs whose index in the full job list ≡ k (mod n), so n
    machines can share one sweep; each job's parameters, seeds and split are the same as
    in an unsharded run (they depend on the spec only)."""
    spec = load_spec(spec_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    jobs = build_jobs(spec)
    if shard is not None:
        k, n = shard
        if not 0 <= k < n:
            raise ValueError(f"shard {k}/{n}: need 0 <= k < n")
        jobs = [j for i, j in enumerate(jobs) if i % n == k]
    todo = [j for j in jobs if not (out / "runs" / j.run_id / "done.json").exists()]
    # interleave large and small N so workers stay balanced and early stops are representative
    if limit is not None:
        todo = todo[:limit]
    progress(f"{len(jobs)} jobs, {len(jobs) - len(todo) if limit is None else '?'} done, "
             f"{len(todo)} to run")
    t_start = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        it = iter(todo)

        def submit():
            if time_budget is not None and time.time() - t_start > time_budget:
                return False
            j = next(it, None)
            if j is None:
                return False
            futs[ex.submit(run_job, j, spec, str(out))] = j
            return True

        for _ in range((workers or 4) * 2):
            if not submit():
                break
        while futs:
            for fu in as_completed(list(futs)):
                j = futs.pop(fu)
                try:
                    r = fu.result()
                    n_ok += 1
                    progress(f"[{n_ok + n_fail}/{len(todo)}] {j.run_id} N={j.params.get('particles.count')} "
                             f"{r['elapsed_s']:.0f}s")
                except Exception as e:  # keep going (spec 8.1)
                    n_fail += 1
                    with open(out / "failures.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"run_id": j.run_id, "params": j.params, "error": repr(e),
                                             "trace": traceback.format_exc()}) + "\n")
                    progress(f"FAILED {j.run_id}: {e!r}")
                submit()
                break
    write_manifest(out)
    return {"ok": n_ok, "failed": n_fail}


def write_manifest(out: str | Path) -> int:
    """Rebuild manifest.csv / manifest.jsonl from all done markers."""
    import csv

    out = Path(out)
    rows = []
    for d in sorted((out / "runs").glob("*/done.json")):
        rows += json.loads(d.read_text())["rows"]
    if not rows:
        return 0
    for r in rows:                       # rows written before these columns existed
        r.setdefault("system_id", r["run_id"])
        r.setdefault("condition_index", 0)
        r.setdefault("noise_rpm_scaled", False)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # same table under the name the Hugging Face "audiofolder" loader expects
    (out / "metadata.csv").write_bytes((out / "manifest.csv").read_bytes())
    return len(rows)
