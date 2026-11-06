"""Re-render the audio of an existing sweep with other acoustic settings (no dynamics).

Rendering works only from the saved run outputs, so a dataset's audio can be redone in
seconds per clip when the acoustic model changes (e.g. a sim2real improvement), while
the expensive dynamics are reused.

Each clip keeps its original per-render conditions (mic gain, DRR, noise level, render
seed → same impulse-response and noise draws). On top of that:

* ``fixed``  — acoustic settings applied to every clip, e.g. ``{"direct_attack_s": 2e-4}``
* ``axes``   — new per-render acoustic axes, e.g.
  ``{"acoustics.impact_gain_sigma": {"dist": "uniform", "low": 0.6, "high": 1.5}}``,
  drawn from the clip's render seed, so all conditions of a multi-condition system share
  the draw (same recording setup) and a rerun gives identical files.

Output layout mirrors a sweep (``runs/<run_id>/done.json`` + ``audio/`` + manifests), so
:func:`gransim.sweep.runner.write_manifest` and :func:`gransim.sweep.hf_upload.upload`
work on it unchanged::

    gransim rerender datasets/v1 --out datasets/v1_r2 \\
        --axis acoustics.impact_gain_sigma=uniform:0.6:1.5 --set acoustics.direct_attack_s=2e-4
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

SALT = 0x2E2E  # separates the new-axis draws from every other use of the render seed

# acoustic row fields refreshed from the new render's metadata
ROW_FROM_META = ("mic_gain", "noise_gain", "noise_file", "drr", "ring_tau_s", "eq", "pulse_shape",
                 "tau_mode", "tau_c_median_s", "band_split_hz", "highpass_hz", "pp_gain",
                 "clip_fraction", "rms", "impact_gain_sigma", "impact_gain_norm", "position_weighting")


def parse_axis(text: str) -> tuple[str, dict]:
    """'acoustics.x=uniform:0.6:1.5' → ('acoustics.x', {'dist': 'uniform', 'low': 0.6, 'high': 1.5});
    'acoustics.y=choice:a,b' → ('acoustics.y', {'dist': 'choice', 'values': ['a', 'b']})."""
    key, rhs = text.split("=", 1)
    if not key.startswith("acoustics."):
        raise ValueError(f"only acoustic axes can be re-rendered, got {key}")
    if rhs.startswith("choice:"):
        return key, {"dist": "choice", "values": rhs[len("choice:"):].split(",")}
    dist, lo, hi = rhs.split(":")
    return key, {"dist": dist, "low": float(lo), "high": float(hi)}


def draw_new_axes(axes: dict, render_seed: int) -> dict:
    from gransim.sweep.sampling import _from_unit

    rng = np.random.default_rng([int(render_seed), SALT])
    return {k: _from_unit(float(rng.random()), spec) for k, spec in sorted(axes.items())}


def rerender_run(src: str, run_id: str, render_params: list, axes: dict, fixed: dict,
                 out: str, tag: str) -> dict:
    from gransim.acoustics.assets import write_wav
    from gransim.acoustics.render import load_run, render
    from gransim.config import AcousticConfig, deep_merge
    from gransim.sweep.sampling import nested

    src, out = Path(src), Path(out)
    done = out / "runs" / run_id / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    old = json.loads((src / "runs" / run_id / "done.json").read_text())
    cfg, ev, wf, win = load_run(src / "runs" / run_id)
    rows = []
    for row in old["rows"]:
        k = int(row["render"])
        rp = render_params[k]
        new = draw_new_axes(axes, row["render_seed"])
        ac_data = cfg.acoustics.model_dump()
        for layer in (nested(rp).get("acoustics", {}), fixed, nested(new).get("acoustics", {})):
            ac_data = deep_merge(ac_data, layer)
        ac = AcousticConfig.model_validate(ac_data)
        if abs(ac.mic_gain - row["mic_gain"]) > 1e-9 and "mic_gain" not in fixed \
                and "acoustics.mic_gain" not in axes:
            raise RuntimeError(f"{run_id} k{k}: rebuilt render conditions do not match the "
                               f"dataset (mic_gain {ac.mic_gain} vs {row['mic_gain']})")
        r = render(cfg, ev, wf, win, seed=int(row["render_seed"]), acoustics=ac)
        name = Path(row["file_name"])
        (out / name.parent).mkdir(parents=True, exist_ok=True)
        write_wav(out / name, r.fs, r.audio, bits=ac.bits)
        m = r.meta
        nr = dict(row)
        nr.update({f: m.get(f) for f in ROW_FROM_META})
        nr.update({"direct_attack_s": ac.direct_attack_s, "impact_source": ac.impact_source, "rerender": tag,
                   "rms_original": row.get("rms"), "clip_fraction_original": row.get("clip_fraction")})
        rows.append(nr)
    rec = {"run_id": run_id, "elapsed_s": old.get("elapsed_s"), "rows": rows, "rerender": tag}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rec, default=float))
    return rec


def rerender_dataset(src: str | Path, out: str | Path, axes: dict | None = None,
                     fixed: dict | None = None, workers: int = 4, tag: str = "r2",
                     limit: int | None = None, progress=print) -> dict:
    from gransim.sweep.runner import load_spec, write_manifest
    from gransim.sweep.sampling import build_jobs

    src, out = Path(src), Path(out)
    axes, fixed = dict(axes or {}), dict(fixed or {})
    fixed = {k.removeprefix("acoustics."): v for k, v in fixed.items()}
    spec = load_spec(src / "sweep_spec.yaml")
    jobs = {j.run_id: j for j in build_jobs(spec)}
    out.mkdir(parents=True, exist_ok=True)
    spec_out = dict(spec)
    spec_out["rerender"] = {"source": str(src), "tag": tag, "fixed": fixed, "axes": axes, "salt": SALT}
    spec_out["axes"] = {**spec["axes"], **axes}
    (out / "sweep_spec.yaml").write_text(yaml.safe_dump(spec_out, sort_keys=False, allow_unicode=True))
    run_ids = sorted(p.parent.name for p in (src / "runs").glob("*/done.json"))
    todo = [r for r in run_ids if not (out / "runs" / r / "done.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    progress(f"{len(run_ids)} runs in {src}, {len(todo)} to re-render")
    n_ok = n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rerender_run, str(src), r, jobs[r].render_params, axes, fixed, str(out), tag): r
                for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                fu.result()
                n_ok += 1
                if n_ok % 50 == 0:
                    progress(f"[{n_ok}/{len(todo)}] {time.time() - t0:.0f}s")
            except Exception as e:
                n_fail += 1
                with open(out / "failures.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"run_id": r, "error": repr(e)}) + "\n")
                progress(f"FAILED {r}: {e!r}")
    n_rows = write_manifest(out)
    return {"ok": n_ok, "failed": n_fail, "rows": n_rows}
