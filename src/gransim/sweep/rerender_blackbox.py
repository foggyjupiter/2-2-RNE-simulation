"""Re-render a downloaded sweep with the black-box acoustics (:mod:`gransim.acoustics.blackbox`).

Only the saved dynamics are used (runs/<id>/config_snapshot.yaml, events.npz,
wall_force.npz, done.json); no DEM is rerun. Every clip keeps its file name, labels and
split, and gets a new acoustic set-up drawn from its render seed (so a rerun gives
identical files). Output layout is that of a sweep, so the manifests are written by
:func:`gransim.sweep.runner.write_manifest` and the ML side reads it unchanged::

    python scripts/rerender_blackbox.py data/src/v3 data/bbx/v3 --workers 4
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# columns of the old rows that described the old acoustics; replaced or dropped
OLD_ACOUSTIC = ("mic_gain", "noise_gain", "noise_file", "noise_start", "noise_phase_aligned",
                "noise_rpm_scaled", "drr", "ring_tau_s", "eq", "impact_gain_sigma",
                "impact_gain_norm", "impact_source", "position_weighting", "direct_attack_s",
                "pp_gain", "clip_fraction", "rms", "tau_c_median_s", "rerender",
                "rms_original", "clip_fraction_original")


def rerender_run(src: str, run_id: str, out: str) -> dict:
    from gransim.acoustics.assets import write_wav
    from gransim.acoustics.blackbox import render_bb
    from gransim.acoustics.render import load_run

    src, out = Path(src), Path(out)
    done = out / "runs" / run_id / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    old = json.loads((src / "runs" / run_id / "done.json").read_text())
    cfg, ev, wf, win = load_run(src / "runs" / run_id)
    rows = []
    for row in old["rows"]:
        y, m = render_bb(cfg, ev, wf, win, int(row["render_seed"]),
                         duration_s=float(row.get("duration_s", 5.0)))
        name = Path(row["file_name"])
        (out / name.parent).mkdir(parents=True, exist_ok=True)
        write_wav(out / name, 48000, y, bits=16)
        nr = {k: v for k, v in row.items() if k not in OLD_ACOUSTIC}
        nr.update({"mic_gain": m["bb_mic_gain"], "noise_gain": m["bb_noise_gain"],
                   "noise_file": m["noise_file"], "drr": m["bb_drr"],
                   "pp_gain": m["bb_pp_gain"], "position_weighting": m["bb_position_weighting"],
                   "impact_gain_sigma": 0.0, "impact_source": "synthetic", "eq": False,
                   "clip_fraction": m["clip_fraction"], "rms": m["rms"],
                   "tau_c_median_s": m["tau_c_median_s"],
                   "particle_E_acoustic_pa": m["particle_E_acoustic_pa"]})
        nr.update({k: v for k, v in m.items() if k.startswith("bb_") or k == "renderer"})
        rows.append(nr)
    rec = {"run_id": run_id, "elapsed_s": old.get("elapsed_s"), "rows": rows, "rerender": "blackbox"}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rec, default=float))
    return rec


def rerender_dataset(src: str | Path, out: str | Path, workers: int = 4,
                     limit: int | None = None, progress=print) -> dict:
    from gransim.sweep.runner import write_manifest

    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_ids = sorted(p.parent.name for p in (src / "runs").glob("*/done.json")
                     if (p.parent / "events.npz").exists() and (p.parent / "wall_force.npz").exists()
                     and (p.parent / "config_snapshot.yaml").exists())
    todo = [r for r in run_ids if not (out / "runs" / r / "done.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    progress(f"{len(run_ids)} runs in {src}, {len(todo)} to re-render")
    n_ok = n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rerender_run, str(src), r, str(out)): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                fu.result()
                n_ok += 1
                if n_ok % 50 == 0:
                    progress(f"[{n_ok}/{len(todo)}] {time.time() - t0:.0f}s")
            except Exception as e:  # keep going; failures are listed
                n_fail += 1
                with open(out / "failures.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"run_id": r, "error": repr(e)}) + "\n")
                progress(f"FAILED {r}: {e!r}")
    n_rows = write_manifest(out)
    return {"ok": n_ok, "failed": n_fail, "rows": n_rows}
