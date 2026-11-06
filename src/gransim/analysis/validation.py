"""Run many dynamics cases in parallel and summarise each (phase indicators + event stats).

Used by the stiffness-convergence test and the Γc(N) reproduction (spec 10.1).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from gransim.config import SimConfig
from gransim.events.tracker import IMPACT

PHASE_BINS = 20


def event_statistics(events: np.ndarray, duration: float) -> dict:
    """Acoustically relevant statistics of impact events over a window of ``duration`` s."""
    imp = events[events["contact_type"] == IMPACT]
    J = imp["J_n"]
    out = {
        "impact_rate": imp.size / duration,
        "impact_rate_pw": int((imp["kind"] == 1).sum()) / duration,
        "impact_rate_pp": int((imp["kind"] == 0).sum()) / duration,
        # Σ J² per second: proportional to the energy of the impact-pulse sound component
        "impulse_sq_rate": float((J ** 2).sum() / duration),
        "impulse_sq_rate_pw": float((J[imp["kind"] == 1] ** 2).sum() / duration),
        "wall_impulse_rate": float(events["J_n"][events["kind"] == 1].sum() / duration),
    }
    if J.size:
        q = np.quantile(J, [0.1, 0.5, 0.9])
        out.update(J_q10=float(q[0]), J_q50=float(q[1]), J_q90=float(q[2]))
        # impulse-squared-weighted phase histogram (what the ear would hear per phase)
        h, _ = np.histogram(imp["phase"], bins=PHASE_BINS, range=(0, 1), weights=J ** 2)
        out["phase_hist_J2"] = (h / h.sum()).tolist() if h.sum() > 0 else [0.0] * PHASE_BINS
        h, _ = np.histogram(imp["phase"], bins=PHASE_BINS, range=(0, 1))
        out["phase_hist"] = (h / h.sum()).tolist()
    else:
        out.update(J_q10=0.0, J_q50=0.0, J_q90=0.0,
                   phase_hist_J2=[0.0] * PHASE_BINS, phase_hist=[0.0] * PHASE_BINS)
    return out


def wall_force_statistics(force: np.ndarray, fs: float, hp_cutoff: float = 100.0) -> dict:
    """Kurtosis-type order parameters of the wall force (proxy for the sound, spec 9.1).

    The first study's best acoustic order parameter was the excess kurtosis of the
    waveform (ψ3 = K − 3): large when sound arrives as one sharp burst per cycle
    (solid-like), small when it is spread out (fluid-like). The wall force is the
    sound source, so its kurtosis is the natural pre-acoustics proxy.
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.stats import kurtosis

    fb = force[0] - force[0].mean()
    total = force.sum(axis=0)
    sos = butter(4, hp_cutoff, btype="highpass", fs=fs, output="sos")
    hp = sosfiltfilt(sos, total)
    return {
        "kurtosis_bottom": float(kurtosis(fb)) if fb.std() > 0 else 0.0,
        "kurtosis_total_hp": float(kurtosis(hp)) if hp.std() > 0 else 0.0,
        "force_rms_hp": float(hp.std()),
    }


def run_case(cfg_data: dict) -> dict:
    """Run one configuration (given as a plain dict) and return a flat summary."""
    from gransim.analysis.phase import phase_indicators
    from gransim.dynamics.engine import run_dynamics
    from gransim.events.tracker import collision_interval_check

    cfg = SimConfig.model_validate(cfg_data)
    tic = time.perf_counter()
    res = run_dynamics(cfg)
    t0 = res.wall_force.t0
    T = cfg.time.record_time
    ind = phase_indicators(res.events, res.diagnostics, cfg.particles.count, (t0, t0 + T))
    st = event_statistics(res.events, T)
    st.update(wall_force_statistics(res.wall_force.force, res.wall_force.fs))
    chk = collision_interval_check(res.events, res.event_params["binary_contact_time"])
    return {
        "N": cfg.particles.count,
        "gamma": res.derived["gamma"],
        "frequency": res.derived["frequency_hz"],
        "seed": cfg.seed,
        "model": cfg.contact.model,
        "contact_time": res.derived["contact_time_pp_s"],
        "dt": res.derived["dt_s"],
        "interval_ratio": chk["ratio"],
        "elapsed_s": time.perf_counter() - tic,
        **ind,
        **st,
    }


def case_key(cfg_data: dict) -> str:
    """Stable hash of a config dict (cache file name)."""
    blob = json.dumps(cfg_data, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def run_many(cfgs: Iterable[SimConfig | dict], workers: Optional[int] = None,
             progress: Optional[Callable[[int, int, dict], None]] = None,
             cache_dir: Optional[str | Path] = None,
             time_budget: Optional[float] = None) -> list[Optional[dict]]:
    """Run configs in parallel processes (spawn context); results keep the input order.

    With ``cache_dir`` every finished case is stored as ``<hash>.json`` and reused, so
    an interrupted batch resumes where it stopped. ``time_budget`` [s] stops submitting
    new cases after that long (unfinished entries are returned as None).
    """
    data = [c.model_dump(mode="json") if isinstance(c, SimConfig) else c for c in cfgs]
    if cache_dir is not None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        results: list = [None] * len(data)
        todo = []
        for k, d in enumerate(data):
            f = cache / f"{case_key(d)}.json"
            if f.exists():
                results[k] = json.loads(f.read_text())
            else:
                todo.append(k)

        def _store(k, r):
            (cache / f"{case_key(data[k])}.json").write_text(json.dumps(r))
            results[k] = r
            if progress:
                progress(sum(x is not None for x in results), len(data), r)

        _execute([data[k] for k in todo], workers,
                 lambda i, r: _store(todo[i], r), time_budget)
        return results
    results = [None] * len(data)

    def _set(k, r):
        results[k] = r
        if progress:
            progress(sum(x is not None for x in results), len(data), r)

    _execute(data, workers, _set, time_budget)
    return results


def _execute(data: list[dict], workers: Optional[int], on_result, time_budget) -> None:
    workers = workers or max(1, (os.cpu_count() or 1))
    start = time.perf_counter()
    if workers == 1:
        for k, d in enumerate(data):
            if time_budget is not None and time.perf_counter() - start > time_budget:
                return
            on_result(k, run_case(d))
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        pending = {}
        it = iter(enumerate(data))
        exhausted = False

        def submit():
            nonlocal exhausted
            if exhausted or (time_budget is not None and time.perf_counter() - start > time_budget):
                return False
            try:
                k, d = next(it)
            except StopIteration:
                exhausted = True
                return False
            pending[ex.submit(run_case, d)] = k
            return True

        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in done:
                k = pending.pop(f)
                on_result(k, f.result())
                submit()
