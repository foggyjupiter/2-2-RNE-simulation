"""Save / load the event table and the per-wall force time series (npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gransim.dynamics.engine import WallForce
from gransim.dynamics.walls import WALL_NAMES
from gransim.events.tracker import EVENT_DTYPE


_TIME_FIELDS = ("t_start", "t_peak", "t_end")


def save_events(path: str | Path, events: np.ndarray, params: dict, record_window: tuple) -> None:
    """events.npz: one array per field + classification thresholds + record window [s].

    Times stay float64; other float fields are stored as float32 (≈7 significant
    digits, far below model uncertainty) to keep files small.
    """
    def _arr(name):
        a = events[name]
        if a.dtype == np.float64 and name not in _TIME_FIELDS:
            return a.astype(np.float32)
        return a

    np.savez_compressed(
        path, **{name: _arr(name) for name in EVENT_DTYPE.names},
        **{f"param_{k}": np.asarray(v) for k, v in params.items()},
        record_window=np.asarray(record_window, dtype=float))


def load_events(path: str | Path) -> tuple[np.ndarray, dict, tuple]:
    with np.load(path) as z:
        n = z["t_start"].size
        ev = np.zeros(n, EVENT_DTYPE)
        for name in EVENT_DTYPE.names:
            ev[name] = z[name]
        params = {k[len("param_"):]: z[k].item() for k in z.files if k.startswith("param_")}
        win = tuple(z["record_window"].tolist())
    return ev, params, win


def save_wall_force(path: str | Path, wf: WallForce) -> None:
    """wall_force.npz: force [6, n] in N (bin averages), fs [Hz], t0 [s], wall names."""
    np.savez_compressed(path, force=wf.force.astype(np.float32), fs=wf.fs, t0=wf.t0,
                        walls=np.array(WALL_NAMES))


def load_wall_force(path: str | Path) -> WallForce:
    with np.load(path) as z:
        return WallForce(force=z["force"].astype(np.float64), fs=float(z["fs"]), t0=float(z["t0"]))
