"""Identifiability check (spec 8.1): where do N and the material mix in the sound?

No learning: every clip is described by the acoustic statistics of
:func:`gransim.analysis.sim2real.features` (level, band levels, centroid, onset rate,
ψ2, ψ3, period-folded peakiness), standardised. For each clip the k nearest clips
from *other* dynamics runs are found; if clips that sound alike have very different
N, then N is not identifiable from these statistics in that region.

Reported:
  * neighbour |ΔN| (median over clips) overall and by bins of N × each material axis
  * fraction of clips whose neighbours differ by > ``big_dn`` particles
  * contrast pairs: statistic distance of same-N pairs (material changed) relative to
    the distance between clips whose N differs by ~10 at similar material
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STAT_KEYS = ["rms", "clip_fraction", "centroid_hz", "band_db_500_2000", "band_db_2000_4000",
             "band_db_4000_8000", "band_db_8000_12000", "band_db_12000_16000",
             "band_db_16000_24000", "onset_rate", "kurtosis", "entropy_psi2", "fold_peak_to_mean"]
MATERIAL_KEYS = ["particle_diameter_m", "particle_density", "particle_mass_kg",
                 "restitution_pp", "restitution_pw", "friction_pp", "friction_pw"]


def clip_stats(out: str | Path, rows: list[dict], f_shake: float, workers: int = 4) -> np.ndarray:
    """Feature matrix [n_clips, len(STAT_KEYS)] (cached in out/identifiability_stats.npz)."""
    from concurrent.futures import ProcessPoolExecutor

    out = Path(out)
    cache = out / "identifiability_stats.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["files"].tolist(), z["X"]))
    todo = [r["file_name"] for r in rows if r["file_name"] not in have]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f, v in zip(todo, ex.map(_one, [str(out / f) for f in todo],
                                         [f_shake] * len(todo), chunksize=8)):
                have[f] = v
        np.savez(cache, files=np.array(list(have)), X=np.stack(list(have.values())))
    return np.stack([have[r["file_name"]] for r in rows])


def _one(path: str, f_shake: float) -> np.ndarray:
    from gransim.acoustics.assets import read_wav
    from gransim.analysis.sim2real import features

    fs, x = read_wav(path)
    d = features(x, fs, f_shake)
    return np.array([d[k] for k in STAT_KEYS], float)


def analyse(rows: list[dict], X: np.ndarray, k: int = 5, big_dn: int = 15) -> dict:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    runs = np.array([r["run_id"] for r in rows])
    N = np.array([r["N"] for r in rows], float)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    D[runs[:, None] == runs[None, :]] = np.inf          # never compare a run with itself
    nn = np.argsort(D, axis=1)[:, :k]
    dn = np.abs(N[nn] - N[:, None])                      # [n, k]
    med = np.median(dn, axis=1)
    rep = {"n_clips": len(rows), "n_runs": int(len(set(runs))), "k": k,
           "neighbour_dN_median": float(np.median(med)),
           "neighbour_dN_mean": float(med.mean()),
           f"frac_neighbours_dN_gt_{big_dn}": float(np.mean(med > big_dn))}
    # by N bin × material bin
    nb = np.digitize(N, [20, 40, 60, 80])
    rep["by_N"] = {f"{int(lo)}-{int(hi)}": float(np.median(med[nb == b]))
                   for b, (lo, hi) in enumerate([(0, 19), (20, 39), (40, 59), (60, 79), (80, 100)])
                   if (nb == b).any()}
    rep["by_material"] = {}
    for key in MATERIAL_KEYS:
        v = np.array([r[key] for r in rows], float)
        q = np.quantile(v, [1 / 3, 2 / 3])
        mb = np.digitize(v, q)
        grid = {}
        for b in range(5):
            for m in range(3):
                sel = (nb == b) & (mb == m)
                if sel.sum() >= 4:
                    grid[f"N{b}_m{m}"] = float(np.median(med[sel]))
        rep["by_material"][key] = {"bin_edges": [float(x) for x in q], "median_dN": grid}
    # which material axis best explains the confusion: correlation of |ΔN| with the
    # neighbours' material difference
    corr = {}
    for key in MATERIAL_KEYS:
        v = np.array([r[key] for r in rows], float)
        dv = np.abs(v[nn] - v[:, None]).ravel() / (v.std() + 1e-12)
        corr[key] = float(np.corrcoef(dv, dn.ravel())[0, 1])
    rep["dN_vs_material_difference_corr"] = corr
    # contrast pairs (same N, other material) vs ΔN≈10 at similar material
    pair = {}
    for i, r in enumerate(rows):
        if r.get("pair_type") == "same_N":
            pair.setdefault((r["pair_id"], int(r["render"])), []).append(i)
    d_same_n = [np.sqrt(D[a, b]) for a, b in (v for v in pair.values() if len(v) == 2)
                if np.isfinite(D[a, b])]
    if d_same_n:
        mat = np.stack([[r[k] for k in MATERIAL_KEYS] for r in rows]).astype(float)
        mz = (mat - mat.mean(0)) / (mat.std(0) + 1e-12)
        dm = np.sqrt(((mz[:, None, :] - mz[None, :, :]) ** 2).sum(-1))
        sel = (np.abs(N[:, None] - N[None, :]) >= 8) & (np.abs(N[:, None] - N[None, :]) <= 12) \
            & (dm < 1.0) & np.isfinite(D)
        ref = np.sqrt(D[sel]) if sel.any() else np.array([np.nan])
        pair = {"same_N_pairs": len(d_same_n), "median_dist_same_N_other_material": float(np.median(d_same_n)),
                "median_dist_dN10_similar_material": float(np.nanmedian(ref)),
                "ratio": float(np.median(d_same_n) / np.nanmedian(ref))}
    rep["contrast_pairs"] = pair
    return rep


def run(out: str | Path, f_shake: float = 6.2, k: int = 5, workers: int = 4,
        max_clips: int | None = None) -> dict:
    """Load the sweep manifest, compute statistics, write identifiability.json + figure."""
    out = Path(out)
    rows = [json.loads(line) for line in open(out / "manifest.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["N"] > 0]
    if max_clips:
        rows = rows[:max_clips]
    X = clip_stats(out, rows, f_shake, workers)
    rep = analyse(rows, X, k)
    (out / "identifiability.json").write_text(json.dumps(rep, indent=1))
    _plot(out, rows, X, k)
    return rep


def _plot(out: Path, rows, X, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    runs = np.array([r["run_id"] for r in rows])
    N = np.array([r["N"] for r in rows], float)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    D[runs[:, None] == runs[None, :]] = np.inf
    med = np.median(np.abs(N[np.argsort(D, 1)[:, :k]] - N[:, None]), 1)
    keys = [("particle_diameter_m", 1e3, "diameter [mm]"), ("particle_density", 1.0, "density [kg/m³]"),
            ("restitution_pw", 1.0, "e (particle-wall)"), ("restitution_pp", 1.0, "e (particle-particle)")]
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    for a, (key, sc, lab) in zip(ax[0], keys):
        v = np.array([r[key] for r in rows]) * sc
        s = a.scatter(N, v, c=med, s=10, cmap="viridis", vmin=0, vmax=30)
        a.set_xlabel("N")
        a.set_ylabel(lab)
    fig.colorbar(s, ax=ax[0].tolist(), label=f"median |ΔN| of {k} nearest-sounding clips")
    fig.savefig(out / "identifiability.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
