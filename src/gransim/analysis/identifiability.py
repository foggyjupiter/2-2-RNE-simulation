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
  * predictability (a floor, not a model): linear ridge regression from the statistics,
    5-fold cross-validated by dynamics run — N with the material unknown vs known, and
    each material property from the sound alone
  * variance decomposition: share of each statistic's variance explained by N alone
    and by N + material
  * Cramér-Rao bound on the N error with the material known vs unknown (the
    theoretical cost of not knowing the material, for these statistics)
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


def _transform(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    i = {k: n for n, k in enumerate(STAT_KEYS)}
    X[:, i["rms"]] = np.log(X[:, i["rms"]])
    X[:, i["clip_fraction"]] = np.sqrt(X[:, i["clip_fraction"]])
    X[:, i["onset_rate"]] = np.log1p(X[:, i["onset_rate"]])
    return X


def _ridge_cv(F, y, folds, alphas=(0.01, 0.1, 1, 10, 30, 100)):
    best = None
    for a in alphas:
        p = np.zeros_like(y)
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
            A = (F[tr] - mu) / sd
            b = y[tr].mean()
            w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (y[tr] - b))
            p[te] = ((F[te] - mu) / sd) @ w + b
        e = float(np.abs(p - y).mean())
        if best is None or e < best[0]:
            best = (e, p)
    return best


def predictability(rows: list[dict], X: np.ndarray, seed: int = 0) -> dict:
    """Linear-ridge floor of how well N and the material follow from the statistics."""
    L = _transform(X)
    N = np.array([r["N"] for r in rows], float)
    mat = {"diameter_mm": np.array([r["particle_diameter_m"] for r in rows]) * 1e3,
           "density": np.array([r["particle_density"] for r in rows]),
           "restitution_pp": np.array([r["restitution_pp"] for r in rows]),
           "restitution_pw": np.array([r["restitution_pw"] for r in rows]),
           "friction_pp": np.array([r["friction_pp"] for r in rows]),
           "friction_pw": np.array([r["friction_pw"] for r in rows])}
    M = np.stack([np.log(mat["diameter_mm"]), np.log(mat["density"])] +
                 [mat[k] for k in list(mat)[2:]], 1)
    runs = np.array([r["run_id"] for r in rows])
    ur = np.random.default_rng(seed).permutation(np.unique(runs))
    fold_of = {r: i % 5 for i, r in enumerate(ur)}
    folds = np.array([fold_of[r] for r in runs])
    out = {"note": "linear ridge on 13 summary statistics, 5-fold CV grouped by run; a floor "
                   "for what a learned model can reach, not an estimate of it",
           "N_mean_guess_mae": float(np.abs(N - N.mean()).mean()),
           "N_mae_material_unknown": _ridge_cv(L, N, folds)[0],
           "N_mae_material_known": _ridge_cv(np.hstack([L, M]), N, folds)[0],
           "properties": {}}
    for k, v in mat.items():
        e, p = _ridge_cv(L, v, folds)
        out["properties"][k] = {"mae": e, "mean_guess_mae": float(np.abs(v - v.mean()).mean()),
                                "r2": float(1 - np.mean((p - v) ** 2) / np.var(v))}
    # variance of each statistic explained by N alone vs N + material (in-sample, quadratic in log N)
    ln = np.log1p(N)[:, None]
    PN = np.hstack([ln, ln ** 2, np.ones_like(ln)])
    PNM = np.hstack([PN, M, M * ln])
    dec = {}
    for i, key in enumerate(STAT_KEYS):
        y = L[:, i]
        r = []
        for F in (PN, PNM):
            w = np.linalg.lstsq(F, y, rcond=None)[0]
            r.append(float(1 - np.var(y - F @ w) / np.var(y)))
        dec[key] = {"r2_N": r[0], "r2_N_material": r[1]}
    out["variance_decomposition"] = dec
    return out


def fisher_bound(rows: list[dict], X: np.ndarray, n_bins=((1, 20), (20, 50), (50, 80), (80, 101))) -> dict:
    """Cramér-Rao lower bound on the N error from the summary statistics.

    Model: statistics x = μ(N, material) + ε, ε ~ N(0, Σ). μ is fitted per statistic as a
    quadratic polynomial in (log(1+N), log d, log ρ, e_pp, e_pw, μ_pp, μ_pw) with the N
    interactions; Σ is the residual covariance. Fisher information I = Jᵀ Σ⁻¹ J with J the
    Jacobian of μ at each clip's parameters. Then
        material known:    σ_N ≥ 1 / sqrt(I_NN)
        material unknown:  σ_N ≥ sqrt((I⁻¹)_NN)   (material = nuisance parameters)
    The ratio is the information cost of not knowing the material, for these statistics.
    A bound for this feature set, not for the raw audio; a learned model can do better
    than these features but not better than its own input.
    """
    L = _transform(X)
    N = np.array([r["N"] for r in rows], float)
    keep = N > 0
    L, N = L[keep], N[keep]
    R = [r for r, k in zip(rows, keep) if k]
    th = np.column_stack([np.log1p(N), np.log([r["particle_diameter_m"] for r in R]),
                          np.log([r["particle_density"] for r in R]),
                          [r["restitution_pp"] for r in R], [r["restitution_pw"] for r in R],
                          [r["friction_pp"] for r in R], [r["friction_pw"] for r in R]])
    mu_t, sd_t = th.mean(0), th.std(0)
    Z = (th - mu_t) / sd_t
    P = Z.shape[1]

    def design(Z):
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(P)] + [Z[:, i] ** 2 for i in range(P)] \
            + [Z[:, 0] * Z[:, i] for i in range(1, P)]
        return np.column_stack(cols)

    def jac(Z):                                   # d design / d Z  → [n, n_cols, P]
        n = len(Z)
        J = np.zeros((n, 1 + 2 * P + (P - 1), P))
        for i in range(P):
            J[:, 1 + i, i] = 1.0
            J[:, 1 + P + i, i] = 2 * Z[:, i]
        for k, i in enumerate(range(1, P)):
            J[:, 1 + 2 * P + k, 0] = Z[:, i]
            J[:, 1 + 2 * P + k, i] = Z[:, 0]
        return J

    D = design(Z)
    W, *_ = np.linalg.lstsq(D, L, rcond=None)             # [n_cols, n_stats]
    res = L - D @ W
    Sigma = np.cov(res.T) + 1e-9 * np.eye(L.shape[1])
    Si = np.linalg.inv(Sigma)
    Jd = jac(Z)                                           # [n, cols, P]
    dN_dz0 = (1 + N) * sd_t[0]                            # dN / dz0 (z0 = standardised log(1+N))
    known, unknown = [], []
    for k in range(len(Z)):
        Jx = (Jd[k].T @ W).T                               # [n_stats, P]: d stats / d z
        I = Jx.T @ Si @ Jx
        known.append(dN_dz0[k] / np.sqrt(max(I[0, 0], 1e-12)))
        try:
            unknown.append(dN_dz0[k] * np.sqrt(max(np.linalg.inv(I)[0, 0], 0.0)))
        except np.linalg.LinAlgError:
            unknown.append(np.inf)
    known, unknown = np.array(known), np.array(unknown)
    out = {"note": "Cramér-Rao bound on the N error (particles, 1 sigma) from the 13 summary "
                   "statistics; quadratic mean model, residual covariance",
           "fit_r2_per_stat": {k: float(1 - res[:, i].var() / L[:, i].var()) for i, k in enumerate(STAT_KEYS)},
           "median_sigma_N_material_known": float(np.median(known)),
           "median_sigma_N_material_unknown": float(np.median(unknown)),
           "by_N": {}}
    for lo, hi in n_bins:
        m = (N >= lo) & (N < hi)
        if m.any():
            out["by_N"][f"{lo}-{hi - 1}"] = {"known": float(np.median(known[m])),
                                            "unknown": float(np.median(unknown[m]))}
    return out


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
    rep["predictability"] = predictability(rows, X)
    rep["fisher_bound"] = fisher_bound(rows, X)
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
