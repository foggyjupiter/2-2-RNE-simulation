"""Report figures from the saved results (results/figures/*.png).

    python scripts/make_figures.py --recordings <first-study data-all> [--runs runs/s2r] [--dataset datasets/v1]

  fig_sim2real_features.png  real (mean ± sd over 15 clips) vs simulation per N, key features
  fig_spectrograms.png       real vs simulated mel spectrograms, same N, 1.5 s excerpts
  fig_identifiability.png    Cramér-Rao bound on the N error, material known vs unknown
  fig_dataset_params.png     distributions of N and the material properties in the dataset
  fig_shake_speed.png        shaking frequency estimated from each real recording vs N
  fig_restitution_probe.png  periodicity and onset rate vs N: real vs restitution settings
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
# categorical slots (fixed order): real, current simulation, best-fitting simulation
C_REAL, C_SIM, C_FIT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"


def sim2real_features():
    p = ROOT / "results" / "sim2real" / "comparison.json"
    if not p.exists():
        return
    rep = json.loads(p.read_text())
    table = rep["table"]
    Ns = sorted(int(n) for n in table)
    feats = [("rms", "RMS"), ("clip_fraction", "clipping fraction"), ("kurtosis", "kurtosis ψ3"),
             ("centroid_hz", "spectral centroid [Hz]"), ("onset_rate", "detected onsets [1/s]"),
             ("fold_peak_to_mean", "period-folded peakiness")]
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    for a, (k, lab) in zip(ax.ravel(), feats):
        rows = [{r["feature"]: r for r in table[str(n)]}[k] for n in Ns]
        a.errorbar(Ns, [r["real_mean"] for r in rows], yerr=[r["real_std"] for r in rows],
                   fmt="o-", ms=3, label="real (15 clips)")
        a.plot(Ns, [r["sim_mean"] for r in rows], "s--", ms=3, label="simulation")
        a.set_xlabel("N")
        a.set_title(lab)
    ax[0, 0].legend()
    fig.suptitle(f"Simulation vs first-study recordings (372 rpm), mic gain {rep['mic_gain']:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sim2real_features.png", dpi=130)
    plt.close(fig)


def spectrograms(recordings: str, runs: str):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from gransim.acoustics.assets import read_wav
    from gransim.acoustics.render import load_run, render
    from gransim.config import AcousticConfig
    from scipy import signal

    Ns = [5, 20, 100]
    fig, ax = plt.subplots(2, len(Ns), figsize=(12, 5), sharey=True)
    for j, N in enumerate(Ns):
        f_real = sorted(glob.glob(f"{recordings}/{N}-*.wav"))
        rd = Path(runs) / f"N{N}_s0"
        if not f_real or not rd.exists():
            continue
        fs, xr = read_wav(f_real[0])
        cfg, ev, wf, win = load_run(rd)
        ac = AcousticConfig.model_validate({**cfg.acoustics.model_dump(), "drr": 6.0, "mic_gain": 0.92})
        xs = render(cfg, ev, wf, win, seed=0, acoustics=ac).audio
        for i, (x, lab) in enumerate(((xr, "real"), (xs, "simulation"))):
            seg = x[int(1.0 * fs):int(2.5 * fs)]
            f, t, Z = signal.stft(seg, fs, nperseg=1024, noverlap=1024 - 240)
            S = 20 * np.log10(np.abs(Z) + 1e-6)
            ax[i, j].pcolormesh(t, f / 1000, S, vmin=S.max() - 70, vmax=S.max(), shading="auto", cmap="magma")
            ax[i, j].set_title(f"{lab}, N = {N}")
            ax[i, j].set_xlabel("time [s]")
            if j == 0:
                ax[i, j].set_ylabel("kHz")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectrograms.png", dpi=130)
    plt.close(fig)


def identifiability():
    p = ROOT / "results" / "identifiability" / "identifiability.json"
    if not p.exists():
        return
    fb = json.loads(p.read_text()).get("fisher_bound")
    if not fb:
        return
    bins = list(fb["by_N"])
    x = np.arange(len(bins))
    fig, a = plt.subplots(figsize=(5.5, 3.6))
    a.bar(x - 0.2, [fb["by_N"][b]["known"] for b in bins], 0.4, label="material known")
    a.bar(x + 0.2, [fb["by_N"][b]["unknown"] for b in bins], 0.4, label="material unknown")
    a.set_xticks(x)
    a.set_xticklabels(bins)
    a.set_xlabel("N")
    a.set_ylabel("Cramér-Rao bound on N error (1σ)")
    a.set_title("Information in 13 summary statistics (single condition)")
    a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_identifiability.png", dpi=130)
    plt.close(fig)


def dataset_params(dataset: str):
    p = Path(dataset) / "manifest.jsonl"
    if not p.exists():
        return
    rows = [json.loads(l) for l in open(p)]
    rows = [r for r in rows if r["render"] == 0]
    cols = [("N", 1, "N"), ("particle_diameter_m", 1e3, "diameter [mm]"),
            ("particle_density", 1, "density [kg/m³]"), ("restitution_pp", 1, "e particle-particle"),
            ("restitution_pw", 1, "e particle-wall"), ("friction_pw", 1, "μ particle-wall")]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5))
    for a, (k, sc, lab) in zip(ax.ravel(), cols):
        v = np.array([r[k] for r in rows], float) * sc
        sp = np.array([r["split"] for r in rows])
        a.hist([v[sp == s] for s in ("train", "val", "test", "test_ood", "pairs")], bins=25, stacked=True,
               label=["train", "val", "test", "test_ood", "pairs"])
        a.set_title(lab)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"Dataset v1: {len(rows)} dynamics runs")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_params.png", dpi=130)
    plt.close(fig)


def shake_speed(recordings: str):
    """Real shaker speed per clip (from the sound): drifts 6.0-6.7 Hz at a nominal 6.2 Hz."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from gransim.acoustics.assets import read_wav
    from gransim.analysis.order_params import estimate_shake_frequency

    Ns, fs_ = [], []
    for N in range(5, 101, 5):
        for f in sorted(glob.glob(f"{recordings}/{N}-*.wav")):
            fs, x = read_wav(f)
            Ns.append(N)
            fs_.append(estimate_shake_frequency(x, fs, 6.2))
    if not Ns:
        return
    Ns, fs_ = np.array(Ns), np.array(fs_)
    fig, a = plt.subplots(figsize=(6, 3.4))
    jit = np.random.default_rng(0).uniform(-1.2, 1.2, len(Ns))
    a.plot(Ns + jit, fs_, "o", ms=4, color=C_REAL, alpha=0.45, mec="none", label="each recording")
    u = np.unique(Ns)
    a.plot(u, [fs_[Ns == n].mean() for n in u], "-", lw=2, color=C_REAL, label="mean per N")
    a.axhline(6.2, color=INK2, lw=1, ls="--")
    a.text(117, 6.205, "nominal 6.2 Hz", va="bottom", ha="right", color=INK2, fontsize=8)
    a.set_xlabel("N (particles)")
    a.set_ylabel("shaking frequency from the sound [Hz]")
    a.set_title("First-study recordings: the shaker speed drifts between clips", color=INK)
    a.set_xlim(0, 118)
    a.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shake_speed.png", dpi=130)
    plt.close(fig)


def restitution_probe():
    """Periodicity (per-clip folded peakiness) and onset rate vs N: recordings vs e = 0.8
    (data default) vs e_pp 0.95 / e_pw 0.7 (best fit). From results/sim2real/restitution_probe/."""
    d = ROOT / "results" / "sim2real" / "restitution_probe"
    try:
        base = next(r for r in json.loads((d / "egrid.json").read_text())
                    if r["set"] == "e0.8" and r["sigma"] == 0.0)
        fit = next(r for r in json.loads((d / "egrid_split.json").read_text())
                   if r["set"] == "pp0.95 pw0.7" and r["sigma"] == 0.0)
    except (FileNotFoundError, StopIteration):
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, key, lab, log in ((ax[0], "fold_peak_to_mean", "periodicity (folded peak / mean)", False),
                             (ax[1], "onset_rate", "detected impact onsets [1/s]", True)):
        series = [("recordings", base, "real", C_REAL, "o"),
                  ("simulation e = 0.8", base, "sim", C_SIM, "s"),
                  ("simulation e_pp 0.95, e_pw 0.7", fit, "sim", C_FIT, "^")]
        for name, rep, which, col, mk in series:
            Ns = sorted(int(n) for n in rep["table"])
            v = [rep["table"][str(n)][key][which] for n in Ns]
            a.plot(Ns, v, "-", marker=mk, lw=2, ms=6, color=col, label=name, mec="white", mew=1)
            a.annotate(name.replace("simulation ", "sim "), (Ns[-1], v[-1]), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        a.set_xlabel("N (particles)")
        a.set_title(lab, color=INK)
        a.set_xlim(0, 135)
        if log:
            a.set_yscale("log")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Restitution vs the first-study recordings (372 rpm; per-clip shaker frequency)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_restitution_probe.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", help="first-study data-all folder")
    ap.add_argument("--runs", default=str(ROOT / "runs" / "s2r"))
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "v1"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim2real_features()
    identifiability()
    dataset_params(a.dataset)
    restitution_probe()
    if a.recordings:
        spectrograms(a.recordings, a.runs)
        shake_speed(a.recordings)
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
