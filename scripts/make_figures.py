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
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False})
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
    feats = [("rms", "RMS 음량"), ("clip_fraction", "클리핑 비율"), ("kurtosis", "첨도 ψ3"),
             ("centroid_hz", "스펙트럼 무게중심 [Hz]"), ("onset_rate", "검출된 충돌 수 [1/s]"),
             ("fold_peak_to_mean", "주기 접힘 봉우리")]
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    for a, (k, lab) in zip(ax.ravel(), feats):
        rows = [{r["feature"]: r for r in table[str(n)]}[k] for n in Ns]
        a.errorbar(Ns, [r["real_mean"] for r in rows], yerr=[r["real_std"] for r in rows],
                   fmt="o-", ms=3, label="실측 (클립 15개)")
        a.plot(Ns, [r["sim_mean"] for r in rows], "s--", ms=3, label="시뮬레이션")
        a.set_xlabel("개수 N")
        a.set_title(lab)
    ax[0, 0].legend()
    fig.suptitle(f"시뮬레이션과 1차 연구 녹음 비교 (372 rpm, 마이크 게인 {rep['mic_gain']:.2f})")
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
        for i, (x, lab) in enumerate(((xr, "실측"), (xs, "시뮬레이션"))):
            seg = x[int(1.0 * fs):int(2.5 * fs)]
            f, t, Z = signal.stft(seg, fs, nperseg=1024, noverlap=1024 - 240)
            S = 20 * np.log10(np.abs(Z) + 1e-6)
            ax[i, j].pcolormesh(t, f / 1000, S, vmin=S.max() - 70, vmax=S.max(), shading="auto", cmap="magma")
            ax[i, j].set_title(f"{lab}, N = {N}")
            ax[i, j].set_xlabel("시간 [s]")
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
    a.bar(x - 0.2, [fb["by_N"][b]["known"] for b in bins], 0.4, label="물질 앎")
    a.bar(x + 0.2, [fb["by_N"][b]["unknown"] for b in bins], 0.4, label="물질 모름")
    a.set_xticks(x)
    a.set_xticklabels(bins)
    a.set_xlabel("개수 N")
    a.set_ylabel("N 오차의 크라메르-라오 하한 (1σ)")
    a.set_title("요약 통계 13개에 담긴 개수 정보 (단일 조건)")
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
    cols = [("N", 1, "개수 N"), ("particle_diameter_m", 1e3, "지름 [mm]"),
            ("particle_density", 1, "밀도 [kg/m³]"), ("restitution_pp", 1, "반발계수 입자-입자"),
            ("restitution_pw", 1, "반발계수 입자-벽"), ("friction_pw", 1, "마찰계수 입자-벽")]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5))
    for a, (k, sc, lab) in zip(ax.ravel(), cols):
        v = np.array([r[k] for r in rows], float) * sc
        sp = np.array([r["split"] for r in rows])
        a.hist([v[sp == s] for s in ("train", "val", "test", "test_ood", "pairs")], bins=25, stacked=True,
               label=["학습", "검증", "시험", "처음 보는 물질", "대조 쌍"])
        a.set_title(lab)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"데이터셋 {Path(dataset).name}: 동역학 실행 {len(rows)}개")
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
    a.plot(Ns + jit, fs_, "o", ms=4, color=C_REAL, alpha=0.45, mec="none", label="녹음 하나하나")
    u = np.unique(Ns)
    a.plot(u, [fs_[Ns == n].mean() for n in u], "-", lw=2, color=C_REAL, label="개수별 평균")
    a.axhline(6.2, color=INK2, lw=1, ls="--")
    a.text(117, 6.205, "공칭 6.2 Hz", va="bottom", ha="right", color=INK2, fontsize=8)
    a.set_xlabel("개수 N")
    a.set_ylabel("소리에서 추정한 가진 주파수 [Hz]")
    a.set_title("1차 연구 녹음: 녹음마다 가진 속도가 다르다", color=INK)
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
    for a, key, lab, log in ((ax[0], "fold_peak_to_mean", "주기성 (접힌 봉우리 / 평균)", False),
                             (ax[1], "onset_rate", "검출된 충돌 수 [1/s]", True)):
        series = [("실측 녹음", base, "real", C_REAL, "o"),
                  ("시뮬레이션 e = 0.8", base, "sim", C_SIM, "s"),
                  ("시뮬레이션 e_pp 0.95, e_pw 0.7", fit, "sim", C_FIT, "^")]
        for name, rep, which, col, mk in series:
            Ns = sorted(int(n) for n in rep["table"])
            v = [rep["table"][str(n)][key][which] for n in Ns]
            a.plot(Ns, v, "-", marker=mk, lw=2, ms=6, color=col, label=name, mec="white", mew=1)
            a.annotate(name.replace("시뮬레이션 ", "시뮬 "), (Ns[-1], v[-1]), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        a.set_xlabel("개수 N")
        a.set_title(lab, color=INK)
        a.set_xlim(0, 135)
        if log:
            a.set_yscale("log")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("반발계수 설정별 1차 연구 녹음과의 비교 (372 rpm, 클립별 가진 주파수)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_restitution_probe.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", help="first-study data-all folder")
    ap.add_argument("--runs", default=str(ROOT / "runs" / "s2r"))
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "v3"))
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
