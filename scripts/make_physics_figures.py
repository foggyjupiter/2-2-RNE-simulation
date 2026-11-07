"""물리 분석 그림 (results/physics/ → results/figures/).

    python scripts/physics_bouncing.py; python scripts/physics_sync_grid.py
    python scripts/physics_energy.py;   python scripts/physics_resolution.py
    python scripts/make_physics_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "physics"
OUT = ROOT / "results" / "figures"
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "font.size": 9, "axes.unicode_minus": False})
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#9cc0ee", "#2a78d6", "#0d3a73"])
RANGE_JSON = ROOT / "results" / "ml" / "range_table.json"


def _axes(ax, grid="both"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    if grid in ("both", "y"):
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    if grid in ("both", "x"):
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _logx(ax, subs=(1.0, 2.0, 5.0)):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_sync():
    b = json.loads((P / "sync" / "bouncing_1d.json").read_text())
    g = json.loads((P / "sync" / "grid.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.5), facecolor=SURF, gridspec_kw={"width_ratios": [1.1, 1.0, 1.1]})
    # (가) 입자 하나: 1차원 이론 vs 3차원 DEM
    ax = axs[0]; _axes(ax)
    e = np.array([r["e_w"] for r in b["scan_e"]])
    ax.axvspan(0.55, 0.92, color=GRID, alpha=0.6, lw=0)
    ax.text(0.735, 0.03, "학습 데이터의\n벽 반발계수 범위", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.plot(e, [r["R_J"] for r in b["scan_e"]], color=C1, lw=1.8, label="동기화 R (1차원 이론)")
    ax.plot(e, [r["ceiling_share"] for r in b["scan_e"]], color=C2, lw=1.4, ls="--", label="천장 충돌 비율 (1차원 이론)")
    ed = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.scatter(ed, [g["events"][f"{x}_1"]["R_J"] for x in ed], color=C1, s=22, zorder=3, edgecolor=SURF, lw=0.8,
               label="동기화 R (3차원 DEM, 입자 1개)")
    ax.scatter(ed, [g["events"][f"{x}_1"]["ceiling_share"] for x in ed], color=C2, s=22, marker="s", zorder=3,
               edgecolor=SURF, lw=0.8, label="천장 비율 (3차원 DEM)")
    ax.set_xlabel("벽 반발계수 e_w", color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_title("(가) 입자 하나 (Γ = 2.07)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK)
    # (나) Γ × e_w 지도
    ax = axs[1]
    mp = b["map"]
    Rm = np.array(mp["R_J"], float)
    G = np.array(mp["gamma"]); E = np.array(mp["e_w"])
    im = ax.pcolormesh(G, E, Rm, cmap=BLUES, vmin=0, vmax=1, shading="nearest")
    ax.axvline(2.07, color=C2, lw=1.4)
    ax.text(2.12, 0.93, "실험 Γ", color=C2, fontsize=8, va="top")
    ax.axhspan(0.4, 0.6, xmin=0, xmax=1, color="none", ec=INK, lw=0.0)
    ax.plot([2.07], [0.5], marker="o", color=INK, ms=5)
    ax.text(2.2, 0.47, "실측 BB 추정", color=INK, fontsize=7.5)
    ax.set_xlabel("가진 세기 Γ (진폭 1.34 cm 고정)", color=INK2)
    ax.set_ylabel("벽 반발계수 e_w", color=INK2)
    ax.set_title("(나) 동기화 지도 (1차원 이론)", loc="left", color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("동기화 R", color=INK2); cb.outline.set_visible(False)
    # (다) 소리 주기성: 실측 vs DEM
    ax = axs[2]; _axes(ax)
    Ns = [5, 10, 20, 35, 50, 100]
    real = [g["sound"][f"0.5_{n}"]["real"] for n in Ns]
    rs = [g["sound"][f"0.5_{n}"]["real_std"] for n in Ns]
    for x, c, lab in [(0.3, C4, "e_w 0.3"), (0.5, C3, "e_w 0.5"), (0.7, C2, "e_w 0.7 (처음 보정값)"), (0.9, C1, "e_w 0.9")]:
        ax.plot(Ns, [g["sound"][f"{x}_{n}"]["sim"] for n in Ns], color=c, lw=1.6, marker="o", ms=3, label=f"DEM {lab}")
    ax.errorbar(Ns, real, yerr=rs, color=INK, lw=0, elinewidth=1, capsize=2, marker="D", ms=4, label="실측 녹음 (±1σ)")
    _logx(ax)
    ax.set_xlabel("개수 N", color=INK2)
    ax.set_ylabel("소리 주기성 (봉우리/평균)", color=INK2)
    ax.set_title("(다) 소리의 주기성: 실측 vs DEM", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sync_wall.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_energy():
    z = np.load(P / "energy_balance_points.npz", allow_pickle=True)
    res = json.loads((P / "energy_balance.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), facecolor=SURF)
    ax = axs[0]; _axes(ax)
    sc = ax.scatter(z["X"], z["theta"], c=z["d"] * 1000, cmap=BLUES, s=6, lw=0, alpha=0.9)
    fit = res["A"]["fit"]
    xx = np.geomspace(z["X"].min(), z["X"].max(), 50)
    a = np.mean(np.log(z["theta"]) - fit["slope"] * np.log(z["X"]))
    ax.plot(xx, np.exp(a + fit["slope"] * np.log(xx)), color=C2, lw=1.6,
            label=f"θ ∝ x^-{-fit['slope']:.2f} (R² {fit['r2']:.2f})")
    _logx(ax, subs=(1.0, 3.0)); ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"손실 변수 x = ε_w(1 + {fit['k']:.0f}F) + {fit['c']:.2f}·ε_pp·F", color=INK2)
    ax.set_ylabel("들뜸 θ = v² / (Aω)²", color=INK2)
    ax.set_title("(가) 에너지 수지: 들뜸은 손실 × 층 수로 정해진다", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02); cb.set_label("지름 (mm)", color=INK2); cb.outline.set_visible(False)
    ax = axs[1]; _axes(ax)
    sysid = z["sys2"]; Aw2 = z["Aw2"] ** 2; v2 = z["v2"]; gam = z["gamma2"]
    rng = np.random.default_rng(1)
    for s in rng.choice(np.unique(sysid), 40, replace=False):
        k = np.argsort(Aw2[sysid == s])
        ax.plot(Aw2[sysid == s][k], v2[sysid == s][k], color=C1, lw=0.7, alpha=0.45)
        ax.scatter(Aw2[sysid == s][k], v2[sysid == s][k], c=np.where(gam[sysid == s][k] >= 2, C1, C2), s=6, lw=0)
    xx = np.geomspace(0.15, 0.9, 10)
    hi = gam >= 2
    k0 = np.exp(np.median(np.log(v2[hi]) - res["B"][1]["alpha"] / 2 * np.log(Aw2[hi])))
    ax.plot(xx, k0 * xx ** (res["B"][1]["alpha"] / 2), color=INK, lw=1.4, ls="--",
            label=f"Γ ≥ 2: v² ∝ (Aω)^{res['B'][1]['alpha']:.2f} (이론 2)")
    _logx(ax, subs=(1.0, 2.0, 5.0)); ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("바닥 최대 속도² (Aω)² [m²/s²]", color=INK2)
    ax.set_ylabel("입자 평균 속도² v² [m²/s²]", color=INK2)
    ax.set_title("(나) 같은 입자를 5가지 속도로 (선 하나 = 시스템 하나)", loc="left", color=INK, fontsize=10)
    ax.scatter([], [], c=C2, s=10, label="Γ < 2 (고체형: 급격히 변함)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_energy_balance.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_resolution():
    r = json.loads((P / "resolution.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.4), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]; _axes(ax)
    N = np.array(r["N"]); mu = np.array(r["features_mean"]["log_rms"]); sd = np.array(r["features_std"]["log_rms"])
    m = N > 0
    ax.axvspan(1, 21, color=GRID, alpha=0.6, lw=0)
    ax.text(4.5, 0.97, "입자 하나씩\n(F < 0.3)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=INK2, fontsize=7.5)
    ax.errorbar(N[m], 2 * mu[m], yerr=2 * sd[m], color=INK, lw=0, elinewidth=1, marker="o", ms=3, capsize=1.5,
                label="실측 녹음 (±1σ)")
    pr = r["loudness_exponent_by_range"]
    for (a, b_), c in [((5, 25), C3), ((30, 100), C2)]:
        s = (N >= a) & (N <= b_)
        k, c0 = np.polyfit(np.log(N[s]), 2 * mu[s], 1)
        xx = np.linspace(a, b_, 20)
        ax.plot(xx, c0 + k * np.log(xx), color=c, lw=2, label=f"N {a}~{b_}: 소리 에너지 ∝ N^{k:.2f}")
    _logx(ax)
    ax.set_xlim(4, 110)
    ax.set_xlabel("개수 N", color=INK2)
    ax.set_ylabel("log 소리 에너지 (log RMS²)", color=INK2)
    ax.set_title("(가) 음량은 개수보다 느리게 커진다", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    ax = axs[1]; _axes(ax)
    rows = r["rows"]
    Nr = [x["N"] for x in rows]
    ax.axvspan(0, 21, color=GRID, alpha=0.6, lw=0)
    ax.plot(Nr, [x["dN_rms"] for x in rows], color=C2, lw=1.6, label="실측 한계: 음량만")
    ax.plot(Nr, [x["dN_all"] for x in rows], color=INK, lw=1.8, label="실측 한계: 물리 특징 7개")
    if RANGE_JSON.exists():
        rt = json.loads(RANGE_JSON.read_text())
        ranges = [(5, 25), (30, 50), (55, 75), (80, 100)]
        for key, c, lab, mk in [("V1_nosig_blind", C1, "블라인드 모델 (V1) 실측 MAE", "o"),
                                ("V7_nosig_real_labeled_5", C3, "실측 정답 5개 모델 (V7) 실측 MAE", "s")]:
            vals = [rt[key]["real"][i + 1]["mae"] for i in range(4)]
            for (a, b_), v in zip(ranges, vals):
                ax.plot([a, b_], [v, v], color=c, lw=2.2, solid_capstyle="butt")
            ax.plot([], [], color=c, lw=2.2, label=lab)
        v = [rt["V1_nosig_blind"]["test"][i + 1]["mae"] for i in range(4)]
        for (a, b_), vv in zip(ranges, v):
            ax.plot([a, b_], [vv, vv], color=C1, lw=1.2, ls=":")
        ax.plot([], [], color=C1, lw=1.2, ls=":", label="블라인드 모델 시뮬레이션 MAE")
    ax.set_xlim(0, 102); ax.set_ylim(0, 22)
    ax.set_xlabel("개수 N", color=INK2)
    ax.set_ylabel("가를 수 있는 개수 차이 δN / 오차 (개)", color=INK2)
    ax.set_title("(나) 개수 분해능의 한계와 모델 오차", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=INK, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_resolution.png", dpi=200, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    fig_sync()
    fig_energy()
    fig_resolution()
    print("saved", OUT)
