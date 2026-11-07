"""보고서 결과 그림 (results/ml/summary.json → results/figures/).

    python scripts/ml_summary.py > results/ml/summary_table.md   # ML 레포 outputs/에서 수치 모으기
    python scripts/make_result_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "font.size": 9, "axes.unicode_minus": False})
C1, C2 = "#2a78d6", "#eb6834"            # categorical slots 1, 2 (fixed order)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
R = {r["name"]: r for r in json.loads((ROOT / "results" / "ml" / "summary.json").read_text())}


def _axes(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_cause_sigma():
    rows = [("보정 전 소리 (R0)", "R0_v1old_blind"), ("+ 실제 충돌음만 (W2)", "W2_clicks_only_blind"),
            ("+ 충돌별 세기 퍼짐만 (W1)", "W1_sigma_only_blind"), ("+ 둘 다 = 보정한 소리 (R1)", "R1_r2_blind")]
    vals = [R[k]["real"] for _, k in rows]
    fig, ax = plt.subplots(figsize=(6.2, 2.4), facecolor=SURF)
    _axes(ax)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.6, color=C1, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, 32)
    ax.set_xlabel("실측 녹음 개수 오차 (MAE, 개)", color=INK2)
    ax.set_title("같은 동역학(v1), 소리만 바꿔 학습한 블라인드 모델의 실측 오차", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cause_sigma.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_nosig_real():
    rows = [("블라인드", "A1_blind", "V1_nosig_blind"),
            ("크기·무게 앎", "A2_known_size_weight", "V2_nosig_known_size_weight"),
            ("물성 전부 앎", "A3_known_all", "V3_nosig_known_all"),
            ("DANN (실측 라벨 없음)", "C1_dann", "V4_nosig_dann"),
            ("실측 라벨 1개/개수", "G_real_labeled_1", "V11_nosig_real_labeled_1"),
            ("실측 라벨 2개/개수", "G_real_labeled_2", "V6_nosig_real_labeled_2"),
            ("실측 라벨 5개/개수", "G_real_labeled_5", "V7_nosig_real_labeled_5")]
    rows = [r for r in rows if r[1] in R and r[2] in R]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), facecolor=SURF, sharey=True)
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for ax, key, title, xmax in [(axs[0], "test", "시뮬레이션 test", 12), (axs[1], "real", "실측 녹음 (1차 연구)", 30)]:
        _axes(ax)
        a = [R[r[1]][key] for r in rows]
        b = [R[r[2]][key] for r in rows]
        ax.barh(y + h / 2, a, height=h, color=C1, edgecolor=SURF, linewidth=1.5, label="세기 퍼짐 있음 (v3)")
        ax.barh(y - h / 2, b, height=h, color=C2, edgecolor=SURF, linewidth=1.5, label="세기 퍼짐 없음 (v3_nosig)")
        for yi, v in zip(y - h / 2, b):
            ax.text(v + xmax * 0.012, yi, f"{v:.1f}", va="center", color=INK, fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("개수 오차 (MAE, 개)", color=INK2)
    axs[0].set_yticks(y, [r[0] for r in rows], color=INK)
    axs[1].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_nosig_real.png", dpi=200, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_cause_sigma()
    fig_nosig_real()
