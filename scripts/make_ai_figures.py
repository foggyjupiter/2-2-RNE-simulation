"""한글 보고서(인공지능 부분)용 그림 3장 (results/ml/*, 학습 레포 outputs/ 수치)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "font.size": 9, "axes.unicode_minus": False})
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_axisbelow(True)


def fig_improvement():
    rows = [("1차 연구 방식 재현\n(BB만 학습, A4)", 29.4, 14, GRAY),
            ("첫 블라인드 CNN\n(보정 음향 v3, A1)", 25.4, 15, GRAY),
            ("세기 퍼짐 제거 데이터\n(V1)", 13.7, 19, C1),
            ("V1 + 여러 가진 속도 집합 모델\n(H9)", 11.2, 35, C1),
            ("V1 + 정답 없는 실측 적응 DANN\n(V4·Q3·Q4 평균)", 11.3, 26, C3),
            ("V1 + DANN + 실제로 잰 크기·무게\n(V8·Q5·Q6 평균)", 9.8, 30, C3),
            ("V1 + 개수별 실측 정답 5개\n(V7·Q1·Q2 평균)", 6.3, 51, C2)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=SURF)
    _ax(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.4, yi, f"{r[1]:.1f}개  (±5개 이내 {r[2]}%)", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=8.5)
    ax.set_xlim(0, 38)
    ax.set_xlabel("실측 녹음 개수 오차 (MAE, 개) — 낮을수록 좋음", color=INK2)
    ax.text(0.99, 0.02, "회색: 기준선 / 파랑: 실측 녹음 미사용 / 초록: 정답 없는 실측 / 주황: 실측 정답 소량",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "ai_improvement.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def fig_cv():
    groups = ["DANN", "DANN + 크기·무게", "실측 정답 5개"]
    f0 = [9.7, 7.4, 6.2]           # 11~15번 평가, 1~5번으로 적응·학습
    f1 = [12.5, 11.1, 6.1]         # 11~15번 평가, 6~10번으로 적응·학습
    f0b = [10.3, 7.3, 6.2]         # 6~10번 평가, 1~5번
    f2 = [11.4, 11.0, 6.5]         # 6~10번 평가, 11~15번
    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=SURF)
    _ax(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    x = np.arange(3)
    w = 0.2
    for k, (v, c, lab) in enumerate([(f0, GRAY, "1~5번 적응 → 11~15번 평가"), (f1, C1, "6~10번 적응 → 11~15번 평가"),
                                     (f0b, "#c9c7c1", "1~5번 적응 → 6~10번 평가"), (f2, C3, "11~15번 적응 → 6~10번 평가")]):
        b = ax.bar(x + (k - 1.5) * w, v, w, color=c, label=lab)
        for xi, vi in zip(x + (k - 1.5) * w, v):
            ax.text(xi, vi + 0.2, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, color=INK)
    ax.set_ylabel("실측 개수 오차 (MAE, 개)", color=INK2)
    ax.set_ylim(0, 15)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ai_crossval.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def fig_range():
    rt = json.loads((ROOT / "results" / "ml" / "range_table.json").read_text())
    labels = ["0", "5~25", "30~50", "55~75", "80~100"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=SURF)
    _ax(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for key, c, lab, ls in [("V1_nosig_blind", GRAY, "블라인드 한 속도 (V1)", "-"),
                            ("H9_nosig_set_per_set", C1, "블라인드 여러 속도 (H9)", "-"),
                            ("V8_nosig_dann_known_size_weight", C3, "DANN + 크기·무게 (V8)", "-"),
                            ("V7_nosig_real_labeled_5", C2, "실측 정답 5개 (V7)", "-")]:
        v = [s["mae"] if s else np.nan for s in rt[key]["real"]]
        ax.plot(labels, v, color=c, lw=2, marker="o", ms=4, ls=ls, label=lab)
    v = [s["mae"] if s else np.nan for s in rt["V1_nosig_blind"]["test"]]
    ax.plot(labels, v, color=INK2, lw=1.2, ls=":", marker="s", ms=3, label="V1 시뮬레이션 test")
    ax.set_xlabel("실제 개수 구간", color=INK2)
    ax.set_ylabel("개수 오차 (MAE, 개)", color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ai_range.png", dpi=220, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    fig_improvement()
    fig_cv()
    fig_range()
    print("saved")
