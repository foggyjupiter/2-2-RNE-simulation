"""한글 보고서(인공지능 부분)용 그림: 결과 그래프 3장(results/ml/*, 학습 레포 outputs/ 수치)과 도식 2장."""
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


def _box(ax, x, y, w, h, text, fc, ec, fs=8.5, bold=False, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, p, q, color=INK2, ls="-", rad=0.0, lw=1.2):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))


def fig_flow():
    """그림 1: 인공지능 파트 전체 흐름."""
    fig, ax = plt.subplots(figsize=(7.4, 3.3), facecolor=SURF)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.45); ax.axis("off")
    L1, L2, L3 = "#e8f1fc", "#fdeee6", "#e6f6ef"
    w, h, y = 1.78, 1.5, 2.05
    xs = [0.05, 2.08, 4.11, 6.14, 8.17]
    texts = ["물리 시뮬레이터\n동역학(DEM) 계산\n→ 충돌음 합성\n(물리 파트 참고)",
             "학습 데이터\n물성·녹음 조건\n무작위화\n약 5.4만 클립",
             "입력\n로그-멜\n스펙트로그램\n+ 물리 특징 220개",
             "CNN\n개수·물성\n추정값 ± 불확도",
             "평가\n시뮬레이션 test\n처음 보는 물질\n실측 녹음 210개"]
    for i, (x, t) in enumerate(zip(xs, texts)):
        _box(ax, x, y, w, h, t, L3 if i == 4 else L1, C3 if i == 4 else C1, fs=8.2)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    # 소리만 다시 합성 경로 (평가 → 시뮬레이터, 위쪽으로 돌아감)
    top = y + h + 0.42
    xr, xl = 8.17 + w / 2, 0.05 + w / 2
    ax.plot([xr, xr, xl], [y + h + 0.03, top, top], color=C2, lw=1.2, ls="--")
    _arrow(ax, (xl, top), (xl, y + h), color=C2, ls="--")
    ax.text((xl + xr) / 2, top + 0.08, "원인 진단: 동역학은 그대로 두고 소리만 다시 합성 (몇 시간 안에 데이터 수정)",
            fontsize=7.8, color=C2, ha="center", va="bottom")
    # 아래 줄: 실측 녹음 → (1~5번) 판별기 → CNN, (6~15번) → 평가
    _box(ax, 5.45, 0.15, 1.9, 1.25, "판별기\n시뮬레이션 / 실측\n(DANN, 실측 적응)", L2, C2, fs=8)
    _arrow(ax, (6.9, 1.4), (6.9, y), color=C2)
    _box(ax, 8.17, 0.15, w, 1.25, "1차 연구 실측 녹음\n(BB, 개수별 15개)", "#f4f3f0", GRAY, fs=8)
    _arrow(ax, (8.17, 0.62), (7.35, 0.62), color=GRAY)
    ax.text(7.76, 0.7, "1~5번", fontsize=7.5, color=INK2, ha="center", va="bottom")
    _arrow(ax, (8.17 + w / 2, 1.4), (8.17 + w / 2, y), color=GRAY)
    ax.text(8.17 + w / 2 + 0.08, 1.72, "6~15번\n(평가 전용)", fontsize=7.5, color=INK2, ha="left", va="center")
    fig.savefig(OUT / "ai_flow.png", dpi=220, facecolor=SURF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_model():
    """그림 2: CountNet(왼쪽)과 다중 조건 SetCountNet(오른쪽) 구조."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8), facecolor=SURF)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    L1, L2, L3 = "#e8f1fc", "#fdeee6", "#e6f6ef"
    ax.text(2.35, 5.15, "기본 모델 (CountNet)", ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    _box(ax, 0.1, 3.75, 2.15, 0.8, "로그-멜 스펙트로그램\n(48 kHz, 128 대역)", L1, C1, fs=7.8)
    _box(ax, 2.5, 3.75, 2.15, 0.8, "물리 특징 220개\n(음량·주기성·첨도 등)", L1, C1, fs=7.8)
    _box(ax, 0.1, 2.6, 2.15, 0.8, "합성곱 4단\n→ 시간 방향 요약", L1, C1, fs=7.8)
    _box(ax, 1.25, 1.45, 2.3, 0.8, "합치기 → 공유 층\n(내부 표현 128)", L1, C1, fs=7.8)
    _box(ax, 0.1, 0.1, 2.15, 1.0, "출력\n개수 N, 물성 8가지\n(추정값 ± 불확도)", L3, C3, fs=7.8)
    _box(ax, 2.5, 0.1, 2.15, 1.0, "판별기\n시뮬레이션 / 실측\n(실측 적응 때만)", L2, C2, fs=7.8, ls="--")
    _arrow(ax, (1.175, 3.75), (1.175, 3.4))
    _arrow(ax, (1.175, 2.6), (1.9, 2.25))
    _arrow(ax, (3.575, 3.75), (2.9, 2.25))
    _arrow(ax, (1.9, 1.45), (1.175, 1.1))
    _arrow(ax, (2.9, 1.45), (3.575, 1.1), color=C2, ls="--")
    ax.plot([5.0, 5.0], [0.1, 5.0], color=GRID, lw=1)
    ax.text(7.55, 5.15, "다중 조건 모델 (SetCountNet)", ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    for x, t in [(5.3, "녹음 1\n300 rpm"), (6.55, "녹음 2\n372 rpm"), (8.6, "녹음 K\n650 rpm")]:
        _box(ax, x, 3.85, 1.1, 0.7, t, L1, C1, fs=7.8)
        _arrow(ax, (x + 0.55, 3.85), (x + 0.55, 3.4))
    ax.text(8.12, 4.2, "…", ha="center", va="center", fontsize=12, color=INK2)
    _box(ax, 5.3, 2.6, 4.4, 0.8, "각 녹음을 같은 CountNet으로 처리\n+ 녹음마다 가진 세기 Γ 표시", L1, C1, fs=7.8)
    _box(ax, 5.3, 1.45, 4.4, 0.8, "묶음 공통 음량 빼기 (마이크 게인·질량 지우기)\n녹음 사이 비교 + 이웃 조건 사이 변화", L1, C1, fs=7.8)
    _box(ax, 5.3, 0.1, 2.3, 1.0, "출력\n개수 N, 물성\n(추정값 ± 불확도)", L3, C3, fs=7.8)
    _arrow(ax, (7.5, 2.6), (7.5, 2.25))
    _arrow(ax, (6.45, 1.45), (6.45, 1.1))
    ax.text(7.85, 0.6, "실측처럼 녹음이 1개뿐이면\n'1개짜리 묶음'으로 넣는다", ha="left", va="center", fontsize=7.4, color=INK2)
    fig.savefig(OUT / "ai_model.png", dpi=220, facecolor=SURF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    fig_improvement()
    fig_cv()
    fig_range()
    fig_flow()
    fig_model()
    print("saved")
