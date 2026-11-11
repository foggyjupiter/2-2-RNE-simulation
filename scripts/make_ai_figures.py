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
    """실제 상자(7×7×5) 재실험 결과 (results/ml/box7_compare.md). 괄호 = 예전 5×5×7 상자 실측."""
    rows = [("1차 연구 방식 재현\n(물질 앎, BB만 학습)", 28.1, 19, GRAY, 29.4),
            ("BB만 학습한 본 모델\n(물질 앎)", 9.6, 41, GRAY, 18.5),
            ("블라인드\n(물질 모름)", 9.8, 38, C1, 13.7),
            ("블라인드,\n여러 가진 속도로 학습", 9.6, 40, C1, 11.2),
            ("+ 정답 없는 실측 적응\n(3겹 평균)", 7.8, 41, C3, 11.3),
            ("+ 정답 없는 실측 적응\n+ 실제로 잰 크기·무게", 8.9, 33, C3, 9.8),
            ("+ 실측 정답 개수별 5개\n(3겹 평균)", 6.8, 47, C2, 6.3)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=SURF)
    _ax(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.4, yi, f"{r[1]:.1f}개  (±5개 이내 {r[2]}%)", va="center", color=INK, fontsize=8.2)
    ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=8.5)
    ax.set_xlim(0, 40)
    ax.set_xlabel("실측 녹음(실험 상자 7×7×5)의 개수 오차 (MAE, 개) — 낮을수록 좋음", color=INK2)
    ax.set_title("회색: 기준선 / 파랑: 실측 녹음 미사용 / 초록: 정답 없는 실측 / 주황: 실측 정답 소량",
                 loc="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "ai_improvement.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def fig_cv():
    """적응·정답 학습에 쓴 실측 녹음 묶음별 실측 오차 (실제 상자 재실험)."""
    groups = ["DANN", "DANN + 크기·무게", "실측 정답 5개"]
    folds = [([7.9, 10.3, 6.3], GRAY, "1~5번 사용 (V4·V8·V7)"), ([7.6, 8.0, 5.8], C1, "6~10번 사용 (Q3·Q5·Q1)"),
             ([8.0, 8.4, 8.3], C3, "11~15번 사용 (Q4·Q6·Q2)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=SURF)
    _ax(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    x = np.arange(3)
    w = 0.26
    for k, (v, c, lab) in enumerate(folds):
        ax.bar(x + (k - 1) * w, v, w, color=c, label=lab)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 0.2, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, color=INK)
    ax.set_ylabel("실측 개수 오차 (MAE, 개)", color=INK2)
    ax.set_ylim(0, 13)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ai_crossval.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def fig_range():
    """개수 구간별 실측 오차 (실제 상자 재실험, results/ml/box7_compare.json)."""
    rt = json.loads((ROOT / "results" / "ml" / "box7_compare.json").read_text())
    labels = ["0", "5~25", "30~50", "55~75", "80~100"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=SURF)
    _ax(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for key, c, lab in [("V1_nosig_blind", GRAY, "블라인드 한 속도 (V1)"),
                        ("H9_nosig_set_per_set", C1, "블라인드 여러 속도 (H9)"),
                        ("V4_nosig_dann", C3, "DANN (V4)"),
                        ("V7_nosig_real_labeled_5", C2, "실측 정답 5개 (V7)")]:
        v = [s["mae"] if s else np.nan for s in rt[key]["ranges_new"]]
        ax.plot(labels, v, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_xlabel("실제 개수 구간", color=INK2)
    ax.set_ylabel("실측 개수 오차 (MAE, 개)", color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ai_range.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def _real_preds(e):
    p = ROOT / "results" / "ml" / "box7_raw" / "experiments_box7" / e / "predictions.jsonl"
    return [r for r in map(json.loads, p.read_text().splitlines()) if r.get("split") == "real"]


def fig_vs_physics():
    """물리 기반 방법(물리 파트 표 6·8·9)과 인공지능(실제 상자 재실험, 교차검증 평균)의 실측 비교."""
    R4 = [(5, 25), (30, 50), (55, 75), (80, 100)]
    NS = [5, 10, 30, 50, 70, 90, 100]
    groups = {"blind": ["V1_nosig_blind"],
              "dann": ["V4_nosig_dann", "Q3_cv_dann_f1", "Q4_cv_dann_f2"],
              "lab": ["V7_nosig_real_labeled_5", "Q1_cv_real_labeled_5_f1", "Q2_cv_real_labeled_5_f2"]}
    rng, mean = {}, {}
    for g, es in groups.items():
        a, b = [], []
        for e in es:
            P = _real_preds(e)
            a.append([np.mean([abs(r["pred"] - r["N"]) for r in P if lo <= r["N"] <= hi]) for lo, hi in R4])
            b.append([np.mean([r["pred"] for r in P if r["N"] == n]) for n in NS])
        rng[g], mean[g] = np.mean(a, 0), np.mean(b, 0)
    phys_known = [2.2, 6.1, 7.3, 7.6]        # 물리 파트 표 6 (물질 앎, 4개 개수로 보정, 5초)
    phys_blind = [4.7, 6.6, 20.4, 35.9]      # 물리 파트 표 8 (물질 모름)
    phys_mean = [4, 10, 31, 45, 41, 49, 65]  # 물리 파트 표 9 (물질 모름, 개수별 평균 추정)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1), facecolor=SURF, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a1, a2):
        _ax(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    lab = ["5~25", "30~50", "55~75", "80~100"]
    a1.plot(lab, phys_blind, color=INK, lw=2, marker="s", ms=4, label="물리 기반, 물질 모름")
    a1.plot(lab, phys_known, color=INK, lw=1.5, ls="--", marker="s", ms=3.5, mfc=SURF, label="물리 기반, 물질 앎 (4개 개수 보정)")
    a1.plot(lab, rng["blind"], color=C1, lw=2, marker="o", ms=4, label="인공지능 블라인드")
    a1.plot(lab, rng["dann"], color=C3, lw=2, marker="o", ms=4, label="인공지능 + 정답 없는 실측 적응")
    a1.plot(lab, rng["lab"], color=C2, lw=1.5, ls="--", marker="o", ms=3.5, mfc=SURF, label="인공지능 + 실측 정답 5개")
    a1.set_xlabel("실제 개수 구간", color=INK2)
    a1.set_ylabel("실측 MAE (개)", color=INK2)
    a1.set_title("(a) 개수 구간별 오차", fontsize=9, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=6.6, labelcolor=INK, loc="upper left")
    a2.plot([0, 100], [0, 100], color=GRID, lw=1.2, ls="-")
    a2.text(66, 80, "추정 = 실제", fontsize=7, color=INK2, rotation=38)
    a2.plot(NS, phys_mean, color=INK, lw=2, marker="s", ms=4, label="물리 기반, 물질 모름")
    a2.plot(NS, mean["blind"], color=C1, lw=2, marker="o", ms=4, label="인공지능 블라인드")
    a2.plot(NS, mean["dann"], color=C3, lw=2, marker="o", ms=4, label="인공지능 + 정답 없는 실측 적응")
    a2.set_xlim(0, 105); a2.set_ylim(0, 105)
    a2.set_xlabel("실제 개수 N", color=INK2)
    a2.set_ylabel("평균 추정 개수", color=INK2)
    a2.set_title("(b) 물질을 모를 때 개수별 평균 추정값", fontsize=9, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=6.6, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ai_vs_physics.png", dpi=220, facecolor=SURF)
    plt.close(fig)
    print("vs physics", {k: np.round(v, 1).tolist() for k, v in rng.items()}, {k: np.round(v).tolist() for k, v in mean.items()})


def fig_probe():
    """개발 단계 모델(V4) 내부 표현에서 선형 회귀로 복원한 물리량의 R² (results/ml/interpret/V4_nosig_dann.json, 시뮬레이션 시험 클립)."""
    r2 = json.loads((ROOT / "results" / "ml" / "interpret" / "V4_nosig_dann.json").read_text())["probe_r2"]
    learned = [("개수 N", "N"), ("질량", "mass"), ("지름", "diameter"), ("벽 반발계수", "e_pw"), ("탄성률", "E"), ("마찰계수", "mu_pp")]
    phase = [("가진 위상 집중도", "gt_phase_concentration"), ("평균 높이", "gt_height_mean"), ("자유 비행 비율", "gt_free_flight"),
             ("집단성", "gt_collectivity"), ("입자당 충돌 빈도", "gt_impact_rate_per_particle"), ("지속 접촉 비율", "gt_sustained_fraction")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.35), facecolor=SURF, sharex=True)
    for ax, items, color, title in ((axes[0], learned, C1, "학습한 목표"), (axes[1], phase, C2, "운동 상태 지표 (학습하지 않음)")):
        _ax(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        labels = [a for a, _ in items][::-1]
        vals = [max(0.0, r2[k]) for _, k in items][::-1]
        y = np.arange(len(items))
        ax.barh(y, vals, color=color, height=0.62)
        for yi, v, lab in zip(y, vals, labels):
            ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=7.6, color=INK,
                    fontweight="bold" if lab == "가진 위상 집중도" else "normal")
        ax.set_yticks(y, labels, fontsize=8, color=INK)
        for t in ax.get_yticklabels():
            if t.get_text() == "가진 위상 집중도":
                t.set_fontweight("bold")
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=8.6, color=INK, loc="left")
        ax.set_xlabel("내부 표현에서 복원한 R²", color=INK2, fontsize=8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "ai_probe.png", dpi=220, facecolor=SURF)
    plt.close(fig)


def fig_phase_map():
    """실제 상자(7×7×5) 시뮬레이션 1,188회의 상: (a) 층 수-비탄성 평면, (b) N, (c) εF에 대한 가진 위상 집중도.
    datasets/box7pack_v3_nosig/manifest.csv (시뮬레이션 정답 지표)."""
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    m = pd.read_csv(ROOT / "datasets" / "box7pack_v3_nosig" / "manifest.csv").drop_duplicates("run_id")
    m = m[m.N > 0]
    d = m.particle_diameter_m.values
    F = m.N.values * d ** 2 / (0.07 * 0.07)
    eps = 0.5 * ((1 - m.restitution_pp.values ** 2) + (1 - m.restitution_pw.values ** 2))
    y = m.gt_phase_concentration.values

    def r2_binned(x, nb=25):
        edges = np.quantile(x, np.linspace(0, 1, nb + 1))
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, nb - 1)
        mu = np.array([y[idx == k].mean() for k in range(nb)])
        return 1 - ((y - mu[idx]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rN, rE = r2_binned(m.N.values.astype(float)), r2_binned(np.log(eps * F))
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#dbe8f8", "#2a78d6", "#0d3a73"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), facecolor=SURF)
    o = np.argsort(y)
    kw = dict(c=y[o], cmap=cmap, lw=0, vmin=0.1, vmax=0.6)
    for ax in axs:
        _ax(ax)
    sc = axs[0].scatter(F[o], eps[o], s=9, **kw)
    xx = np.logspace(-2.5, 0.2, 50)
    for c in (0.02, 0.05, 0.1, 0.2):
        axs[0].plot(xx, c / xx, color=INK2, lw=0.7, ls=":")
    axs[0].set_xlabel("층 수 F = N d²/A  (점선: εF 일정)", color=INK2)
    axs[0].set_ylabel("비탄성 ε", color=INK2)
    axs[0].set_title("(a) 상 지도 (색: 위상 집중도)", loc="left", fontsize=9, color=INK)
    axs[1].scatter(m.N.values[o], y[o], s=7, **kw)
    axs[1].set_xlabel("개수 N", color=INK2)
    axs[1].set_ylabel("가진 위상 집중도", color=INK2)
    axs[1].set_title(f"(b) 개수로 보면 (R² {rN:.2f})", loc="left", fontsize=9, color=INK)
    axs[2].scatter((eps * F)[o], y[o], s=7, **kw)
    axs[2].set_xlabel("εF", color=INK2)
    axs[2].set_title(f"(c) εF로 보면 (R² {rE:.2f})", loc="left", fontsize=9, color=INK)
    for ax in (axs[0], axs[2]):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylim(0.08, 0.72)
    axs[0].set_xlim(0.003, 1.6)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label("가진 위상 집중도", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "ai_phase_map.png", dpi=220, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"phase map: runs {len(m)}, R2 N {rN:.2f}, epsF {rE:.2f}")


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
    fig_vs_physics()
    fig_probe()
    fig_phase_map()
    fig_flow()
    fig_model()
    print("saved")
