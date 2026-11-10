"""1차 연구 물리 분석을 물성이 다른 입자로 확장 (학습 없음, 시뮬레이션 정답 지표 사용).

A. 단일 가진(v3, Γ ≈ 1.9~2.2): 상 지표가 N보다 층 수 F = N d²/A, 그리고 εF (ε = 1 − e²)로
   더 잘 정리되는가 (1차원 비모수 회귀의 교차검증 R²).
B. 다중 가진(v2, 시스템 440개 × Γ 1.35~6.3): 시스템마다 상전이 Γc를 찾고, 무차원 가진 세기
   S = A²ω²/(g d) = Γ A / d 의 임계값 S_c가 1차 연구의 S_c = K (εF)² 를 따르는가.
C. 1차 연구가 상을 판정한 소리 지표(ψ1 주기 접힘 봉우리, ψ2 스펙트럼 엔트로피, ψ3 첨도)가
   시뮬레이션의 실제 상 지표를 얼마나 따라가는가 (v3_nosig 소리, 물성별).

    python scripts/physics_scaling.py   → results/physics/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physics"
D = ROOT / "datasets"
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from gransim.config import load_config  # noqa: E402
_BOX = load_config(ROOT / "configs" / "experiment_replica.yaml").container.size
BOX = _BOX[0] * _BOX[1]                 # 바닥 넓이 A
PHASE = ["gt_phase_concentration", "gt_collectivity", "gt_height_std", "gt_free_flight", "gt_sustained_fraction"]


def cv_r2_1d(x, y, bins=24, folds=5, seed=0):
    """y ≈ g(x), g = 분위 구간 평균 (단조 가정 없음). 5겹 교차검증 R²."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    pred = np.empty(len(x))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        edges = np.unique(np.quantile(x[tr], np.linspace(0, 1, bins + 1)))
        b_tr = np.clip(np.searchsorted(edges, x[tr], side="right") - 1, 0, len(edges) - 2)
        means = np.array([y[tr][b_tr == j].mean() if np.any(b_tr == j) else np.nan for j in range(len(edges) - 1)])
        means = pd.Series(means).interpolate(limit_direction="both").values
        b_te = np.clip(np.searchsorted(edges, x[te], side="right") - 1, 0, len(edges) - 2)
        pred[te] = means[b_te]
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def part_a(lines, res):
    m = pd.read_csv(D / "v3" / "manifest.csv").drop_duplicates("run_id")
    m = m[m.N > 0]
    d = m.particle_diameter_m.values
    F = m.N.values * d ** 2 / BOX
    epp, epw = 1 - m.restitution_pp.values ** 2, 1 - m.restitution_pw.values ** 2
    xs = {"N": m.N.values, "F = N d²/A": F, "ε_pp F": epp * F, "ε_pw F": epw * F,
          "ε̄ F (두 반발의 평균)": 0.5 * (epp + epw) * F}
    lines += ["## A. 상은 개수보다 εF로 정해지는가 (v3, Γ ≈ 1.9~2.2, 물성 전부 무작위, 실행 %d개)" % len(m), "",
              "상 지표마다, 아래 변수 하나로만 설명할 때의 교차검증 R² (1 = 완전히 그 변수로 정해짐).", "",
              "| 상 지표 | " + " | ".join(xs) + " |", "|---|" + "---|" * len(xs)]
    res["A"] = {}
    for p in PHASE:
        y = m[p].values
        ok = np.isfinite(y)
        r = {k: float(cv_r2_1d(np.log(v[ok]), y[ok])) for k, v in xs.items()}
        res["A"][p] = r
        lines.append(f"| {p} | " + " | ".join(f"{r[k]:.2f}" for k in xs) + " |")
    lines.append("")


def part_b(lines, res):
    m = pd.read_csv(D / "v2" / "manifest.csv").drop_duplicates(["system_id", "condition_index"])
    m = m[m.N > 0].sort_values(["system_id", "gamma"])
    rows = []
    thr = 0.45                          # 위상 집중도: 고체형(≈0.77) ↔ 유체형(≈0.15)의 중간
    for sid, g in m.groupby("system_id"):
        G = g.gamma.values; y = g.gt_phase_concentration.values
        if len(G) < 4 or not (y.max() > thr > y.min()):
            continue
        j = np.where((y[:-1] >= thr) & (y[1:] < thr))[0]
        if len(j) == 0:
            continue
        j = j[0]
        lg = np.interp(thr, [y[j + 1], y[j]], [np.log(G[j + 1]), np.log(G[j])])
        Gc = float(np.exp(lg))
        r = g.iloc[0]
        d, A = r.particle_diameter_m, r.amplitude_m
        F = r.N * d ** 2 / BOX
        rows.append({"system": sid, "N": r.N, "d_mm": d * 1000, "e_pp": r.restitution_pp, "e_pw": r.restitution_pw,
                     "F": F, "Gamma_c": Gc, "S_c": Gc * A / d,
                     "epsF_pp": (1 - r.restitution_pp ** 2) * F, "epsF_pw": (1 - r.restitution_pw ** 2) * F})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "critical_per_system.csv", index=False)
    lines += [f"## B. 시스템마다 찾은 상전이 지점 (v2, 위상 집중도가 {thr}을 지나는 Γ, 시스템 {len(t)}개)", "",
              "S = A²ω²/(g d) = Γ A / d (무차원 가진 세기). log S_c = log K + p · log x 맞춤:", "",
              "| x | 기울기 p | R² | 1차 연구식 S_c = K (εF)² 이면 p = 2 |", "|---|---|---|---|"]
    res["B"] = {"n_systems": len(t)}
    for name, col in [("F", "F"), ("ε_pp F", "epsF_pp"), ("ε_pw F", "epsF_pw"), ("N", "N")]:
        x = np.log(t[col].values); yv = np.log(t.S_c.values)
        p, c = np.polyfit(x, yv, 1)
        r2 = 1 - np.sum((yv - (p * x + c)) ** 2) / np.sum((yv - yv.mean()) ** 2)
        res["B"][name] = {"slope": float(p), "r2": float(r2), "K": float(np.exp(c))}
        lines.append(f"| {name} | {p:.2f} | {r2:.2f} | |")
    # Γc 자체 (1차 연구 그림과 같은 축)
    x = np.log(t.N.values); yv = np.log(t.Gamma_c.values)
    p, c = np.polyfit(x, yv, 1)
    lines += ["", f"Γc는 N과 거의 무관 (log Γc vs log N 기울기 {p:.2f}); Γc 중앙값 {t.Gamma_c.median():.2f} "
              f"(사분위 {t.Gamma_c.quantile(.25):.2f}~{t.Gamma_c.quantile(.75):.2f}).", ""]
    res["B"]["Gamma_c_vs_N_slope"] = float(p)
    return t


def part_c(lines, res):
    S = Path("/tmp/claude-0/-home-user/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/bias")
    if not (S / "nosig.npz").exists():
        lines += ["## C. (소리 특징 파일이 없어 건너뜀)", ""]
        return
    import sys
    sys.path.insert(0, "/home/user/rne_machine_learning")
    from rneml.features import SCALAR_NAMES
    X = np.load(S / "nosig.npz")["X"]
    m = pd.read_csv(S / "meta_nosig.csv")
    nm = list(SCALAR_NAMES)
    psi = {"ψ1 주기 접힘 봉우리": nm.index("fold_peak"), "ψ2 스펙트럼 엔트로피": nm.index("entropy"),
           "ψ3 첨도": nm.index("kurtosis")}
    from scipy.stats import spearmanr
    ok = (m.N > 0).values & np.isfinite(X).all(1)
    d = m.particle_diameter_m.values * 1000
    lines += ["## C. 1차 연구의 소리 지표는 실제 상을 따라가는가 (v3_nosig 소리, 클립 %d개)" % ok.sum(), "",
              "스피어만 순위 상관. 괄호 안: 지름 4~5.5 mm / 6.5~8 mm 입자만.", "",
              "| 소리 지표 | 위상 집중도 | 집단성 | 높이 분산 | N |", "|---|---|---|---|---|"]
    res["C"] = {}
    for name, j in psi.items():
        cells = []
        for col in ["gt_phase_concentration", "gt_collectivity", "gt_height_std", "N"]:
            r = spearmanr(X[ok, j], m[col].values[ok])[0]
            sm = ok & (d < 5.5); lg = ok & (d > 6.5)
            rs = spearmanr(X[sm, j], m[col].values[sm])[0]; rl = spearmanr(X[lg, j], m[col].values[lg])[0]
            res["C"].setdefault(name, {})[col] = [float(r), float(rs), float(rl)]
            cells.append(f"{r:+.2f} ({rs:+.2f} / {rl:+.2f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 1차 연구 물리 분석의 물성 일반화 (학습 없음)", ""]
    res = {}
    part_a(lines, res)
    part_b(lines, res)
    part_c(lines, res)
    (OUT / "physics_scaling.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump(res, open(OUT / "physics_scaling.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(lines))
