"""에너지 수지로 본 상(相)의 스케일링 — 왜 εF인가 (학습 없음, 시뮬레이션 정답 지표).

간단한 에너지 수지 (기체처럼 튀는 입자, 평균 속력 v):
  · 바닥 충돌 빈도 ν_w ~ g / v   (높이 ~ v²/g 를 오르내리는 시간의 역수)
  · 입자끼리 충돌 빈도 ν_pp ~ n σ v ~ (N / (A_box · v²/g)) · d² · v = F · g / v
    → ν_pp / ν_w ~ F (층 수 F = N d² / A_box = "한 번 바닥에 닿는 동안 다른 입자와 부딪치는 횟수")
  · 얻는 에너지: 움직이는 바닥에 부딪칠 때 ~ m (Aω)²  (바닥 속도 Aω 규모)
  · 잃는 에너지: 바닥 충돌마다 ~ ε_w · m v², 입자끼리 충돌마다 ~ ε_pp · m v²   (ε = 1 − e²)
  균형: ν_w · m (Aω)² ≈ ν_w · ε_w · m v² + ν_pp · ε_pp · m v²
  ⇒ θ ≡ v² / (Aω)² ≈ 1 / (ε_w + c · ε_pp · F)        (c는 기하 상수)

즉 입자가 얼마나 들뜨는지(→ 상)는 개수 N이 아니라 "비탄성 × 층 수"로 정해지고, 가진 세기는
가속도 Γ가 아니라 속도 Aω로 들어온다. 이를 v3(Γ 거의 고정, 물성 무작위)와 v2(같은 시스템을
5가지 속도로)로 확인한다.

    python scripts/physics_energy.py   → results/physics/energy_balance.{md,json}, 그림
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from physics_scaling import cv_r2_1d  # noqa: E402

OUT = ROOT / "results" / "physics"
D = ROOT / "datasets"
sys.path.insert(0, str(ROOT / "src"))
from gransim.config import load_config  # noqa: E402
_BOX = load_config(ROOT / "configs" / "experiment_replica.yaml").container.size
BOX = _BOX[0] * _BOX[1]                 # 바닥 넓이 A


def load(ds):
    m = pd.read_csv(D / ds / "manifest.csv")
    m = m.drop_duplicates("run_id" if ds == "v3" else ["system_id", "condition_index"])
    m = m[m.N > 0].copy()
    d = m.particle_diameter_m.values
    m["F"] = m.N * d ** 2 / BOX
    m["eps_pp"] = 1 - m.restitution_pp ** 2
    m["eps_pw"] = 1 - m.restitution_pw ** 2
    Aw = m.amplitude_m * 2 * np.pi * m.frequency_hz
    m["Aw"] = Aw
    m["theta"] = 2 * m.gt_ke_rel_per_particle / (m.particle_mass_kg * Aw ** 2)   # 입자 평균 운동에너지 / 바닥 최대 운동에너지
    return m


def fit_theory(m, y, cs=np.round(np.arange(0.0, 6.01, 0.25), 2), ks=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0)):
    """log y = a − b · log(ε_w (1 + k F) + c ε_pp F): k, c를 격자로 골라 R² 최대.
    k = 0 이 가장 단순한 기체 수지, k > 0 은 층이 두꺼울수록 바닥에 더 자주 닿는 효과."""
    best = None
    ly = np.log(y)
    for k in ks:
        for c in cs:
            x = np.log(m.eps_pw * (1 + k * m.F) + c * m.eps_pp * m.F)
            b, a = np.polyfit(x, ly, 1)
            r2 = 1 - np.var(ly - (a + b * x)) / np.var(ly)
            if best is None or r2 > best["r2"]:
                best = {"k": float(k), "c": float(c), "slope": float(b), "r2": float(r2)}
    return best


def loglin_r2(x, y):
    b, a = np.polyfit(x, y, 1)
    return float(b), float(1 - np.var(y - (a + b * x)) / np.var(y))


def main():
    res, L = {}, ["# 에너지 수지로 본 상의 스케일링 (학습 없음)", ""]
    m = load("v3")
    ok = np.isfinite(m.theta) & (m.theta > 0)
    m = m[ok]
    # 1) 들뜸 θ 를 한 변수로 설명: N, F, εF, 이론 변수
    th = fit_theory(m, m.theta.values)
    th0 = fit_theory(m, m.theta.values, ks=(0.0,))
    X = m.eps_pw * (1 + th["k"] * m.F) + th["c"] * m.eps_pp * m.F
    xs = {"N": m.N, "F": m.F, "ε̄F": 0.5 * (m.eps_pp + m.eps_pw) * m.F,
          f"이론 ε_w(1+{th['k']:.0f}F) + {th['c']:.2f}·ε_pp·F": X}
    targets = {"들뜸 θ = v²/(Aω)²": np.log(m.theta.values), "위상 집중도": m.gt_phase_concentration.values,
               "집단성": m.gt_collectivity.values, "높이 분산": m.gt_height_std.values,
               "충돌 빈도/입자": np.log(m.gt_impact_rate_per_particle.values)}
    L += [f"## A. v3 (Γ ≈ 1.9~2.2, 물성 무작위, 실행 {len(m)}개): 한 변수 교차검증 R²", "",
          "거듭제곱 법칙 맞춤 log θ = a − b·log(x) (선형 R²):", "",
          f"- 가장 단순한 기체 수지 x = ε_w + c·ε_pp·F: c = {th0['c']:.2f}, b = {-th0['slope']:.2f} (이론 1), R² = {th0['r2']:.2f}",
          f"- 바닥 충돌이 층 수와 함께 느는 형태 x = ε_w(1 + kF) + c·ε_pp·F: k = {th['k']:.0f}, c = {th['c']:.2f}, b = {-th['slope']:.2f}, R² = {th['r2']:.2f}",
          f"- 1차 연구 변수 x = ε̄F: b = {-loglin_r2(np.log(0.5 * (m.eps_pp + m.eps_pw) * m.F), np.log(m.theta.values))[0]:.2f}, R² = {loglin_r2(np.log(0.5 * (m.eps_pp + m.eps_pw) * m.F), np.log(m.theta.values))[1]:.2f}", "",
          "층 수 구간별 log θ vs log ε̄F 기울기 (R²): " + ", ".join(
              f"F {lo}~{hi}: {loglin_r2(np.log(0.5 * (m.eps_pp + m.eps_pw) * m.F)[(m.F >= lo) & (m.F < hi)], np.log(m.theta.values)[(m.F >= lo) & (m.F < hi)])[0]:.2f} "
              f"({loglin_r2(np.log(0.5 * (m.eps_pp + m.eps_pw) * m.F)[(m.F >= lo) & (m.F < hi)], np.log(m.theta.values)[(m.F >= lo) & (m.F < hi)])[1]:.2f})"
              for lo, hi in [(0, 0.3), (0.3, 1), (1, 3)]), "",
          "비모수 한 변수 교차검증 R² (곡선 모양 자유):", "",
          "| 지표 | " + " | ".join(xs) + " |", "|---|" + "---|" * len(xs)]
    res["A"] = {"fit": th, "fit_simple": th0, "r2": {}}
    for k, y in targets.items():
        r = {n: float(cv_r2_1d(np.log(np.asarray(v, float)), y)) for n, v in xs.items()}
        res["A"]["r2"][k] = r
        L.append(f"| {k} | " + " | ".join(f"{r[n]:.2f}" for n in xs) + " |")
    L.append("")
    # 2) 개수 하나만 늘릴 때 ν_pp/ν_w ~ F 의 흔적: 입자당 충돌 빈도 ∝ (g/v)(1 + kF)
    v = np.sqrt(m.theta) * m.Aw
    rate_norm = m.gt_impact_rate_per_particle * v / 9.81
    kfit = np.polyfit(m.F, rate_norm, 1)
    res["A"]["rate_vs_F"] = {"intercept": float(kfit[1]), "slope": float(kfit[0]),
                             "r2": float(np.corrcoef(m.F, rate_norm)[0, 1] ** 2)}
    L += [f"입자당 충돌 빈도 × v/g = {kfit[1]:.2f} + {kfit[0]:.2f}·F (R² {res['A']['rate_vs_F']['r2']:.2f}) "
          "— 바닥 충돌(상수항)에 층 수에 비례하는 입자끼리 충돌이 더해진다.", ""]
    # 3) v2: 같은 시스템을 다른 속도로 → θ가 속도 Aω에만 묶이는가 (v² ∝ (Aω)^α)
    m2 = load("v2")
    m2 = m2[np.isfinite(m2.theta) & (m2.theta > 0)]
    m2["lv2"] = np.log(2 * m2.gt_ke_rel_per_particle / m2.particle_mass_kg)
    m2["lAw"] = np.log(m2.Aw)
    rows = []
    for name, sub in [("전체", m2), ("Γ ≥ 2 (유체형)", m2[m2.gamma >= 2]), ("Γ < 2", m2[m2.gamma < 2])]:
        dv = sub.lv2 - sub.groupby("system_id").lv2.transform("mean")
        dx = sub.lAw - sub.groupby("system_id").lAw.transform("mean")
        a = float((dv * dx).sum() / (dx * dx).sum())
        r2 = float(1 - np.var(dv - a * dx) / np.var(dv))
        rows.append({"subset": name, "alpha": a, "r2_within": r2, "n": int(len(sub))})
    res["B"] = rows
    L += ["## B. v2 (같은 시스템 × 5가지 속도): 입자 운동에너지 v² ∝ (Aω)^α, 시스템 안에서의 기울기", "",
          "| 범위 | α (이론 2) | 시스템 안 R² | 행 |", "|---|---|---|---|"]
    L += [f"| {r['subset']} | {r['alpha']:.2f} | {r['r2_within']:.2f} | {r['n']} |" for r in rows]
    # Γ 범위별 θ의 이론 변수 맞춤
    for name, sub in [("Γ ≥ 2", m2[m2.gamma >= 2]), ("Γ < 2", m2[m2.gamma < 2])]:
        f2 = fit_theory(sub, sub.theta.values)
        res.setdefault("B_fit", {})[name] = f2
        L.append(f"\nv2 {name}: log θ ~ log(ε_w + c ε_pp F) 맞춤 c = {f2['c']:.1f}, 기울기 {-f2['slope']:.2f}, R² {f2['r2']:.2f}")
    (OUT / "energy_balance.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / "energy_balance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez(OUT / "energy_balance_points.npz", X=X.values, theta=m.theta.values, pc=m.gt_phase_concentration.values,
             d=m.particle_diameter_m.values, N=m.N.values, c=th["c"],
             Aw2=m2.Aw.values, v2=np.exp(m2.lv2.values), sys2=m2.system_id.astype(str).values, gamma2=m2.gamma.values)
    print("\n".join(L))


if __name__ == "__main__":
    main()
