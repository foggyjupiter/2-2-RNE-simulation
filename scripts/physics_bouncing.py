"""흔들리는 상자 속 공 하나의 1차원 모형 — 적은 입자가 왜 가진에 동기화되는가 (학습 없음).

실제 상자(높이 5 cm, BB 6 mm → 공 중심이 움직일 수 있는 높이 H = 4.4 cm)를 크랭크 진폭
A = 1.34 cm, 진동수 f로 위아래로 흔든다. 공은 바닥·천장과만 부딪치고(반발계수 e_w), 사이에서는
자유낙하한다. 바닥 위치는 A sin(ωt)를 정확히 쓴다 (높이 튀기 근사 없음). 상대 속도가 아주
작아지면 (연속 미세 충돌) 공이 바닥에 붙어 함께 움직이다가 바닥 가속도가 −g보다 작아지는
순간 (sin ωt = 1/Γ) 떨어진다 — 완전 비탄성 공의 이륙 조건.

입자끼리 부딪치지 않으므로, 서로 다른 초기 조건에서 출발한 공들을 합친 충돌 위상 분포가
"서로 부딪치지 않는 입자 기체"의 동기화 정도다. 모든 공이 같은 주기 궤도로 끌려가면
(위상 고정) 충돌이 한 위상에 몰리고, 혼돈이면 퍼진다.

    python scripts/physics_bouncing.py   → results/physics/sync/bouncing_1d.json, 그림
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physics" / "sync"
G = 9.81
A = 0.0134            # 크랭크 진폭 [m]
sys.path.insert(0, str(ROOT / "src"))
from gransim.config import load_config  # noqa: E402
_BOX = load_config(ROOT / "configs" / "experiment_replica.yaml").container.size
H = _BOX[2] - 0.006   # 공 중심이 움직일 수 있는 높이 [m] (상자 높이 − BB 지름)
TAU_RING = 0.062      # 실측 울림 감쇠 시간 [s] — 소리 주기성 지표에서 충돌 에너지를 이만큼 번지게 함


@njit(cache=True)
def _sim(e, omega, n_cycles, z0, v0, t0, v_stick, max_imp):
    """공 하나. 반환: 충돌 시각, 벽(0 바닥 / 1 천장), 충격량/질량 |Δv|."""
    T = 2 * math.pi / omega
    gam = A * omega * omega / G
    ts = np.empty(max_imp)
    wall = np.empty(max_imp, np.int64)
    J = np.empty(max_imp)
    k = 0
    t, z, v = t0, z0, v0
    t_end = t0 + n_cycles * T
    dt = T / 2000.0
    stuck = False
    while t < t_end and k < max_imp:
        if stuck:
            # 바닥에 붙어 이동 → sin(ωt) = 1/Γ (cos > 0) 인 다음 순간에 이륙
            if gam <= 1.0:
                break
            ph = math.asin(1.0 / gam)
            n = math.floor((omega * t - ph) / (2 * math.pi)) + 1.0
            tt = (ph + 2 * math.pi * n) / omega
            if math.sin(omega * t) > 1.0 / gam and math.cos(omega * t) > 0:
                tt = t                                   # 이미 이륙 조건
            t = tt
            z = A * math.sin(omega * t)
            v = A * omega * math.cos(omega * t)
            stuck = False
            continue
        # 다음 충돌 찾기: s(τ) = z(τ) − 바닥, 0 또는 H에 닿는 첫 순간
        tau = 0.0
        hit = -1
        prev_tau = 0.0
        while tau < 4 * T:
            tau += dt
            zz = z + v * tau - 0.5 * G * tau * tau
            zf = A * math.sin(omega * (t + tau))
            s = zz - zf
            if s < 0.0:
                hit = 0
                break
            if s > H:
                hit = 1
                break
            prev_tau = tau
        if hit < 0:
            break
        lo, hi = prev_tau, tau
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            zz = z + v * mid - 0.5 * G * mid * mid
            s = zz - A * math.sin(omega * (t + mid))
            if (hit == 0 and s < 0.0) or (hit == 1 and s > H):
                hi = mid
            else:
                lo = mid
        tau = hi
        t = t + tau
        vb = v - G * tau
        u = A * omega * math.cos(omega * t)
        zf = A * math.sin(omega * t)
        vrel = vb - u
        vnew = u - e * vrel
        ts[k] = t
        wall[k] = hit
        J[k] = abs(vnew - vb)
        k += 1
        z = zf if hit == 0 else zf + H
        v = vnew
        if hit == 0 and abs(vnew - u) < v_stick:
            stuck = True
    return ts[:k], wall[:k], J[:k]


def run_balls(e, f, n_balls=24, n_cycles=160, n_skip=60, seed=0, v_stick=2e-3):
    """초기 조건이 다른 공 n_balls개 (서로 충돌 없음). 앞 n_skip 주기는 버림."""
    rng = np.random.default_rng(seed)
    omega = 2 * math.pi * f
    T = 1 / f
    out_t, out_J, out_w = [], [], []
    for b in range(n_balls):
        t0 = rng.uniform(0, T)
        z0 = A * math.sin(omega * t0) + rng.uniform(0.002, H - 0.002)
        v0 = rng.uniform(-0.8, 0.8)
        ts, w, J = _sim(e, omega, n_cycles, z0, v0, t0, v_stick, 200000)
        m = ts > t0 + n_skip * T
        out_t.append(ts[m]); out_J.append(J[m]); out_w.append(w[m])
    return out_t, out_J, out_w, T


def sync_measures(ts_list, J_list, T, nbin=64):
    """R_J: 충격량 가중 위상 집중도 (gransim phase_concentration_impulse와 같은 정의),
    fold: 충돌 에너지(J²)를 울림 시간만큼 번진 뒤 주기로 접은 윤곽의 봉우리/평균 (소리 주기성 대응)."""
    t = np.concatenate(ts_list); J = np.concatenate(J_list)
    if t.size == 0:
        return float("nan"), float("nan")
    ph = np.mod(t / T, 1.0)
    z = np.exp(2j * np.pi * ph)
    R = float(abs((J * z).sum() / J.sum()))
    h = np.bincount(np.minimum((ph * nbin).astype(int), nbin - 1), weights=J ** 2, minlength=nbin)
    x = np.arange(nbin) / nbin
    ker = np.exp(-x * T / TAU_RING)                      # 원형으로 이어 붙인 지수 감쇠
    prof = np.real(np.fft.ifft(np.fft.fft(h) * np.fft.fft(ker)))
    return R, float(prof.max() / prof.mean())


def stability_band(e, m=1):
    """높이 튀기 근사(Holmes 1982)에서 주기 m 궤도가 안정한 Γ 구간 (천장 없음)."""
    a = math.pi * m * (1 - e) / (1 + e)
    return a, math.sqrt(a * a + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f0 = 6.2
    res = {"A_m": A, "H_m": H, "f_hz": f0, "gamma": A * (2 * math.pi * f0) ** 2 / G}
    # 1) 실험 조건(Γ = 2.07)에서 벽 반발계수별 동기화와 분기 그림
    es = np.round(np.arange(0.10, 0.96, 0.025), 3)
    scan, bif = [], []
    for e in es:
        ts, J, w, T = run_balls(float(e), f0)
        R, fold = sync_measures(ts, J, T)
        nimp = np.mean([len(x) for x in ts]) / 100            # 공 하나가 한 주기에 부딪치는 횟수
        ceil = float(np.mean(np.concatenate(w))) if sum(len(x) for x in w) else 0.0
        scan.append({"e_w": float(e), "R_J": R, "fold": fold, "impacts_per_cycle": float(nimp),
                     "ceiling_share": ceil})
        # 분기 그림: 바닥 충돌 위상 (공 8개 × 마지막 40 주기)
        for x, ww, jj in zip(ts[:8], w[:8], J[:8]):
            sel = (ww == 0) & (x > x.max() - 40 * T) & (jj > 0.02)
            bif += [[float(e), float(p)] for p in np.mod(x[sel] / T, 1.0)]
        print(f"e_w {e:.3f}  R_J {R:.2f}  fold {fold:.2f}  충돌/주기 {nimp:.2f}  천장 비율 {ceil:.2f}", flush=True)
    res["scan_e"] = scan
    # 2) Γ (= 진동수, 크랭크 진폭 고정) × e_w 지도
    fs = np.round(np.arange(4.0, 10.01, 0.25), 2)
    es2 = np.round(np.arange(0.1, 0.96, 0.05), 2)
    grid = np.full((len(es2), len(fs)), np.nan)
    gridf = np.full_like(grid, np.nan)
    for i, e in enumerate(es2):
        for j, f in enumerate(fs):
            ts, J, w, T = run_balls(float(e), float(f), n_balls=12, n_cycles=120, n_skip=50)
            grid[i, j], gridf[i, j] = sync_measures(ts, J, T)
    res["map"] = {"e_w": es2.tolist(), "f_hz": fs.tolist(),
                  "gamma": [A * (2 * math.pi * f) ** 2 / G for f in fs],
                  "R_J": grid.tolist(), "fold": gridf.tolist()}
    res["bands_m1"] = {str(e): stability_band(float(e)) for e in es2}
    (OUT / "bouncing_1d.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / "bouncing_1d_bif.npy", np.array(bif))
    print("saved", OUT)


if __name__ == "__main__":
    main()
