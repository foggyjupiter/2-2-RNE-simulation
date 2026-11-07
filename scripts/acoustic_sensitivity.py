"""음향 파라미터 민감도: 각 파라미터를 현실적인 범위의 양 끝으로 바꾸면 소리 특징이
"개수 몇 개만큼" 움직이는가 (실측 녹음 사용 없음, 시뮬레이션만).

같은 동역학 실행(BB, 372 rpm, 벽 반발계수 0.5, N 5~100)의 소리를 기준 설정과, 파라미터 하나만
바꾼 설정으로 다시 합성해 특징 변화 Δf를 재고, 기준 설정에서 N에 따른 특징 기울기 df/dN으로
나눠 개수 등가 |ΔN| = |Δf / (df/dN)|를 구한다. 이 값이 크면 그 파라미터를 모르는 채로 고정하면
개수 추정이 그만큼 틀릴 수 있다는 뜻 (= 무작위화가 꼭 필요).

    python scripts/acoustic_sensitivity.py   → results/sim2real/acoustic_sensitivity.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gransim.acoustics.render import load_run, render  # noqa: E402
from gransim.analysis.sim2real import features  # noqa: E402
from gransim.config import AcousticConfig  # noqa: E402

RUNS = ROOT / "runs" / "sync_probe"
NS = [5, 10, 20, 35, 50, 100]
SEEDS = [0, 1, 2]
BASE = {"impact_gain_sigma": 0.0, "impact_source": "synthetic", "drr": 1.0, "ring_tau": 0.062,
        "highpass_hz": 500.0, "pp_gain": 0.5, "direct_attack_s": 0.0, "mic_gain": 0.92,
        "noise_gain": 1.0, "eq": True, "position_weighting": 0.0}
VARY = {   # 파라미터: (낮은 값, 높은 값) — 현실적으로 가능한 범위의 양 끝
    "ring_tau": (0.03, 0.15),              # 방 잔향 RT60 ≈ 6.9 τ: 0.2 ~ 1.0 s
    "drr": (0.1, 3.0),                     # 직접음/울림 비 −10 ~ +5 dB
    "highpass_hz": (200.0, 1000.0),        # 상자 방사·마이크 저역 차단
    "pp_gain": (0.1, 1.0),                 # 입자끼리 충돌음 크기 (벽 대비)
    "direct_attack_s": (0.0, 1.0e-3),      # 충돌음 상승 시간 (상자 모드 형성·초기 반사)
    "mic_gain": (0.29, 2.9),               # 절대 게인 ±10 dB (클리핑·잡음 대비 포함)
    "noise_gain": (0.32, 3.2),             # 잡음 대비 −10 ~ +10 dB
    "eq": (False, True),                   # 등화기 유무 (≈ 마이크·방 스펙트럼 차이 몇 dB)
    "position_weighting": (0.0, 2.0),      # 벽 위치별 방사 (판 모드 모양)
    "particle_youngs_modulus": (1.5e9, 6.0e10),  # 충돌 시간 (플라스틱 ~ 유리) — 물성, 비교용
}
FEATS = {"log_rms": lambda d: np.log(d["rms"]), "fold_peak": lambda d: d["fold_peak_to_mean"],
         "kurtosis": lambda d: np.log(max(d["kurtosis"], 1e-3)), "onset_rate": lambda d: d["onset_rate"],
         "centroid_khz": lambda d: d["centroid_hz"] / 1e3, "hf_16_24k_db": lambda d: d["band_db_16000_24000"],
         "clip_frac": lambda d: d["clip_fraction"]}


def feats_for(n, over):
    cfg, ev, wf, win = load_run(RUNS / f"pw0.5_N{n}")
    ac = cfg.acoustics.model_dump()
    ac.update(BASE)
    ac.update(over)
    ac = AcousticConfig.model_validate(ac)
    rows = []
    for s in SEEDS:
        r = render(cfg, ev, wf, win, seed=s, acoustics=ac)
        d = features(r.audio, r.fs, 6.2)
        rows.append({k: f(d) for k, f in FEATS.items()})
    return {k: float(np.mean([r[k] for r in rows])) for k in FEATS}


def main():
    base = {n: feats_for(n, {}) for n in NS}
    # df/dN: 이웃 개수 사이 기울기 (로그 N 기준 보간)
    lnN = np.log(NS)
    slope = {}
    for k in FEATS:
        y = np.array([base[n][k] for n in NS])
        g = np.gradient(y, lnN)                      # df/dlnN
        slope[k] = {n: g[i] / n for i, n in enumerate(NS)}   # df/dN
    res = {"base": base, "slope": slope, "vary": {}}
    for p, (lo, hi) in VARY.items():
        out = {}
        for tag, val in (("lo", lo), ("hi", hi)):
            out[tag] = {n: feats_for(n, {p: val}) for n in NS}
        res["vary"][p] = out
        print(p, "done", flush=True)
    # 요약: 각 파라미터를 lo→hi로 바꿀 때 특징별 개수 등가 |ΔN| (N 10~50 중앙값)
    L = ["# 음향 파라미터 민감도 (개수 등가 |ΔN|, 낮은 값 → 높은 값, N 10~50 중앙값)", "",
         "| 파라미터 (낮은 ~ 높은) | " + " | ".join(FEATS) + " | 최대 |", "|---|" + "---|" * (len(FEATS) + 1)]
    summ = {}
    for p, (lo, hi) in VARY.items():
        vals = {}
        for k in FEATS:
            dn = []
            for n in [10, 20, 35, 50]:
                df = res["vary"][p]["hi"][n][k] - res["vary"][p]["lo"][n][k]
                s = slope[k][n]
                dn.append(abs(df / s) if abs(s) > 1e-12 else np.inf)
            vals[k] = float(np.median(dn))
        summ[p] = vals
        L.append(f"| {p} ({lo:g} ~ {hi:g}) | " + " | ".join(f"{min(v, 999):.0f}" for v in vals.values())
                 + f" | {min(max(vals.values()), 999):.0f} |")
    res["summary"] = summ
    out = ROOT / "results" / "sim2real"
    (out / "acoustic_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "acoustic_sensitivity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
