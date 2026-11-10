"""상자 치수를 바꾼 뒤 음향 보정을 다시 한다: 1/3옥타브 등화기와 마이크 게인 (학습 없음).

예전 보정(마이크 게인 0.92, assets/impulse_response/eq.npz)은 5×5×7 cm 상자 시뮬레이션으로
맞춘 값이다. 상자가 바닥 7×7 cm·높이 5 cm로 바뀌면 충돌 빈도·세기가 달라지므로 다시 맞춘다.
방법은 처음 보정과 같다 (gransim.analysis.sim2real.run_sim2real):
  1) 등화기를 끄고 합성 → 실측(N ≤ 20)과 평균 스펙트럼 비를 1/3옥타브로 다듬어 새 등화기
  2) 새 등화기를 켜고 다시 합성 → 모든 N에 하나인 마이크 게인을 로그 RMS 최소제곱으로 재보정
입력 실행: runs/sync_probe/<이름>_N{5..100} (BB 6 mm, 372 rpm, physics_sync_grid가 고른 벽 반발계수)

    python scripts/recalibrate_box.py --e 0.5
    → assets/impulse_response/eq.npz (이전 것은 eq_box5x5x7.npz로 보관)
      results/sim2real/recalibration_box7.{json,md}
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gransim.acoustics.transfer import EQ_FILE  # noqa: E402
from gransim.analysis.sim2real import run_sim2real  # noqa: E402

RUNS = ROOT / "runs" / "sync_probe"
OUT = ROOT / "results" / "sim2real"
REC = "/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all"
NS_REAL = [5, 10, 20, 35, 50, 100]
BASE_AC = {"impact_gain_sigma": 0.0, "impact_source": "synthetic"}   # 처음 보정과 같은 조건


def name(e, n):
    return f"{'base' if e == 0.7 else f'pw{e}'}_N{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=float, required=True, help="입자-벽 반발계수 (격자에서 고른 값)")
    ap.add_argument("--n-render", type=int, default=3)
    a = ap.parse_args()
    dirs = {n: [str(RUNS / name(a.e, n))] for n in NS_REAL if (RUNS / name(a.e, n) / "summary.json").exists()}
    if len(dirs) < len(NS_REAL):
        raise SystemExit(f"실행이 모자람: {sorted(dirs)}")
    old = {"mic_gain": 0.92}
    with np.load(EQ_FILE) as z:
        old_eq = {"f": z["f"], "gain_db": z["gain_db"]}
    # 1) 등화기 없이 → 새 등화기
    r0 = run_sim2real(REC, dirs, 6.2, n_render=a.n_render, acoustics_overrides={**BASE_AC, "eq": False})
    backup = EQ_FILE.with_name("eq_box5x5x7.npz")
    if not backup.exists():
        shutil.copyfile(EQ_FILE, backup)
    np.savez(EQ_FILE, f=r0["eq"]["f"], gain_db=r0["eq"]["gain_db"])
    # 2) 새 등화기로 → 마이크 게인
    r1 = run_sim2real(REC, dirs, 6.2, n_render=a.n_render, acoustics_overrides=BASE_AC)
    bands = [250, 500, 1000, 2000, 4000, 8000, 16000]
    eqtab = {fc: (float(np.interp(fc, old_eq["f"], old_eq["gain_db"])),
                  float(np.interp(fc, r0["eq"]["f"], r0["eq"]["gain_db"]))) for fc in bands}
    res = {"e_pw": a.e, "mic_gain": r1["mic_gain"], "log_rms_rmse": r1["log_rms_rmse"],
           "mic_gain_without_eq": r0["mic_gain"], "old": old, "eq_db_old_new": eqtab,
           "table": {str(n): r1["table"][n] for n in r1["table"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recalibration_box7.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["# 음향 재보정 (상자 바닥 7×7 cm, 높이 5 cm)", "",
         f"- 동역학: BB 6 mm·0.12 g, 372 rpm, 입자끼리 e 0.95, 입자-벽 e {a.e}, 요철 0.5°",
         f"- 마이크 게인: {r1['mic_gain']:.3f} (예전 5×5×7 상자 0.92), 로그 RMS 오차 {r1['log_rms_rmse']:.3f}", "",
         "| 대역 (Hz) | 예전 등화기 (dB) | 새 등화기 (dB) |", "|---|---|---|"]
    L += [f"| {fc} | {o:+.1f} | {n:+.1f} |" for fc, (o, n) in eqtab.items()]
    L += ["", "## 특징 비교 (새 보정, 실측 평균 vs 시뮬레이션 평균)", "",
          "| N | 특징 | 실측 | 시뮬레이션 | 통과 |", "|---|---|---|---|---|"]
    for n in r1["table"]:
        for row in r1["table"][n]:
            L.append(f"| {n} | {row['feature']} | {row['real_mean']:.3g} | {row['sim_mean']:.3g} | "
                     f"{'예' if row.get('pass') else '아니오'} |")
    (OUT / "recalibration_box7.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
