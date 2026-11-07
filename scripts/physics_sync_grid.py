"""3차원 DEM에서 벽 반발계수 × 개수에 따른 동기화, 그리고 실측 주기성과의 비교 (학습 없음).

실행: results/physics/sync/launch_grid.sh (BB 6 mm, 372 rpm, 입자끼리 e 0.95, 요철 0.5°,
벽 반발계수 e_w 0.3~0.9 × N 1~100). 충돌 기록에서
  R_J      충격량 가중 위상 집중도 (모든 충돌)
  천장 비율  천장(윗벽) 충돌이 충격량에서 차지하는 비율
  입자끼리 비율 입자끼리 충돌의 충격량 비율
을 재고, 실측 녹음이 있는 N(5~100)은 소리를 합성해 실측 주기성(fold_peak_to_mean)과 비교한다.

    python scripts/physics_sync_grid.py   → results/physics/sync/grid.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gransim.analysis.sim2real import run_sim2real  # noqa: E402

RUNS = ROOT / "runs" / "sync_probe"
OUT = ROOT / "results" / "physics" / "sync"
REC = "/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all"
ES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NS = [1, 2, 5, 10, 20, 35, 50, 100]
NS_REAL = [5, 10, 20, 35, 50, 100]


def name(e, n):
    return f"{'base' if e == 0.7 else f'pw{e}'}_N{n}"


def event_measures(run):
    z = np.load(run / "events.npz")
    t0, t1 = z["record_window"]
    m = (z["t_start"] >= t0) & (z["t_start"] <= t1) & (z["contact_type"] == 0)
    J = np.abs(z["J_n"][m]); ph = z["phase"][m]; w = z["wall_id"][m]
    if J.sum() == 0:
        return {}
    zz = np.exp(2j * np.pi * ph)
    return {"R_J": float(abs((J * zz).sum() / J.sum())),
            "R_J_bottom": float(abs((J * zz)[w == 0].sum() / max(J[w == 0].sum(), 1e-30))),
            "ceiling_share": float(J[w == 1].sum() / J.sum()),
            "pp_share": float(J[w == -1].sum() / J.sum()),
            "impacts_per_particle_per_cycle": float(m.sum() / (t1 - t0) / 6.2 / max(int(run.name.split("_N")[1]), 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"events": {}, "sound": {}}
    for e in ES:
        for n in NS:
            r = RUNS / name(e, n)
            if (r / "events.npz").exists():
                res["events"][f"{e}_{n}"] = event_measures(r)
    # 소리 주기성: 실측과 같은 방식 (세기 퍼짐 0, 합성 클릭 — sync_probe 비교와 같은 조건)
    for e in ES:
        dirs = {n: [str(RUNS / name(e, n))] for n in NS_REAL if (RUNS / name(e, n) / "summary.json").exists()}
        rep = run_sim2real(REC, dirs, 6.2, n_render=3,
                           acoustics_overrides={"impact_gain_sigma": 0.0, "impact_source": "synthetic"})
        for n in dirs:
            row = {r_["feature"]: r_ for r_ in rep["table"][n]}
            res["sound"][f"{e}_{n}"] = {"sim": row["fold_peak_to_mean"]["sim_mean"],
                                        "real": row["fold_peak_to_mean"]["real_mean"],
                                        "real_std": row["fold_peak_to_mean"]["real_std"]}
        print(e, {n: round(res["sound"][f"{e}_{n}"]["sim"], 2) for n in dirs}, flush=True)
    # 개수마다 실측 주기성에 가장 가까운 벽 반발계수 (e에 대해 선형 보간)
    best = {}
    for n in NS_REAL:
        ee = [e for e in ES if f"{e}_{n}" in res["sound"]]
        s = np.array([res["sound"][f"{e}_{n}"]["sim"] for e in ee])
        real = res["sound"][f"{ee[0]}_{n}"]["real"]
        k = int(np.argmin(abs(s - real)))
        best[n] = {"e_best_grid": ee[k], "sim_at_best": float(s[k]), "real": float(real),
                   "err_at_0.7": float(abs(res["sound"].get(f"0.7_{n}", {}).get("sim", np.nan) - real))}
    res["best_e_per_N"] = best
    (OUT / "grid.json").write_text(json.dumps(res, indent=1))
    L = ["# 벽 반발계수 × 개수: 동기화 (3차원 DEM, BB 372 rpm)", "",
         "## 충격량 가중 위상 집중도 R_J (1 = 모든 충돌이 한 위상)", "",
         "| e_w | " + " | ".join(f"N {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for e in ES:
        L.append(f"| {e} | " + " | ".join(f"{res['events'].get(f'{e}_{n}', {}).get('R_J', np.nan):.2f}" for n in NS) + " |")
    L += ["", "## 천장 충돌 비율 (충격량)", "", "| e_w | " + " | ".join(f"N {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for e in ES:
        L.append(f"| {e} | " + " | ".join(f"{res['events'].get(f'{e}_{n}', {}).get('ceiling_share', np.nan):.2f}" for n in NS) + " |")
    L += ["", "## 입자끼리 충돌 비율 (충격량)", "", "| e_w | " + " | ".join(f"N {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for e in ES:
        L.append(f"| {e} | " + " | ".join(f"{res['events'].get(f'{e}_{n}', {}).get('pp_share', np.nan):.2f}" for n in NS) + " |")
    L += ["", "## 소리 주기성 (주기 접힘 봉우리/평균) vs 실측", "",
          "| e_w | " + " | ".join(f"N {n}" for n in NS_REAL) + " | 평균 오차 |", "|---|" + "---|" * (len(NS_REAL) + 1),
          "| **실측** | " + " | ".join(f"{best[n]['real']:.2f}" for n in NS_REAL) + " | – |"]
    for e in ES:
        v = [res["sound"].get(f"{e}_{n}", {}).get("sim", np.nan) for n in NS_REAL]
        err = np.nanmean([abs(x - best[n]["real"]) for x, n in zip(v, NS_REAL)])
        L.append(f"| {e} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {err:.2f} |")
    L += ["", "개수마다 실측에 가장 가까운 e_w: " + ", ".join(f"N {n}: {best[n]['e_best_grid']}" for n in NS_REAL)]
    (OUT / "grid.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
