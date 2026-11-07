"""launch_sync.sh 실행들을 실측과 비교 → egrid_sync.json, sync_summary.md.

주 지표: 주기성 (fold_peak_to_mean; 클립마다 추정한 가진 주파수로 접음) — N 10~35에서 실측이
합성보다 크게 높았던 것 (실측 3.9 / 3.2 / 2.6 vs 보정값 합성 ~2.0~2.2).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from gransim.analysis.sim2real import run_sim2real  # noqa: E402

REC = "/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all"
RUNS = ROOT / "runs" / "sync_probe"
SETS = ["base", "r0", "r0.2", "pw0.5", "pp0.8", "mu0.5"]
NS = [10, 20, 35, 100]
FEATS = ["fold_peak_to_mean", "onset_rate", "kurtosis", "rms"]


def main():
    out = []
    for s in SETS:
        dirs = {n: [str(RUNS / f"{s}_N{n}")] for n in NS if (RUNS / f"{s}_N{n}" / "summary.json").exists()}
        if len(dirs) < len(NS):
            print("skip", s, sorted(dirs))
            continue
        rep = run_sim2real(REC, dirs, 6.2, n_render=3,
                           acoustics_overrides={"impact_gain_sigma": 0.0, "impact_source": "synthetic"})
        tab = {str(n): {r["feature"]: {"real": r["real_mean"], "sim": r["sim_mean"]} for r in rep["table"][n]
                        if r["feature"] in FEATS} for n in NS}
        fe = np.mean([abs(tab[str(n)]["fold_peak_to_mean"]["sim"] - tab[str(n)]["fold_peak_to_mean"]["real"])
                      for n in NS])
        out.append({"set": s, "gain": rep["mic_gain"], "fold_err": float(fe), "table": tab})
        print(s, "fold err", round(fe, 2), {n: round(tab[str(n)]["fold_peak_to_mean"]["sim"], 2) for n in NS}, flush=True)
    (Path(__file__).parent / "egrid_sync.json").write_text(json.dumps(out, indent=1, default=float))
    real = {n: out[0]["table"][str(n)]["fold_peak_to_mean"]["real"] for n in NS}
    lines = ["| 설정 | " + " | ".join(f"N {n}" for n in NS) + " | 평균 오차 |", "|---|" + "---|" * (len(NS) + 1),
             "| **실측** | " + " | ".join(f"{real[n]:.2f}" for n in NS) + " | – |"]
    for r in out:
        lines.append(f"| {r['set']} | " + " | ".join(f"{r['table'][str(n)]['fold_peak_to_mean']['sim']:.2f}" for n in NS)
                     + f" | {r['fold_err']:.2f} |")
    (Path(__file__).parent / "sync_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
