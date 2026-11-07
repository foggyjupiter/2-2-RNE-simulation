"""소리로 개수를 얼마나 잘게 가를 수 있나 — 실측 녹음의 개수 분해능 (학습 없음).

1차 연구 녹음 (N = 0~100, 개수별 15개)에서 물리 특징의 개수별 평균 μ(N)과 녹음 간 퍼짐 Σ(N)을 재고,
  δN(N) = 1 / sqrt( μ'(N)ᵀ Σ(N)⁻¹ μ'(N) )         (피셔 정보 → 크라메르-라오 하한)
으로 "같은 BB를 아는 상태에서 이 특징들로 가를 수 있는 개수 차이"를 구한다. 음량 하나만 쓰면
δN = σ(log RMS) / |d log RMS / dN|. 음량이 N^p 로 커지면 d log RMS/dN = p/(2N) 이므로 δN ∝ N:
같은 비율(%)로만 가를 수 있다 (베버 법칙과 같은 꼴).

    python scripts/physics_resolution.py   → results/physics/resolution.{md,json}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gransim.acoustics.assets import read_wav  # noqa: E402
from gransim.analysis.sim2real import features  # noqa: E402

REC = "/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all"
OUT = ROOT / "results" / "physics"
FEATS = ["log_rms", "onset_rate", "kurtosis", "fold_peak_to_mean", "centroid_khz", "band_db_16000_24000",
         "clip_fraction"]


def real_table():
    cache = OUT / "real_features.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    tab = {}
    for f in sorted(glob.glob(REC + "/*.wav")):
        n, i = Path(f).stem.split("-")
        fs, x = read_wav(f)
        d = features(x, fs, 6.2)
        row = {"rec": int(i), "log_rms": float(np.log(d["rms"])), "onset_rate": d["onset_rate"],
               "kurtosis": float(d["kurtosis"]), "fold_peak_to_mean": d["fold_peak_to_mean"],
               "centroid_khz": d["centroid_hz"] / 1000, "band_db_16000_24000": d["band_db_16000_24000"],
               "clip_fraction": d["clip_fraction"], "f_shake": d["f_shake_est"]}
        tab.setdefault(int(n), []).append(row)
    cache.write_text(json.dumps(tab))
    return tab


def main():
    tab = real_table()
    Ns = np.array(sorted(tab))
    mu = {k: np.array([np.mean([r[k] for r in tab[n]]) for n in Ns]) for k in FEATS}
    sd = {k: np.array([np.std([r[k] for r in tab[n]], ddof=1) for n in Ns]) for k in FEATS}
    res, L = {"N": Ns.tolist()}, ["# 실측 녹음의 개수 분해능 (학습 없음, 1차 연구 녹음 개수별 15개)", ""]
    # 음량 지수 p: RMS² ∝ N^p (N ≥ 5)
    m = Ns >= 5
    p = float(np.polyfit(np.log(Ns[m]), 2 * mu["log_rms"][m], 1)[0])
    res["loudness_exponent_p"] = p
    L += [f"소리 에너지(RMS²) ∝ N^{p:.2f} (N 5~100). 개수가 2배가 되어도 소리 에너지는 {2 ** p:.2f}배.", ""]
    # 개수별 분해능: 이웃 개수와의 중앙 차분으로 μ'(N)
    rows = []
    for j, n in enumerate(Ns):
        if n == 0:
            continue
        lo, hi = max(j - 2, 1), min(j + 2, len(Ns) - 1)              # 이웃 ±10개 (N=0 제외)에서 국소 선형 기울기
        idx = range(lo, hi + 1)
        nn = np.concatenate([[Ns[i]] * len(tab[Ns[i]]) for i in idx]).astype(float)
        X = np.concatenate([np.array([[r[k] for k in FEATS] for r in tab[Ns[i]]]) for i in idx])
        der = np.array([np.polyfit(nn, X[:, q], 1)[0] for q in range(len(FEATS))])
        Xc = np.concatenate([np.array([[r[k] for k in FEATS] for r in tab[Ns[i]]]) -
                             np.array([[r[k] for k in FEATS] for r in tab[Ns[i]]]).mean(0) for i in idx])
        C = np.cov(Xc.T)
        C = 0.7 * C + 0.3 * np.diag(np.diag(C))                    # 표본 45개라 수축
        info = float(der @ np.linalg.solve(C, der))
        i_rms = FEATS.index("log_rms")
        d_rms = abs(der[i_rms]) / np.sqrt(C[i_rms, i_rms])
        rows.append({"N": int(n), "dN_all": float(1 / np.sqrt(info)) if info > 0 else float("inf"),
                     "dN_rms": float(1 / d_rms) if d_rms > 0 else float("inf"),
                     "rms_cv": float(sd["log_rms"][j])})
    res["rows"] = rows
    L += ["| N | 음량만 δN | 특징 7개 δN | 녹음 간 음량 퍼짐 (log RMS 표준편차) |", "|---|---|---|---|"]
    L += [f"| {r['N']} | {r['dN_rms']:.1f} | {r['dN_all']:.1f} | {r['rms_cv']:.3f} |" for r in rows]
    # 구간 요약 (중앙값)
    rng = [(5, 25), (30, 50), (55, 75), (80, 100)]
    summ = {}
    for a, b in rng:
        s = [r for r in rows if a <= r["N"] <= b]
        summ[f"{a}-{b}"] = {"dN_rms": float(np.median([r["dN_rms"] for r in s])),
                            "dN_all": float(np.median([r["dN_all"] for r in s]))}
    res["ranges"] = summ
    L += ["", "| 구간 | 음량만 δN (중앙값) | 특징 7개 δN (중앙값) |", "|---|---|---|"]
    L += [f"| {k} | {v['dN_rms']:.1f} | {v['dN_all']:.1f} |" for k, v in summ.items()]
    # 음량 지수의 구간별 값: 실측 vs DEM (벽 충돌 충격량² 흐름, 입자 속력)
    Nn = np.array(Ns, float)
    pr = {}
    for a, b in [(5, 25), (30, 100)]:
        s_ = (Nn >= a) & (Nn <= b)
        pr[f"{a}-{b}"] = float(np.polyfit(np.log(Nn[s_]), 2 * mu["log_rms"][s_], 1)[0])
    res["loudness_exponent_by_range"] = pr
    dem = {}
    runs = ROOT / "runs" / "sync_probe"
    for tag, e in [("pw0.5", 0.5), ("base", 0.7)]:
        rows_ = []
        for n in [1, 2, 5, 10, 20, 35, 50, 100]:
            z = np.load(runs / f"{tag}_N{n}" / "events.npz"); dg = np.load(runs / f"{tag}_N{n}" / "diagnostics.npz")
            t0, t1 = z["record_window"]; mm = (z["t_start"] >= t0) & (z["t_start"] <= t1)
            J = np.abs(z["J_n"][mm]); w = z["wall_id"][mm]
            md = (dg["t"] >= t0) & (dg["t"] <= t1)
            v = float(np.sqrt(2 * dg["ke_rel"][md].mean() / n / 1.2e-4))
            rows_.append({"N": n, "wall_J2_rate": float((J[w >= 0] ** 2).sum() / (t1 - t0)),
                          "pp_J2_rate": float((J[w < 0] ** 2).sum() / (t1 - t0)), "v_rms": v})
        NN = np.array([r_["N"] for r_ in rows_], float)
        EW = np.array([r_["wall_J2_rate"] for r_ in rows_]); VV = np.array([r_["v_rms"] for r_ in rows_])
        fit = {}
        for a, b in [(1, 20), (20, 100)]:
            s_ = (NN >= a) & (NN <= b)
            fit[f"{a}-{b}"] = {"p_wall": float(np.polyfit(np.log(NN[s_]), np.log(EW[s_]), 1)[0]),
                               "v_exp": float(np.polyfit(np.log(NN[s_]), np.log(VV[s_]), 1)[0])}
        dem[f"e_w={e}"] = {"rows": rows_, "fit": fit}
    res["dem_loudness"] = dem
    L += ["", "## 음량이 개수에 따라 커지는 정도 (소리 에너지 ∝ N^p)", "",
          "| | N 5~25 (또는 1~20) | N 30~100 (또는 20~100) |", "|---|---|---|",
          f"| 실측 녹음 RMS² | {pr['5-25']:.2f} | {pr['30-100']:.2f} |"]
    for k, v_ in dem.items():
        L.append(f"| DEM {k} 벽 충돌 충격량² | {v_['fit']['1-20']['p_wall']:.2f} | {v_['fit']['20-100']['p_wall']:.2f} |")
        L.append(f"| DEM {k} 입자 속력 지수 | {v_['fit']['1-20']['v_exp']:.2f} | {v_['fit']['20-100']['v_exp']:.2f} |")
    res["features_mean"] = {k: v.tolist() for k, v in mu.items()}
    res["features_std"] = {k: v.tolist() for k, v in sd.items()}
    (OUT / "resolution.json").write_text(json.dumps(res, indent=1))
    (OUT / "resolution.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
