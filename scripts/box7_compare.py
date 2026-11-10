"""상자 정정(7×7×5) 재실험 결과를 예전(5×5×7) 결과와 같은 지표로 나란히 놓는다.

새 결과: Hugging Face experiments_box7/<실험>/{metrics.json, provenance.json, predictions.jsonl}
예전 결과: 학습 레포 outputs/<실험>/metrics.json (RNEML_OUTPUTS, 기본 /home/user/rne_machine_learning/outputs)

    python scripts/box7_compare.py   → results/ml/box7_compare.{json,md}
"""
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(os.environ.get("RNEML_OUTPUTS", "/home/user/rne_machine_learning/outputs"))
REPO = "FoggyJupiter/rne-granular-acoustics"
CACHE = ROOT / "results" / "ml" / "box7_raw"
EXPS = [("A4_first_study_repro", "1차 연구 방식 재현 (A4)"), ("A5_bb_trained", "BB만 학습한 본 모델 (A5)"),
        ("F_rms", "음량만 회귀 (F)"), ("F_gbm_physics", "손특징 부스팅 (F)"),
        ("V1_nosig_blind", "블라인드, 한 속도 (V1)"), ("H9_nosig_set_per_set", "블라인드, 여러 속도 (H9)"),
        ("L1_lowe_blind", "블라인드 + 낮은 벽 반발 보충 (L1)"),
        ("V4_nosig_dann", "DANN, 1~5번 적응 (V4)"), ("Q3_cv_dann_f1", "DANN, 6~10번 (Q3)"),
        ("Q4_cv_dann_f2", "DANN, 11~15번 (Q4)"),
        ("V8_nosig_dann_known_size_weight", "DANN + 크기·무게, 1~5번 (V8)"),
        ("Q5_cv_dann_known_size_weight_f1", "DANN + 크기·무게, 6~10번 (Q5)"),
        ("Q6_cv_dann_known_size_weight_f2", "DANN + 크기·무게, 11~15번 (Q6)"),
        ("H11_nosig_set_dann", "여러 속도 + DANN (H11)"),
        ("V7_nosig_real_labeled_5", "실측 정답 5개, 1~5번 (V7)"),
        ("Q1_cv_real_labeled_5_f1", "실측 정답 5개, 6~10번 (Q1)"),
        ("Q2_cv_real_labeled_5_f2", "실측 정답 5개, 11~15번 (Q2)")]
RANGES = [(0, 0), (5, 25), (30, 50), (55, 75), (80, 100)]


def get(e, f):
    try:
        return Path(hf_hub_download(REPO, f"experiments_box7/{e}/{f}", repo_type="dataset", local_dir=str(CACHE)))
    except Exception:  # noqa: BLE001
        return None


def summ(m):
    if m is None:
        return None
    g = lambda d, *k: (lambda x: x)(d.get(k[0], {}).get(k[1]) if len(k) == 2 else d.get(k[0]))  # noqa: E731
    out = {"test": g(m, "test", "mae"), "ood": g(m, "test_ood", "mae"), "real": g(m, "real", "mae"),
           "real_bias": g(m, "real", "bias"), "real_w5": g(m, "real", "within_5"),
           "speed": (m.get("real_speed") or {}).get("within_N_slope_per_hz")}
    for split in ("test", "real"):
        c = ((m.get(split) or {}).get("conformal") or {}).get("0.1") or {}
        out[f"{split}_cov90"] = c.get("coverage")
    return out


def ranges(pred_file):
    if pred_file is None:
        return None
    P = [json.loads(line) for line in open(pred_file)]
    R = [p for p in P if p.get("split") == "real"]
    res = []
    for lo, hi in RANGES:
        v = np.array([p["pred"] - p["N"] for p in R if lo <= p["N"] <= hi])
        res.append({"mae": float(np.abs(v).mean()), "bias": float(v.mean())} if len(v) else None)
    return res


def main():
    have = {e.path.split("/")[-1] for e in HfApi().list_repo_tree(REPO, path_in_repo="experiments_box7", repo_type="dataset")}
    res = {}
    for e, lab in EXPS:
        new = old = prov = rng = None
        if e in have:
            mf = get(e, "metrics.json")
            new = summ(json.load(open(mf))) if mf else None
            pf = get(e, "provenance.json")
            prov = json.load(open(pf)).get("git_commit", "")[:7] if pf else None
            rng = ranges(get(e, "predictions.jsonl")) if new else None
        if (OLD / e / "metrics.json").exists():
            old = summ(json.load(open(OLD / e / "metrics.json")))
        res[e] = {"label": lab, "new": new, "old": old, "commit": prov, "ranges_new": rng}
    out = ROOT / "results" / "ml"
    (out / "box7_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"  # noqa: E731
    L = ["# 상자 정정(바닥 7×7 cm, 높이 5 cm) 재실험 vs 예전(5×5×7) 결과", "",
         "MAE(개). 실측 = 1차 연구 녹음 개수별 6~15번 210개. 새 결과는 HF experiments_box7/, 예전은 학습 레포 outputs/.", "",
         "| 모델 | 새 test | 새 OOD | **새 실측** | 새 ±5 | 새 편향 | 예전 실측 | 예전 ±5 | 커밋 |", "|---|---|---|---|---|---|---|---|---|"]
    for e, r in res.items():
        n, o = r["new"] or {}, r["old"] or {}
        bias = "–" if n.get("real_bias") is None else f"{n['real_bias']:+.1f}"
        L.append(f"| {r['label']} | {f(n.get('test'))} | {f(n.get('ood'))} | **{f(n.get('real'))}** | {pct(n.get('real_w5'))} | "
                 f"{bias} | {f(o.get('real'))} | {pct(o.get('real_w5'))} | {r['commit'] or '–'} |")
    # 3겹 평균
    def avg(keys, which="new"):
        v = [((res[k][which] or {}).get("real")) for k in keys]
        return None if any(x is None for x in v) else float(np.mean(v))
    L += ["", "## 3겹 교차검증 평균 (실측 MAE)", "", "| 방법 | 새 | 예전 |", "|---|---|---|"]
    for lab, ks in [("DANN", ["V4_nosig_dann", "Q3_cv_dann_f1", "Q4_cv_dann_f2"]),
                    ("DANN + 크기·무게", ["V8_nosig_dann_known_size_weight", "Q5_cv_dann_known_size_weight_f1", "Q6_cv_dann_known_size_weight_f2"]),
                    ("실측 정답 5개", ["V7_nosig_real_labeled_5", "Q1_cv_real_labeled_5_f1", "Q2_cv_real_labeled_5_f2"])]:
        L.append(f"| {lab} | {f(avg(ks))} | {f(avg(ks, 'old'))} |")
    L += ["", "## 개수 구간별 실측 MAE (편향) — 새 결과", "",
          "| 모델 | N 0 | N 5~25 | N 30~50 | N 55~75 | N 80~100 |", "|---|---|---|---|---|---|"]
    for e, r in res.items():
        if r["ranges_new"]:
            L.append(f"| {r['label']} | " + " | ".join("–" if s is None else f"{s['mae']:.1f} ({s['bias']:+.1f})"
                                                     for s in r["ranges_new"]) + " |")
    L += ["", "## 불확도·속도 강건성 — 새 결과", "", "| 모델 | 시뮬레이션 90% 포함률 | 실측 90% 포함률 | 속도 민감도 (개/Hz) |",
          "|---|---|---|---|"]
    for e, r in res.items():
        n = r["new"] or {}
        if n:
            L.append(f"| {r['label']} | {f(n.get('test_cov90'), 2)} | {f(n.get('real_cov90'), 2)} | {f(n.get('speed'), 0)} |")
    (out / "box7_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
