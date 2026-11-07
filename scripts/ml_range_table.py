"""개수 구간별 오차 표 (학습 코드 레포 outputs/*/predictions.jsonl → results/ml/range_table.{json,md}).

    RNEML_OUTPUTS=/home/user/rne_machine_learning/outputs python scripts/ml_range_table.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(os.environ.get("RNEML_OUTPUTS", "/home/user/rne_machine_learning/outputs"))
RANGES = [(0, 0), (5, 25), (30, 50), (55, 75), (80, 100)]
MODELS = [("V1_nosig_blind", "블라인드, 한 속도 (V1)"),
          ("H9_nosig_set_per_set", "블라인드, 여러 속도 (H9)"),
          ("V2_nosig_known_size_weight", "크기·무게 앎 (V2)"),
          ("V4_nosig_dann", "DANN (V4)"),
          ("V8_nosig_dann_known_size_weight", "DANN + 크기·무게 (V8)"),
          ("V7_nosig_real_labeled_5", "실측 정답 5개/개수 (V7)")]


def stats(P):
    out = []
    for lo, hi in RANGES:
        v = np.array([p["pred"] - p["N"] for p in P if lo <= p["N"] <= hi])
        out.append({"mae": float(np.abs(v).mean()), "bias": float(v.mean()), "n": int(len(v))} if len(v) else None)
    return out


def main():
    res = {}
    for key, _ in MODELS:
        f = OUTPUTS / key / "predictions.jsonl"
        if not f.exists():
            continue
        P = [json.loads(line) for line in open(f)]
        res[key] = {"real": stats([p for p in P if p["split"] == "real"]),
                    "test": stats([p for p in P if p["split"] == "test"])}
    out = ROOT / "results" / "ml"
    (out / "range_table.json").write_text(json.dumps(res, indent=1))
    head = " | ".join("N 0" if lo == hi else f"N {lo}~{hi}" for lo, hi in RANGES)
    L = ["실측 MAE (편향)", "", f"| 모델 | {head} |", "|---|" + "---|" * len(RANGES)]
    for key, lab in MODELS:
        if key in res:
            L.append(f"| {lab} | " + " | ".join(f"{s['mae']:.1f} ({s['bias']:+.1f})" for s in res[key]["real"]) + " |")
    L += ["", "시뮬레이션 test MAE (편향)", "", f"| 모델 | {head} |", "|---|" + "---|" * len(RANGES)]
    for key, lab in MODELS:
        if key in res:
            L.append(f"| {lab} | " + " | ".join("–" if s is None else f"{s['mae']:.1f} ({s['bias']:+.1f})"
                                                for s in res[key]["test"]) + " |")
    (out / "range_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
