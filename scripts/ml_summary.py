"""모든 실험의 핵심 수치 요약 (보고서 4장용)."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/home/user/rne_machine_learning')
from rneml.suite import EXPERIMENTS
import os
O = Path(os.environ.get('RNEML_OUTPUTS', '/home/user/rne_machine_learning/outputs'))
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d
rows = []
for prio, name, group, data, base, over in EXPERIMENTS:
    p = O / name / 'metrics.json'
    if not p.exists(): continue
    d = json.loads(p.read_text())
    rows.append(dict(name=name, group=group, data=data,
        test=g(d,'test','mae'), ood=g(d,'test_ood','mae'), real=g(d,'real','mae'), bias=g(d,'real','bias'),
        w5=g(d,'real','within_5'), cov=g(d,'real','in_2sigma'), d_r2=g(d,'properties_test','diameter','r2'),
        rho_r2=g(d,'properties_test','density','r2'), m_r2=g(d,'properties_test','mass','r2'),
        d_real=g(d,'real_properties','diameter','pred_median'), m_real=g(d,'real_properties','mass','pred_median'),
        degen=g(d,'degeneracy_test','corr_N_error_with','diameter'), slope=g(d,'real_speed','within_N_slope_per_hz')))
f = lambda v, s='{:.1f}': '–' if v is None else s.format(v)
print('| 실험 | 묶음 | 데이터 | test | OOD | 실측 MAE | 실측 편향 | 실측 ±5 | 지름 R² | 실측 지름(mm) | N–지름 오차 상관 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['name']} | {r['group']} | {r['data']} | {f(r['test'])} | {f(r['ood'])} | {f(r['real'])} | {f(r['bias'],'{:+.1f}')} | "
          f"{f(r['w5'],'{:.0%}')} | {f(r['d_r2'],'{:.2f}')} | {f(r['d_real'],'{:.2f}')} | {f(r['degen'],'{:+.2f}')} |")
json.dump(rows, open(Path(__file__).resolve().parents[1] / 'results' / 'ml' / 'summary.json', 'w'), indent=1, default=float)
