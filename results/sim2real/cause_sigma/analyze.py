"""원인 진단: 같은 동역학 위에 음향만 다른 세 합성음(v1 원본 / v1_sig 세기 퍼짐만 / v1_clk 실제 충돌음만)으로
같은 회귀기(특징 → N, 재료 정보 없음)를 학습해 실측 녹음에서의 오차와 편향을 비교하고,
실측과 가장 멀어진 특징을 찾는다."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, '/home/user/rne_machine_learning')
from rneml.features import SCALAR_NAMES, BASIC_SCALARS, MOD_NAMES, SK_NAMES

O = Path(__file__).parent
m = pd.read_csv(O / 'manifest_v1.csv')
R = np.load(O / 'real.npz'); XR, NR = R['X'], R['N']
sets = {k: np.load(O / f'{k}.npz')['X'] for k in ['v1', 'v1_sig', 'v1_clk']}
names = np.array(SCALAR_NAMES)
groups = {'전체': np.arange(len(names)),
          '기본 스칼라': np.arange(len(BASIC_SCALARS)),
          '변조 스펙트럼': np.arange(len(BASIC_SCALARS), len(BASIC_SCALARS) + len(MOD_NAMES)),
          '스펙트럼 첨도': np.arange(len(BASIC_SCALARS) + len(MOD_NAMES), len(names))}
rng = np.random.default_rng(0)
runs = m.run_id.unique(); test_runs = set(rng.choice(runs, len(runs) // 5, replace=False))
te = m.run_id.isin(test_runs).values; tr = ~te
y = m.N.values.astype(float)
bins = [(0, 0), (5, 25), (30, 50), (55, 75), (80, 100)]
out = {}
lines = ['# 원인 진단: 소리만 바꾼 세 합성음으로 같은 회귀기를 학습했을 때 실측 오차', '',
         '같은 v1 동역학(1600 실행 × 4 렌더), 재료 정보 없이 특징 220개 → N (기울기 부스팅 회귀). '
         '실측 = 첫 연구 녹음 각 N의 6~15번 (210개).', '']
for gname, idx in groups.items():
    lines += [f'## 특징: {gname} ({len(idx)}개)', '',
              '| 학습 소리 | 시뮬 MAE | 실측 MAE | ' + ' | '.join(f'N {a}–{b} 편향' if a != b else f'N {a} 편향' for a, b in bins) + ' |',
              '|---|---|---|' + '---|' * len(bins)]
    for k, X in sets.items():
        ok = ~np.isnan(X).any(1)
        reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        reg.fit(X[tr & ok][:, idx], y[tr & ok])
        ps = reg.predict(X[te & ok][:, idx]); pr = np.clip(reg.predict(XR[:, idx]), 0, 100)
        sim_mae = np.mean(np.abs(ps - y[te & ok])); real_mae = np.mean(np.abs(pr - NR))
        bias = [np.mean(pr[(NR >= a) & (NR <= b)] - NR[(NR >= a) & (NR <= b)]) for a, b in bins]
        out[f'{gname}/{k}'] = {'sim_mae': sim_mae, 'real_mae': real_mae, 'bias': bias}
        lines.append(f'| {k} | {sim_mae:.1f} | {real_mae:.1f} | ' + ' | '.join(f'{b:+.1f}' for b in bias) + ' |')
    lines.append('')
# 특징별 거리: BB 근처 재료(지름 4.5–7.5 mm, 밀도 700–1500)의 시뮬 클립, N 구간별 실측 평균과의 표준화 차이
d = m.particle_diameter_m.values * 1000; rho = m.particle_density.values
bb = (d > 4.5) & (d < 7.5) & (rho > 700) & (rho < 1500)
dist = {}
for k, X in sets.items():
    D = []
    for a, b in bins[1:]:
        s = bb & (y >= a - 2) & (y <= b + 2); r = (NR >= a) & (NR <= b)
        mu, sd = np.nanmean(X[s], 0), np.nanstd(X[s], 0) + 1e-6
        D.append(np.abs(np.nanmean(XR[r], 0) - mu) / sd)
    dist[k] = np.mean(D, 0)
lines += ['## 실측과 가장 멀어진 특징 (BB 비슷한 재료, N 구간 평균 표준화 거리)', '',
          '| 특징 | v1 원본 | v1_sig 세기 퍼짐 | v1_clk 실제 충돌음 |', '|---|---|---|---|']
order = np.argsort(-(np.maximum(dist['v1_sig'], dist['v1_clk']) - dist['v1']))
for i in order[:12]:
    lines.append(f'| {names[i]} | {dist["v1"][i]:.2f} | {dist["v1_sig"][i]:.2f} | {dist["v1_clk"][i]:.2f} |')
lines += ['', '| 데이터 | 특징 220개 평균 거리 | 기본 스칼라 평균 | 변조 평균 | 첨도 평균 |', '|---|---|---|---|---|']
for k in sets:
    lines.append(f'| {k} | {dist[k].mean():.2f} | ' + ' | '.join(f'{dist[k][groups[g]].mean():.2f}' for g in ['기본 스칼라', '변조 스펙트럼', '스펙트럼 첨도']) + ' |')
(O / 'cause_report.md').write_text('\n'.join(lines), encoding='utf-8')
json.dump({k: v for k, v in out.items()}, open(O / 'cause_metrics.json', 'w'), default=float, indent=1)
print('\n'.join(lines))
