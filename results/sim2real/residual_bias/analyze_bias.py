"""남은 실측 편향 (블라인드 V1: N 5~25 +17, N 80~100 -10)의 원인 분석. 학습 없이.

1. 최근접 이웃: 실측 클립마다 특징이 가장 비슷한 시뮬레이션 클립(v3_nosig) 20개의 N.
   - 전체 물성에서 찾기  → 블라인드 모델이 볼 수 있는 답
   - BB와 비슷한 물성에서만 찾기 → 물질을 알 때의 답
   두 경우 모두 편향이 같으면 소리 자체의 차이, BB 제한에서 사라지면 물질 혼동.
2. N 구간별로 실측과 가장 다른 특징 (BB 비슷한 물성, 표준화 거리).
3. 블라인드 모델(V1)의 실측 잔차와 상관이 큰 특징 (같은 N 안에서, 가진 속도 포함).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/rne_machine_learning')
from rneml.features import SCALAR_NAMES, BASIC_SCALARS, MOD_NAMES

O = Path(__file__).parent
m = pd.read_csv(O / 'meta_nosig.csv')
X = np.load(O / 'nosig.npz')['X']; R = np.load(O / 'real.npz'); XR, NR, names = R['X'], R['N'], R['name']
ok = ~np.isnan(X).any(1)
mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-9
Z, ZR = (X - mu) / sd, (XR - mu) / sd
N = m.N.values.astype(float)
d = m.particle_diameter_m.values * 1000; rho = m.particle_density.values
bb = ok & (d > 5.4) & (d < 6.6) & (rho > 850) & (rho < 1300)
bins = [(0, 0), (5, 25), (30, 50), (55, 75), (80, 100)]
nm = np.array(SCALAR_NAMES)
lines = ['# 남은 실측 편향의 원인 (학습 없이, v3_nosig)', '',
         f'시뮬레이션 클립 {ok.sum()}개 (그중 BB와 비슷한 물성: 지름 5.4~6.6 mm, 밀도 850~1300 → {bb.sum()}개), 실측 {len(NR)}개 (개수별 6~15번).', '']

def knn_bias(feat_idx, pool, k=20):
    P = Z[pool][:, feat_idx]; Np = N[pool]; dp = d[pool]
    est, dia = [], []
    for z in ZR[:, feat_idx]:
        dist = ((P - z) ** 2).sum(1)
        nn = np.argpartition(dist, k)[:k]
        est.append(np.median(Np[nn])); dia.append(np.median(dp[nn]))
    est = np.array(est); e = est - NR
    return [float(e[(NR >= a) & (NR <= b)].mean()) for a, b in bins], float(np.abs(e).mean()), np.array(dia)

groups = {'특징 220개': np.arange(len(nm)), '기본 스칼라 12개': np.arange(len(BASIC_SCALARS)),
          '변조 스펙트럼 192개': np.arange(len(BASIC_SCALARS), len(BASIC_SCALARS) + len(MOD_NAMES))}
lines += ['## 1. 최근접 이웃이 말하는 N (실측 클립마다 이웃 20개의 N 중앙값 − 참 N)', '',
          '| 특징 | 찾는 범위 | MAE | N 0 | N 5~25 | N 30~50 | N 55~75 | N 80~100 | 이웃의 지름 (mm, 중앙값) |', '|---|---|---|---|---|---|---|---|---|']
res = {}
for g, idx in groups.items():
    for pname, pool in [('전체 물성', ok), ('BB 비슷한 물성', bb)]:
        b, mae, dia = knn_bias(idx, pool)
        res[f'{g}/{pname}'] = {'bias': b, 'mae': mae, 'diameter_median': float(np.median(dia))}
        lines.append(f'| {g} | {pname} | {mae:.1f} | ' + ' | '.join(f'{x:+.1f}' for x in b) + f' | {np.median(dia):.2f} |')
lines.append('')
# 블라인드 모델 V1의 실측 편향 (비교용)
pr = [json.loads(l) for l in open('/home/user/rne_machine_learning/outputs/V1_nosig_blind/predictions.jsonl')]
pr = {x['run_id']: x for x in pr if x['split'] == 'real'}
P1 = np.array([max(pr[n]['pred'], 0) for n in names]); fs = np.array([pr[n].get('f_shake_est', np.nan) for n in names])
e1 = P1 - NR
lines += ['비교: 블라인드 모델 V1의 실측 편향 ' + ', '.join(f'N {a}~{b} {e1[(NR >= a) & (NR <= b)].mean():+.1f}' for a, b in bins), '']
# 2. N 구간별로 실측과 가장 다른 특징 (BB 비슷한 물성)
lines += ['## 2. N 구간별로 실측과 가장 다른 특징 (BB 비슷한 물성의 시뮬레이션 기준, 표준화 거리 = (실측 평균 − 시뮬 평균) / 시뮬 표준편차)', '']
for a, b in bins[1:]:
    s = bb & (N >= a - 2) & (N <= b + 2); r = (NR >= a) & (NR <= b)
    dz = (XR[r].mean(0) - X[s].mean(0)) / (X[s].std(0) + 1e-9)
    top = np.argsort(-np.abs(dz))[:6]
    lines.append(f'- N {a}~{b}: ' + ', '.join(f'{nm[i]} {dz[i]:+.1f}' for i in top))
lines.append('')
# 3. 같은 N 안에서 V1 잔차와 상관이 큰 특징
res_w = np.zeros_like(e1)
for n in np.unique(NR):
    msk = NR == n; res_w[msk] = e1[msk] - e1[msk].mean()
Xw = XR.copy()
for n in np.unique(NR):
    msk = NR == n; Xw[msk] -= Xw[msk].mean(0)
cor = np.array([np.corrcoef(Xw[:, j], res_w)[0, 1] if Xw[:, j].std() > 0 else 0 for j in range(Xw.shape[1])])
fw = fs - np.array([fs[NR == n].mean() for n in NR])
cf = np.corrcoef(fw[np.isfinite(fw)], res_w[np.isfinite(fw)])[0, 1]
top = np.argsort(-np.abs(np.nan_to_num(cor)))[:8]
lines += ['## 3. 같은 N 안에서 블라인드 모델(V1) 잔차와 함께 움직이는 것', '',
          f'- 가진 속도(클립별 추정): 상관 {cf:+.2f}',
          '- 특징: ' + ', '.join(f'{nm[i]} {cor[i]:+.2f}' for i in top), '']
(O / 'bias_report.md').write_text('\n'.join(lines), encoding='utf-8')
json.dump(res, open(O / 'bias_metrics.json', 'w'), indent=1)
print('\n'.join(lines))
