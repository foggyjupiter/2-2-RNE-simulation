"""원인 진단용 특징 계산: 같은 동역학(v1) 위에 음향만 다른 세 데이터셋(v1 원본, v1_sig 세기 퍼짐만,
v1_clk 실제 충돌음만)과 실측 녹음(각 N의 6~15번)의 스칼라 특징 220개."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/rne_machine_learning')
from rneml.features import extract

D = Path('/home/user/2-2-RNE-simulation/datasets')
REAL = Path('/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all')
OUT = Path(__file__).parent

def sc(p):
    try:
        return extract(p)[1]
    except Exception:
        return np.full(220, np.nan, np.float32)

if __name__ == '__main__':
    m = pd.read_csv(D / 'v1_sig/manifest.csv')
    m.to_csv(OUT / 'manifest_v1.csv', index=False)
    real = sorted(p for p in REAL.glob('*.wav') if int(p.stem.split('-')[1]) >= 6)
    with ProcessPoolExecutor(4) as ex:
        X = np.stack(list(ex.map(sc, [str(p) for p in real], chunksize=8)))
        np.savez(OUT / 'real.npz', X=X, N=np.array([int(p.stem.split('-')[0]) for p in real]))
        print('real', X.shape, flush=True)
        for name in ['v1', 'v1_sig', 'v1_clk']:
            X = np.stack(list(ex.map(sc, [str((D / 'v1_audio' if name == 'v1' else D / name) / f) for f in m.file_name], chunksize=16)))
            np.savez(OUT / f'{name}.npz', X=X)
            print(name, X.shape, np.isnan(X).any(1).sum(), flush=True)
