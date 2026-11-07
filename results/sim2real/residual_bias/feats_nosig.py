"""남은 실측 편향 분석용 특징: v3_nosig 6400 클립 + 실측 210개(파일명 포함)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/rne_machine_learning')
from rneml.features import extract

D = Path('/home/user/2-2-RNE-simulation/datasets/v3_nosig_audio/v3_nosig')
REAL = Path('/tmp/claude-0/-home-user-2-2-RNE-simulation/04e80129-3b37-5580-b51d-630d931e5d63/scratchpad/cnn1/data-all')
OUT = Path(__file__).parent

def sc(p):
    try:
        return extract(p)[1]
    except Exception:
        return np.full(220, np.nan, np.float32)

if __name__ == '__main__':
    m = pd.read_csv(D / 'metadata.csv')
    m.to_csv(OUT / 'meta_nosig.csv', index=False)
    real = sorted(p for p in REAL.glob('*.wav') if int(p.stem.split('-')[1]) >= 6)
    with ProcessPoolExecutor(4) as ex:
        X = np.stack(list(ex.map(sc, [str(p) for p in real], chunksize=8)))
        np.savez(OUT / 'real.npz', X=X, N=np.array([int(p.stem.split('-')[0]) for p in real]), name=np.array([p.stem for p in real]))
        X = np.stack(list(ex.map(sc, [str(D / f) for f in m.file_name], chunksize=16)))
        np.savez(OUT / 'nosig.npz', X=X)
        print('done', X.shape, np.isnan(X).any(1).sum(), flush=True)
