"""Block until a sharded sweep under PREFIX is complete on the Hugging Face dataset.

    python wait_markers.py PREFIX N [INTERVAL_S] [TOTAL_RUNS]

Complete = all PREFIX/_shards/shard_{k}_of_N.json markers exist, or (fallback, when a marker
upload was lost or a replacement machine finished the shard) TOTAL_RUNS run done markers
PREFIX/runs/*/done.json are on the repo."""
import sys
import time

from huggingface_hub import HfApi

prefix, n = sys.argv[1], int(sys.argv[2])
interval = int(sys.argv[3]) if len(sys.argv) > 3 else 300
total = int(sys.argv[4]) if len(sys.argv) > 4 else None
want = {f"{prefix}/_shards/shard_{k}_of_{n}.json" for k in range(n)}
while True:
    try:
        files = HfApi().list_repo_files("FoggyJupiter/rne-granular-acoustics", repo_type="dataset")
        have = set(files)
        runs = sum(1 for f in files if f.startswith(prefix + "/runs/") and f.endswith("/done.json"))
        if want <= have or (total and runs >= total):
            print(f"all {n} {prefix} shards done (markers {len(want & have)}/{n}, runs {runs})", flush=True)
            break
        print(time.strftime("%H:%M"), f"{prefix}: {len(want & have)}/{n} shards done, runs uploaded {runs}"
              + (f"/{total}" if total else ""), flush=True)
    except Exception as e:
        print("poll error", repr(e)[:100], flush=True)
    time.sleep(interval)
