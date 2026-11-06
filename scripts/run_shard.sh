#!/bin/bash
# One machine's share of a sweep shared by several machines (cloud sessions).
#   scripts/run_shard.sh SPEC OUT PREFIX K N
# Runs jobs with index ≡ K (mod N), uploads its files to the Hugging Face dataset under
# PREFIX every 20 min (no manifest: the merge step `gransim upload OUT --prefix PREFIX
# --fetch` writes the full one), then uploads a marker PREFIX/_shards/shard_K_of_N.json.
set -u
SPEC=$1; OUT=$2; PREFIX=$3; K=$4; N=$5
REPO=${REPO:-FoggyJupiter/rne-granular-acoustics}
export HF_TOKEN=${HF_TOKEN:-proxy-injected}
mkdir -p "$OUT"
gransim sweep "$SPEC" --out "$OUT" --workers "$(nproc)" --shard "$K/$N" > "$OUT.sweep.log" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 10; done
  gransim upload "$OUT" --repo "$REPO" --prefix "$PREFIX" --no-manifest >> "$OUT.upload.log" 2>&1
done
wait $PID; RC=$?
gransim upload "$OUT" --repo "$REPO" --prefix "$PREFIX" --no-manifest >> "$OUT.upload.log" 2>&1
python3 - "$OUT" "$REPO" "$PREFIX" "$K" "$N" "$RC" <<'PY'
import json, sys, time
from pathlib import Path
from huggingface_hub import HfApi
out, repo, prefix, k, n, rc = sys.argv[1:]
done = len(list(Path(out, "runs").glob("*/done.json")))
fails = Path(out, "failures.jsonl")
nfail = len(fails.read_text().splitlines()) if fails.exists() else 0
marker = {"shard": f"{k}/{n}", "runs_done": done, "failures": nfail, "sweep_rc": int(rc),
          "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
p = Path(out).with_suffix(".marker.json"); p.write_text(json.dumps(marker))
HfApi().upload_file(path_or_fileobj=str(p), path_in_repo=f"{prefix.rstrip('/')}/_shards/shard_{k}_of_{n}.json",
                    repo_id=repo, repo_type="dataset", commit_message=f"shard {k}/{n} done")
print(json.dumps(marker))
PY
echo "SHARD $K/$N DONE rc=$RC"
