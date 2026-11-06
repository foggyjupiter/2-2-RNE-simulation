"""Incremental upload of a sweep folder to a Hugging Face dataset repo.

Uploads finished runs only (those with ``runs/<id>/done.json``): their wav clips and
the files needed to re-render them (events, wall force, config snapshot, summary) plus
the done marker (so a new machine can download the folder and resume the sweep),
plus ``metadata.csv`` / ``manifest.jsonl`` at the root (``metadata.csv`` makes the
repo loadable with ``datasets.load_dataset("audiofolder", ...)``). Already uploaded
paths are tracked in ``<out>/.hf_uploaded.txt``; a commit whose response was lost
(gateway error) is verified against the repo listing before retrying.

Authentication: the token comes from the environment (HF_TOKEN) or, in the cloud
session, is injected by the network proxy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RUN_FILES = ("config_snapshot.yaml", "events.npz", "wall_force.npz", "summary.json", "done.json")


def _pending(out: Path, done_set: set[str]) -> list[str]:
    paths = []
    for d in sorted((out / "runs").glob("*/done.json")):
        rec = json.loads(d.read_text())
        for r in rec["rows"]:
            if r["file_name"] not in done_set:
                paths.append(r["file_name"])
        for f in RUN_FILES:
            p = f"runs/{rec['run_id']}/{f}"
            if p not in done_set and (out / p).exists():
                paths.append(p)
    return paths


def upload(out: str | Path, repo_id: str, batch: int = 800, max_retries: int = 4,
           extra_files: dict | None = None, log=print) -> int:
    """Upload new files in batches; returns the number of files uploaded."""
    from huggingface_hub import CommitOperationAdd, HfApi

    from gransim.sweep.runner import write_manifest

    out = Path(out)
    api = HfApi()
    track = out / ".hf_uploaded.txt"
    done_set = set(track.read_text().split()) if track.exists() else set()
    n_rows = write_manifest(out)
    if n_rows:
        (out / "metadata.csv").write_bytes((out / "manifest.csv").read_bytes())
    todo = _pending(out, done_set)
    n = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(out / p)) for p in chunk]
        _commit(api, repo_id, ops, f"add {len(chunk)} files ({chunk[0]} …)", max_retries, log)
        with open(track, "a", encoding="utf-8") as fh:
            fh.write("\n".join(chunk) + "\n")
        n += len(chunk)
        log(f"uploaded {n}/{len(todo)}")
    meta = ["metadata.csv", "manifest.csv", "manifest.jsonl", "sweep_spec.yaml"]
    ops = [CommitOperationAdd(path_in_repo=m, path_or_fileobj=str(out / m))
           for m in meta if (out / m).exists()]
    for dst, src in (extra_files or {}).items():
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)))
    if ops:
        _commit(api, repo_id, ops, f"update manifest ({n_rows} clips)", max_retries, log)
    return n


def _commit(api, repo_id, ops, message, max_retries, log):
    """create_commit with retries. The gateway sometimes answers 502 although the commit
    went through, so after an error the latest commit title is checked first."""
    for attempt in range(max_retries + 1):
        try:
            api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops,
                              commit_message=message)
            return
        except Exception as e:
            time.sleep(5 * 2 ** attempt)
            try:
                last = api.list_repo_commits(repo_id, repo_type="dataset")[0]
                if last.title == message:
                    log(f"commit confirmed after error: {type(e).__name__}")
                    return
            except Exception:
                pass
            if attempt == max_retries:
                raise
            log(f"retry {attempt + 1} after {type(e).__name__}")
