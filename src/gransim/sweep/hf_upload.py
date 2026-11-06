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
           extra_files: dict | None = None, log=print, prefix: str = "",
           manifest: bool = True, manifest_only: bool = False) -> int:
    """Upload new files in batches; returns the number of files uploaded.

    ``prefix`` puts the whole folder under a sub-directory of the repo (e.g. a
    companion dataset ``bb_fixed/`` next to the main one).
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    from gransim.sweep.runner import write_manifest

    out = Path(out)
    api = HfApi()
    track = out / ".hf_uploaded.txt"
    done_set = set(track.read_text().split()) if track.exists() else set()
    n_rows = write_manifest(out)
    if n_rows:
        (out / "metadata.csv").write_bytes((out / "manifest.csv").read_bytes())
    # manifest=False: one shard of a sweep shared by several machines — upload its files but
    # not a manifest that would list only this shard (the merge step writes the full one).
    # manifest_only=True: only (re)write the manifest files (after a merge).
    todo = [] if manifest_only else _pending(out, done_set)
    n = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=prefix + p, path_or_fileobj=str(out / p)) for p in chunk]
        _commit(api, repo_id, ops, f"add {len(chunk)} files ({chunk[0]} …)", max_retries, log)
        with open(track, "a", encoding="utf-8") as fh:
            fh.write("\n".join(chunk) + "\n")
        n += len(chunk)
        log(f"uploaded {n}/{len(todo)}")
    meta = ["metadata.csv", "manifest.csv", "manifest.jsonl", "sweep_spec.yaml"] if manifest else []
    ops = [CommitOperationAdd(path_in_repo=prefix + m, path_or_fileobj=str(out / m))
           for m in meta if (out / m).exists()]
    for dst, src in (extra_files or {}).items():
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)))
    if ops:
        _commit(api, repo_id, ops, f"update manifest ({n_rows} clips)", max_retries, log)
    return n


def fetch_remote(out: str | Path, repo_id: str, prefix: str, log=print, audio: bool = True) -> int:
    """Merge step of a sharded sweep: download every shard's ``done.json`` and audio under
    ``prefix`` into ``out`` (layout of a local sweep, prefix stripped), mark them as
    uploaded, and rebuild the full manifest. Returns the number of rows."""
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    from gransim.sweep.runner import write_manifest

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/") + "/"
    with tempfile.TemporaryDirectory(dir=out.parent if out.parent.exists() else None) as tmp:
        pats = [f"{prefix}runs/*/done.json"] + ([f"{prefix}audio/**"] if audio else [])
        snapshot_download(repo_id, repo_type="dataset", local_dir=tmp, allow_patterns=pats)
        src = Path(tmp) / prefix
        got = []
        for p in sorted(src.rglob("*")):
            if p.is_file():
                rel = p.relative_to(src)
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.move(str(p), dst)
                got.append(rel.as_posix())
    track = out / ".hf_uploaded.txt"
    have = set(track.read_text().split()) if track.exists() else set()
    with open(track, "a", encoding="utf-8") as fh:
        fh.write("".join(f + "\n" for f in got if f not in have))
    n = write_manifest(out)
    log(f"fetched {len(got)} files from {repo_id}/{prefix} → {out}; manifest {n} rows")
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
