"""상자 7×7×5 데이터: 몫이 다 끝난 데이터셋을 받아 전체 목록을 만들고, 학습 세션이 파일 하나로
받을 수 있게 tar로 묶어 올린다 (Hugging Face API 한도를 아끼기 위해).

API 한도(계정당 5분에 1000회)를 아끼는 방법:
  - 완료 확인: 완료 표시 파일 경로들을 get_paths_info 한 번으로 확인 (전체 목록 받지 않음)
  - 대체 확인(완료 표시 업로드가 빠졌을 때): <prefix>/runs 폴더 한 단계만 나열해 실행 수를 셈
  - 내려받기: <prefix> 하위만 나열한 뒤 파일별로 받음 (전체 저장소 목록을 받지 않음)
  - 학습 세션은 box7_pack/<이름>.tar 하나만 받음

디스크가 작으므로 데이터셋을 하나씩 처리하고, tar에 넣은 파일은 바로 지운다.

    python scripts/box7_pack.py            (끝날 때까지 돌며 datasets/box7_pack_status.log에 기록)
"""
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gransim.sweep.runner import write_manifest  # noqa: E402

REPO = "FoggyJupiter/rne-granular-acoustics"
D = ROOT / "datasets"
STATUS = D / "box7_pack_status.log"
# (이름, 몫 수, 실행 수, 설정 파일) — 작은 것부터 (먼저 끝나고, 디스크도 적게 씀)
SETS = [("bb_fixed", 4, 300, "sweep_bb_fixed.yaml"), ("lowe", 3, 300, "sweep_lowe.yaml"),
        ("v3_nosig", 12, 1200, "sweep_main.yaml"), ("multi_rpm_nosig", 15, 1800, "sweep_multi_rpm.yaml")]
api = HfApi()


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def retry(fn, *a, tries=8, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except HfHubHTTPError as e:
            if i == tries - 1:
                raise
            wait = 30 * (i + 1) if "429" in str(e) else 10 * (i + 1)
            log(f"  HF 오류, {wait}s 뒤 재시도: {str(e)[:120]}")
            time.sleep(wait)


def complete(name, n, total, check_runs):
    prefix = f"box7/{name}"
    want = [f"{prefix}/_shards/shard_{k}_of_{n}.json" for k in range(n)]
    have = {p.path for p in retry(api.get_paths_info, REPO, want, repo_type="dataset")}
    if len(have) == n:
        return True, f"완료 표시 {n}/{n}"
    if check_runs:
        runs = sum(1 for _ in retry(api.list_repo_tree, REPO, path_in_repo=f"{prefix}/runs", repo_type="dataset"))
        if runs >= total:
            return True, f"완료 표시 {len(have)}/{n}, 실행 {runs}/{total} (실행 수로 완료 판정)"
        return False, f"완료 표시 {len(have)}/{n}, 실행 {runs}/{total}"
    return False, f"완료 표시 {len(have)}/{n}"


def pack(name, spec):
    prefix = f"box7/{name}"
    out = D / f"box7pack_{name}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "configs" / "box7" / spec, out / "sweep_spec.yaml")
    files = [f.path for f in retry(api.list_repo_tree, REPO, path_in_repo=prefix, recursive=True, repo_type="dataset")
             if hasattr(f, "size") and (f.path.endswith("/done.json") or "/audio/" in f.path)]
    # 완료 표시가 다 있어도 업로드가 한도에 걸려 빠진 실행이 있을 수 있음 → 모자라면 묶지 않고 빠진 몫을 기록
    have_runs = {p.split("/")[-2] for p in files if p.endswith("/done.json")}
    total = next(t for s, _n, t, _sp in SETS if s == name)
    if len(have_runs) < total:
        from gransim.sweep.runner import load_spec
        from gransim.sweep.sampling import build_jobs
        n = next(_n for s, _n, t, _sp in SETS if s == name)
        jobs = build_jobs(load_spec(ROOT / "configs" / "box7" / spec))
        miss = sorted({i % n for i, j in enumerate(jobs) if j.run_id not in have_runs})
        log(f"{name}: 실행 {len(have_runs)}/{total}만 올라옴 — 빠진 몫 {miss} (해당 세션 재업로드 또는 RESUME=1 대체 필요)")
        raise RuntimeError("incomplete upload")
    log(f"{name}: 받을 파일 {len(files)}개")

    def get(path):
        rel = path[len(prefix) + 1:]
        if (out / rel).exists():
            return
        p = retry(hf_hub_download, REPO, path, repo_type="dataset", local_dir=str(out / "_dl"))
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, out / rel)

    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(get, files), 1):
            if i % 1000 == 0:
                log(f"{name}: {i}/{len(files)} 받음")
    shutil.rmtree(out / "_dl", ignore_errors=True)
    n_rows = write_manifest(out)
    (out / "metadata.csv").write_bytes((out / "manifest.csv").read_bytes())
    log(f"{name}: 목록 {n_rows}줄")
    # 전체 목록을 데이터셋 폴더에도 올림 (기록용)
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=f"{prefix}/{m}", path_or_fileobj=str(out / m))
           for m in ("metadata.csv", "manifest.csv", "manifest.jsonl", "sweep_spec.yaml") if (out / m).exists()]
    retry(api.create_commit, REPO, ops, repo_type="dataset", commit_message=f"{prefix}: merged manifest ({n_rows} clips)")
    # tar: 목록 + 음성 (넣은 음성은 바로 지움)
    tar_path = D / f"box7pack_{name}.tar"
    with tarfile.open(tar_path, "w") as tf:
        for m in ("metadata.csv", "manifest.csv", "manifest.jsonl", "sweep_spec.yaml"):
            tf.add(out / m, arcname=m)
        for wav in sorted((out / "audio").rglob("*.wav")):
            tf.add(wav, arcname=wav.relative_to(out).as_posix())
            wav.unlink()
    log(f"{name}: tar {tar_path.stat().st_size / 1e9:.2f} GB, 올리는 중")
    retry(api.upload_file, path_or_fileobj=str(tar_path), path_in_repo=f"box7_pack/{name}.tar", repo_id=REPO,
          repo_type="dataset", commit_message=f"box7 pack {name} ({n_rows} clips)")
    tar_path.unlink()
    shutil.rmtree(out / "audio", ignore_errors=True)
    log(f"PACKED {name} ({n_rows} clips)")


def main():
    todo = list(SETS)
    polls = 0
    while todo:
        polls += 1
        for item in list(todo):
            name, n, total, spec = item
            try:
                ok, msg = complete(name, n, total, check_runs=(polls % 3 == 0))
            except Exception as e:  # noqa: BLE001
                log(f"{name}: 확인 실패 {str(e)[:120]}")
                continue
            log(f"{name}: {msg}")
            if ok:
                try:
                    pack(name, spec)
                    todo.remove(item)
                except Exception as e:  # noqa: BLE001
                    log(f"{name}: 묶기 실패, 다음에 다시 {str(e)[:200]}")
        if todo:
            time.sleep(180)
    log("ALL PACKED")


if __name__ == "__main__":
    main()
