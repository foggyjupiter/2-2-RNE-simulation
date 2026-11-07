"""docs/REPORT_DRAFT.md → docs/2차연구_보고서.docx (그림 삽입, 한글 글꼴 서식).

    python scripts/build_report_docx.py
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT_DRAFT.md"
BUILD = ROOT / "docs" / "report_build"
OUT = ROOT / "docs" / "2차연구_보고서.docx"
FONT = "Malgun Gothic"            # 맑은 고딕 (Windows·한글 기본)

# (이 줄로 시작하는 단락 뒤에 넣을 그림, 캡션)
FIGS = [
    ("**발견 1 — 실제 가진 속도는 녹음마다 다르다**", "results/figures/fig_shake_speed.png",
     "클립마다 소리에서 추정한 가진 주파수. 공칭 6.2 Hz와 달리 6.04~6.64 Hz이고 N = 95·100에서 빠르다."),
    ("**발견 2 — 입자끼리는 거의 탄성, 입자-벽은 손실이 크다**", "results/figures/fig_restitution_probe.png",
     "반발계수 설정별 실측과 합성음의 비교. 입자끼리 0.95, 입자-벽 0.7이 가장 잘 맞는다 (이후 4.4절에서 입자-벽 0.5로 갱신)."),
    ("**적은 입자: 벽 반발계수가 동기화를 정한다**", "results/figures/fig_sync_wall.png",
     "(가) 흔들리는 상자 속 공 하나: 벽 반발계수에 따른 동기화 R(충돌 위상이 한 곳에 몰린 정도)과 천장 충돌 비율. "
     "선은 1차원 이론, 점은 3차원 DEM. (나) 가진 세기 Γ와 벽 반발계수에 따른 동기화 지도 (1차원 이론). "
     "(다) 소리의 주기성: 실측 녹음과 벽 반발계수별 DEM. 0.5가 가장 잘 맞는다."),
    ("**상은 개수가 아니라 εF로 정해진다**", "results/figures/fig_phase_collapse.png",
     "시뮬레이션의 위상 집중도를 개수(왼쪽)와 εF(오른쪽)에 대해 그린 것. 색은 지름. 개수로는 지름마다 다른 곡선이지만 εF로는 한 곡선으로 모인다."),
    ("**에너지 수지로 본 들뜸**", "results/figures/fig_energy_balance.png",
     "(가) 입자의 들뜸 θ = v²/(Aω)²를 에너지 수지 이론의 손실 변수에 대해 그린 것 (색은 지름). "
     "(나) 같은 입자를 5가지 속도로 흔들 때 입자 운동에너지와 바닥 속도의 관계 (선 하나 = 시스템 하나)."),
    ("**음량은 개수보다 느리게 커진다**", "results/figures/fig_resolution.png",
     "(가) 실측 녹음의 소리 에너지와 개수: 적은 개수에서 N^0.86, 많은 개수에서 N^0.48. "
     "(나) 실측 녹음으로 가를 수 있는 개수 차이(δN)와 모델의 개수 구간별 오차. 회색은 입자 하나씩 튀는 영역 (F < 0.3)."),
    ("**보정한 소리가 실측에서 오히려 나빴던 원인**", "results/figures/fig_cause_sigma.png",
     "같은 동역학(v1)에 소리만 바꿔 학습한 블라인드 모델의 실측 오차. 충돌별 세기 퍼짐이 악화의 거의 전부를 만든다."),
    ("**원인을 뺀 주 데이터 (v3_nosig, 묶음 V)**", "results/figures/fig_nosig_real.png",
     "세기 퍼짐 있음/없음 데이터로 학습한 모델의 시뮬레이션·실측 개수 오차."),
    ("- **모델이 상(相)을 스스로 배웠다**", "results/ml/interpret/embedding_map_V4.png",
     "최종 모델(V4) 내부 표현의 주성분 지도. 개수(N) 방향으로 정렬되고, 지름·밀도·반발계수는 그 위에 섞여 있다."),
]


def preprocess(md: str) -> str:
    lines = md.splitlines()
    # 제목에서 '보고서 초안 — ' 빼기, 머리 인용 블록 빼기
    out, skip_quote = [], True
    for ln in lines:
        if ln.startswith("# 보고서 초안 — "):
            ln = "# " + ln[len("# 보고서 초안 — "):]
        if skip_quote and ln.startswith(">"):
            continue
        if ln.startswith("## 국문 초록"):
            skip_quote = False
        out.append(ln)
    # pandoc은 목록 앞에 빈 줄이 있어야 목록으로 읽음
    fixed = []
    for ln in out:
        is_item = bool(re.match(r"^\s*(-|\d+\.) ", ln))
        if is_item and fixed and fixed[-1].strip() and not re.match(r"^\s*(-|\d+\.) ", fixed[-1]) \
                and not fixed[-1].startswith("|") and not fixed[-1].startswith("  "):
            fixed.append("")
        fixed.append(ln)
    text = "\n".join(fixed)
    text = text.replace("## 5. 결론 및 논의 (초안)", "## 5. 결론 및 논의")
    text = text.replace("## 참고문헌 (초안 — 본문에서 쓰는 것만 남기고 형식 맞추기)", "## 참고문헌")
    # 그림 넣기 (단락 끝 다음 줄에)
    n, num = 0, {}
    paras = text.split("\n\n")
    res = []
    for p in paras:
        res.append(p)
        for key, path, cap in FIGS:
            if p.lstrip().startswith(key) or ("\n" + key) in p:
                n += 1
                res.append(f"![그림 {n}. {cap}]({ROOT / path}){{width=16cm}}")
                num[Path(path).name] = n
    text = "\n\n".join(res)
    # 본문의 그림 파일 경로를 그림 번호로
    for name, k in num.items():
        text = re.sub(r"그림 `(?:results/[\w/]*/)?" + re.escape(name) + "`", f"그림 {k}", text)
        text = re.sub(r"`(?:results/[\w/]*/)?" + re.escape(name) + "`", f"그림 {k}", text)
    return text


def reference_doc(path: Path):
    ref = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                         capture_output=True, check=True).stdout
    tmp = path.with_suffix(".tmp.docx")
    tmp.write_bytes(ref)
    with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                s = data.decode("utf-8")
                s = re.sub(r"<w:rFonts [^>]*/>",
                           f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>', s)
                # 표 글자를 조금 작게, 본문 줄 간격
                data = s.encode("utf-8")
            if item.filename.endswith("/"):
                continue
            zout.writestr(item, data)
    tmp.unlink()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    md = preprocess(SRC.read_text(encoding="utf-8"))
    (BUILD / "report.md").write_text(md, encoding="utf-8")
    ref = BUILD / "reference.docx"
    reference_doc(ref)
    subprocess.run(["pandoc", str(BUILD / "report.md"), "-f", "markdown+pipe_tables+implicit_figures",
                    "-o", str(OUT), "--reference-doc", str(ref), "--resource-path", str(ROOT)], check=True)
    print(OUT)


if __name__ == "__main__":
    main()
