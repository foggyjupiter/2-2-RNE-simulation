"""오른쪽 파일(요약서 물리 쪽 작성본)의 '연구결과 요약문'·'과제 수행 과정' 표를 왼쪽 파일(본문 최종본)에 옮기고,
인공지능 관련 문단을 같은 칸에 추가한다. 왼쪽 파일의 다른 레코드는 건드리지 않는다.

    python3 merge_summary.py <왼쪽.hwp> <오른쪽.hwp> <출력.hwp>
"""
import struct
import sys

import olefile

import cfb
from docinfo_map import Mapper
from hwprec import Rec, deflate, inflate, para_header, para_text, paragraphs, parse, serialize


def load(path):
    ole = olefile.OleFileIO(path)
    streams = [(tuple(e), ole.openstream(e).read()) for e in ole.listdir(streams=True, storages=False)]
    ole.close()
    get = dict(("/".join(p), b) for p, b in streams)
    return streams, parse(inflate(get["BodyText/Section0"])), parse(inflate(get["DocInfo"]))


def top_text(sec, a, e):
    return next((para_text(r) for r in sec[a:e] if r.tag == 67 and r.level == 1), "")


def find_table_para(sec, first_cell_text):
    """첫 칸의 글이 first_cell_text로 시작하는 표를 담은 최상위 문단 범위."""
    for a, e in paragraphs(sec):
        if "[11:tbl ]" not in top_text(sec, a, e):
            continue
        for i in range(a, e):
            if sec[i].tag == 67 and sec[i].level == 3 and para_text(sec[i]).startswith(first_cell_text):
                return a, e
    raise KeyError(first_cell_text)


# ----------------------------------------------------------------------------- 번호 대응
def remap(recs, M):
    out = []
    for r in recs:
        d = bytearray(r.data)
        if r.tag == 66:
            ps = struct.unpack_from("<H", d, 8)[0]
            struct.pack_into("<H", d, 8, M.parashape(ps))
            d[10] = M.style(d[10])
        elif r.tag == 68:
            for k in range(len(d) // 8):
                pos, cs = struct.unpack_from("<II", d, 8 * k)
                struct.pack_into("<II", d, 8 * k, pos, M.charshape(cs))
        elif r.tag == 72 and len(d) >= 34:
            bf = struct.unpack_from("<H", d, 32)[0]
            if bf:
                struct.pack_into("<H", d, 32, M.borderfill(bf))
        elif r.tag == 77:
            rows = struct.unpack_from("<H", d, 4)[0]
            o = 18 + 2 * rows
            bf = struct.unpack_from("<H", d, o)[0]
            if bf:
                struct.pack_into("<H", d, o, M.borderfill(bf))
            if len(d) >= o + 4:
                nz = struct.unpack_from("<H", d, o + 2)[0]
                for z in range(nz):
                    q = o + 4 + 10 * z + 8
                    zb = struct.unpack_from("<H", d, q)[0]
                    if zb:
                        struct.pack_into("<H", d, q, M.borderfill(zb))
        elif r.tag in (76, 85, 88):
            pass
        out.append(Rec(r.tag, r.level, d))
    return out


# ----------------------------------------------------------------------------- 칸 다루기
def para_end(recs, i):
    lv = recs[i].level
    j = i + 1
    while j < len(recs) and recs[j].level > lv:
        j += 1
    return j


def cell(recs, level, row, col, nth_table=0):
    """level의 칸(LIST_HEADER) 위치와 그 칸의 문단 머리 위치들. nth_table: 같은 level의 몇 번째 표인지."""
    tables = [i for i, r in enumerate(recs) if r.tag == 77 and r.level == level]
    t = tables[nth_table]
    for i in range(t + 1, len(recs)):
        r = recs[i]
        if r.level < level:
            break
        if r.tag == 72 and r.level == level and len(r.data) >= 34:
            c, rw = struct.unpack_from("<HH", r.data, 8)
            if (rw, c) == (row, col):
                paras, j = [], i + 1
                while j < len(recs) and recs[j].level >= level and not (recs[j].tag == 72 and recs[j].level == level) \
                        and not (recs[j].tag == 77 and recs[j].level == level):
                    if recs[j].tag == 66 and recs[j].level == level:
                        paras.append(j)
                    j += 1
                return i, paras
    raise KeyError((level, row, col))


def new_para(tpl_hdr, tpl_cs, text, level, last=False):
    d = bytearray(tpl_hdr.data)
    if text:
        t = text.encode("utf-16le") + b"\x0d\x00"
        n = len(t) // 2
    else:
        t, n = b"", 1
    struct.pack_into("<II", d, 0, n | (0x80000000 if last else 0), 0)
    struct.pack_into("<HHHI", d, 12, 1, 0, 0, 0x80000000)
    out = [Rec(66, level, d)]
    if t:
        out.append(Rec(67, level + 1, t))
    out.append(Rec(68, level + 1, struct.pack("<II", 0, tpl_cs)))
    return out


def insert_in_cell(recs, level, row, col, after_text, texts, nth_table=0, tpl_text=None):
    """칸 안에서 글이 after_text로 시작하는 문단 바로 뒤에 texts(빈 문자열 = 빈 줄)를 넣는다.
    after_text가 None이면 칸의 마지막 문단 뒤에 넣는다. 서식은 tpl_text 문단(없으면 after 문단)을 따른다."""
    lh, paras = cell(recs, level, row, col, nth_table)

    def ptext(i):
        j = i + 1
        return para_text(recs[j]) if recs[j].tag == 67 else ""
    if after_text is None:
        k = paras[-1]
    else:
        k = next(i for i in paras if ptext(i).startswith(after_text))
    tk = k if tpl_text is None else next(i for i in paras if ptext(i).startswith(tpl_text))
    tpl_hdr = recs[tk]
    tpl_cs = next(struct.unpack_from("<I", r.data, 4)[0] for r in recs[tk + 1:tk + 4] if r.tag == 68)
    is_last = k == paras[-1]
    new = []
    for m, t in enumerate(texts):
        new += new_para(tpl_hdr, tpl_cs, t, level, last=(is_last and m == len(texts) - 1))
    if is_last:   # 마지막 문단 표시를 새 끝 문단으로 옮김
        d = bytearray(recs[k].data)
        n = struct.unpack_from("<I", d, 0)[0] & 0x7FFFFFFF
        struct.pack_into("<I", d, 0, n)
        recs[k] = Rec(66, recs[k].level, d)
    e = para_end(recs, k)
    recs[e:e] = new
    d = bytearray(recs[lh].data)
    struct.pack_into("<H", d, 0, struct.unpack_from("<H", d, 0)[0] + len(texts))
    recs[lh] = Rec(72, recs[lh].level, d)


# ----------------------------------------------------------------------------- 인공지능 문단
AI_SUM = {
    "purpose": ["", "특히 내부 물체의 종류를 모르는 상황까지 다루기 위해, 과립계 물리 이론에 기반한 모형과 물성을 무작위로 바꾼 시뮬레이션으로 학습한 "
                    "인공지능 모형을 함께 개발하고, 같은 실측 녹음으로 두 방법을 비교한다."],
    "ai": ["",
           "물성(지름, 밀도, 반발계수, 마찰계수, 탄성률)과 녹음 조건을 무작위로 바꾼 시뮬레이션 소리 1만 2,600개로, 스펙트로그램을 입력받아 개수와 물성을 "
           "불확도와 함께 추정하는 합성곱 신경망(CNN)을 학습하였다. 물질을 모르는 블라인드 모델은 실측 녹음에서 평균 오차 9.8개로, 같은 조건의 1차 연구 방식"
           "(28.1개)보다 65% 정확했고 물질을 아는 모델(9.6개)과 비슷했다. 같은 개수 범위(5~100개)에서 물리 기반 블라인드 방법(16.9개)보다도 정확했으며"
           "(10.3개, 녹음 장치 특성은 1차 연구 녹음에 맞춘 조건), 차이는 55개 이상에서 컸다.",
           "",
           "정답 없는 실측 녹음으로 도메인 적대 학습을 하면 평균 오차가 7.8개로, 개수별 정답 녹음 5개를 더하면 6.8개로 줄었다. 90% 예측 구간은 실측에서 "
           "실제 개수를 89% 포함해 불확도도 믿을 만했다. 또 개수만 배운 모델이 내부에 고체·기체 같은 거동을 가르는 위상 집중도(R² 0.95)를 스스로 만들어, "
           "인공지능의 판단을 과립계 물리로 해석할 수 있었다."],
    "conclusion": ["인공지능 기법은 물질을 모를 때, 특히 개수가 많을 때 물리 기반 방법보다 정확했고, 물리 기반 방법은 물질을 알 때 적은 보정 녹음으로 더 정확하고 "
                   "해석이 분명했다. 두 방법 모두 큰 개수에서 과소추정이 남아, 그 원인이 시뮬레이터에 있음을 보였다."],
    "plan": ["시뮬레이터에 충돌 시점을 흩뜨리는 기작(가진기의 비정현 운동, 수평 진동)을 넣어 두 방법에 공통인 큰 개수의 과소추정을 줄이고, 녹음 장치 조건까지 "
             "넓게 무작위화해 장치를 모르는 경우로 인공지능 모형을 확장한다. 또 개수 구간에 따라 물리 기반 방법과 인공지능을 결합하는 추정을 시험하고, "
             "다른 입자와 여러 가진 속도의 실측 녹음으로 검증한다."],
}
AI_PROC = {
    "problems": ["",
                 "4. 인공지능이 시뮬레이션에서는 맞지만 실측에서는 틀리는 문제",
                 "문제: 시뮬레이션 소리로 학습한 모델이 시뮬레이션에서는 잘 맞아도 실측 녹음에서는 크게 틀렸고, 특히 적은 개수를 많게 세었다.",
                 "원인: 소리만 바꿔 다시 합성한 대조 실험으로, 실측 진폭 분포를 맞추려고 넣은 충돌별 세기 무작위화가 개수 단서(첨도, 변동계수)를 흐린 것이 "
                 "원인임을 찾았다.",
                 "해결: 세기 무작위화를 빼고, 개수를 모르는 실측 녹음으로 시뮬레이션과 실측의 차이를 줄이는 도메인 적대 학습을 적용하였다.",
                 "",
                 "5. 인공지능 학습용 시뮬레이터의 상자 치수 오류",
                 "문제: 보고서 작성 중, 인공지능 학습용 시뮬레이터의 상자가 실제 용기(7x7x5 cm)와 다른 5x5x7 cm로 설정되어 있었음을 발견하였다. "
                 "이 데이터로 학습한 모델의 실측 성능은 보고할 수 없었다.",
                 "해결: 치수와 음향 보정을 고친 뒤, 실측 검증에 필요한 데이터(동역학 계산 3,600회, 소리 1만 2,600개)와 핵심 실험 17개를 클라우드 병렬 세션 "
                 "50개로 하루 안에 다시 만들고 학습하였다. 실측 성능은 모두 고친 시뮬레이터로 다시 학습한 결과만 썼다."],
    "tool": ["Claude Code (인공지능 파트)"],
    "purpose": ["시뮬레이터·인공지능 코드 작성과 실험 실행 (인공지능 파트)"],
    "content": ["인공지능 파트에서는 Claude Code로 DEM 시뮬레이터와 학습 코드를 작성·검증하고, 클라우드 병렬 세션으로 데이터 생성과 학습 실험을 실행·정리하였다. "
                "보고서의 결과 수치는 모두 실험 결과 파일에서 직접 확인하였다."],
    "learning": ["시뮬레이션으로 만든 데이터로 신경망을 학습하고 실제 녹음에 적용하는 과정을 익혔다. 시뮬레이션에서 잘 맞는 모델이 실측에서 틀릴 수 있다는 것"
                 "(시뮬레이션-실측 간극)을 겪으며, 소리만 바꾼 대조 실험으로 원인을 찾는 법, 도메인 적대 학습, 불확도와 예측 구간, 교차검증 같은 평가 방법을 배웠다. "
                 "또 모델 내부를 들여다보는 선형 탐침으로 인공지능의 판단을 물리량으로 해석할 수 있음을 알게 되었다."],
}


def main(left, right, out):
    Ls, Lsec, Ldi = load(left)
    _, Rsec, Rdi = load(right)
    M = Mapper(Rdi, Ldi)

    # 오른쪽 표 두 개 (요약문, 과제 수행 과정)
    ra, rb = find_table_para(Rsec, "과 제 명")
    rc, rd = find_table_para(Rsec, "외부 전문가")
    sum_recs = remap(Rsec[ra:rb], M)
    proc_recs = remap(Rsec[rc:rd], M)

    # 인공지능 문단 추가 (요약문)
    insert_in_cell(sum_recs, 2, 1, 1, None, AI_SUM["purpose"])
    insert_in_cell(sum_recs, 2, 2, 1, "내부 물체를 모르는 상황에서는 물질의 물성", AI_SUM["ai"])
    insert_in_cell(sum_recs, 2, 2, 1, "큰 개수에서의", AI_SUM["conclusion"])
    _, p = cell(sum_recs, 2, 3, 1)
    assert len(p) == 1
    lh, paras = cell(sum_recs, 2, 3, 1)
    tpl_hdr = sum_recs[paras[0]]
    tpl_cs = next(struct.unpack_from("<I", r.data, 4)[0] for r in sum_recs[paras[0] + 1:paras[0] + 3] if r.tag == 68)
    new = []
    for t in AI_SUM["plan"]:
        new += new_para(tpl_hdr, tpl_cs, t, 2)
    sum_recs[paras[0]:paras[0]] = new                     # 빈 끝 문단 앞에
    d = bytearray(sum_recs[lh].data)
    struct.pack_into("<H", d, 0, struct.unpack_from("<H", d, 0)[0] + len(AI_SUM["plan"]))
    sum_recs[lh] = Rec(72, 2, d)

    # 인공지능 문단 추가 (과제 수행 과정)
    insert_in_cell(proc_recs, 2, 1, 1, "해결: 정해진 관측량 대신", AI_PROC["problems"])
    insert_in_cell(proc_recs, 4, 1, 2, None, AI_PROC["tool"])
    insert_in_cell(proc_recs, 4, 2, 2, None, AI_PROC["purpose"])
    insert_in_cell(proc_recs, 4, 3, 2, None, AI_PROC["content"])
    insert_in_cell(proc_recs, 2, 6, 1, "인공지능", AI_PROC["learning"], tpl_text="진동하는 과립계의 상 변화")

    # 왼쪽 문서에 끼우기 (뒤에서부터)
    la, lb = find_table_para(Lsec, "과 제 명")
    lc, ld = find_table_para(Lsec, "외부 전문가")
    assert lb <= lc
    new_sec = Lsec[:la] + sum_recs + Lsec[lb:lc] + proc_recs + Lsec[ld:]

    ents = []
    for p, b in Ls:
        key = "/".join(p)
        if key == "BodyText/Section0":
            b = deflate(serialize(new_sec))
        elif key == "DocInfo":
            b = deflate(serialize(M.D.recs))
        ents.append((p, b))
    cfb.write(out, ents)
    print("saved", out)
    print("문서 정보에 추가한 서식:", M.added)
    print("대응: 문단 모양", M.ps, "글자 모양", M.cs, "스타일", M.st)


if __name__ == "__main__":
    main(*sys.argv[1:4])
