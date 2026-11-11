"""최종본 .hwp의 인공지능 절(Ⅱ-4, Ⅲ-5, Ⅳ-3, Ⅴ-3, Ⅵ 인공지능 결론, 참고문헌 [3]~)을 줄인 내용(content_short.py)으로 바꾼다.
물리 파트·표지·요약서 등 나머지 레코드는 그대로 둔다. 서식 번호는 최종본(한글에서 다시 저장되어 번호가 바뀜)의
물리 파트 문단에서 직접 찾아 쓴다. 요약문의 인공지능 문단 두 곳만 본문과 맞게 고친다.

    python3 shrink_ai.py <최종본.hwp> <content_short.py> <출력.hwp>
"""
import struct
import sys
from pathlib import Path

from hwpbuild import ROOT, TEXT_W, Doc, load_content, rng, split_inline
from hwprec import Rec, para_header, para_text, paragraphs


class Final(Doc):
    """서식 번호를 최종본의 물리 파트에서 찾는다."""

    def __init__(self, path):
        super().__init__(path)
        # 한글에서 다시 저장한 문서에는 줄 배치 정보(PARA_LINE_SEG)가 있어 템플릿 레코드 위치가 밀린다 → 빼고 쓴다
        self.t_eq = [r for r in self.t_eq if r.tag != 69]
        self.t_fig = [r for r in self.t_fig if r.tag != 69]
        self.t_tbl = [r for r in self.t_tbl if r.tag != 69]

    def style_of(self, prefix, nth=0):
        hits = []
        for a, e in paragraphs(self.sec):
            t = next((para_text(r) for r in self.sec[a:e] if r.tag == 67 and r.level == 1), "")
            if t.strip().startswith(prefix):
                hits.append(a)
        a = hits[nth]
        h = para_header(self.sec[a])
        cs = next(struct.unpack_from("<I", r.data, 4)[0] for r in self.sec[a + 1:a + 4] if r.tag == 68)
        return dict(ps=h["ps"], style=h["style"], cs=cs)

    def styles(self):
        S = {}
        S["II_top"] = self.style_of("3. 교차 거듭제곱 모형과 통계적 추정")
        S["II_sub"] = self.style_of("3.1. 교차 거듭제곱 모형")
        S["II_body"] = self.style_of("수직으로 흔들리는 과립층은")
        S["III_top"] = self.style_of("2. 시뮬레이션 환경 구성")
        S["III_sub"] = self.style_of("4.1. 문제 설정")
        S["III_body"] = self.style_of("같은 N이라도 입자의 지름")
        S["IV_top"] = self.style_of("2. 내부 물체를 모를 때")
        S["IV_sub"] = self.style_of("2.1. 시뮬레이션 피팅 결과")
        S["IV_body"] = self.style_of("실측 녹음 300개의 MAE는")
        S["V_sub"] = self.style_of("2. 내부 물체를 모를 때 큰 개수에서")
        S["V_body"] = self.style_of("시뮬레이션 안에서 평가해도")
        S["ref"] = self.style_of("[2] 윤형주")
        return S

    # 표: 셀 서식과 테두리를 표 8에서 읽는다
    def table(self, caption, head, rows, widths):
        T = self.t_tbl
        para_h, ptext, pcs, ctrl, cap_lh, cap_h, cap_t, cap_cs, trec = T[:9]
        cells = [k for k, r in enumerate(T) if r.tag == 72 and r.level == 2 and len(r.data) >= 34]
        ncol_t = struct.unpack_from("<H", trec.data, 6)[0]
        hc0, hcl, bc0, bcl = cells[0], cells[ncol_t - 1], cells[ncol_t], cells[2 * ncol_t - 1]

        def info(k):
            bf = struct.unpack_from("<H", T[k].data, 32)[0]
            h = para_header(T[k + 1])
            cs = next(struct.unpack_from("<I", r.data, 4)[0] for r in T[k + 2:k + 5] if r.tag == 68)
            return bf, h["ps"], h["style"], cs
        H0, HL, B0, BL = info(hc0), info(hcl), info(bc0), info(bcl)
        tot = sum(widths)
        ws = [int(TEXT_W * w / tot) for w in widths]
        ws[-1] = TEXT_W - sum(ws[:-1])
        nrow, ncol = 1 + len(rows), len(head)
        HH, HB = 1946, 1698
        out = [self._hdr(para_h, 9, 1 << 11, para_header(para_h)["ps"], para_header(para_h)["style"]),
               ptext.copy(), pcs.copy()]
        c = bytearray(ctrl.data)
        struct.pack_into("<II", c, 16, TEXT_W, HH + HB * len(rows))
        struct.pack_into("<I", c, 36, rng.getrandbits(31))
        out.append(Rec(71, 1, c))
        out.append(cap_lh.copy())
        out += self.para([caption], ps=para_header(cap_h)["ps"], style=para_header(cap_h)["style"],
                         cs=struct.unpack_from("<I", cap_cs.data, 4)[0], level=2, last=True)
        t = bytearray(trec.data[:18])
        struct.pack_into("<HH", t, 4, nrow, ncol)
        t += struct.pack("<" + "H" * nrow, *([ncol] * nrow))
        nrow_t = struct.unpack_from("<H", trec.data, 4)[0]
        t += trec.data[18 + 2 * nrow_t:]
        out.append(Rec(77, 2, t))
        for r_i, row in enumerate([head] + rows):
            hdr = r_i == 0
            for c_i, txt in enumerate(row):
                lh = bytearray(T[hc0 if hdr else bc0].data)
                lines = str(txt).split("\n")
                last_col = c_i == ncol - 1
                bf, ps, st, cs = (HL if last_col else H0) if hdr else (BL if last_col else B0)
                struct.pack_into("<H", lh, 0, len(lines))
                struct.pack_into("<HHHHII", lh, 8, c_i, r_i, 1, 1, ws[c_i], HH if hdr else HB)
                struct.pack_into("<H", lh, 32, bf)
                struct.pack_into("<I", lh, 34, ws[c_i])
                out.append(Rec(72, 2, lh))
                for k, line in enumerate(lines):
                    out += self.para([line], ps=ps, style=st, cs=cs, level=2, last=(k == len(lines) - 1))
        return out


def render(doc, S, blocks, group):
    out, prev_head = [], True
    body = S[{"II": "II_body", "III": "III_body", "IV": "IV_body"}.get(group, "V_body")]
    empty = lambda: doc.para([], ps=body["ps"], style=body["style"], cs=body["cs"])
    for b in blocks:
        k = b[0]
        if k == "h3":
            t = b[1]
            top = len(t.split(" ")[0].rstrip(".").split(".")) == 1
            if not prev_head:
                out += empty()
            if group == "II":
                s = S["II_top"] if top else S["II_sub"]
                out += doc.para([("  " if top else "    ") + t], **s)
            elif group in ("III", "IV"):
                s = S[f"{group}_top"] if top else S[f"{group}_sub"]
                out += doc.para([("  " + t) if top else t], **s)
            else:
                out += doc.para([t], **S["V_sub"])
            prev_head = True
            continue
        prev_head = False
        if k == "p":
            out += doc.para(split_inline(b[1]), **body)
        elif k == "eq":
            out += doc.equation(b[1], b[2])
        elif k == "table":
            _, cap, head, rows, widths = b
            out += doc.table(cap, head, rows, widths or [1] * len(head))
        elif k == "fig":
            _, cap, f = b
            out += doc.figure(ROOT / f, cap)
    return out


def top_paras(sec):
    out = []
    for a, e in paragraphs(sec):
        t = next((para_text(r) for r in sec[a:e] if r.tag == 67 and r.level == 1), "")
        out.append((a, e, t.replace("¶", "").strip(), para_header(sec[a])["nchars"]))
    return out


def section_range(sec, head, nxt, after=None):
    """head로 시작하는 문단부터, nxt 문단 앞의 빈 문단들 바로 앞까지 (레코드 위치)."""
    P = top_paras(sec)
    i0 = 0
    if after:
        i0 = next(i for i, x in enumerate(P) if x[2].startswith(after))
    s = next(i for i in range(i0, len(P)) if P[i][2].startswith(head))
    n = next(i for i in range(s + 1, len(P)) if P[i][2].startswith(nxt))
    e = n
    while e - 1 > s and P[e - 1][2] == "" and P[e - 1][3] == 1:
        e -= 1
    return P[s][0], P[e][0]


def set_para_text(recs, i, text):
    """문단(머리 위치 i)의 글을 바꾼다. 글자 모양은 처음 것 하나로, 줄 배치 정보는 지운다(한글이 다시 계산)."""
    e = i + 1
    while e < len(recs) and recs[e].level > recs[i].level:
        e += 1
    lv = recs[i].level
    kids = recs[i + 1:e]
    cs_rec = next(r for r in kids if r.tag == 68)
    cs0 = struct.unpack_from("<I", cs_rec.data, 4)[0]
    t = text.encode("utf-16le") + b"\x0d\x00"
    d = bytearray(recs[i].data)
    last = struct.unpack_from("<I", d, 0)[0] & 0x80000000
    struct.pack_into("<I", d, 0, (len(t) // 2) | last)
    struct.pack_into("<HHH", d, 12, 1, 0, 0)
    rest = [r for r in kids if r.tag not in (67, 68, 69, 70)]
    recs[i:e] = [Rec(66, lv, d), Rec(67, lv + 1, t), Rec(68, lv + 1, struct.pack("<II", 0, cs0))] + rest


SUMMARY_FIX = {
    "물성(지름, 밀도, 반발계수, 마찰계수, 탄성률)과 녹음 조건을 무작위로 바꾼 시뮬레이션 소리":
        "물성(지름, 밀도, 반발계수, 마찰계수, 탄성률)과 녹음 조건을 무작위로 바꾼 시뮬레이션 소리 4800개로, 스펙트로그램을 입력받아 개수와 물성을 불확도와 함께 "
        "추정하는 합성곱 신경망(CNN)을 학습하였다. 물질을 모르는 블라인드 모델은 실측 녹음에서 평균 오차 9.8개로, 같은 조건의 1차 연구 방식(28.1개)보다 65% "
        "정확했고 물질을 아는 모델(9.6개)과 비슷했다. 같은 개수 범위(5~100개)에서 물리 기반 블라인드 방법(16.9개)보다도 정확했으며(10.3개, 녹음 장치 특성은 "
        "1차 연구 녹음에 맞춘 조건), 차이는 55개 이상에서 컸다.",
    "정답 없는 실측 녹음으로 도메인 적대 학습을 하면":
        "정답 없는 실측 녹음으로 도메인 적대 학습을 하면 평균 오차가 7.8개로, 개수별 정답 녹음 5개를 더하면 6.8개로 줄었다. 90% 예측 구간은 실측에서 "
        "실제 개수를 89% 포함해 불확도도 믿을 만했다.",
}


def main(src, content, dst):
    doc = Final(src)
    S = doc.styles()
    G = load_content(content)

    # 쓰지 않게 되는 인공지능 그림(BIN0002~)을 문서에서 뺀다 (물리 그림 BIN0001만 남김)
    doc.di = [r for r in doc.di if not (r.tag == 18 and struct.unpack_from("<H", r.data, 2)[0] > 1)]
    doc.n_bins = 1
    doc.streams = [(p, b) for p, b in doc.streams if not (p[0] == "BinData" and p[1] != "BIN0001.png")]

    sec = doc.sec
    spots = []
    a, b = section_range(sec, "[3] S. S. Stevens", "【별첨】")
    rs = S["ref"]
    refs = []
    for i, (k, t) in enumerate(G["refs"]):
        refs += doc.para([f"[{3 + i}] " + t], **rs)
    spots.append((a, b, refs))
    a, b = section_range(sec, "물리 기반 방법은 물질을 모르면", "청소년 과학창의연구", after="Ⅵ. 결론")
    spots.append((a, b, render(doc, S, G["VI"], "VI")))
    a, b = section_range(sec, "3. 인공지능을 이용한 추정", "Ⅵ. 결론", after="Ⅴ. 고찰")
    spots.append((a, b, render(doc, S, G["V"], "V")))
    a, b = section_range(sec, "3. 인공지능을 이용한 추정", "Ⅴ. 고찰", after="Ⅳ. 연구 결과")
    spots.append((a, b, render(doc, S, G["IV"], "IV")))
    a, b = section_range(sec, "5. 인공지능을 이용한 추정 방법론", "Ⅳ. 연구 결과")
    spots.append((a, b, render(doc, S, G["III"], "III")))
    a, b = section_range(sec, "4. 인공지능 관련 이론", "Ⅲ. 연구 방법 및 절차")
    spots.append((a, b, render(doc, S, G["II"], "II")))
    for a, b, recs in sorted(spots, key=lambda x: -x[0]):
        sec[a:b] = recs

    # 요약문의 인공지능 문단: 본문에서 뺀 내용(내부 표현 분석)을 빼고, 학습 데이터 수를 본문과 맞춤
    for prefix, new in SUMMARY_FIX.items():
        hits = [i for i, r in enumerate(sec) if r.tag == 66 and i + 1 < len(sec) and sec[i + 1].tag == 67
                and para_text(sec[i + 1]).startswith(prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        set_para_text(sec, hits[0], new)

    # 남은 그림이 없는 BinData를 가리키지 않는지 확인
    for r in sec:
        if r.tag == 85:
            bid = struct.unpack_from("<H", r.data, 71)[0]
            assert bid <= doc.n_bins + len(doc.new_bins), bid
    doc.save(dst)
    print("saved", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
