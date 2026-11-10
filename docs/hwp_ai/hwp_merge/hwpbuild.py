"""보내준 .hwp(물리 파트가 채워진 보고서)를 그대로 두고, 인공지능 파트를 같은 서식으로 끼워 넣는다.

새 문단·표·그림·수식은 문서 안에 이미 있는 물리 파트의 문단·표(표 8)·그림(그림 1)·수식 레코드를 복제해
글자 모양·문단 모양·테두리 번호를 그대로 쓴다. 원래 문서의 레코드는 하나도 지우거나 바꾸지 않는다
(문서 정보의 그림 개수와 BIN_DATA 항목만 추가).

    python3 hwpbuild.py <입력.hwp> <content_merge.py> <출력.hwp>
"""
import importlib.util
import random
import re
import struct
import sys
from pathlib import Path

import olefile

import cfb
from hwprec import Rec, deflate, inflate, para_header, para_text, paragraphs, parse, serialize

ROOT = Path("/home/user/2-2-RNE-simulation")
TEXT_W = 42980            # 물리 파트 전체 폭 표(표 6·8)의 폭
rng = random.Random(20261111)


# --------------------------------------------------------------------------- 수식 크기
def eq_size(latex, inline=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    t = re.sub(r"\\text\{([^}]*)\}", lambda m: r"\mathrm{" + "W" * len(m.group(1)) + "}", latex)
    t = re.sub(r"\\le(?![a-z])", r"\\leq", t)
    w, h, _d, _, _ = MathTextParser("path").parse(f"${t}$", dpi=72, prop=FontProperties(size=10))
    tall = r"\frac" in latex
    W = int(w * 100 * 1.15) + 200
    if tall:
        H, base = max(2438, int(h * 100 * 1.6)), 66
    else:
        H, base = (975 if inline else 1163), 86
    return W, H, base


# --------------------------------------------------------------------------- 문서
class Doc:
    def __init__(self, path):
        ole = olefile.OleFileIO(path)
        self.streams = [(tuple(e), ole.openstream(e).read()) for e in ole.listdir(streams=True, storages=False)]
        ole.close()
        get = dict(("/".join(p), b) for p, b in self.streams)
        self.sec = parse(inflate(get["BodyText/Section0"]))
        self.di = parse(inflate(get["DocInfo"]))
        self.new_bins = []                   # (id, png bytes)
        self.n_bins = struct.unpack_from("<I", next(r for r in self.di if r.tag == 17).data, 0)[0]
        self._templates()

    # 템플릿: 물리 파트의 실제 레코드
    def _find_para(self, prefix):
        for a, e in paragraphs(self.sec):
            t = next((para_text(r) for r in self.sec[a:e] if r.tag == 67 and r.level == 1), "")
            if t.startswith(prefix):
                return a, e
        raise KeyError(prefix)

    def _templates(self):
        s = self.sec
        self.t_body = s[self._find_para("희박 영역에서 세기, 첨도")[0]]       # 본문 문단 머리 (ps 7)
        a, e = self._find_para("[11:eqed]¶")                                   # 단독 수식 문단
        self.t_eq = s[a:e]
        a, e = self._find_para("[11:gso ]")                                    # 그림 1
        self.t_fig = s[a:e]
        # 표 8 (헤더 5칸 + 본문 5칸)
        a, e = None, None
        for a0, e0 in paragraphs(s):
            if any(r.tag == 67 and r.level == 3 and para_text(r).startswith("표 8.") for r in s[a0:e0]):
                a, e = a0, e0
        self.t_tbl = s[a:e]

    # ------------------------------------------------------------------ 문단 조각
    @staticmethod
    def _hdr(tpl, nchars, mask, ps, style, ncs=1, last=False, level=0):
        d = bytearray(tpl.data)
        struct.pack_into("<IIH", d, 0, nchars | (0x80000000 if last else 0), mask, ps)
        d[10] = style
        d[11] = 0
        struct.pack_into("<HHHI", d, 12, ncs, 0, 0, 0x80000000)
        return Rec(66, level, d)

    def para(self, parts, ps=7, style=1, cs=6, level=0, last=False):
        """parts: 문자열 또는 ('eq', latex, script) 의 목록. 빈 목록이면 빈 문단."""
        text = bytearray()
        ctrls = []
        mask = 0
        n = 0
        for part in parts:
            if isinstance(part, str):
                text += part.encode("utf-16le")
                n += len(part.encode("utf-16le")) // 2
            else:
                _, latex, script = part
                text += bytes.fromhex("0b006465716500000000000000000b00")
                n += 8
                mask |= 1 << 11
                ctrls += self._eq_ctrl(latex, script, level + 1, inline=True)
        if n == 0 and not ctrls:
            return [self._hdr(self.t_body, 1, 0, ps, style, level=level, last=last),
                    Rec(68, level + 1, struct.pack("<II", 0, cs))]
        text += b"\x0d\x00"
        n += 1
        return ([self._hdr(self.t_body, n, mask, ps, style, level=level, last=last),
                 Rec(67, level + 1, text), Rec(68, level + 1, struct.pack("<II", 0, cs))] + ctrls)

    def _eq_ctrl(self, latex, script, level, inline):
        ch, eq = self.t_eq[3], self.t_eq[4]
        W, H, base = eq_size(latex, inline)
        c = bytearray(ch.data)
        struct.pack_into("<II", c, 16, W, H)
        struct.pack_into("<I", c, 36, rng.getrandbits(31))
        p = bytearray(eq.data)
        n0 = struct.unpack_from("<H", p, 4)[0]
        tail = bytearray(p[6 + 2 * n0:])
        struct.pack_into("<h", tail, 8, base)
        sc = script.encode("utf-16le")
        e = p[:4] + struct.pack("<H", len(sc) // 2) + sc + bytes(tail)
        return [Rec(71, level, c), Rec(88, level + 1, e)]

    def equation(self, latex, script):
        """단독 수식 문단 (물리 파트 식 문단과 같은 모양)."""
        h = para_header(self.t_eq[0])
        return ([self._hdr(self.t_eq[0], 9, 1 << 11, h["ps"], h["style"]), self.t_eq[1].copy(), self.t_eq[2].copy()]
                + self._eq_ctrl(latex, script, 1, inline=False))

    # ------------------------------------------------------------------ 표
    def table(self, caption, head, rows, widths):
        T = self.t_tbl
        para_h, ptext, pcs, ctrl, cap_lh, cap_h, cap_t, cap_cs, trec = T[:9]
        cells = [k for k, r in enumerate(T) if r.tag == 72 and r.level == 2 and len(r.data) >= 34]
        head_lh, body_lh = T[cells[0]], T[cells[5]]
        tot = sum(widths)
        ws = [int(TEXT_W * w / tot) for w in widths]
        ws[-1] = TEXT_W - sum(ws[:-1])
        nrow, ncol = 1 + len(rows), len(head)
        HH, HB = 1946, 1698
        out = []
        out.append(self._hdr(para_h, 9, 1 << 11, para_header(para_h)["ps"], para_header(para_h)["style"]))
        out += [ptext.copy(), Rec(68, 1, struct.pack("<II", 0, 6))]
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
        t += trec.data[18 + 2 * 2:]                  # 테두리 번호 + 영역 정보 (표 8은 2행)
        out.append(Rec(77, 2, t))
        for r_i, row in enumerate([head] + rows):
            hdr = r_i == 0
            for c_i, txt in enumerate(row):
                lh = bytearray((head_lh if hdr else body_lh).data)
                lines = str(txt).split("\n")
                struct.pack_into("<H", lh, 0, len(lines))
                struct.pack_into("<HHHHII", lh, 8, c_i, r_i, 1, 1, ws[c_i], HH if hdr else HB)
                bf = (28 if c_i < ncol - 1 else 2) if hdr else (17 if c_i < ncol - 1 else 29)
                struct.pack_into("<H", lh, 32, bf)
                struct.pack_into("<I", lh, 34, ws[c_i])
                out.append(Rec(72, 2, lh))
                for k, line in enumerate(lines):
                    out += self.para([line], ps=48 if hdr else 65, style=26 if hdr else 0, cs=60 if hdr else 0,
                                     level=2, last=(k == len(lines) - 1))
        return out

    # ------------------------------------------------------------------ 그림
    def figure(self, png_path, caption):
        F = self.t_fig
        png = Path(png_path).read_bytes()
        pw, ph = struct.unpack(">II", png[16:24])
        self.n_bins += 1
        bid = self.n_bins
        self.new_bins.append((bid, png))
        dw, dh = pw * 75, ph * 75                       # 96 dpi 기준 원본 크기 (HWPUNIT)
        cw = TEXT_W
        chh = int(cw * ph / pw)
        out = [self._hdr(F[0], 9, 1 << 11, para_header(F[0])["ps"], para_header(F[0])["style"]),
               F[1].copy(), F[2].copy()]
        c = bytearray(F[3].data)
        struct.pack_into("<II", c, 16, cw, chh)
        struct.pack_into("<I", c, 36, rng.getrandbits(31))
        out.append(Rec(71, 1, c))
        out.append(F[4].copy())                              # 캡션 목록 머리
        cap_h = F[5]
        cap_cs = struct.unpack_from("<I", F[7].data, 4)[0]
        out += self.para([caption], ps=para_header(cap_h)["ps"], style=para_header(cap_h)["style"], cs=cap_cs,
                         level=2, last=True)
        sc = bytearray(F[8].data)
        struct.pack_into("<iiHHIIII", sc, 8, 0, 0, 0, 1, dw, dh, cw, chh)
        struct.pack_into("<Hii", sc, 40, 0, cw // 2, chh // 2)
        assert struct.unpack_from("<H", sc, 50)[0] == 1
        o = 52
        struct.pack_into("<6d", sc, o, 1, 0, 0, 0, 1, 0)
        struct.pack_into("<6d", sc, o + 48, cw / dw, 0, 0, 0, chh / dh, 0)
        struct.pack_into("<6d", sc, o + 96, 1, 0, 0, 0, 1, 0)
        out.append(Rec(76, 2, sc))
        pic = bytearray(F[9].data)
        struct.pack_into("<8i", pic, 12, 0, 0, dw, 0, dw, dh, 0, dh)
        struct.pack_into("<4i", pic, 44, 0, 0, dw, dh)
        struct.pack_into("<H", pic, 71, bid)
        struct.pack_into("<I", pic, 74, rng.getrandbits(31))
        struct.pack_into("<II", pic, 82, dw, dh)
        out.append(Rec(85, 3, pic))
        return out

    # ------------------------------------------------------------------ 삽입과 저장
    def insert(self, index, recs):
        self.sec[index:index] = recs

    def save(self, path):
        # 문서 정보: 그림 개수와 BIN_DATA 항목
        di = list(self.di)
        k_idm = next(i for i, r in enumerate(di) if r.tag == 17)
        idm = bytearray(di[k_idm].data)
        struct.pack_into("<I", idm, 0, self.n_bins)
        di[k_idm] = Rec(17, di[k_idm].level, idm)
        k_last_bin = max(i for i, r in enumerate(di) if r.tag == 18)
        tpl = di[k_last_bin]
        new = []
        for bid, _ in self.new_bins:
            d = bytearray(tpl.data)
            struct.pack_into("<H", d, 2, bid)
            new.append(Rec(18, tpl.level, d))
        di[k_last_bin + 1:k_last_bin + 1] = new
        ents = []
        for p, b in self.streams:
            key = "/".join(p)
            if key == "BodyText/Section0":
                b = deflate(serialize(self.sec))
            elif key == "DocInfo":
                b = deflate(serialize(di))
            ents.append((p, b))
        for bid, png in self.new_bins:
            ents.append((("BinData", f"BIN{bid:04X}.png"), deflate(png)))
        cfb.write(path, ents)


# --------------------------------------------------------------------------- 내용 → 레코드
INLINE = [("N̂", r"\hat{N}", "hat{N}"), ("σ̂", r"\hat{\sigma}", "hat{sigma}"),
          ("⟨a⁴⟩/⟨a²⟩²", r"\langle a^4\rangle/\langle a^2\rangle^2", "LEFT < a ^{4} RIGHT > / LEFT < a ^{2} RIGHT > ^{2}"),
          ("e⁹", r"e^9", "e ^{9}"), ("e_n", r"e_n", "e _{n}")]


def split_inline(text):
    parts, i = [], 0
    pat = re.compile("|".join(re.escape(k) for k, _, _ in INLINE))
    for m in pat.finditer(text):
        if m.start() > i:
            parts.append(text[i:m.start()])
        k = next(x for x in INLINE if x[0] == m.group(0))
        parts.append(("eq", k[1], k[2]))
        i = m.end()
    if i < len(text):
        parts.append(text[i:])
    return parts


def load_content(path):
    spec = importlib.util.spec_from_file_location("content_merge", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    groups, cur, tno = {}, None, m.TABLE_START
    for b in m.B:
        k = b[0]
        if k == "h2":
            t = b[1]
            cur = ("intro" if t.startswith("Ⅰ") else "II" if t.startswith("Ⅱ") else "III" if t.startswith("Ⅲ") else "IV" if t.startswith("Ⅳ")
                   else "V" if t.startswith("Ⅴ") else "VI" if t.startswith("Ⅵ") else "refs")
            groups[cur] = []
            continue
        if cur is None or k in ("title", "note", "blank"):
            continue
        if k == "table":
            b = ("table", f"표 {tno}. {b[1]}", b[2], b[3], b[4])
            tno += 1
        if k == "fig":
            num, cap = re.match(r"그림\s*(\d+)\.\s*(.*)", b[1]).groups()
            b = ("fig", f"[그림 {num}] {cap}", b[2])
        groups[cur].append(b)
    return groups


def render(doc, blocks, group):
    out, prev_head = [], True
    cs = {"III": 115, "intro": 77}.get(group, 6)
    for b in blocks:
        k = b[0]
        if k == "h3":
            t = b[1]
            top = re.match(r"^\d+\.\s", t) is not None
            if not prev_head:
                out += doc.para([], cs=cs)
            if group == "II":
                out += doc.para([("  " if top else "    ") + t], ps=2, style=0, cs=46 if top else 6)
            elif top and group in ("III", "IV"):
                out += doc.para(["  " + t], ps=2, style=0, cs=55)
            else:
                out += doc.para([t], cs=cs)
            prev_head = True
            continue
        prev_head = False
        if k == "p":
            out += doc.para(split_inline(b[1]), cs=cs)
        elif k == "ref":
            out += doc.para([b[1]], cs=52)
        elif k == "eq":
            out += doc.equation(b[1], b[2])
        elif k == "table":
            _, cap, head, rows, widths = b
            out += doc.table(cap, head, rows, widths or [1] * len(head))
        elif k == "fig":
            _, cap, f = b
            out += doc.figure(ROOT / f, cap)
    return out


def main(src, content, dst):
    doc = Doc(src)
    G = load_content(content)
    s = doc.sec
    P = doc._find_para
    # 삽입 위치 (원래 문서 기준, 뒤에서부터 넣는다)
    spots = []
    a, e = P("[2] 윤형주")
    refs = [x[1] for x in G["refs"]]
    refs = [f"[{3 + i}] " + r for i, r in enumerate(refs)]
    spots.append((e, sum((doc.para([r], cs=52) for r in refs), [])))
    a, e = P("Ⅵ. 결론")
    a2, e2 = P("청소년 과학창의연구 학술지는")
    spots.append((a2, render(doc, G["VI"], "VI") + doc.para([], cs=6)))
    a, e = P("같은 클리핑 수준에서 비교하면")
    spots.append((e, doc.para([], cs=6) + render(doc, G["V"], "V")))
    a, e = P("Ⅴ. 고찰")
    # Ⅴ 앞의 빈 문단들 바로 앞 = 표 9 문단 끝
    k = a
    while k > 0 and para_header(s[[i for i, _ in paragraphs(s) if i < k][-1]])["nchars"] == 1:
        k = [i for i, _ in paragraphs(s) if i < k][-1]
    spots.append((k, doc.para([], cs=6) + render(doc, G["IV"], "IV")))
    a, e = P("Ⅳ. 연구 결과")
    spots.append((a, render(doc, G["III"], "III") + doc.para([], cs=115)))
    a, e = P("추정값: [11:eqed]가 최소인")
    spots.append((e, doc.para([], cs=6) + render(doc, G["II"], "II")))
    a, e = P("이를 위해 과립계 물리 기반")
    spots.append((e, doc.para([], cs=77) + render(doc, G["intro"], "intro")))
    for idx, recs in sorted(spots, key=lambda x: -x[0]):
        doc.insert(idx, recs)
    doc.save(dst)
    print("saved", dst, "bins", doc.n_bins)


if __name__ == "__main__":
    main(*sys.argv[1:4])
