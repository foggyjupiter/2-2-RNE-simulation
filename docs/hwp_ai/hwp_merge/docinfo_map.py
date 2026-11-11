"""두 HWP의 문서 정보(DocInfo) 사이에서 서식 번호를 내용으로 대응시킨다.

글꼴(FACE_NAME)은 언어별 묶음 안의 순번, 글자 모양은 글꼴 번호 7개를 대응시킨 뒤 바이트 비교,
문단 모양은 바이트 비교(탭·번호·테두리 정의가 같은지 함께 확인), 스타일은 이름으로 대응시킨다.
대응이 없으면 오른쪽 정의를 왼쪽 끝에 추가한다(글꼴은 해당 언어 묶음 끝에 삽입).
"""
import struct

from hwprec import Rec

LANGS = 7


class DI:
    def __init__(self, recs):
        self.recs = list(recs)

    def idm(self):
        r = next(r for r in self.recs if r.tag == 17)
        return list(struct.unpack_from("<" + "i" * (len(r.data) // 4), r.data, 0))

    def set_idm(self, vals):
        k = next(i for i, r in enumerate(self.recs) if r.tag == 17)
        d = bytearray(self.recs[k].data)
        struct.pack_into("<" + "i" * len(vals), d, 0, *vals)
        self.recs[k] = Rec(17, self.recs[k].level, d)

    def faces(self):
        """언어별 글꼴 레코드 목록 [(레코드 위치, 바이트)...] × 7"""
        counts = self.idm()[1:1 + LANGS]
        pos = [i for i, r in enumerate(self.recs) if r.tag == 19]
        out, k = [], 0
        for c in counts:
            out.append([(pos[k + j], self.recs[pos[k + j]].data) for j in range(c)])
            k += c
        return out

    def list(self, tag):
        return [(i, r.data) for i, r in enumerate(self.recs) if r.tag == tag]


def face_name(d):
    n = struct.unpack_from("<H", d, 1)[0]
    return d[3:3 + 2 * n].decode("utf-16le")


def style_name(d):
    n = struct.unpack_from("<H", d, 0)[0]
    return d[2:2 + 2 * n].decode("utf-16le")


def style_ids(d):
    n = struct.unpack_from("<H", d, 0)[0]
    o = 2 + 2 * n
    m = struct.unpack_from("<H", d, o)[0]
    o += 2 + 2 * m
    # property(1), next style(1), lang(2), para shape(2), char shape(2)
    prop, nxt, lang, ps, cs = struct.unpack_from("<BBhHH", d, o)
    return o, nxt, ps, cs


class Mapper:
    """오른쪽(src) 번호 → 왼쪽(dst) 번호. 없으면 dst에 추가."""

    def __init__(self, src_recs, dst_recs):
        self.S, self.D = DI(src_recs), DI(dst_recs)
        self.face = {}
        self.cs, self.ps, self.st, self.bf, self.tab, self.num = {}, {}, {}, {}, {}, {}
        self.added = []

    # ---- 테두리·탭·번호: 같은 순번이 같은 내용인지 확인하고, 다르면 추가
    def _simple(self, tag, cache, idx, one_based):
        if idx in cache:
            return cache[idx]
        S, D = self.S.list(tag), self.D.list(tag)
        k = idx - 1 if one_based else idx
        if k < 0:
            cache[idx] = idx
            return idx
        data = S[k][1]
        for j, (_, dd) in enumerate(D):
            if dd == data:
                cache[idx] = j + 1 if one_based else j
                return cache[idx]
        # 추가
        last = D[-1][0]
        self.D.recs.insert(last + 1, Rec(tag, self.D.recs[last].level, data))
        self.added.append((tag, idx))
        cache[idx] = len(D) + 1 if one_based else len(D)
        self._bump(tag)
        return cache[idx]

    def _bump(self, tag):
        order = {20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14}
        v = self.D.idm()
        v[order[tag]] += 1
        self.D.set_idm(v)

    def borderfill(self, i):   # 1부터
        return self._simple(20, self.bf, i, True)

    def tabdef(self, i):       # 0부터
        return self._simple(22, self.tab, i, False)

    def numbering(self, i):    # 1부터 (0 = 없음)
        if i == 0:
            return 0
        return self._simple(23, self.num, i, True)

    # ---- 글꼴
    def face_id(self, lang, i):
        key = (lang, i)
        if key in self.face:
            return self.face[key]
        sf, df = self.S.faces(), self.D.faces()
        data = sf[lang][i][1]
        nm = face_name(data)
        for j, (_, dd) in enumerate(df[lang]):
            if face_name(dd) == nm:          # 글꼴 정보(형태 정보 10바이트)는 저장할 때마다 달라질 수 있어 이름으로
                self.face[key] = j
                return j
        last = df[lang][-1][0]
        self.D.recs.insert(last + 1, Rec(19, self.D.recs[last].level, data))
        v = self.D.idm()
        v[1 + lang] += 1
        self.D.set_idm(v)
        self.added.append(("face", key))
        self.face[key] = len(df[lang])
        return self.face[key]

    # ---- 글자 모양
    def charshape(self, i):
        if i in self.cs:
            return self.cs[i]
        d = bytearray(self.S.list(21)[i][1])
        fids = struct.unpack_from("<7H", d, 0)
        struct.pack_into("<7H", d, 0, *[self.face_id(l, f) for l, f in enumerate(fids)])
        if len(d) >= 70:
            bf = struct.unpack_from("<H", d, 68)[0]
            if bf:
                struct.pack_into("<H", d, 68, self.borderfill(bf))
        d = bytes(d)
        for j, (_, dd) in enumerate(self.D.list(21)):
            if dd == d:
                self.cs[i] = j
                return j
        D = self.D.list(21)
        last = D[-1][0]
        self.D.recs.insert(last + 1, Rec(21, self.D.recs[last].level, d))
        self._bump(21)
        self.added.append(("charshape", i))
        self.cs[i] = len(D)
        return self.cs[i]

    # ---- 문단 모양
    def parashape(self, i):
        if i in self.ps:
            return self.ps[i]
        d = bytearray(self.S.list(25)[i][1])
        # property1(4) 왼(4) 오(4) 들여(4) 위(4) 아래(4) 줄간격(4) tabdef(2) 번호/글머리(2) 테두리(2) ...
        tab, num, bf = struct.unpack_from("<HHH", d, 28)
        head = struct.unpack_from("<I", d, 0)[0]
        head_type = (head >> 23) & 3          # 0 없음, 1 개요, 2 번호, 3 글머리
        struct.pack_into("<H", d, 28, self.tabdef(tab))
        if head_type == 2:
            struct.pack_into("<H", d, 30, self.numbering(num))
        elif head_type == 3 and num:
            raise NotImplementedError("글머리표 대응은 필요할 때 구현")
        if bf:
            struct.pack_into("<H", d, 32, self.borderfill(bf))
        d = bytes(d)
        for j, (_, dd) in enumerate(self.D.list(25)):
            if dd == d:
                self.ps[i] = j
                return j
        D = self.D.list(25)
        last = D[-1][0]
        self.D.recs.insert(last + 1, Rec(25, self.D.recs[last].level, d))
        self._bump(25)
        self.added.append(("parashape", i))
        self.ps[i] = len(D)
        return self.ps[i]

    # ---- 스타일: 이름으로
    def style(self, i):
        if i in self.st:
            return self.st[i]
        name = style_name(self.S.list(26)[i][1])
        for j, (_, dd) in enumerate(self.D.list(26)):
            if style_name(dd) == name:
                self.st[i] = j
                return j
        raise KeyError(f"스타일 '{name}'이 왼쪽 문서에 없음")
