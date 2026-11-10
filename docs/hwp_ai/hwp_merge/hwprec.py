"""HWP 5 레코드 단위 읽기·쓰기 (olefile로 읽고, 스트림 내용은 직접 파싱)."""
import struct
import zlib

import olefile

TAG = {16: "DOCUMENT_PROPERTIES", 17: "ID_MAPPINGS", 18: "BIN_DATA", 19: "FACE_NAME", 20: "BORDER_FILL",
       21: "CHAR_SHAPE", 22: "TAB_DEF", 23: "NUMBERING", 24: "BULLET", 25: "PARA_SHAPE", 26: "STYLE",
       66: "PARA_HEADER", 67: "PARA_TEXT", 68: "PARA_CHAR_SHAPE", 69: "PARA_LINE_SEG", 70: "PARA_RANGE_TAG",
       71: "CTRL_HEADER", 72: "LIST_HEADER", 73: "PAGE_DEF", 74: "FOOTNOTE_SHAPE", 75: "PAGE_BORDER_FILL",
       76: "SHAPE_COMPONENT", 77: "TABLE", 78: "SHAPE_COMPONENT_LINE", 79: "SHAPE_COMPONENT_RECTANGLE",
       85: "SHAPE_COMPONENT_PICTURE", 87: "CTRL_DATA", 88: "EQEDIT"}


class Rec:
    __slots__ = ("tag", "level", "data")

    def __init__(self, tag, level, data):
        self.tag, self.level, self.data = tag, level, bytes(data)

    def name(self):
        return TAG.get(self.tag, str(self.tag))

    def copy(self, level_shift=0):
        return Rec(self.tag, self.level + level_shift, self.data)

    def __repr__(self):
        return f"<{self.name()} L{self.level} {len(self.data)}B>"


def parse(buf):
    out, i = [], 0
    while i < len(buf):
        h = struct.unpack_from("<I", buf, i)[0]
        i += 4
        tag, level, size = h & 0x3FF, (h >> 10) & 0x3FF, h >> 20
        if size == 0xFFF:
            size = struct.unpack_from("<I", buf, i)[0]
            i += 4
        out.append(Rec(tag, level, buf[i:i + size]))
        i += size
    return out


def serialize(recs):
    b = bytearray()
    for r in recs:
        n = len(r.data)
        if n >= 0xFFF:
            b += struct.pack("<II", r.tag | (r.level << 10) | (0xFFF << 20), n)
        else:
            b += struct.pack("<I", r.tag | (r.level << 10) | (n << 20))
        b += r.data
    return bytes(b)


def inflate(b):
    return zlib.decompress(b, -15)


def deflate(b):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(b) + c.flush()


def read_streams(path):
    """{경로 문자열: 원래 바이트}와 압축 여부."""
    ole = olefile.OleFileIO(path)
    streams = {}
    for e in ole.listdir(streams=True, storages=False):
        streams["/".join(e)] = ole.openstream(e).read()
    fh = streams["FileHeader"]
    compressed = bool(struct.unpack_from("<I", fh, 36)[0] & 1)
    ole.close()
    return streams, compressed


def para_text(rec):
    """PARA_TEXT를 사람이 읽을 문자열로 (컨트롤 문자는 표시)."""
    d = rec.data
    s, i = [], 0
    while i < len(d):
        c = struct.unpack_from("<H", d, i)[0]
        if c < 32:
            if c in (0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31):  # char control (1 wchar)
                s.append({13: "¶", 10: "↵", 24: "-", 30: "~", 31: " "}.get(c, f"<{c}>"))
                i += 2
            else:  # inline/extended control: 8 wchars
                cid = d[i + 2:i + 6][::-1].decode("latin1")
                s.append(f"[{c}:{cid}]")
                i += 16
        else:
            s.append(chr(c))
            i += 2
    return "".join(s)


def para_header(rec):
    d = rec.data
    nchars, mask, ps = struct.unpack_from("<IIH", d, 0)
    style, divide = d[10], d[11]
    ncs, nrt, nls, inst = struct.unpack_from("<HHHI", d, 12)
    return dict(nchars=nchars & 0x7FFFFFFF, last=bool(nchars >> 31), mask=mask, ps=ps, style=style, divide=divide,
                ncs=ncs, nrt=nrt, nls=nls, inst=inst, rest=d[22:])


def paragraphs(recs, level=0, start=0, end=None):
    """level에서 시작하는 문단들의 (시작, 끝) 레코드 범위."""
    end = len(recs) if end is None else end
    idx = [i for i in range(start, end) if recs[i].tag == 66 and recs[i].level == level]
    out = []
    for k, i in enumerate(idx):
        j = idx[k + 1] if k + 1 < len(idx) else end
        # 끝: 같은 level 이하의 다른 레코드가 나오기 전까지
        e = i + 1
        while e < j and recs[e].level > level:
            e += 1
        out.append((i, e))
    return out


def char_shapes(rec):
    d = rec.data
    return [struct.unpack_from("<II", d, 8 * k) for k in range(len(d) // 8)]
