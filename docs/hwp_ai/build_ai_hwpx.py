"""content.py → 인공지능 부분 작성본 .hwpx (참고 hwpx의 스타일·쪽 설정을 그대로 사용).

    python docs/hwp_ai/build_ai_hwpx.py <참고.hwpx> <출력.hwpx>

참고 hwpx에서 header.xml(글꼴·문단·글자·테두리 스타일), settings.xml, 첫 문단의 쪽 설정(secPr)을
가져오고, 본문(section0.xml)은 content.py의 블록으로 새로 만든다. 그림은 넣지 않고 자리(테두리 상자)와
캡션만 둔다. 한글에서 열어 .hwp로 다시 저장하면 된다.
"""
import importlib.util
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

HERE = Path(__file__).resolve().parent
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"

# 참고 파일 스타일 번호 (header.xml)
S = {
    "title": (6, 14), "h1": (7, 13), "h2": (7, 13), "h3": (11, 6), "p": (9, 2), "ref": (12, 2),
    "note": (13, 16), "cap": (4, 15), "eq_head": (4, 7), "eq_body": (4, 16),
    "cell": (13, 16), "cell_kv": (13, 15), "cell_head": (4, 7), "kv_cap": (13, 7),
}
BF_CELL, BF_HEAD, BF_TABLE = 9, 4, 9
TEXT_W = 41950                       # 표 폭 (본문 폭 42520 HWPUNIT 안쪽)

_pid = [1000000000]
_tid = [1700000000]
_z = [100]


def nid():
    _pid[0] += 1
    return _pid[0]


def run(text, cpr):
    return f'<hp:run charPrIDRef="{cpr}"><hp:t>{escape(text)}</hp:t></hp:run>'


def para(text, style, ppr_cpr=None, inner=None):
    ppr, cpr = ppr_cpr or S[style]
    body = inner if inner is not None else run(text, cpr)
    return (f'<hp:p id="{nid()}" paraPrIDRef="{ppr}" styleIDRef="0" pageBreak="0" '
            f'columnBreak="0" merged="0">{body}</hp:p>')


def est_lines(text, width, pt):
    """셀 안 글줄 수 추정: 한글 1자 ≈ 글자 크기, 영숫자 ≈ 0.55배."""
    w = 0.0
    for ch in text:
        w += pt * (1.0 if ord(ch) > 0x2E80 else 0.55)
    usable = max(width - 1020, 1000)
    return max(1, math.ceil(w / usable))


def cell(text, col, row, width, head=False, kv_text=False, span=(1, 1)):
    ppr, cpr = S["cell_head"] if head else (S["cell_kv"] if kv_text else S["cell"])
    pt = 1000 if (head or kv_text) else 900
    lines = text.split("\n") if text else [""]
    ps = "".join(para(l, None, (ppr, cpr)) for l in lines)
    n = sum(est_lines(l, width, pt) for l in lines)
    h = int(n * pt * 1.6 + 284)
    bf = BF_HEAD if head else BF_CELL
    xml = (f'<hp:tc name="" header="{1 if head else 0}" hasMargin="0" protect="0" editable="0" dirty="0" '
           f'borderFillIDRef="{bf}"><hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" '
           f'vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" textWidth="0" textHeight="0" '
           f'hasTextRef="0" hasNumRef="0">{ps}</hp:subList><hp:cellAddr colAddr="{col}" rowAddr="{row}"/>'
           f'<hp:cellSpan colSpan="{span[0]}" rowSpan="{span[1]}"/><hp:cellSz width="{width}" height="{{H}}"/>'
           f'<hp:cellMargin left="510" right="510" top="142" bottom="142"/></hp:tc>')
    return xml, h


def table_xml(rows, widths, head_rows=1, kv=False, caption=None):
    """rows: list of list of str. widths: 비율. head_rows: 머리글 행 수. kv: 첫 열을 머리글처럼."""
    tot = sum(widths)
    ws = [int(TEXT_W * w / tot) for w in widths]
    ws[-1] = TEXT_W - sum(ws[:-1])
    trs, total_h = [], 0
    for r, row in enumerate(rows):
        cells, hs = [], []
        for c, txt in enumerate(row):
            head = r < head_rows or (kv and c == 0)
            x, h = cell(txt, c, r, ws[c], head=head, kv_text=(kv and c > 0))
            cells.append(x)
            hs.append(h)
        rh = max(hs)
        total_h += rh
        trs.append("<hp:tr>" + "".join(x.replace("{H}", str(rh)) for x in cells) + "</hp:tr>")
    _tid[0] += 7
    _z[0] += 1
    cap = ""
    if caption:
        cap = (f'<hp:caption side="BOTTOM" fullSz="0" width="8504" gap="850" lastWidth="{TEXT_W}">'
               f'<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="TOP" linkListIDRef="0" '
               f'linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
               f'<hp:p id="{nid()}" paraPrIDRef="{S["cap"][0]}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
               f'<hp:run charPrIDRef="{S["cap"][1]}"><hp:t>표 </hp:t><hp:ctrl><hp:autoNum num="1" numType="TABLE">'
               f'<hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar="" supscript="0"/></hp:autoNum>'
               f'</hp:ctrl><hp:t>. {escape(caption)}</hp:t></hp:run></hp:p></hp:subList></hp:caption>')
    tbl = (f'<hp:tbl id="{_tid[0]}" zOrder="{_z[0]}" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" '
           f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" pageBreak="CELL" repeatHeader="1" '
           f'rowCnt="{len(rows)}" colCnt="{len(widths)}" cellSpacing="0" borderFillIDRef="{BF_TABLE}" noAdjust="0">'
           f'<hp:sz width="{TEXT_W}" widthRelTo="ABSOLUTE" height="{total_h}" heightRelTo="ABSOLUTE" protect="0"/>'
           f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
           f'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
           f'<hp:outMargin left="141" right="141" top="141" bottom="141"/>{cap}'
           f'<hp:inMargin left="510" right="510" top="141" bottom="141"/>{"".join(trs)}</hp:tbl>')
    return (f'<hp:p id="{nid()}" paraPrIDRef="{S["cap"][0]}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="{S["p"][1]}">{tbl}<hp:t/></hp:run></hp:p>')


def render(blocks):
    out, n_eq = [], 0
    for b in blocks:
        k = b[0]
        if k in ("title", "h1", "h2", "h3", "p", "ref", "note"):
            out.append(para(b[1], k))
        elif k == "blank":
            out.append(para("", "p"))
        elif k == "table":
            _, cap, head, rows, widths = b
            widths = widths or [1] * len(head)
            out.append(table_xml([head] + rows, widths, head_rows=1, caption=cap))
            out.append(para("", "p"))
        elif k == "kv":
            _, cap, rows = b
            out.append(para("■ " + cap, "kv_cap", S["kv_cap"]))
            out.append(table_xml([[a, t] for a, t in rows], [15, 85], head_rows=0, kv=True))
            out.append(para("", "p"))
        elif k == "fig":
            _, cap, file, what = b
            box = f"[그림 자리]  넣을 파일: {file}\n내용: {what}"
            out.append(table_xml([[box]], [1], head_rows=0))
            out.append(para(cap, "cap"))
            out.append(para("", "p"))
        elif k == "eq":
            n_eq += 1
            _, desc, script = b
            out.append(para(f"[수식 {n_eq}] {desc}", "eq_head"))
            out.append(para(f"한글 수식 편집기 입력: {script}", "eq_body"))
    return "".join(out)


def first_paragraph(ref_section: str) -> str:
    """참고 본문의 첫 문단에서 쪽 설정(secPr)·단 설정(ctrl)이 든 run만 남긴다."""
    root = etree.fromstring(ref_section.encode("utf-8"))
    p0 = root[0]
    for r in list(p0.findall(f"{{{HP}}}run")):
        if r.find(f"{{{HP}}}secPr") is None:
            p0.remove(r)
    for la in p0.findall(f"{{{HP}}}linesegarray"):
        p0.remove(la)
    s = etree.tostring(p0, encoding="unicode")
    return re.sub(r' xmlns:\w+="[^"]+"', "", s)


def preview_png(path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "WenQuanYi Zen Hei"})
    fig = plt.figure(figsize=(2.48, 3.51), dpi=100, facecolor="white")
    fig.text(0.5, 0.75, title, ha="center", va="center", fontsize=8, wrap=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(ref_hwpx: str, out_hwpx: str):
    spec = importlib.util.spec_from_file_location("content", HERE / "content.py")
    content = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(content)
    work = Path(out_hwpx).with_suffix(".build")
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile(ref_hwpx) as z:
        z.extractall(work)
    sec_path = work / "Contents" / "section0.xml"
    ref_sec = sec_path.read_text(encoding="utf-8")
    head = ref_sec[: ref_sec.index("<hp:p ")]           # xml 선언 + <hs:sec ...> (이름공간 포함)
    body = first_paragraph(ref_sec) + render(content.B)
    sec_path.write_text(head + body + "</hs:sec>", encoding="utf-8")
    # 그림 파일 제거, 매니페스트 정리
    shutil.rmtree(work / "BinData", ignore_errors=True)
    hpf = work / "Contents" / "content.hpf"
    s = hpf.read_text(encoding="utf-8")
    s = re.sub(r'<opf:item id="image\d+"[^>]*/>', "", s)
    s = re.sub(r"<opf:title>.*?</opf:title>", "<opf:title>R&amp;E 연구보고서 인공지능 부분</opf:title>", s)
    hpf.write_text(s, encoding="utf-8")
    title = "R&E 연구보고서 — 인공지능 부분 작성본"
    texts = [b[1] for b in content.B if b[0] in ("title", "h1", "h2", "p")][:12]
    (work / "Preview" / "PrvText.txt").write_text("\r\n".join(f"<{t[:120]}>" for t in texts), encoding="utf-8")
    preview_png(work / "Preview" / "PrvImage.png", title)
    # 검증: 본문 XML이 올바른지
    etree.fromstring((head + body + "</hs:sec>").encode("utf-8"))
    # zip: mimetype은 맨 앞, 무압축
    out = Path(out_hwpx)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(work / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        for f in sorted(work.rglob("*")):
            if f.is_file() and f.name != "mimetype":
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
