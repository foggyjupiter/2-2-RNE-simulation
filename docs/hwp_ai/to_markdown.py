"""content.py → Markdown (워드 사본용). docs/hwp_ai에서 실행한 뒤 pandoc으로 .docx 변환.
그림은 저장소 results/ 경로로 넣고, 수식은 TeX로 넣는다(pandoc이 Word 수식으로 바꾼다)."""
import importlib.util, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("content", HERE / os.environ.get("CONTENT", "content.py")); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
L, nq = [], 0
nt = getattr(c, "TABLE_START", 1) - 1
cell = lambda s: s.replace("|", "/").replace("\n", "<br>")
for b in c.B:
    k = b[0]
    if k == "title": L += [f"# {b[1]}", ""]
    elif k == "h1": L += [f"## {b[1]}", ""]
    elif k == "h2": L += [f"### {b[1]}", ""]
    elif k == "h3": L += [f"#### {b[1]}", ""]
    elif k in ("p", "ref"): L += [b[1], ""]
    elif k == "note": L += [f"*{b[1]}*", ""]
    elif k == "blank": pass
    elif k == "table":
        nt += 1; _, cap, head, rows, _w = b
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows] + ["", f"**표 {nt}. {cap}**", ""]
    elif k == "kv":
        _, cap, rows = b
        L += [f"**■ {cap}**", "", "| 항목 | 내용 |", "|---|---|"] + [f"| {cell(a)} | {cell(t)} |" for a, t in rows] + [""]
    elif k == "fig":
        _, cap, f, what = b; L += [f"![{cap}](../../{f}){{width=95%}}", ""]
    elif k == "eq":
        L += ["$$" + b[1] + "$$", ""]
Path(sys.argv[1]).write_text("\n".join(L), encoding="utf-8")
