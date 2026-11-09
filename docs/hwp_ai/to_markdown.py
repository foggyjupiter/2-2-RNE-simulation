"""content.py → Markdown (워드 사본용, pandoc으로 .docx 변환)."""
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("content", HERE / "content.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
L, nt, nq = [], 0, 0
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
        _, cap, f, what = b; L += [f"> [그림 자리] 넣을 파일: `{f}`  ", f"> 내용: {what}", "", f"**{cap}**", ""]
    elif k == "eq":
        nq += 1; L += [f"**[수식 {nq}] {b[1]}** — 한글 수식 편집기 입력: `{b[2]}`", ""]
Path(sys.argv[1]).write_text("\n".join(L), encoding="utf-8")
