"""Render solution_doc.md to solution_doc.pdf.

solution_doc.md is the single source. The previous generator hardcoded its own
copy of every sentence, so the PDF that gets submitted carried claims the
markdown had already corrected. This one reads the markdown and nothing else.

    python scripts/generate_solution_pdf.py            # final: refuses unfilled [[REMEASURE]] markers, enforces 2 pages
    python scripts/generate_solution_pdf.py --draft    # renders markers in red, page limit is a warning
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
MARK = "[[REMEASURE]]"
INK, MUTED, RULE, HEAD_BG = (colors.HexColor(c) for c in ("#0f172a", "#475569", "#cbd5e1", "#e2e8f0"))

BASE = dict(fontName="Helvetica", textColor=INK)
STYLES = {
    "title": ParagraphStyle("t", **{**BASE, "fontName": "Helvetica-Bold"}, fontSize=14, leading=17, spaceAfter=1),
    "sub": ParagraphStyle("s", **{**BASE, "textColor": MUTED}, fontSize=8, leading=10, spaceAfter=4),
    "h1": ParagraphStyle("h", **{**BASE, "fontName": "Helvetica-Bold"}, fontSize=10, leading=12, spaceBefore=5, spaceAfter=2),
    "body": ParagraphStyle("b", **BASE, fontSize=8.1, leading=10.1, spaceAfter=2),
    "bullet": ParagraphStyle("u", **BASE, fontSize=8.1, leading=10.1, leftIndent=9, bulletIndent=1, spaceAfter=1),
    "cell": ParagraphStyle("c", **BASE, fontSize=7.6, leading=9.2),
    "cellh": ParagraphStyle("ch", **{**BASE, "fontName": "Helvetica-Bold"}, fontSize=7.6, leading=9.2),
}


def inline(text: str, draft: bool) -> str:
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font name="Courier" size="7.4">\1</font>', t)
    if draft:
        t = t.replace(MARK, '<font color="#dc2626"><b>[[REMEASURE]]</b></font>')
    return t


def table(rows, draft):
    data = [[Paragraph(inline(c, draft), STYLES["cellh" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return t


def flow(md: str, draft: bool):
    out, rows, lines = [], [], md.splitlines()
    for i, raw in enumerate(lines + [""]):
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                rows.append(cells)
            continue
        if rows:
            out += [table(rows, draft), Spacer(1, 3)]
            rows = []
        if not line:
            continue
        if line.startswith("# "):
            out.append(Paragraph(inline(line[2:], draft), STYLES["title"]))
        elif line.startswith("## "):
            out.append(Paragraph(inline(line[3:], draft), STYLES["h1"]))
        elif line.startswith("- "):
            out.append(Paragraph(inline(line[2:], draft), STYLES["bullet"], bulletText="•"))
        elif i == 1:
            out += [Paragraph(inline(line, draft), STYLES["sub"]), HRFlowable(width="100%", thickness=0.6, color=RULE)]
        else:
            out.append(Paragraph(inline(line, draft), STYLES["body"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "solution_doc.md"))
    ap.add_argument("--out", default=str(ROOT / "solution_doc.pdf"))
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()

    md = Path(a.src).read_text(encoding="utf-8")
    left = md.count(MARK)
    if left and not a.draft:
        print(f"REFUSING: {left} {MARK} markers left in {a.src}. Fill each from the command beside it, or pass --draft.", file=sys.stderr)
        return 2

    doc = SimpleDocTemplate(a.out, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=30, bottomMargin=30,
                            title="Promo Finder: Solution Document")
    doc.build(flow(md, a.draft))

    import pymupdf

    pages = pymupdf.open(a.out).page_count
    print(f"wrote {a.out}: {pages} page(s), {left} unfilled marker(s)")
    if pages > a.max_pages:
        print(f"{'WARNING' if a.draft else 'REFUSING'}: {pages} pages exceeds --max-pages {a.max_pages}", file=sys.stderr)
        return 0 if a.draft else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
