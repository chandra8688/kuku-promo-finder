"""Last gate before submitting. Exit 0 only when nothing known-false or unfilled remains.

Checks the markdown sources and the text of the rendered solution_doc.pdf, since the
PDF is what gets read. Each banned phrase is a claim this repo once made that was
false for the shipped code.

    python scripts/check_submission.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "ARCHITECTURE.md", "solution_doc.md", "eval_report.md"]
BANNED = {
    "[[REMEASURE]]": "unfilled number",
    "STALE — do not quote": "stale evaluation report not regenerated",
    "k=3 sample self-consistency": "run.py annotates once",
    "Each episode is annotated k times": "run.py annotates once",
    "Under k=3 repeated sampling": "run.py annotates once",
    "15/15 (100": "circular precision@15 figure",
    "$0.15": "cost never measured",
    "100% deterministic": "LLM stages are not deterministic",
    "millisecond-accurate": "screenplay timings are centiseconds",
    "25/25 (100%) recall": "inconsistent dev recall figure",
    "12/25 (51%)": "12/25 is 48%",
    "Continuous Scene Constraint": "scene boundaries are not enforced",
}


def pdf_text(path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        return ""
    return "\n".join(p.get_text() for p in pymupdf.open(path)) if path.exists() else ""


def main() -> int:
    problems = []
    sources = {d: (ROOT / d).read_text(encoding="utf-8") for d in DOCS if (ROOT / d).exists()}
    pdf = ROOT / "solution_doc.pdf"
    sources["solution_doc.pdf"] = pdf_text(pdf)
    if not sources["solution_doc.pdf"]:
        problems.append("solution_doc.pdf: missing or unreadable (python scripts/generate_solution_pdf.py)")
    for name, text in sources.items():
        for phrase, why in BANNED.items():
            n = text.count(phrase)
            if n:
                problems.append(f"{name}: {n}x '{phrase}' ({why})")
    md = sources.get("solution_doc.md", "")
    flat = " ".join(sources["solution_doc.pdf"].split())
    probe = "Our own pipeline drifted."
    if md and probe in md and probe not in flat:
        problems.append("solution_doc.pdf is older than solution_doc.md; regenerate it")
    for p in problems:
        print("FAIL", p)
    print("submission check:", "PASS" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
