"""Last gate before submitting. Exit 0 only when nothing known-false or unfilled remains.

Checks the repo's markdown and, if given, the exported solution doc PDF (it is
submitted through the form, not committed). Each banned phrase is a claim this repo
once made that was false for the shipped code.

    python scripts/check_submission.py [path/to/solution_doc.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "ARCHITECTURE.md", "AI_LOG.md", "eval/eval_report.md"]
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


# app.py must read numbers from files the pipeline wrote, never carry them as text.
APP_PATTERNS = {
    r"\d+(?:\.\d+)?/\d+\s*\(\d+%\)": "hardcoded metric like 13/15 (87%)",
    r"passed in \d": "hardcoded test-run result",
    r"\b[0-9a-f]{16}\b": "hardcoded config hash",
    r"\d[\d,]* shingles": "hardcoded leak-check result",
}
# a test count in a document goes stale with the next test added
DOC_PATTERNS = {r"\b\d+ tests? pass": "test count typed into a document"}


def pdf_text(path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        return ""
    return "\n".join(p.get_text() for p in pymupdf.open(path)) if path.exists() else ""


def main() -> int:
    problems = []
    sources = {d: (ROOT / d).read_text(encoding="utf-8") for d in DOCS if (ROOT / d).exists()}
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
        sources[pdf.name] = pdf_text(pdf)
        if not sources[pdf.name]:
            problems.append(f"{pdf}: missing or unreadable (pip install pymupdf)")
    for name, text in sources.items():
        for phrase, why in BANNED.items():
            n = text.count(phrase)
            if n:
                problems.append(f"{name}: {n}x '{phrase}' ({why})")
    import re

    app = ROOT / "app.py"
    if app.exists():
        code = app.read_text(encoding="utf-8")
        for pat, why in APP_PATTERNS.items():
            for m in re.finditer(pat, code):
                problems.append(f"app.py: '{m.group(0)}' ({why})")
    for name, text in sources.items():
        for pat, why in DOC_PATTERNS.items():
            for m in re.finditer(pat, text):
                problems.append(f"{name}: '{m.group(0)}' ({why})")
    for p in problems:
        print("FAIL", p)
    print("submission check:", "PASS" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
