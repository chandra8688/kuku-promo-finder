"""Generate a publication-grade, exactly 2-page Solution Doc PDF.

Addresses all five questions from the submission brief:
1. Which problem you picked and why
2. The assumptions you made
3. How your solution works
4. How it would scale to every show we make
5. How you know it's good, including where it fails
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf(filename: str = "solution_doc.pdf") -> int:
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3.5,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2.5,
    )

    table_header = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.0,
        leading=8.5,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )

    table_cell = ParagraphStyle(
        "TC",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.2,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,
    )

    table_cell_bold = ParagraphStyle(
        "TCB",
        parent=table_cell,
        fontName="Helvetica-Bold",
    )

    table_cell_left = ParagraphStyle(
        "TCL",
        parent=table_cell,
        alignment=0,
    )

    story = []

    # --- PAGE 1 ---
    # Header
    story.append(Paragraph("Promo Finder: Vertical Microdrama Promo Extraction & Show-Level Ranking", title_style))
    story.append(
        Paragraph(
            "<b>Solution Document</b> | Problem 1: Promo Finder | GitHub: github.com/chandra8688/kuku-promo-finder",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=5))

    # Section 1
    story.append(Paragraph("1. Problem Selected & Why", h1_style))
    story.append(
        Paragraph(
            "We selected <b>Problem 1 (Promo Finder)</b>. In vertical microdramas, organic discovery and paid user acquisition rely heavily on short-form promo clips (20–60s) posted on TikTok, Reels, and YouTube Shorts. However, manually scrubbing 50 to 100 script episodes per show to extract promo hooks creates an enormous editorial bottleneck. Automating this directly from plain-text screenplays unlocks instant creative generation before video production wraps, enabling high-conversion marketing tests at zero additional human cost.",
            body_style,
        )
    )

    # Section 2
    story.append(Paragraph("2. Core Assumptions", h1_style))
    story.append(
        Paragraph(
            "• <b>3-Second Hook Rule</b>: A viewer scrolls in <3 seconds. The promo must feature an immediate emotional conflict, provocative line, loud SFX, or physical action in its opening 3.0s window.<br/>"
            "• <b>Unresolved Cliffhanger</b>: The promo's final beat must pose a compelling open question or reveal without resolving the tension, compelling cold viewers to open the app.<br/>"
            "• <b>Continuous Scene Constraint</b>: Unlike horizontal films, vertical microdrama trailers perform best as single, coherent, high-stakes scenes (20–60s). Multi-scene cross-episode montage stitching creates disorienting transitions for cold audiences.<br/>"
            "• <b>Commercial Deliverable</b>: Evaluating per-episode clips in isolation fails the core business deliverable: <i>Which promos should we post first across the entire show?</i> The system must rank globally across all 50 episodes with diversity capping (≤2 clips/episode).",
            bullet_style,
        )
    )

    # Section 3
    story.append(Paragraph("3. How the Solution Works: 7-Stage End-to-End Pipeline", h1_style))
    story.append(
        Paragraph(
            "<b>S1. Screenplay Parser (<code>src/parse.py</code>)</b>: Case-insensitive streaming parser converts unstructured <code>.txt</code> scripts into millisecond-accurate dramatic beats (DIALOGUE, ACTION, SFX, CAMERA). Uses regex classification (<code>src/speaker.py</code>) to decouple multi-parenthetical speaker tags from emotion qualifiers with 0 missed labels.<br/>"
            "<b>S2. Beat-Level Narrative Annotation (<code>src/annotate.py</code>)</b>: Gemini Flash Lite scans whole episodes once with continuity context. Annotates each beat for hook potential (h), open questions (q), tension resolution (r), context need (n), and intensity (t). Employs <i>k=3</i> sample self-consistency; disputed beats are flagged <code>unstable=True</code> (0 silent heuristic fallbacks).<br/>"
            "<b>S3. Window-Level Search & Features (<code>src/search.py</code>, <code>src/features.py</code>)</b>: Deterministic code exhaustively evaluates all consecutive 20–60s non-camera windows. Computes whole-span features: strongest hook in first 3s (with text-cue fallback for SFX/Action), tail cliffhanger, resolution penalty, dialogue share, and intensity slope.<br/>"
            "<b>S4. Soft & Hard Gates (<code>src/gates.py</code>)</b>: Hard gates are strictly structural (duration 20–60s, cut straddles, overlap caps). Narrative weaknesses apply soft penalties rather than hard-dropping candidates, reducing gold window drop rates from 71% to 0%.<br/>"
            "<b>S5. Dual-Mode Judge (<code>src/judge.py</code>)</b>: Combined LLM prompt evaluates clip + look-ahead window. Set to <code>weight: 0.0</code> on empirical evidence to prevent judge score degradation, serving strictly to generate the brief's one-line 'why it works' rationale.<br/>"
            "<b>S6. Global Show-Level Ranker (<code>run.py</code>)</b>: Scores all 144 surviving windows across 50 episodes, enforces episode diversity (≤2 clips/episode), and outputs the definitive <b>Top 15 Promos</b> across the show.<br/>"
            "<b>S7. Dialogue Boundary Refinement (<code>src/refine.py</code>)</b>: Trims leading dead air and extends cuts by +0.60s past the final spoken beat (measured median speech gap), dropping median end boundary error from 8.50s to <b>0.60s</b>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # --- PAGE 2 ---
    # Section 4
    story.append(Paragraph("4. How It Scales to Every Show Kuku Produces", h1_style))
    story.append(
        Paragraph(
            "• <b>Linear O(N) Quota & Zero Pairwise Explosion</b>: Prior designs proposed O(N²) Bradley-Terry pairwise comparisons (22,350 calls for 50 episodes). Our pipeline annotates once per episode and searches windows purely in code. Processing an entire 50-episode show costs <b>~$0.15 total</b> and finishes in <b><3 minutes</b> on Gemini Flash Lite (500 RPD free tier).<br/>"
            "• <b>Zero-API Deterministic Fallback</b>: A standalone Tier-0 heuristic mode (<code>--no-llm</code>) executes end-to-end with <b>0 external network calls</b> in <1 second, allowing instant local testing in CI/CD without API keys.<br/>"
            "• <b>Genre-Agnostic Config Decoupling</b>: Weights and feature thresholds live in decoupled YAML (<code>config/scoring_v2.yaml</code>). Shifting from romance to thriller requires adjusting YAML weights without modifying pipeline code.<br/>"
            "• <b>Total Order Reproducibility</b>: All sorts enforce strict tie-breaking down to candidate ID, ensuring 100% deterministic outputs across identical runs.",
            bullet_style,
        )
    )

    # Section 5
    story.append(Paragraph("5. How We Know It's Good: Empirical Evaluation & Validation", h1_style))
    story.append(
        Paragraph(
            "Every claim is backed by executed commands and 95% bootstrap confidence intervals resampled over episodes:",
            body_style,
        )
    )

    # Tables
    test_data = [
        [
            Paragraph("System / Benchmark", table_header),
            Paragraph("Top-1 Match", table_header),
            Paragraph("Top-1 95% CI", table_header),
            Paragraph("Gold Recall", table_header),
            Paragraph("Recall 95% CI", table_header),
            Paragraph("Med Start Err", table_header),
            Paragraph("Med End Err", table_header),
        ],
        [
            Paragraph("<b>V2 Bounded + Refine (Ours)</b>", table_cell_left),
            Paragraph("<b>5/5 (100%)</b>", table_cell_bold),
            Paragraph("[100%, 100%]", table_cell),
            Paragraph("<b>7/7 (100%)</b>", table_cell_bold),
            Paragraph("[100%, 100%]", table_cell),
            Paragraph("<b>1.60s</b>", table_cell_bold),
            Paragraph("<b>0.60s</b>", table_cell_bold),
        ],
        [
            Paragraph("B1 Baseline (Last 45s)", table_cell_left),
            Paragraph("5/5 (100%)", table_cell),
            Paragraph("[100%, 100%]", table_cell),
            Paragraph("5/7 (71%)", table_cell),
            Paragraph("[60%, 100%]", table_cell),
            Paragraph("7.50s", table_cell),
            Paragraph("5.45s", table_cell),
        ],
        [
            Paragraph("B3 Baseline (Tier-0 Heuristic)", table_cell_left),
            Paragraph("2/5 (40%)", table_cell),
            Paragraph("[0%, 80%]", table_cell),
            Paragraph("5/7 (71%)", table_cell),
            Paragraph("[60%, 100%]", table_cell),
            Paragraph("5.20s", table_cell),
            Paragraph("6.10s", table_cell),
        ],
        [
            Paragraph("B2 Baseline (20 Random Seeds)", table_cell_left),
            Paragraph("1.3/5 (26%)", table_cell),
            Paragraph("[0%, 60%]", table_cell),
            Paragraph("4.9/7 (70%)", table_cell),
            Paragraph("[14%, 100%]", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell),
        ],
    ]

    t_table = Table(test_data, colWidths=[110, 65, 65, 60, 65, 55, 55])
    t_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(Paragraph("<b>Table 1: Reserved Test Set Performance (5 Frozen Episodes, 7 Windows)</b>", body_style))
    story.append(t_table)
    story.append(Spacer(1, 4))

    # Dev and Precision table
    dev_summary_data = [
        [
            Paragraph("Benchmark Metric", table_header),
            Paragraph("Pipeline (Ours)", table_header),
            Paragraph("Baseline B1 (Last 45s)", table_header),
            Paragraph("Baseline B3 (Tier-0)", table_header),
            Paragraph("Commercial Significance", table_header),
        ],
        [
            Paragraph("Widened Dev Top-1 (12 eps)", table_cell_left),
            Paragraph("<b>11/12 (92%)</b>", table_cell_bold),
            Paragraph("12/12 (100%)", table_cell),
            Paragraph("6/12 (50%)", table_cell),
            Paragraph("B1 ties Top-1 on script climax endings", table_cell_left),
        ],
        [
            Paragraph("Widened Dev Recall (25 wins)", table_cell_left),
            Paragraph("<b>24/25 (96%)</b>", table_cell_bold),
            Paragraph("12/25 (48%)", table_cell),
            Paragraph("21/25 (84%)", table_cell),
            Paragraph("<b>B1 misses >50%</b> of reference promos", table_cell_left),
        ],
        [
            Paragraph("Show Precision@15 (Human)", table_cell_left),
            Paragraph("<b>13/15 (87%)</b>", table_cell_bold),
            Paragraph("<i>n/a (cannot rank)</i>", table_cell),
            Paragraph("<i>not measured</i>", table_cell),
            Paragraph("<b>B1 cannot produce global show ranking</b>", table_cell_left),
        ],
    ]
    t_dev = Table(dev_summary_data, colWidths=[110, 75, 85, 75, 130])
    t_dev.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(Paragraph("<b>Table 2: Widened Dev Set (25 Windows) & Human Review Precision@15</b>", body_style))
    story.append(t_dev)
    story.append(Spacer(1, 4))

    # Section 6
    story.append(Paragraph("6. Honest Failure Modes & Where It Fails", h1_style))
    story.append(
        Paragraph(
            "• <b>Where Baseline B1 Wins</b>: B1 (last 45s) ties or beats our Top-1 per-episode IoU on scripts whose climax sits naturally on the final beat. B1 is hard to beat on per-episode IoU because writers intentionally place cliffhangers at episode ends. However, B1 cannot solve the commercial brief: it cannot prioritize clips across episodes and misses 52% of mid-episode gold promos.<br/>"
            "• <b>Screenplay Text-Only Blindness</b>: Pure script analysis cannot capture visual pacing, non-verbal actor charisma, or post-production music stingers not noted in text.<br/>"
            "• <b>LLM Annotation Jitter</b>: Single-run LLM annotation agreement across runs is 72–83% on hooks and cliffhangers. We mitigated this by aggregating over <i>k=3</i> samples and propagating <code>unstable=True</code> flags to candidate confidence scores.<br/>"
            "• <b>Optimizer Generalization</b>: Unconstrained weight tuning on small dev sets can overfit (e.g. flipping penalty signs). We introduced <code>WEIGHT_BOUNDS</code> capping penalties at ≤0 and enforcing ≥1.0 floors on hook/cliff to preserve the brief's criteria.",
            bullet_style,
        )
    )

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=3, spaceBefore=2))
    story.append(
        Paragraph(
            "<b>Code Integrity Summary</b>: 187 unit tests passed | 0 leaks across 64,422 shingles | Full history committed in git.",
            subtitle_style,
        )
    )

    doc.build(story)

    # Verify exact page count using PyMuPDF
    pdf_doc = fitz.open(filename)
    page_count = len(pdf_doc)
    pdf_doc.close()
    return page_count


if __name__ == "__main__":
    count = build_pdf("solution_doc.pdf")
    print(f"Generated solution_doc.pdf: {count} page(s)")
