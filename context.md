# Context & Comprehensive Technical Specification: Promo Finder

> **Document Purpose**: Single source of truth (SSOT) capturing the complete problem statement, system architecture, empirical discoveries, current implementation status, limitations, and future roadmap. Designed to allow any AI agent or engineer to pick up, maintain, modify, or extend `promo-finder` without losing context or violating core constraints.

---

## 1. Problem Statement

### 1.1 Core Mission
The target is to ingest vertical microdrama screenplay episodes (plain-text format) and autonomously discover, extract, and rank **high-converting promo clip candidates** suitable for social video marketing (TikTok, Instagram Reels, YouTube Shorts, ReelShort).

### 1.2 The Four Strict Promo Criteria
Every candidate promo clip must satisfy four non-negotiable requirements:
1. **Duration**: Strictly between **20.0 and 60.0 seconds** in total length.
2. **Hook (First 3 Seconds)**: An immediate narrative, visual, or acoustic hook within the initial 3.0 seconds of the clip to prevent scrolling.
3. **Unresolved Cliffhanger Ending**: The clip must terminate abruptly on an unresolved, high-stakes question, tension spike, accusation, or revelation.
4. **Stand-Alone Clarity**: The clip must be intelligible and compelling to a "cold viewer" who has zero prior context from earlier episodes.

### 1.3 Key Required Outputs
- **Episode Candidates (`promos_per_episode.*`)**: Top candidates for each episode with exact start/end timestamps, duration, hook summary, cliffhanger description, clarity score, and provenance.
- **Show-Level Ranking (`promos_ranked.*`)**: The top 15 promo candidates across the entire series to post first, balanced with diversity constraints (e.g. at most 2 clips per episode, low-confidence clips ranked after others).
- **Audit Manifests (`run_manifest.json`)**: Exact run provenance including git commit hash, model configs, token usage, latency, and parser warning summaries.

---

## 2. Core Operational Principles & Constraints

Any AI agent modifying or extending this codebase must adhere strictly to these non-negotiable rules:

1. **Confidentiality (Top Priority)**:
   - `data/`, `outputs/`, `runs/`, `cache/`, `docs_notes/`, and `.env` are strictly gitignored.
   - Screenplay dialogue, character names, loglines, and proprietary candidate excerpts must **never** be committed, printed to terminal logs, or leaked into public docs.
2. **Centralized LLM Gateway (`src/llm.py`)**:
   - Direct calls to third-party SDKs outside `src/llm.py` are strictly forbidden.
   - All LLM access flows through a token-bucket rate limiter (15 RPM), exponential backoff retry handler, disk cache, and fallback chaining.
3. **Beat IDs Over Timestamps**:
   - The LLM **never** calculates or outputs raw timestamps (seconds, `mm:ss`).
   - The LLM processes and emits discrete screenplay **Beat IDs**. Deterministic Python code maps beat boundaries to exact timestamps and durations.
4. **Empirical Metrics Only**:
   - Never hallucinate or estimate numbers. Report measured counts ($n$ of $N$) and exact latencies.
5. **Continuous Failure Logging (`AI_LOG.md`)**:
   - Every model failure, API quota error, schema drift, or parser anomaly must be documented in `AI_LOG.md` with timestamp, model, context, observed failure, and corrective action.
6. **Gold Reservation Lock (`src/reserved.py`)**:
   - 10 episodes (5 gold: `003, 013, 023, 033, 043`; 5 shadow: `008, 018, 028, 038, 048`) are reserved for unbiased evaluation and locked against model calls during development.

---

## 3. Architecture & Implementation Approach

```
screenplays (.txt)
       │
       ▼
[Stage S1: parse.py] ───────────────────────────┐
       │ (Structured beats: CAMERA, DIALOGUE,   │ (Heuristic Fallback)
       │  ACTION, SFX + Timestamps)             ▼
       ▼                                  [src/tier0.py]
[Stage S2: annotate.py] (LLM: cheap role)       │ (Zero LLM calls)
       │ (h: hook, q: open-q, r: resolve,       │
       │  n: context, t: intensity)             │
       ▼                                        │
[Stage S3a: windows.py / candidates.py]         │
       │ (Enumerate continuous 20-60s windows,  │
       │  score = hook + cliff + intensity)     │
       ▼                                        │
[Stage S4: gates.py]                            │
       │ (20-60s, hook <= 3s, speech cutpoint,  │
       │  IoU <= 0.5 deduplication)             │
       ▼                                        │
[Stage S5: judge.py] (LLM: judge role)          │
       │ (Look-ahead after window, cliffhanger  │
       │  resolution check, clarity score)      │
       ▼                                        │
[Stage S6: run.py Show Ranker] ◄────────────────┘
       │ (Primary: S3a score + Judge Gate pass;
       │  Diversity cap: <=2 per ep; Top 15)
       ▼
outputs/ (promos_per_episode.*, promos_ranked.*, run_manifest.json)
```

### 3.1 Stage-by-Stage Breakdown

| Stage | Module | Type | Responsibility & Implementation |
|---|---|---|---|
| **S1: Parse & Enrich** | `src/parse.py` | Code | Case-insensitive parser. Parses scene headings, timing blocks (`Start: mm:ss.ss \| End: mm:ss.ss`), and beat headers (`**CAMERA:**`, `**ACTION:**`, `**SFX:**`, `**SPEAKER (Emotion):**`). Enriches beats with duration, gaps, and acoustic bounds. Generates `parse_report.json`. |
| **S2: Beat Annotation** | `src/annotate.py` | LLM (`cheap`) | Prompts `prompts/s2_v1.md` with whole episode beats in context. Scores each beat: `hook_potential` ($0..3$), `open_question_after` ($0..3$), `resolves_tension` (bool), `needs_context` ($0..3$), `intensity` ($0..3$). |
| **S3a: Window Search** | `src/windows.py` | Code | Algorithmic sliding window across beat boundaries between 20.0s and 60.0s. Computes objective score: $\text{Hook}_{0..3s} + \text{Cliff}_{\text{last}} + \text{Intensity} - \text{Penalty}_{\text{context}} - \text{Penalty}_{\text{resolution}}$. |
| **S4: Hard Gates** | `src/gates.py` | Code | Strict deterministic filters: duration $20..60$s, valid beat IDs, non-camera hook in relative first 3.0s, speech cutpoint protection (cannot cut mid-spoken beat), IoU $\le 0.5$ deduplication. |
| **S5: Verify (Judge)** | `src/judge.py` | LLM (`judge`) | Evaluates candidate clip + 12s look-ahead after window. Scores hook, cliffhanger, clarity ($0..3$), and `resolved_after` (bool). Discards clips that defuse tension immediately in the look-ahead window (`JUDGE_RESOLVED_AFTER`). |
| **S6: Show Ranking** | `run.py` | Code | Orders surviving candidates across the entire show. **Adopted Decision Rule**: ordered primarily by S3a score among candidates passing the judge gate, with diversity capping ($\le 2$ clips/episode) and low-confidence demotion. |
| **Tier-0 Fallback** | `src/tier0.py` | Code | Deterministic heuristic candidate generator requiring zero LLM calls. Used when API quotas fail or when `--no-llm` is specified. |
| **Reviewer CLI** | `scripts/review.py` | CLI Tool | Terminal review interface allowing humans to review candidate timings and log accept/reject labels to `labels.csv` with standard reason codes. |

---

## 4. Empirical Screenplay Baseline (The 50 Raw Episodes)

Analysis of all 50 screenplay text files (`528,811` characters total) in `data/raw/` established critical empirical boundaries:
- **Total Episodes**: 50 episodes.
- **Episode Duration**: Range $70.50$s to $200.75$s (Median: $117.78$s / ~1.96 min).
- **Total Beats**: 3,727 beats across the show (Median: $73.0$ beats/episode, Max: 121 beats).
- **Beat Breakdown by Kind**:
  - `CAMERA`: 1,754 beats (**47.06%**) — camera shots, angles, and framing.
  - `DIALOGUE`: 1,487 beats (**39.90%**) — spoken character lines.
  - `ACTION`: 372 beats (**9.98%**) — character movement, physical interactions.
  - `SFX`: 114 beats (**3.06%**) — sound cues, stingers.
- **First 3 Seconds Reality**: **Only 34.0%** of episodes (17 of 50) have spoken dialogue in their first 3.0 seconds. 66% start with CAMERA or ACTION. The "first 3 seconds hook" must therefore apply to the *clip's relative start window*, not the episode's absolute timestamp, and must accept high-impact action/camera movements.
- **Median Dialogue Gap**: Median silence between consecutive spoken lines is **0.60s** (Mean: 1.31s). S4 enforces a 0.6s acoustic buffer to prevent cutting trailing speech.

---

## 5. Architectural Review & Key Design Decisions

During adversarial review (`REVIEW.md`), several proposed stages from early planning were simplified or cut to guarantee reliability under free-tier API quotas (15 RPM):
1. **Cut Bradley-Terry Pairwise Ranking (S6)**: Pairwise comparison with order-swapping requires $\binom{N}{2} \times 2 = 300..2,675$ LLM calls, causing severe HTTP 429 quota exhaustion. Replaced with deterministic S3a scoring + Judge gate.
2. **Cut Free-Form Editor Proposals (S3b)**: Algorithmic window search (S3a) already achieves 100% continuous candidate coverage at zero API cost; S3b produced redundant, overlapping intervals.
3. **Cut Cross-Episode Multi-Segment Stitching**: Stitching disjointed scenes across episodes created confusing, incoherent montages for cold viewers. Single continuous windows were adopted as the primary candidate format.
4. **Cut Dynamic Cold-Viewer Retest Loop**: Mutating clip boundaries during evaluation introduced duration gate violations and stateful bugs. Verification was made strictly read-only.

---

## 6. What Has Been Perfectly Accomplished

The following components are fully implemented, verified, test-covered, and frozen:

1. **Deterministic Parser (`src/parse.py`)**:
   - Handles all 50 episodes with zero crashes.
   - Accurately normalizes double-parenthetical character headers, timestamps, and multi-line actions.
2. **LLM Gateway & Platform (`src/llm.py`)**:
   - Unified rate-limiting token bucket (15 RPM, 250k TPM).
   - Exponential backoff retry handling HTTP 429 and HTTP 503 errors.
   - Automatic fallback chaining (`cheap -> judge` model) during live API saturation.
   - Cache keyed by `stage + prompt_version + model + input_hash`.
3. **Model Selection & Quota Probing (`config/models.yaml`)**:
   - Verified active Google Gemini models under free-tier quotas:
     - `cheap`: `gemini-3.5-flash-lite` (500 requests/day, 15 RPM).
     - `judge`: `gemini-3.1-flash-lite` (500 requests/day, 15 RPM).
     - `strong`: `gemini-3.6-flash` (20 requests/day, 5 RPM).
     - `alt`: `gemini-3.5-flash-lite`.
4. **Hard Gates & Heuristic Search (`src/windows.py`, `src/gates.py`, `src/candidates.py`)**:
   - Sub-second window enumeration.
   - Strict 20–60s duration gating, relative 3s hook gating, and mid-sentence speech protection.
5. **Tier-0 Deterministic Baseline (`src/tier0.py`)**:
   - Generates candidate promos with zero LLM calls, providing a reliable baseline and automatic fallback.
6. **Gold Evaluation Benchmark (`scripts/evaluate.py`, `eval_report.md`, `gold/gold_public.csv`)**:
   - 7 audited reference windows across 5 reserved episodes.
   - Full evaluation across IoU $\ge 0.5$ and $\ge 0.3$ measuring Top-1, Top-3, recall, and boundary error against B1 (last 45s), B2 (random 20 seeds), and B3 (Tier-0).
   - Evaluated and documented the three ranking rules: (1) S3a only, (2) S3a order + judge gate (Adopted), (3) Judge-ranked.
7. **Comprehensive Test Suite (`tests/`)**:
   - 14 test modules (104 passing tests, 2 clean skips when `data/raw/` is absent).
   - Fully runnable in fresh clones without local test data.
8. **Confidentiality & Leak Checker (`scripts/leak_check.py`)**:
   - Verified 0 matching 8-word dialogue shingles and 0 denylist speaker/location names in tracked files.
9. **Clean Submission Packaging (`submission` branch)**:
   - Single commit, 73 tracked files, verified clean tree, documented input format specifications, and `requirements-extra.txt` for optional dependencies.

---

## 7. What We Are Currently Lacking (Current Gaps & Limitations)

While the pipeline is robust and production-grade on screenplays, several specific capabilities remain missing:

1. **Lack of Multimodal / Video Pacing Signals**:
   - Screenplay text does not reflect video editing cuts, music drops, visual micro-expressions, actor pacing, or sound effects added in post-production. A scene that reads slowly in text may have fast-paced video editing.
2. **AI-Proposed Gold Windows (Human Ground Truth Gap)**:
   - The gold set currently contains 7 reference windows derived via empirical diagnostics and structural criteria, but lacks validation from real viral marketing performance (e.g. TikTok/Reels CTR or view-through rates).
3. **No Multi-Scene Montage Stitching**:
   - The current engine extracts single continuous windows. It cannot yet stitch a 3-second hook from Scene 1 with a 30-second confrontation from Scene 4 of the same episode.
4. **Lack of Global Long-Arc Narrative Graph**:
   - S2 processes each episode in context, but does not maintain a cross-episode knowledge graph of character backstories, hidden relationships, or long-term secrets.
5. **Free-Tier API Rate Saturation**:
   - Free-tier rate limits (15 RPM, 500 RPD) require sequential processing (~3.3 minutes per stage for 50 episodes). Batch regression testing without cache remains slow.
6. **No Dynamic Visual Framing (Aspect Ratio / 9:16 Cropping)**:
   - The system outputs timestamps and text provenance, but does not output video crop coordinates (e.g., dynamic face-tracking or camera reframing for vertical 9:16 aspect ratio).

---

## 8. Remaining Roadmap & Pending Tasks

For future AI agents continuing this work, tasks are organized by priority:

### Phase 1: Human Ground-Truth Calibration (Immediate Next Step)
- [ ] **Human Review Campaign**: Use `scripts/review.py` to label the primary candidates across all 50 episodes.
- [ ] **Weight Fine-Tuning**: Optimize S3a scoring weights ($\text{Hook}$, $\text{Cliff}$, $\text{Intensity}$, $\text{Context}$, $\text{Resolution}$) against human approval labels.

### Phase 2: Intra-Episode Multi-Segment Stitching (Feature Enhancement)
- [ ] **Multi-Segment Window Generator**: Extend `src/windows.py` to support 2-segment intra-episode candidates (e.g., Hook segment $A$ + Bridge + Cliffhanger segment $B$), constrained by:
  - Total combined duration $\le 60$s.
  - Smooth audio transition buffer between segments.
  - S4 continuity gate ensuring speaker continuity between segments.

### Phase 3: Global Series Entity Graph (Clarity & Hook Improvement)
- [ ] **Series Character & Lore Profiler**: Extract global character relationships, aliases, and secret hierarchies across all 50 episodes.
- [ ] **Enriched Clarity Prompting**: Inject the character profile into S5 Judge prompts so the judge accurately detects when a cold viewer will be alienated by unexplained character relationships.

### Phase 4: Video Pipeline Integration (Multimodal Evolution)
- [ ] **Whisper Audio Alignment**: If `.mp4` video files are provided, align screenplay timestamps with Whisper speech-to-text timestamps to compensate for actor delivery deviations.
- [ ] **FFmpeg Clip Slicer**: Build an export script (`scripts/export_clips.py`) that slices verified promo windows from `.mp4` video files using exact beat cutpoints.
- [ ] **Automated Subtitle / Captions Stamping**: Render burn-in vertical captions for the first 3-second hook to maximize sound-off social media viewing.

### Phase 5: Production Batching & Infrastructure
- [ ] **Google Gemini Batch API**: Implement asynchronous batch request dispatching via `src/llm.py` to bypass 15 RPM online rate limits for 500+ episode catalogs.
- [ ] **Self-Hosted vLLM / Ollama Backend**: Test local open-weight models (e.g. Llama 3.3 70B, Qwen 2.5 32B) using the existing `openai_compatible` provider in `config/models.yaml`.

---

## 9. File Tree & Component Map

```
promo-finder/
├── .gitignore                   # Ignores data/, outputs/, cache/, runs/, .env
├── AGENTS.md                    # Agent operational constraints and rules
├── AI_LOG.md                    # Running log of model errors, 429s, and drift
├── ARCHITECTURE.md              # Architecture specification & implementation status
├── README.md                    # Project documentation, CLI usage, input format specs
├── REVIEW.md                    # Adversarial review of architecture against empirical data
├── context.md                   # THIS FILE — comprehensive system context & roadmap
├── requirements.txt             # Core dependencies (Python 3.11+, pydantic, pyyaml, etc.)
├── requirements-extra.txt       # Optional auxiliary dependencies (scipy)
├── config/
│   ├── aliases.yaml             # Character name alias mapping
│   ├── config.yaml              # Pipeline, cache, and gateway settings
│   ├── models.yaml              # Model role configurations (cheap, judge, strong, alt)
│   └── scoring.yaml             # S3a objective function weights and gate thresholds
├── gold/
│   └── gold_public.csv          # Public gold reference set (7 windows across 5 episodes)
├── prompts/
│   ├── s2_v1.md                 # S2 dense beat annotation prompt
│   ├── judge_v1.md              # S5 pointwise judge prompt
│   └── judge_v2.md              # S5 listwise comparative ranking prompt
├── run.py                       # Unified end-to-end CLI pipeline runner
├── src/
│   ├── __init__.py
│   ├── annotate.py              # S2 LLM beat annotation stage
│   ├── baselines.py             # Evaluation baselines (B1 last 45s, B2 random)
│   ├── candidates.py            # S3a + S4 candidate extraction orchestrator
│   ├── gates.py                 # S4 hard gates (duration, hook, speech boundary, IoU)
│   ├── judge.py                 # S5 judge verification and clarity scoring
│   ├── llm.py                   # Centralized gateway: rate-limiting, retries, caching
│   ├── parse.py                 # S1 screenplay parser and featurizer
│   ├── plan.py                  # Pre-flight quota calculator and run planner
│   ├── reserved.py              # Gold-set episode locking logic
│   ├── sensitivity.py           # Threshold sensitivity analysis utilities
│   ├── tier0.py                 # Deterministic heuristic fallback generator (zero-LLM)
│   └── windows.py               # Algorithmic window search and scoring
├── scripts/
│   ├── build_final_outputs.py   # Formats promos_per_episode and promos_ranked
│   ├── compute_cost_metrics.py  # Calculates token spend and dollar costs
│   ├── diagnose_gold.py         # Diagnoses gate failures on gold reference windows
│   ├── evaluate.py              # Evaluates candidate outputs against gold set
│   ├── leak_check.py            # Confidentiality leak checker (shingles + denylist)
│   ├── list_models.py           # Queries provider API for available model IDs
│   ├── probe_models.py          # Probes models for parameter support and quotas
│   ├── review.py                # Terminal-based promo review CLI
│   ├── run_primary.py           # Executes primary pipeline run
│   ├── run_stability.py         # Multi-run stability evaluation harness
│   ├── show_clip.py             # Terminal-only gold clip inspector
│   └── show_window.py           # Terminal-only candidate window inspector
└── tests/
    ├── test_annotate.py         # S2 annotation parser unit tests
    ├── test_gates.py            # S4 gate validation unit tests
    ├── test_judge.py            # S5 judge schema and scoring unit tests
    ├── test_leak.py             # Screenplay leak check unit tests
    ├── test_llm.py              # Gateway retry, backoff, and cache unit tests
    ├── test_parse.py            # Screenplay parsing and warning unit tests
    ├── test_reserved.py         # Reserved episode lock unit tests
    ├── test_review.py           # Terminal reviewer CLI unit tests
    ├── test_run.py              # End-to-end pipeline runner unit tests
    ├── test_sensitivity.py      # Threshold sensitivity unit tests
    ├── test_show_window.py      # Window display inspector unit tests
    ├── test_tier0.py            # Tier-0 heuristic candidate unit tests
    ├── test_validate_gold.py    # Gold set ingestion unit tests
    └── test_windows.py          # S3a window enumeration and scoring unit tests
```

---

## 10. CLI Quick Reference for Incoming Agents

```bash
# 1. Run full pipeline on screenplay directory with LLM
python run.py --input <path_to_screenplays> --out outputs/

# 2. Run deterministic zero-LLM heuristic mode (instant, free)
python run.py --input <path_to_screenplays> --out outputs/ --no-llm

# 3. Run specific episodes only
python run.py --input <path_to_screenplays> --out outputs/ --episodes episode-001 episode-002

# 4. Run test suite
python -m pytest

# 5. Run confidentiality leak check
python scripts/leak_check.py

# 6. Launch terminal promo reviewer
python scripts/review.py --promos outputs/promos_per_episode.json --out labels.csv

# 7. Evaluate against public gold reference set
python scripts/evaluate.py --judged outputs/ --gold gold/gold_public.csv
```

