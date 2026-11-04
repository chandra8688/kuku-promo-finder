# AGENTS.md — Promo Finder

Operational guide and behavioral specifications for AI agents working on the `promo-finder` repository.

---

## 1. Project Mission & Objectives

**Target:** Read microdrama screenplays and identify promo clip candidates that meet the following strict criteria:
- **Duration:** 20 to 60 seconds in total length.
- **Hook (First 3 Seconds):** Immediate narrative or visual hook within the first 3 seconds to capture attention.
- **Unresolved Cliffhanger Ending:** The clip terminates abruptly on an unresolved, high-stakes question, tension spike, or revelation.
- **Stand-Alone Clarity:** The clip must be intelligible to a cold viewer without needing external context from preceding episodes.

---

## 2. Fundamental Operating Rules

Agents working on this codebase must adhere strictly to these six non-negotiable rules:

1. **Confidentiality:** `data/`, `outputs/`, and `.env` must remain strictly gitignored at all times. Screenplay packs, raw scripts, and extracted proprietary promo candidates are confidential. Never stage or commit screenplay data.
2. **Centralized LLM Gateway:** Every LLM call must route through `src/llm.py` exclusively. Direct calls to third-party SDKs outside `src/llm.py` are strictly forbidden. This ensures unified token tracking, cost monitoring, fallback handling, and error logging.
3. **Beat IDs Over Timestamps:** The LLM **never** outputs raw timestamps (seconds, mm:ss). The LLM processes and outputs discrete screenplay **Beat IDs**. Deterministic Python code calculates and maps beat ranges to exact timestamps and durations.
4. **Empirical Metrics Only:** Never invent, hallucinate, or estimate data or performance metrics. Report measured numbers only (e.g., actual character counts, real duration computations, benchmark outputs, measured latency).
5. **Incremental Cadence:** Work in small, verifiable steps.
   - Present a concise plan first.
   - Implement the code changes.
   - Run tests and display actual, observed terminal output before reporting a task as complete.
6. **Continuous Failure Logging:** Maintain `AI_LOG.md`. Log any model failure, hallucinated output, format breach, or misjudgment immediately, including root cause and mitigation.

---

## 3. Agent Roles & Model Routing (`config.yaml`)

Models are configured by functional role in `config.yaml`:

| Role | Default Model | Responsibility |
|---|---|---|
| `cheap` | `gemini-2.5-flash` | Fast scanning, screenplay chunking, token-efficient pre-filtering, and preliminary beat tagging. |
| `strong` | `gemini-2.5-pro` | Deep screenplay parsing, thematic beat analysis, hook detection (0–3s), and cliffhanger tension evaluation. |
| `judge` | `gemini-2.5-pro` | Objective multi-criteria evaluation, stand-alone clarity assessment, scoring, and promo candidate validation. |

---

## 4. Repository Skeleton & Conventions

```
promo-finder/
├── .gitignore          # Ignores data/, outputs/, .env, and build artifacts
├── AGENTS.md           # Agent operational guidelines and strict constraints
├── AI_LOG.md           # Running record of model failures and drift
├── config.yaml         # Model role mappings (cheap, strong, judge)
├── requirements.txt    # Project dependencies (Python 3.11+)
├── data/               # [GITIGNORED] Confidential screenplay inputs
├── outputs/            # [GITIGNORED] Generated promo candidate outputs & reports
├── src/                # Core application source code
│   ├── __init__.py
│   └── llm.py          # Mandatory centralized LLM gateway
└── tests/              # Unit, integration, and evaluation test suites
```

---

## 5. Development Guidelines for Agents

- **Python Version:** Python 3.11+ compatible code.
- **Typing & Validation:** Use standard typing (`typing`) and Pydantic models for structured outputs and schemas.
- **Determinism:** Any timing, beat-to-seconds conversion, or candidate filtering logic must be deterministic and test-covered.
- **Testing:** Add targeted tests in `tests/` for new functionality. Always verify with `python -m unittest discover tests` or `pytest`.

