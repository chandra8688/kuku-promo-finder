# AGENTS.md — Promo Finder

Operating rules for any agent working in this repository. Read `ARCHITECTURE.md` for
what the pipeline does and `context/TASKS.md` in the upgrade bundle for current work.

## 1. Mission

Read microdrama screenplays and return ranked promo clip candidates that are 20 to 60
seconds long, hook within their own first 3 seconds, end on an unresolved beat, and
stand alone for a cold viewer.

## 2. Non-negotiable rules

1. **Confidentiality.** `data/`, `outputs/`, `cache/`, `runs/`, `posthoc/`, `fixtures/`
   and `.env` stay gitignored. Never stage screenplay text, character names, loglines
   or extracted dialogue. Committed artefacts carry scores and beat IDs, never text.
2. **One gateway.** Every model call goes through `src/llm.py`. No provider SDK or HTTP
   call anywhere else. That is what makes token accounting, caching and fallback real.
3. **Beat IDs, not timestamps.** The model returns beat IDs. Code converts to seconds.
   If a prompt ever asks a model for a time, that is a bug.
4. **Measured numbers only.** Never write a metric you did not produce with a command.
   Paste the command and its output into the report. No estimates, no remembered
   figures, no rounding a bad number up.
5. **Small steps.** Plan, change, run tests, show the real output, then commit. One
   task per commit.
6. **Log failures.** Every model failure, schema deviation, hallucination or wrong
   guess goes into `AI_LOG.md` with root cause and fix. This file is an asset. Do not
   tidy it, do not delete entries, do not soften them.
7. **Never tune on the reserved episodes.** Episodes 3, 13, 23, 33 and 43 and their
   gold windows are the test set. All tuning happens on dev episodes. If you look at
   the test numbers and then change a weight, the report has to say so.
8. **Never claim a stage helps without an ablation.** One row, one change.

## 3. Roles and routing

Models are configured per role in `config/models.yaml`, not in a root `config.yaml`.
Current mapping, set after capability probes and quota failures recorded in `AI_LOG.md`:

| Role | Model | Used for | Free-tier notes |
|---|---|---|---|
| `cheap` | `gemini-3.5-flash-lite` | S2 beat annotation | 500 rpd, `thinking_budget` unsupported |
| `judge` | `gemini-3.1-flash-lite` | combined judge scoring | 500 rpd, `max_output_tokens` 2048 |
| `strong` | `gemini-3.6-flash` | top-N second opinion, listwise rerank | 20 rpd, use sparingly |
| `alt` | `gemini-3.5-flash-lite` | model-swap stability checks | 500 rpd |

Check the actual limits in `config/models.yaml` before any batch run. A role's
`max_output_tokens` must cover its largest job: S2 over 120 beats needs 4096, and
pointing a 2048-token role at S2 truncates the JSON, which has already happened once.

## 4. Layout

```
promo-finder/
├── run.py                  # single entrypoint, --input / --out / --no-llm / --strict
├── AGENTS.md ARCHITECTURE.md AI_LOG.md README.md eval_report.md REVIEW.md
├── config/
│   ├── models.yaml         # roles -> models, quotas, token caps
│   ├── config.yaml         # gateway, cache, runs
│   ├── scoring.yaml        # v1 weights, thresholds, frozen decision rule
│   ├── scoring_v2.yaml     # v2 features, weights, soft gates, refine, judge mode
│   └── aliases.yaml
├── gold/                   # gold_public.csv committed; dev labels and protocol
├── prompts/                # versioned prompt registry
├── src/
│   ├── llm.py              # the only gateway
│   ├── parse.py speaker.py # S1
│   ├── annotate.py consistency.py  # S2 and k-sample aggregation
│   ├── windows.py features.py score.py search.py  # S3
│   ├── gates.py            # S4, structural rejects only
│   ├── judge.py            # S5
│   ├── plan.py refine.py   # S6 and S7
│   ├── evalkit.py          # IoU matching, bootstrap CIs, paired tests
│   ├── baselines.py tier0.py reserved.py
├── scripts/                # eval_v2, review, make_fixtures, tune_weights, leak_check
│   └── archive/            # superseded one-off scripts, indexed
└── tests/
```

## 5. Development

- Python 3.11+, standard `typing`, Pydantic for any model output schema.
- Anything converting beats to seconds, filtering candidates or ordering them must be
  deterministic and covered by a test. Sorts need a total order; tie-break down to
  candidate ID.
- `python -m pytest tests/ -q` before every commit.
- Work offline where possible: `python scripts/make_fixtures.py` builds synthetic
  episodes in the real input format so logic can be tested with no quota and no
  contact with `data/raw`.
- Prompt changes bump the prompt version, because the version is part of the cache key.
  Changing a prompt without bumping it serves stale cached answers.
