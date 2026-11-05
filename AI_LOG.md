# AI_LOG.md

Running log of LLM errors, hallucinations, schema deviations, and unexpected model outputs observed during the development and execution of `promo-finder`.

## Logging Guidelines
Whenever any model produces an output that fails verification, breaks rules, or requires correction, log an entry here with:
- **Timestamp**: Date and time of incident
- **Role / Model**: Role (`cheap`, `strong`, `judge`) and model name from `config.yaml`
- **Task / Prompt Context**: What task the model was performing
- **Error Description**: What went wrong (e.g., hallucinated timestamps, missed 3s hook, inaccurate beat IDs, invalid JSON)
- **Observed Output**: Snippet or description of wrong output
- **Corrective Action**: Code safeguard, prompt adjustment, or validation logic applied

---

## Log Entries

| Timestamp | Role / Model | Task / Context | Error Description | Observed Output | Corrective Action |
|-----------|--------------|----------------|-------------------|-----------------|-------------------|
| 2026-11-05 | System Setup | Prompt 0: Model configuration | Agent filled config.yaml with model names from memory (gemini-2.5-flash/pro); unverified and probably outdated. | `gemini-2.5-flash`, `gemini-2.5-pro` (from memory) | Replaced with "TODO_SET_AFTER_LIST_MODELS" pending verification via scripts/list_models.py |
| 2026-11-05 | cheap / gemini-3.5-flash | S2 Beat Annotation (episode-001) | JSON output truncated at line 286 (EOF error) because max_output_tokens=2048 was exhausted by verbose multi-line JSON indentation across 45 beats. | `EOF while parsing an object at line 286 column 12` | Increased cheap max_output_tokens to 4096 in config/models.yaml; added compact JSON instruction to prompts/s2_v1.md. |
| 2026-11-05 | Codebase Audit | Part A: 8-word shingle leak check v1 | `scripts/leak_check.py` detected 17 matching shingles in tracked files: `REVIEW.md` (1) and `tests/test_parse.py` (16). Root cause: synthetic timestamp sequences in test fixtures colliding with raw episode header tokens. | `REVIEW.md: 1`, `tests/test_parse.py: 16` | Upgraded to leak check v2 (shingle boilerplate filter + name denylist). Scrubbed `REVIEW.md`, `tests/test_parse.py`, `tests/test_annotate.py`. |
| 2026-11-05 | judge / gemini-3.6-flash | S5 Judge pipeline (dev episodes) | Daily free-tier quota exhausted on first batch call. API reports `Generate content requests per day limit exceeded on Free Tier`, limit=20 req/day. Request sent to `gemini-3.5-flash` returned error with `quotaDimensions.model: gemini-3.6-flash`, proving the API maps `gemini-3.5-flash` under the `gemini-3.6-flash` quota pool on the free tier. | `HTTP 429 RESOURCE_EXHAUSTED (quotaDimensions.model: gemini-3.6-flash)` | Reclassified 429 errors into per-minute (retry) vs per-day (no retry). Re-mapped `cheap` to `gemini-3.5-flash-lite` (500 RPD) and `judge` to `gemini-3.1-flash-lite` (500 RPD) in config/models.yaml. |
| 2026-11-05 | cheap / gemini-3.5-flash-lite | Capability Probe (PART B) | `gemini-3.5-flash-lite` rejected `thinkingConfig` with HTTP 400 (`thinkingConfig is not supported`). | `HTTP 400: Request contains an invalid argument: thinkingConfig is not supported` | Set `thinking_budget: null` for `gemini-3.5-flash-lite` in config/models.yaml. |
| 2026-11-05 | alt / gemma-4-31b-it | Capability Probe (PART B) | `gemma-4-31b-it` rejected `thinkingBudget` with HTTP 400 and second probe call timed out at 30s. | `HTTP 400: Thinking budget is not supported / Read operation timed out` | Reverted `alt` role to `gemini-3.6-flash` per specification. |
| 2026-11-05 | strong / gemini-3.8-flash | Capability Re-Probe (PART D) | `gemini-3.8-flash` failed all three capability probe calls with HTTP 503 (service unavailable / high demand). `gemma-4-31b-it` passed system prompt (47.8s) but failed JSON output with HTTP 503 and thinkingBudget with HTTP 400. | `HTTP 503: This model is currently experiencing high demand` | Mapped `strong` role to `gemini-3.6-flash` per fallback rule; mapped `alt` to `gemini-3.5-flash-lite` for judge swap test. |
| 2026-11-05 | Pipeline / S4 Gates | Dev Episode Survivor Analysis (PART E) | Under initial rigid thresholds, an episode yielded 0 survivors (`episode-005`) and another yielded 1 survivor (`episode-002`), failing the target minimum candidate quota. | `0 survivors in episode-005, 1 survivor in episode-002` | Introduced configurable `hook_window_s: 3.0` (Part A) and fallback mechanism (Part B) admitting candidates failing ONLY `HOOK_TOO_WEAK` / `WEAK_CLIFFHANGER` up to `min_survivors: 2` with `low_confidence=True`. Never admits structural, straddle, or resolved-ending failures. |
