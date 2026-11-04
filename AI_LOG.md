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
| 2026-11-05 | System Setup | Prompt 0: Model configuration | Prompt 0 - agent filled config.yaml with model names from memory (gemini-2.5-flash/pro); unverified and probably outdated; replaced with a placeholder pending a check of the IDs my API key can use. | `gemini-2.5-flash`, `gemini-2.5-pro` (from memory) | Replaced with "TODO_SET_AFTER_LIST_MODELS" in config.yaml pending verification via scripts/list_models.py |

