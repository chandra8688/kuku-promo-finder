# Where AI got it wrong

This project was built with AI coding assistants (Gemini in Antigravity, then Claude).
These are the mistakes that mattered, what caught them, and the fix. The full
chronological log is in `notes/ai_log_full.md`.

| # | What went wrong | How it was caught | Fix |
|---|---|---|---|
| 1 | The assistant filled `config/models.yaml` with model names from memory (`gemini-2.5-flash/pro`), unverified and out of date. | Checking the names against the models the key can reach. | Listed the models the key can actually reach before configuring any role. |
| 2 | Stability runs reported perfect agreement between "independent" annotation runs. The cache key ignored the run label, so runs 2 and 3 were cache hits of run 1. | `calls.jsonl` showed 100% cached calls at 0.0 s latency. | Run label added to the cache key; re-ran uncached; real agreement is 64–89% per field. |
| 3 | A baseline scored 0/5 because the evaluation script read the wrong JSON key and loaded no windows. | The number was implausibly low for a keyword baseline. | Fixed the key; the baseline scores 3/5. A suspicious number gets traced before it is reported. |
| 4 | Precision@15 was reported as 15/15, computed from the pipeline's own gates, which every promo passes by construction. | Review of what the metric actually measured. | Replaced with a person's approve/reject labels; `scripts/check_submission.py` now blocks the old claim. |
| 5 | The LLM judge was trusted to rank clips, but in a validity test it preferred control clips (last-45-second and random windows) over the picks in 65% of 176 pairs and changed its top pick when the order flipped. | Pre-registered test with controls and reversed order. | The judge was demoted to a gate, not a ranker. |
| 6 | After that, the show-level ranking ignored the judge entirely. 4 of the shipped top 15 opened on hooks the judge scored 0–1/3, and the order followed position-in-episode. | Comparing judge and annotator scores on the shipped top 15. | `src/showrank.py`: a promo leads only if both models agree on hook and cliffhanger. |
| 7 | A clip ranked 6th ended on a beat the annotator marked as resolving the tension; only the judge's resolution verdict was checked. | The new rule checker (`scripts/validate_promos.py`), and the same clip got the lowest blind rating. | The annotator's verdict now gates too; 15/15 pass. |
| 8 | A test written by the assistant used the show's character names, and one commit message did too. | `tests/test_leak.py` failed. | Made-up names; history squashed before publishing. |
| 9 | The strongest model returned 503 (overloaded) three times and the run lost its "why it works" lines. | Run output. | Finalists call falls back to the judge model. |
