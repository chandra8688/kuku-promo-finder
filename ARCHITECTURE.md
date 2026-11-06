# ARCHITECTURE

Goal: given microdrama screenplays, return ranked 20 to 60 second promo clips that
hook inside their first 3 seconds, end unresolved, and make sense to someone who has
not seen the episode. Each promo carries its episode, the start and end of every
segment in order, its hook, its cliffhanger, and one line on why it works.

This file describes **what runs**. Anything specified but not built is in the last
section, named as such.

## Principles

1. The model never emits timestamps. It returns beat IDs; code converts them to
   seconds. This is the main defence against hallucinated timings.
2. Annotate once per episode, then search in code. Search is free and deterministic;
   annotation is the thing that costs quota.
3. Every stage sits behind a config flag and has to justify itself with an ablation row.
4. Every candidate carries its own provenance: the beat IDs it spans, each scoring
   term and its contribution, any boundary-refinement ops, and the model and prompt
   version behind each score.
5. Model access only through `src/llm.py`. Models swap by config, never by code edit.

## Stages

**S1 Parse, `src/parse.py`.** Case-insensitive parser over the screenplay text into
beats of `{episode, scene, beat_id, kind, speaker_raw, speaker_norm, emotion_label,
start_s, end_s, text}`, where `kind` is DIALOGUE, ACTION, SFX or CAMERA. Alias map
folds character spellings together. Per-beat features: duration, gap to the previous
beat, concurrency with camera beats. Emits `parse_report.json` and fails loudly on an
unknown header unless `--strict` is off. Speaker headers may carry several
parentheticals, for example `MAYA (O.S.) (COMMANDING)`; placement and mic tags are
separated from the emotion label.

**S2 Annotate, `src/annotate.py`, one model call per episode.** Whole episode in
context together with its logline and continuity notes. Per beat: `h` hook potential,
`q` open question after this beat, `r` resolves tension, `n` needs prior context, `t`
intensity, each 0 to 3 except `r` which is boolean. The shipped runner annotates each
episode once. Beats the model fails to score are null-filled and counted as missing,
never as zero; any window containing one is `low_confidence`. A k-sample aggregation
helper (`src/consistency.py`: median per numeric field, majority for `r`, `unstable`
flag on disagreement) exists and was used to measure stability, but `run.py` does not
call it.

**S3 Window search, `src/engine.py` over `src/windows.py`, code only.** `run.py`, the
evaluation and the weight fit all select candidates through `src/engine.py` with the one
config `config/scoring_v2.yaml`; its path, hash and effective weight vector are written
into every candidates file and the run manifest. Enumerate every
window of consecutive non-camera beats whose duration falls in 20 to 60 seconds.
For each window compute features over the whole span, not off its first and last beat:

- hook: strongest hook in the window's own first 3 seconds, over dialogue, SFX, action
  and camera beats, discounted by kind, with a text-cue fallback so a slap or a crash
  zoom can carry a hook when nobody speaks;
- cliffhanger: strongest open question in the final seconds, so a closing camera or
  action beat does not erase the question asked just before it;
- resolution: whether anything in that tail resolves the tension;
- context need: the opening seconds, not only beat one;
- intensity: duration-weighted mean, plus peak and the early-to-late slope;
- position: where the window sits in the episode;
- clarity proxies: dialogue share, distinct speaker count;
- recorded for the reviewer but not scored: annotation coverage, scenes crossed, largest
  gap between beats.

Score is a weighted sum of those terms. Every term and its contribution is stored on
the candidate. Weights live in `config/scoring_v2.yaml` and are fitted on dev episodes
only, never on the reserved evaluation episodes.

**S4 Gates, `src/gates.py`, code only.** Hard rejects are structural only: duration
out of range, unknown beat IDs, unordered or overlapping segments, a cut that slices
an outside spoken beat in half, and overlap duplicates above the IoU and containment
caps. Narrative weakness is not a hard reject. A weak hook, a weak cliffhanger or a
resolved tail subtracts from the score and raises `low_confidence`, so the candidate
stays visible and ranked rather than silently disappearing.

**S5 Judge, `src/judge.py`, one model call per candidate batch.** One combined call
scores hook, cliffhanger and stand-alone clarity, and looks ahead past the cut to say
whether the tension resolves. The composite folds into the final score at a configured
weight, currently 0.0, and the judge's own hook and cliff scores are not used by any
sort, so S5 does not affect ranking. It removes a clip only when it and S2 both say the
tail resolves; removals are written to `judge_rejected.json`. A judge failure (parse
error, missing from a batch) is not a verdict: the clip stays, marked `low_confidence`.
The judge also writes the brief's one-line "why". Rationale: the judge failed its own pre-set ranking validity test, 34.7% pooled
win rate against a 65% bar and top-1 agreement of 4 of 10 across argument orders, so it
is treated as one noisy signal among several rather than an authority.

**S6 Select and rank, `run.py`.** Per episode, greedy diverse selection by score with
IoU and containment caps. Across the show, the top 15 by the frozen rule with at most
2 clips per episode, low-confidence candidates after the rest, and a total order that
tie-breaks down to candidate ID so two runs cannot disagree on equal scores. Selection
order is frozen **before** boundary refinement and is never re-sorted by a refined or
judge score. The two listwise LLM ranking calls run only with `--rank-mode listwise`.
If fewer than 15 clips qualify, fewer are returned and the shortfall is in the manifest.

**S7 Refine, `src/refine.py`, code only.** Boundary polish on the selected windows:
trim dead lead-in to the first real beat so the hook lands immediately (a camera opener
with a strong cue is kept), and end the cut 0.6 seconds after the last dialogue, SFX or
action beat, not the last spoken line, since the beat after the final line is often the
cliffhanger. Every op is recorded and applied only if the result passes the duration
limits and does not slice an outside spoken beat. Refinement never reorders the list.

**S8 Review, `scripts/review.py`.** Terminal reviewer that prints each clip's screenplay
beats, offset from the clip start with the first 3 seconds marked, and hides the model's
scores and "why" unless `--show-model` is passed. Approve or reject with a reason code;
`labels.csv` records decisions only, never text.

## Platform

`src/llm.py` is the only path to a model: provider routing by role (cheap, judge,
strong, alt), a token-bucket rate limiter, retry with backoff that distinguishes
per-minute from per-day quota errors, schema-validated JSON with one repair retry, and
a fallback chain. On exhausted retries an episode is marked failed and excluded from
metrics; it is not silently downgraded to the heuristic path, because comparing a
heuristic run against a model run is not a measurement. Cache keyed by stage, prompt
version, model, input hash and sample index. Run manifest records commit hash, config
hash, models, token counts and latency.

## Evaluation

Reserved test set of 5 episodes and their windows, frozen, evaluated once. Dev set of
further labelled episodes used for every tuning decision, with the labelling protocol
written down before labelling and self-agreement reported. Metrics: top-1 and top-3
window match at IoU >= 0.5 and >= 0.3, gold recall, median start and end boundary
error, each with a bootstrap confidence interval resampled over episodes, and a paired
bootstrap against the baseline. Baselines: last 45 seconds of each episode, 20 seeded
random windows, and the deterministic Tier-0 heuristic. Per-stage ablations, k-run
stability with the comparable-episode count stated, model-swap overlap, judge order
sensitivity, and cost and latency per episode.

## Specified but not built

- Cold-viewer stage as a separate blind judge. Stand-alone clarity is scored inside the
  combined judge call instead.
- S3b free-form model candidate proposals. Code search covers the space deterministically
  at zero quota cost.
- Multi-segment stitching and cross-episode beam search. Single continuous windows only.
- Bradley-Terry pairwise ranking. Removed on a call-budget estimate of 300 to 22,350
  extra calls; the pointwise composite replaces it.
- Learned ranker, embeddings, batch API, multilingual run.
