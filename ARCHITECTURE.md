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
intensity, each 0 to 3 except `r` which is boolean. Each episode is annotated k times
and the samples are aggregated per beat, median for the numeric fields and majority for
`r`; beats whose samples disagree are flagged `unstable` and that flag travels to any
candidate containing them.

**S3 Window search, `src/windows.py` and `src/search.py`, code only.** Enumerate every
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
- clarity proxies: dialogue share, distinct speaker count.

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
weight (currently set to 0.0 in `config/scoring_v2.yaml`, meaning S5 does not affect
candidate ranking order and serves strictly to generate the brief's one-line "why it works" rationale).
There is one hard judge reject, a resolved ending, and only when S2 agrees that
the tail resolves; a single model opinion is not allowed to delete the top-ranked
window. Rationale: the judge failed its own pre-set ranking validity test, 34.7% pooled
win rate against a 65% bar and top-1 agreement of 4 of 10 across argument orders, so it
is treated as one noisy signal among several rather than an authority.

**S6 Select and rank, `run.py`.** Per episode, greedy diverse selection by score with
IoU and containment caps. Across the show, the top 15 by the frozen rule with at most
2 clips per episode, low-confidence candidates after the rest, and a total order that
tie-breaks down to candidate ID so two runs cannot disagree on equal scores. Selection
order is frozen **before** boundary refinement.

**S7 Refine, `src/refine.py`, code only.** Boundary polish on the selected windows:
trim lead-in silence to the first real beat so the hook lands immediately, and move the
cut to the end of the last spoken beat plus 0.6 seconds, the measured median gap
between spoken lines. Every op is recorded and the duration limits are re-checked after
each one; an op that would break them is dropped. Refinement never reorders the list.

**S8 Review, `scripts/review.py`.** Terminal reviewer: approve or reject with a reason
code, labels appended to `labels.csv`, no dialogue printed.

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
