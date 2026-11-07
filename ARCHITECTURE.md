# Architecture

Design decisions, in the order they matter. `README.md` has the pipeline overview.

## 1. Code owns time; the model owns judgment

The parser turns each script into timed beats. Models only ever see beats by local
index and return indices; code maps them back to seconds. A prompt never asks for a
timestamp (checked in `src/judge.py`), so a hallucinated time cannot reach the output.

## 2. Annotate once, search exhaustively

One annotation call per episode scores every non-camera beat (0–3: hook, open
question, context needed, intensity; resolves-tension flag). Code then scores every
20–60 s window from those beat scores: `src/features.py` (hook in the window's first
3 s, cliffhanger in its last 6 s, context need in its first 5 s, intensity shape),
`src/score.py` (weights in `config/scoring_v2.yaml`, fitted on dev episodes only, with
penalty signs fixed), `src/gates.py` (reject cuts through a spoken line; length;
soft flags for weak hook or cliff), `src/refine.py` (trim dead air at both ends).
About 2 model calls per episode, whatever the number of windows.

## 3. Two models must agree before a promo leads

Inside an episode, the window score picks the best few clips. Across the show it
cannot: most finalists hit the top of the 0–3 scale, so its order there followed
position-in-episode. `src/showrank.py` therefore ranks the show by the **agreed**
score: the lower of the annotator's and the judge's hook and cliffhanger. A promo
leads only if both are at least 2/3 and neither model says it resolves itself.
Then clarity, payoff and an early-episode bonus (`config/show_rank.yaml`).
Each promo gets a `rank_reason` built from these numbers.

## 4. One extra call for the words a marketer needs

`src/finalists.py` sends the top 25 finalists, as text, to the `strong` role once:
hook type and a one-line "why it works". Wave 1 (first five posts) is spread over
at least three hook types, so the first results also say which hook this audience
stops for. If the strong model is busy it falls back to the judge model; if every
model fails, the code-built reasons are used.

## 5. One gateway for every call

`src/llm.py`: providers `gemini`, `openai_compatible`, `mock`; per-role config in
`config/models.yaml`; disk cache keyed on model + prompt; per-minute and per-day
quota tracking; retries with backoff; JSON schema validation with one repair pass;
fallback chain on 503; a call log per run in `runs/`.

## 6. Confidential input stays out of git

`data/`, outputs and caches are gitignored. `tests/test_leak.py` fails if any tracked
file shares an 8-word run with the scripts or contains a character name.

## Not built (and why)

- Multi-segment stitching: the brief says promos are usually one continuous stretch.
- Video or audio analysis: no footage was provided.
- Pacing inside long clips: the score reads the opening and the ending only.
- A learned ranker: needs editor decisions and ad results to learn from;
  `scripts/review.py` collects the first.
