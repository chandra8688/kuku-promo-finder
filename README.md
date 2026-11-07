# Promo Finder

Reads the screenplays of a vertical microdrama and returns a ranked list of promo clips:
each one 20–60 s, opening on a hook in its first 3 seconds, ending on an unresolved
moment, and understandable without the rest of the show. Every promo comes with its
segment times, hook line, cliffhanger line, a one-line reason it works, and why it ranks
where it does.

Built for the Kuku AI Generalist assignment. The show's scripts are confidential and are
**not** in this repo; `examples/demo_show/` is a made-up 3-episode show you can run instead.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py --input examples/demo_show --out outputs_demo --no-llm   # no API key needed
```

With a free Gemini key (Google AI Studio) in `.env` as `GEMINI_API_KEY=...`:

```bash
python run.py --input examples/demo_show --out outputs_demo           # ~7 calls, ~20 s
python -m scripts.validate_promos --run outputs_demo                   # checks every promo against the brief's rules
```

To run on a new show, point `--input` at a folder of its `.txt` scripts. Nothing else changes.
Expected output for the demo show is in `examples/demo_output/`.

## How it works

```
scripts (.txt) -> 1 parse -> 2 annotate beats (LLM) -> 3 search windows -> 4 judge (LLM) -> 5 rank show (+1 LLM call)
```

1. **Parse** (`src/parse.py`): every camera, action, sound and dialogue element with its start and end time.
   Tolerates header and timestamp format drift; merges name variants such as `LADY X` / `X` automatically.
2. **Annotate** (`src/annotate.py`): one call per episode scores each beat 0–3 for hook, open question,
   context needed, intensity and whether it resolves tension. The model returns **beat IDs, never timestamps**;
   code turns IDs into times, so a timestamp can't be invented.
3. **Search** (`src/windows.py`, `features.py`, `score.py`, `gates.py`): every 20–60 s window is scored
   (hook in the first 3 s, cliffhanger at the end, intensity, context penalty) and windows that cut a spoken
   line or break a rule are rejected.
4. **Judge** (`src/judge.py`): one call per episode; a "cold viewer" prompt scores the top windows for hook,
   cliffhanger, clarity and payoff, and whether what follows resolves the clip.
5. **Rank the show** (`src/showrank.py`): a promo leads the list only if **both** models rate its hook and
   cliffhanger at least 2/3 and neither says it resolves itself. Order = agreed hook + agreed cliffhanger +
   clarity + payoff + a small bonus for early episodes (a new viewer starts at episode 1).
   One call (`src/finalists.py`) labels each finalist's hook type and writes its "why it works" line;
   the first five posts are spread across at least three hook types. Weights: `config/show_rank.yaml`.

Every model call goes through `src/llm.py`: caching, rate limits, retries, schema repair, fallback models.
More detail: `ARCHITECTURE.md`.

## Swapping the model

Models are set per role in `config/models.yaml` (`cheap` annotates, `judge` judges, `strong` writes the
finalist lines). For any OpenAI-compatible API, change one role:

```yaml
  judge:
    provider: "openai_compatible"
    base_url: "https://api.your-provider.com/v1"
    api_key_env: "YOUR_KEY_ENV_VAR"
    model_name: "your-model"
```

## Outputs (`--out` folder)

| File | What it holds |
|---|---|
| `promos_ranked.csv` / `.json` | Top 15 across the show: rank, wave, segments, hook, cliffhanger, why, rank reason, all scores |
| `promos_per_episode.csv` / `.json` | Every surviving promo, ranked within its episode |
| `run_manifest.json` | Models, config hash, calls, tokens, time |
| `aliases_auto.json` | Speaker name variants that were merged |

## How good is it (on the assignment's 50 episodes)

| Check | Result |
|---|---|
| Brief's hard rules on the top 15 (`scripts/validate_promos.py`) | 15/15 on length, hook in 3 s, no mid-line cut, unresolved ending |
| Blind rating, top 15 vs last-45-seconds clips, 1–5 (`eval/blind_rating.json`) | hook 4.20 vs 3.86 · cliffhanger 3.93 vs 3.57 · standalone 3.93 vs 3.71 · would post 12/15 vs 11/14 |
| Ranking order, same rating | ranks 1–5: 5/5 would post, hook 4.8 · ranks 6–15: 7/10, hook 3.9 |
| Endings the writers call "cliffhanger" in their own notes (`eval/eval_endings.json`) | 12/12 flagged by the annotator |
| New, unseen show with format drift (`examples/demo_show`) | runs end to end; 6/6 promos pass the rules |
| Cost of a full run | ~102 calls, ~390k input tokens, ~7 min on the free tier; re-runs from cache in ~30 s |

Caveats: the blind rating has one rater (the author, who knows the show) and ~15 clips per side.
Earlier reference-window metrics and baselines are in `eval/eval_report.md`.

## Known limits

- Text only: a hook that is purely visual is judged from its camera and action description.
- Long clips can sag in the middle; the score checks the opening and the ending, not the pacing between.
- One continuous segment per promo; multi-segment stitching is not built.
- Annotations vary between runs (same top clip in 39/50 episodes across two runs).

## Repo map

`run.py` CLI · `src/` pipeline · `config/` models, weights, ranking · `prompts/` all prompts ·
`scripts/` checks and evaluation · `eval/` results (numbers only) · `examples/` demo show and output ·
`tests/` 225 tests, including a leak check that blocks script text and character names from being committed.
