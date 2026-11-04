# REVIEW.md — Adversarial Design Review of ARCHITECTURE.md

## Executive Summary
This review critically examines the proposed architecture for `promo-finder` against the empirical realities of the 50 microdrama screenplay episodes in `data/raw/` and the practical constraints of the Kuku AI Generalist Assignment (free API tier, ~15 requests per minute, 2–3 hour scope).

The current design is conceptually thorough but architecturally bloated. In its current form, **it would require between 1,345 and 2,675 LLM calls to process all 50 episodes**, resulting in an execution time of **1.5 to 3.0 hours** at 15 RPM—exhausting daily free-tier rate limits and failing the assignment's mandate for a lightweight, reliable prototype. Furthermore, core assumptions regarding spoken hooks in the first 3 seconds and multi-segment cross-episode beam search contradict empirical screenplay data.

---

## (a) Facts Verified from Real Data (`data/raw/`)

All statistics below were computed directly from the 50 screenplay text files in `data/raw/`. No screenplay dialogue or confidential story content has been leaked.

| Metric | Measured Value | Computation Rule / Regular Expression |
|---|---|---|
| **Total Episodes** | `50` | Count of `.txt` files in `data/raw/` matching `episode-*.txt`. |
| **Episode Duration Range** | `70.50s – 200.75s` (1.18m – 3.35m) | Min and max timestamp of final beat: `End:\s*\(?(\d+):(\d+\.?\d*)\)?` converted via `int(m)*60 + float(s)`. (Matches `Scene end:` exactly across all files). |
| **Episode Duration Median** | `117.78s` (~1.96m) | Median of the 50 episode duration values. |
| **Total Characters Across All 50 Files** | `528,811` chars | Total character count across all files (`len(f.read())`). |
| **Episode Character Count (Median / Longest)** | Median: `10,567.5` chars<br>Max: `15,835` chars | Episode 3 (`episode-003-Episode 1671705.txt`) is the longest at 15,835 characters. |
| **Total Beats Across 50 Files** | `3,727` beats | Count of timing blocks matching `Start:\s*\(?(\d+):(\d+\.?\d*)\)?\s*\|\s*End:\s*\(?(\d+):(\d+\.?\d*)\)?`. |
| **Total Beats Per Episode (Median / Max)** | Median: `73.0` beats<br>Max: `121` beats (Min: `44`) | Distribution of beat counts per file. |
| **Share: CAMERA Beats** | `1,754` beats (**47.06%**) | Preceding header starts with `**CAMERA` (e.g., `**CAMERA:**`). |
| **Share: DIALOGUE Beats** | `1,487` beats (**39.90%**) | Preceding header is a character name, e.g., `**CHARACTER (Emotion):**` or `**CHARACTER:**`. |
| **Share: ACTION Beats** | `372` beats (**9.98%**) | Preceding header starts with `**ACTION` (e.g., `**ACTION:**`). |
| **Share: SOUND FX (SFX) Beats** | `114` beats (**3.06%**) | Preceding header starts with `**SOUND FX` (e.g., `**SOUND FX:**`). Counted strictly separate from dialogue. |
| **Episodes with Spoken Line in First 3s** | `17 / 50` (**34.0%**) | Episodes where earliest DIALOGUE beat has `start_s <= 3.0s`. (33 of 50 episodes start with CAMERA or ACTION). |
| **Median Gap Between Consecutive Spoken Lines** | `0.60s` (Mean: `1.31s`) | For all 1,437 consecutive dialogue pairs $(D_i, D_{i+1})$ in an episode: `gap = start_s_{i+1} - end_s_i`. |

*Illustrative sanitized header examples:*
- CAMERA beat: `**CAMERA:**` followed by `Start: (00:00.00) | End: (00:04.80)`
- DIALOGUE beat: `**LADY JULIANA (Confused, Breathless):**` followed by `Start: (00:12.40) | End: (00:14.20)`
- SFX beat: `**SOUND FX:**` followed by `Start: (00:45.00) | End: (00:46.50)`

---

## (b) Top 10 Risks Ranked by Severity

### 1. Severe API Rate Limit & Quota Exhaustion in Verification (S5) & Ranking (S6) [CRITICAL]
- **Risk:** Running 3 separate judges with $2\times$ sampling plus $O(N^2)$ Bradley-Terry pairwise comparisons across 50 episodes generates 1,345 to 2,675 API calls. On a standard free tier (~15 RPM, 1,500 daily requests), this run will take 1.5 to 3 hours or crash with HTTP 429 quota exhaustion.
- **Mitigation:** Eliminate pairwise LLM ranking completely. Consolidate S5 into a single combined Judge prompt (evaluating hook, cliffhanger, and clarity in one pass with structured JSON) with single sampling. Pointwise score = $1.5 \times Hook + 2.0 \times Cliff + 1.0 \times Clarity$. Reduces calls by ~85%.

### 2. High False Rejection Rate From the 3-Second Hook Rule (S4) [HIGH]
- **Risk:** 66% of episodes (33 out of 50) have NO spoken dialogue in their first 3 seconds, and nearly half (47.06%) of all screenplay beats are CAMERA directions. S4 hard-gating candidates on a spoken hook in the first 3 seconds of an episode will prematurely discard high-tension scenes.
- **Mitigation:** Clarify that the "first 3 seconds" applies to the *candidate clip's relative start window*, not the episode's absolute 00:00 timestamp. Furthermore, allow high-impact CAMERA movements (e.g., rapid zoom, whip pan) and SFX (e.g., slap, gunshot stinger) to qualify as valid hooks alongside dialogue.

### 3. Whole-Episode Dense Beat Annotation (S2) Schema Failure & JSON Truncation [HIGH]
- **Risk:** Prompting an LLM to output a JSON array annotating 73 to 121 individual beats per episode requires ~2,200 output tokens. Free-tier models will routinely truncate, hallucinate beat IDs, or drop closing brackets, breaking downstream JSON parsing across all 50 episodes.
- **Mitigation:** Strip out CAMERA beats (47% of total) before prompting S2, or replace dense per-beat annotation with scene-level tension/hook scoring. Have the LLM return only the IDs of top candidate beats rather than an exhaustive array for all 120 beats.

### 4. Narrative Incoherence from Multi-Segment Cross-Episode Beam Search (S3a) [HIGH]
- **Risk:** Microdrama ads on TikTok/ReelShort are overwhelmingly single continuous scenes (20–45s) ending on a dramatic climax. Stitching 2–3 non-contiguous segments across episode boundaries creates disjointed, incomprehensible montages that confuse cold viewers and explode beam search complexity.
- **Mitigation:** Enforce single continuous windows as the primary candidate generator. Restrict multi-segment stitching to within the same scene, and disable cross-episode stitching by default unless an explicit CLI flag `--allow-cross-episode` is set.

### 5. Incoherent Cutpoints Truncating Spoken Lines (S1 & S3a) [MEDIUM]
- **Risk:** Screenplay beats slice lines at exact recorded boundaries. Slicing strictly on beat boundaries can clip trailing audio or cut mid-sentence if dialogue overlaps with an action beat or if natural delivery extends slightly past the nominal timestamp.
- **Mitigation:** In S1/S3a, snap window start boundaries to the start of the initial dialogue/sound beat, and extend the cliffhanger cut point through the end of the final dialogue beat plus an acoustic padding buffer of 0.6s (the empirical median dialogue gap).

### 6. Duplication and Competition Between S3a (Window Search) and S3b (Editor Proposals) [MEDIUM]
- **Risk:** Running both exhaustive algorithmic window search (S3a) and free-form LLM candidate generation (S3b) generates duplicated or slightly offset candidate intervals (e.g., Beats 12–25 vs 13–26). Unioning them balloons candidate count before S5, doubling the verification load.
- **Mitigation:** Make S3a algorithmic window search the primary generator (0 LLM calls, deterministic, fast). Demote S3b to a fallback mode or ablation flag, running it only if S3a finds zero candidates meeting minimum heuristic thresholds.

### 7. Subjectivity & Disagreement Deadlocks in S5 Multi-Judge Verification [MEDIUM]
- **Risk:** S5 mandates that if judges disagree, the candidate is routed to `needs_human`. Microdrama quality is inherently subjective; LLM judges will routinely disagree on nuanced "confusion scores", dumping 40%+ of candidates into an unreviewed human queue and failing batch automation.
- **Mitigation:** Replace unanimous consensus with weighted scalar averaging. If the composite score exceeds a minimum threshold (e.g., $\ge 3.5 / 5.0$), accept the candidate; mark borderline candidates with an audit flag rather than halting automated execution.

### 8. Dynamic State Mutation During Cold Viewer Retesting (S5) [MEDIUM]
- **Risk:** The design states: "add context segment and retest once". Dynamically expanding candidate boundaries during verification turns a read-only evaluation stage into a mutation stage, risking violation of S4's 60-second upper duration gate.
- **Mitigation:** Keep verification strictly read-only. If a candidate lacks stand-alone clarity, penalize its clarity score or reject it outright. Do not dynamically mutate clip windows inside the judge loop.

### 9. Show-Specific Overfitting to Regency / Cathedral Tropes [LOW]
- **Risk:** The 50 episodes in `data/raw/` all come from a single costume/historical microdrama ("Lady Juliana"). Prompts tuned to specific character names, titles, or cathedral wedding tropes will fail when evaluated on contemporary billionaire or revenge microdramas.
- **Mitigation:** Abstract prompt instructions to generic dramatic primitives: power reversals, shocking physical acts, social humiliations, and abrupt cliffhangers.

### 10. Combinatorial Explosion in Ablation Testing (Principle 3) [LOW]
- **Risk:** "Every stage is behind a config flag and must justify itself by ablation." Running full ablations over 7 stages ($2^7 = 128$ configurations) across 50 episodes is physically impossible within the 48-hour assignment window on a free API tier.
- **Mitigation:** Pre-define exactly three ablation tiers in `config.yaml`:
  1. *Baseline Heuristic* (S1 + S3a rule-based ranking; 0 LLM calls).
  2. *Standard Fast* (S1 + S2 scene annotations + S3a + 1-pass Judge; ~100 calls).
  3. *Full Production* (all stages enabled).

---

## (c) Call and Token Budget Table (50 Episodes)

### Baseline Assumptions:
- **Episode Count:** 50 episodes.
- **Average Script Size:** 10,576 characters ($\approx 2,644$ tokens at 4 characters/token).
- **Prompt Overhead:** System instructions + few-shot examples $\approx 2,000$ characters ($\approx 500$ tokens).
- **Surviving Candidates:** S3a + S3b produce 3 surviving candidates per episode after S4 hard gates = **150 candidates total across 50 episodes**.
- **Clip Excerpt Size:** Average promo clip is ~40 seconds ($\approx 25$ beats $\approx 3,000$ characters $\approx 750$ tokens).
- **Pairwise Ranking (S6):** Evaluated strictly *intra-episode* among the 3 candidates per episode. Number of pairs per episode = $\binom{3}{2} = 3$ pairs. With order swapped ($2\times$) = **6 comparisons per episode $\times 50$ episodes = 300 comparisons total**. (Note: Cross-episode global ranking would require $\binom{150}{2} \times 2 = 22,350$ calls, which is completely unfeasible).
- **Cold-Viewer Retest:** Assumes 30% of candidates (45 candidates) trigger context addition and 1 retest call.
- **Token Estimation Rule:** 1 token $\approx$ 4 characters for both input and output.

| Stage | Calls per Episode | Total Calls (50 Eps) | Input Tokens / Call | Output Tokens / Call | Total Input Tokens | Total Output Tokens | Total Tokens | Feasibility @ 15 RPM |
|---|---|---|---|---|---|---|---|---|
| **S2: Beat Annotation** | 1 | 50 | 3,144 | 2,190 | 157,200 | 109,500 | 266,700 | 3.3 min |
| **S3a: Window Search** | 0 *(Code)* | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 min (Instant) |
| **S3b: Editor Proposals** | 1 | 50 | 3,144 | 375 | 157,200 | 18,750 | 175,950 | 3.3 min |
| **S5: Cliffhanger Look-Ahead (x2)** | 6 | 300 | 1,200 | 100 | 360,000 | 30,000 | 390,000 | 20.0 min |
| **S5: Cold Viewer (x2)** | 6 | 300 | 1,000 | 100 | 300,000 | 30,000 | 330,000 | 20.0 min |
| **S5: Cold-Viewer Retest (1x on 30%)** | 0.9 | 45 | 1,200 | 100 | 54,000 | 4,500 | 58,500 | 3.0 min |
| **S5: Hook-Payoff Coherence (x2)** | 6 | 300 | 1,000 | 100 | 300,000 | 30,000 | 330,000 | 20.0 min |
| **S6: Pairwise Ranking (Order Swapped)** | 6 | 300 | 1,750 | 75 | 525,000 | 22,500 | 547,500 | 20.0 min |
| **TOTAL PIPELINE** | **26.9** | **1,345** | — | — | **1,853,400** | **245,250** | **2,098,650** | **89.7 minutes (~1.5 hours)** |

*(Note: If 5 candidates survive per episode instead of 3, total calls surge to **2,675 calls**, total tokens reach **4,115,150**, and runtime becomes **178.3 minutes / 2.97 hours**).*

### Bottleneck Identification:
- **Primary Bottleneck:** **Stage S6 Pairwise Ranking** (300 to 1,000 calls, 20 to 67 minutes) and **Stage S5 Multi-Judge Verification** (945 calls, 63 minutes).
- Together, **S5 and S6 account for 92.5% of all LLM calls** in the pipeline.
- At 15 RPM, sequential execution takes **~90 minutes** for 3 candidates/ep and **~180 minutes** for 5 candidates/ep, which exceeds the continuous uptime reliability of free APIs and violates the assignment's guidance to keep solutions lean and runnable.

---

## (d) Contradictions and Ambiguities in the Design

1. **Principle 1 vs. S5 Cliffhanger Look-Ahead Mutation:**
   - *Principle 1:* "LLM never outputs timestamps; it references beat IDs and code converts."
   - *S5 Specification:* "resolved? move cut earlier or reject".
   - *Contradiction:* How does the LLM "move cut earlier"? If it outputs seconds, it breaks Principle 1. If it outputs a beat ID, how does the look-ahead prompt supply valid beat IDs outside the clip without passing the full episode?
2. **S5 Retest Mutation vs. S4 Hard Gates:**
   - S5 Cold Viewer permits: "add context segment and retest once". If context is prepended to a 55-second promo clip, the resulting clip length will exceed the 60-second hard limit enforced in S4. The design does not specify whether mutated clips re-trigger S4 gates.
3. **S3a Scoring vs. S3b Free-Form Proposals:**
   - S3a uses a mathematical objective function on beat boundaries. S3b generates free-form candidate intervals. If S3b candidates are unioned with S3a, how are S3b candidates scored before S4/S5? Do they bypass S3a scoring, or does S3a evaluate them retroactively?
4. **Intra-Episode vs. Global Ranking Ambiguity in S6:**
   - The design specifies "pairwise comparisons with order swapped, fit Bradley-Terry", but omits whether ranking is performed *within each episode* or *across all 50 episodes*. Cross-episode global ranking ($\binom{150}{2} \times 2 = 22,350$ calls) is physically impossible under free-tier limits, while intra-episode ranking fails to answer the user's primary prompt: *"Which promos would you post first across the show?"*
5. **Beat Boundary Granularity vs. Strict 20–60s Duration:**
   - In microdramas, individual beats vary from 1.0s to 6.5s. Combinations of beats will frequently sum to 19.4s or 60.5s. S4 strictly rejects clips outside 20–60s with zero tolerance. Without micro-trimming rules, valid dramatic sequences will be discarded due to a 0.5s margin error.

---

## (e) What to Cut First If Running Out of Time (In Strict Order)

1. **Cut 1: S6 Bradley-Terry Pairwise Ranking [Saves 300–1,000 calls / 20–67 min]**
   - *Reason:* Replace with deterministic pointwise composite scoring ($1.5 \times Hook + 2.0 \times Cliff + 1.0 \times Clarity$). Completely eliminates the single biggest latency bottleneck while still producing an objective global rank ordering.
2. **Cut 2: S5 $2\times$ Sampling and Triplicate Separate Judges [Saves 600–1,000 calls / 40–67 min]**
   - *Reason:* Consolidate Cliffhanger, Cold Viewer, and Hook-Payoff into a single, unified Judge evaluation prompt with single sampling ($1\times$). Emits all 3 scores in one structured JSON response per candidate.
3. **Cut 3: S5 Dynamic Cold-Viewer Retest Loop [Saves 45–75 calls / 3–5 min]**
   - *Reason:* Eliminate dynamic window mutation during evaluation. If a clip is not intelligible to a cold viewer, penalize its clarity score. Keeps S5 strictly read-only and eliminates stateful boundary bugs.
4. **Cut 4: S3b Free-Form LLM Editor Proposals [Saves 50 calls / 3.3 min]**
   - *Reason:* S3a algorithmic window search already scans all continuous 20–60s candidate windows deterministically at 0 API cost. Dropping S3b eliminates candidate duplication and saves 50 full-episode LLM calls.
5. **Cut 5: S3a Multi-Segment Stitching & Cross-Episode Beam Search [Saves massive code complexity]**
   - *Reason:* Restrict S3a to single continuous windows within single episodes. Splicing across non-contiguous scenes creates jarring transitions that cold viewers dislike and adds significant engineering overhead with minimal quality gain.

---

## (f) Recommended Build Order

Following the assignment's mandate ("a simple, honest submission that works... pick something narrow... spend 2 to 3 hours"):

```
Milestone 1: Deterministic Screenplay Parser & Featurizer (S1)
└── Milestone 2: Algorithmic Window Generator & S4 Hard Gates (S3a + S4)
    └── Milestone 3: Centralized LLM Gateway (src/llm.py)
        └── Milestone 4: Consolidated Pointwise Judge (Simplified S5)
            └── Milestone 5: Heuristic Ranker & Formatted Output Generator
                └── Milestone 6 (Optional): Dense Beat Annotation (S2)
```

1. **Milestone 1: Deterministic Screenplay Parser & Featurizer (S1)**
   - Pure Python. Parse the 50 `.txt` files into structured beat dictionaries (`beat_id`, `scene`, `kind`, `start_s`, `end_s`, `speaker`, `text`).
   - Validate timestamps and emit `parse_report.json`. (Zero LLM calls).
2. **Milestone 2: Algorithmic Window Search & Hard Gates (S3a + S4)**
   - Pure Python. Slide windows on beat boundaries between 20.0s and 60.0s.
   - Enforce hard gates: 20–60s duration, hook in relative 0–3s window, dedupe overlapping windows (>50% IoU).
   - Generates our baseline candidates (e.g. last 45s of each episode) instantly. (Zero LLM calls).
3. **Milestone 3: Centralized LLM Gateway (`src/llm.py`)**
   - Enforce single entrypoint with rate-limiting token bucket (15 RPM), exponential backoff retry, and cost/token tracking.
4. **Milestone 4: Consolidated Pointwise Judge (Simplified S5)**
   - A single LLM call per candidate clip assessing:
     1. Hook score (0–3)
     2. Cliffhanger score (0–3)
     3. Stand-alone clarity (0–3)
     4. One-line why
   - Pointwise composite score calculated deterministically in code.
5. **Milestone 5: Formatted Output Generator (`outputs/`)**
   - Output structured promos matching the Kuku problem brief:
     - Episode name / number
     - Start and end timestamps of each segment in order
     - Hook description
     - Cliffhanger description
     - One-line why it works
     - Rank order across the entire show
6. **Milestone 6 (Optional Ablation): Add S2 Beat Annotation**
   - Add S2 only after Milestones 1–5 are verified working end-to-end, evaluating whether S2 improves candidate selection over S3a algorithmic search.

