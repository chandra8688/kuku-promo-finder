# Promo Finder: Vertical Microdrama Promo Extraction & Show-Level Ranking
**Solution Document | Problem 1: Promo Finder**  
**Repository**: [github.com/chandra8688/kuku-promo-finder](https://github.com/chandra8688/kuku-promo-finder)

---

## 1. Problem Selected & Why
We selected **Problem 1 (Promo Finder)**. In vertical microdramas, organic discovery and paid user acquisition rely heavily on short-form promo clips (20–60s) posted on TikTok, Instagram Reels, and YouTube Shorts. However, manually scrubbing 50 to 100 script episodes per show to extract promo hooks creates an enormous editorial bottleneck. Automating this directly from plain-text screenplays unlocks instant creative generation before video production wraps, enabling high-conversion marketing tests at zero additional human cost.

---

## 2. Core Assumptions
- **3-Second Hook Rule**: A viewer scrolls in <3 seconds. The promo must feature an immediate emotional conflict, provocative line, loud SFX, or physical action in its opening 3.0s window.
- **Unresolved Cliffhanger**: The promo's final beat must pose a compelling open question or reveal without resolving the tension, compelling cold viewers to open the app.
- **Continuous Scene Constraint**: Unlike horizontal films, vertical microdrama trailers perform best as single, coherent, high-stakes scenes (20–60s). Multi-scene cross-episode montage stitching creates disorienting transitions for cold audiences.
- **Commercial Deliverable**: Evaluating per-episode clips in isolation fails the core business deliverable: *Which promos should we post first across the entire show?* The system must rank globally across all 50 episodes with diversity capping ($\le 2$ clips/episode).

---

## 3. How the Solution Works: 7-Stage End-to-End Pipeline
- **S1. Screenplay Parser (`src/parse.py`)**: Case-insensitive streaming parser converts unstructured `.txt` scripts into millisecond-accurate dramatic beats (DIALOGUE, ACTION, SFX, CAMERA). Uses regex classification (`src/speaker.py`) to decouple multi-parenthetical speaker tags from emotion qualifiers with 0 missed labels.
- **S2. Beat-Level Narrative Annotation (`src/annotate.py`)**: Gemini Flash Lite scans whole episodes once with continuity context. Annotates each beat for hook potential ($h$), open questions ($q$), tension resolution ($r$), context need ($n$), and intensity ($t$). Employs $k=3$ sample self-consistency; disputed beats are flagged `unstable=True` (0 silent heuristic fallbacks).
- **S3. Window-Level Search & Features (`src/search.py`, `src/features.py`)**: Deterministic code exhaustively evaluates all consecutive 20–60s non-camera windows. Computes whole-span features: strongest hook in first 3s (with text-cue fallback for SFX/Action), tail cliffhanger, resolution penalty, dialogue share, and intensity slope.
- **S4. Soft & Hard Gates (`src/gates.py`)**: Hard gates are strictly structural (duration 20–60s, cut straddles, overlap caps). Narrative weaknesses apply soft penalties rather than hard-dropping candidates, reducing gold window drop rates from 71% to 0%.
- **S5. Dual-Mode Judge (`src/judge.py`)**: Combined LLM prompt evaluates clip + look-ahead window. Set to `weight: 0.0` on empirical evidence to prevent judge score degradation, serving strictly to generate the brief's one-line "why it works" rationale.
- **S6. Global Show-Level Ranker (`run.py`)**: Scores all 144 surviving windows across 50 episodes, enforces episode diversity ($\le 2$ clips/episode), and outputs the definitive **Top 15 Promos** across the show.
- **S7. Dialogue Boundary Refinement (`src/refine.py`)**: Trims leading dead air and extends cuts by +0.60s past the final spoken beat (measured median speech gap), dropping median end boundary error from 8.50s to **0.60s**.

---

## 4. How It Scales to Every Show Kuku Produces
- **Linear $O(N)$ Quota & Zero Pairwise Explosion**: Prior designs proposed $O(N^2)$ Bradley-Terry pairwise comparisons (22,350 calls for 50 episodes). Our pipeline annotates once per episode and searches windows purely in code. Processing an entire 50-episode show costs **~$0.15 total** and finishes in **<3 minutes** on Gemini Flash Lite (500 RPD free tier).
- **Zero-API Deterministic Fallback**: A standalone Tier-0 heuristic mode (`--no-llm`) executes end-to-end with **0 external network calls** in <1 second, allowing instant local testing in CI/CD without API keys.
- **Genre-Agnostic Config Decoupling**: Weights and feature thresholds live in decoupled YAML (`config/scoring_v2.yaml`). Shifting from romance to thriller requires adjusting YAML weights without modifying pipeline code.
- **Total Order Reproducibility**: All sorts enforce strict tie-breaking down to candidate ID, ensuring 100% deterministic outputs across identical runs.

---

## 5. How We Know It's Good: Empirical Evaluation & Validation

### Table 1: Reserved Test Set Performance (5 Frozen Episodes, 7 Windows)
| System / Benchmark | Top-1 Match | Top-1 95% CI | Gold Recall | Recall 95% CI | Med Start Err | Med End Err |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V2 Bounded + Refine (Ours)** | **5/5 (100%)** | [100%, 100%] | **7/7 (100%)** | [100%, 100%] | **1.60s** | **0.60s** |
| B1 Baseline (Last 45s) | 5/5 (100%) | [100%, 100%] | 5/7 (71%) | [60%, 100%] | 7.50s | 5.45s |
| B3 Baseline (Tier-0 Heuristic) | 2/5 (40%) | [0%, 80%] | 5/7 (71%) | [60%, 100%] | 5.20s | 6.10s |
| B2 Baseline (20 Random Seeds) | 1.3/5 (26%) | [0%, 60%] | 4.9/7 (70%) | [14%, 100%] | — | — |

### Table 2: Widened Dev Set (25 Windows) & Human Review Precision@15
| Benchmark Metric | Pipeline (Ours) | Baseline B1 (Last 45s) | Baseline B3 (Tier-0) | Commercial Significance |
|---|:---:|:---:|:---:|---|
| **Widened Dev Top-1 (12 eps)** | **11/12 (92%)** | 12/12 (100%) | 6/12 (50%) | B1 ties Top-1 on script climax endings |
| **Widened Dev Recall (25 wins)** | **24/25 (96%)** | 12/25 (48%) | 21/25 (84%) | **B1 misses >50%** of reference promos |
| **Show Precision@15 (Human)** | **13/15 (87%)** | *n/a (cannot rank)* | *not measured* | **B1 cannot produce global show ranking** |

---

## 6. Honest Failure Modes & Where It Fails
- **Where Baseline B1 Wins**: B1 (last 45s) ties or beats our Top-1 per-episode IoU on scripts whose climax sits naturally on the final beat. B1 is hard to beat on per-episode IoU because writers intentionally place cliffhangers at episode ends. However, B1 cannot solve the commercial brief: it cannot prioritize clips across episodes and misses 52% of mid-episode gold promos.
- **Screenplay Text-Only Blindness**: Pure script analysis cannot capture visual pacing, non-verbal actor charisma, or post-production music stingers not noted in text.
- **LLM Annotation Jitter**: Single-run LLM annotation agreement across runs is 72–83% on hooks and cliffhangers. We mitigated this by aggregating over $k=3$ samples and propagating `unstable=True` flags to candidate confidence scores.
- **Optimizer Generalization**: Unconstrained weight tuning on small dev sets can overfit (e.g. flipping penalty signs). We introduced `WEIGHT_BOUNDS` capping penalties at $\le 0$ and enforcing $\ge 1.0$ floors on hook/cliff to preserve the brief's criteria.

---
**Code Integrity**: 187 unit tests passed | 0 leaks across 64,422 shingles | Full history committed in git.
