# Promo Finder — Evaluation Report (superseded)

> **Superseded.** This report describes an earlier pipeline run (`outputs_final`, before the show-level ranking and the reveal-ending rule). Current evidence: `eval/blind_rating.json`, `eval/eval_endings.json` and the README's results table. It is kept for its baselines and stability numbers.

> [!IMPORTANT]
> **Evaluation Scope**: 5 episodes, 7 reference windows (IoU threshold $\ge 0.5$).
> **Reference provenance**: 7 `ai_reviewed`. 7 of 7 windows were not verified by a person, so these numbers measure agreement with an AI-assisted reference, not with human editors.
> **Dataset Partition**: Reserved **Test Set** (`3, 13, 23, 33, 43`). Held out, frozen, and untouched.
> **Development Set**: reference windows in `gold/gold_dev.csv`, disjoint from these episodes. Weights were fitted on it, so any dev-set number is in-sample.
> **Baseline Rule**: Baseline B1 (last 45s), B2 (random), and B3 (Tier-0) are kept in all tables.

---

## 1. Primary Evaluation Table (IoU $\ge 0.5$)

| System | Top-1 | Top-1 CI | Top-3 | Recall | Recall CI | Med start err | Med end err |
|---|---|---|---|---|---|---|---|
| V2 features+refine | 4/5 | 80% [40, 100] | 4/5 | 6/7 | 86% [62, 100] | 5.55 | 4.98 |
| V2 no refine | 4/5 | 80% [40, 100] | 4/5 | 6/7 | 86% [62, 100] | 5.55 | 4.98 |
| B1 last 45s | 5/5 | 100% [100, 100] | 5/5 | 5/7 | 71% [56, 100] | 7.5 | 5.45 |
| B3 Tier-0 | 2/5 | 40% [0, 80] | 5/5 | 5/7 | 71% [56, 100] | 5.2 | 6.1 |
| B2 random (20 seeds) | mean 1.3/5 | range [0, 4] | - | mean 4.9 | range [1, 7] | - | - |

---

## 2. Paired Bootstrap Comparison vs Baseline (B1 Last 45s)

| System | vs | mean delta | 95% CI | P(better) | significant |
|---|---|---|---|---|---|
| V2 features+refine | B1 last 45s | -0.2 | [-0.6, 0.0] | 0.0 | False |
| V2 no refine | B1 last 45s | -0.2 | [-0.6, 0.0] | 0.0 | False |
| B3 Tier-0 | B1 last 45s | -0.6 | [-1.0, -0.2] | 0.0 | True |

---

## 3. Commercial Framing & Show-Level Precision@15

**Why per-episode Top-1 is not the headline metric:**
- The business objective is to decide **which promos to post first across the entire series**.
- Baseline B1 (last 45s) outputs exactly one unranked, fixed-length candidate per episode with no score.
- Consequently, **B1 cannot produce a show-level ranking** and cannot prioritize content across episodes.

| System | Precision@15 | Labelled By | Ordering Capability |
|---|---|---|---|
| Pipeline (V2 features + refine) | 13/15 (87%, CI [67, 100]) | human review via scripts/review.py | Global score ranking across 50 episodes with diversity cap <= 2 |
| B1 last 45s | n/a | - | Cannot rank across show (1 unranked clip per episode) |
| B3 Tier-0 (deterministic) | not labelled | - | Deterministic heuristic rule score ranking |

---

## 4. Match Details (V2 Features + Boundary Refinement)

| Episode | Pred Rank | Gold ID | Candidate Window | Gold Window | IoU | Start Error | End Error |
|---|---|---|---|---|---|---|---|
| episode-003 | 5 | E03_REF002 | [90.4s - 129.9s] | [85.6s - 140.3s] | 0.7221 | 4.80s | 10.40s |
| episode-013 | 1 | E13_REF001 | [68.5s - 107.1s] | [62.2s - 102.5s] | 0.7572 | 6.30s | 4.60s |
| episode-023 | 1 | E23_REF001 | [86.0s - 131.8s] | [79.0s - 131.5s] | 0.8617 | 7.00s | 0.30s |
| episode-033 | 2 | E33_REF001 | [64.4s - 123.4s] | [64.4s - 122.8s] | 0.9898 | 0.00s | 0.60s |
| episode-033 | 1 | E33_REF002 | [124.5s - 174.6s] | [126.8s - 164.5s] | 0.7519 | 2.30s | 10.14s |
| episode-043 | 1 | E43_REF001 | [76.1s - 109.6s] | [69.0s - 115.0s] | 0.7289 | 7.10s | 5.36s |

## Evaluation history of the reserved test set

The reserved set has been evaluated five times. Rows 1 to 4 used annotation run A (`data/annotated`). Row 5 used annotation run B (`outputs_final/annotated`, the run that produced the shipped promos), with the code and weights of row 4 unchanged; nothing was tuned on it.

| # | Commit | Change | V2 Top-1 | V2 Recall | Med end err |
|---|---|---|---|---|---|
| 1 | 78e50fd | default weights | 5/5 (100%) | 5/7 (71%) | 3.95s |
| 2 | 4920b78 | unconstrained dev fit | 3/5 (60%) | 7/7 (100%) | 2.40s |
| 3 | fef6b3b | bounded dev fit | 5/5 (100%) | 7/7 (100%) | 0.60s |
| 4 | e01ebbf | bounded fit, single engine and config (this revision) | 4/5 (80%) | 6/7 (86%) | 4.98s |
| 5 | 348acd6 | row 4 unchanged, annotation run B (shipped run) | 5/5 (100%) | 7/7 (100%) | 0.60s |

## Reserved test set, annotation run B (shipped run)

Generated from `eval_test_run_b.json` (same code and weights as row 4; annotations from `outputs_final/annotated`).

| System | Top-1 | Top-1 CI | Top-3 | Recall | Recall CI | Med start err | Med end err |
|---|---|---|---|---|---|---|---|
| V2 features+refine | 5/5 | 100% [100, 100] | 5/5 | 7/7 | 100% [100, 100] | 7.1 | 0.6 |
| V2 no refine | 5/5 | 100% [100, 100] | 5/5 | 7/7 | 100% [100, 100] | 7.1 | 0.3 |
| B1 last 45s | 5/5 | 100% [100, 100] | 5/5 | 5/7 | 71% [56, 100] | 7.5 | 5.45 |
| B3 Tier-0 | 2/5 | 40% [0, 80] | 5/5 | 5/7 | 71% [56, 100] | 5.2 | 6.1 |
| B2 random (20 seeds) | mean 1.3/5 | range [0, 4] | - | mean 4.9 | range [1, 7] | - | - |
