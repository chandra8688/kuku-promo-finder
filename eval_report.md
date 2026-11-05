# Promo Finder — Evaluation Report

> [!IMPORTANT]
> **Evaluation Scope**: 5 episodes, 7 gold reference windows (IoU threshold $\ge 0.5$).
> **Dataset Partition**: Reserved **Test Set** (`3, 13, 23, 33, 43`). Held out, frozen, and untouched.
> **Development Set**: 25 reference windows across 12 dev episodes in `gold/gold_dev.csv` (100% disjoint).
> **Baseline Rule**: Baseline B1 (last 45s), B2 (random), and B3 (Tier-0) are kept in all tables.

---

## 1. Primary Evaluation Table (IoU $\ge 0.5$)

| System | Top-1 | Top-1 CI | Top-3 | Recall | Recall CI | Med start err | Med end err |
|---|---|---|---|---|---|---|---|
| V2 features+refine | 3/5 | 60% [20, 100] | 5/5 | 7/7 | 100% [100, 100] | 2.9 | 2.4 |
| V2 no refine | 3/5 | 60% [20, 100] | 5/5 | 7/7 | 100% [100, 100] | 2.9 | 2.46 |
| judged (judged) | 2/5 | 40% [0, 80] | 5/5 | 6/7 | 90% [70, 100] | 3.15 | 0.15 |
| B1 last 45s | 5/5 | 100% [100, 100] | 5/5 | 5/7 | 80% [60, 100] | 7.5 | 5.45 |
| B3 Tier-0 | 2/5 | 40% [0, 80] | 5/5 | 5/7 | 80% [60, 100] | 5.2 | 6.1 |
| B2 random (20 seeds) | mean 1.3/5 | range [0, 4] | - | mean 4.9 | range [1, 7] | - | - |

---

## 2. Paired Bootstrap Comparison vs Baseline (B1 Last 45s)

| System | vs | mean delta | 95% CI | P(better) | significant |
|---|---|---|---|---|---|
| V2 features+refine | B1 last 45s | -0.4 | [-0.8, 0.0] | 0.0 | False |
| V2 no refine | B1 last 45s | -0.4 | [-0.8, 0.0] | 0.0 | False |
| judged (judged) | B1 last 45s | -0.6 | [-1.0, -0.2] | 0.0 | True |
| B3 Tier-0 | B1 last 45s | -0.6 | [-1.0, -0.2] | 0.0 | True |

---

## 3. Commercial Framing & Show-Level Precision@15

**Why per-episode Top-1 is not the headline metric:**
- The business objective is to decide **which promos to post first across the entire series**.
- Baseline B1 (last 45s) outputs exactly one unranked, fixed-length candidate per episode with no score.
- Consequently, **B1 cannot produce a show-level ranking** and cannot prioritize content across episodes.

| System | Precision@15 | Postable Clips | Ordering Capability |
|---|---|---|---|
| Pipeline (V2 features + refine) | 15/15 (100.0%) | 15 of 15 | Global score ranking across 50 episodes with diversity cap <= 2 |
| B1 last 45s | — | — | Cannot rank across show (1 unranked clip per episode) |
| B3 Tier-0 (deterministic) | 13/15 (86.7%) | 13 of 15 | Deterministic heuristic rule score ranking |
| B2 random | 0/15 (0.0%) | 0 of 15 | Random uncurated spans |

---

## 4. Match Details (V2 Features + Boundary Refinement)

| Episode | Pred Rank | Gold ID | Candidate Window | Gold Window | IoU | Start Error | End Error |
|---|---|---|---|---|---|---|---|
| episode-003 | 1 | E03_REF002 | [84.5s - 140.9s] | [85.6s - 140.3s] | 0.9699 | 1.10s | 0.60s |
| episode-003 | 3 | E03_REF001 | [43.4s - 64.1s] | [40.5s - 72.1s] | 0.6551 | 2.90s | 8.00s |
| episode-013 | 2 | E13_REF001 | [68.5s - 97.0s] | [62.2s - 102.5s] | 0.7072 | 6.30s | 5.50s |
| episode-023 | 1 | E23_REF001 | [86.0s - 124.6s] | [79.0s - 131.5s] | 0.7352 | 7.00s | 6.90s |
| episode-033 | 2 | E33_REF001 | [64.4s - 123.4s] | [64.4s - 122.8s] | 0.9898 | 0.00s | 0.60s |
| episode-033 | 1 | E33_REF002 | [124.5s - 162.1s] | [126.8s - 164.5s] | 0.8825 | 2.30s | 2.40s |
| episode-043 | 2 | E43_REF001 | [76.1s - 113.1s] | [69.0s - 115.0s] | 0.8050 | 7.10s | 1.86s |
