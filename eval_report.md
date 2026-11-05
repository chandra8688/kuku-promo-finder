# Promo Finder — Evaluation Report

> [!IMPORTANT]
> **Evaluation Scope**: 4 episodes, 4 gold reference windows (IoU threshold $\ge 0.5$).
> **Dataset Source**: `C:\Users\CHANDR~1\AppData\Local\Temp\tmpmxkdo6mf\gold_synthetic.csv`.
> **Partition**: Development benchmark for validation and tuning.

---

## 1. Primary Evaluation Table (IoU $\ge 0.5$)

| System | Top-1 | Top-1 CI | Top-3 | Recall | Recall CI | Med start err | Med end err |
|---|---|---|---|---|---|---|---|
| V2 features+refine | 4/4 | 100% [100, 100] | 4/4 | 4/4 | 100% [100, 100] | 0.0 | 18.18 |
| V2 no refine | 4/4 | 100% [100, 100] | 4/4 | 4/4 | 100% [100, 100] | 0.0 | 18.17 |
| B1 last 45s | 0/4 | 0% [0, 0] | 0/4 | 0/4 | 0% [0, 0] | None | None |
| B3 Tier-0 | 0/4 | 0% [0, 0] | 4/4 | 4/4 | 100% [100, 100] | 1.5 | 0.0 |
| B2 random (20 seeds) | mean 0.5/4 | range [0, 2] | - | mean 3.8 | range [2, 4] | - | - |

---

## 2. Paired Bootstrap Comparison vs Baseline (B1 Last 45s)

| System | vs | mean delta | 95% CI | P(better) | significant |
|---|---|---|---|---|---|
| V2 features+refine | B1 last 45s | 1.0 | [1.0, 1.0] | 1.0 | True |
| V2 no refine | B1 last 45s | 1.0 | [1.0, 1.0] | 1.0 | True |
| B3 Tier-0 | B1 last 45s | 0.0 | [0.0, 0.0] | 0.0 | False |

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
| episode-901 | 1 | F901_REF001 | [13.2s - 71.9s] | [13.2s - 45.7s] | 0.5530 | 0.00s | 26.23s |
| episode-902 | 1 | F902_REF001 | [30.8s - 87.5s] | [30.8s - 69.0s] | 0.6743 | 0.00s | 18.46s |
| episode-903 | 1 | F903_REF001 | [16.8s - 63.1s] | [16.8s - 57.0s] | 0.8676 | 0.00s | 6.12s |
| episode-904 | 1 | F904_REF001 | [35.4s - 60.2s] | [35.4s - 78.1s] | 0.5802 | 0.00s | 17.89s |
