# Gold-Set Evaluation Report

> [!IMPORTANT]
> **Gold Set Provenance**: AI-proposed, AI-reviewed, human-unverified.
> Evaluated across the 5 gold episodes (`3, 13, 23, 33, 43`), total 7 gold windows.
>
> **Ranking Rule Clarification**: "Pipeline Final Ranking" in earlier output referred to row (3) Judge-ranked.
> The frozen adopted rule is row (2) **S3a order + judge gate**, which is the exact ranking used by `run.py` and `outputs/promos_per_episode.*`.
>
> **Gate Removal Impact**: The judge gate removed the S3a rank-1 window in **2 of 5** gold episodes (`episode-003` and `episode-033`) due to `JUDGE_RESOLVED_AFTER`.
>
> **Audit Note on Baseline B3**: B3 first reported 0 of 5 and 0 of 7 because the evaluation script read the wrong JSON key; corrected after audit.

## 1. Primary Evaluation Table (IoU >= 0.5)

| System / Ranking Rule | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) | Start Errors (Listed) | End Errors (Listed) |
|---|---|---|---|---|---|---|---|
| **(1) S3a only (no judge)** | 5 of 5 | 5 of 5 | 5 of 7 | 3.80s | 8.50s | [3.8, 6.3, 18.6, 0.0, 2.3, 13.7, 2.5] | [0.0, 13.7, 0.0, 0.3, 8.5, 9.54, 12.76] |
| **(2) S3a order + judge gate (ADOPTED)** | 3 of 5 | 5 of 5 | 5 of 7 | 3.80s | 8.50s | [3.8, 6.3, 18.6, 0.0, 2.3, 13.7, 2.5] | [0.0, 13.7, 0.0, 0.3, 8.5, 9.54, 12.76] |
| **(3) Judge-ranked (rejected rule)** | 2 of 5 | 5 of 5 | 5 of 7 | 5.05s | 0.15s | [3.8, 6.3, 18.6, 0.0, 12.4, 2.5] | [0.0, 13.7, 0.0, 0.3, 0.0, 12.76] |
| **B1 Last 45s** | 5 of 5 | 5 of 5 | 5 of 7 | 7.50s | 5.45s | [15.15, 11.68, 7.5, 2.24, 0.96] | [5.45, 16.38, 0.0, 9.54, 0.0] |
| **B2 Random (20 seeds)** | mean 2.2 of 5 [range 1, 4] | mean 3.45 of 5 [range 2, 5] | mean 3.85 of 7 [range 3, 6] | n/a | n/a | n/a | n/a |
| **B3 Tier-0 Top 3** | 3 of 5 | 5 of 5 | 5 of 7 | 7.70s | 6.10s | [4.8, 25.3, 9.2, 15.5, 5.2, 7.7, 2.4] | [13.0, 0.0, 6.1, 7.5, 17.5, 3.0, 2.46] |

## 2. Secondary Evaluation Table (IoU >= 0.3)

| System / Ranking Rule | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) | Start Errors (Listed) | End Errors (Listed) |
|---|---|---|---|---|---|---|---|
| **(1) S3a only (no judge)** | 5 of 5 | 5 of 5 | 7 of 7 | 5.05s | 9.02s | [3.8, 17.8, 6.3, 18.6, 8.2, 0.0, 2.3, 0.0, 13.7, 2.5] | [0.0, 6.8, 13.7, 0.0, 22.0, 0.3, 8.5, 38.3, 9.54, 12.76] |
| **(2) S3a order + judge gate (ADOPTED)** | 5 of 5 | 5 of 5 | 7 of 7 | 5.05s | 9.02s | [17.8, 3.8, 6.3, 8.2, 18.6, 0.0, 0.0, 2.3, 13.7, 2.5] | [6.8, 0.0, 13.7, 22.0, 0.0, 0.3, 38.3, 8.5, 9.54, 12.76] |
| **(3) Judge-ranked (rejected rule)** | 5 of 5 | 5 of 5 | 6 of 7 | 6.30s | 6.80s | [17.8, 3.8, 8.2, 6.3, 18.6, 0.0, 0.0, 12.4, 2.5] | [6.8, 0.0, 22.0, 13.7, 0.0, 0.3, 38.3, 0.0, 12.76] |
| **B1 Last 45s** | 5 of 5 | 5 of 5 | 5 of 7 | 7.50s | 5.45s | [15.15, 11.68, 7.5, 2.24, 0.96] | [5.45, 16.38, 0.0, 9.54, 0.0] |
| **B2 Random (20 seeds)** | mean 2.85 of 5 [range 1, 5] | mean 4.15 of 5 [range 3, 5] | mean 5.05 of 7 [range 3, 7] | n/a | n/a | n/a | n/a |
| **B3 Tier-0 Top 3** | 4 of 5 | 5 of 5 | 5 of 7 | 9.20s | 8.76s | [23.2, 4.8, 25.3, 9.2, 15.0, 15.5, 5.2, 7.7, 2.3, 2.4, 17.0] | [18.2, 13.0, 0.0, 6.1, 13.7, 7.5, 17.5, 3.0, 18.0, 2.46, 8.76] |

## 3. Match Details (Adopted Rule: S3a order + judge gate, IoU >= 0.5)

| Episode | Rank | Matched Gold ID | Candidate Window | Gold Window | IoU | Start Error | End Error |
|---|---|---|---|---|---|---|---|
| episode-003 | 2 | E03_REF002 | [81.8s - 140.3s] | [85.6s - 140.3s] | 0.935 | 3.80s | 0.00s |
| episode-013 | 1 | E13_REF001 | [68.5s - 88.8s] | [62.2s - 102.5s] | 0.504 | 6.30s | 13.70s |
| episode-013 | 3 | E13_REF001 | [80.8s - 102.5s] | [62.2s - 102.5s] | 0.538 | 18.60s | 0.00s |
| episode-023 | 1 | E23_REF001 | [79.0s - 131.2s] | [79.0s - 131.5s] | 0.994 | 0.00s | 0.30s |
| episode-033 | 2 | E33_REF002 | [124.5s - 156.0s] | [126.8s - 164.5s] | 0.73 | 2.30s | 8.50s |
| episode-033 | 3 | E33_REF002 | [140.5s - 174.0s] | [126.8s - 164.5s] | 0.508 | 13.70s | 9.54s |
| episode-043 | 1 | E43_REF001 | [66.5s - 102.2s] | [69.0s - 115.0s] | 0.685 | 2.50s | 12.76s |

- **Median Start Error**: 3.80s
- **Median End Error**: 8.50s
