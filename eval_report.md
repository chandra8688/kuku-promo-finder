# Gold-Set Evaluation Report

> [!IMPORTANT]
> **Gold Set Provenance**: AI-proposed, AI-reviewed, human-unverified.
> Evaluated across the 5 gold episodes (`3, 13, 23, 33, 43`), total 7 gold windows.
>
> **Audit Note on Baseline B3**: B3 first reported 0 of 5 and 0 of 7 because the evaluation script read the wrong JSON key; corrected after audit.

## 1. Primary Evaluation Table (IoU >= 0.5)

| System | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) |
|---|---|---|---|---|---|
| **Pipeline Final Ranking** | 3 of 5 | 5 of 5 | 5 of 7 | 3.80s | 8.50s |
| **Pipeline S3a Order** | 5 of 5 | 5 of 5 | 5 of 7 | 3.80s | 8.50s |
| **B1 Last 45s** | 5 of 5 | 5 of 5 | 5 of 7 | 7.50s | 5.45s |
| **B2 Random (20 seeds)** | 2.2 of 5 (range [1, 4]) | 3.45 of 5 (range [2, 5]) | 3.85 of 7 (range [3, 6]) | n/a | n/a |
| **B3 Tier-0 Top 3** | 3 of 5 | 5 of 5 | 5 of 7 | 7.70s | 6.10s |

## 2. Secondary Evaluation Table (IoU >= 0.3)

| System | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) |
|---|---|---|---|---|---|
| **Pipeline Final Ranking** | 5 of 5 | 5 of 5 | 7 of 7 | 5.05s | 9.02s |
| **Pipeline S3a Order** | 5 of 5 | 5 of 5 | 7 of 7 | 5.05s | 9.02s |
| **B1 Last 45s** | 5 of 5 | 5 of 5 | 5 of 7 | 7.50s | 5.45s |
| **B2 Random (20 seeds)** | 2.85 of 5 (range [1, 5]) | 4.15 of 5 (range [3, 5]) | 5.05 of 7 (range [3, 7]) | n/a | n/a |
| **B3 Tier-0 Top 3** | 4 of 5 | 5 of 5 | 5 of 7 | 9.20s | 8.76s |

## 3. Match Details & Boundary Errors (Pipeline Final Ranking, IoU >= 0.5)

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

## 4. Pipeline Windows With No Gold Match (Per Episode)

| Episode | Pipeline Evaluated Windows | Matched Windows | Unmatched Windows |
|---|---|---|---|
| episode-003 | 2 | 1 | 1 |
| episode-013 | 3 | 2 | 1 |
| episode-023 | 2 | 1 | 1 |
| episode-033 | 3 | 2 | 1 |
| episode-043 | 2 | 1 | 1 |
