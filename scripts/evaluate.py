"""Evaluation of Promo Finder Pipeline and Baselines against Gold Standard Set.

Gold set: gold/gold.csv (AI-proposed, AI-reviewed, human-unverified)
5 evaluation episodes: 3, 13, 23, 33, 43 (7 gold windows).

Systems (K=3 windows per episode):
1. Pipeline final ranking (data/judged/)
2. Pipeline S3a order before judge (data/candidates/)
3. B1 last 45s (src/baselines.py)
4. B2 random (20 seeds x 3 windows; mean and range across seeds)
5. B3 Tier-0 top 3 (data/tier0/)

Matches evaluated at IoU >= 0.5 (primary) and IoU >= 0.3 (secondary).
Reports raw counts (n of N), never percentages.
Never prints screenplay text or character names.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.baselines import generate_b1_baseline, generate_b2_baseline, load_episode_durations
from src.windows import compute_interval_iou

GOLD_EPISODES = [3, 13, 23, 33, 43]
GOLD_EP_STRS = [f"episode-{ep:03d}" for ep in GOLD_EPISODES]


def load_gold_windows(gold_csv_path: str | Path = "gold/gold.csv") -> List[Dict[str, Any]]:
    """Load gold windows from gold/gold.csv."""
    path = Path(gold_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {gold_csv_path}")

    gold_windows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gold_windows.append({
                "gold_id": r["gold_id"],
                "episode": int(r["episode"]),
                "episode_str": f"episode-{int(r['episode']):03d}",
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "duration_s": round(float(r["end_s"]) - float(r["start_s"]), 2),
                "status": r.get("status", ""),
            })
    return gold_windows


def load_pipeline_final_ranked(judged_dir: str | Path = "data/judged", k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Load top K windows per episode from judged outputs."""
    p = Path(judged_dir)
    res: Dict[str, List[Dict[str, Any]]] = {}
    for ep_str in GOLD_EP_STRS:
        file_path = p / f"{ep_str}.json"
        if not file_path.exists():
            res[ep_str] = []
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        judged = data.get("judged", [])
        # Already ranked by frozen ranking rule
        res[ep_str] = judged[:k]
    return res


def load_pipeline_s3a_ranked(candidates_dir: str | Path = "data/candidates", k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Load top K windows per episode by S3a score (before judge) from candidates outputs."""
    p = Path(candidates_dir)
    res: Dict[str, List[Dict[str, Any]]] = {}
    for ep_str in GOLD_EP_STRS:
        file_path = p / f"{ep_str}.json"
        if not file_path.exists():
            res[ep_str] = []
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cands = [c for c in data.get("candidates", []) if c.get("passed_all_gates")]
        # Sort by S3a score desc
        cands.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        res[ep_str] = cands[:k]
    return res


def load_tier0_ranked(tier0_dir: str | Path = "data/tier0", k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Load top K windows per episode from Tier-0 baseline outputs."""
    p = Path(tier0_dir)
    res: Dict[str, List[Dict[str, Any]]] = {}
    for ep_str in GOLD_EP_STRS:
        file_path = p / f"{ep_str}.json"
        if not file_path.exists():
            res[ep_str] = []
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        survivors = data.get("survivors", [])
        survivors.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        res[ep_str] = survivors[:k]
    return res


def evaluate_system_against_gold(
    system_windows_by_ep: Dict[str, List[Dict[str, Any]]],
    gold_windows: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate candidate windows against gold windows for the 5 episodes."""
    gold_by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for g in gold_windows:
        gold_by_ep.setdefault(g["episode_str"], []).append(g)

    episodes_top1_match = 0
    episodes_top3_match = 0
    gold_recalled_set = set()

    matches_list = []
    unmatched_per_ep = {}

    for ep_str in GOLD_EP_STRS:
        ep_gold = gold_by_ep.get(ep_str, [])
        ep_cands = system_windows_by_ep.get(ep_str, [])[:top_k]

        matched_cands_in_ep = 0

        for rank_idx, cand in enumerate(ep_cands, start=1):
            c_st = cand.get("start_s", 0.0)
            c_en = cand.get("end_s", 0.0)

            cand_matched = False
            best_iou = 0.0
            best_gold = None

            for g in ep_gold:
                g_st = g["start_s"]
                g_en = g["end_s"]
                iou = compute_interval_iou(c_st, c_en, g_st, g_en)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gold = g

            if best_gold is not None:
                cand_matched = True
                matched_cands_in_ep += 1
                gold_recalled_set.add(best_gold["gold_id"])

                st_err = round(abs(c_st - best_gold["start_s"]), 2)
                en_err = round(abs(c_en - best_gold["end_s"]), 2)

                matches_list.append({
                    "episode": ep_str,
                    "rank": rank_idx,
                    "gold_id": best_gold["gold_id"],
                    "cand_window": [c_st, c_en],
                    "gold_window": [best_gold["start_s"], best_gold["end_s"]],
                    "iou": round(best_iou, 3),
                    "start_error_s": st_err,
                    "end_error_s": en_err,
                })

            if rank_idx == 1 and cand_matched:
                episodes_top1_match += 1

        if matched_cands_in_ep > 0:
            episodes_top3_match += 1

        unmatched_per_ep[ep_str] = len(ep_cands) - matched_cands_in_ep

    start_errors = [m["start_error_s"] for m in matches_list]
    end_errors = [m["end_error_s"] for m in matches_list]
    med_st_err = statistics.median(start_errors) if start_errors else 0.0
    med_en_err = statistics.median(end_errors) if end_errors else 0.0

    return {
        "iou_threshold": iou_threshold,
        "episodes_evaluated": len(GOLD_EP_STRS),
        "episodes_top1_match": episodes_top1_match,
        "episodes_top1_ratio": f"{episodes_top1_match} of {len(GOLD_EP_STRS)}",
        "episodes_top3_match": episodes_top3_match,
        "episodes_top3_ratio": f"{episodes_top3_match} of {len(GOLD_EP_STRS)}",
        "gold_windows_total": len(gold_windows),
        "gold_windows_recalled": len(gold_recalled_set),
        "gold_recall_ratio": f"{len(gold_recalled_set)} of {len(gold_windows)}",
        "matches": matches_list,
        "start_errors": start_errors,
        "end_errors": end_errors,
        "median_start_error_s": round(med_st_err, 2),
        "median_end_error_s": round(med_en_err, 2),
        "unmatched_windows_per_episode": unmatched_per_ep,
    }


def evaluate_b2_random_baseline(
    gold_windows: List[Dict[str, Any]],
    durations: Dict[str, float],
    iou_threshold: float = 0.5,
    num_seeds: int = 20,
) -> Dict[str, Any]:
    """Evaluate B2 random baseline across 20 seeds."""
    b2_dict = generate_b2_baseline(durations, seeds=range(num_seeds), windows_per_episode=3)

    seed_top1 = []
    seed_top3 = []
    seed_recall = []

    for s in range(num_seeds):
        windows_s = b2_dict[s]
        eval_res = evaluate_system_against_gold(windows_s, gold_windows, iou_threshold=iou_threshold)
        seed_top1.append(eval_res["episodes_top1_match"])
        seed_top3.append(eval_res["episodes_top3_match"])
        seed_recall.append(eval_res["gold_windows_recalled"])

    return {
        "num_seeds": num_seeds,
        "iou_threshold": iou_threshold,
        "top1_match_mean": round(statistics.mean(seed_top1), 2),
        "top1_match_min": min(seed_top1),
        "top1_match_max": max(seed_top1),
        "top3_match_mean": round(statistics.mean(seed_top3), 2),
        "top3_match_min": min(seed_top3),
        "top3_match_max": max(seed_top3),
        "recall_mean": round(statistics.mean(seed_recall), 2),
        "recall_min": min(seed_recall),
        "recall_max": max(seed_recall),
    }


def run_full_evaluation() -> Tuple[Dict[str, Any], str]:
    """Run evaluation for all systems across IoU 0.5 and IoU 0.3."""
    gold_windows = load_gold_windows("gold/gold.csv")
    durations = load_episode_durations("data/parsed")

    # Load systems
    sys_final = load_pipeline_final_ranked("data/judged", k=3)
    sys_s3a = load_pipeline_s3a_ranked("data/candidates", k=3)
    sys_b1 = generate_b1_baseline(durations)
    sys_tier0 = load_tier0_ranked("data/tier0", k=3)

    systems = {
        "pipeline_final": sys_final,
        "pipeline_s3a": sys_s3a,
        "b1_last45s": sys_b1,
        "b3_tier0": sys_tier0,
    }

    report_data: Dict[str, Any] = {
        "gold_provenance": "AI-proposed, AI-reviewed, human-unverified",
        "gold_episodes": GOLD_EPISODES,
        "total_gold_windows": len(gold_windows),
        "results_iou_0_5": {},
        "results_iou_0_3": {},
        "b2_random_0_5": {},
        "b2_random_0_3": {},
    }

    # Evaluate at IoU >= 0.5
    for name, sys_wins in systems.items():
        report_data["results_iou_0_5"][name] = evaluate_system_against_gold(
            sys_wins, gold_windows, iou_threshold=0.5
        )

    # Evaluate at IoU >= 0.3
    for name, sys_wins in systems.items():
        report_data["results_iou_0_3"][name] = evaluate_system_against_gold(
            sys_wins, gold_windows, iou_threshold=0.3
        )

    # B2 random baseline
    report_data["b2_random_0_5"] = evaluate_b2_random_baseline(gold_windows, durations, iou_threshold=0.5)
    report_data["b2_random_0_3"] = evaluate_b2_random_baseline(gold_windows, durations, iou_threshold=0.3)

    # Format Markdown Report
    lines = []
    lines.append("# Gold-Set Evaluation Report")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Gold Set Provenance**: AI-proposed, AI-reviewed, human-unverified.")
    lines.append("> Evaluated across the 5 gold episodes (`3, 13, 23, 33, 43`), total 7 gold windows.")
    lines.append("")
    lines.append("## 1. Primary Evaluation Table (IoU >= 0.5)")
    lines.append("")
    lines.append("| System | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) |")
    lines.append("|---|---|---|---|---|---|")

    pf_5 = report_data["results_iou_0_5"]["pipeline_final"]
    lines.append(f"| **Pipeline Final Ranking** | {pf_5['episodes_top1_ratio']} | {pf_5['episodes_top3_ratio']} | {pf_5['gold_recall_ratio']} | {pf_5['median_start_error_s']:.2f}s | {pf_5['median_end_error_s']:.2f}s |")

    ps_5 = report_data["results_iou_0_5"]["pipeline_s3a"]
    lines.append(f"| **Pipeline S3a Order** | {ps_5['episodes_top1_ratio']} | {ps_5['episodes_top3_ratio']} | {ps_5['gold_recall_ratio']} | {ps_5['median_start_error_s']:.2f}s | {ps_5['median_end_error_s']:.2f}s |")

    b1_5 = report_data["results_iou_0_5"]["b1_last45s"]
    lines.append(f"| **B1 Last 45s** | {b1_5['episodes_top1_ratio']} | {b1_5['episodes_top3_ratio']} | {b1_5['gold_recall_ratio']} | {b1_5['median_start_error_s']:.2f}s | {b1_5['median_end_error_s']:.2f}s |")

    b2_5 = report_data["b2_random_0_5"]
    b2_top1_str = f"{b2_5['top1_match_mean']} of 5 (range [{b2_5['top1_match_min']}, {b2_5['top1_match_max']}])"
    b2_top3_str = f"{b2_5['top3_match_mean']} of 5 (range [{b2_5['top3_match_min']}, {b2_5['top3_match_max']}])"
    b2_rec_str = f"{b2_5['recall_mean']} of 7 (range [{b2_5['recall_min']}, {b2_5['recall_max']}])"
    lines.append(f"| **B2 Random (20 seeds)** | {b2_top1_str} | {b2_top3_str} | {b2_rec_str} | - | - |")

    b3_5 = report_data["results_iou_0_5"]["b3_tier0"]
    lines.append(f"| **B3 Tier-0 Top 3** | {b3_5['episodes_top1_ratio']} | {b3_5['episodes_top3_ratio']} | {b3_5['gold_recall_ratio']} | {b3_5['median_start_error_s']:.2f}s | {b3_5['median_end_error_s']:.2f}s |")

    lines.append("")
    lines.append("## 2. Secondary Evaluation Table (IoU >= 0.3)")
    lines.append("")
    lines.append("| System | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) |")
    lines.append("|---|---|---|---|---|---|")

    pf_3 = report_data["results_iou_0_3"]["pipeline_final"]
    lines.append(f"| **Pipeline Final Ranking** | {pf_3['episodes_top1_ratio']} | {pf_3['episodes_top3_ratio']} | {pf_3['gold_recall_ratio']} | {pf_3['median_start_error_s']:.2f}s | {pf_3['median_end_error_s']:.2f}s |")

    ps_3 = report_data["results_iou_0_3"]["pipeline_s3a"]
    lines.append(f"| **Pipeline S3a Order** | {ps_3['episodes_top1_ratio']} | {ps_3['episodes_top3_ratio']} | {ps_3['gold_recall_ratio']} | {ps_3['median_start_error_s']:.2f}s | {ps_3['median_end_error_s']:.2f}s |")

    b1_3 = report_data["results_iou_0_3"]["b1_last45s"]
    lines.append(f"| **B1 Last 45s** | {b1_3['episodes_top1_ratio']} | {b1_3['episodes_top3_ratio']} | {b1_3['gold_recall_ratio']} | {b1_3['median_start_error_s']:.2f}s | {b1_3['median_end_error_s']:.2f}s |")

    b2_3 = report_data["b2_random_0_3"]
    b2_top1_str_3 = f"{b2_3['top1_match_mean']} of 5 (range [{b2_3['top1_match_min']}, {b2_3['top1_match_max']}])"
    b2_top3_str_3 = f"{b2_3['top3_match_mean']} of 5 (range [{b2_3['top3_match_min']}, {b2_3['top3_match_max']}])"
    b2_rec_str_3 = f"{b2_3['recall_mean']} of 7 (range [{b2_3['recall_min']}, {b2_3['recall_max']}])"
    lines.append(f"| **B2 Random (20 seeds)** | {b2_top1_str_3} | {b2_top3_str_3} | {b2_rec_str_3} | - | - |")

    b3_3 = report_data["results_iou_0_3"]["b3_tier0"]
    lines.append(f"| **B3 Tier-0 Top 3** | {b3_3['episodes_top1_ratio']} | {b3_3['episodes_top3_ratio']} | {b3_3['gold_recall_ratio']} | {b3_3['median_start_error_s']:.2f}s | {b3_3['median_end_error_s']:.2f}s |")

    lines.append("")
    lines.append("## 3. Match Details & Boundary Errors (Pipeline Final Ranking, IoU >= 0.5)")
    lines.append("")
    if pf_5["matches"]:
        lines.append("| Episode | Rank | Matched Gold ID | Candidate Window | Gold Window | IoU | Start Error | End Error |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in pf_5["matches"]:
            cw = f"[{m['cand_window'][0]:.1f}s - {m['cand_window'][1]:.1f}s]"
            gw = f"[{m['gold_window'][0]:.1f}s - {m['gold_window'][1]:.1f}s]"
            lines.append(f"| {m['episode']} | {m['rank']} | {m['gold_id']} | {cw} | {gw} | {m['iou']} | {m['start_error_s']:.2f}s | {m['end_error_s']:.2f}s |")
        lines.append(f"\n- **Median Start Error**: {pf_5['median_start_error_s']:.2f}s")
        lines.append(f"- **Median End Error**: {pf_5['median_end_error_s']:.2f}s")
    else:
        lines.append("No matches at IoU >= 0.5.")

    lines.append("")
    lines.append("## 4. Pipeline Windows With No Gold Match (Per Episode)")
    lines.append("")
    lines.append("| Episode | Pipeline Evaluated Windows | Matched Windows | Unmatched Windows |")
    lines.append("|---|---|---|---|")
    for ep_str in GOLD_EP_STRS:
        total_k = len(sys_final.get(ep_str, []))
        unmatched = pf_5["unmatched_windows_per_episode"].get(ep_str, 0)
        matched = total_k - unmatched
        lines.append(f"| {ep_str} | {total_k} | {matched} | {unmatched} |")

    md_report = "\n".join(lines) + "\n"

    # Write evaluation_report.json and eval_report.md
    Path("data/evaluation_report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    Path("evaluation_report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    Path("eval_report.md").write_text(md_report, encoding="utf-8")

    return report_data, md_report


if __name__ == "__main__":
    rep, md = run_full_evaluation()
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Gold Provenance : {rep['gold_provenance']}")
    print(f"Pipeline Final  : Top-1 = {rep['results_iou_0_5']['pipeline_final']['episodes_top1_ratio']}, Top-3 = {rep['results_iou_0_5']['pipeline_final']['episodes_top3_ratio']}, Recall = {rep['results_iou_0_5']['pipeline_final']['gold_recall_ratio']}")
    print(f"Pipeline S3a    : Top-1 = {rep['results_iou_0_5']['pipeline_s3a']['episodes_top1_ratio']}, Top-3 = {rep['results_iou_0_5']['pipeline_s3a']['episodes_top3_ratio']}, Recall = {rep['results_iou_0_5']['pipeline_s3a']['gold_recall_ratio']}")
    print(f"B1 Last 45s     : Top-1 = {rep['results_iou_0_5']['b1_last45s']['episodes_top1_ratio']}, Top-3 = {rep['results_iou_0_5']['b1_last45s']['episodes_top3_ratio']}, Recall = {rep['results_iou_0_5']['b1_last45s']['gold_recall_ratio']}")
    print(f"B2 Random (mean): Top-1 = {rep['b2_random_0_5']['top1_match_mean']} of 5, Top-3 = {rep['b2_random_0_5']['top3_match_mean']} of 5, Recall = {rep['b2_random_0_5']['recall_mean']} of 7")
    print(f"B3 Tier-0       : Top-1 = {rep['results_iou_0_5']['b3_tier0']['episodes_top1_ratio']}, Top-3 = {rep['results_iou_0_5']['b3_tier0']['episodes_top3_ratio']}, Recall = {rep['results_iou_0_5']['b3_tier0']['gold_recall_ratio']}")
    print("Wrote data/evaluation_report.json and eval_report.md")

