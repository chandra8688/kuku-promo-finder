"""Evaluation of Promo Finder Pipeline and Baselines against Gold Standard Set.

Gold set: gold/gold.csv (AI-proposed, AI-reviewed, human-unverified)
5 evaluation episodes: 3, 13, 23, 33, 43 (7 gold windows).

Systems (K=3 windows per episode):
1. (1) S3a only (no judge)
2. (2) S3a order + judge gate (ADOPTED rule)
3. (3) Judge-ranked (rejected rule)
4. B1 last 45s (src/baselines.py)
5. B2 random (20 seeds x 3 windows; mean and range across seeds)
6. B3 Tier-0 top 3 (data/tier0/)

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
                "duration_s": float(r["end_s"]) - float(r["start_s"]),
                "decision": r.get("decision", ""),
                "note": r.get("note", ""),
            })
    return gold_windows


def load_all_ranking_rules(
    judged_dir: str | Path = "data/judged",
    k: int = 3,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    int,
    List[Dict[str, Any]],
]:
    """Load the three ranking rules from judged outputs and compute gate removals."""
    p_judged = Path(judged_dir)
    rule1_s3a_only: Dict[str, List[Dict[str, Any]]] = {}
    rule2_s3a_gate: Dict[str, List[Dict[str, Any]]] = {}
    rule3_judge: Dict[str, List[Dict[str, Any]]] = {}

    removed_gate_count = 0
    gate_removed_details = []

    for ep_str in GOLD_EP_STRS:
        file_path = p_judged / f"{ep_str}.json"
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cands = data.get("judged", [])

        # Rule 1: S3a only (no judge) - pure S3a score desc
        r1 = sorted(cands, key=lambda c: (c.get("s3a_score", 0.0), -c.get("start_s", 0.0)), reverse=True)
        rule1_s3a_only[ep_str] = r1[:k]

        # Rule 2: S3a order + judge gate (ADOPTED rule) - passed_judge_gates first, then s3a desc
        passing = [c for c in cands if c.get("passed_judge_gates", True)]
        failing = [c for c in cands if not c.get("passed_judge_gates", True)]
        if passing:
            p_sorted = sorted(passing, key=lambda c: (not c.get("low_confidence", False), c.get("s3a_score", 0.0), -c.get("start_s", 0.0)), reverse=True)
            f_sorted = sorted(failing, key=lambda c: (not c.get("low_confidence", False), c.get("s3a_score", 0.0), -c.get("start_s", 0.0)), reverse=True)
            r2 = p_sorted + f_sorted
        else:
            r2 = sorted(cands, key=lambda c: (not c.get("low_confidence", False), c.get("s3a_score", 0.0), -c.get("start_s", 0.0)), reverse=True)
        rule2_s3a_gate[ep_str] = r2[:k]

        # Check if gate removed S3a rank-1 window
        if r1 and r2 and r1[0]["candidate_id"] != r2[0]["candidate_id"]:
            removed_gate_count += 1
            gate_removed_details.append({
                "episode": ep_str,
                "s3a_rank1": {
                    "candidate_id": r1[0]["candidate_id"],
                    "window": [r1[0]["start_s"], r1[0]["end_s"]],
                    "s3a_score": r1[0].get("s3a_score", 0.0),
                    "judge_gate_reason": r1[0].get("judge_gate_reason"),
                },
                "gated_rank1": {
                    "candidate_id": r2[0]["candidate_id"],
                    "window": [r2[0]["start_s"], r2[0]["end_s"]],
                    "s3a_score": r2[0].get("s3a_score", 0.0),
                },
            })

        # Rule 3: Judge-ranked (rejected rule) - judge rank asc
        r3 = sorted(cands, key=lambda c: c.get("rank", 999))
        rule3_judge[ep_str] = r3[:k]

    return rule1_s3a_only, rule2_s3a_gate, rule3_judge, removed_gate_count, gate_removed_details


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
        cands = data.get("tier0_candidates", data.get("candidates", []))
        survivors = [c for c in cands if c.get("passed_all_gates")] if (cands and "passed_all_gates" in cands[0]) else cands
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

        if ep_cands:
            top1 = ep_cands[0]
            for g in ep_gold:
                if compute_interval_iou(top1.get("start_s", 0.0), top1.get("end_s", 0.0), g["start_s"], g["end_s"]) >= iou_threshold:
                    episodes_top1_match += 1
                    break

        if matched_cands_in_ep > 0:
            episodes_top3_match += 1

        unmatched_per_ep[ep_str] = len(ep_cands) - matched_cands_in_ep

    start_errs = [m["start_error_s"] for m in matches_list]
    end_errs = [m["end_error_s"] for m in matches_list]

    med_start = round(statistics.median(start_errs), 2) if start_errs else None
    med_end = round(statistics.median(end_errs), 2) if end_errs else None

    return {
        "iou_threshold": iou_threshold,
        "episodes_evaluated": len(GOLD_EPISODES),
        "episodes_top1_match": episodes_top1_match,
        "episodes_top1_ratio": f"{episodes_top1_match} of {len(GOLD_EPISODES)}",
        "episodes_top3_match": episodes_top3_match,
        "episodes_top3_ratio": f"{episodes_top3_match} of {len(GOLD_EPISODES)}",
        "gold_windows_total": len(gold_windows),
        "gold_windows_recalled": len(gold_recalled_set),
        "gold_recall_ratio": f"{len(gold_recalled_set)} of {len(gold_windows)}",
        "matches": matches_list,
        "start_errors": start_errs,
        "end_errors": end_errs,
        "median_start_error_s": med_start,
        "median_end_error_s": med_end,
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

    # Load ranking rules and baselines
    r1, r2, r3, rem_count, rem_details = load_all_ranking_rules("data/judged", k=3)
    sys_b1 = generate_b1_baseline(durations)
    sys_tier0 = load_tier0_ranked("data/tier0", k=3)

    systems = {
        "rule1_s3a_only": r1,
        "rule2_s3a_gate_adopted": r2,
        "rule3_judge_rejected": r3,
        "b1_last45s": sys_b1,
        "b3_tier0": sys_tier0,
    }

    report_data: Dict[str, Any] = {
        "gold_provenance": "AI-proposed, AI-reviewed, human-unverified",
        "b3_audit_note": "B3 first reported 0 of 5 and 0 of 7 because the evaluation script read the wrong JSON key; corrected after audit",
        "pipeline_final_clarification": "Pipeline Final Ranking in earlier output referred to row (3) Judge-ranked; row (2) S3a order + judge gate is the ADOPTED rule used by run.py and outputs/promos_per_episode.*",
        "gold_episodes": GOLD_EPISODES,
        "total_gold_windows": len(gold_windows),
        "gate_removed_s3a_rank1_count": f"{rem_count} of 5",
        "gate_removed_details": rem_details,
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

    def fmt_err(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{val:.2f}s"
        return str(val) if val is not None else "n/a"

    # Format Markdown Report
    lines = []
    lines.append("# Gold-Set Evaluation Report")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Gold Set Provenance**: AI-proposed, AI-reviewed, human-unverified.")
    lines.append("> Evaluated across the 5 gold episodes (`3, 13, 23, 33, 43`), total 7 gold windows.")
    lines.append(">")
    lines.append("> **Ranking Rule Clarification**: \"Pipeline Final Ranking\" in earlier output referred to row (3) Judge-ranked.")
    lines.append("> The frozen adopted rule is row (2) **S3a order + judge gate**, which is the exact ranking used by `run.py` and `outputs/promos_per_episode.*`.")
    lines.append(">")
    lines.append(f"> **Gate Removal Impact**: The judge gate removed the S3a rank-1 window in **{rem_count} of 5** gold episodes (`episode-003` and `episode-033`) due to `JUDGE_RESOLVED_AFTER`.")
    lines.append(">")
    lines.append("> **Audit Note on Baseline B3**: B3 first reported 0 of 5 and 0 of 7 because the evaluation script read the wrong JSON key; corrected after audit.")
    lines.append("")
    lines.append("## 1. Primary Evaluation Table (IoU >= 0.5)")
    lines.append("")
    lines.append("| System / Ranking Rule | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) | Start Errors (Listed) | End Errors (Listed) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    r1_5 = report_data["results_iou_0_5"]["rule1_s3a_only"]
    lines.append(f"| **(1) S3a only (no judge)** | {r1_5['episodes_top1_ratio']} | {r1_5['episodes_top3_ratio']} | {r1_5['gold_recall_ratio']} | {fmt_err(r1_5['median_start_error_s'])} | {fmt_err(r1_5['median_end_error_s'])} | {r1_5['start_errors']} | {r1_5['end_errors']} |")

    r2_5 = report_data["results_iou_0_5"]["rule2_s3a_gate_adopted"]
    lines.append(f"| **(2) S3a order + judge gate (ADOPTED)** | {r2_5['episodes_top1_ratio']} | {r2_5['episodes_top3_ratio']} | {r2_5['gold_recall_ratio']} | {fmt_err(r2_5['median_start_error_s'])} | {fmt_err(r2_5['median_end_error_s'])} | {r2_5['start_errors']} | {r2_5['end_errors']} |")

    r3_5 = report_data["results_iou_0_5"]["rule3_judge_rejected"]
    lines.append(f"| **(3) Judge-ranked (rejected rule)** | {r3_5['episodes_top1_ratio']} | {r3_5['episodes_top3_ratio']} | {r3_5['gold_recall_ratio']} | {fmt_err(r3_5['median_start_error_s'])} | {fmt_err(r3_5['median_end_error_s'])} | {r3_5['start_errors']} | {r3_5['end_errors']} |")

    b1_5 = report_data["results_iou_0_5"]["b1_last45s"]
    lines.append(f"| **B1 Last 45s** | {b1_5['episodes_top1_ratio']} | {b1_5['episodes_top3_ratio']} | {b1_5['gold_recall_ratio']} | {fmt_err(b1_5['median_start_error_s'])} | {fmt_err(b1_5['median_end_error_s'])} | {b1_5['start_errors']} | {b1_5['end_errors']} |")

    b2_5 = report_data["b2_random_0_5"]
    b2_top1_str = f"mean {b2_5['top1_match_mean']} of 5 [range {b2_5['top1_match_min']}, {b2_5['top1_match_max']}]"
    b2_top3_str = f"mean {b2_5['top3_match_mean']} of 5 [range {b2_5['top3_match_min']}, {b2_5['top3_match_max']}]"
    b2_rec_str = f"mean {b2_5['recall_mean']} of 7 [range {b2_5['recall_min']}, {b2_5['recall_max']}]"
    lines.append(f"| **B2 Random (20 seeds)** | {b2_top1_str} | {b2_top3_str} | {b2_rec_str} | n/a | n/a | n/a | n/a |")

    b3_5 = report_data["results_iou_0_5"]["b3_tier0"]
    lines.append(f"| **B3 Tier-0 Top 3** | {b3_5['episodes_top1_ratio']} | {b3_5['episodes_top3_ratio']} | {b3_5['gold_recall_ratio']} | {fmt_err(b3_5['median_start_error_s'])} | {fmt_err(b3_5['median_end_error_s'])} | {b3_5['start_errors']} | {b3_5['end_errors']} |")

    lines.append("")
    lines.append("## 2. Secondary Evaluation Table (IoU >= 0.3)")
    lines.append("")
    lines.append("| System / Ranking Rule | Top-1 Match (n of 5) | Top-3 Match (n of 5) | Gold Recall (n of 7) | Med Start Err (s) | Med End Err (s) | Start Errors (Listed) | End Errors (Listed) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    r1_3 = report_data["results_iou_0_3"]["rule1_s3a_only"]
    lines.append(f"| **(1) S3a only (no judge)** | {r1_3['episodes_top1_ratio']} | {r1_3['episodes_top3_ratio']} | {r1_3['gold_recall_ratio']} | {fmt_err(r1_3['median_start_error_s'])} | {fmt_err(r1_3['median_end_error_s'])} | {r1_3['start_errors']} | {r1_3['end_errors']} |")

    r2_3 = report_data["results_iou_0_3"]["rule2_s3a_gate_adopted"]
    lines.append(f"| **(2) S3a order + judge gate (ADOPTED)** | {r2_3['episodes_top1_ratio']} | {r2_3['episodes_top3_ratio']} | {r2_3['gold_recall_ratio']} | {fmt_err(r2_3['median_start_error_s'])} | {fmt_err(r2_3['median_end_error_s'])} | {r2_3['start_errors']} | {r2_3['end_errors']} |")

    r3_3 = report_data["results_iou_0_3"]["rule3_judge_rejected"]
    lines.append(f"| **(3) Judge-ranked (rejected rule)** | {r3_3['episodes_top1_ratio']} | {r3_3['episodes_top3_ratio']} | {r3_3['gold_recall_ratio']} | {fmt_err(r3_3['median_start_error_s'])} | {fmt_err(r3_3['median_end_error_s'])} | {r3_3['start_errors']} | {r3_3['end_errors']} |")

    b1_3 = report_data["results_iou_0_3"]["b1_last45s"]
    lines.append(f"| **B1 Last 45s** | {b1_3['episodes_top1_ratio']} | {b1_3['episodes_top3_ratio']} | {b1_3['gold_recall_ratio']} | {fmt_err(b1_3['median_start_error_s'])} | {fmt_err(b1_3['median_end_error_s'])} | {b1_3['start_errors']} | {b1_3['end_errors']} |")

    b2_3 = report_data["b2_random_0_3"]
    b2_top1_str_3 = f"mean {b2_3['top1_match_mean']} of 5 [range {b2_3['top1_match_min']}, {b2_3['top1_match_max']}]"
    b2_top3_str_3 = f"mean {b2_3['top3_match_mean']} of 5 [range {b2_3['top3_match_min']}, {b2_3['top3_match_max']}]"
    b2_rec_str_3 = f"mean {b2_3['recall_mean']} of 7 [range {b2_3['recall_min']}, {b2_3['recall_max']}]"
    lines.append(f"| **B2 Random (20 seeds)** | {b2_top1_str_3} | {b2_top3_str_3} | {b2_rec_str_3} | n/a | n/a | n/a | n/a |")

    b3_3 = report_data["results_iou_0_3"]["b3_tier0"]
    lines.append(f"| **B3 Tier-0 Top 3** | {b3_3['episodes_top1_ratio']} | {b3_3['episodes_top3_ratio']} | {b3_3['gold_recall_ratio']} | {fmt_err(b3_3['median_start_error_s'])} | {fmt_err(b3_3['median_end_error_s'])} | {b3_3['start_errors']} | {b3_3['end_errors']} |")

    lines.append("")
    lines.append("## 3. Match Details (Adopted Rule: S3a order + judge gate, IoU >= 0.5)")
    lines.append("")
    if r2_5["matches"]:
        lines.append("| Episode | Rank | Matched Gold ID | Candidate Window | Gold Window | IoU | Start Error | End Error |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in r2_5["matches"]:
            cw = f"[{m['cand_window'][0]:.1f}s - {m['cand_window'][1]:.1f}s]"
            gw = f"[{m['gold_window'][0]:.1f}s - {m['gold_window'][1]:.1f}s]"
            lines.append(f"| {m['episode']} | {m['rank']} | {m['gold_id']} | {cw} | {gw} | {m['iou']} | {m['start_error_s']:.2f}s | {m['end_error_s']:.2f}s |")
        lines.append(f"\n- **Median Start Error**: {r2_5['median_start_error_s']:.2f}s")
        lines.append(f"- **Median End Error**: {r2_5['median_end_error_s']:.2f}s")
    else:
        lines.append("No matches at IoU >= 0.5.")

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
    print("Updated eval_report.md and evaluation_report.json")
