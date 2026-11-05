"""Post-Hoc Diagnostics (scripts/run_posthoc_diagnostics.py).

Implements Section E diagnostics:
E1. Duration distribution (reserved rank-1, gold durations, top-3 within 1.0s of 20.0s)
E2. Extra boundary metrics (|end error| <= 3s, |start error| <= 5s & |end error| <= 5s)
E3. Gate ablation on 5 gold episodes using S3a order
E4. Failure-log inputs (no screenplay text)

Outputs saved to posthoc/diagnostics.json and posthoc/diagnostics.md.
Never prints screenplay text, loglines, or character names.
Reports raw counts (n of N), never percentages.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baselines import generate_b1_baseline, generate_b2_baseline, load_episode_durations
from src.gates import evaluate_window_gates
from src.reserved import RESERVED_EPISODES
from src.windows import compute_interval_iou, enumerate_windows_for_episode, load_scoring_config

RESERVED_LIST = sorted(list(RESERVED_EPISODES))
GOLD_EPISODES = [3, 13, 23, 33, 43]
GOLD_EP_STRS = [f"episode-{ep:03d}" for ep in GOLD_EPISODES]


def load_gold_windows() -> List[Dict[str, Any]]:
    res = []
    with open("gold/gold.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res.append({
                "gold_id": r["gold_id"],
                "episode": int(r["episode"]),
                "episode_str": f"episode-{int(r['episode']):03d}",
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "duration_s": round(float(r["end_s"]) - float(r["start_s"]), 2),
            })
    return res


def run_e1_duration_report(gold_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1. Rank-1 window durations for 10 reserved episodes
    rank1_durations = {}
    top3_within_1s_of_20_count = 0
    total_top3_evaluated = 0

    top3_within_1s_of_20_gold_eps = 0
    total_top3_gold_eps = 0

    for ep in RESERVED_LIST:
        judged_file = Path(f"data/judged/{ep}.json")
        if judged_file.exists():
            with open(judged_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            judged = d.get("judged", [])
            if judged:
                rank1_durations[ep] = judged[0]["duration_s"]
            for c in judged[:3]:
                total_top3_evaluated += 1
                dur = c["duration_s"]
                if 20.0 <= dur <= 21.0:
                    top3_within_1s_of_20_count += 1
                if ep in GOLD_EP_STRS:
                    total_top3_gold_eps += 1
                    if 20.0 <= dur <= 21.0:
                        top3_within_1s_of_20_gold_eps += 1

    # 2. 7 gold durations
    gold_durations = {g["gold_id"]: g["duration_s"] for g in gold_windows}

    return {
        "reserved_rank1_durations_s": rank1_durations,
        "gold_durations_s": gold_durations,
        "top3_within_1s_of_20_reserved": f"{top3_within_1s_of_20_count} of {total_top3_evaluated}",
        "top3_within_1s_of_20_gold_eps": f"{top3_within_1s_of_20_gold_eps} of {total_top3_gold_eps}",
    }


def evaluate_boundary_metrics_for_windows(
    system_windows_by_ep: Dict[str, List[Dict[str, Any]]],
    gold_windows: List[Dict[str, Any]],
    top_k: int = 3,
) -> Tuple[int, int]:
    """Returns (gold_recalled_end3s, gold_recalled_st5s_en5s) as raw counts out of len(gold_windows)."""
    gold_by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for g in gold_windows:
        gold_by_ep.setdefault(g["episode_str"], []).append(g)

    recalled_end3 = set()
    recalled_both5 = set()

    for ep_str in GOLD_EP_STRS:
        ep_gold = gold_by_ep.get(ep_str, [])
        ep_cands = system_windows_by_ep.get(ep_str, [])[:top_k]

        for g in ep_gold:
            g_st = g["start_s"]
            g_en = g["end_s"]
            gid = g["gold_id"]

            for c in ep_cands:
                c_st = c["start_s"]
                c_en = c["end_s"]

                # Overlap check (must have positive intersection)
                inter = max(0.0, min(c_en, g_en) - max(c_st, g_st))
                if inter <= 0.0:
                    continue

                end_err = abs(c_en - g_en)
                start_err = abs(c_st - g_st)

                if end_err <= 3.0:
                    recalled_end3.add(gid)

                if start_err <= 5.0 and end_err <= 5.0:
                    recalled_both5.add(gid)

    return len(recalled_end3), len(recalled_both5)


def run_e2_boundary_metrics(gold_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1. Pipeline Final
    pipe_final = {}
    for ep in GOLD_EP_STRS:
        with open(f"data/judged/{ep}.json", "r", encoding="utf-8") as f:
            pipe_final[ep] = json.load(f).get("judged", [])

    # 2. Pipeline S3a
    pipe_s3a = {}
    for ep in GOLD_EP_STRS:
        with open(f"data/candidates/{ep}.json", "r", encoding="utf-8") as f:
            cands = [c for c in json.load(f).get("candidates", []) if c.get("passed_all_gates")]
            cands.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            pipe_s3a[ep] = cands

    # 3. B1 last 45s
    durations = load_episode_durations()
    b1_dict = generate_b1_baseline(durations)

    # 4. B2 random (20 seeds)
    b2_dict = generate_b2_baseline(durations, seeds=range(20), windows_per_episode=3)

    # 5. B3 Tier-0
    b3_dict = {}
    for ep in GOLD_EP_STRS:
        with open(f"data/tier0/{ep}.json", "r", encoding="utf-8") as f:
            cands = [c for c in json.load(f).get("candidates", []) if c.get("passed_all_gates")]
            cands.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            b3_dict[ep] = cands

    total_gold = len(gold_windows)

    # Evaluate each system
    fin_end3, fin_both5 = evaluate_boundary_metrics_for_windows(pipe_final, gold_windows)
    s3a_end3, s3a_both5 = evaluate_boundary_metrics_for_windows(pipe_s3a, gold_windows)
    b1_end3, b1_both5 = evaluate_boundary_metrics_for_windows(b1_dict, gold_windows, top_k=1)
    b3_end3, b3_both5 = evaluate_boundary_metrics_for_windows(b3_dict, gold_windows)

    # B2 across 20 seeds
    b2_end3_list = []
    b2_both5_list = []
    for s in range(20):
        e3, b5 = evaluate_boundary_metrics_for_windows(b2_dict[s], gold_windows)
        b2_end3_list.append(e3)
        b2_both5_list.append(b5)

    b2_end3_mean = statistics.mean(b2_end3_list)
    b2_end3_min = min(b2_end3_list)
    b2_end3_max = max(b2_end3_list)

    b2_both5_mean = statistics.mean(b2_both5_list)
    b2_both5_min = min(b2_both5_list)
    b2_both5_max = max(b2_both5_list)

    return {
        "gold_windows_total": total_gold,
        "pipeline_final": {
            "end_error_le_3s": f"{fin_end3} of {total_gold}",
            "both_errors_le_5s": f"{fin_both5} of {total_gold}",
        },
        "pipeline_s3a": {
            "end_error_le_3s": f"{s3a_end3} of {total_gold}",
            "both_errors_le_5s": f"{s3a_both5} of {total_gold}",
        },
        "b1_last45s": {
            "end_error_le_3s": f"{b1_end3} of {total_gold}",
            "both_errors_le_5s": f"{b1_both5} of {total_gold}",
        },
        "b2_random": {
            "end_error_le_3s": f"mean {b2_end3_mean:.2f} of {total_gold} (range {b2_end3_min}-{b2_end3_max})",
            "both_errors_le_5s": f"mean {b2_both5_mean:.2f} of {total_gold} (range {b2_both5_min}-{b2_both5_max})",
        },
        "b3_tier0": {
            "end_error_le_3s": f"{b3_end3} of {total_gold}",
            "both_errors_le_5s": f"{b3_both5} of {total_gold}",
        },
    }


def run_e3_gate_ablation(gold_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoring_config = load_scoring_config()
    weights = scoring_config.get("weights", {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0})
    thresholds = scoring_config.get("thresholds", {})
    min_dur = thresholds.get("min_duration_s", 20.0)
    max_dur = thresholds.get("max_duration_s", 60.0)
    max_iou = thresholds.get("max_iou_overlap", 0.5)
    max_containment = thresholds.get("max_containment_overlap", 0.6)
    hook_window_s = thresholds.get("hook_window_s", 3.0)

    # Pre-load enumerated windows and beats for the 5 gold episodes
    ep_data_map = {}
    for ep in GOLD_EP_STRS:
        with open(f"data/parsed/{ep}.json", "r", encoding="utf-8") as f:
            p_data = json.load(f)
        with open(f"data/annotated/{ep}.json", "r", encoding="utf-8") as f:
            a_data = json.load(f)

        windows = enumerate_windows_for_episode(
            parsed_data=p_data,
            annotated_data=a_data,
            weights=weights,
            min_duration_s=min_dur,
            max_duration_s=max_dur,
            hook_window_s=hook_window_s,
        )
        ep_data_map[ep] = {
            "windows": windows,
            "beats": p_data.get("beats", []),
        }

    gold_by_ep = {}
    for g in gold_windows:
        gold_by_ep.setdefault(g["episode_str"], []).append(g)

    ablation_configs = [
        ("(a) Frozen gates (hook>=2, cliff>=2)", 2, 2),
        ("(b) Hook gate off (hook>=0, cliff>=2)", 0, 2),
        ("(c) Cliff gate off (hook>=2, cliff>=0)", 2, 0),
        ("(d) Both off (hook>=0, cliff>=0)", 0, 0),
    ]

    results = []

    for name, min_h, min_c in ablation_configs:
        survivors_counts = []
        top1_matches = 0
        top3_matches = 0
        gold_recalled = set()

        for ep in GOLD_EP_STRS:
            win_list = [dict(w) for w in ep_data_map[ep]["windows"]]
            beats_list = ep_data_map[ep]["beats"]

            all_eval, survs, _ = evaluate_window_gates(
                windows=win_list,
                episode_beats=beats_list,
                min_hook=min_h,
                min_cliff=min_c,
                min_duration_s=min_dur,
                max_duration_s=max_dur,
                max_iou=max_iou,
                max_containment=max_containment,
                top_k=3,
                min_survivors=0,
            )

            survivors_counts.append(len(survs))

            # S3a score sort
            survs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            top3 = survs[:3]

            ep_gold = gold_by_ep.get(ep, [])

            # Top-1
            if top3:
                c1 = top3[0]
                hit1 = any(compute_interval_iou(c1["start_s"], c1["end_s"], g["start_s"], g["end_s"]) >= 0.5 for g in ep_gold)
                if hit1:
                    top1_matches += 1

            # Top-3
            hit3 = False
            for c in top3:
                for g in ep_gold:
                    if compute_interval_iou(c["start_s"], c["end_s"], g["start_s"], g["end_s"]) >= 0.5:
                        hit3 = True
                        gold_recalled.add(g["gold_id"])
            if hit3:
                top3_matches += 1

        mean_surv = statistics.mean(survivors_counts)
        results.append({
            "configuration": name,
            "min_hook": min_h,
            "min_cliff": min_c,
            "mean_survivors_per_ep": round(mean_surv, 2),
            "top1_matches": f"{top1_matches} of 5",
            "top3_matches": f"{top3_matches} of 5",
            "gold_recall": f"{len(gold_recalled)} of 7",
        })

    return {
        "title": "POST-HOC DIAGNOSTIC - not tuned, not used for the headline",
        "ablations": results,
    }


def run_e4_failure_log_inputs(gold_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1. Gold windows failing S4
    with open("data/gold_gate_diagnosis.json", "r", encoding="utf-8") as f:
        diag_list = json.load(f)

    gold_failing_s4 = []
    for g in diag_list:
        failed = [f for f in g.get("failed_gates", []) if f != "PASS_ALL_GATES"]
        if failed:
            gold_failing_s4.append({
                "gold_id": g.get("gold_id"),
                "episode": g.get("episode"),
                "timing": g.get("timing"),
                "s3a_features": g.get("s3a_features", {}),
                "gate_reason_codes": failed,
                "judge_scores": g.get("judge_scores", {}),
            })

    # 2. Pipeline top-3 windows with no gold match (IoU < 0.5)
    gold_by_ep = {}
    for g in gold_windows:
        gold_by_ep.setdefault(g["episode_str"], []).append(g)

    unmatched_top3_windows = []

    for ep in GOLD_EP_STRS:
        with open(f"data/judged/{ep}.json", "r", encoding="utf-8") as f:
            judged_list = json.load(f).get("judged", [])

        ep_gold = gold_by_ep.get(ep, [])

        for rank_idx, c in enumerate(judged_list[:3], start=1):
            c_st = c["start_s"]
            c_en = c["end_s"]
            cid = c["candidate_id"]

            best_iou = 0.0
            best_gid = None
            for g in ep_gold:
                val = compute_interval_iou(c_st, c_en, g["start_s"], g["end_s"])
                if val > best_iou:
                    best_iou = val
                    best_gid = g["gold_id"]

            if best_iou < 0.5:
                unmatched_top3_windows.append({
                    "episode": ep,
                    "candidate_id": cid,
                    "rank_in_episode": rank_idx,
                    "start_s": c_st,
                    "end_s": c_en,
                    "duration_s": c["duration_s"],
                    "best_gold_id": best_gid,
                    "best_gold_iou": round(best_iou, 3),
                    "s3a_features": {
                        "s3a_score": c.get("s3a_score"),
                    },
                    "gate_reason_codes": c.get("judge_gate_reason", "PASSED"),
                    "judge_scores": {
                        "hook": c.get("hook"),
                        "cliff": c.get("cliff"),
                        "clarity": c.get("clarity"),
                        "payoff": c.get("payoff"),
                        "final_score": c.get("final_score"),
                    },
                })

    return {
        "gold_windows_failing_s4": gold_failing_s4,
        "unmatched_pipeline_top3_windows": unmatched_top3_windows,
    }


def main() -> None:
    gold_windows = load_gold_windows()

    e1 = run_e1_duration_report(gold_windows)
    e2 = run_e2_boundary_metrics(gold_windows)
    e3 = run_e3_gate_ablation(gold_windows)
    e4 = run_e4_failure_log_inputs(gold_windows)

    diagnostics = {
        "E1_duration_report": e1,
        "E2_boundary_metrics": e2,
        "E3_gate_ablation": e3,
        "E4_failure_log_inputs": e4,
    }

    out_dir = Path("posthoc")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "diagnostics.json"
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    # Generate Markdown table
    md_lines = [
        "# POST-HOC DIAGNOSTICS REPORT",
        "",
        "> [!NOTE]",
        "> POST-HOC DIAGNOSTIC - not tuned, not used for the headline.",
        "",
        "## E1. Duration Distribution",
        "",
        "### Rank-1 Window Durations for 10 Reserved Episodes",
        "| Episode | Rank-1 Duration (s) |",
        "|---|---|",
    ]
    for ep, dur in e1["reserved_rank1_durations_s"].items():
        md_lines.append(f"| {ep} | {dur:.2f}s |")

    md_lines.extend([
        "",
        "### Gold Window Durations (7 Reference Windows)",
        "| Gold ID | Duration (s) |",
        "|---|---|",
    ])
    for gid, dur in e1["gold_durations_s"].items():
        md_lines.append(f"| {gid} | {dur:.2f}s |")

    md_lines.extend([
        "",
        f"- Pipeline Top-3 Windows within 1.0s of 20.0s (Reserved Episodes): **{e1['top3_within_1s_of_20_reserved']}**",
        f"- Pipeline Top-3 Windows within 1.0s of 20.0s (5 Gold Episodes): **{e1['top3_within_1s_of_20_gold_eps']}**",
        "",
        "## E2. Boundary Error Metrics (5 Gold Episodes, 7 Gold Windows)",
        "",
        "| System | Gold Windows Matched (\\|end error\\| <= 3s) | Gold Windows Matched (\\|start\\| <= 5s & \\|end\\| <= 5s) |",
        "|---|---|---|",
        f"| Pipeline Final (Top 3) | {e2['pipeline_final']['end_error_le_3s']} | {e2['pipeline_final']['both_errors_le_5s']} |",
        f"| Pipeline S3a Order (Top 3) | {e2['pipeline_s3a']['end_error_le_3s']} | {e2['pipeline_s3a']['both_errors_le_5s']} |",
        f"| B1 Last 45s (1 Window) | {e2['b1_last45s']['end_error_le_3s']} | {e2['b1_last45s']['both_errors_le_5s']} |",
        f"| B2 Random (20 seeds, 3 Win) | {e2['b2_random']['end_error_le_3s']} | {e2['b2_random']['both_errors_le_5s']} |",
        f"| B3 Tier-0 (Top 3) | {e2['b3_tier0']['end_error_le_3s']} | {e2['b3_tier0']['both_errors_le_5s']} |",
        "",
        "## E3. Gate Ablation on 5 Gold Episodes (S3a Order, No Judge)",
        "",
        "**POST-HOC DIAGNOSTIC - not tuned, not used for the headline**",
        "",
        "| Configuration | Mean Survivors / Ep | Top-1 Matches (IoU >= 0.5) | Top-3 Matches (IoU >= 0.5) | Gold Recall (IoU >= 0.5) |",
        "|---|---|---|---|---|",
    ])
    for ab in e3["ablations"]:
        md_lines.append(
            f"| {ab['configuration']} | {ab['mean_survivors_per_ep']} | {ab['top1_matches']} | {ab['top3_matches']} | {ab['gold_recall']} |"
        )

    md_lines.extend([
        "",
        "## E4. Failure-Log Inputs (Metadata Only, Zero Text)",
        "",
        "### Gold Windows Failing S4 Hard Gates",
        "| Gold ID | Episode | Window Interval | S3a Features (H, Q, R, N, T, Score) | S4 Failed Reason Codes | Judge Scores (H, Q, C, P, Final) |",
        "|---|---|---|---|---|---|",
    ])
    for g in e4["gold_windows_failing_s4"]:
        f_feat = f"H={g['s3a_features'].get('hook')}, Q={g['s3a_features'].get('cliff')}, R={g['s3a_features'].get('resolved_end')}, N={g['s3a_features'].get('context')}, T={g['s3a_features'].get('intensity')}, S={g['s3a_features'].get('score')}"
        j_feat = f"H={g['judge_scores'].get('hook')}, Q={g['judge_scores'].get('cliff')}, C={g['judge_scores'].get('clarity')}, P={g['judge_scores'].get('payoff')}, Tot={g['judge_scores'].get('final_score')}"
        reasons = ", ".join(g["gate_reason_codes"])
        timing = g.get("timing", "")
        md_lines.append(f"| {g['gold_id']} | {g['episode']} | {timing} | {f_feat} | {reasons} | {j_feat} |")

    md_lines.extend([
        "",
        "### Pipeline Top-3 Windows with No Gold Match (IoU < 0.5)",
        "| Episode | Candidate ID | Rank | Window Interval | Duration | Best Gold ID | Best IoU | S3a Score | Judge Scores (H, Q, C, P, Final) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for u in e4["unmatched_pipeline_top3_windows"]:
        j_feat = f"H={u['judge_scores']['hook']}, Q={u['judge_scores']['cliff']}, C={u['judge_scores']['clarity']}, P={u['judge_scores']['payoff']}, Tot={u['judge_scores']['final_score']}"
        md_lines.append(
            f"| {u['episode']} | {u['candidate_id']} | #{u['rank_in_episode']} | [{u['start_s']:.2f}s - {u['end_s']:.2f}s] | {u['duration_s']:.2f}s | {u['best_gold_id']} | {u['best_gold_iou']:.3f} | {u['s3a_features']['s3a_score']:.2f} | {j_feat} |"
        )

    md_path = out_dir / "diagnostics.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated diagnostics in {json_path} and {md_path}")


if __name__ == "__main__":
    main()
