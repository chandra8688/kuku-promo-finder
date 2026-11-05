"""Gate Diagnosis on 7 Gold Windows (scripts/diagnose_gold.py).

Diagnostic only (no tuning). For each of the 7 gold windows, reports:
- which S4 gates it fails (reason codes)
- its S3a score rank among the episode's enumerated windows
- its S3a narrative features (hook, cliff, resolved_end, context, intensity, score)
- judge scores when evaluated once by judge role (hook, cliff, resolved_after, clarity, payoff, final_score)

Never prints screenplay text or character names.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.gates import evaluate_window_gates
from src.judge import CandidateEvaluation, EpisodeBatchJudgeOutput, build_judge_prompt, load_judge_system_prompt
from src.llm import LLMResult, call
from src.windows import check_straddle_cuts, enumerate_windows_for_episode, load_scoring_config


def diagnose_gold_windows() -> List[Dict[str, Any]]:
    with open("gold/gold.csv", "r", encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))

    cfg = load_scoring_config()
    weights = cfg.get("weights", {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0})
    thresholds = cfg.get("thresholds", {})
    min_dur = thresholds.get("min_duration_s", 20.0)
    max_dur = thresholds.get("max_duration_s", 60.0)
    min_h = thresholds.get("min_hook", 2)
    min_c = thresholds.get("min_cliff", 2)
    hook_window_s = thresholds.get("hook_window_s", 3.0)

    diagnosis_results = []

    for r in gold_rows:
        gid = r["gold_id"]
        ep_num = int(r["episode"])
        ep_id = f"episode-{ep_num:03d}"
        gold_st = float(r["start_s"])
        gold_en = float(r["end_s"])
        gold_dur = round(gold_en - gold_st, 2)

        with open(f"data/parsed/{ep_id}.json", "r", encoding="utf-8") as pf:
            parsed_data = json.load(pf)
        with open(f"data/annotated/{ep_id}.json", "r", encoding="utf-8") as af:
            ann_data = json.load(af)

        all_beats = parsed_data.get("beats", [])
        annotations = ann_data.get("annotations", {})

        # 1. Enumerate all windows for this episode to determine S3a rank
        enumerated = enumerate_windows_for_episode(
            parsed_data=parsed_data,
            annotated_data=ann_data,
            weights=weights,
            min_duration_s=min_dur,
            max_duration_s=max_dur,
            hook_window_s=hook_window_s,
        )
        total_enumerated = len(enumerated)

        # 2. Extract non-camera beats for this gold window
        clip_beats = [
            b for b in all_beats
            if b.get("kind") != "CAMERA"
            and (b.get("start_s", 0.0) >= gold_st - 0.05)
            and (b.get("end_s", 0.0) <= gold_en + 0.05)
        ]
        if not clip_beats:
            clip_beats = [
                b for b in all_beats
                if b.get("kind") != "CAMERA"
                and (b.get("end_s", 0.0) > gold_st)
                and (b.get("start_s", 0.0) < gold_en)
            ]

        beat_ids = [b["beat_id"] for b in clip_beats]
        first_b_id = beat_ids[0] if beat_ids else ""
        last_b_id = beat_ids[-1] if beat_ids else ""

        # Compute S3a features
        first_ann = annotations.get(first_b_id) or {}
        last_ann = annotations.get(last_b_id) or {}

        # Hook = max S2 hook over beats starting within hook_window_s of window start
        candidate_hook_beats = [
            b["beat_id"] for b in clip_beats
            if b.get("start_s", 0.0) <= gold_st + hook_window_s + 0.05
        ]
        hook_vals = [(annotations.get(bid) or {}).get("h", 0) for bid in candidate_hook_beats]
        feat_hook = max(hook_vals) if hook_vals else (first_ann.get("h", 0) or 0)
        feat_cliff = last_ann.get("q", 0) or 0
        feat_resolved = bool(last_ann.get("r", False))
        feat_context = first_ann.get("n", 0) or 0
        intensities = [(annotations.get(bid) or {}).get("t", 0) for bid in beat_ids]
        feat_intensity = round(sum(intensities) / max(1, len(intensities)), 3)

        feat_score = round(
            weights.get("hook", 1.0) * feat_hook
            + weights.get("cliff", 1.0) * feat_cliff
            + weights.get("intensity", 1.0) * feat_intensity
            - weights.get("context", 1.0) * feat_context,
            3,
        )

        # 3. Check S4 gates directly
        failed_gates = []
        if not (min_dur <= gold_dur <= max_dur):
            failed_gates.append(f"DURATION({gold_dur}s)")

        straddles_spoken, _ = check_straddle_cuts(
            start_cut=gold_st,
            end_cut=gold_en,
            all_beats=all_beats,
            window_beat_ids=set(beat_ids),
        )
        if straddles_spoken:
            failed_gates.append("CUT_STRADDLES_SPOKEN")

        if feat_hook < min_h:
            failed_gates.append(f"HOOK_TOO_WEAK({feat_hook}<{min_h})")

        if feat_resolved:
            failed_gates.append("RESOLVED_ENDING")

        if feat_cliff < min_c:
            failed_gates.append(f"WEAK_CLIFFHANGER({feat_cliff}<{min_c})")

        # Rank among enumerated windows by score desc
        # Count how many enumerated windows have score > feat_score
        s3a_rank = 1 + sum(1 for w in enumerated if w.get("score", 0.0) > feat_score)

        # 4. Judge evaluation (single call per gold window, role judge)
        gold_cand_dict = {
            "candidate_id": gid,
            "episode": ep_id,
            "beat_ids": beat_ids,
            "start_s": gold_st,
            "end_s": gold_en,
            "duration_s": gold_dur,
        }

        sys_prompt = load_judge_system_prompt(Path("prompts/judge_v1.md"))
        u_prompt, n_beats, after_avail = build_judge_prompt(gold_cand_dict, parsed_data, ann_data)

        # Schema for single candidate evaluation
        j_res = call(
            role="judge",
            system=sys_prompt,
            user=u_prompt,
            schema=CandidateEvaluation,
            stage="s5_gate_diagnosis",
            prompt_version="judge_v1",
            enable_cache=True,
        )
        j_out: Optional[CandidateEvaluation] = j_res.parsed
        if j_out:
            judge_scores = {
                "hook": j_out.hook,
                "cliff": j_out.cliff,
                "resolved_after": j_out.resolved_after,
                "clarity": j_out.clarity,
                "payoff": j_out.payoff,
                "final_score": j_out.hook + j_out.cliff + j_out.clarity + j_out.payoff,
            }
        else:
            judge_scores = {
                "hook": 0,
                "cliff": 0,
                "resolved_after": False,
                "clarity": 0,
                "payoff": 0,
                "final_score": 0,
            }

        diag_entry = {
            "gold_id": gid,
            "episode": ep_id,
            "timing": f"[{gold_st:.2f}s - {gold_en:.2f}s, {gold_dur:.1f}s]",
            "s3a_features": {
                "hook": feat_hook,
                "cliff": feat_cliff,
                "resolved_end": feat_resolved,
                "context": feat_context,
                "intensity": feat_intensity,
                "score": feat_score,
            },
            "s3a_rank": f"{s3a_rank} of {total_enumerated}",
            "failed_gates": failed_gates if failed_gates else ["PASS_ALL_GATES"],
            "judge_scores": judge_scores,
        }
        diagnosis_results.append(diag_entry)

    return diagnosis_results


def print_diagnosis_report(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print("GATE DIAGNOSIS REPORT (7 Gold Windows, S4 Gates & S3a Rank)")
    print("=" * 110)
    header = (
        f"{'Gold ID':<11} | {'Ep':<12} | {'Timing':<24} | {'S3a Rank':<14} | "
        f"{'S3a (H/Q/R/N/T/Score)':<24} | {'Failed Gates':<28} | {'Judge Score'}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        gid = r["gold_id"]
        ep = r["episode"]
        timing = r["timing"]
        rank_str = r["s3a_rank"]
        f = r["s3a_features"]
        feat_str = f"H={f['hook']} Q={f['cliff']} R={int(f['resolved_end'])} N={f['context']} T={f['intensity']:.2f} S={f['score']:.2f}"
        gates_str = ", ".join(r["failed_gates"])
        j = r["judge_scores"]
        judge_str = f"H={j['hook']} Q={j['cliff']} R_aft={int(j['resolved_after'])} C={j['clarity']} P={j['payoff']} (Tot={j['final_score']})"

        print(
            f"{gid:<11} | {ep:<12} | {timing:<24} | {rank_str:<14} | "
            f"{feat_str:<24} | {gates_str:<28} | {judge_str}"
        )

    print("=" * 110 + "\n")

    out_file = Path("data/gold_gate_diagnosis.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote diagnosis JSON to {out_file}")


if __name__ == "__main__":
    res = diagnose_gold_windows()
    print_diagnosis_report(res)

