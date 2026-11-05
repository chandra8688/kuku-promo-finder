"""Run Part F: Judge Validity Controls on Dev Episodes.

Controls per dev episode (max 15 extra calls total):
1. Up to 2 windows rejected only for RESOLVED_ENDING.
2. The B1 baseline window (last 45s, non-camera beats).
3. One seeded B2 baseline window (seed=0, window 0, non-camera beats).

Reports:
- Mean judge scores and resolved_after rate for controls vs selected picks.
- Assessment of whether judge marks resolved endings.
Never prints screenplay text or character names!
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.baselines import generate_b1_baseline, generate_b2_baseline, load_episode_durations
from src.judge import judge_candidate, load_judge_system_prompt
from src.llm import call

DEV_EPISODES = ["episode-001", "episode-002", "episode-005"]


def populate_beats_for_time_range(
    ep_id: str,
    start_s: float,
    end_s: float,
    parsed_data: Dict[str, Any],
) -> List[str]:
    """Find all non-camera beat IDs that start >= start_s and end <= end_s."""
    all_beats = parsed_data.get("beats", [])
    matched_ids = [
        b["beat_id"]
        for b in all_beats
        if b.get("kind") != "CAMERA"
        and b.get("start_s", 0.0) >= start_s - 0.05
        and b.get("end_s", 0.0) <= end_s + 0.05
    ]
    # If strictly enclosed has too few beats, take beats overlapping range
    if not matched_ids:
        matched_ids = [
            b["beat_id"]
            for b in all_beats
            if b.get("kind") != "CAMERA"
            and b.get("end_s", 0.0) > start_s
            and b.get("start_s", 0.0) < end_s
        ]
    return matched_ids


def collect_controls() -> Dict[str, List[Dict[str, Any]]]:
    durations = load_episode_durations("data/parsed")
    b1_dict = generate_b1_baseline(durations)
    b2_dict = generate_b2_baseline(durations, seeds=[0], windows_per_episode=1)[0]

    controls: Dict[str, List[Dict[str, Any]]] = {}

    for ep in DEV_EPISODES:
        controls[ep] = []
        parsed_p = Path(f"data/parsed/{ep}.json")
        parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
        cand_p = Path(f"data/candidates/{ep}.json")
        cand_data = json.loads(cand_p.read_text(encoding="utf-8"))

        # 1. Up to 2 windows rejected only for RESOLVED_ENDING
        res_cand = [
            c for c in cand_data.get("candidates", [])
            if c.get("first_failed_gate") == "RESOLVED_ENDING"
        ]
        # Sort by score descending to get best 2
        res_cand.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for idx, rc in enumerate(res_cand[:2]):
            c_copy = dict(rc)
            c_copy["control_type"] = "RESOLVED_ENDING_REJECT"
            c_copy["control_name"] = f"ctrl_resolved_{idx+1}"
            controls[ep].append(c_copy)

        # 2. B1 baseline window
        b1_win = b1_dict.get(ep, [{}])[0]
        b1_beats = populate_beats_for_time_range(
            ep, b1_win["start_s"], b1_win["end_s"], parsed_data
        )
        b1_cand = {
            "candidate_id": f"{ep}_ctrl_b1",
            "episode": ep,
            "start_s": b1_win["start_s"],
            "end_s": b1_win["end_s"],
            "duration_s": b1_win["duration_s"],
            "beat_ids": b1_beats,
            "control_type": "BASELINE_B1",
            "control_name": "ctrl_b1_last45s",
        }
        controls[ep].append(b1_cand)

        # 3. One seeded B2 baseline window
        b2_win = b2_dict.get(ep, [{}])[0]
        b2_beats = populate_beats_for_time_range(
            ep, b2_win["start_s"], b2_win["end_s"], parsed_data
        )
        b2_cand = {
            "candidate_id": f"{ep}_ctrl_b2",
            "episode": ep,
            "start_s": b2_win["start_s"],
            "end_s": b2_win["end_s"],
            "duration_s": b2_win["duration_s"],
            "beat_ids": b2_beats,
            "control_type": "BASELINE_B2",
            "control_name": "ctrl_b2_seed0",
        }
        controls[ep].append(b2_cand)

    return controls


def run_controls_judge(role: str = "judge"):
    controls_by_ep = collect_controls()
    total_controls = sum(len(c) for c in controls_by_ep.values())
    print(f"[Part F] Running judge (role={role}) on {total_controls} control candidates...")

    judged_controls = []

    for ep, cand_list in controls_by_ep.items():
        parsed_p = Path(f"data/parsed/{ep}.json")
        ann_p = Path(f"data/annotated/{ep}.json")
        parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
        ann_data = json.loads(ann_p.read_text(encoding="utf-8"))

        for cand in cand_list:
            res = judge_candidate(
                candidate=cand,
                parsed_data=parsed_data,
                annotated_data=ann_data,
                role=role,
                stage=f"s5_judge_control_{role}",
                enable_cache=True,
            )
            res["control_type"] = cand.get("control_type")
            res["control_name"] = cand.get("control_name")
            judged_controls.append(res)
            print(f"  {ep} {cand.get('control_name')}: hook={res.get('hook')}, cliff={res.get('cliff')}, clarity={res.get('clarity')}, payoff={res.get('payoff')}, final_score={res.get('final_score')}, resolved_after={res.get('resolved_after')}")

    out_file = Path(f"data/judged_controls_{role}.json")
    out_file.write_text(json.dumps(judged_controls, indent=2), encoding="utf-8")
    print(f"[Part F] Saved {len(judged_controls)} judged controls to {out_file}")
    return judged_controls


if __name__ == "__main__":
    run_controls_judge(role="judge")

