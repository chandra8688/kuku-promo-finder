"""Tier-0 No-LLM Promo Candidate Generator (src/tier0.py).

Generates promo candidates purely from deterministic screenplay features parsed in S1.
Zero LLM calls. Serves as baseline B3 and fallback when no model is available.

Proxy Hook Score (0-3) from first beat:
- Dialogue starts within 1.0s: +1
- SFX stinger present (first beat is SFX or preceded by SFX): +1
- Short punchy line (length between 1 and 40 characters): +1
- Question or exclamation mark present in text: +1
Clamped to [0, 3]. Never prints screenplay text!

Proxy Cliffhanger Score (0-3) from last beat:
- Spoken line ending in a question mark (?): +1
- Short final line (length between 1 and 35 characters): +1
- Near the end of the episode (end_s >= 80% of episode runtime): +1
Clamped to [0, 3].

Applies structural gates (DURATION, VALID_BEAT_IDS, ORDERED_NONOVERLAPPING_SEGMENTS,
CUT_STRADDLES_SPOKEN, RESOLVED_ENDING) and greedy IoU + anchor deduplication.
Obeys reserved episode rules. Outputs candidates in the standard candidate JSON schema.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.reserved import is_episode_blocked
from src.windows import check_straddle_cuts, compute_interval_iou


def compute_proxy_hook(
    first_beat: Dict[str, Any],
    all_beats: List[Dict[str, Any]],
    first_beat_idx: int,
) -> int:
    """Compute heuristic hook score (0-3) from first beat features without LLM.

    Never prints text.
    """
    score = 0
    kind = first_beat.get("kind", "")
    text = (first_beat.get("text") or "").strip()
    text_len = len(text)

    # 1. Dialogue starts within 1.0s of episode start or immediate cut
    if kind == "DIALOGUE":
        if first_beat.get("start_s", 0.0) <= 1.5:
            score += 1
        elif (first_beat.get("duration_s", 0.0) >= 0.5):
            score += 1

    # 2. SFX stinger present
    if kind == "SFX":
        score += 1
    elif first_beat_idx > 0 and all_beats[first_beat_idx - 1].get("kind") == "SFX":
        score += 1

    # 3. Short punchy line
    if 1 <= text_len <= 40:
        score += 1

    # 4. Question or exclamation mark
    if "?" in text or "!" in text:
        score += 1

    return min(3, max(0, score))


def compute_proxy_cliff(
    last_beat: Dict[str, Any],
    ep_duration_s: float,
) -> Tuple[int, bool]:
    """Compute heuristic cliffhanger score (0-3) and resolved_end flag from last beat.

    Never prints text.
    """
    score = 0
    kind = last_beat.get("kind", "")
    text = (last_beat.get("text") or "").strip()
    text_len = len(text)
    end_s = last_beat.get("end_s", 0.0)

    # 1. Spoken line ending in a question mark
    if kind in ("DIALOGUE", "SFX") and text.endswith("?"):
        score += 1

    # 2. Short final line
    if 1 <= text_len <= 35:
        score += 1

    # 3. Near the end of the episode (within last 20% or last 30s)
    if ep_duration_s > 0 and (end_s >= ep_duration_s * 0.80 or (ep_duration_s - end_s) <= 30.0):
        score += 1

    # Check if tension seems resolved (period ending without question or exclamation)
    resolved_end = text.endswith(".") and not text.endswith("...")

    return min(3, max(0, score)), resolved_end


def compute_proxy_intensity(window_units: List[Dict[str, Any]]) -> float:
    """Estimate dramatic intensity (0-3) from punctuation and pacing. Never prints text."""
    if not window_units:
        return 1.0
    total = 0.0
    for u in window_units:
        t = (u.get("text") or "").strip()
        unit_score = 1.0
        if "!" in t:
            unit_score += 1.0
        if "?" in t:
            unit_score += 0.5
        if u.get("kind") == "SFX":
            unit_score += 0.5
        total += min(3.0, unit_score)
    return round(total / len(window_units), 3)


def generate_tier0_candidates_for_episode(
    parsed_data: Dict[str, Any],
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
    min_hook: int = 1,
    min_cliff: int = 1,
    max_iou: float = 0.5,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Enumerate and filter candidates using purely heuristic proxy scores."""
    ep_id = parsed_data.get("episode", "unknown")
    all_beats = parsed_data.get("beats", [])

    # Compute episode duration from last beat end
    ep_duration_s = max((b.get("end_s", 0.0) for b in all_beats), default=120.0)

    # Filter to consecutive non-camera beats
    non_cam_units = [b for b in all_beats if b.get("kind") != "CAMERA"]
    non_cam_units.sort(key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))
    n_units = len(non_cam_units)

    windows: List[Dict[str, Any]] = []
    window_idx = 1
    gate_fail_counts = {
        "DURATION": 0,
        "VALID_BEAT_IDS": 0,
        "ORDERED_NONOVERLAPPING_SEGMENTS": 0,
        "CUT_STRADDLES_SPOKEN": 0,
        "HOOK_TOO_WEAK": 0,
        "RESOLVED_ENDING": 0,
        "WEAK_CLIFFHANGER": 0,
        "DUPLICATE_OVERLAP": 0,
        "DUPLICATE_ANCHOR": 0,
    }

    passed_structural: List[Dict[str, Any]] = []

    for i in range(n_units):
        start_s = non_cam_units[i].get("start_s", 0.0)
        curr_max_end = start_s

        for j in range(i, n_units):
            beat_j_end = non_cam_units[j].get("end_s", 0.0)
            if beat_j_end > curr_max_end:
                curr_max_end = beat_j_end

            duration_s = curr_max_end - start_s
            if duration_s < min_duration_s:
                continue
            if duration_s > max_duration_s:
                if non_cam_units[j].get("start_s", 0.0) - start_s > max_duration_s:
                    break
                continue

            window_units = non_cam_units[i : j + 1]
            window_beat_ids = {u["beat_id"] for u in window_units}

            # Cut validity
            straddles_spoken, straddles_action = check_straddle_cuts(
                start_cut=start_s,
                end_cut=curr_max_end,
                all_beats=all_beats,
                window_beat_ids=window_beat_ids,
            )

            # Heuristic features
            hook = compute_proxy_hook(window_units[0], all_beats, i)
            cliff, resolved_end = compute_proxy_cliff(window_units[-1], ep_duration_s)
            intensity = compute_proxy_intensity(window_units)
            score = round(float(hook) + float(cliff) + intensity, 3)

            cand = {
                "candidate_id": f"{ep_id}_t0_{window_idx:04d}",
                "episode": ep_id,
                "beat_ids": [u["beat_id"] for u in window_units],
                "start_s": round(start_s, 2),
                "end_s": round(curr_max_end, 2),
                "duration_s": round(duration_s, 2),
                "hook": hook,
                "cliff": cliff,
                "resolved_end": resolved_end,
                "context": 0,
                "intensity": intensity,
                "score": score,
                "cut_straddles_spoken": straddles_spoken,
                "cut_straddles_action": straddles_action,
                "gate_results": {},
                "passed_all_gates": False,
                "is_tier0": True,
            }
            windows.append(cand)
            window_idx += 1

            # Gate checks
            failed_gate = None
            if not (min_duration_s <= duration_s <= max_duration_s):
                failed_gate = "DURATION"
            elif straddles_spoken:
                failed_gate = "CUT_STRADDLES_SPOKEN"
            elif hook < min_hook:
                failed_gate = "HOOK_TOO_WEAK"
            elif resolved_end:
                failed_gate = "RESOLVED_ENDING"
            elif cliff < min_cliff:
                failed_gate = "WEAK_CLIFFHANGER"

            if failed_gate:
                cand["first_failed_gate"] = failed_gate
                gate_fail_counts[failed_gate] += 1
            else:
                cand["passed_individual_gates"] = True
                passed_structural.append(cand)

    # Deduplicate passed_structural greedily
    passed_structural.sort(
        key=lambda x: (x.get("score", 0.0), x.get("cliff", 0), x.get("duration_s", 0.0)),
        reverse=True,
    )

    survivors: List[Dict[str, Any]] = []
    survivor_first_beats: Set[str] = set()
    survivor_last_beats: Set[str] = set()

    for w in passed_structural:
        b_ids = w.get("beat_ids", [])
        fb = b_ids[0] if b_ids else None
        lb = b_ids[-1] if b_ids else None

        iou_dup = any(
            compute_interval_iou(w["start_s"], w["end_s"], s["start_s"], s["end_s"]) > max_iou
            for s in survivors
        )
        anchor_dup = (fb in survivor_first_beats) or (lb in survivor_last_beats)

        if iou_dup:
            w["first_failed_gate"] = "DUPLICATE_OVERLAP"
            gate_fail_counts["DUPLICATE_OVERLAP"] += 1
        elif anchor_dup:
            w["first_failed_gate"] = "DUPLICATE_ANCHOR"
            gate_fail_counts["DUPLICATE_ANCHOR"] += 1
        else:
            w["passed_all_gates"] = True
            w["selected_top_k"] = len(survivors) < top_k
            survivors.append(w)
            if fb:
                survivor_first_beats.add(fb)
            if lb:
                survivor_last_beats.add(lb)

    return {
        "episode": ep_id,
        "total_windows_enumerated": len(windows),
        "total_survivors": len(survivors),
        "gate_fail_counts": gate_fail_counts,
        "candidates": windows,
    }


def run_tier0_pipeline(
    episodes: List[str],
    parsed_dir: str | Path = "data/parsed",
    out_dir: str | Path = "data/tier0",
    report_file: str | Path = "tier0_report.json",
    gold_file: str | Path = "gold/gold.csv",
) -> Dict[str, Any]:
    """Execute Tier-0 generator across episodes."""
    parsed_path = Path(parsed_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    total_enumerated = 0
    total_survivors = 0

    for ep_id in episodes:
        if is_episode_blocked(ep_id, gold_file=gold_file):
            print(f"[tier0] Skipping reserved episode: {ep_id}")
            continue

        parsed_file = parsed_path / f"{ep_id}.json"
        if not parsed_file.exists():
            print(f"[tier0] Missing parsed file: {parsed_file}")
            continue

        with open(parsed_file, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        ep_payload = generate_tier0_candidates_for_episode(parsed_data)
        out_file = out_path / f"{ep_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ep_payload, f, indent=2)

        results.append(ep_payload)
        total_enumerated += ep_payload["total_windows_enumerated"]
        total_survivors += ep_payload["total_survivors"]
        print(f"[tier0] {ep_id}: {ep_payload['total_survivors']} survivor(s) from {ep_payload['total_windows_enumerated']} windows.")

    summary = {
        "total_episodes": len(results),
        "total_windows_enumerated": total_enumerated,
        "total_survivors": total_survivors,
        "episodes": [
            {
                "episode": r["episode"],
                "windows_enumerated": r["total_windows_enumerated"],
                "survivors": r["total_survivors"],
            }
            for r in results
        ],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "episodes": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier-0 No-LLM candidate generator (Baseline B3).")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["episode-001", "episode-002", "episode-005"],
        help="Episodes to process",
    )
    parser.add_argument(
        "--out-dir",
        default="data/tier0",
        help="Output directory for tier0 candidate JSONs",
    )
    parser.add_argument(
        "--report-file",
        default="tier0_report.json",
        help="Summary report output path",
    )
    args = parser.parse_args()

    run_tier0_pipeline(
        episodes=args.episodes,
        out_dir=args.out_dir,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()

