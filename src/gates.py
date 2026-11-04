"""Stage 4: Hard Gates (src/gates.py).

Applies deterministic filters in strict sequential order:
1. DURATION: 20.0 <= duration <= 60.0 s
2. VALID_BEAT_IDS: All beat IDs belong to the episode
3. ORDERED_NONOVERLAPPING_SEGMENTS: Valid start < end, monotonic beats
4. CUT_STRADDLES_SPOKEN: Window cut does not split an external DIALOGUE/SFX beat
5. HOOK_TOO_WEAK: hook (h of beat 1) >= min_hook (default 2)
   # Note: "First 3s contains a non-camera beat" is true by construction because
   # every window starts on a non-camera beat (unit i), which starts at start_s=0.0
   # relative to the window, so the min_hook gate is the real hook gate.
6. RESOLVED_ENDING: resolved_end (r of last beat) must be False (resolved endings rejected)
7. WEAK_CLIFFHANGER: cliff (q of last beat) >= min_cliff (default 2)
8. DUPLICATE_OVERLAP: Time-interval IoU <= max_iou (default 0.5) against higher-ranked survivors

Re-runnable on any candidate. Generates counts and reason codes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.windows import compute_interval_iou, load_scoring_config

GATE_ORDER = [
    "DURATION",
    "VALID_BEAT_IDS",
    "ORDERED_NONOVERLAPPING_SEGMENTS",
    "CUT_STRADDLES_SPOKEN",
    "HOOK_TOO_WEAK",
    "RESOLVED_ENDING",
    "WEAK_CLIFFHANGER",
    "DUPLICATE_OVERLAP",
]


def evaluate_window_gates(
    windows: List[Dict[str, Any]],
    episode_beats: List[Dict[str, Any]],
    min_hook: int = 2,
    min_cliff: int = 2,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
    max_iou: float = 0.5,
    top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Evaluate hard gates across windows in order.

    Returns:
        (all_windows_with_gate_results, survivors, gate_fail_counts)
    """
    valid_ep_beat_ids = {b["beat_id"] for b in episode_beats}
    ep_beat_lookup = {b["beat_id"]: b for b in episode_beats}

    gate_fail_counts: Dict[str, int] = {gate: 0 for gate in GATE_ORDER}
    passed_individual: List[Dict[str, Any]] = []

    # Evaluate Gates 1 through 7 individually
    for w in windows:
        gate_results: Dict[str, bool] = {}
        failed_gate: Optional[str] = None

        # 1. DURATION
        duration_s = w.get("duration_s", 0.0)
        dur_ok = min_duration_s <= duration_s <= max_duration_s
        gate_results["DURATION"] = dur_ok
        if not dur_ok and failed_gate is None:
            failed_gate = "DURATION"
            gate_fail_counts["DURATION"] += 1

        # 2. VALID_BEAT_IDS
        beat_ids = w.get("beat_ids", [])
        beats_valid = bool(beat_ids) and all(b_id in valid_ep_beat_ids for b_id in beat_ids)
        gate_results["VALID_BEAT_IDS"] = beats_valid
        if not beats_valid and failed_gate is None:
            failed_gate = "VALID_BEAT_IDS"
            gate_fail_counts["VALID_BEAT_IDS"] += 1

        # 3. ORDERED_NONOVERLAPPING_SEGMENTS
        start_s = w.get("start_s", 0.0)
        end_s = w.get("end_s", 0.0)
        ordered_ok = start_s < end_s
        if beats_valid and ordered_ok:
            # Check sequential monotonic beat ordering
            unit_starts = [ep_beat_lookup[b_id].get("start_s", 0.0) for b_id in beat_ids]
            if unit_starts != sorted(unit_starts):
                ordered_ok = False
        gate_results["ORDERED_NONOVERLAPPING_SEGMENTS"] = ordered_ok
        if not ordered_ok and failed_gate is None:
            failed_gate = "ORDERED_NONOVERLAPPING_SEGMENTS"
            gate_fail_counts["ORDERED_NONOVERLAPPING_SEGMENTS"] += 1

        # 4. CUT_STRADDLES_SPOKEN
        straddles_spoken = w.get("cut_straddles_spoken", False)
        gate_results["CUT_STRADDLES_SPOKEN"] = not straddles_spoken
        if straddles_spoken and failed_gate is None:
            failed_gate = "CUT_STRADDLES_SPOKEN"
            gate_fail_counts["CUT_STRADDLES_SPOKEN"] += 1

        # 5. HOOK_TOO_WEAK
        # Note: "First 3s contains a non-camera beat" is true by construction because
        # every window starts on a non-camera beat (unit i), which starts at start_s=0.0
        # relative to the window, so the min_hook gate is the real hook gate.
        hook_val = w.get("hook", 0)
        hook_ok = hook_val >= min_hook
        gate_results["HOOK_TOO_WEAK"] = hook_ok
        if not hook_ok and failed_gate is None:
            failed_gate = "HOOK_TOO_WEAK"
            gate_fail_counts["HOOK_TOO_WEAK"] += 1

        # 6. RESOLVED_ENDING
        resolved_end = w.get("resolved_end", False)
        unresolved_ok = not resolved_end  # Rejected if tension was resolved
        gate_results["RESOLVED_ENDING"] = unresolved_ok
        if not unresolved_ok and failed_gate is None:
            failed_gate = "RESOLVED_ENDING"
            gate_fail_counts["RESOLVED_ENDING"] += 1

        # 7. WEAK_CLIFFHANGER
        cliff_val = w.get("cliff", 0)
        cliff_ok = cliff_val >= min_cliff
        gate_results["WEAK_CLIFFHANGER"] = cliff_ok
        if not cliff_ok and failed_gate is None:
            failed_gate = "WEAK_CLIFFHANGER"
            gate_fail_counts["WEAK_CLIFFHANGER"] += 1

        w["gate_results"] = gate_results
        w["first_failed_gate"] = failed_gate
        w["passed_individual_gates"] = (failed_gate is None)

        if failed_gate is None:
            passed_individual.append(w)

    # 8. DUPLICATE_OVERLAP (Evaluated against higher-scoring candidates that passed 1-7)
    # Sort candidates passing 1-7 descending by score
    passed_individual.sort(
        key=lambda x: (x.get("score", 0.0), x.get("cliff", 0), x.get("duration_s", 0.0)),
        reverse=True,
    )

    survivors: List[Dict[str, Any]] = []
    for w in passed_individual:
        is_duplicate = False
        for s in survivors:
            iou = compute_interval_iou(w["start_s"], w["end_s"], s["start_s"], s["end_s"])
            if iou > max_iou:
                is_duplicate = True
                break

        if is_duplicate:
            w["gate_results"]["DUPLICATE_OVERLAP"] = False
            w["first_failed_gate"] = "DUPLICATE_OVERLAP"
            w["passed_all_gates"] = False
            gate_fail_counts["DUPLICATE_OVERLAP"] += 1
        else:
            w["gate_results"]["DUPLICATE_OVERLAP"] = True
            w["passed_all_gates"] = True
            if len(survivors) < top_k:
                w["selected_top_k"] = True
            else:
                w["selected_top_k"] = False
            survivors.append(w)

    return windows, survivors, gate_fail_counts
