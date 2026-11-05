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
8. DUPLICATE_OVERLAP / DUPLICATE_ANCHOR:
   - DUPLICATE_OVERLAP: Time-interval IoU > max_iou (default 0.5) against a higher-ranked survivor
   - DUPLICATE_ANCHOR: Shares the same first beat ID or same last beat ID as a higher-ranked survivor

Re-runnable on any candidate. Generates counts and reason codes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.windows import compute_containment_ratio, compute_interval_iou, load_scoring_config

GATE_ORDER = [
    "DURATION",
    "VALID_BEAT_IDS",
    "ORDERED_NONOVERLAPPING_SEGMENTS",
    "CUT_STRADDLES_SPOKEN",
    "HOOK_TOO_WEAK",
    "RESOLVED_ENDING",
    "WEAK_CLIFFHANGER",
    "DUPLICATE_OVERLAP",
    "DUPLICATE_ANCHOR",
    "DUPLICATE_CONTAINED",
]


def evaluate_window_gates(
    windows: List[Dict[str, Any]],
    episode_beats: List[Dict[str, Any]],
    min_hook: int = 2,
    min_cliff: int = 2,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
    max_iou: float = 0.5,
    max_containment: float = 0.6,
    top_k: int = 3,
    min_survivors: int = 0,
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

    for w in windows:
        w.setdefault("low_confidence", False)

    # 8. DUPLICATE_OVERLAP / DUPLICATE_ANCHOR / DUPLICATE_CONTAINED
    # Evaluated against higher-scoring candidates that passed gates 1-7.
    # A candidate is a duplicate if ANY of:
    #   (a) IoU > max_iou with a higher-ranked survivor  → reason: DUPLICATE_OVERLAP
    #   (b) Same first beat ID as any kept survivor       → reason: DUPLICATE_ANCHOR
    #   (c) Same last beat ID as any kept survivor        → reason: DUPLICATE_ANCHOR
    #   (d) Overlap / shorter_duration > max_containment  → reason: DUPLICATE_CONTAINED
    # Sort candidates passing 1-7 descending by score
    passed_individual.sort(
        key=lambda x: (x.get("score", 0.0), x.get("cliff", 0), x.get("duration_s", 0.0)),
        reverse=True,
    )

    survivors: List[Dict[str, Any]] = []
    survivor_first_beats: Set[str] = set()
    survivor_last_beats: Set[str] = set()

    for w in passed_individual:
        beat_ids_w = w.get("beat_ids", [])
        first_beat = beat_ids_w[0] if beat_ids_w else None
        last_beat = beat_ids_w[-1] if beat_ids_w else None

        # Check IoU overlap
        iou_dup = False
        for s in survivors:
            iou = compute_interval_iou(w["start_s"], w["end_s"], s["start_s"], s["end_s"])
            if iou > max_iou:
                iou_dup = True
                break

        # Check anchor overlap (shared first or last beat)
        anchor_dup = (
            (first_beat is not None and first_beat in survivor_first_beats)
            or (last_beat is not None and last_beat in survivor_last_beats)
        )

        # Check containment overlap (overlap / shorter_duration > 0.6)
        contained_dup = False
        if not iou_dup and not anchor_dup:
            for s in survivors:
                cont = compute_containment_ratio(w["start_s"], w["end_s"], s["start_s"], s["end_s"])
                if cont > max_containment:
                    contained_dup = True
                    break

        if iou_dup:
            w["gate_results"]["DUPLICATE_OVERLAP"] = False
            w["gate_results"]["DUPLICATE_ANCHOR"] = True
            w["gate_results"]["DUPLICATE_CONTAINED"] = True
            w["first_failed_gate"] = "DUPLICATE_OVERLAP"
            w["passed_all_gates"] = False
            gate_fail_counts["DUPLICATE_OVERLAP"] += 1
        elif anchor_dup:
            w["gate_results"]["DUPLICATE_OVERLAP"] = True
            w["gate_results"]["DUPLICATE_ANCHOR"] = False
            w["gate_results"]["DUPLICATE_CONTAINED"] = True
            w["first_failed_gate"] = "DUPLICATE_ANCHOR"
            w["passed_all_gates"] = False
            gate_fail_counts["DUPLICATE_ANCHOR"] += 1
        elif contained_dup:
            w["gate_results"]["DUPLICATE_OVERLAP"] = True
            w["gate_results"]["DUPLICATE_ANCHOR"] = True
            w["gate_results"]["DUPLICATE_CONTAINED"] = False
            w["first_failed_gate"] = "DUPLICATE_CONTAINED"
            w["passed_all_gates"] = False
            gate_fail_counts["DUPLICATE_CONTAINED"] += 1
        else:
            w["gate_results"]["DUPLICATE_OVERLAP"] = True
            w["gate_results"]["DUPLICATE_ANCHOR"] = True
            w["gate_results"]["DUPLICATE_CONTAINED"] = True
            w["passed_all_gates"] = True
            w["low_confidence"] = False
            w["selected_top_k"] = len(survivors) < top_k
            survivors.append(w)
            if first_beat:
                survivor_first_beats.add(first_beat)
            if last_beat:
                survivor_last_beats.add(last_beat)

    # 9. Fallback if fewer than min_survivors:
    # Add windows that fail ONLY HOOK_TOO_WEAK or WEAK_CLIFFHANGER, best S3a score first,
    # up to min_survivors, with low_confidence=True.
    # Never admit windows that fail structural (DURATION, VALID_BEAT_IDS, ORDERED_NONOVERLAPPING_SEGMENTS),
    # straddle (CUT_STRADDLES_SPOKEN), or resolved-ending (RESOLVED_ENDING) gates!
    if len(survivors) < min_survivors:
        eligible_fallback: List[Dict[str, Any]] = []
        for w in windows:
            if w in survivors:
                continue
            gr = w.get("gate_results", {})
            # Must pass all structural, straddle, and resolved ending gates
            structural_ok = (
                gr.get("DURATION", False)
                and gr.get("VALID_BEAT_IDS", False)
                and gr.get("ORDERED_NONOVERLAPPING_SEGMENTS", False)
                and gr.get("CUT_STRADDLES_SPOKEN", False)
                and gr.get("RESOLVED_ENDING", False)
            )
            if not structural_ok:
                continue

            # Must have failed ONLY HOOK_TOO_WEAK or WEAK_CLIFFHANGER (or both)
            failed_g = w.get("first_failed_gate")
            if failed_g in ("HOOK_TOO_WEAK", "WEAK_CLIFFHANGER"):
                eligible_fallback.append(w)

        eligible_fallback.sort(
            key=lambda x: (x.get("score", 0.0), x.get("cliff", 0), x.get("duration_s", 0.0)),
            reverse=True,
        )

        for cand in eligible_fallback:
            if len(survivors) >= min_survivors:
                break

            b_ids = cand.get("beat_ids", [])
            fb = b_ids[0] if b_ids else None
            lb = b_ids[-1] if b_ids else None

            # Dedupe check against existing survivors
            is_dup = False
            for s in survivors:
                if compute_interval_iou(cand["start_s"], cand["end_s"], s["start_s"], s["end_s"]) > max_iou:
                    is_dup = True
                    break
                if compute_containment_ratio(cand["start_s"], cand["end_s"], s["start_s"], s["end_s"]) > max_containment:
                    is_dup = True
                    break
            if is_dup:
                continue

            if (fb and fb in survivor_first_beats) or (lb and lb in survivor_last_beats):
                continue

            cand["passed_all_gates"] = True
            cand["low_confidence"] = True
            cand["is_fallback"] = True
            cand["selected_top_k"] = True
            cand["gate_results"]["DUPLICATE_OVERLAP"] = True
            cand["gate_results"]["DUPLICATE_ANCHOR"] = True
            cand["gate_results"]["DUPLICATE_CONTAINED"] = True
            survivors.append(cand)
            if fb:
                survivor_first_beats.add(fb)
            if lb:
                survivor_last_beats.add(lb)

    return windows, survivors, gate_fail_counts

