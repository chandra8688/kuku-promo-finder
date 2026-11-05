"""Stage 4: Gates (src/gates.py).

Applies deterministic filters with soft narrative gates:
1. Hard structural gates (strictly reject):
   - DURATION: 20.0 <= duration <= 60.0 s
   - VALID_BEAT_IDS: All beat IDs belong to the episode
   - ORDERED_NONOVERLAPPING_SEGMENTS: Valid start < end, monotonic beats
   - CUT_STRADDLES_SPOKEN: Window cut does not split an external DIALOGUE/SFX beat
   - DUPLICATE_OVERLAP: Time-interval IoU > max_iou (default 0.5) against a higher-ranked survivor
   - DUPLICATE_ANCHOR: Shares the same first or last beat ID with a higher-ranked survivor
   - DUPLICATE_CONTAINED: Containment ratio > max_containment (default 0.6)

2. Soft narrative gates (advisory flags + score penalties via src.score, do not delete):
   - HOOK_TOO_WEAK: hook < min_hook (default 2)
   - RESOLVED_ENDING: resolved tail or resolved last
   - WEAK_CLIFFHANGER: cliff < min_cliff (default 2)
   Candidates failing multiple soft gates are retained with low_confidence=True.

Re-runnable on any candidate. Generates counts and reason codes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.features import window_features
from src.refine import refine_boundaries
from src.score import rank_key, score_window, soft_gate_flags
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
    refine: bool = True,
    refine_cfg: Optional[Dict[str, Any]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Evaluate structural hard gates and narrative soft gates across windows.

    Returns:
        (all_windows_with_gate_results, survivors, gate_fail_counts)
    """
    valid_ep_beat_ids = {b["beat_id"] for b in episode_beats}
    ep_beat_lookup = {b["beat_id"]: b for b in episode_beats}

    gate_fail_counts: Dict[str, int] = {gate: 0 for gate in GATE_ORDER}
    passed_individual: List[Dict[str, Any]] = []

    soft_gates_cfg = {"min_hook": float(min_hook), "min_cliff": float(min_cliff)}

    for w in windows:
        if "start_s_raw" not in w:
            w["start_s_raw"] = w.get("start_s", 0.0)
        if "end_s_raw" not in w:
            w["end_s_raw"] = w.get("end_s", 0.0)
        if "refine_ops" not in w:
            w["refine_ops"] = []
        if "score_refined" not in w:
            w["score_refined"] = w.get("score", 0.0)

        gate_results: Dict[str, bool] = {}
        failed_hard_gate: Optional[str] = None

        # 1. DURATION (Hard)
        duration_s = w.get("duration_s", 0.0)
        dur_ok = min_duration_s <= duration_s <= max_duration_s
        gate_results["DURATION"] = dur_ok
        if not dur_ok and failed_hard_gate is None:
            failed_hard_gate = "DURATION"
            gate_fail_counts["DURATION"] += 1

        # 2. VALID_BEAT_IDS (Hard)
        beat_ids = w.get("beat_ids", [])
        beats_valid = bool(beat_ids) and all(b_id in valid_ep_beat_ids for b_id in beat_ids)
        gate_results["VALID_BEAT_IDS"] = beats_valid
        if not beats_valid and failed_hard_gate is None:
            failed_hard_gate = "VALID_BEAT_IDS"
            gate_fail_counts["VALID_BEAT_IDS"] += 1

        # 3. ORDERED_NONOVERLAPPING_SEGMENTS (Hard)
        start_s = w.get("start_s", 0.0)
        end_s = w.get("end_s", 0.0)
        ordered_ok = start_s < end_s
        if beats_valid and ordered_ok:
            unit_starts = [ep_beat_lookup[b_id].get("start_s", 0.0) for b_id in beat_ids]
            if unit_starts != sorted(unit_starts):
                ordered_ok = False
        gate_results["ORDERED_NONOVERLAPPING_SEGMENTS"] = ordered_ok
        if not ordered_ok and failed_hard_gate is None:
            failed_hard_gate = "ORDERED_NONOVERLAPPING_SEGMENTS"
            gate_fail_counts["ORDERED_NONOVERLAPPING_SEGMENTS"] += 1

        # 4. CUT_STRADDLES_SPOKEN (Hard)
        straddles_spoken = w.get("cut_straddles_spoken", False)
        gate_results["CUT_STRADDLES_SPOKEN"] = not straddles_spoken
        if straddles_spoken and failed_hard_gate is None:
            failed_hard_gate = "CUT_STRADDLES_SPOKEN"
            gate_fail_counts["CUT_STRADDLES_SPOKEN"] += 1

        # 5-7. Soft Gates (Advisory flags, do not drop structural survivors)
        if "features" in w and isinstance(w["features"], dict):
            s_flags = soft_gate_flags(w["features"], soft_gates_cfg)
            hook_ok = not s_flags.get("WEAK_HOOK", False)
            unresolved_ok = not s_flags.get("RESOLVED_TAIL", False)
            cliff_ok = not s_flags.get("WEAK_CLIFFHANGER", False)
            low_conf = s_flags.get("low_confidence", False)
        else:
            hook_ok = float(w.get("hook", 0)) >= min_hook
            unresolved_ok = not bool(w.get("resolved_end", False))
            cliff_ok = float(w.get("cliff", 0)) >= min_cliff
            low_conf = (sum([not hook_ok, not unresolved_ok, not cliff_ok]) >= 2)
            s_flags = {
                "WEAK_HOOK": not hook_ok,
                "WEAK_CLIFFHANGER": not cliff_ok,
                "RESOLVED_TAIL": not unresolved_ok,
                "low_confidence": low_conf,
            }

        gate_results["HOOK_TOO_WEAK"] = hook_ok
        if not hook_ok:
            gate_fail_counts["HOOK_TOO_WEAK"] += 1

        gate_results["RESOLVED_ENDING"] = unresolved_ok
        if not unresolved_ok:
            gate_fail_counts["RESOLVED_ENDING"] += 1

        gate_results["WEAK_CLIFFHANGER"] = cliff_ok
        if not cliff_ok:
            gate_fail_counts["WEAK_CLIFFHANGER"] += 1

        w["gate_results"] = gate_results
        w["first_failed_gate"] = failed_hard_gate
        w["passed_individual_gates"] = (failed_hard_gate is None)
        w["soft_gate_flags"] = s_flags
        w["low_confidence"] = low_conf

        if failed_hard_gate is None:
            passed_individual.append(w)

    # 8. DUPLICATE_OVERLAP / DUPLICATE_ANCHOR / DUPLICATE_CONTAINED
    # Sorted by total rank key (low_confidence last, then score descending)
    passed_individual.sort(key=rank_key)

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
            w["selected_top_k"] = len(survivors) < top_k
            survivors.append(w)
            if first_beat:
                survivor_first_beats.add(first_beat)
            if last_beat:
                survivor_last_beats.add(last_beat)

    # Boundary refinement: strictly after selection and deduplication.
    # Note: survivors are NOT re-sorted afterwards to preserve ranking freeze.
    if refine and episode_beats:
        r_cfg = {"min_duration_s": min_duration_s, "max_duration_s": max_duration_s}
        if refine_cfg:
            r_cfg.update(refine_cfg)
        ep_end = max((b.get("end_s", 0.0) for b in episode_beats), default=0.0)
        for w in survivors:
            s_raw = w.get("start_s_raw", w["start_s"])
            e_raw = w.get("end_s_raw", w["end_s"])
            b_ids = w.get("beat_ids")
            new_s, new_e, ops = refine_boundaries(
                episode_beats, s_raw, e_raw, b_ids, r_cfg
            )
            w["start_s_raw"] = s_raw
            w["end_s_raw"] = e_raw
            w["start_s"] = new_s
            w["end_s"] = new_e
            w["duration_s"] = round(new_e - new_s, 2)
            w["refine_ops"] = ops

            if annotations:
                f = window_features(episode_beats, annotations, new_s, new_e, None, ep_end)
                sc, bd = score_window(f, weights)
                w["score_refined"] = sc
                w["breakdown_refined"] = bd
                w["features_refined"] = f
            else:
                w["score_refined"] = w.get("score", 0.0)

    return windows, survivors, gate_fail_counts
