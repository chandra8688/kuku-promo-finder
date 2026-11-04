"""Stage 3a: Window Search (src/windows.py).

Enumerates consecutive non-camera beat windows (units) over 20.0s <= duration <= 60.0s.
Evaluates cut validity (straddle rule for DIALOGUE/SFX vs ACTION).
Computes narrative features from S2 annotations:
  - hook (h of first beat)
  - cliff (q of last beat)
  - resolved_end (r of last beat)
  - context (n of first beat)
  - intensity (mean t over window)
Computes score = hook + cliff + intensity - context from config/scoring.yaml.
Greedily ranks and dedupes top K candidates with time-interval IoU <= 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from src.reserved import is_episode_blocked


def load_scoring_config(config_path: str | Path = "config/scoring.yaml") -> Dict[str, Any]:
    """Load weights and threshold parameters from YAML config."""
    p = Path(config_path)
    if not p.exists():
        return {
            "weights": {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0},
            "thresholds": {
                "min_duration_s": 20.0,
                "max_duration_s": 60.0,
                "min_hook": 2,
                "min_cliff": 2,
                "max_iou_overlap": 0.5,
                "top_k": 3,
            },
        }
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_interval_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Calculate intersection-over-union (IoU) between two 1D time intervals."""
    intersection = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return (intersection / union) if union > 0.0 else 0.0


def check_straddle_cuts(
    start_cut: float,
    end_cut: float,
    all_beats: List[Dict[str, Any]],
    window_beat_ids: Set[str],
    eps: float = 0.01,
) -> Tuple[bool, bool]:
    """Check if beats outside the window straddle the start or end cut.

    Returns:
        (cut_straddles_spoken, cut_straddles_action)
    """
    cut_straddles_spoken = False
    cut_straddles_action = False

    for b in all_beats:
        if b["beat_id"] in window_beat_ids:
            continue

        b_start = b.get("start_s", 0.0)
        b_end = b.get("end_s", 0.0)
        kind = b.get("kind", "")

        # Check start cut straddle: b_start < start_cut < b_end
        straddles_start = (b_start + eps < start_cut) and (start_cut + eps < b_end)
        # Check end cut straddle: b_start < end_cut < b_end
        straddles_end = (b_start + eps < end_cut) and (end_cut + eps < b_end)

        if straddles_start or straddles_end:
            if kind in ("DIALOGUE", "SFX"):
                cut_straddles_spoken = True
            elif kind == "ACTION":
                cut_straddles_action = True

    return cut_straddles_spoken, cut_straddles_action


def enumerate_windows_for_episode(
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
) -> List[Dict[str, Any]]:
    """Enumerate all valid-duration windows of consecutive non-camera beats.

    Extracts S2 annotations and computes features and score.
    """
    if weights is None:
        cfg = load_scoring_config()
        weights = cfg.get("weights", {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0})

    w_hook = weights.get("hook", 1.0)
    w_cliff = weights.get("cliff", 1.0)
    w_intensity = weights.get("intensity", 1.0)
    w_context = weights.get("context", 1.0)

    ep_id = parsed_data.get("episode", "unknown")
    all_beats = parsed_data.get("beats", [])
    annotations = annotated_data.get("annotations", {})

    # C1: Non-camera beats sorted by (start_s, end_s). Camera beats are never window members.
    non_cam_units = [b for b in all_beats if b.get("kind") != "CAMERA"]
    non_cam_units.sort(key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))

    n_units = len(non_cam_units)
    windows: List[Dict[str, Any]] = []
    window_idx = 1

    # C2: Enumerate over consecutive units i..j
    for i in range(n_units):
        start_s = non_cam_units[i].get("start_s", 0.0)
        curr_max_end = start_s

        for j in range(i, n_units):
            beat_j_end = non_cam_units[j].get("end_s", 0.0)
            if beat_j_end > curr_max_end:
                curr_max_end = beat_j_end

            duration_s = curr_max_end - start_s

            # Early break if duration exceeds max_duration_s
            # (since end_s generally increases, though we clamp strictly by duration filter)
            if duration_s < min_duration_s:
                continue
            if duration_s > max_duration_s:
                # If beat starts after max duration window, later beats will also exceed
                if non_cam_units[j].get("start_s", 0.0) - start_s > max_duration_s:
                    break
                continue

            window_units = non_cam_units[i : j + 1]
            window_beat_ids = {u["beat_id"] for u in window_units}

            # C3: Cut validity (straddle rule)
            straddles_spoken, straddles_action = check_straddle_cuts(
                start_cut=start_s,
                end_cut=curr_max_end,
                all_beats=all_beats,
                window_beat_ids=window_beat_ids,
            )

            # C4: Features from S2 annotations
            first_b_id = window_units[0]["beat_id"]
            last_b_id = window_units[-1]["beat_id"]

            first_ann = annotations.get(first_b_id) or {}
            last_ann = annotations.get(last_b_id) or {}

            hook = first_ann.get("h", 0) or 0
            cliff = last_ann.get("q", 0) or 0
            resolved_end = bool(last_ann.get("r", False))
            context = first_ann.get("n", 0) or 0

            # Mean intensity over all beats in window
            intensities = [
                (annotations.get(u["beat_id"]) or {}).get("t", 0) or 0
                for u in window_units
            ]
            intensity = sum(intensities) / len(intensities) if intensities else 0.0

            # C5: Score calculation
            score = (w_hook * hook) + (w_cliff * cliff) + (w_intensity * intensity) - (w_context * context)

            candidate = {
                "candidate_id": f"{ep_id}_w{window_idx:04d}",
                "episode": ep_id,
                "beat_ids": [u["beat_id"] for u in window_units],
                "start_s": round(start_s, 2),
                "end_s": round(curr_max_end, 2),
                "duration_s": round(duration_s, 2),
                "hook": hook,
                "cliff": cliff,
                "resolved_end": resolved_end,
                "context": context,
                "intensity": round(intensity, 3),
                "score": round(score, 3),
                "cut_straddles_spoken": straddles_spoken,
                "cut_straddles_action": straddles_action,
                "gate_results": {},
                "passed_all_gates": False,
            }
            windows.append(candidate)
            window_idx += 1

    return windows


def deduplicate_windows_greedy(
    windows: List[Dict[str, Any]],
    top_k: int = 3,
    max_iou: float = 0.5,
) -> List[Dict[str, Any]]:
    """C7: Greedily select top K windows by score with time-interval IoU <= max_iou."""
    # Sort descending by score, tie-break by duration or cliff
    sorted_windows = sorted(
        windows,
        key=lambda w: (w["score"], w["cliff"], w["duration_s"]),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    for w in sorted_windows:
        if len(kept) >= top_k:
            break
        # Check IoU against all already kept
        is_dup = False
        for k in kept:
            iou = compute_interval_iou(w["start_s"], w["end_s"], k["start_s"], k["end_s"])
            if iou > max_iou:
                is_dup = True
                break
        if not is_dup:
            kept.append(w)

    return kept
