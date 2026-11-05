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

from src.features import window_features
from src.reserved import is_episode_blocked
from src.score import DEFAULT_WEIGHTS, hard_gate_ok, score_window, soft_gate_flags


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
                "hook_window_s": 3.0,
                "min_survivors": 2,
                "max_containment_overlap": 0.6,
            },
        }
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_interval_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Calculate intersection-over-union (IoU) between two 1D time intervals."""
    intersection = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return (intersection / union) if union > 0.0 else 0.0


def compute_containment_ratio(s1: float, e1: float, s2: float, e2: float) -> float:
    """Calculate overlap / min(duration1, duration2)."""
    dur1 = max(0.0, e1 - s1)
    dur2 = max(0.0, e2 - s2)
    shorter = min(dur1, dur2)
    if shorter <= 0.0:
        return 0.0
    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    return overlap / shorter


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


def _v1_features(
    window_units: List[Dict[str, Any]],
    annotations: Dict[str, Any],
    start_s: float,
    hook_window_s: float,
    weights: Optional[Dict[str, float]],
) -> Tuple[int, int, bool, int, float, float]:
    w = weights or {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}
    first_b_id = window_units[0]["beat_id"]
    last_b_id = window_units[-1]["beat_id"]

    first_ann = annotations.get(first_b_id) or {}
    last_ann = annotations.get(last_b_id) or {}

    hook_beats = [
        u for u in window_units
        if u.get("start_s", 0.0) <= start_s + hook_window_s + 1e-4
    ]
    hook = max(
        [(annotations.get(u["beat_id"]) or {}).get("h", 0) or 0 for u in hook_beats],
        default=0,
    )
    cliff = int(last_ann.get("q", 0) or 0)
    resolved_end = bool(last_ann.get("r", False))
    context = int(first_ann.get("n", 0) or 0)

    intensities = [
        (annotations.get(u["beat_id"]) or {}).get("t", 0) or 0
        for u in window_units
    ]
    intensity = sum(intensities) / len(intensities) if intensities else 0.0

    w_hook = w.get("hook", 1.0)
    w_cliff = w.get("cliff", 1.0)
    w_intensity = w.get("intensity", 1.0)
    w_context = w.get("context", 1.0)
    score = (w_hook * hook) + (w_cliff * cliff) + (w_intensity * intensity) - (w_context * context)

    return hook, cliff, resolved_end, context, intensity, score


def enumerate_windows_for_episode(
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
    hook_window_s: float = 3.0,
    score_version: str = "v2",
) -> List[Dict[str, Any]]:
    """Enumerate all valid-duration windows of consecutive non-camera beats.

    Extracts S2 annotations and computes features and score.
    Hook = max S2 hook score over non-camera beats starting within hook_window_s of window start.
    """
    ep_id = parsed_data.get("episode", "unknown")
    all_beats = parsed_data.get("beats", [])
    episode_end_s = parsed_data.get("episode_end_s") or max((b.get("end_s", 0.0) for b in all_beats), default=0.0)

    if "annotations" in annotated_data and isinstance(annotated_data["annotations"], dict):
        annotations = annotated_data["annotations"]
    elif "beats" in annotated_data and isinstance(annotated_data["beats"], list):
        annotations = {b["beat_id"]: b for b in annotated_data["beats"]}
    else:
        annotations = annotated_data

    cfg_v2 = None
    if score_version == "v2":
        p_v2 = Path("config/scoring_v2.yaml")
        if p_v2.exists():
            with open(p_v2, "r", encoding="utf-8") as f:
                cfg_v2 = yaml.safe_load(f) or {}
    feat_cfg = dict(cfg_v2.get("features", {})) if cfg_v2 else {}
    if hook_window_s is not None:
        feat_cfg["hook_window_s"] = hook_window_s
    weights_v2 = weights if weights is not None else (cfg_v2.get("weights") if cfg_v2 else None)

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
            if duration_s < min_duration_s:
                continue
            if duration_s > max_duration_s:
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
            ids = [u["beat_id"] for u in window_units]
            if score_version == "v1":
                hook, cliff, resolved_end, context, intensity, score = _v1_features(
                    window_units, annotations, start_s, hook_window_s, weights
                )
                feats, breakdown = None, None
            else:
                feats = window_features(
                    all_beats, annotations, start_s, curr_max_end, ids, episode_end_s, feat_cfg
                )
                feats["cut_straddles_spoken"] = straddles_spoken
                feats["cut_straddles_action"] = straddles_action
                score, breakdown = score_window(feats, weights_v2)
                hook = int(feats["hook"])
                cliff = int(feats["cliff"])
                resolved_end = bool(feats["resolved_tail"] or feats["resolved_last"])
                context = int(feats["context"])
                intensity = float(feats["intensity"])

            candidate = {
                "candidate_id": f"{ep_id}_w{window_idx:04d}",
                "episode": ep_id,
                "beat_ids": ids,
                "start_s": round(start_s, 2),
                "end_s": round(curr_max_end, 2),
                "duration_s": round(duration_s, 2),
                "start_s_raw": round(start_s, 2),
                "end_s_raw": round(curr_max_end, 2),
                "refine_ops": [],
                "hook": hook,
                "cliff": cliff,
                "resolved_end": resolved_end,
                "context": context,
                "intensity": round(intensity, 3),
                "score": round(score, 3),
                "score_refined": round(score, 3),
                "cut_straddles_spoken": straddles_spoken,
                "cut_straddles_action": straddles_action,
                "features": feats,
                "breakdown": breakdown,
                "gate_results": {},
                "passed_all_gates": False,
                "low_confidence": False,
            }
            windows.append(candidate)
            window_idx += 1

    return windows


def deduplicate_windows_greedy(
    windows: List[Dict[str, Any]],
    top_k: int = 3,
    max_iou: float = 0.5,
    max_containment: float = 0.6,
    beats: Optional[List[Dict[str, Any]]] = None,
    refine: bool = False,
    refine_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """C7: Greedily select top K windows by score with IoU <= max_iou and containment <= max_containment."""
    # Sort descending by score, tie-break by duration or cliff
    sorted_windows = sorted(
        windows,
        key=lambda w: (w["score"], w["cliff"], w["duration_s"]),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    kept_first_beats: Set[str] = set()
    kept_last_beats: Set[str] = set()

    for w in sorted_windows:
        if len(kept) >= top_k:
            break
        b_ids = w.get("beat_ids", [])
        fb = b_ids[0] if b_ids else None
        lb = b_ids[-1] if b_ids else None

        # Check IoU and containment
        is_dup = False
        for k in kept:
            iou = compute_interval_iou(w["start_s"], w["end_s"], k["start_s"], k["end_s"])
            if iou > max_iou:
                is_dup = True
                break
            cont = compute_containment_ratio(w["start_s"], w["end_s"], k["start_s"], k["end_s"])
            if cont > max_containment:
                is_dup = True
                break

        if is_dup:
            continue

        if (fb and fb in kept_first_beats) or (lb and lb in kept_last_beats):
            continue

        kept.append(w)
        if fb:
            kept_first_beats.add(fb)
        if lb:
            kept_last_beats.add(lb)

    if refine and beats:
        from src.refine import refine_boundaries

        r_cfg = {"min_duration_s": 20.0, "max_duration_s": 60.0}
        if refine_cfg:
            r_cfg.update(refine_cfg)
        for w in kept:
            s_raw = w.get("start_s_raw", w["start_s"])
            e_raw = w.get("end_s_raw", w["end_s"])
            b_ids = w.get("beat_ids")
            new_s, new_e, ops = refine_boundaries(beats, s_raw, e_raw, b_ids, r_cfg)
            w["start_s_raw"] = s_raw
            w["end_s_raw"] = e_raw
            w["start_s"] = new_s
            w["end_s"] = new_e
            w["duration_s"] = round(new_e - new_s, 2)
            w["refine_ops"] = ops
            w.setdefault("score_refined", w.get("score", 0.0))

    return kept

