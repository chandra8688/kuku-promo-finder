"""Transparent scoring for promo windows.

score = sum(weight_i * term_i). Every term is returned in the breakdown so a
candidate can carry its own provenance. Nothing here calls a model.

Soft gates replace the hard HOOK_TOO_WEAK / WEAK_CLIFFHANGER / RESOLVED_ENDING
rejects: they subtract from the score and set flags instead of deleting the
candidate. Keep only structural checks hard.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DEFAULT_WEIGHTS: Dict[str, float] = {
    "hook": 1.4,
    "hook_latency": -0.25,
    "cliff": 2.0,
    "intensity": 0.8,
    "peak_intensity": 0.3,
    "intensity_slope": 0.4,
    "context": -1.0,
    "tail_pos": 0.6,
    "resolved_tail": -2.0,
    "resolved_last": -1.5,
    "straddle_spoken": -0.75,
    "straddle_action": -0.2,
    "speaker_overload": -0.15,
    "silence": -0.5,
}

SOFT_GATES: Dict[str, float] = {
    "min_hook": 2.0,
    "min_cliff": 2.0,
    "max_context": 2.0,
    "min_dialogue_share": 0.15,
    "max_speakers": 4,
    "max_gap_s": 5.0,
}


def score_terms(f: Dict[str, Any], gates: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    g = dict(SOFT_GATES)
    g.update(gates or {})
    return {
        "hook": float(f.get("hook", 0.0)),
        "hook_latency": float(f.get("hook_latency_s", 0.0)),
        "cliff": float(f.get("cliff", 0.0)),
        "intensity": float(f.get("intensity", 0.0)),
        "peak_intensity": float(f.get("peak_intensity", 0.0)),
        "intensity_slope": float(f.get("intensity_slope", 0.0)),
        "context": float(f.get("context", 0.0)),
        "tail_pos": float(f.get("tail_pos", 0.0)),
        "resolved_tail": 1.0 if f.get("resolved_tail") else 0.0,
        "resolved_last": 1.0 if f.get("resolved_last") else 0.0,
        "straddle_spoken": 1.0 if f.get("cut_straddles_spoken") else 0.0,
        "straddle_action": 1.0 if f.get("cut_straddles_action") else 0.0,
        "speaker_overload": max(float(f.get("speaker_count", 0)) - g["max_speakers"], 0.0),
        "silence": max(g["min_dialogue_share"] - float(f.get("dialogue_share", 0.0)), 0.0),
    }


def score_window(
    f: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    gates: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    terms = score_terms(f, gates)
    contrib = {k: round(w.get(k, 0.0) * v, 4) for k, v in terms.items()}
    total = round(sum(contrib.values()), 4)
    breakdown = {
        "total": total,
        "terms": {k: round(v, 4) for k, v in terms.items()},
        "contrib": contrib,
        "weights": {k: w.get(k, 0.0) for k in terms},
    }
    return total, breakdown


def soft_gate_flags(f: Dict[str, Any], gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Advisory flags. None of these drop a candidate on their own.

    low_confidence fires on two or more narrative weaknesses, or on any beat the
    annotator never scored. CROSSES_SCENE and LARGE_GAP are recorded for the
    reviewer but deliberately do not move the ranking until they are measured.
    """
    g = dict(SOFT_GATES)
    g.update(gates or {})
    narrative = {
        "WEAK_HOOK": float(f.get("hook", 0.0)) < g["min_hook"],
        "WEAK_CLIFFHANGER": float(f.get("cliff", 0.0)) < g["min_cliff"],
        "HIGH_CONTEXT_NEED": float(f.get("context", 0.0)) > g["max_context"],
        "RESOLVED_TAIL": bool(f.get("resolved_tail")),
        "MOSTLY_SILENT": float(f.get("dialogue_share", 0.0)) < g["min_dialogue_share"],
    }
    flags: Dict[str, Any] = dict(narrative)
    flags["MISSING_ANNOTATIONS"] = int(f.get("ann_missing", 0)) > 0
    flags["CROSSES_SCENE"] = int(f.get("scene_count", 0)) > 1
    flags["LARGE_GAP"] = float(f.get("max_gap_s", 0.0)) > g.get("max_gap_s", 5.0)
    flags["soft_fail_count"] = sum(1 for v in narrative.values() if v)
    flags["low_confidence"] = flags["soft_fail_count"] >= 2 or flags["MISSING_ANNOTATIONS"]
    return flags


def hard_gate_ok(
    f: Dict[str, Any],
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
) -> Tuple[bool, Optional[str]]:
    """Structural only. Narrative quality belongs in the score, not here."""
    d = float(f.get("duration_s", 0.0))
    if d < min_duration_s - 1e-6:
        return False, "DURATION_SHORT"
    if d > max_duration_s + 1e-6:
        return False, "DURATION_LONG"
    if int(f.get("beat_count", 0)) < 1:
        return False, "EMPTY_WINDOW"
    if f.get("cut_straddles_spoken"):
        return False, "CUT_STRADDLES_SPOKEN"
    return True, None


def rank_key(cand: Dict[str, Any]) -> tuple:
    """Deterministic ordering. Never leave a tie to dict/float ordering."""
    return (
        0 if not cand.get("low_confidence") else 1,
        -float(cand.get("score", 0.0)),
        -float(cand.get("cliff", 0.0)),
        -float(cand.get("hook", 0.0)),
        float(cand.get("start_s", 0.0)),
        str(cand.get("candidate_id", "")),
    )
