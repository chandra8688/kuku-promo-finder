"""Boundary refinement.

Trims dead lead-in so the hook lands immediately, and places the cut just after
the last content beat (dialogue, SFX or action) plus an acoustic pad. Every op is
recorded, and is applied only if the result still passes the duration limits and
the spoken-straddle rule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.features import cue_score

DEFAULTS: Dict[str, Any] = {
    "min_duration_s": 20.0,
    "max_duration_s": 60.0,
    "tail_pad_s": 0.6,
    "max_lead_silence_s": 1.0,
    "max_trailing_dead_s": 1.5,
    "hook_kinds": ("DIALOGUE", "SFX", "ACTION"),
    "keep_visual_cue_min": 2,
}


def _ok(start: float, end: float, c: Dict[str, Any]) -> bool:
    d = end - start
    return c["min_duration_s"] - 1e-6 <= d <= c["max_duration_s"] + 1e-6


def _straddles(beats: Sequence[Dict[str, Any]], cut: float, keep: set, eps: float = 0.01) -> bool:
    """Same rule as the CUT_STRADDLES_SPOKEN hard gate: a cut may not slice an
    outside DIALOGUE or SFX beat in half."""
    for b in beats:
        if b.get("beat_id") in keep or b.get("kind") not in ("DIALOGUE", "SFX"):
            continue
        if b.get("start_s", 0.0) + eps < cut < b.get("end_s", 0.0) - eps:
            return True
    return False


def refine_boundaries(
    beats: Sequence[Dict[str, Any]],
    start_s: float,
    end_s: float,
    member_ids: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, List[str]]:
    """Polish boundaries without losing the hook or the cliffhanger.

    The end anchors on the last content beat of any kind (dialogue, SFX or
    action), not the last spoken line: a slap or a door after the final line is
    usually the cliffhanger. Every op is checked against the duration limits and
    the spoken-straddle rule before it is applied, so refinement can never turn
    a valid window into one the hard gates would reject.
    """
    c = dict(DEFAULTS)
    c.update(cfg or {})
    ops: List[str] = []

    if member_ids is not None:
        keep = set(member_ids)
        win = [b for b in beats if b.get("beat_id") in keep]
    else:
        win = [
            b
            for b in beats
            if b.get("end_s", 0.0) > start_s + 1e-9
            and b.get("start_s", 0.0) < end_s - 1e-9
            and b.get("kind") != "CAMERA"
        ]
        keep = {b.get("beat_id") for b in win}
    win = sorted(win, key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))
    if not win:
        return round(start_s, 2), round(end_s, 2), ops

    def valid(s: float, e: float) -> bool:
        return _ok(s, e, c) and not _straddles(beats, s, keep) and not _straddles(beats, e, keep)

    content = [b for b in win if b.get("kind") in c["hook_kinds"]]
    if content:
        first = content[0].get("start_s", start_s)
        lead = first - start_s
        visual_opener = any(
            b.get("kind") == "CAMERA"
            and b.get("end_s", 0.0) > start_s + 1e-9
            and b.get("start_s", 0.0) < first - 1e-9
            and cue_score(b.get("text") or "") >= c["keep_visual_cue_min"]
            for b in beats
        )
        if lead > c["max_lead_silence_s"] and not visual_opener and valid(first, end_s):
            start_s = first
            ops.append(f"trim_lead_silence:{round(lead, 2)}s")

        last_end = max(b.get("end_s", 0.0) for b in content)
        # The pad never runs past the episode's last timestamp: there is no
        # footage after it.
        episode_end = max((b.get("end_s", 0.0) for b in beats), default=last_end)
        target = min(last_end + c["tail_pad_s"], max(episode_end, last_end))
        if target > end_s + 1e-6 and valid(start_s, target):
            ops.append(f"extend_to_last_beat:{round(target - end_s, 2)}s")
            end_s = target
        elif end_s - last_end > c["max_trailing_dead_s"] and valid(start_s, target):
            ops.append(f"trim_trailing_dead:{round(end_s - target, 2)}s")
            end_s = target

    if not _ok(start_s, end_s, c):
        return round(start_s, 2), round(end_s, 2), ops + ["duration_violation_unrefined"]
    return round(start_s, 2), round(end_s, 2), ops


def refine_candidate(
    cand: Dict[str, Any],
    beats: Sequence[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    s, e, ops = refine_boundaries(
        beats, float(cand["start_s"]), float(cand["end_s"]), cand.get("beat_ids"), cfg
    )
    out = dict(cand)
    out["start_s_raw"], out["end_s_raw"] = cand["start_s"], cand["end_s"]
    out["start_s"], out["end_s"] = s, e
    out["duration_s"] = round(e - s, 2)
    out["refine_ops"] = ops
    return out
