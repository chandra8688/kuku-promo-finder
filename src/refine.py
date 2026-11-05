"""Boundary refinement.

Beat-boundary windows inherit dead air at the head and clipped lines at the
tail. This trims the lead-in so the hook lands immediately, and moves the cut to
the end of the last spoken beat plus an acoustic pad. Every op is reversible and
recorded; duration limits are re-checked after each one and the op is dropped if
it would break them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULTS: Dict[str, Any] = {
    "min_duration_s": 20.0,
    "max_duration_s": 60.0,
    "tail_pad_s": 0.6,
    "max_lead_silence_s": 1.0,
    "max_trailing_dead_s": 1.5,
    "hook_kinds": ("DIALOGUE", "SFX", "ACTION"),
}


def _ok(start: float, end: float, c: Dict[str, Any]) -> bool:
    d = end - start
    return c["min_duration_s"] - 1e-6 <= d <= c["max_duration_s"] + 1e-6


def refine_boundaries(
    beats: Sequence[Dict[str, Any]],
    start_s: float,
    end_s: float,
    member_ids: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, List[str]]:
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
    win = sorted(win, key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))
    if not win:
        return round(start_s, 2), round(end_s, 2), ops

    hook_beats = [b for b in win if b.get("kind") in c["hook_kinds"]]
    if hook_beats:
        first = hook_beats[0].get("start_s", start_s)
        lead = first - start_s
        if lead > c["max_lead_silence_s"] and _ok(first, end_s, c):
            start_s = first
            ops.append(f"trim_lead_silence:{round(lead, 2)}s")

    spoken = [b for b in win if b.get("kind") == "DIALOGUE"]
    if spoken:
        last_end = max(b.get("end_s", 0.0) for b in spoken)
        target = last_end + c["tail_pad_s"]
        if target > end_s + 1e-6 and _ok(start_s, target, c):
            ops.append(f"extend_to_last_line:{round(target - end_s, 2)}s")
            end_s = target
        elif end_s - last_end > c["max_trailing_dead_s"] and _ok(start_s, target, c):
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
