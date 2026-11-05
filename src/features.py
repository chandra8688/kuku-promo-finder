"""Window-level feature extraction for S3a candidates.

Replaces the single-beat reads in src/windows.py (hook from first beat, cliff and
resolved from the last beat only). Everything here is deterministic and needs no
model call. Shapes expected:

    beats       list of {beat_id, kind, speaker_norm, start_s, end_s, text}
                kind in {DIALOGUE, ACTION, SFX, CAMERA}
    annotations {beat_id: {h, q, r, n, t}}   missing beat -> treated as zeros
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

KIND_WEIGHT = {"DIALOGUE": 1.0, "SFX": 1.0, "ACTION": 0.9, "CAMERA": 0.6}

CUE_TIERS = (
    (3, r"\b(slap|slaps|gunshot|gun\s?shot|scream|screams|crash|smash|explosion|shatter|stab|stabs|slam|slams|choke|gasps?\s+loudly)\b"),
    (2, r"\b(gasp|shout|shouts|yell|yells|thud|siren|alarm|knock|knocks|whip\s?pan|snap\s?zoom|crash\s?zoom|shove|grabs?)\b"),
    (1, r"\b(door|footsteps|phone|rings?|buzz|zoom|push\s?in|close\s?up|rain|thunder)\b"),
)
_CUE_RE = [(s, re.compile(p, re.IGNORECASE)) for s, p in CUE_TIERS]

DEFAULTS: Dict[str, Any] = {
    "hook_window_s": 3.0,
    "tail_window_s": 6.0,
    "ctx_window_s": 5.0,
    "kind_weight": KIND_WEIGHT,
    "use_cues": True,
    "strong_hook_min": 2.0,
}


def cue_score(text: str) -> int:
    if not text:
        return 0
    for s, rx in _CUE_RE:
        if rx.search(text):
            return s
    return 0


def _overlaps(b: Dict[str, Any], lo: float, hi: float) -> bool:
    return b.get("end_s", 0.0) > lo + 1e-9 and b.get("start_s", 0.0) < hi - 1e-9


def _ann(annotations: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return annotations.get(b.get("beat_id")) or {}


def _num(d: Dict[str, Any], k: str) -> float:
    v = d.get(k, 0)
    return 0.0 if v is None else float(v)


def window_features(
    beats: Sequence[Dict[str, Any]],
    annotations: Dict[str, Any],
    start_s: float,
    end_s: float,
    member_ids: Optional[Sequence[str]] = None,
    episode_end_s: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    dur = max(end_s - start_s, 1e-6)
    kw = c["kind_weight"]

    inside = [b for b in beats if _overlaps(b, start_s, end_s)]
    if member_ids is None:
        members = [b for b in inside if b.get("kind") != "CAMERA"]
    else:
        keep = set(member_ids)
        members = [b for b in inside if b.get("beat_id") in keep]
    members = sorted(members, key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))
    if not members:
        members = sorted(inside, key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)))

    hook_hi = start_s + float(c["hook_window_s"])
    hook = 0.0
    hook_latency = float(c["hook_window_s"])
    hook_kind = None
    for b in sorted(inside, key=lambda b: b.get("start_s", 0.0)):
        if not _overlaps(b, start_s, hook_hi):
            continue
        h = _num(_ann(annotations, b), "h")
        if c["use_cues"] and b.get("kind") in ("SFX", "ACTION", "CAMERA"):
            h = max(h, float(cue_score(b.get("text") or "")))
        v = kw.get(b.get("kind"), 0.5) * h
        if v > hook:
            hook, hook_kind = v, b.get("kind")
        if v >= float(c["strong_hook_min"]):
            hook_latency = min(hook_latency, max(b.get("start_s", 0.0) - start_s, 0.0))

    tail_lo = end_s - float(c["tail_window_s"])
    tail = [b for b in members if _overlaps(b, tail_lo, end_s)] or members[-1:]
    cliff = max((_num(_ann(annotations, b), "q") for b in tail), default=0.0)
    resolved_tail = any(bool(_ann(annotations, b).get("r")) for b in tail)
    last_ann = _ann(annotations, members[-1]) if members else {}
    cliff_last = _num(last_ann, "q")
    resolved_last = bool(last_ann.get("r"))

    ctx_hi = start_s + float(c["ctx_window_s"])
    ctx_beats = [b for b in members if _overlaps(b, start_s, ctx_hi)] or members[:1]
    context = max((_num(_ann(annotations, b), "n") for b in ctx_beats), default=0.0)
    context_mean = (
        sum(_num(_ann(annotations, b), "n") for b in members) / len(members) if members else 0.0
    )

    acc = tot = peak = 0.0
    pts: List[tuple] = []
    for b in members:
        w = max(min(b.get("end_s", 0.0), end_s) - max(b.get("start_s", 0.0), start_s), 0.0)
        t = _num(_ann(annotations, b), "t")
        acc += w * t
        tot += w
        peak = max(peak, t)
        pts.append(((max(b.get("start_s", 0.0), start_s) + min(b.get("end_s", 0.0), end_s)) / 2.0, t))
    intensity = acc / tot if tot > 0 else 0.0
    third = dur / 3.0
    early = [t for m, t in pts if m < start_s + third]
    late = [t for m, t in pts if m >= end_s - third]
    slope = (sum(late) / len(late) if late else 0.0) - (sum(early) / len(early) if early else 0.0)

    spoken = [b for b in members if b.get("kind") == "DIALOGUE"]
    spoken_dur = sum(
        max(min(b.get("end_s", 0.0), end_s) - max(b.get("start_s", 0.0), start_s), 0.0) for b in spoken
    )
    speakers = {b.get("speaker_norm") for b in spoken if b.get("speaker_norm")}

    return {
        "duration_s": round(dur, 2),
        "hook": round(hook, 3),
        "hook_kind": hook_kind,
        "hook_latency_s": round(hook_latency, 2),
        "cliff": round(cliff, 3),
        "cliff_last": round(cliff_last, 3),
        "resolved_tail": resolved_tail,
        "resolved_last": resolved_last,
        "context": round(context, 3),
        "context_mean": round(context_mean, 3),
        "intensity": round(intensity, 3),
        "peak_intensity": round(peak, 3),
        "intensity_slope": round(slope, 3),
        "dialogue_share": round(spoken_dur / dur, 3),
        "speaker_count": len(speakers),
        "sfx_count": sum(1 for b in members if b.get("kind") == "SFX"),
        "beat_count": len(members),
        "tail_pos": round(min(end_s / episode_end_s, 1.0), 3) if episode_end_s else 0.0,
        "has_tail_pos": episode_end_s is not None,
    }
