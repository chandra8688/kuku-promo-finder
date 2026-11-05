"""Reference window search wired to features + score + refine.

This is the shape src/windows.py should take after integration: enumerate,
feature-extract, hard-gate on structure only, score, dedupe, refine boundaries,
then re-score on the refined window. No model calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from .features import window_features
    from .refine import refine_boundaries
    from .score import hard_gate_ok, rank_key, score_window, soft_gate_flags
    from .evalkit import iou
except ImportError:
    from features import window_features
    from refine import refine_boundaries
    from score import hard_gate_ok, rank_key, score_window, soft_gate_flags
    from evalkit import iou


def straddle_flags(
    beats: Sequence[Dict[str, Any]], start_s: float, end_s: float, member_ids: Set[str]
) -> Tuple[bool, bool]:
    """True when a beat outside the window is cut in half by either boundary."""
    spoken = action = False
    for b in beats:
        if b.get("beat_id") in member_ids:
            continue
        bs, be = b.get("start_s", 0.0), b.get("end_s", 0.0)
        for cut in (start_s, end_s):
            if bs + 1e-9 < cut < be - 1e-9:
                if b.get("kind") == "DIALOGUE":
                    spoken = True
                elif b.get("kind") in ("ACTION", "SFX"):
                    action = True
    return spoken, action


def containment(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    small = min(a[1] - a[0], b[1] - b[0])
    return inter / small if small > 0 else 0.0


def enumerate_candidates(
    parsed: Dict[str, Any],
    annotated: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
) -> List[Dict[str, Any]]:
    beats = parsed.get("beats", [])
    ann = annotated.get("annotations", {}) if annotated else {}
    ep = parsed.get("episode", "unknown")
    ep_end = parsed.get("episode_end_s") or (max((b.get("end_s", 0.0) for b in beats), default=0.0))

    units = sorted(
        [b for b in beats if b.get("kind") != "CAMERA"],
        key=lambda b: (b.get("start_s", 0.0), b.get("end_s", 0.0)),
    )
    out: List[Dict[str, Any]] = []
    n = len(units)
    idx = 1
    for i in range(n):
        s = units[i].get("start_s", 0.0)
        cur = s
        for j in range(i, n):
            cur = max(cur, units[j].get("end_s", 0.0))
            d = cur - s
            if d < min_duration_s - 1e-6:
                continue
            if d > max_duration_s + 1e-6:
                if units[j].get("start_s", 0.0) - s > max_duration_s:
                    break
                continue
            ids = [u["beat_id"] for u in units[i : j + 1]]
            f = window_features(beats, ann, s, cur, ids, ep_end, cfg)
            f["cut_straddles_spoken"], f["cut_straddles_action"] = straddle_flags(beats, s, cur, set(ids))
            ok, reason = hard_gate_ok(f, min_duration_s, max_duration_s)
            if not ok:
                continue
            sc, bd = score_window(f, weights)
            flags = soft_gate_flags(f)
            out.append(
                {
                    "candidate_id": f"{ep}_v2w{idx:04d}",
                    "episode": ep,
                    "beat_ids": ids,
                    "start_s": round(s, 2),
                    "end_s": round(cur, 2),
                    "duration_s": round(d, 2),
                    "start_s_raw": round(s, 2),
                    "end_s_raw": round(cur, 2),
                    "refine_ops": [],
                    "score": sc,
                    "score_refined": sc,
                    "breakdown": bd,
                    "features": f,
                    "flags": flags,
                    "low_confidence": flags["low_confidence"],
                    "hook": f["hook"],
                    "cliff": f["cliff"],
                }
            )
            idx += 1
    return out


def select_diverse(
    cands: Sequence[Dict[str, Any]],
    top_k: int = 3,
    max_iou: float = 0.5,
    max_containment: float = 0.6,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for c in sorted(cands, key=rank_key):
        if len(kept) >= top_k:
            break
        a = (c["start_s"], c["end_s"])
        dup = False
        for k in kept:
            b = (k["start_s"], k["end_s"])
            if iou(*a, *b) > max_iou or containment(a, b) > max_containment:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def search_episode(
    parsed: Dict[str, Any],
    annotated: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    refine: bool = True,
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
) -> List[Dict[str, Any]]:
    beats = parsed.get("beats", [])
    ann = annotated.get("annotations", {}) if annotated else {}
    ep_end = parsed.get("episode_end_s") or (max((b.get("end_s", 0.0) for b in beats), default=0.0))
    cands = enumerate_candidates(parsed, annotated, weights, cfg, min_duration_s, max_duration_s)
    picked = select_diverse(cands, top_k)
    if not refine:
        return picked

    out = []
    for c in picked:
        s, e, ops = refine_boundaries(
            beats, c["start_s"], c["end_s"], c["beat_ids"], {"min_duration_s": min_duration_s, "max_duration_s": max_duration_s}
        )
        f = window_features(beats, ann, s, e, None, ep_end, cfg)
        f["cut_straddles_spoken"], f["cut_straddles_action"] = straddle_flags(
            beats, s, e, {b["beat_id"] for b in beats if b.get("end_s", 0) > s and b.get("start_s", 0) < e and b.get("kind") != "CAMERA"}
        )
        sc, bd = score_window(f, weights)
        flags = soft_gate_flags(f)
        d = dict(c)
        d.update(
            {
                "start_s_raw": c["start_s"],
                "end_s_raw": c["end_s"],
                "start_s": s,
                "end_s": e,
                "duration_s": round(e - s, 2),
                "refine_ops": ops,
                "score_refined": sc,
                "breakdown_refined": bd,
                "features_refined": f,
                "flags_refined": flags,
                "hook_refined": f["hook"],
                "cliff_refined": f["cliff"],
            }
        )
        out.append(d)
    # order is frozen before refinement on purpose: polishing a boundary must not
    # move a candidate up or down the list
    return out


def baseline_last_n(parsed: Dict[str, Any], n_s: float = 45.0) -> List[Dict[str, Any]]:
    beats = parsed.get("beats", [])
    end = parsed.get("episode_end_s") or max((b.get("end_s", 0.0) for b in beats), default=0.0)
    start = max(end - n_s, 0.0)
    return [
        {
            "candidate_id": f"{parsed.get('episode')}_b1",
            "episode": parsed.get("episode"),
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "duration_s": round(end - start, 2),
            "score": 0.0,
        }
    ]


def baseline_random(parsed: Dict[str, Any], seed: int = 0, k: int = 3, dur_lo: float = 20.0, dur_hi: float = 60.0) -> List[Dict[str, Any]]:
    import random

    rng = random.Random(seed)
    beats = parsed.get("beats", [])
    end = parsed.get("episode_end_s") or max((b.get("end_s", 0.0) for b in beats), default=0.0)
    out = []
    for i in range(k):
        d = rng.uniform(dur_lo, min(dur_hi, max(dur_lo, end)))
        s = rng.uniform(0.0, max(end - d, 0.0))
        out.append(
            {
                "candidate_id": f"{parsed.get('episode')}_r{seed}_{i}",
                "episode": parsed.get("episode"),
                "start_s": round(s, 2),
                "end_s": round(s + d, 2),
                "duration_s": round(d, 2),
                "score": 0.0,
            }
        )
    return out
