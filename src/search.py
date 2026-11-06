"""Evaluation-facing wrappers over the production candidate engine, plus baselines.

Nothing in this file selects candidates on its own. See src/engine.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.score import rank_key


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
    """All structurally valid scored windows, from the production engine."""
    from src import engine

    wins = engine.enumerate(parsed, annotated, weights)
    return [w for w in wins if min_duration_s - 1e-6 <= w.get("duration_s", 0.0) <= max_duration_s + 1e-6]


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
    """Top-k picks exactly as run.py makes them, before the judge.

    This used to be a second implementation of enumeration, dedupe and
    refinement. It is now a wrapper over src/engine.py so evaluation and tuning
    measure the code that ships.
    """
    from src import engine

    _, _, surv, _ = engine.run_episode(parsed, annotated, weights=weights, top_k=top_k, refine=refine)
    return sorted(surv, key=rank_key)[:top_k]


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
