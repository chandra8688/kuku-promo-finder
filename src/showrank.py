"""Show-level ranking: which promos to post first, and why.

S3a (src/score.py) picks the best windows inside each episode. Across the
show it saturates: most finalists score cliff 3 and peak intensity 3, so its
order among them is driven by position-in-episode, not promo quality. This
module ranks the per-episode finalists with a second, transparent score:

  1. Consensus gate: annotator AND judge both rate hook >= 2 and cliff >= 2
     (0-3 scales), and the clip does not resolve itself.
  2. Post score: agreed hook + agreed cliff + clarity + payoff + funnel bonus.
  3. At most N promos per episode; wave 1 spread across hook types.
  4. A plain-language rank_reason for every promo, built from its numbers.

Pure functions, no model calls. Config: config/show_rank.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "show_rank.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def episode_number(ep: str) -> int:
    m = re.findall(r"\d+", str(ep))
    return int(m[0]) if m else 0


def funnel_bonus(ep: str, cfg: Dict[str, Any]) -> float:
    """1.0 for early episodes, falling linearly to 0.0."""
    f = cfg.get("funnel", {})
    full = int(f.get("full_until_episode", 10))
    zero = int(f.get("zero_from_episode", 30))
    n = episode_number(ep)
    if n <= full:
        return 1.0
    if n >= zero:
        return 0.0
    return round((zero - n) / float(zero - full), 3)


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def agreed(promo: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Consensus hook and cliff: the lower of the two models' scores.

    A promo is only as strong as the less convinced model says. When the judge
    did not score it (no-LLM run, judge failure), the annotator score is used
    and the promo is marked single_model.
    """
    s = promo.get("scores", {})
    a = s.get("annotator", {}) or {}
    ah, ac = _f(a.get("hook")), _f(a.get("cliff"))
    jh, jc = _f(s.get("hook")), _f(s.get("cliff"))
    judged = bool(s.get("judged"))
    if judged and jh is not None and jc is not None:
        hook = jh if ah is None else min(ah, jh)
        cliff = jc if ac is None else min(ac, jc)
    else:
        hook, cliff = ah, ac
    return {"hook": hook, "cliff": cliff, "single_model": not judged}


def gate(promo: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    """Return the reason a promo cannot lead the show list, or None."""
    g = cfg.get("gates", {})
    ag = agreed(promo)
    if promo.get("low_confidence"):
        return "low_confidence"
    if promo.get("scores", {}).get("resolved_after"):
        return "resolves_itself"
    if (promo.get("scores", {}).get("annotator") or {}).get("resolved_last"):
        return "last_beat_resolves"
    if ag["hook"] is None or ag["hook"] < float(g.get("min_hook", 2.0)):
        return "hook_not_agreed"
    if ag["cliff"] is None or ag["cliff"] < float(g.get("min_cliff", 2.0)):
        return "cliffhanger_not_agreed"
    return None


def post_score(promo: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    w = cfg.get("weights", {})
    s = promo.get("scores", {})
    ag = agreed(promo)
    clarity = _f(s.get("clarity")) or 0.0
    payoff = _f(s.get("payoff")) or 0.0
    return round(
        float(w.get("hook", 1.0)) * (ag["hook"] or 0.0)
        + float(w.get("cliff", 1.0)) * (ag["cliff"] or 0.0)
        + float(w.get("clarity", 0.75)) * clarity
        + float(w.get("payoff", 0.5)) * payoff
        + float(w.get("funnel", 0.9)) * funnel_bonus(promo["episode"], cfg),
        3,
    )


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "?"
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def rank_reason(promo: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    s = promo.get("scores", {})
    ag = agreed(promo)
    who = "annotator only" if ag["single_model"] else "both models agree"
    parts = [
        f"hook {_fmt(ag['hook'])}/3 ({who})",
        f"cliffhanger {_fmt(ag['cliff'])}/3",
    ]
    if not ag["single_model"]:
        parts.append(f"clear to a new viewer {_fmt(_f(s.get('clarity')))}/3")
    n = episode_number(promo["episode"])
    fb = funnel_bonus(promo["episode"], cfg)
    if fb >= 1.0:
        parts.append(f"episode {n}: leads straight into the first episodes")
    else:
        parts.append(f"episode {n}")
    if promo.get("hook_type"):
        parts.append(f"hook type: {promo['hook_type'].replace('_', ' ')}")
    reason = gate(promo, cfg)
    if reason:
        parts.append(f"below the bar: {reason.replace('_', ' ')}")
    return " · ".join(parts)


def _spread_wave1(ranked: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Greedy re-order of the first wave so it covers several hook types.

    Only swaps in a promo whose post score is within max_score_gap of the one
    it displaces, so quality is never traded for variety by much.
    """
    w = cfg.get("wave1", {})
    size, need, gap = int(w.get("size", 5)), int(w.get("min_hook_types", 3)), float(w.get("max_score_gap", 1.0))
    if not all(p.get("hook_type") for p in ranked[: size * 3]):
        return ranked
    pool = list(ranked)
    wave: List[Dict[str, Any]] = []
    while pool and len(wave) < size:
        types = {p["hook_type"] for p in wave}
        slots_left = size - len(wave)
        must_diversify = len(types) + slots_left <= need and len(types) < need
        pick = pool[0]
        if must_diversify and pick["hook_type"] in types:
            for cand in pool[1:]:
                if cand["hook_type"] not in types and pick["post_score"] - cand["post_score"] <= gap:
                    pick = cand
                    break
        wave.append(pick)
        pool.remove(pick)
    return wave + pool


def rank_show(
    promos: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    finalist_info: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Rank per-episode promos across the show. Returns the top_n list."""
    cfg = cfg or load_config()
    top_n = int(cfg.get("top_n", 15))
    cap = int(cfg.get("max_per_episode", 2))
    finalist_info = finalist_info or {}

    scored = []
    for p in promos:
        q = dict(p)
        info = finalist_info.get(q.get("candidate_id"), {})
        if info:
            q["hook_type"] = info.get("hook_type")
            q["why_it_works"] = info.get("why_it_works")
        ag = agreed(q)
        q["agreed_hook"], q["agreed_cliff"] = ag["hook"], ag["cliff"]
        q["post_score"] = post_score(q, cfg)
        q["funnel_bonus"] = funnel_bonus(q["episode"], cfg)
        q["below_bar"] = gate(q, cfg)
        scored.append(q)

    # Promos that pass the consensus gate first, then by post score, then S3a.
    scored.sort(
        key=lambda q: (
            q["below_bar"] is None,
            q["post_score"],
            float(q.get("scores", {}).get("s3a_score", 0.0)),
            -episode_number(q["episode"]),
        ),
        reverse=True,
    )

    picked: List[Dict[str, Any]] = []
    per_ep: Dict[str, int] = {}
    for q in scored:
        if per_ep.get(q["episode"], 0) >= cap:
            continue
        picked.append(q)
        per_ep[q["episode"]] = per_ep.get(q["episode"], 0) + 1
        if len(picked) == top_n:
            break

    passing = [q for q in picked if q["below_bar"] is None]
    rest = [q for q in picked if q["below_bar"] is not None]
    ordered = _spread_wave1(passing, cfg) + rest

    wave_size = int(cfg.get("wave1", {}).get("size", 5))
    for i, q in enumerate(ordered, start=1):
        q["show_rank"] = i
        q["wave"] = 1 if i <= wave_size and q["below_bar"] is None else 2
        q["rank_reason"] = rank_reason(q, cfg)
        q["why"] = q.get("why_it_works") or q["rank_reason"]
    return ordered
