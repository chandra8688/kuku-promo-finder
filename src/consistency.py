"""Self-consistency for S2 annotations and run-to-run stability.

Annotating an episode k times and taking the median per beat removes most of the
single-sample jitter that made the earlier stability numbers unusable. Beats
where the samples disagree get flagged so the disagreement travels with the
candidate instead of silently moving a rank.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .evalkit import topk_overlap
except ImportError:
    from evalkit import topk_overlap

NUMERIC = ("h", "q", "n", "t")


def aggregate_annotations(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """samples: list of {beat_id: {h,q,r,n,t}}. Median for numbers, majority for r."""
    if not samples:
        return {}
    if len(samples) == 1:
        return dict(samples[0])
    beat_ids = sorted({bid for s in samples for bid in s})
    out: Dict[str, Any] = {}
    for bid in beat_ids:
        rows = [s[bid] for s in samples if bid in s and s[bid]]
        if not rows:
            continue
        agg: Dict[str, Any] = {}
        spread = 0.0
        for k in NUMERIC:
            vals = [float(r.get(k) or 0) for r in rows]
            agg[k] = int(round(statistics.median(vals)))
            if len(vals) > 1:
                spread = max(spread, statistics.pstdev(vals))
        r_votes = [1 if r.get("r") else 0 for r in rows]
        agg["r"] = sum(r_votes) * 2 > len(r_votes)
        agg["n_samples"] = len(rows)
        agg["disagreement"] = round(spread, 3)
        agg["unstable"] = spread >= 0.75 or 0 < sum(r_votes) < len(r_votes)
        out[bid] = agg
    return out


def annotation_agreement(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) < 2:
        return {"n_samples": len(samples), "comparable_beats": 0}
    common = set(samples[0])
    for s in samples[1:]:
        common &= set(s)
    common = sorted(common)
    per_field = {}
    for k in NUMERIC:
        exact = 0
        for bid in common:
            vals = {int(s[bid].get(k) or 0) for s in samples}
            exact += 1 if len(vals) == 1 else 0
        per_field[k] = round(exact / len(common), 4) if common else 0.0
    r_exact = sum(1 for bid in common if len({bool(s[bid].get("r")) for s in samples}) == 1)
    return {
        "n_samples": len(samples),
        "comparable_beats": len(common),
        "exact_agreement": per_field,
        "r_agreement": round(r_exact / len(common), 4) if common else 0.0,
    }


def run_stability(
    runs: Dict[str, Dict[str, List[str]]],
    k: int = 5,
    comparable_only: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """runs: {run_id: {episode: [candidate_id ranked]}}. Compares every pair."""
    ids = sorted(runs)
    out: Dict[str, Any] = {"runs": ids, "pairs": {}}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = runs[ids[i]], runs[ids[j]]
            eps = sorted(set(a) & set(b))
            if comparable_only is not None:
                allow = set(comparable_only)
                eps = [e for e in eps if e in allow]
            if not eps:
                out["pairs"][f"{ids[i]}|{ids[j]}"] = {"episodes": 0}
                continue
            top1 = sum(1 for e in eps if a[e] and b[e] and a[e][0] == b[e][0])
            ov = [topk_overlap(a[e], b[e], k) for e in eps]
            out["pairs"][f"{ids[i]}|{ids[j]}"] = {
                "episodes": len(eps),
                "top1_agreement": f"{top1}/{len(eps)}",
                "top1_rate": round(top1 / len(eps), 4),
                f"mean_top{k}_overlap": round(sum(ov) / len(ov), 4),
            }
    return out
