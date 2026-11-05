"""Evaluation kit: IoU matching, bootstrap CIs, paired comparison, rank stats.

Pure stdlib. No scipy, no numpy. Everything resamples over EPISODES, not over
windows, because windows inside one episode are not independent.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def match_one_to_one(
    preds: Sequence[Dict[str, Any]],
    golds: Sequence[Dict[str, Any]],
    thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """Greedy max-IoU matching, each gold and each pred used at most once."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            v = iou(float(p["start_s"]), float(p["end_s"]), float(g["start_s"]), float(g["end_s"]))
            if v >= thresh:
                pairs.append((v, pi, gi))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_g, out = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        p, g = preds[pi], golds[gi]
        out.append(
            {
                "pred_rank": pi + 1,
                "pred": [float(p["start_s"]), float(p["end_s"])],
                "gold_id": g.get("gold_id"),
                "gold": [float(g["start_s"]), float(g["end_s"])],
                "iou": round(v, 4),
                "start_err": round(abs(float(p["start_s"]) - float(g["start_s"])), 2),
                "end_err": round(abs(float(p["end_s"]) - float(g["end_s"])), 2),
            }
        )
    return out


def episode_metrics(
    preds: Sequence[Dict[str, Any]],
    golds: Sequence[Dict[str, Any]],
    thresh: float = 0.5,
    k_list: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    full = match_one_to_one(preds, golds, thresh)
    m: Dict[str, Any] = {"n_gold": len(golds), "n_pred": len(preds), "matches": full}
    for k in k_list:
        hit = any(mt["pred_rank"] <= k for mt in full)
        m[f"top{k}_hit"] = 1 if hit else 0
    m["recall"] = len({mt["gold_id"] for mt in full}) / len(golds) if golds else 0.0
    m["start_errs"] = [mt["start_err"] for mt in full]
    m["end_errs"] = [mt["end_err"] for mt in full]
    return m


def system_metrics(
    preds_by_ep: Dict[str, Sequence[Dict[str, Any]]],
    golds_by_ep: Dict[str, Sequence[Dict[str, Any]]],
    thresh: float = 0.5,
    k_list: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    per_ep = {}
    for ep, golds in sorted(golds_by_ep.items()):
        per_ep[ep] = episode_metrics(preds_by_ep.get(ep, []), golds, thresh, k_list)
    eps = list(per_ep)
    n_gold = sum(per_ep[e]["n_gold"] for e in eps)
    matched = sum(len({m["gold_id"] for m in per_ep[e]["matches"]}) for e in eps)
    s_errs = [v for e in eps for v in per_ep[e]["start_errs"]]
    e_errs = [v for e in eps for v in per_ep[e]["end_errs"]]
    out: Dict[str, Any] = {
        "n_episodes": len(eps),
        "n_gold": n_gold,
        "gold_recall": matched / n_gold if n_gold else 0.0,
        "gold_matched": matched,
        "median_start_err": round(statistics.median(s_errs), 2) if s_errs else None,
        "median_end_err": round(statistics.median(e_errs), 2) if e_errs else None,
        "per_episode": per_ep,
    }
    for k in k_list:
        hits = [per_ep[e][f"top{k}_hit"] for e in eps]
        out[f"top{k}"] = sum(hits)
        out[f"top{k}_rate"] = sum(hits) / len(hits) if hits else 0.0
    return out


def bootstrap_ci(
    values_by_ep: Dict[str, float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260101,
) -> Tuple[float, float, float]:
    """Mean with a percentile CI, resampling episodes with replacement."""
    eps = sorted(values_by_ep)
    if not eps:
        return 0.0, 0.0, 0.0
    vals = [values_by_ep[e] for e in eps]
    point = sum(vals) / len(vals)
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return round(point, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap(
    a_by_ep: Dict[str, float],
    b_by_ep: Dict[str, float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260101,
) -> Dict[str, Any]:
    """Is system A better than B on the same episodes? Resample episodes together."""
    eps = sorted(set(a_by_ep) & set(b_by_ep))
    if not eps:
        return {"n": 0}
    diffs = [a_by_ep[e] - b_by_ep[e] for e in eps]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return {
        "n": n,
        "mean_delta": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_a_better": round(sum(1 for b in boots if b > 0) / n_boot, 4),
        "significant": bool(lo > 0 or hi < 0),
    }


def kendall_tau(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Tau-b over paired ranks. O(n^2), fine for n <= a few hundred."""
    n = len(a)
    if n < 2:
        return 0.0
    con = dis = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1
                tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif (da > 0) == (db > 0):
                con += 1
            else:
                dis += 1
    denom = ((con + dis + ta) * (con + dis + tb)) ** 0.5
    return round((con - dis) / denom, 4) if denom else 0.0


def topk_overlap(a: Sequence[Any], b: Sequence[Any], k: int = 5) -> float:
    """Divides by the effective k so a short list is not punished for being short."""
    eff = min(k, len(a), len(b))
    if eff <= 0:
        return 0.0
    return round(len(set(list(a)[:eff]) & set(list(b)[:eff])) / eff, 4)


def markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def fmt_ci(point: float, lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"{point*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]"
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"
