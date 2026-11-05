"""Human review labels -> show-level precision@k.

The point of this module is that precision@k must come from a person deciding
whether a clip is postable, not from re-checking the filters the pipeline already
applied. A metric built out of the pipeline's own gates returns 100% on random
input, which is how the previous version passed unnoticed.

Labels come from scripts/review.py, which appends rows of
    candidate_id, episode, start_s, end_s, decision, reason_code
with decision in {approve, reject}.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from .evalkit import iou
except ImportError:
    from evalkit import iou

MATCH_IOU = 0.9


def load_labels(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Returns {candidate_id: row}. Later rows win, so a re-review overrides."""
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("candidate_id") or "").strip()
            dec = (row.get("decision") or "").strip().lower()
            if not cid or dec not in ("approve", "reject"):
                continue
            out[cid] = {
                "candidate_id": cid,
                "episode": (row.get("episode") or "").strip(),
                "start_s": float(row["start_s"]) if row.get("start_s") else None,
                "end_s": float(row["end_s"]) if row.get("end_s") else None,
                "decision": dec,
                "reason_code": (row.get("reason_code") or "").strip(),
            }
    return out


def match_label(promo: Dict[str, Any], labels: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Exact candidate_id, else same episode and a near-identical window."""
    cid = promo.get("candidate_id")
    if cid and cid in labels:
        return labels[cid]
    ps, pe, ep = promo.get("start_s"), promo.get("end_s"), str(promo.get("episode", ""))
    if ps is None or pe is None:
        return None
    for row in labels.values():
        if row["start_s"] is None or row["end_s"] is None:
            continue
        if row["episode"] and ep and row["episode"] != ep:
            continue
        if iou(float(ps), float(pe), row["start_s"], row["end_s"]) >= MATCH_IOU:
            return row
    return None


def precision_at_k(
    promos: Sequence[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    k: int = 15,
    n_boot: int = 2000,
    seed: int = 20260101,
) -> Dict[str, Any]:
    """Precision over the top k, counted only on clips a person actually labelled.

    The resampling unit is the clip, because this is one ranked list rather than a
    per-episode measurement. `complete` is False whenever any of the top k is
    unlabelled; callers must not print a bare percentage in that case.
    """
    top = list(promos)[:k]
    n_eval = len(top)
    decisions: List[int] = []
    unlabelled: List[str] = []
    rejected: List[Dict[str, Any]] = []
    for p in top:
        row = match_label(p, labels)
        if row is None:
            unlabelled.append(str(p.get("candidate_id") or f"{p.get('episode')}@{p.get('start_s')}"))
            continue
        ok = row["decision"] == "approve"
        decisions.append(1 if ok else 0)
        if not ok:
            rejected.append({"candidate_id": row["candidate_id"], "reason_code": row["reason_code"]})

    n_lab = len(decisions)
    out: Dict[str, Any] = {
        "k": k,
        "n_in_list": n_eval,
        "n_labelled": n_lab,
        "n_approved": sum(decisions),
        "coverage": round(n_lab / n_eval, 4) if n_eval else 0.0,
        "complete": bool(n_eval) and n_lab == n_eval,
        "unlabelled": unlabelled,
        "rejected": rejected,
    }
    if not n_lab:
        out.update({"precision": None, "ci_low": None, "ci_high": None})
        return out

    point = sum(decisions) / n_lab
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(decisions[rng.randrange(n_lab)] for _ in range(n_lab)) / n_lab)
    boots.sort()
    out.update(
        {
            "precision": round(point, 4),
            "ci_low": round(boots[int(0.025 * n_boot)], 4),
            "ci_high": round(boots[min(int(0.975 * n_boot), n_boot - 1)], 4),
        }
    )
    return out


def format_cell(res: Dict[str, Any]) -> str:
    """One table cell. Never a bare percentage on partial labels."""
    if not res or not res.get("n_labelled"):
        return "not labelled"
    body = f"{res['n_approved']}/{res['n_labelled']}"
    if res.get("precision") is not None:
        body += f" ({res['precision']*100:.0f}%, CI [{res['ci_low']*100:.0f}, {res['ci_high']*100:.0f}])"
    if not res.get("complete"):
        body += f" — partial, {res['n_labelled']} of {res['n_in_list']} labelled"
    return body
