"""The candidate engine. One implementation, three callers.

run.py (via src/candidates.py), scripts/eval_v2.py (via src/search.py) and
scripts/tune_weights.py all select candidates through this module. Before it
existed there were three copies of selection: src/gates.py in production,
src/search.py in evaluation, and an inline re-implementation in the tuner. The
last two skipped the shared-anchor dedupe rule, so the weights were fitted and
the metrics reported on a selection procedure run.py never executes.

Features do not depend on weights, so enumerate() is the expensive call and
rescore() + select() is cheap. The tuner relies on that split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.gates import evaluate_window_gates
from src.score import score_window
from src.windows import enumerate_windows_for_episode, load_scoring_config, load_v2_config

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_duration_s": 20.0,
    "max_duration_s": 60.0,
    "min_hook": 2,
    "min_cliff": 2,
    "max_iou_overlap": 0.5,
    "max_containment_overlap": 0.6,
    "top_k": 3,
    "min_survivors": 2,
    "hook_window_s": 3.0,
}


def thresholds() -> Dict[str, Any]:
    t = dict(DEFAULT_THRESHOLDS)
    t.update(load_scoring_config().get("thresholds") or {})
    return t


def v2_weights() -> Dict[str, float]:
    return dict(load_v2_config().get("weights") or {})


def _annotations(annotated: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not annotated:
        return {}
    if isinstance(annotated.get("annotations"), dict):
        return annotated["annotations"]
    return annotated


def enumerate(parsed: Dict[str, Any], annotated: Optional[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    t = thresholds()
    return enumerate_windows_for_episode(
        parsed_data=parsed,
        annotated_data=annotated or {"annotations": {}},
        weights=weights if weights is not None else v2_weights(),
        min_duration_s=t["min_duration_s"],
        max_duration_s=t["max_duration_s"],
        hook_window_s=t["hook_window_s"],
    )


def rescore(windows: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """Fresh shallow copies with score and breakdown recomputed. Features untouched."""
    out = []
    for w in windows:
        c = dict(w)
        sc, bd = score_window(w["features"], weights)
        c["score"] = sc
        c["breakdown"] = bd
        for k in ("score_refined", "breakdown_refined", "features_refined", "gate_results",
                  "passed_all_gates", "selected_top_k", "first_failed_gate"):
            c.pop(k, None)
        c["start_s"] = w.get("start_s_raw", w["start_s"])
        c["end_s"] = w.get("end_s_raw", w["end_s"])
        c["duration_s"] = round(c["end_s"] - c["start_s"], 2)
        out.append(c)
    return out


def select(
    windows: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    annotated: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
    refine: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    t = thresholds()
    return evaluate_window_gates(
        windows=windows,
        episode_beats=parsed.get("beats", []),
        min_hook=t["min_hook"],
        min_cliff=t["min_cliff"],
        min_duration_s=t["min_duration_s"],
        max_duration_s=t["max_duration_s"],
        max_iou=t["max_iou_overlap"],
        max_containment=t["max_containment_overlap"],
        top_k=top_k if top_k is not None else t["top_k"],
        min_survivors=t["min_survivors"],
        refine=refine,
        annotations=_annotations(annotated),
        weights=weights if weights is not None else v2_weights(),
    )


def run_episode(
    parsed: Dict[str, Any],
    annotated: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
    refine: bool = True,
):
    """Returns (windows, all_evaluated, survivors, gate_fail_counts)."""
    w = weights if weights is not None else v2_weights()
    wins = enumerate(parsed, annotated, w)
    all_eval, surv, counts = select(wins, parsed, annotated, w, top_k, refine)
    return wins, all_eval, surv, counts
