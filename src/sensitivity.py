"""Weight Sensitivity Analysis (src/sensitivity.py).

Performs perturbation testing on dev episodes:
- Independent perturbations: 4 weights x [x0.5, x1.5] = 8 runs
- 20 random weight vectors uniformly sampled in [0.5, 1.5]^4
Total: 28 perturbed configurations.
Measures mean overlap (IoU >= 0.5) against the default top-3 candidate selection.

Structural gates: DURATION, VALID_BEAT_IDS, ORDERED_NONOVERLAPPING_SEGMENTS,
                  CUT_STRADDLES_SPOKEN, RESOLVED_ENDING.
When structural_gates_only=True, the pool excludes windows rejected by
HOOK_TOO_WEAK or WEAK_CLIFFHANGER (before the weighted-score dedupe step).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from src.gates import evaluate_window_gates
from src.windows import compute_interval_iou

# Gates whose failure does NOT exclude a window from the structural pool.
_NON_STRUCTURAL_GATES = {"HOOK_TOO_WEAK", "WEAK_CLIFFHANGER", "DUPLICATE_OVERLAP", "DUPLICATE_ANCHOR"}


def generate_perturbation_weights(seed: int = 42) -> List[Dict[str, float]]:
    """Generate 28 perturbed weight vectors (8 independent + 20 random)."""
    base = {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}
    weights_list: List[Dict[str, float]] = []

    # 1. Independent perturbations: 4 weights x [0.5, 1.5]
    for key in ["hook", "cliff", "intensity", "context"]:
        for factor in [0.5, 1.5]:
            w = dict(base)
            w[key] = round(base[key] * factor, 3)
            weights_list.append(w)

    # 2. 20 random weight vectors in [0.5, 1.5]
    rng = random.Random(seed)
    for _ in range(20):
        w = {
            "hook": round(rng.uniform(0.5, 1.5), 3),
            "cliff": round(rng.uniform(0.5, 1.5), 3),
            "intensity": round(rng.uniform(0.5, 1.5), 3),
            "context": round(rng.uniform(0.5, 1.5), 3),
        }
        weights_list.append(w)

    return weights_list


def compute_top_k_for_weights(
    windows: List[Dict[str, Any]],
    weights: Dict[str, float],
    top_k: int = 3,
    max_iou: float = 0.5,
    structural_gates_only: bool = False,
) -> List[Dict[str, Any]]:
    """Re-score candidates passing gates 1-7 (or structural-only subset) and select top K.

    Args:
        windows: All windows with gate results already populated.
        weights: Scoring weight vector.
        top_k: Maximum survivors.
        max_iou: IoU threshold for greedy deduplication.
        structural_gates_only: When True, include windows that only failed
            HOOK_TOO_WEAK, WEAK_CLIFFHANGER, DUPLICATE_OVERLAP, or DUPLICATE_ANCHOR
            (i.e., passed all five structural gates).
    """
    w_hook = weights.get("hook", 1.0)
    w_cliff = weights.get("cliff", 1.0)
    w_intensity = weights.get("intensity", 1.0)
    w_context = weights.get("context", 1.0)

    if structural_gates_only:
        # Include windows where failure (if any) was only in non-structural gates.
        valid = [
            w for w in windows
            if w.get("passed_individual_gates", False)
            or w.get("first_failed_gate") in _NON_STRUCTURAL_GATES
        ]
    else:
        # Filter to candidates that passed individual gates 1-7
        valid = [w for w in windows if w.get("passed_individual_gates", True)]

    # Rescore
    rescored = []
    for w in valid:
        c = dict(w)
        score = (
            w_hook * c.get("hook", 0)
            + w_cliff * c.get("cliff", 0)
            + w_intensity * c.get("intensity", 0.0)
            - w_context * c.get("context", 0)
        )
        c["score"] = round(score, 3)
        rescored.append(c)

    rescored.sort(
        key=lambda x: (x["score"], x.get("cliff", 0), x.get("duration_s", 0.0)),
        reverse=True,
    )

    survivors: List[Dict[str, Any]] = []
    for cand in rescored:
        if len(survivors) >= top_k:
            break
        is_dup = False
        for s in survivors:
            iou = compute_interval_iou(cand["start_s"], cand["end_s"], s["start_s"], s["end_s"])
            if iou > max_iou:
                is_dup = True
                break
        if not is_dup:
            survivors.append(cand)

    return survivors


def calculate_overlap_ratio(
    default_top_k: List[Dict[str, Any]],
    perturbed_top_k: List[Dict[str, Any]],
    match_iou_threshold: float = 0.5,
) -> float:
    """Calculate share of default top-K matching the perturbed top-K at IoU >= 0.5."""
    if not default_top_k:
        return 1.0 if not perturbed_top_k else 0.0

    matches = 0
    for d in default_top_k:
        found = False
        for p in perturbed_top_k:
            iou = compute_interval_iou(d["start_s"], d["end_s"], p["start_s"], p["end_s"])
            if iou >= match_iou_threshold:
                found = True
                break
        if found:
            matches += 1

    return matches / len(default_top_k)


def run_weight_sensitivity_analysis(
    episodes_windows: Dict[str, List[Dict[str, Any]]],
    seed: int = 42,
    top_k: int = 3,
    max_iou: float = 0.5,
    structural_gates_only: bool = False,
) -> Dict[str, Any]:
    """Execute 28 weight perturbations and calculate mean overlap with default top-3.

    Args:
        episodes_windows: Dict mapping episode ID → list of windows (with gate results).
        seed: RNG seed for reproducibility.
        top_k: Candidates per episode.
        max_iou: Greedy dedup IoU threshold.
        structural_gates_only: When True, pool = windows passing only structural gates
            (DURATION, VALID_BEAT_IDS, ORDERED_NONOVERLAPPING_SEGMENTS,
             CUT_STRADDLES_SPOKEN, RESOLVED_ENDING); HOOK_TOO_WEAK and WEAK_CLIFFHANGER
            failures are ignored for pool membership.
    """
    perturbations = generate_perturbation_weights(seed=seed)
    default_weights = {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}

    # Report pool sizes per episode when structural_gates_only=True
    pool_sizes: Dict[str, int] = {}
    if structural_gates_only:
        for ep_id, windows in episodes_windows.items():
            pool = [
                w for w in windows
                if w.get("passed_individual_gates", False)
                or w.get("first_failed_gate") in _NON_STRUCTURAL_GATES
            ]
            pool_sizes[ep_id] = len(pool)

    episode_overlaps: Dict[str, List[float]] = {ep: [] for ep in episodes_windows}
    config_mean_overlaps: List[float] = []

    for weights in perturbations:
        conf_overlaps = []
        for ep_id, windows in episodes_windows.items():
            default_top3 = compute_top_k_for_weights(
                windows, default_weights, top_k=top_k, max_iou=max_iou,
                structural_gates_only=structural_gates_only,
            )
            perturbed_top3 = compute_top_k_for_weights(
                windows, weights, top_k=top_k, max_iou=max_iou,
                structural_gates_only=structural_gates_only,
            )
            ratio = calculate_overlap_ratio(default_top3, perturbed_top3, match_iou_threshold=0.5)
            episode_overlaps[ep_id].append(ratio)
            conf_overlaps.append(ratio)
        config_mean_overlaps.append(sum(conf_overlaps) / max(1, len(conf_overlaps)))

    overall_mean = (
        sum(config_mean_overlaps) / len(config_mean_overlaps)
        if config_mean_overlaps
        else 0.0
    )

    per_ep_means = {
        ep: round(sum(scores) / len(scores), 3) if scores else 0.0
        for ep, scores in episode_overlaps.items()
    }

    result: Dict[str, Any] = {
        "total_perturbations": len(perturbations),
        "overall_mean_overlap": round(overall_mean, 3),
        "per_episode_mean_overlap": per_ep_means,
        "structural_gates_only": structural_gates_only,
    }
    if structural_gates_only:
        result["pool_sizes"] = pool_sizes
    return result

