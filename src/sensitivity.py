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
from typing import Any, Dict, List, Set, Tuple

from src.gates import evaluate_window_gates
from src.windows import compute_interval_iou

# Gates whose failure does NOT exclude a window from the structural pool.
_NON_STRUCTURAL_GATES = {"HOOK_TOO_WEAK", "WEAK_CLIFFHANGER", "DUPLICATE_OVERLAP", "DUPLICATE_ANCHOR", "DUPLICATE_CONTAINED"}


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


def run_extreme_weight_check(
    episodes_windows: Dict[str, List[Dict[str, Any]]],
    top_k: int = 3,
    max_iou: float = 0.5,
    structural_gates_only: bool = False,
) -> Dict[str, Any]:
    """Check whether zeroing one weight at a time changes the top-K selection.

    Uses weight vectors where one weight is 100 and the others are 0.
    Returns per-episode, per-weight overlap vs default (1.0=identical, <1.0=changed).
    This deliberately uses extreme values outside the [0.5, 1.5] perturbation range
    to verify the sensitivity function is not broken.
    """
    default_weights = {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}
    weight_keys = ["hook", "cliff", "intensity", "context"]

    results: Dict[str, Any] = {}
    for ep_id, windows in episodes_windows.items():
        ep_result: Dict[str, float] = {}
        default_top3 = compute_top_k_for_weights(
            windows, default_weights, top_k=top_k, max_iou=max_iou,
            structural_gates_only=structural_gates_only,
        )
        for key in weight_keys:
            extreme = {k: (100.0 if k == key else 0.0) for k in weight_keys}
            extreme_top3 = compute_top_k_for_weights(
                windows, extreme, top_k=top_k, max_iou=max_iou,
                structural_gates_only=structural_gates_only,
            )
            overlap = calculate_overlap_ratio(default_top3, extreme_top3)
            ep_result[f"{key}@100"] = round(overlap, 3)
        results[ep_id] = ep_result

    # Overall: fraction of (episode, weight) combos where top-k changed
    all_overlaps = [v for ep in results.values() for v in ep.values()]
    n_changed = sum(1 for o in all_overlaps if o < 1.0)
    return {
        "per_episode": results,
        "total_combos": len(all_overlaps),
        "combos_changed": n_changed,
        "sensitivity_confirmed": n_changed > 0,
    }


def compute_kendall_tau_top_k(
    default_top: List[Dict[str, Any]],
    perturbed_top: List[Dict[str, Any]],
) -> float:
    """Compute Kendall's tau between default top-K and perturbed ranking.

    Compares relative ordering of items in default_top.
    If an item from default_top does not appear in perturbed_top,
    its rank is assigned len(perturbed_top) + 1.
    """
    n = len(default_top)
    if n < 2:
        return 1.0

    p_ranks = {cand["candidate_id"]: idx for idx, cand in enumerate(perturbed_top)}
    fallback_rank = len(perturbed_top) + 1

    concordant = 0
    discordant = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        c1_id = default_top[i]["candidate_id"]
        r1 = p_ranks.get(c1_id, fallback_rank)
        for j in range(i + 1, n):
            c2_id = default_top[j]["candidate_id"]
            r2 = p_ranks.get(c2_id, fallback_rank)
            if r1 < r2:
                concordant += 1
            elif r1 > r2:
                discordant += 1

    return (concordant - discordant) / total_pairs


def get_distinct_hook_cliff_combos(
    windows: List[Dict[str, Any]],
    structural_gates_only: bool = False,
) -> Set[Tuple[int, int]]:
    """Return distinct (hook, cliff) tuples among valid windows."""
    if structural_gates_only:
        valid = [
            w for w in windows
            if w.get("passed_individual_gates", False)
            or w.get("first_failed_gate") in _NON_STRUCTURAL_GATES
        ]
    else:
        valid = [w for w in windows if w.get("passed_individual_gates", True)]
    return {(w.get("hook", 0), w.get("cliff", 0)) for w in valid}


def run_weight_sensitivity_analysis(
    episodes_windows: Dict[str, List[Dict[str, Any]]],
    seed: int = 42,
    top_k: int = 3,
    max_iou: float = 0.5,
    structural_gates_only: bool = False,
) -> Dict[str, Any]:
    """Execute 28 weight perturbations and calculate overlap, top-1 stability, and Kendall tau.

    Args:
        episodes_windows: Dict mapping episode ID → list of windows (with gate results).
        seed: RNG seed for reproducibility.
        top_k: Candidates per episode for set overlap.
        max_iou: Greedy dedup IoU threshold.
        structural_gates_only: When True, pool = windows passing only structural gates
            (DURATION, VALID_BEAT_IDS, ORDERED_NONOVERLAPPING_SEGMENTS,
             CUT_STRADDLES_SPOKEN, RESOLVED_ENDING); HOOK_TOO_WEAK and WEAK_CLIFFHANGER
            failures are ignored for pool membership.
    """
    perturbations = generate_perturbation_weights(seed=seed)
    default_weights = {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}

    # Report pool sizes and distinct (hook, cliff) combos per episode
    pool_sizes: Dict[str, int] = {}
    distinct_combos: Dict[str, int] = {}
    for ep_id, windows in episodes_windows.items():
        if structural_gates_only:
            pool = [
                w for w in windows
                if w.get("passed_individual_gates", False)
                or w.get("first_failed_gate") in _NON_STRUCTURAL_GATES
            ]
        else:
            pool = [w for w in windows if w.get("passed_individual_gates", True)]
        pool_sizes[ep_id] = len(pool)
        distinct_combos[ep_id] = len({(w.get("hook", 0), w.get("cliff", 0)) for w in pool})

    episode_overlaps: Dict[str, List[float]] = {ep: [] for ep in episodes_windows}
    episode_top1: Dict[str, List[float]] = {ep: [] for ep in episodes_windows}
    episode_tau: Dict[str, List[float]] = {ep: [] for ep in episodes_windows}

    config_mean_overlaps: List[float] = []
    config_mean_top1: List[float] = []
    config_mean_tau: List[float] = []

    # Precompute default top-k and top-5 per episode
    defaults_top_k = {
        ep_id: compute_top_k_for_weights(
            windows, default_weights, top_k=top_k, max_iou=max_iou,
            structural_gates_only=structural_gates_only,
        )
        for ep_id, windows in episodes_windows.items()
    }
    defaults_top_5 = {
        ep_id: compute_top_k_for_weights(
            windows, default_weights, top_k=5, max_iou=max_iou,
            structural_gates_only=structural_gates_only,
        )
        for ep_id, windows in episodes_windows.items()
    }

    for weights in perturbations:
        conf_overlaps = []
        conf_top1 = []
        conf_tau = []

        for ep_id, windows in episodes_windows.items():
            def_k = defaults_top_k[ep_id]
            def_5 = defaults_top_5[ep_id]

            pert_k = compute_top_k_for_weights(
                windows, weights, top_k=top_k, max_iou=max_iou,
                structural_gates_only=structural_gates_only,
            )
            pert_5 = compute_top_k_for_weights(
                windows, weights, top_k=5, max_iou=max_iou,
                structural_gates_only=structural_gates_only,
            )

            # 1. Set overlap (IoU >= 0.5)
            ratio = calculate_overlap_ratio(def_k, pert_k, match_iou_threshold=0.5)
            episode_overlaps[ep_id].append(ratio)
            conf_overlaps.append(ratio)

            # 2. Top-1 stability
            t1 = 1.0 if (def_k and pert_k and def_k[0].get("candidate_id") == pert_k[0].get("candidate_id")) else 0.0
            episode_top1[ep_id].append(t1)
            conf_top1.append(t1)

            # 3. Kendall tau over top 5
            tau = compute_kendall_tau_top_k(def_5, pert_5)
            episode_tau[ep_id].append(tau)
            conf_tau.append(tau)

        config_mean_overlaps.append(sum(conf_overlaps) / max(1, len(conf_overlaps)))
        config_mean_top1.append(sum(conf_top1) / max(1, len(conf_top1)))
        config_mean_tau.append(sum(conf_tau) / max(1, len(conf_tau)))

    overall_mean = sum(config_mean_overlaps) / len(config_mean_overlaps) if config_mean_overlaps else 0.0
    overall_top1 = sum(config_mean_top1) / len(config_mean_top1) if config_mean_top1 else 0.0
    overall_tau = sum(config_mean_tau) / len(config_mean_tau) if config_mean_tau else 0.0

    per_ep_overlap = {
        ep: round(sum(scores) / len(scores), 3) if scores else 0.0
        for ep, scores in episode_overlaps.items()
    }
    per_ep_top1 = {
        ep: round(sum(scores) / len(scores), 3) if scores else 0.0
        for ep, scores in episode_top1.items()
    }
    per_ep_tau = {
        ep: round(sum(scores) / len(scores), 3) if scores else 0.0
        for ep, scores in episode_tau.items()
    }

    result: Dict[str, Any] = {
        "total_perturbations": len(perturbations),
        "overall_mean_overlap": round(overall_mean, 3),
        "per_episode_mean_overlap": per_ep_overlap,
        "overall_top1_stability": round(overall_top1, 3),
        "per_episode_top1_stability": per_ep_top1,
        "overall_kendall_tau_top5": round(overall_tau, 3),
        "per_episode_kendall_tau_top5": per_ep_tau,
        "structural_gates_only": structural_gates_only,
        "pool_sizes": pool_sizes,
        "distinct_hook_cliff_combos": distinct_combos,
    }
    return result

