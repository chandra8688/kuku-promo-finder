"""Unit tests for weight sensitivity analysis (src/sensitivity.py).

Tests:
1. Synthetic test where one weight is set to 100 and others to 0,
   asserting the top candidate ordering changes vs default weights.
2. Extreme weight check helper functionality.
3. Overlap ratio calculations.
"""

import pytest
from src.sensitivity import (
    compute_top_k_for_weights,
    calculate_overlap_ratio,
    run_weight_sensitivity_analysis,
    run_extreme_weight_check,
)


def test_synthetic_weight_extreme_changes_ordering():
    """Verify that setting one weight to 100 and others to 0 flips candidate ranking.

    Candidate 1: high hook (3), low cliff (1)
    Candidate 2: low hook (1), high cliff (3)
    """
    windows = [
        {
            "candidate_id": "cand_hook_heavy",
            "start_s": 0.0,
            "end_s": 30.0,
            "duration_s": 30.0,
            "hook": 3,
            "cliff": 1,
            "intensity": 1.0,
            "context": 0,
            "passed_individual_gates": True,
        },
        {
            "candidate_id": "cand_cliff_heavy",
            "start_s": 40.0,
            "end_s": 70.0,
            "duration_s": 30.0,
            "hook": 1,
            "cliff": 3,
            "intensity": 1.0,
            "context": 0,
            "passed_individual_gates": True,
        },
    ]

    default_weights = {"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0}
    hook_extreme = {"hook": 100.0, "cliff": 0.0, "intensity": 0.0, "context": 0.0}
    cliff_extreme = {"hook": 0.0, "cliff": 100.0, "intensity": 0.0, "context": 0.0}

    # Under hook extreme: cand_hook_heavy must be #1
    top_hook = compute_top_k_for_weights(windows, hook_extreme, top_k=2)
    assert top_hook[0]["candidate_id"] == "cand_hook_heavy"
    assert top_hook[0]["score"] == 300.0

    # Under cliff extreme: cand_cliff_heavy must be #1
    top_cliff = compute_top_k_for_weights(windows, cliff_extreme, top_k=2)
    assert top_cliff[0]["candidate_id"] == "cand_cliff_heavy"
    assert top_cliff[0]["score"] == 300.0

    # The top candidate ordering strictly changed between the two configurations
    assert top_hook[0]["candidate_id"] != top_cliff[0]["candidate_id"]


def test_extreme_weight_check_detects_sensitivity():
    """Verify run_extreme_weight_check detects changes when features diverge."""
    windows = [
        {
            "candidate_id": "c1",
            "start_s": 0.0,
            "end_s": 25.0,
            "duration_s": 25.0,
            "hook": 3,
            "cliff": 1,
            "intensity": 1.0,
            "context": 0,
            "passed_individual_gates": True,
        },
        {
            "candidate_id": "c2",
            "start_s": 30.0,
            "end_s": 55.0,
            "duration_s": 25.0,
            "hook": 1,
            "cliff": 3,
            "intensity": 1.0,
            "context": 0,
            "passed_individual_gates": True,
        },
    ]

    res = run_extreme_weight_check({"test_ep": windows}, top_k=1)
    assert res["sensitivity_confirmed"] is True
    assert res["combos_changed"] > 0


def test_weight_sensitivity_metrics():
    """Verify run_weight_sensitivity_analysis computes overlap, top-1 stability, and Kendall tau."""
    from src.sensitivity import run_weight_sensitivity_analysis, get_distinct_hook_cliff_combos

    windows = [
        {
            "candidate_id": f"c{i}",
            "start_s": float(i * 30),
            "end_s": float(i * 30 + 25),
            "duration_s": 25.0,
            "hook": 3 - (i % 3),
            "cliff": 1 + (i % 3),
            "intensity": 1.0 + i * 0.1,
            "context": 0,
            "passed_individual_gates": True,
        }
        for i in range(10)
    ]

    res = run_weight_sensitivity_analysis({"test_ep": windows}, top_k=3)
    assert "overall_mean_overlap" in res
    assert "overall_top1_stability" in res
    assert "overall_kendall_tau_top5" in res
    assert "distinct_hook_cliff_combos" in res
    assert 0.0 <= res["overall_mean_overlap"] <= 1.0
    assert 0.0 <= res["overall_top1_stability"] <= 1.0
    assert -1.0 <= res["overall_kendall_tau_top5"] <= 1.0

    combos = get_distinct_hook_cliff_combos(windows)
    assert len(combos) > 0

