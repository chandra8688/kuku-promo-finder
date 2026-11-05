import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalkit import (
    bootstrap_ci,
    iou,
    kendall_tau,
    markdown_table,
    match_one_to_one,
    paired_bootstrap,
    system_metrics,
    topk_overlap,
)


def test_iou_basics():
    assert iou(0, 10, 0, 10) == 1.0
    assert iou(0, 10, 10, 20) == 0.0
    assert round(iou(0, 10, 5, 15), 4) == round(5 / 15, 4)


def test_matching_is_one_to_one():
    preds = [{"start_s": 0, "end_s": 10}, {"start_s": 1, "end_s": 11}]
    golds = [{"gold_id": "g1", "start_s": 0, "end_s": 10}]
    m = match_one_to_one(preds, golds)
    assert len(m) == 1
    assert m[0]["pred_rank"] == 1 and m[0]["gold_id"] == "g1"


def test_matching_respects_threshold():
    preds = [{"start_s": 0, "end_s": 10}]
    golds = [{"gold_id": "g1", "start_s": 8, "end_s": 20}]
    assert match_one_to_one(preds, golds, 0.5) == []
    assert len(match_one_to_one(preds, golds, 0.1)) == 1


def test_system_metrics_counts_hits_and_recall():
    golds = {
        "ep1": [{"gold_id": "a", "start_s": 0, "end_s": 30}],
        "ep2": [{"gold_id": "b", "start_s": 0, "end_s": 30}, {"gold_id": "c", "start_s": 60, "end_s": 90}],
    }
    preds = {
        "ep1": [{"start_s": 0, "end_s": 30}],
        "ep2": [{"start_s": 100, "end_s": 130}, {"start_s": 60, "end_s": 90}],
    }
    m = system_metrics(preds, golds)
    assert m["top1"] == 1
    assert m["top3"] == 2
    assert m["gold_matched"] == 2 and m["n_gold"] == 3


def test_bootstrap_ci_brackets_the_point_estimate():
    p, lo, hi = bootstrap_ci({"e1": 1.0, "e2": 0.0, "e3": 1.0, "e4": 1.0}, n_boot=500)
    assert lo <= p <= hi
    assert p == 0.75


def test_bootstrap_ci_is_degenerate_when_every_episode_agrees():
    p, lo, hi = bootstrap_ci({"e1": 1.0, "e2": 1.0}, n_boot=200)
    assert (p, lo, hi) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_detects_a_clear_win():
    a = {f"e{i}": 1.0 for i in range(10)}
    b = {f"e{i}": 0.0 for i in range(10)}
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["mean_delta"] == 1.0 and r["significant"] is True


def test_paired_bootstrap_stays_humble_on_a_coin_flip():
    a = {"e1": 1.0, "e2": 0.0, "e3": 1.0, "e4": 0.0}
    b = {"e1": 0.0, "e2": 1.0, "e3": 0.0, "e4": 1.0}
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["significant"] is False


def test_paired_bootstrap_is_reproducible():
    a = {"e1": 1.0, "e2": 0.0, "e3": 1.0}
    b = {"e1": 0.0, "e2": 0.0, "e3": 1.0}
    assert paired_bootstrap(a, b, 400, seed=5) == paired_bootstrap(a, b, 400, seed=5)


def test_kendall_tau_endpoints():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_topk_overlap():
    assert topk_overlap(["a", "b", "c"], ["c", "b", "x"], 3) == round(2 / 3, 4)


def test_topk_overlap_uses_effective_k_for_short_lists():
    assert topk_overlap(["x", "y"], ["y", "x"], 5) == 1.0
    assert topk_overlap([], ["a"], 3) == 0.0


def test_markdown_table_shape():
    t = markdown_table([{"a": 1, "b": 2}], ["a", "b"])
    assert t.splitlines()[0] == "| a | b |"
    assert len(t.splitlines()) == 3
