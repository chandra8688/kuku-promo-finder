import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consistency import aggregate_annotations, annotation_agreement, run_stability


S1 = {"b1": {"h": 3, "q": 2, "r": False, "n": 0, "t": 3}, "b2": {"h": 1, "q": 1, "r": True, "n": 1, "t": 1}}
S2 = {"b1": {"h": 2, "q": 2, "r": False, "n": 0, "t": 3}, "b2": {"h": 1, "q": 2, "r": True, "n": 1, "t": 2}}
S3 = {"b1": {"h": 2, "q": 3, "r": False, "n": 0, "t": 2}, "b2": {"h": 0, "q": 1, "r": False, "n": 1, "t": 1}}


def test_single_sample_passes_through():
    assert aggregate_annotations([S1]) == S1


def test_median_and_majority():
    agg = aggregate_annotations([S1, S2, S3])
    assert agg["b1"]["h"] == 2
    assert agg["b1"]["q"] == 2
    assert agg["b2"]["r"] is True  # 2 of 3 say resolved
    assert agg["b1"]["n_samples"] == 3


def test_disagreement_is_flagged():
    agg = aggregate_annotations([S1, S2, S3])
    assert agg["b2"]["unstable"] is True  # r split 2/1
    assert agg["b2"]["disagreement"] > 0


def test_beat_missing_from_one_sample_still_aggregates():
    agg = aggregate_annotations([S1, {"b1": S2["b1"]}])
    assert agg["b2"]["n_samples"] == 1
    assert agg["b1"]["n_samples"] == 2


def test_agreement_report():
    rep = annotation_agreement([S1, S2, S3])
    assert rep["comparable_beats"] == 2
    assert 0.0 <= rep["exact_agreement"]["h"] <= 1.0
    assert rep["r_agreement"] == 0.5


def test_run_stability_pairs():
    runs = {
        "primary": {"ep1": ["a", "b", "c"], "ep2": ["x", "y"]},
        "run2": {"ep1": ["a", "c", "b"], "ep2": ["y", "x"]},
    }
    st = run_stability(runs, k=3)
    pair = st["pairs"]["primary|run2"]
    assert pair["episodes"] == 2
    assert pair["top1_agreement"] == "1/2"
    assert pair["mean_top3_overlap"] == 1.0


def test_run_stability_respects_comparable_filter():
    runs = {"a": {"ep1": ["x"], "ep2": ["y"]}, "b": {"ep1": ["x"], "ep2": ["z"]}}
    st = run_stability(runs, k=1, comparable_only=["ep1"])
    assert st["pairs"]["a|b"]["episodes"] == 1
    assert st["pairs"]["a|b"]["top1_rate"] == 1.0
