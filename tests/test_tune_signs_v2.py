import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("tw", ROOT / "scripts" / "tune_weights.py")
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)


def test_penalty_terms_cannot_go_positive():
    for key in ("resolved_tail", "resolved_last", "context", "straddle_spoken",
                "straddle_action", "silence", "speaker_overload", "hook_latency"):
        assert max(tw.allowed_grid(key)) <= 0.0, key
        assert tw.allowed_grid(key), key


def test_reward_terms_cannot_go_negative():
    for key in ("intensity", "peak_intensity", "intensity_slope", "tail_pos"):
        assert min(tw.allowed_grid(key)) >= 0.0, key


def test_the_two_named_criteria_cannot_be_zeroed_out():
    # "hook in the first 3s" and "unresolved cliffhanger" are stated criteria;
    # a fit must not delete either by driving its weight to 0
    for key in ("hook", "cliff"):
        assert min(tw.allowed_grid(key)) >= 1.0, key
    assert tw.bound_violations({"cliff": 0.25}) != []
    assert tw.bound_violations({"hook": 0.0}) != []


def test_unconstrained_term_keeps_the_full_grid():
    assert tw.allowed_grid("something_new") == tw.GRID


def test_violations_detected():
    bad = tw.bound_violations({"resolved_tail": 1.4, "straddle_action": 0.25, "hook": -1.0})
    assert len(bad) == 3
    assert any("resolved_tail" in b for b in bad)


def test_the_shipped_bad_config_is_rejected():
    # exactly what config/scoring_v2.yaml held after the unconstrained fit
    shipped = {
        "cliff": 0.8, "context": -1.0, "hook": 2.0, "hook_latency": -0.25,
        "intensity": 0.8, "intensity_slope": 0.4, "peak_intensity": 0.3,
        "resolved_last": -1.5, "resolved_tail": 1.4, "silence": -0.5,
        "speaker_overload": -0.15, "straddle_action": 0.25, "straddle_spoken": -0.75,
        "tail_pos": 0.6,
    }
    bad = tw.bound_violations(shipped)
    # resolved_tail +1.4 and straddle_action +0.25 flipped sign; cliff 0.8 fell below the floor
    assert len(bad) == 3
    assert any("resolved_tail" in b for b in bad)
    assert any("cliff" in b for b in bad)


def test_defaults_satisfy_the_constraints():
    from src.score import DEFAULT_WEIGHTS
    assert tw.bound_violations(dict(DEFAULT_WEIGHTS)) == []
