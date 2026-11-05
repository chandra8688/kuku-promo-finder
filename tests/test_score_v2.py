import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.score import DEFAULT_WEIGHTS, hard_gate_ok, rank_key, score_window, soft_gate_flags

GOOD = {
    "duration_s": 38.0,
    "hook": 3.0,
    "hook_latency_s": 0.0,
    "cliff": 3.0,
    "intensity": 2.4,
    "peak_intensity": 3.0,
    "intensity_slope": 0.6,
    "context": 0.0,
    "tail_pos": 0.9,
    "resolved_tail": False,
    "resolved_last": False,
    "dialogue_share": 0.7,
    "speaker_count": 2,
    "beat_count": 12,
}


def test_breakdown_sums_to_total():
    total, bd = score_window(GOOD)
    assert abs(sum(bd["contrib"].values()) - total) < 1e-6
    assert set(bd["terms"]) == set(bd["contrib"])


def test_resolved_ending_is_penalised_not_fatal():
    base, _ = score_window(GOOD)
    bad = dict(GOOD, resolved_tail=True, resolved_last=True)
    worse, _ = score_window(bad)
    assert worse < base
    ok, reason = hard_gate_ok(bad)
    assert ok is True and reason is None


def test_weak_hook_is_a_flag_not_a_reject():
    weak = dict(GOOD, hook=0.0, cliff=0.0)
    flags = soft_gate_flags(weak)
    assert flags["WEAK_HOOK"] and flags["WEAK_CLIFFHANGER"]
    assert flags["low_confidence"] is True
    assert hard_gate_ok(weak)[0] is True


def test_hard_gates_only_fire_on_structure():
    assert hard_gate_ok(dict(GOOD, duration_s=12.0))[1] == "DURATION_SHORT"
    assert hard_gate_ok(dict(GOOD, duration_s=75.0))[1] == "DURATION_LONG"
    assert hard_gate_ok(dict(GOOD, beat_count=0))[1] == "EMPTY_WINDOW"
    assert hard_gate_ok(dict(GOOD, cut_straddles_spoken=True))[1] == "CUT_STRADDLES_SPOKEN"


def test_zero_weights_give_zero_score():
    total, _ = score_window(GOOD, {k: 0.0 for k in DEFAULT_WEIGHTS})
    assert total == 0.0


def test_rank_key_is_total_and_deterministic():
    a = {"candidate_id": "a", "score": 5.0, "cliff": 3, "hook": 3, "start_s": 10.0}
    b = {"candidate_id": "b", "score": 5.0, "cliff": 3, "hook": 3, "start_s": 10.0}
    c = {"candidate_id": "c", "score": 5.0, "cliff": 3, "hook": 3, "start_s": 10.0, "low_confidence": True}
    assert sorted([c, b, a], key=rank_key) == [a, b, c]
    assert rank_key(a) != rank_key(b)


def test_hook_latency_penalty_prefers_an_immediate_hook():
    late, _ = score_window(dict(GOOD, hook_latency_s=2.8))
    early, _ = score_window(GOOD)
    assert early > late
