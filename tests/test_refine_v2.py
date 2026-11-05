import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.refine import refine_boundaries, refine_candidate


def mk(bid, kind, s, e):
    return {"beat_id": bid, "kind": kind, "start_s": s, "end_s": e, "text": ""}


def test_lead_silence_is_trimmed_to_the_first_real_beat():
    beats = [mk("b1", "CAMERA", 0.0, 4.0), mk("b2", "DIALOGUE", 4.0, 10.0), mk("b3", "DIALOGUE", 10.2, 30.0)]
    s, e, ops = refine_boundaries(beats, 0.0, 30.0, ["b2", "b3"])
    assert s == 4.0
    assert any(o.startswith("trim_lead_silence") for o in ops)


def test_end_extends_to_finish_the_last_line():
    beats = [mk("b1", "DIALOGUE", 0.0, 20.0), mk("b2", "DIALOGUE", 20.1, 26.0)]
    s, e, ops = refine_boundaries(beats, 0.0, 24.0, ["b1", "b2"])
    assert e == 26.6
    assert any(o.startswith("extend_to_last_line") for o in ops)


def test_trailing_dead_air_is_trimmed_back():
    beats = [mk("b1", "DIALOGUE", 0.0, 22.0), mk("b2", "ACTION", 22.0, 40.0)]
    s, e, ops = refine_boundaries(beats, 0.0, 40.0, ["b1", "b2"])
    assert e == 22.6
    assert any(o.startswith("trim_trailing_dead") for o in ops)


def test_refinement_never_breaks_the_duration_limits():
    beats = [mk("b1", "DIALOGUE", 0.0, 20.5), mk("b2", "DIALOGUE", 20.5, 59.9)]
    s, e, ops = refine_boundaries(beats, 0.0, 59.9, ["b1", "b2"])
    assert 20.0 <= e - s <= 60.0


def test_no_extension_when_it_would_exceed_the_cap():
    beats = [mk("b1", "DIALOGUE", 0.0, 59.8)]
    s, e, ops = refine_boundaries(beats, 0.0, 59.8, ["b1"])
    assert e == 59.8
    assert not any(o.startswith("extend") for o in ops)


def test_candidate_keeps_its_raw_boundaries():
    beats = [mk("b1", "CAMERA", 0.0, 5.0), mk("b2", "DIALOGUE", 5.0, 30.0)]
    out = refine_candidate({"start_s": 0.0, "end_s": 30.0, "beat_ids": ["b2"]}, beats)
    assert out["start_s_raw"] == 0.0 and out["start_s"] == 5.0
    assert out["duration_s"] == 25.6 or out["duration_s"] == 25.0


def test_empty_window_is_returned_untouched():
    s, e, ops = refine_boundaries([], 1.0, 31.0, [])
    assert (s, e, ops) == (1.0, 31.0, [])
