import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import cue_score, window_features


def mk(bid, kind, s, e, text="", spk=None):
    return {"beat_id": bid, "kind": kind, "start_s": s, "end_s": e, "text": text, "speaker_norm": spk}


BEATS = [
    mk("b1", "CAMERA", 0.0, 2.0, "Crash zoom to her face."),
    mk("b2", "SFX", 1.0, 2.2, "A loud slap echoes."),
    mk("b3", "DIALOGUE", 2.2, 6.0, "You lied to me.", "MAYA"),
    mk("b4", "DIALOGUE", 6.2, 12.0, "I had no choice.", "RUDRA"),
    mk("b5", "ACTION", 12.1, 14.0, "She steps back."),
    mk("b6", "DIALOGUE", 14.1, 22.0, "Who told you?", "MAYA"),
    mk("b7", "DIALOGUE", 22.5, 26.0, "It is finished now.", "RUDRA"),
]
ANN = {
    "b1": {"h": 0, "q": 0, "r": False, "n": 0, "t": 1},
    "b2": {"h": 0, "q": 1, "r": False, "n": 0, "t": 2},
    "b3": {"h": 2, "q": 2, "r": False, "n": 1, "t": 2},
    "b4": {"h": 1, "q": 2, "r": False, "n": 0, "t": 2},
    "b5": {"h": 1, "q": 1, "r": False, "n": 0, "t": 2},
    "b6": {"h": 2, "q": 3, "r": False, "n": 0, "t": 3},
    "b7": {"h": 1, "q": 0, "r": True, "n": 0, "t": 1},
}


def test_cue_score_tiers():
    assert cue_score("A loud slap echoes.") == 3
    assert cue_score("He shouts from the hall.") == 2
    assert cue_score("The door closes.") == 1
    assert cue_score("") == 0
    assert cue_score("Nothing notable at all here.") == 0


def test_hook_uses_cue_when_s2_scored_it_zero():
    f = window_features(BEATS, ANN, 1.0, 22.0, ["b2", "b3", "b4", "b5", "b6"], 30.0)
    # b2 has h=0 from S2 but the slap cue carries it
    assert f["hook"] >= 3.0
    assert f["hook_kind"] == "SFX"


def test_camera_hook_is_discounted_not_ignored():
    f = window_features(BEATS, ANN, 0.0, 22.0, None, 30.0)
    assert f["hook"] > 0


def test_cliff_reads_the_tail_not_only_the_last_beat():
    # window ends on b7 which resolves; b6 (q=3) still sits inside the 6s tail
    f = window_features(BEATS, ANN, 2.2, 26.0, ["b3", "b4", "b5", "b6", "b7"], 30.0)
    assert f["cliff_last"] == 0.0
    assert f["cliff"] == 3.0
    assert f["resolved_tail"] is True
    assert f["resolved_last"] is True


def test_intensity_is_duration_weighted():
    f = window_features(BEATS, ANN, 2.2, 22.0, ["b3", "b4", "b5", "b6"], 30.0)
    plain = (2 + 2 + 2 + 3) / 4
    assert f["intensity"] != round(plain, 3)
    assert 2.0 <= f["intensity"] <= 3.0


def test_slope_is_positive_when_tension_rises():
    f = window_features(BEATS, ANN, 2.2, 22.0, ["b3", "b4", "b5", "b6"], 30.0)
    assert f["intensity_slope"] > 0


def test_shares_and_counts():
    f = window_features(BEATS, ANN, 2.2, 22.0, ["b3", "b4", "b5", "b6"], 30.0)
    assert f["speaker_count"] == 2
    assert f["beat_count"] == 4
    assert 0.0 < f["dialogue_share"] <= 1.0
    assert f["tail_pos"] == round(22.0 / 30.0, 3)


def test_missing_annotations_do_not_crash():
    f = window_features(BEATS, {}, 2.2, 22.0, ["b3", "b4", "b5", "b6"], 30.0)
    assert f["cliff"] == 0.0
    assert f["hook"] >= 0.0


def test_none_valued_annotations_are_treated_as_zero():
    ann = {"b3": {"h": None, "q": None, "r": None, "n": None, "t": None}}
    f = window_features(BEATS, ann, 2.2, 22.0, ["b3", "b4"], 30.0)
    assert f["intensity"] == 0.0
