import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speaker import classify_speaker_header as c


def test_single_emotion():
    assert c("MAYA (ANGRY):") == ("MAYA", "ANGRY", [])


def test_double_parenthetical_is_the_regression_case():
    assert c("MAYA (O.S.) (COMMANDING):") == ("MAYA", "COMMANDING", ["O.S."])
    assert c("RUDRA (V.O.) (WHISPERING)") == ("RUDRA", "WHISPERING", ["V.O."])


def test_tag_only_header_has_no_emotion():
    assert c("MAYA (O.S.):") == ("MAYA", None, ["O.S."])


def test_bare_name():
    assert c("SHANKAR") == ("SHANKAR", None, [])


def test_three_groups():
    name, emo, mods = c("KAVYA (CONT'D) (O.S.) (BREATHLESS, SHAKING):")
    assert name == "KAVYA"
    assert emo == "BREATHLESS, SHAKING"
    assert mods == ["CONT'D", "O.S."]


def test_case_and_space_normalised():
    assert c("  maya   devi  (angry) ") == ("MAYA DEVI", "ANGRY", [])


def test_alias_applied():
    assert c("MAYA (ANGRY)", {"MAYA": "MAYA DEVI"})[0] == "MAYA DEVI"


def test_empty_parens_ignored():
    assert c("MAYA ()") == ("MAYA", None, [])
