"""Unit tests for reserved episodes rule (Part B).

Verifies:
- Reserved episodes are blocked and print a one-line notice when gold/gold.csv is missing
- Reserved episodes are blocked when gold/gold.csv is empty (0 bytes)
- Reserved episodes unlock automatically when gold/gold.csv exists and is non-empty
- Non-reserved episodes (e.g. episode-001) are always unblocked
"""

from pathlib import Path

from src.reserved import RESERVED_EPISODES, is_episode_blocked, is_gold_unlocked


def test_reserved_episodes_list():
    """Verify exact 10 reserved episodes."""
    expected = {
        "episode-003",
        "episode-008",
        "episode-013",
        "episode-018",
        "episode-023",
        "episode-028",
        "episode-033",
        "episode-038",
        "episode-043",
        "episode-048",
    }
    assert RESERVED_EPISODES == expected


def test_reserved_rule_locked_when_missing(tmp_path: Path, capsys):
    """Verify reserved episodes are blocked when gold/gold.csv does not exist."""
    missing_gold = tmp_path / "gold" / "gold.csv"
    assert is_gold_unlocked(missing_gold) is False

    # Blocked for reserved episode
    blocked = is_episode_blocked("episode-003", gold_file=missing_gold)
    assert blocked is True
    out = capsys.readouterr().out
    assert "[RESERVED] Skipping reserved gold episode episode-003" in out

    # Unblocked for normal episode
    unblocked = is_episode_blocked("episode-001", gold_file=missing_gold)
    assert unblocked is False


def test_reserved_rule_locked_when_empty(tmp_path: Path, capsys):
    """Verify reserved episodes stay blocked when gold/gold.csv is 0 bytes."""
    empty_gold = tmp_path / "gold.csv"
    empty_gold.touch()  # 0 bytes
    assert is_gold_unlocked(empty_gold) is False

    blocked = is_episode_blocked("episode-048", gold_file=empty_gold)
    assert blocked is True
    out = capsys.readouterr().out
    assert "[RESERVED] Skipping reserved gold episode episode-048" in out


def test_reserved_rule_unlocked_when_nonempty(tmp_path: Path, capsys):
    """Verify reserved episodes unlock automatically when gold/gold.csv is non-empty."""
    active_gold = tmp_path / "gold.csv"
    active_gold.write_text("candidate_id,episode,start_s,end_s\ngold-1,episode-003,10.0,40.0\n", encoding="utf-8")
    assert is_gold_unlocked(active_gold) is True

    blocked = is_episode_blocked("episode-003", gold_file=active_gold)
    assert blocked is False
    out = capsys.readouterr().out
    assert out == ""  # No skip notice when unblocked
