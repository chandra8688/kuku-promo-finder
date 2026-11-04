"""Reserved episodes locking and unlocking logic (src/reserved.py).

Episodes 003, 008, 013, 018, 023, 028, 033, 038, 043, 048 are reserved for the
human gold set. They stay blocked across annotation, window search, and candidate
generation until gold/gold.csv exists and is non-empty, then unlock automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

RESERVED_EPISODES: Set[str] = {
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


def is_gold_unlocked(gold_file: str | Path = "gold/gold.csv") -> bool:
    """Return True if gold/gold.csv exists and is non-empty (>0 bytes)."""
    p = Path(gold_file)
    try:
        return p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def is_episode_blocked(episode_id: str, gold_file: str | Path = "gold/gold.csv") -> bool:
    """Return True if episode is reserved and gold set is not yet unlocked.

    Prints a one-line notice when an episode is skipped.
    """
    # Extract clean episode key e.g. "episode-003"
    stem = Path(episode_id).stem.split("-Episode")[0].strip()
    if stem in RESERVED_EPISODES:
        if not is_gold_unlocked(gold_file=gold_file):
            print(
                f"[RESERVED] Skipping reserved gold episode {stem}: locked until gold/gold.csv exists and is non-empty."
            )
            return True
    return False
