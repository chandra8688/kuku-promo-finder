"""Reserved episodes locking and unlocking logic (src/reserved.py).

Episodes 003, 008, 013, 018, 023, 028, 033, 038, 043, 048 are reserved for the
human gold set. They stay blocked across annotation, window search, and candidate
generation until gold/gold.csv exists and is non-empty, then unlock automatically.
"""

from __future__ import annotations

import subprocess
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


def is_freeze_tag_present(tag_name: str = "config-frozen-before-gold") -> bool:
    """Return True if the specified git tag exists in the repository."""
    try:
        res = subprocess.run(
            ["git", "tag", "-l", tag_name],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [t.strip() for t in res.stdout.splitlines() if t.strip()]
        return tag_name in tags
    except Exception:
        return False


def is_gold_unlocked(gold_file: str | Path = "gold/gold.csv") -> bool:
    """Return True if gold/gold.csv exists and is non-empty (>0 bytes)."""
    p = Path(gold_file)
    try:
        return p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def is_episode_blocked(
    episode_id: str,
    gold_file: str | Path = "gold/gold.csv",
    freeze_tag: str = "config-frozen-before-gold",
    check_freeze_tag: bool = True,
) -> bool:
    """Return True if episode is reserved and locked.

    Episodes 003, 008, 013, 018, 023, 028, 033, 038, 043, 048 stay locked until:
    1. The git tag config-frozen-before-gold exists.
    2. gold/gold.csv exists and is non-empty.

    Prints a one-line notice when an episode is skipped.
    """
    stem = Path(episode_id).stem.split("-Episode")[0].strip()
    if stem in RESERVED_EPISODES:
        if check_freeze_tag and not is_freeze_tag_present(freeze_tag):
            print(
                f"[RESERVED] Skipping reserved gold episode {stem}: locked until git tag {freeze_tag} exists."
            )
            return True
        if not is_gold_unlocked(gold_file=gold_file):
            print(
                f"[RESERVED] Skipping reserved gold episode {stem}: locked until {gold_file} exists and is non-empty."
            )
            return True
    return False

