"""Reserved episodes locking and unlocking logic (src/reserved.py).

Episodes 003, 008, 013, 018, 023, 028, 033, 038, 043, 048 are reserved for the
human gold set. They stay blocked across annotation, window search, and candidate
generation until gold/gold.csv exists and is non-empty, then unlock automatically.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Set

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


# The folder run.py is processing. Stages call is_episode_blocked(ep_id) without
# an input folder; without this, the lock fired on episodes 3/13/23/33/43 of ANY
# show on a machine lacking the private gold file and freeze tag.
_SCOPE: dict = {"set": False, "input_dir": None, "enforce": False}


def set_scope(input_dir: Optional[str | Path], enforce_gold_lock: bool = False) -> None:
    """Record the run's input folder so every stage applies the same lock rule."""
    _SCOPE.update(set=True, input_dir=input_dir, enforce=enforce_gold_lock)


def clear_scope() -> None:
    _SCOPE.update(set=False, input_dir=None, enforce=False)


def is_episode_blocked(
    episode_id: str,
    gold_file: str | Path = "gold/gold.csv",
    freeze_tag: str = "config-frozen-before-gold",
    check_freeze_tag: bool = True,
    input_dir: Optional[str | Path] = None,
    enforce_gold_lock: bool = False,
) -> bool:
    """Return True if episode is reserved and locked.

    The reserved-episode lock applies only to the assessment pack (input path
    data/raw or enforce_gold_lock=True); it never blocks external input that
    reuses episode names.
    """
    if input_dir is None and _SCOPE["set"]:
        input_dir = _SCOPE["input_dir"]
        enforce_gold_lock = enforce_gold_lock or _SCOPE["enforce"]
    if input_dir is not None and not enforce_gold_lock:
        norm_input = str(Path(input_dir)).replace("\\", "/")
        if "data/raw" not in norm_input:
            try:
                if Path(input_dir).resolve() != Path("data/raw").resolve():
                    return False
            except Exception:
                return False

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

