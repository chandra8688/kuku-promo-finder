"""Unit test for screenplay confidentiality leak check v2.

Skips when data/raw is missing.
Fails if any tracked file contains:
1. Matching 8-word shingles from raw screenplays (excluding structural boilerplate)
2. Character/setting names from speaker_norm or loglines (name denylist)

Reports file names and match counts only; never displays matched text or names.
"""

from pathlib import Path

import pytest

from scripts.leak_check import run_checks


def test_no_raw_screenplay_shingles_or_names_in_tracked_files():
    """Verify zero raw shingles and zero denylist names exist in tracked files."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists() or not list(raw_dir.glob("*.txt")):
        pytest.skip("data/raw missing or contains no .txt files")

    shingle_matches, name_matches, stats = run_checks(
        raw_dir="data/raw",
        parsed_dir="data/parsed",
    )

    assert shingle_matches == {}, f"Shingle leak check failed: matching shingles found in {shingle_matches}"
    assert name_matches == {}, f"Name denylist check failed: matching names found in {name_matches}"
