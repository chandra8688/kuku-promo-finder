import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def temp_freeze_tag():
    """A real git tag that exists only for the duration of one test.

    The reserved-episode lock is keyed on a git tag. Asserting that
    `config-frozen-before-gold` exists only works in a clone that has the tag,
    and tags are not pushed, so a fresh clone fails. Creating and deleting a
    throwaway tag tests the same logic anywhere.
    """
    name = "pytest-temp-freeze-tag"
    subprocess.run(["git", "tag", "-f", name], capture_output=True, text=True)
    try:
        yield name
    finally:
        subprocess.run(["git", "tag", "-d", name], capture_output=True, text=True)
