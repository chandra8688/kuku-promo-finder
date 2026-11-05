"""Unit tests for terminal-only window inspector (tests/test_show_window.py).

Verifies with synthetic data:
- Banner presence (terminal only, writes nothing to files/logs)
- Normalization of episode ID formats
- Non-camera beat filtering and local indexing (1..N)
- Lookahead 12s window filtering and (+1..+M) indexing
- Camera beat exclusion
- End-of-episode handling

Confidentiality: Uses purely synthetic data fixtures; zero real screenplay text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from scripts.show_window import normalize_episode_id, render_window_display
except ImportError:
    from scripts.archive.show_window import normalize_episode_id, render_window_display


def create_synthetic_parsed_episode() -> Dict[str, Any]:
    """Create synthetic parsed episode structure for testing."""
    return {
        "episode": "episode-999",
        "beats": [
            {"beat_id": "ep999-001", "kind": "CAMERA", "start_s": 0.0, "end_s": 4.0, "speaker_norm": None, "text": "EXT. SYNTHETIC STREET - DAY"},
            {"beat_id": "ep999-002", "kind": "ACTION", "start_s": 4.0, "end_s": 12.0, "speaker_norm": None, "text": "A synthetic figure walks briskly."},
            {"beat_id": "ep999-003", "kind": "DIALOGUE", "start_s": 12.0, "end_s": 24.0, "speaker_norm": "SYNTHETIC_A", "text": "This is synthetic test line one."},
            {"beat_id": "ep999-004", "kind": "CAMERA", "start_s": 24.0, "end_s": 26.0, "speaker_norm": None, "text": "CLOSE UP - SYNTHETIC_A"},
            {"beat_id": "ep999-005", "kind": "DIALOGUE", "start_s": 26.0, "end_s": 38.0, "speaker_norm": "SYNTHETIC_B", "text": "This is synthetic test line two."},
            {"beat_id": "ep999-006", "kind": "SFX", "start_s": 38.0, "end_s": 42.0, "speaker_norm": None, "text": "[SYNTHETIC DOOR SLAM]"},
            {"beat_id": "ep999-007", "kind": "ACTION", "start_s": 42.0, "end_s": 50.0, "speaker_norm": None, "text": "Silence falls across the synthetic room."},
            {"beat_id": "ep999-008", "kind": "DIALOGUE", "start_s": 50.0, "end_s": 58.0, "speaker_norm": "SYNTHETIC_A", "text": "Lookahead test line within 12s."},
            {"beat_id": "ep999-009", "kind": "ACTION", "start_s": 58.0, "end_s": 65.0, "speaker_norm": None, "text": "Beyond the 12s window."},
        ],
    }


def test_normalize_episode_id():
    assert normalize_episode_id("episode-003") == "episode-003"
    assert normalize_episode_id("3") == "episode-003"
    assert normalize_episode_id("003") == "episode-003"
    assert normalize_episode_id("episode-003.json") == "episode-003"
    assert normalize_episode_id("episode-043") == "episode-043"


def test_render_window_display_banner():
    data = create_synthetic_parsed_episode()
    out = render_window_display(
        episode_id="episode-999",
        parsed_data=data,
        start_s=4.0,
        end_s=42.0,
    )
    assert "NOTICE: Terminal output only. Confidential screenplay inspection." in out
    assert "This script writes nothing to any files or logs." in out


def test_render_window_display_beats_and_indexing():
    data = create_synthetic_parsed_episode()
    # Window: 4.0s to 42.0s
    # Expected window beats (non-camera):
    # 1: ep999-002 (ACTION, 4-12)
    # 2: ep999-003 (DIALOGUE, 12-24)
    # (ep999-004 is CAMERA -> excluded)
    # 3: ep999-005 (DIALOGUE, 26-38)
    # 4: ep999-006 (SFX, 38-42)
    # Expected after beats (42.0s to 54.0s):
    # +1: ep999-007 (ACTION, 42-50)
    # +2: ep999-008 (DIALOGUE, 50-58) [starts at 50 < 54]
    out = render_window_display(
        episode_id="episode-999",
        parsed_data=data,
        start_s=4.0,
        end_s=42.0,
    )

    # Check local indices
    assert "[ 1]" in out
    assert "[ 2]" in out
    assert "[ 3]" in out
    assert "[ 4]" in out
    assert "[ 5]" not in out.split("--- 12s AFTER WINDOW")[0]  # only 4 beats in window

    # Camera beat excluded
    assert "CLOSE UP" not in out
    assert "EXT. SYNTHETIC" not in out

    # Check lookahead indices
    assert "[+1]" in out
    assert "[+2]" in out
    assert "Lookahead test line within 12s" in out


def test_render_window_display_end_of_episode():
    data = create_synthetic_parsed_episode()
    # Window extending to end of episode: 42.0s to 65.0s
    out = render_window_display(
        episode_id="episode-999",
        parsed_data=data,
        start_s=42.0,
        end_s=65.0,
    )
    assert "[END OF EPISODE reached directly at window end cut]" in out

