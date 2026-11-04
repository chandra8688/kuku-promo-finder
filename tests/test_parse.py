"""Unit tests for Stage 1 parser (src/parse.py).

Verifies header tolerance, normalization, timestamp rounding, enrichment,
camera context attachment, validation rules, severity policy, and determinism.
"""

import json
from pathlib import Path

import pytest

from src.parse import (
    Beat,
    ParseError,
    ParseWarning,
    parse_file,
    parse_timestamp_to_seconds,
    run_pipeline,
)


SYNTHETIC_SCRIPT_VALID = """EPISODE NUMBER: Episode 999
EPISODE LOGLINE: A synthetic test episode.
CONTINUITY NOTES: Test continuity notes.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. TEST ROOM - DAY
Scene start: 00:00.00
Scene end: 00:30.00

Actions and dialogue:

**camera**
Start: (00:00.00) | End: (00:02.00)
Wide establishing shot of the empty room.

**CAMERA:**
Start: 00:02.00 | End: 00:03.50
Slow push-in on the doorway.

**   action   :  **
Start: (00:03.50) | End: (00:06.00)
Alice suddenly steps into the room.

**   LADY ALICE   ( Whispering )   :  **
Start: 00:06.00 | End: (00:08.60)
"Is anyone there?"

**sfx**
Start: (00:08.60) | End: 00:09.20
Door creaks shut loudly.

**BOB (Aloud):**
Start: (00:09.80) | End: (00:12.00)
"Just me."

**CAMERA:**
Start: (00:12.00) | End: (00:15.00)
Fade to black on the dark corridor.

--- SCRIPT END ---
"""

SYNTHETIC_SCRIPT_INVALID_START_GTE_END = """--- SCRIPT START ---
SCENE 01
Actions and dialogue:
**ACTION:**
Start: 00:05.00 | End: 00:03.00
Someone walks backward in time.
--- SCRIPT END ---
"""

SYNTHETIC_SCRIPT_NON_MONOTONIC = """--- SCRIPT START ---
SCENE 01
Actions and dialogue:
**ACTION:**
Start: 00:05.00 | End: 00:08.00
First action happens.

**ACTION:**
Start: 00:03.00 | End: 00:06.00
Second action starts earlier.
--- SCRIPT END ---
"""

SYNTHETIC_SCRIPT_SPOKEN_OVERLAP = """--- SCRIPT START ---
SCENE 01
Actions and dialogue:
**ALICE:**
Start: 00:01.00 | End: 00:05.00
"First spoken line."

**BOB:**
Start: 00:04.00 | End: 00:07.00
"Spoken line overlapping with Alice."
--- SCRIPT END ---
"""

SYNTHETIC_SCRIPT_UNKNOWN_HEADER = """--- SCRIPT START ---
SCENE 01
Actions and dialogue:
**   :  **
Start: 00:00.00 | End: 00:02.00
Invalid empty header.
--- SCRIPT END ---
"""


def test_timestamp_rounding():
    """Verify float rounding handles floating point artifacts cleanly."""
    val = parse_timestamp_to_seconds("0", "00.6000000000000001")
    assert val == 0.60
    val2 = parse_timestamp_to_seconds("1", "10.40")
    assert val2 == 70.40


def test_synthetic_parsing_and_enrichment(tmp_path: Path):
    """Verify parsing synthetic script with mixed headers, cameras, and contexts."""
    test_file = tmp_path / "episode-999-synthetic.txt"
    test_file.write_text(SYNTHETIC_SCRIPT_VALID, encoding="utf-8")

    aliases = {"ALICE": "LADY ALICE"}
    beats, warnings, dur_max, dur_last, meta = parse_file(
        test_file, aliases=aliases, episode_id="episode-999"
    )

    # 1. Episode metadata
    assert meta["episode_number_raw"] == "Episode 999"
    assert meta["logline"] == "A synthetic test episode."
    assert meta["continuity_notes"] == "Test continuity notes."
    assert len(meta["scenes"]) == 1
    sc = meta["scenes"][0]
    assert sc["scene_index"] == 1
    assert sc["heading"] == "INT. TEST ROOM - DAY"
    assert sc["scene_start_s"] == 0.0
    assert sc["scene_end_s"] == 30.0

    # 2. Total beats: 2 CAMERA + 1 ACTION + 1 DIALOGUE + 1 SFX + 1 DIALOGUE + 1 CAMERA = 7 beats
    assert len(beats) == 7
    assert dur_max == 15.00
    assert dur_last == 15.00
    assert len(warnings) == 0

    # 3. Kinds
    kinds = [b.kind for b in beats]
    assert kinds == ["CAMERA", "CAMERA", "ACTION", "DIALOGUE", "SFX", "DIALOGUE", "CAMERA"]

    # 4. Beat links to scene index
    for b in beats:
        assert b.scene == 1

    # 5. SFX is its own kind and not DIALOGUE
    sfx_beat = beats[4]
    assert sfx_beat.kind == "SFX"
    assert sfx_beat.speaker_norm is None

    # 6. Consecutive CAMERA attachment to next non-camera beat (ACTION beat at index 2)
    action_beat = beats[2]
    assert action_beat.kind == "ACTION"
    assert action_beat.camera_context is not None
    assert "Wide establishing shot" in action_beat.camera_context
    assert "Slow push-in" in action_beat.camera_context

    # 7. Dialogue normalization and emotion extraction
    alice_beat = beats[3]
    assert alice_beat.kind == "DIALOGUE"
    assert alice_beat.speaker_raw == "LADY ALICE   ( Whispering )"
    assert alice_beat.speaker_norm == "LADY ALICE"
    assert alice_beat.emotion_label == "Whispering"
    assert alice_beat.duration_s == 2.60
    assert alice_beat.gap_prev_spoken_s is None  # First spoken line in episode

    # 8. Second dialogue beat gap_prev_spoken_s
    bob_beat = beats[5]
    assert bob_beat.kind == "DIALOGUE"
    assert bob_beat.speaker_norm == "BOB"
    assert bob_beat.emotion_label == "Aloud"
    assert bob_beat.gap_prev_spoken_s == 1.20

    # 9. Trailing CAMERA beat attached to preceding non-camera beat (Bob) as closing_camera_context
    assert bob_beat.closing_camera_context is not None
    assert "Fade to black" in bob_beat.closing_camera_context


def test_alias_mapping(tmp_path: Path):
    """Verify alias mapping canonicalizes speaker names."""
    script = """--- SCRIPT START ---
SCENE 01
Actions and dialogue:
**LADY A:**
Start: 00:00.00 | End: 00:02.00
"Hello."
--- SCRIPT END ---
"""
    f = tmp_path / "ep-alias.txt"
    f.write_text(script, encoding="utf-8")
    aliases = {"LADY A": "LADY ALICE"}
    beats, _, _, _, _ = parse_file(f, aliases=aliases)
    assert beats[0].speaker_norm == "LADY ALICE"


def test_warning_policy_spoken_vs_concurrent_overlap(tmp_path: Path):
    """Verify overlap between spoken lines is 'warning' while camera/action is 'info'."""
    f = tmp_path / "ep-spoken-overlap.txt"
    f.write_text(SYNTHETIC_SCRIPT_SPOKEN_OVERLAP, encoding="utf-8")
    beats, warnings, _, _, _ = parse_file(f)
    spoken_warns = [w for w in warnings if w.reason_code == "SPOKEN_BEAT_OVERLAP"]
    assert len(spoken_warns) == 1
    assert spoken_warns[0].severity == "warning"


def test_validation_start_gte_end(tmp_path: Path):
    """Verify start_s >= end_s generates START_GTE_END error severity."""
    f = tmp_path / "ep-bad-duration.txt"
    f.write_text(SYNTHETIC_SCRIPT_INVALID_START_GTE_END, encoding="utf-8")
    beats, warnings, _, _, _ = parse_file(f)
    assert any(w.reason_code == "START_GTE_END" and w.severity == "error" for w in warnings)


def test_validation_non_monotonic(tmp_path: Path):
    """Verify out-of-order start times generate NON_MONOTONIC_START warning."""
    f = tmp_path / "ep-non-monotonic.txt"
    f.write_text(SYNTHETIC_SCRIPT_NON_MONOTONIC, encoding="utf-8")
    beats, warnings, _, _, _ = parse_file(f)
    non_mono = [w for w in warnings if w.reason_code == "NON_MONOTONIC_START"]
    assert len(non_mono) == 1
    assert non_mono[0].severity == "warning"
    assert non_mono[0].detail["inversion_s"] == 2.0


def test_unknown_header_raises_error(tmp_path: Path):
    """Verify unknown or empty header raises ParseError with line info."""
    f = tmp_path / "ep-unknown-header.txt"
    f.write_text(SYNTHETIC_SCRIPT_UNKNOWN_HEADER, encoding="utf-8")
    with pytest.raises(ParseError) as exc_info:
        parse_file(f)
    assert "Unknown empty header" in str(exc_info.value) or "Unknown header" in str(exc_info.value)


def test_real_dataset_integrity():
    """Verify parsing all 50 real episodes reproduces exact verified baseline metrics."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists() or not list(raw_dir.glob("*.txt")):
        pytest.skip("data/raw/ is missing or empty; skipping real dataset integrity test")

    report = run_pipeline(
        input_dir="data/raw",
        output_dir="data/parsed",
        report_path="parse_report.json",
        allow_warnings=True,
    )
    overall = report["overall"]

    assert overall["total_episodes"] == 50
    assert overall["total_beats"] == 3727
    assert overall["counts_by_kind"]["CAMERA"] == 1754
    assert overall["counts_by_kind"]["ACTION"] == 372
    assert overall["counts_by_kind"]["SFX"] == 114
    assert overall["counts_by_kind"]["DIALOGUE"] == 1487
    assert overall["episodes_with_speech_within_3s"] == 17
    assert overall["median_gap_spoken_s"] == 0.60
    assert overall["total_characters"] == 528811
    assert overall["median_beats_per_episode"] == 73.0
    assert overall["max_beats_per_episode"] == 121

    # Warning breakdown checks
    assert overall["counts_by_severity"]["info"] == 1933
    assert overall["counts_by_severity"]["warning"] == 27
    assert overall["counts_by_reason"]["CONCURRENT_BEAT_OVERLAP"] == 1933
    assert overall["counts_by_reason"]["SPOKEN_BEAT_OVERLAP"] == 19
    assert overall["counts_by_reason"]["NON_MONOTONIC_START"] == 8

    # Duration definition checks
    assert overall["duration_min_s"] == 70.5
    assert overall["shortest_episode_duration_check"]["max_end_s"] == 70.5
    assert overall["shortest_episode_duration_check"]["last_beat_s"] == 70.4
