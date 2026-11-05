"""Unit tests for gold set validation and ingestion (tests/test_validate_gold.py).

Verifies with synthetic episodes:
- Boundary duration rules (20.0s <= dur <= 60.0s)
- Start alignment within 0.05s of non-camera beat start_s
- End alignment: Rule 1 (within 0.05s of spoken end), Rule 2 (later <= 0.5s), Rule 3 (episode final end)
- Straddle detection (DIALOGUE / SFX beats)
- Status and source labels ("ai_reviewed" vs "human_audited")
- Exclusion handling for invalid/rejected candidates
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.validate_gold import (
    build_gold_files,
    validate_boundary_rules,
    validate_gold_audit,
)


def create_synthetic_beats() -> List[Dict[str, Any]]:
    """Create a synthetic episode with varied beat types and timings."""
    return [
        {"beat_id": "ep-001", "kind": "CAMERA", "start_s": 0.0, "end_s": 5.0, "text": "Cam 1"},
        {"beat_id": "ep-002", "kind": "ACTION", "start_s": 5.0, "end_s": 15.0, "text": "Act 1"},
        {"beat_id": "ep-003", "kind": "DIALOGUE", "start_s": 15.0, "end_s": 35.0, "text": "Dia 1"},
        {"beat_id": "ep-004", "kind": "SFX", "start_s": 35.0, "end_s": 40.0, "text": "Sfx 1"},
        {"beat_id": "ep-005", "kind": "DIALOGUE", "start_s": 40.0, "end_s": 55.0, "text": "Dia 2"},
        {"beat_id": "ep-006", "kind": "ACTION", "start_s": 55.0, "end_s": 75.0, "text": "Act 2 (Scene closer)"},
    ]


def test_validator_duration_rules():
    """Verify duration bounds [20.0, 60.0]."""
    beats = create_synthetic_beats()
    # Too short: 10s (5.0 to 15.0)
    res_short = validate_boundary_rules(5.0, 15.0, beats, episode_final_end_s=75.0)
    assert res_short["passed"] is False
    assert any("DURATION_OUT_OF_BOUNDS" in f for f in res_short["failures"])

    # Too long: 65s (5.0 to 70.0)
    res_long = validate_boundary_rules(5.0, 70.0, beats, episode_final_end_s=75.0)
    assert res_long["passed"] is False
    assert any("DURATION_OUT_OF_BOUNDS" in f for f in res_long["failures"])

    # Valid duration: 30s (5.0 to 35.0)
    res_valid = validate_boundary_rules(5.0, 35.0, beats, episode_final_end_s=75.0)
    assert res_valid["duration_ok"] is True


def test_validator_start_alignment():
    """Verify start aligns within 0.05s of a non-camera beat start."""
    beats = create_synthetic_beats()
    # Start at 0.0: only CAMERA beat starts at 0.0 -> non-cam starts at 5.0 -> FAIL
    res_cam = validate_boundary_rules(0.0, 35.0, beats, episode_final_end_s=75.0)
    assert res_cam["start_ok"] is False
    assert res_cam["nearest_start_s"] == 5.0

    # Start at 5.02 (within 0.05s of 5.0) -> PASS
    res_close = validate_boundary_rules(5.02, 35.0, beats, episode_final_end_s=75.0)
    assert res_close["start_ok"] is True

    # Start at 8.0 (mid-action beat 5..15) -> FAIL
    res_mid = validate_boundary_rules(8.0, 35.0, beats, episode_final_end_s=75.0)
    assert res_mid["start_ok"] is False


def test_validator_end_rules():
    """Verify the three end rules: exact spoken, later <= 0.5s, or episode final end."""
    beats = create_synthetic_beats()

    # Rule 1: within 0.05s of spoken beat end (35.0)
    r1 = validate_boundary_rules(5.0, 35.02, beats, episode_final_end_s=75.0)
    assert r1["end_ok"] is True
    assert r1["end_rule_matched"] == "RULE_1_SPOKEN_END_EXACT"

    # Rule 2: later by <= 0.5s after spoken beat end (35.30 is 0.30s after 35.0)
    r2 = validate_boundary_rules(5.0, 35.30, beats, episode_final_end_s=75.0)
    assert r2["end_ok"] is True
    assert r2["end_rule_matched"] == "RULE_2_SPOKEN_END_LATER_LE_0_5S"

    # Rule 3: equals episode final end_s (75.0, which ends an ACTION beat)
    r3 = validate_boundary_rules(25.0, 75.0, beats, episode_final_end_s=75.0)
    assert r3["end_ok"] is True
    assert r3["end_rule_matched"] == "RULE_3_EPISODE_FINAL_END"

    # Invalid end: 36.5s (1.5s after 35.0, not episode end)
    r_bad = validate_boundary_rules(5.0, 36.50, beats, episode_final_end_s=75.0)
    assert r_bad["end_ok"] is False
    assert r_bad["end_rule_matched"] is None


def test_validator_straddle_detection():
    """Verify straddle detection blocks cuts through DIALOGUE and SFX beats."""
    beats = create_synthetic_beats()

    # Cut at 25.0 splits DIALOGUE ep-003 (15.0 to 35.0)
    res_split_dialogue = validate_boundary_rules(5.0, 25.0, beats, episode_final_end_s=75.0)
    assert res_split_dialogue["straddle_ok"] is False
    assert "ep-003" in res_split_dialogue["straddles_end"]

    # Cut at 38.0 splits SFX ep-004 (35.0 to 40.0)
    res_split_sfx = validate_boundary_rules(5.0, 38.0, beats, episode_final_end_s=75.0)
    assert res_split_sfx["straddle_ok"] is False
    assert "ep-004" in res_split_sfx["straddles_end"]

    # Cut at 65.0 splits ACTION ep-006 (55.0 to 75.0) -> ACTION is allowed, not DIALOGUE/SFX
    res_split_action = validate_boundary_rules(15.0, 75.0, beats, episode_final_end_s=75.0)
    assert res_split_action["straddle_ok"] is True


def test_validator_after_window_metrics():
    """Verify calculation of spoken beats in the 12s after clip end."""
    beats = create_synthetic_beats()

    # Clip ends at 35.0s. 12s after is 35.0..47.0.
    # Spoken beat ep-005 starts at 40.0 and ends at 55.0 (it falls in 35..47).
    res = validate_boundary_rules(5.0, 35.0, beats, episode_final_end_s=75.0)
    assert res["after_12s_spoken_count"] == 1
    assert res["after_12s_episode_ends"] is False

    # Clip ends at 65.0s. 12s after is 65.0..77.0.
    # Episode ends at 75.0s -> episode_ends_in_12s should be True.
    res_end = validate_boundary_rules(15.0, 75.0, beats, episode_final_end_s=75.0)
    assert res_end["after_12s_episode_ends"] is True


def test_build_gold_files_status_and_source_labels(tmp_path: Path):
    """Verify status and source labeling for empty vs human_confirmed rows."""
    audit_file = tmp_path / "audit.csv"
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    ep_data = {
        "episode": "episode-099",
        "beats": [
            {"beat_id": "b1", "kind": "ACTION", "start_s": 0.0, "end_s": 10.0},
            {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 10.0, "end_s": 35.0},
            {"beat_id": "b3", "kind": "DIALOGUE", "start_s": 35.0, "end_s": 50.0},
        ],
    }
    (parsed_dir / "episode-099.json").write_text(json.dumps(ep_data), encoding="utf-8")

    # Row 1: empty human_confirmed -> ai_reviewed / ai_proposed_ai_reviewed
    # Row 2: human_confirmed=yes -> human_audited / human_audited
    # Row 3: decision=reject -> excluded
    rows = [
        "gold_id,episode,proposed_start_s,proposed_end_s,decision,approved_start_s,approved_end_s,note,human_confirmed",
        "E99_REF001,99,0.0,35.0,accept,0.0,35.0,note1,",
        "E99_REF002,99,10.0,50.0,adjust,10.0,50.0,note2,yes",
        "E99_REF003,99,0.0,15.0,reject,0.0,15.0,note3,",
    ]
    audit_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    results = validate_gold_audit(audit_file=audit_file, parsed_dir=parsed_dir)
    assert len(results) == 3

    gold_csv = tmp_path / "gold.csv"
    gold_public = tmp_path / "gold_public.csv"
    n_written, excluded = build_gold_files(results, gold_csv_path=gold_csv, gold_public_csv_path=gold_public)

    assert n_written == 2
    assert len(excluded) == 1
    assert "E99_REF003" in excluded[0]

    # Inspect gold.csv
    with open(gold_csv, "r", encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))
    assert len(gold_rows) == 2

    # Row 1 check
    r1 = gold_rows[0]
    assert r1["gold_id"] == "E99_REF001"
    assert r1["status"] == "ai_reviewed"
    assert r1["source"] == "ai_proposed_ai_reviewed"
    assert r1["human_confirmed"] == ""

    # Row 2 check
    r2 = gold_rows[1]
    assert r2["gold_id"] == "E99_REF002"
    assert r2["status"] == "human_audited"
    assert r2["source"] == "human_audited"
    assert r2["human_confirmed"] == "yes"

    # Inspect gold_public.csv
    with open(gold_public, "r", encoding="utf-8") as f:
        public_rows = list(csv.DictReader(f))
    assert len(public_rows) == 2
    assert "source" not in public_rows[0]
    assert "human_confirmed" not in public_rows[0]
    assert public_rows[0]["status"] == "ai_reviewed"
    assert public_rows[1]["status"] == "human_audited"

