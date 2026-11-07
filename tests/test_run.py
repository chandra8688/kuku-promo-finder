"""End-to-End tests for unified CLI run.py.

Tests end-to-end execution on a synthetic 2-episode folder with mock LLM provider,
verifies output files (promos_per_episode.json/.csv, promos_ranked.json/.csv, run_manifest.json),
verifies --no-llm mode, and verifies unknown header warning handling.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from run import main
from src.llm import ModelGateway, reset_gateway, set_gateway

SYNTHETIC_EP1 = """--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 01:00.00
Actions and dialogue:
**ACTION**
Start: 00:00.00 | End: 00:02.00
A door slams open.

**ALICE (ANGRY)**
Start: 00:02.00 | End: 00:25.00
Did you really think you could deceive us after everything we built?

**BOB (DEFENSIVE)**
Start: 00:25.00 | End: 00:50.00
I never meant for anyone to get hurt! Why won't you listen to me?
--- SCRIPT END ---
"""

SYNTHETIC_EP2 = """--- SCRIPT START ---
SCENE 01
Scene heading: INT. OFFICE - NIGHT
Scene start: 00:00.00
Scene end: 01:10.00
Actions and dialogue:
**SOUND FX**
Start: 00:00.00 | End: 00:03.00
Thunder claps outside the window.

**CAROL (COLD)**
Start: 00:03.00 | End: 00:30.00
The documents are missing from the vault. Who had the key?

**DAN (NERVOUS)**
Start: 00:30.00 | End: 00:58.00
I swear I left it in the desk! Someone must have taken it!
--- SCRIPT END ---
"""


def test_run_synthetic_no_llm(tmp_path: Path):
    """Verify run.py with --no-llm mode processes synthetic episodes with zero model calls."""
    input_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    input_dir.mkdir()

    (input_dir / "ep_syn_1.txt").write_text(SYNTHETIC_EP1, encoding="utf-8")
    (input_dir / "ep_syn_2.txt").write_text(SYNTHETIC_EP2, encoding="utf-8")

    test_args = [
        "run.py",
        "--input", str(input_dir),
        "--out", str(out_dir),
        "--no-llm",
        "--no-cache",
    ]

    with patch("sys.argv", test_args):
        main()

    # Verify output files exist
    assert (out_dir / "promos_per_episode.json").exists()
    assert (out_dir / "promos_per_episode.csv").exists()
    assert (out_dir / "promos_ranked.json").exists()
    assert (out_dir / "promos_ranked.csv").exists()
    assert (out_dir / "run_manifest.json").exists()

    with open(out_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["mode"] == "tier0_no_llm"
    assert manifest["total_calls"] == 0
    assert manifest["episodes_processed"] == 2


def test_run_synthetic_with_mock_llm(tmp_path: Path):
    """Verify run.py with mock provider executes full pipeline end-to-end."""
    input_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    input_dir.mkdir()

    (input_dir / "ep_syn_1.txt").write_text(SYNTHETIC_EP1, encoding="utf-8")
    (input_dir / "ep_syn_2.txt").write_text(SYNTHETIC_EP2, encoding="utf-8")

    gw = ModelGateway(
        roles_config={
            "cheap": {
                "provider": "mock",
                "model_name": "mock-cheap",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "rpm_limit": 60,
            },
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "rpm_limit": 60,
            },
        },
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )
    set_gateway(gw)

    test_args = [
        "run.py",
        "--input", str(input_dir),
        "--out", str(out_dir),
        "--no-cache",
    ]

    with patch("sys.argv", test_args):
        main()

    # Verify per-episode promos
    per_ep_json = json.loads((out_dir / "promos_per_episode.json").read_text(encoding="utf-8"))
    assert len(per_ep_json) >= 2
    for p in per_ep_json:
        assert "episode" in p
        assert "segments" in p
        assert "duration_s" in p
        assert "hook_line" in p
        assert "cliffhanger_line" in p
        assert "scores" in p
        assert "tier" in p

    # Verify show-level ranked promos
    ranked_json = json.loads((out_dir / "promos_ranked.json").read_text(encoding="utf-8"))
    assert "header" in ranked_json
    assert ranked_json["header"].startswith("Show ranking:")
    promos = ranked_json["ranked_promos"]
    assert len(promos) >= 1
    assert promos[0]["show_rank"] == 1


def test_show_level_ranking_order_and_exports(tmp_path: Path):
    """Agreed scores tie, so S3a breaks the tie; low-confidence goes last."""
    from run import build_show_level_top15, HEADER_TEXT

    out_dir = tmp_path / "ranking_test"
    out_dir.mkdir()

    promos = [
        {
            "episode": "episode-001",
            "candidate_id": "ep001_c1",
            "segments_seconds": "[10.0, 35.0]",
            "segments_mmss": "00:10.00-00:35.00",
            "duration_s": 25.0,
            "scores": {"hook": 2, "cliff": 2, "resolved_after": False, "s3a_score": 6.5, "judge_final_score": 10},
            "low_confidence": False,
            "why": "Note 1",
            "hook_line": "Hook 1",
            "cliffhanger_line": "Cliff 1",
            "tier": "standard",
            "provenance": {"annotator_model": "test", "judge_model": "test"},
        },
        {
            "episode": "episode-002",
            "candidate_id": "ep002_c1",
            "segments_seconds": "[20.0, 50.0]",
            "segments_mmss": "00:20.00-00:50.00",
            "duration_s": 30.0,
            "scores": {"hook": 2, "cliff": 2, "resolved_after": False, "s3a_score": 8.5, "judge_final_score": 9},
            "low_confidence": False,
            "why": "Note 2",
            "hook_line": "Hook 2",
            "cliffhanger_line": "Cliff 2",
            "tier": "standard",
            "provenance": {"annotator_model": "test", "judge_model": "test"},
        },
        {
            "episode": "episode-003",
            "candidate_id": "ep003_c1",
            "segments_seconds": "[05.0, 30.0]",
            "segments_mmss": "00:05.00-00:30.00",
            "duration_s": 25.0,
            "scores": {"hook": 2, "cliff": 2, "resolved_after": False, "s3a_score": 9.0, "judge_final_score": 8},
            "low_confidence": True,
            "why": "Note 3",
            "hook_line": "Hook 3",
            "cliffhanger_line": "Cliff 3",
            "tier": "standard",
            "provenance": {"annotator_model": "test", "judge_model": "test"},
        },
    ]

    for p in promos:
        p["scores"]["judged"] = True
        p["scores"]["annotator"] = {"hook": 2, "cliff": 2}
    # ep002 (S3a 8.5) > ep001 (6.5) on equal post score, then ep003 (low_confidence=True)
    ranked_s3a = build_show_level_top15(promos, out_dir=out_dir, no_llm=True)
    assert ranked_s3a[0]["candidate_id"] == "ep002_c1"
    assert ranked_s3a[1]["candidate_id"] == "ep001_c1"
    assert ranked_s3a[2]["candidate_id"] == "ep003_c1"
    assert ranked_s3a[0]["show_rank"] == 1
    assert ranked_s3a[1]["show_rank"] == 2
    assert ranked_s3a[2]["show_rank"] == 3

    # Check generated files
    csv_text = (out_dir / "promos_ranked.csv").read_text(encoding="utf-8")
    assert HEADER_TEXT in csv_text
    json_data = json.loads((out_dir / "promos_ranked.json").read_text(encoding="utf-8"))
    assert json_data["header"] == HEADER_TEXT
    assert json_data["show_rank_config"]["top_n"] == 15
    assert ranked_s3a[2]["below_bar"] == "low_confidence"

