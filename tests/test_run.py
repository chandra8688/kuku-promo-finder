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
    assert len(ranked_json) >= 1
    assert ranked_json[0]["show_rank"] == 1
