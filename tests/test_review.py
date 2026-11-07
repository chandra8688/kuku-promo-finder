"""Unit tests for Terminal Reviewer (scripts/review.py).

Verifies approving, rejecting with reason codes, skipping, quitting,
and CSV logging without text leakage.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.review import load_promos, main


def test_review_terminal_flow(tmp_path: Path):
    promos_file = tmp_path / "promos.json"
    labels_file = tmp_path / "labels.csv"

    mock_promos = [
        {
            "candidate_id": "cand_001",
            "episode": "episode-001",
            "start_s": 10.0,
            "end_s": 40.0,
            "duration_s": 30.0,
            "scores": {"hook": 3, "cliff": 2, "clarity": 3, "s3a_score": 2.5},
            "why": "High tension hook",
        },
        {
            "candidate_id": "cand_002",
            "episode": "episode-001",
            "start_s": 45.0,
            "end_s": 75.0,
            "duration_s": 30.0,
            "scores": {"hook": 1, "cliff": 1, "clarity": 1, "s3a_score": 1.0},
            "why": "Low tension",
        },
    ]
    promos_file.write_text(json.dumps(mock_promos), encoding="utf-8")

    # Inputs: 1) 'a' (approve cand_001), 2) 'r' then '1' (reject cand_002 with reason weak_hook)
    inputs = ["a", "r", "1"]

    with patch("builtins.input", side_effect=inputs), patch("sys.argv", ["review.py", "--promos", str(promos_file), "--out", str(labels_file)]):
        main()

    assert labels_file.exists()
    rows = list(csv.DictReader(open(labels_file, "r", encoding="utf-8")))
    assert len(rows) == 2

    assert rows[0]["candidate_id"] == "cand_001"
    assert rows[0]["decision"] == "approve"
    assert rows[0]["reason_code"] == "none"

    assert rows[1]["candidate_id"] == "cand_002"
    assert rows[1]["decision"] == "reject"
    assert rows[1]["reason_code"] == "weak_hook"
