"""Unit tests for Tier-0 No-LLM candidate generator (src/tier0.py).

Zero LLM calls. Fully deterministic.
"""

from pathlib import Path
import pytest

from src.tier0 import (
    compute_proxy_hook,
    compute_proxy_cliff,
    compute_proxy_intensity,
    generate_tier0_candidates_for_episode,
    run_tier0_pipeline,
)


def test_proxy_hook_heuristics():
    """Verify heuristic hook scoring."""
    # Beat 1: Dialogue, starts at 0.5s, short line, has exclamation mark
    b1 = {
        "beat_id": "b1",
        "kind": "DIALOGUE",
        "start_s": 0.5,
        "end_s": 3.0,
        "text": "Look out!",
        "duration_s": 2.5,
    }
    hook = compute_proxy_hook(b1, [b1], 0)
    # Starts <= 1.5 (+1), short line (+1), '!' (+1) = 3
    assert hook == 3

    # Beat 2: Action, long line, no punctuation
    b2 = {
        "beat_id": "b2",
        "kind": "ACTION",
        "start_s": 10.0,
        "end_s": 25.0,
        "text": "A very long detailed action sentence describing routine movement without any special punctuation at all.",
        "duration_s": 15.0,
    }
    hook2 = compute_proxy_hook(b2, [b2], 0)
    assert hook2 == 0


def test_proxy_cliff_heuristics():
    """Verify heuristic cliffhanger scoring."""
    # Last beat: ends in question mark, short line, at end of episode (110s of 120s)
    last_beat = {
        "beat_id": "b_last",
        "kind": "DIALOGUE",
        "start_s": 105.0,
        "end_s": 110.0,
        "text": "What have you done?",
        "duration_s": 5.0,
    }
    cliff, resolved = compute_proxy_cliff(last_beat, ep_duration_s=120.0)
    # Ends in ? (+1), short (+1), near end (+1) = 3
    assert cliff == 3
    assert resolved is False


def test_tier0_candidate_generation_clamped_and_clean():
    """Verify generated tier0 candidates obey 20.0 <= duration <= 60.0."""
    beats = [
        {"beat_id": "b1", "kind": "ACTION", "start_s": 0.0, "end_s": 10.0, "text": "Action!"},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 10.0, "end_s": 25.0, "text": "Is this real?"},
        {"beat_id": "b3", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 40.0, "text": "Where are they?"},
        {"beat_id": "b4", "kind": "SFX", "start_s": 40.0, "end_s": 55.0, "text": "THUD!"},
    ]
    parsed_data = {"episode": "ep-synthetic", "beats": beats}
    res = generate_tier0_candidates_for_episode(parsed_data, min_duration_s=20.0, max_duration_s=60.0)

    assert res["episode"] == "ep-synthetic"
    assert res["total_windows_enumerated"] > 0
    for cand in res["candidates"]:
        if cand.get("passed_all_gates"):
            assert 20.0 <= cand["duration_s"] <= 60.0
            assert cand.get("is_tier0") is True


def test_tier0_pipeline_skips_reserved_episodes(tmp_path: Path):
    """Verify Tier-0 generator obeys reserved episode rule."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    # Write a reserved episode file
    with open(parsed_dir / "episode-003.json", "w") as f:
        f.write('{"episode": "episode-003", "beats": []}')

    res = run_tier0_pipeline(
        episodes=["episode-003"],
        parsed_dir=parsed_dir,
        out_dir=tmp_path / "tier0",
        report_file=tmp_path / "rep.json",
        gold_file=tmp_path / "missing_gold.csv",
    )
    # episode-003 is reserved, should be skipped
    assert len(res["episodes"]) == 0

