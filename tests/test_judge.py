"""Offline unit tests for Stage 5 Combined Judge (src/judge.py).

All tests use the mock provider via set_gateway / reset_gateway.
Zero network calls.

Tests:
1. Local index mapping — prompt uses 1..N, no beat IDs, no timestamps
2. END OF EPISODE — after_available=False when no AFTER beats exist
3. Variant generation — straddle rule, duration gate, parent_id linkage
4. apply_judge_gates — hook<2, cliff<2, clarity<1, resolved_after=True all rejected
5. Schema with resolved_after=True triggers variant path (integration smoke)
6. Zero timestamps and zero beat IDs in judge prompt text
7. cache_key_extra causes cache miss when value differs
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any

import pytest
from pydantic import BaseModel

from src.llm import ModelGateway, set_gateway, reset_gateway
from src.judge import (
    JudgeOutputFull,
    CandidateEvaluation,
    EpisodeBatchJudgeOutput,
    build_judge_prompt,
    build_episode_batch_judge_prompt,
    apply_judge_gates,
    generate_earlier_cut_variants,
    run_judge_pipeline,
    _verify_no_beat_ids_or_timestamps,
    LOOK_AHEAD_SECONDS,
)
from src.llm import LLMQuotaExhaustedError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_ROLES = {
    "judge": {
        "provider": "mock",
        "model_name": "mock-judge",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "rpm_limit": 60,
    }
}

VALID_JUDGE_JSON = json.dumps({
    "hook": 2, "cliff": 3, "resolved_after": False,
    "clarity": 2, "payoff": 2,
    "hook_index": 1, "cliff_index": 3,
    "why": "Strong opening confrontation ends on unanswered ultimatum."
})

RESOLVED_JUDGE_JSON = json.dumps({
    "hook": 2, "cliff": 2, "resolved_after": True,
    "clarity": 2, "payoff": 2,
    "hook_index": 1, "cliff_index": 2,
    "why": "Clip ends resolved — tension answered immediately after."
})


def make_parsed_data(beats: List[Dict[str, Any]], episode: str = "episode-tst") -> Dict[str, Any]:
    """Minimal parsed episode data structure."""
    return {"episode": episode, "beats": beats}


def make_annotated_data(beat_ids: List[str], annotations: Dict[str, Any] = None) -> Dict[str, Any]:
    """Minimal annotated episode data structure."""
    anns = annotations or {bid: {"h": 2, "q": 2, "r": False, "n": 0, "t": 2, "e": "tense"} for bid in beat_ids}
    return {"annotations": anns}


def make_beats(n: int = 4, start_offset: float = 0.0) -> List[Dict[str, Any]]:
    """Generate n simple DIALOGUE beats starting at start_offset."""
    beats = []
    t = start_offset
    for i in range(1, n + 1):
        beats.append({
            "beat_id": f"episode-tst-{i:03d}",
            "kind": "DIALOGUE",
            "speaker_norm": f"SPK{i}",
            "speaker_raw": f"SPK{i}",
            "emotion_label": "neutral",
            "start_s": t,
            "end_s": t + 10.0,
            "text": f"Line {i} of dialogue.",
            "camera_context": None,
            "closing_camera_context": None,
        })
        t += 10.0
    return beats


# ---------------------------------------------------------------------------
# Test 1: Local index mapping — no beat IDs, no timestamps in prompt
# ---------------------------------------------------------------------------

def test_prompt_uses_local_indices_not_beat_ids():
    """Prompt text must use [1], [2]... indices, never raw beat_id strings or timestamps."""
    beats = make_beats(4)
    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    candidate = {
        "candidate_id": "ep-tst_w0001",
        "episode": "episode-tst",
        "beat_ids": beat_ids[:3],
        "start_s": 0.0,
        "end_s": 30.0,
        "duration_s": 30.0,
    }

    prompt_text, n_beats, after_avail = build_judge_prompt(candidate, parsed, annotated)

    # Must contain local indices [1], [2], [3]
    for idx in range(1, 4):
        assert f"[{idx}]" in prompt_text, f"Missing local index [{idx}] in prompt"

    # Must NOT contain any beat_id strings
    for bid in beat_ids:
        assert bid not in prompt_text, f"Beat ID leaked into prompt: {bid}"

    # Must NOT contain mm:ss timestamp patterns
    assert not re.search(r"\b\d{1,3}:\d{2}(?:\.\d+)?\b", prompt_text), \
        "Timestamp pattern found in judge prompt"

    assert n_beats == 3


# ---------------------------------------------------------------------------
# Test 2: END OF EPISODE when no AFTER beats exist
# ---------------------------------------------------------------------------

def test_after_end_of_episode_when_no_after_beats():
    """When clip includes all beats, AFTER must show [END OF EPISODE] and after_available=False."""
    beats = make_beats(3)  # 3 beats, 0-30s
    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    # Clip includes ALL beats — nothing left in AFTER window
    candidate = {
        "candidate_id": "ep-tst_w0001",
        "episode": "episode-tst",
        "beat_ids": beat_ids,
        "start_s": 0.0,
        "end_s": 30.0,
        "duration_s": 30.0,
    }

    prompt_text, n_beats, after_avail = build_judge_prompt(candidate, parsed, annotated)
    assert after_avail is False
    assert "END OF EPISODE" in prompt_text


# ---------------------------------------------------------------------------
# Test 3: AFTER window populated when beats fall within 12s window
# ---------------------------------------------------------------------------

def test_after_window_populated_within_12s():
    """Beats starting within [clip_end, clip_end+12) appear in AFTER section."""
    beats = make_beats(5)   # 5 beats, 0-50s
    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    # Clip = first 3 beats (0-30s), beat 4 starts at 30s (within 12s window), beat 5 at 40s (outside)
    candidate = {
        "candidate_id": "ep-tst_w0001",
        "episode": "episode-tst",
        "beat_ids": beat_ids[:3],
        "start_s": 0.0,
        "end_s": 30.0,
        "duration_s": 30.0,
    }

    prompt_text, n_beats, after_avail = build_judge_prompt(candidate, parsed, annotated)
    assert after_avail is True
    assert "AFTER" in prompt_text
    assert "END OF EPISODE" not in prompt_text
    # Beat 4 (local index 4) should appear
    assert "[4]" in prompt_text
    # Beat 5 starts at 40s = clip_end(30) + 10s < 12s → actually 10 < 12, should appear
    # Beat 5 starts at 40s, within 30+12=42s window, so both should appear
    assert "[5]" in prompt_text


# ---------------------------------------------------------------------------
# Test 4: apply_judge_gates — each failure mode
# ---------------------------------------------------------------------------

def _make_judge_out(**kwargs) -> JudgeOutputFull:
    defaults = dict(hook=2, cliff=2, resolved_after=False, clarity=2, payoff=2,
                    hook_index=1, cliff_index=2, why="ok")
    defaults.update(kwargs)
    return JudgeOutputFull(**defaults)


def test_judge_gate_hook_too_weak():
    out = _make_judge_out(hook=1)
    passed, reason = apply_judge_gates(out)
    assert not passed
    assert reason == "JUDGE_HOOK_TOO_WEAK"


def test_judge_gate_cliff_too_weak():
    out = _make_judge_out(cliff=1)
    passed, reason = apply_judge_gates(out)
    assert not passed
    assert reason == "JUDGE_CLIFFHANGER_TOO_WEAK"


def test_judge_gate_clarity_too_low():
    out = _make_judge_out(clarity=0)
    passed, reason = apply_judge_gates(out)
    assert not passed
    assert reason == "JUDGE_CLARITY_TOO_LOW"


def test_judge_gate_resolved_after():
    out = _make_judge_out(resolved_after=True)
    passed, reason = apply_judge_gates(out)
    assert not passed
    assert reason == "JUDGE_RESOLVED_AFTER"


def test_judge_gate_passes_clean():
    out = _make_judge_out()
    passed, reason = apply_judge_gates(out)
    assert passed
    assert reason is None


# ---------------------------------------------------------------------------
# Test 5: Variant generation — parent_id linkage, duration gate, straddle
# ---------------------------------------------------------------------------

def test_variant_generation_parent_id_and_duration():
    """Variants must carry parent_id and satisfy 20s <= duration <= 60s."""
    # 5 beats each 10s → total clip 50s
    beats = make_beats(5)
    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    candidate = {
        "candidate_id": "ep-tst_w0001",
        "episode": "episode-tst",
        "beat_ids": beat_ids,
        "start_s": 0.0,
        "end_s": 50.0,
        "duration_s": 50.0,
        "hook": 2, "cliff": 2, "context": 0, "intensity": 2.0, "score": 6.0,
    }

    variants = generate_earlier_cut_variants(candidate, parsed, annotated)

    # All variants must have parent_id == original candidate_id
    for v in variants:
        assert v["parent_id"] == "ep-tst_w0001"
        assert v["is_variant"] is True
        dur = v["duration_s"]
        assert 20.0 <= dur <= 60.0, f"Variant duration {dur}s out of range"


def test_variant_generation_excludes_too_short():
    """Variants shorter than 20s must be excluded."""
    # 3 beats of 7s each → 21s total clip. Trying to cut to 1 beat (7s) must be skipped.
    beats = []
    t = 0.0
    for i in range(1, 4):
        beats.append({
            "beat_id": f"episode-tst-{i:03d}",
            "kind": "DIALOGUE",
            "speaker_norm": f"SPK{i}",
            "speaker_raw": f"SPK{i}",
            "emotion_label": "neutral",
            "start_s": t,
            "end_s": t + 7.0,
            "text": f"Line {i}.",
            "camera_context": None,
            "closing_camera_context": None,
        })
        t += 7.0

    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    candidate = {
        "candidate_id": "ep-tst_w0002",
        "episode": "episode-tst",
        "beat_ids": beat_ids,
        "start_s": 0.0,
        "end_s": 21.0,
        "duration_s": 21.0,
        "hook": 2, "cliff": 2, "context": 0, "intensity": 2.0, "score": 6.0,
    }

    variants = generate_earlier_cut_variants(candidate, parsed, annotated, min_duration_s=20.0)
    for v in variants:
        assert v["duration_s"] >= 20.0


# ---------------------------------------------------------------------------
# Test 6: Verify no timestamps or beat IDs in prompt (_verify helper)
# ---------------------------------------------------------------------------

def test_verify_no_beat_ids_raises_on_leak():
    """_verify_no_beat_ids_or_timestamps must raise if a beat ID appears."""
    with pytest.raises(ValueError, match="Beat ID leaked"):
        _verify_no_beat_ids_or_timestamps("Some text episode-001-003 more", ["episode-001-003"])


def test_verify_no_timestamps_raises_on_leak():
    """_verify_no_beat_ids_or_timestamps must raise if a mm:ss timestamp appears."""
    with pytest.raises(ValueError, match="Timestamp"):
        _verify_no_beat_ids_or_timestamps("At 1:30 something happened", [])


def test_verify_clean_prompt_passes():
    """Clean prompt with neither beat IDs nor timestamps must not raise."""
    _verify_no_beat_ids_or_timestamps("[1] DIALOGUE SPK1: Hello world.", [])


# ---------------------------------------------------------------------------
# Test 7: cache_key_extra causes cache miss
# ---------------------------------------------------------------------------

def test_cache_key_extra_forces_cache_miss(tmp_path: Path):
    """Two calls with different cache_key_extra values must both be cache misses."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=True,
    )
    gw.register_mock_response("judge", "Response A")
    gw.register_mock_response("judge", "Response B")

    # Call 1 with cache_key_extra="run1"
    res1 = gw.call(
        role="judge", system="sys", user="same user prompt",
        stage="test", prompt_version="v1",
        cache_key_extra="run1",
    )
    # Call 2 with cache_key_extra="run2" — different key → cache miss
    res2 = gw.call(
        role="judge", system="sys", user="same user prompt",
        stage="test", prompt_version="v1",
        cache_key_extra="run2",
    )

    assert res1.text == "Response A"
    assert res1.cached is False
    assert res2.text == "Response B"
    assert res2.cached is False


def test_cache_key_extra_same_produces_hit(tmp_path: Path):
    """Same cache_key_extra on identical input must hit cache on second call."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=True,
    )
    gw.register_mock_response("judge", "Response A")
    gw.register_mock_response("judge", "Response B")

    res1 = gw.call(
        role="judge", system="sys", user="same user prompt",
        stage="test", prompt_version="v1",
        cache_key_extra="run1",
    )
    # Identical call — cache hit
    res2 = gw.call(
        role="judge", system="sys", user="same user prompt",
        stage="test", prompt_version="v1",
        cache_key_extra="run1",
    )

    assert res1.text == "Response A"
    assert res1.cached is False
    assert res2.text == "Response A"
    assert res2.cached is True


# ---------------------------------------------------------------------------
# Test 8: Batched per_episode prompt structure & local indexing
# ---------------------------------------------------------------------------

def test_batched_episode_judge_prompt_structure():
    """Verify build_episode_batch_judge_prompt formats candidates with local indices and zero leaks."""
    beats = make_beats(5)
    beat_ids = [b["beat_id"] for b in beats]
    parsed = make_parsed_data(beats)
    annotated = make_annotated_data(beat_ids)

    c1 = {
        "candidate_id": "ep-tst_w0001",
        "episode": "episode-tst",
        "beat_ids": beat_ids[:2],
        "start_s": 0.0,
        "end_s": 20.0,
        "duration_s": 20.0,
    }
    c2 = {
        "candidate_id": "ep-tst_w0002",
        "episode": "episode-tst",
        "beat_ids": beat_ids[1:4],
        "start_s": 10.0,
        "end_s": 40.0,
        "duration_s": 30.0,
    }

    prompt_text, meta_list = build_episode_batch_judge_prompt([c1, c2], parsed, annotated)

    assert "CANDIDATE 1" in prompt_text
    assert "CANDIDATE 2" in prompt_text
    assert len(meta_list) == 2

    # Verify no beat IDs or timestamps leaked
    for bid in beat_ids:
        assert bid not in prompt_text
    assert not re.search(r"\b\d{1,3}:\d{2}(?:\.\d+)?\b", prompt_text)


# ---------------------------------------------------------------------------
# Test 9: Batched per_episode execution with mock provider
# ---------------------------------------------------------------------------

def test_batched_episode_judge_mock_execution(tmp_path: Path):
    """Verify run_judge_pipeline in per_episode mode parses batched evaluations correctly."""
    gw = ModelGateway(
        roles_config={
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "temperature": 0.0,
                "max_output_tokens": 2048,
                "rpm_limit": 60,
                "rpd_limit": 20,
            }
        },
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )
    set_gateway(gw)

    try:
        # Create synthetic parsed, annotated, and candidates files
        parsed_dir = tmp_path / "parsed"
        annotated_dir = tmp_path / "annotated"
        candidates_dir = tmp_path / "candidates"
        judged_dir = tmp_path / "judged"
        parsed_dir.mkdir()
        annotated_dir.mkdir()
        candidates_dir.mkdir()

        ep = "episode-001"
        beats = [
            {"beat_id": f"{ep}-001", "kind": "ACTION", "start_s": 0.0, "end_s": 10.0, "text": "A1"},
            {"beat_id": f"{ep}-002", "kind": "DIALOGUE", "start_s": 10.0, "end_s": 25.0, "text": "D1"},
            {"beat_id": f"{ep}-003", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 40.0, "text": "D2"},
        ]
        with open(parsed_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "beats": beats}, f)
        with open(annotated_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "annotations": {b["beat_id"]: {"h": 2, "q": 2, "r": False, "n": 0, "t": 2, "e": "x"} for b in beats}}, f)

        c1 = {
            "candidate_id": f"{ep}_w0001",
            "episode": ep,
            "beat_ids": [f"{ep}-001", f"{ep}-002"],
            "start_s": 0.0,
            "end_s": 25.0,
            "duration_s": 25.0,
            "passed_all_gates": True,
            "score": 7.0,
        }
        with open(candidates_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "candidates": [c1]}, f)

        # Register mock response for EpisodeBatchJudgeOutput
        mock_output = EpisodeBatchJudgeOutput(
            evaluations=[
                CandidateEvaluation(
                    candidate_num=1,
                    hook=3,
                    cliff=2,
                    resolved_after=False,
                    clarity=2,
                    payoff=2,
                    hook_index=1,
                    cliff_index=2,
                    why="Good hook and unresolved cliff.",
                )
            ]
        )
        gw.register_mock_response("judge", mock_output.model_dump_json())

        result = run_judge_pipeline(
            episodes=[ep],
            parsed_dir=parsed_dir,
            annotated_dir=annotated_dir,
            candidates_dir=candidates_dir,
            judged_dir=judged_dir,
            gold_file=tmp_path / "nonexistent_gold.csv",
            mode="per_episode",
            force=True,
        )

        assert len(result["episodes"]) == 1
        judged = result["episodes"][0]["judged"]
        assert len(judged) == 1
        assert judged[0]["candidate_id"] == f"{ep}_w0001"
        assert judged[0]["hook"] == 3
        assert judged[0]["cliff"] == 2
        assert judged[0]["final_score"] == 3 + 2 + 2 + 2  # 9
        assert judged[0]["passed_judge_gates"] is True

    finally:
        reset_gateway()


# ---------------------------------------------------------------------------
# Test 10: Quota exhaustion error handling
# ---------------------------------------------------------------------------

def test_quota_exhausted_stops_cleanly_without_uncaught_crash(tmp_path: Path):
    """Verify gateway raises LLMQuotaExhaustedError and pipeline stops gracefully."""
    gw = ModelGateway(
        roles_config={
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "rpm_limit": 60,
                "rpd_limit": 20,
            }
        },
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )
    set_gateway(gw)

    try:
        parsed_dir = tmp_path / "parsed"
        annotated_dir = tmp_path / "annotated"
        candidates_dir = tmp_path / "candidates"
        judged_dir = tmp_path / "judged"
        parsed_dir.mkdir()
        annotated_dir.mkdir()
        candidates_dir.mkdir()

        ep = "episode-001"
        beats = [
            {"beat_id": f"{ep}-001", "kind": "ACTION", "start_s": 0.0, "end_s": 25.0, "text": "A1"},
        ]
        with open(parsed_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "beats": beats}, f)
        with open(annotated_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "annotations": {f"{ep}-001": {"h": 2, "q": 2, "r": False, "n": 0, "t": 2, "e": "x"}}}, f)

        c1 = {
            "candidate_id": f"{ep}_w0001",
            "episode": ep,
            "beat_ids": [f"{ep}-001"],
            "start_s": 0.0,
            "end_s": 25.0,
            "duration_s": 25.0,
            "passed_all_gates": True,
            "score": 7.0,
        }
        with open(candidates_dir / f"{ep}.json", "w") as f:
            json.dump({"episode": ep, "candidates": [c1]}, f)

        # Queue a daily quota exhaustion exception
        gw.register_mock_response("judge", LLMQuotaExhaustedError("Simulated daily quota exhausted"))

        # In per_candidate mode, the exception should be caught cleanly
        res = run_judge_pipeline(
            episodes=[ep],
            parsed_dir=parsed_dir,
            annotated_dir=annotated_dir,
            candidates_dir=candidates_dir,
            judged_dir=judged_dir,
            gold_file=tmp_path / "nonexistent_gold.csv",
            force=True,
        )
        # Did not crash with uncaught exception
        assert isinstance(res, dict)
    finally:
        reset_gateway()


