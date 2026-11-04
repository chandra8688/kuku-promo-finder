"""Offline unit tests for Stage 2 Beat Annotation (src/annotate.py).

Verifies:
- Mapping from local indices (1..N) to stable beat_id
- Missing-index detection and repair retry
- Null-fill fallback when repair still misses indices
- camera_context appears in prompt text with [cam: ...]
- Zero timestamps and zero beat_ids appear anywhere in the generated prompt text
All tests run 100% offline with the mock gateway.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from src.annotate import (
    BeatAnnotation,
    EpisodeAnnotation,
    annotate_episode,
    build_episode_input,
    run_annotation_pipeline,
)
from src.llm import ModelGateway, reset_gateway, set_gateway

MOCK_ROLES_CONFIG = {
    "cheap": {
        "provider": "mock",
        "model_name": "mock-cheap",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "rpm_limit": 60,
    },
    "strong": {
        "provider": "mock",
        "model_name": "mock-strong",
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "rpm_limit": 60,
    },
    "judge": {
        "provider": "mock",
        "model_name": "mock-judge",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "rpm_limit": 60,
    },
}


@pytest.fixture(autouse=True)
def setup_mock_gateway(tmp_path: Path):
    """Ensure a fresh mock gateway is active for every test."""
    gw = ModelGateway(
        roles_config=MOCK_ROLES_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        runs_dir=str(tmp_path / "runs"),
        enable_cache=False,
    )
    set_gateway(gw)
    yield gw
    reset_gateway()


def create_synthetic_episode() -> Dict[str, Any]:
    """Create a synthetic episode with 1 CAMERA beat and 3 dramatic beats."""
    return {
        "episode": "episode-synthetic-01",
        "logline": "A secret meeting reveals unexpected betrayal.",
        "continuity_notes": "Nighttime, raining.",
        "beats": [
            {
                "beat_id": "ep99-001",
                "scene": 1,
                "kind": "CAMERA",
                "speaker_raw": None,
                "speaker_norm": None,
                "emotion_label": None,
                "start_s": 0.0,
                "end_s": 2.5,
                "text": "WIDE ANGLE OF RAINY STREET",
                "duration_s": 2.5,
                "gap_prev_spoken_s": None,
                "camera_context": "WIDE SHOT",
                "closing_camera_context": None,
            },
            {
                "beat_id": "ep99-002",
                "scene": 1,
                "kind": "ACTION",
                "speaker_raw": None,
                "speaker_norm": None,
                "emotion_label": None,
                "start_s": 2.5,
                "end_s": 5.0,
                "text": "A shadowed figure slips through the iron gate.",
                "duration_s": 2.5,
                "gap_prev_spoken_s": None,
                "camera_context": "CLOSE UP ON HANDS",
                "closing_camera_context": None,
            },
            {
                "beat_id": "ep99-003",
                "scene": 1,
                "kind": "DIALOGUE",
                "speaker_raw": "MYSTERIOUS STRANGER",
                "speaker_norm": "STRANGER",
                "emotion_label": "urgent",
                "start_s": 5.0,
                "end_s": 8.0,
                "text": "Did you bring the ledger?",
                "duration_s": 3.0,
                "gap_prev_spoken_s": None,
                "camera_context": None,
                "closing_camera_context": "PAN DOWN",
            },
            {
                "beat_id": "ep99-004",
                "scene": 1,
                "kind": "SFX",
                "speaker_raw": "SFX",
                "speaker_norm": "SFX",
                "emotion_label": None,
                "start_s": 8.0,
                "end_s": 9.5,
                "text": "Loud gunshot rings out in the alley.",
                "duration_s": 1.5,
                "gap_prev_spoken_s": 0.0,
                "camera_context": None,
                "closing_camera_context": None,
            },
        ],
    }


def test_build_episode_input_no_timestamps_no_beat_ids():
    """Verify generated prompt text contains NO timestamps and NO beat_ids."""
    ep_data = create_synthetic_episode()
    prompt_text, idx_to_beat_id, non_cam_beats = build_episode_input(ep_data)

    # 1. Non-camera beats must exclude CAMERA
    assert len(non_cam_beats) == 3
    assert all(b["kind"] != "CAMERA" for b in non_cam_beats)

    # 2. Local indices mapping
    assert idx_to_beat_id == {
        1: "ep99-002",
        2: "ep99-003",
        3: "ep99-004",
    }

    # 3. No beat_ids in prompt text
    assert "ep99-001" not in prompt_text
    assert "ep99-002" not in prompt_text
    assert "ep99-003" not in prompt_text
    assert "ep99-004" not in prompt_text
    assert "beat_id" not in prompt_text

    # 4. No timestamps in prompt text
    # Verify mm:ss pattern or decimal seconds pattern do not appear
    timestamp_pattern = re.compile(r"\b\d+:\d{2}(?:\.\d+)?\b|\b\d+\.\d+s?\b")
    matches = timestamp_pattern.findall(prompt_text)
    assert matches == [], f"Found timestamps in prompt text: {matches}"

    # 5. Check camera contexts correctly rendered
    assert "[cam: CLOSE UP ON HANDS]" in prompt_text
    assert "[closing_cam: PAN DOWN]" in prompt_text

    # 6. Check sequential indices
    assert "[1] (Scene 1) ACTION:" in prompt_text
    assert "[2] (Scene 1) DIALOGUE MYSTERIOUS STRANGER:" in prompt_text
    assert "[3] (Scene 1) SFX SFX:" in prompt_text


def test_mapping_1_to_n_to_beat_id():
    """Verify local indices 1..N map accurately back to beat_ids."""
    ep_data = create_synthetic_episode()
    gw = setup_mock_gateway.__wrapped__(Path(".")) if hasattr(setup_mock_gateway, "__wrapped__") else None
    from src.llm import get_gateway
    gw = get_gateway()

    mock_annotation = EpisodeAnnotation(
        summary="A dramatic exchange ends in violence.",
        next_picks_up_immediately=True,
        ends_on_cliffhanger=True,
        beats=[
            BeatAnnotation(i=1, h=2, q=1, r=False, n=0, t=2, e="suspense"),
            BeatAnnotation(i=2, h=3, q=3, r=False, n=1, t=3, e="fear"),
            BeatAnnotation(i=3, h=3, q=3, r=False, n=0, t=3, e="shock"),
        ],
    )
    gw.register_mock_response("cheap", mock_annotation)

    annotated, telemetry = annotate_episode(ep_data, role="cheap", enable_cache=False)

    assert annotated["episode"] == "episode-synthetic-01"
    assert annotated["summary"] == "A dramatic exchange ends in violence."
    assert annotated["next_picks_up_immediately"] is True
    assert annotated["ends_on_cliffhanger"] is True
    assert annotated["partial"] is False

    ann_map = annotated["annotations"]
    assert set(ann_map.keys()) == {"ep99-002", "ep99-003", "ep99-004"}
    assert ann_map["ep99-002"]["h"] == 2
    assert ann_map["ep99-002"]["e"] == "suspense"
    assert ann_map["ep99-003"]["h"] == 3
    assert ann_map["ep99-003"]["q"] == 3
    assert ann_map["ep99-004"]["e"] == "shock"


def test_missing_index_repair_retry_success():
    """Verify repair retry succeeds when initial response is missing indices."""
    ep_data = create_synthetic_episode()
    from src.llm import get_gateway
    gw = get_gateway()

    # Initial response missing index 2 (has only 1 and 3)
    initial_incomplete = EpisodeAnnotation(
        summary="Initial partial",
        next_picks_up_immediately=False,
        ends_on_cliffhanger=True,
        beats=[
            BeatAnnotation(i=1, h=1, q=0, r=False, n=0, t=1, e="calm"),
            BeatAnnotation(i=3, h=3, q=3, r=False, n=0, t=3, e="shock"),
        ],
    )
    # Repaired response with all indices 1, 2, 3
    repaired_full = EpisodeAnnotation(
        summary="Repaired full",
        next_picks_up_immediately=False,
        ends_on_cliffhanger=True,
        beats=[
            BeatAnnotation(i=1, h=1, q=0, r=False, n=0, t=1, e="calm"),
            BeatAnnotation(i=2, h=2, q=2, r=False, n=1, t=2, e="tense"),
            BeatAnnotation(i=3, h=3, q=3, r=False, n=0, t=3, e="shock"),
        ],
    )

    gw.register_mock_response("cheap", initial_incomplete)
    gw.register_mock_response("cheap", repaired_full)

    annotated, telemetry = annotate_episode(ep_data, role="cheap", enable_cache=False)

    assert annotated["partial"] is False
    assert annotated["annotations"]["ep99-003"]["h"] == 2
    assert annotated["annotations"]["ep99-003"]["e"] == "tense"
    assert telemetry["calls"] == 2  # 1 initial + 1 repair


def test_missing_index_null_fill_fallback():
    """Verify persistent missing index triggers null-fill and partial=True."""
    ep_data = create_synthetic_episode()
    from src.llm import get_gateway
    gw = get_gateway()

    # Both initial and repair response persistently omit index 2
    incomplete = EpisodeAnnotation(
        summary="Persistently partial",
        next_picks_up_immediately=False,
        ends_on_cliffhanger=False,
        beats=[
            BeatAnnotation(i=1, h=1, q=1, r=False, n=0, t=1, e="anticipation"),
            BeatAnnotation(i=3, h=3, q=3, r=False, n=0, t=3, e="dread"),
        ],
    )
    gw.register_mock_response("cheap", incomplete)
    gw.register_mock_response("cheap", incomplete)

    annotated, telemetry = annotate_episode(ep_data, role="cheap", enable_cache=False)

    assert annotated["partial"] is True
    ann_map = annotated["annotations"]

    # Index 2 maps to ep99-003; it should be null-filled
    assert ann_map["ep99-003"] == {
        "h": None,
        "q": None,
        "r": None,
        "n": None,
        "t": None,
        "e": None,
    }
    # Index 1 and 3 should be valid
    assert ann_map["ep99-002"]["h"] == 1
    assert ann_map["ep99-004"]["h"] == 3


def test_camera_context_rendering():
    """Verify camera_context and closing_camera_context formatting."""
    ep_data = {
        "episode": "ep-cam",
        "logline": "Logline",
        "continuity_notes": "Continuity",
        "beats": [
            {
                "beat_id": "b-1",
                "scene": 1,
                "kind": "ACTION",
                "start_s": 0.0,
                "end_s": 1.0,
                "text": "Action line",
                "camera_context": "ZOOM IN",
                "closing_camera_context": "CUT TO BLACK",
            },
            {
                "beat_id": "b-2",
                "scene": 1,
                "kind": "DIALOGUE",
                "speaker_raw": "BOB",
                "start_s": 1.0,
                "end_s": 2.0,
                "text": "Hello",
                "camera_context": None,
                "closing_camera_context": None,
            },
        ],
    }
    prompt_text, _, _ = build_episode_input(ep_data)

    assert "[cam: ZOOM IN]" in prompt_text
    assert "[closing_cam: CUT TO BLACK]" in prompt_text
    # b-2 has no cam tags
    lines = prompt_text.splitlines()
    bob_line = [l for l in lines if "DIALOGUE BOB:" in l][0]
    assert "[cam:" not in bob_line
    assert "[closing_cam:" not in bob_line
