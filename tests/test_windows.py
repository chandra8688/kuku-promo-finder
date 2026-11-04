"""Offline unit tests for Stage 3a Window Search (src/windows.py).

Verifies:
- Units are non-camera beats only (C1)
- Window duration is kept within 20.0 <= dur <= 60.0 (C2)
- Straddle cut rule: external DIALOGUE/SFX invalidates, ACTION is recorded (C3)
- Features and scoring from annotations (C4, C5)
- Greedy IoU deduplication (C7)
"""

from src.windows import (
    check_straddle_cuts,
    compute_interval_iou,
    deduplicate_windows_greedy,
    enumerate_windows_for_episode,
)


def make_synthetic_parsed_data():
    return {
        "episode": "ep-test-windows",
        "beats": [
            # 0. Camera
            {
                "beat_id": "ep-b00",
                "scene": 1,
                "kind": "CAMERA",
                "start_s": 0.0,
                "end_s": 5.0,
                "text": "WIDE VIEW",
            },
            # 1. Action
            {
                "beat_id": "ep-b01",
                "scene": 1,
                "kind": "ACTION",
                "start_s": 5.0,
                "end_s": 20.0,
                "text": "Action runs",
            },
            # 2. Dialogue
            {
                "beat_id": "ep-b02",
                "scene": 1,
                "kind": "DIALOGUE",
                "start_s": 20.0,
                "end_s": 35.0,
                "text": "Question asked",
            },
            # 3. Action
            {
                "beat_id": "ep-b03",
                "scene": 1,
                "kind": "ACTION",
                "start_s": 35.0,
                "end_s": 45.0,
                "text": "Confrontation",
            },
            # 4. Spoken line running across 40.0 to 50.0 (overlaps with b03)
            {
                "beat_id": "ep-b04",
                "scene": 1,
                "kind": "DIALOGUE",
                "start_s": 40.0,
                "end_s": 50.0,
                "text": "Spoken overlap",
            },
        ],
    }


def make_synthetic_annotated_data():
    return {
        "episode": "ep-test-windows",
        "annotations": {
            "ep-b01": {"h": 3, "q": 1, "r": False, "n": 0, "t": 2, "e": "shock"},
            "ep-b02": {"h": 2, "q": 3, "r": False, "n": 1, "t": 3, "e": "fear"},
            "ep-b03": {"h": 1, "q": 2, "r": True, "n": 0, "t": 2, "e": "calm"},
            "ep-b04": {"h": 0, "q": 1, "r": False, "n": 0, "t": 1, "e": "relief"},
        },
    }


def test_units_exclude_camera():
    """Verify camera beats are never window members."""
    p_data = make_synthetic_parsed_data()
    a_data = make_synthetic_annotated_data()

    windows = enumerate_windows_for_episode(
        parsed_data=p_data,
        annotated_data=a_data,
        min_duration_s=20.0,
        max_duration_s=60.0,
    )

    for w in windows:
        assert "ep-b00" not in w["beat_ids"], "CAMERA beat ep-b00 found in window!"


def test_window_durations_clamped():
    """Verify all enumerated windows satisfy 20.0 <= duration <= 60.0."""
    p_data = make_synthetic_parsed_data()
    a_data = make_synthetic_annotated_data()

    windows = enumerate_windows_for_episode(
        parsed_data=p_data,
        annotated_data=a_data,
        min_duration_s=20.0,
        max_duration_s=60.0,
    )

    assert len(windows) > 0
    for w in windows:
        assert 20.0 <= w["duration_s"] <= 60.0


def test_straddle_cut_rule():
    """Verify cut straddling detection for DIALOGUE vs ACTION."""
    all_beats = [
        {"beat_id": "b1", "kind": "ACTION", "start_s": 10.0, "end_s": 25.0},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 35.0},
        {"beat_id": "b3_action", "kind": "ACTION", "start_s": 30.0, "end_s": 40.0},
        {"beat_id": "b4_spoken", "kind": "DIALOGUE", "start_s": 33.0, "end_s": 42.0},
    ]

    # Window covers [b1, b2]: start=10.0, end=35.0
    # b3_action spans 30.0 to 40.0: straddles end cut 35.0 -> cut_straddles_action = True
    # b4_spoken spans 33.0 to 42.0: straddles end cut 35.0 -> cut_straddles_spoken = True
    spoken, action = check_straddle_cuts(
        start_cut=10.0,
        end_cut=35.0,
        all_beats=all_beats,
        window_beat_ids={"b1", "b2"},
    )
    assert spoken is True
    assert action is True


def test_features_and_scoring():
    """Verify hook, cliff, resolved_end, context, intensity, and score calculation."""
    p_data = make_synthetic_parsed_data()
    a_data = make_synthetic_annotated_data()

    # Window from b01 to b02:
    # b01: start=5.0, end=20.0; b02: start=20.0, end=35.0 -> start=5.0, end=35.0, dur=30.0
    windows = enumerate_windows_for_episode(
        parsed_data=p_data,
        annotated_data=a_data,
        weights={"hook": 1.0, "cliff": 1.0, "intensity": 1.0, "context": 1.0},
        min_duration_s=20.0,
        max_duration_s=40.0,
    )

    w_b1_b2 = [w for w in windows if w["beat_ids"] == ["ep-b01", "ep-b02"]][0]
    assert w_b1_b2["hook"] == 3  # h of ep-b01
    assert w_b1_b2["cliff"] == 3  # q of ep-b02
    assert w_b1_b2["context"] == 0  # n of ep-b01
    assert w_b1_b2["resolved_end"] is False  # r of ep-b02
    # Intensity: (2 + 3) / 2 = 2.5
    assert w_b1_b2["intensity"] == 2.5
    # Score: 3 + 3 + 2.5 - 0 = 8.5
    assert w_b1_b2["score"] == 8.5


def test_compute_interval_iou():
    """Verify 1D interval IoU computation."""
    # Complete non-overlap
    assert compute_interval_iou(0.0, 10.0, 20.0, 30.0) == 0.0
    # Exact match
    assert compute_interval_iou(10.0, 30.0, 10.0, 30.0) == 1.0
    # Partial overlap: [10, 30] and [20, 40] -> intersection=10, union=30 -> 1/3
    assert round(compute_interval_iou(10.0, 30.0, 20.0, 40.0), 3) == 0.333
    # Overlap over 50%: [10, 40] and [15, 45] -> intersection=25, union=35 -> 25/35 = 0.714
    assert compute_interval_iou(10.0, 40.0, 15.0, 45.0) > 0.5


def test_greedy_deduplication():
    """Verify greedy selection keeps top K while filtering IoU > 0.5."""
    candidates = [
        {"candidate_id": "c1", "start_s": 10.0, "end_s": 40.0, "duration_s": 30.0, "score": 9.0, "cliff": 3},
        {"candidate_id": "c2", "start_s": 12.0, "end_s": 42.0, "duration_s": 30.0, "score": 8.5, "cliff": 2},  # overlaps c1 heavily (IoU > 0.5)
        {"candidate_id": "c3", "start_s": 50.0, "end_s": 80.0, "duration_s": 30.0, "score": 8.0, "cliff": 2},  # clean distinct window
    ]

    kept = deduplicate_windows_greedy(candidates, top_k=3, max_iou=0.5)
    kept_ids = [k["candidate_id"] for k in kept]
    assert kept_ids == ["c1", "c3"], f"Expected ['c1', 'c3'] but got {kept_ids}"

