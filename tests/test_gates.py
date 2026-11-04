"""Offline unit tests for Stage 4 Hard Gates (src/gates.py).

Verifies each gate individually:
1. DURATION: < 20s or > 60s
2. VALID_BEAT_IDS: Non-existent beat ID
3. ORDERED_NONOVERLAPPING_SEGMENTS: Inverted start >= end
4. CUT_STRADDLES_SPOKEN: Split spoken line
5. HOOK_TOO_WEAK: hook < min_hook (default 2)
6. RESOLVED_ENDING: resolved_end is True (resolved cliffhanger)
7. WEAK_CLIFFHANGER: cliff < min_cliff (default 2)
8. DUPLICATE_OVERLAP: IoU > 0.5 against higher-ranked survivor
"""

from src.gates import evaluate_window_gates


def make_sample_episode_beats():
    return [
        {"beat_id": "b1", "kind": "ACTION", "start_s": 0.0, "end_s": 10.0},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 10.0, "end_s": 25.0},
        {"beat_id": "b3", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 40.0},
        {"beat_id": "b4", "kind": "SFX", "start_s": 40.0, "end_s": 50.0},
    ]


def make_clean_candidate():
    return {
        "candidate_id": "clean_cand",
        "beat_ids": ["b1", "b2", "b3"],
        "start_s": 0.0,
        "end_s": 40.0,
        "duration_s": 40.0,
        "hook": 2,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 7.5,
    }


def test_gate_1_duration():
    """Verify DURATION gate rejects windows shorter than 20s or longer than 60s."""
    beats = make_sample_episode_beats()
    short_cand = make_clean_candidate()
    short_cand["duration_s"] = 15.0

    long_cand = make_clean_candidate()
    long_cand["duration_s"] = 65.0

    _, survivors, fail_counts = evaluate_window_gates([short_cand, long_cand], beats)
    assert len(survivors) == 0
    assert fail_counts["DURATION"] == 2


def test_gate_2_valid_beat_ids():
    """Verify VALID_BEAT_IDS gate rejects invalid / non-existent beat IDs."""
    beats = make_sample_episode_beats()
    bad_cand = make_clean_candidate()
    bad_cand["beat_ids"] = ["b1", "non_existent_id"]

    _, survivors, fail_counts = evaluate_window_gates([bad_cand], beats)
    assert len(survivors) == 0
    assert fail_counts["VALID_BEAT_IDS"] == 1


def test_gate_3_ordered_segments():
    """Verify ORDERED_NONOVERLAPPING_SEGMENTS rejects inverted start/end."""
    beats = make_sample_episode_beats()
    inverted_cand = make_clean_candidate()
    inverted_cand["start_s"] = 40.0
    inverted_cand["end_s"] = 10.0

    _, survivors, fail_counts = evaluate_window_gates([inverted_cand], beats)
    assert len(survivors) == 0
    assert fail_counts["ORDERED_NONOVERLAPPING_SEGMENTS"] == 1


def test_gate_4_cut_straddles_spoken():
    """Verify CUT_STRADDLES_SPOKEN rejects cut splitting spoken line."""
    beats = make_sample_episode_beats()
    straddle_cand = make_clean_candidate()
    straddle_cand["cut_straddles_spoken"] = True

    _, survivors, fail_counts = evaluate_window_gates([straddle_cand], beats)
    assert len(survivors) == 0
    assert fail_counts["CUT_STRADDLES_SPOKEN"] == 1


def test_gate_5_hook_too_weak():
    """Verify HOOK_TOO_WEAK rejects hook < 2."""
    beats = make_sample_episode_beats()
    weak_hook = make_clean_candidate()
    weak_hook["hook"] = 1  # < min_hook (2)

    _, survivors, fail_counts = evaluate_window_gates([weak_hook], beats, min_hook=2)
    assert len(survivors) == 0
    assert fail_counts["HOOK_TOO_WEAK"] == 1


def test_gate_6_resolved_ending():
    """Verify RESOLVED_ENDING rejects windows where last beat resolved tension."""
    beats = make_sample_episode_beats()
    resolved_cand = make_clean_candidate()
    resolved_cand["resolved_end"] = True

    _, survivors, fail_counts = evaluate_window_gates([resolved_cand], beats)
    assert len(survivors) == 0
    assert fail_counts["RESOLVED_ENDING"] == 1


def test_gate_7_weak_cliffhanger():
    """Verify WEAK_CLIFFHANGER rejects cliff < 2."""
    beats = make_sample_episode_beats()
    weak_cliff = make_clean_candidate()
    weak_cliff["cliff"] = 1  # < min_cliff (2)

    _, survivors, fail_counts = evaluate_window_gates([weak_cliff], beats, min_cliff=2)
    assert len(survivors) == 0
    assert fail_counts["WEAK_CLIFFHANGER"] == 1


def test_gate_8_duplicate_overlap():
    """Verify DUPLICATE_OVERLAP rejects IoU > 0.5 against higher-scoring survivor."""
    beats = make_sample_episode_beats()
    c1 = make_clean_candidate()
    c1["candidate_id"] = "cand_high"
    c1["score"] = 9.0
    c1["start_s"] = 0.0
    c1["end_s"] = 30.0

    c2 = make_clean_candidate()
    c2["candidate_id"] = "cand_dup"
    c2["score"] = 8.0
    c2["start_s"] = 5.0
    c2["end_s"] = 32.0  # Overlaps c1 with IoU = 25/32 = 0.781 > 0.5

    _, survivors, fail_counts = evaluate_window_gates([c1, c2], beats, max_iou=0.5)
    assert len(survivors) == 1
    assert survivors[0]["candidate_id"] == "cand_high"
    assert fail_counts["DUPLICATE_OVERLAP"] == 1
