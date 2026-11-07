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
9. DUPLICATE_ANCHOR: Shared first or last beat ID with higher-ranked survivor
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
    """Verify HOOK_TOO_WEAK is a soft flag: survivor kept with flag set."""
    beats = make_sample_episode_beats()
    weak_hook = make_clean_candidate()
    weak_hook["hook"] = 1  # < min_hook (2)

    _, survivors, fail_counts = evaluate_window_gates([weak_hook], beats, min_hook=2)
    assert len(survivors) == 1
    assert survivors[0]["gate_results"]["HOOK_TOO_WEAK"] is False
    assert fail_counts["HOOK_TOO_WEAK"] == 1


def test_gate_6_resolved_ending():
    """Verify RESOLVED_ENDING is a soft flag: survivor kept with flag set."""
    beats = make_sample_episode_beats()
    resolved_cand = make_clean_candidate()
    resolved_cand["resolved_end"] = True

    _, survivors, fail_counts = evaluate_window_gates([resolved_cand], beats)
    assert len(survivors) == 1
    assert survivors[0]["gate_results"]["RESOLVED_ENDING"] is False
    assert fail_counts["RESOLVED_ENDING"] == 1


def test_gate_7_weak_cliffhanger():
    """Verify WEAK_CLIFFHANGER is a soft flag: survivor kept with flag set."""
    beats = make_sample_episode_beats()
    weak_cliff = make_clean_candidate()
    weak_cliff["cliff"] = 1  # < min_cliff (2)

    _, survivors, fail_counts = evaluate_window_gates([weak_cliff], beats, min_cliff=2)
    assert len(survivors) == 1
    assert survivors[0]["gate_results"]["WEAK_CLIFFHANGER"] is False
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


def test_gate_9_duplicate_anchor_shared_first_beat():
    """Verify DUPLICATE_ANCHOR rejects candidate sharing first beat ID with higher-scored survivor.

    Design: c1 covers [0..21s] (beats b1,b2), c2 covers [0..58s] (beats b1,b2,b3,b4).
    They share first beat b1 but IoU = 21/58 ≈ 0.36 < 0.5, so IoU gate does NOT fire.
    DUPLICATE_ANCHOR must reject c2.
    """
    # Larger beat set so we can construct two non-overlapping-IoU windows
    beats = [
        {"beat_id": "b1", "kind": "ACTION",   "start_s":  0.0,  "end_s": 10.0},
        {"beat_id": "b2", "kind": "DIALOGUE",  "start_s": 10.0,  "end_s": 21.0},
        {"beat_id": "b3", "kind": "DIALOGUE",  "start_s": 21.0,  "end_s": 42.0},
        {"beat_id": "b4", "kind": "SFX",       "start_s": 42.0,  "end_s": 58.0},
    ]

    # c1 scores higher, wins, anchors b1 as first beat
    c1 = {
        "candidate_id": "cand_anchor_high",
        "beat_ids": ["b1", "b2"],
        "start_s": 0.0,
        "end_s": 21.0,
        "duration_s": 21.0,
        "hook": 2,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 9.0,
    }
    # c2 starts at same first beat (b1), but spans much longer → IoU ≈ 21/58 ≈ 0.36 < 0.5
    c2 = {
        "candidate_id": "cand_anchor_dup_first",
        "beat_ids": ["b1", "b2", "b3", "b4"],
        "start_s": 0.0,
        "end_s": 58.0,
        "duration_s": 58.0,
        "hook": 2,
        "cliff": 2,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 6.0,
    }

    _, survivors, fail_counts = evaluate_window_gates([c1, c2], beats, max_iou=0.5)
    assert len(survivors) == 1
    assert survivors[0]["candidate_id"] == "cand_anchor_high"
    assert fail_counts["DUPLICATE_ANCHOR"] == 1
    assert c2["first_failed_gate"] == "DUPLICATE_ANCHOR"


def test_gate_9_duplicate_anchor_shared_last_beat():
    """Verify DUPLICATE_ANCHOR rejects candidate sharing last beat ID with higher-scored survivor.

    Design: c1 covers [b2,b3] (10..58s), c2 covers [b3] only (21..58s?).
    Wait — c2 must be ≥20s duration. Let me use b3+b4 = 21..58 = 37s.
    c1 covers b1..b4 = [0..58s] (58s), c2 covers [b3,b4] = [21..58s] (37s).
    Both end on b4 (last beat).
    IoU = (58-21) / (58-0) = 37/58 ≈ 0.638 > 0.5 → IoU fires first. Need different layout.

    Better: c1=[b1,b2]=[0..21s], c2=[b3,b2]? Not monotonic. Use separate beat sets.
    Design with beats that allow IoU < 0.5 while sharing last beat:
    c1=[0..22s] beats=[b1,b2] (last=b2), c2=[2..22s]=[b3_start2..b2] – not clean.

    Use: b2 ends at 22s. c1=[b1,b2]=[0..22s], c2=[b_new, b2] where b_new=[1..10s].
    But c2=[1..22s]=[21s duration], IoU=(22-1)/(22-0)=21/22≈0.95>0.5.

    Cleanest solution: separate the episodes so the LAST beat anchor test uses
    different windows where start is far apart but last beat is same.
    c1=[b1,b2]=[0..22s], c2=[b3,b4,b2]? Needs non-monotonic check...

    Use beats: b1[0-5], b2[5-22], b3[22-50], b4[50-72 → cap 60s but not needed].
    c1=[b1,b2]=[0..22s=22s duration, scores high]. Last beat=b2.
    c2=[b3,b2]? Beats must be monotonic. Not possible.

    Correct approach: beats where there is a beat shared as last but windows are far apart.
    beat_ids must be ordered by start_s. So 'last beat shared' means two windows
    both end on the same beat (same beat_id as final entry). They must be monotonic.
    If c1=[b1,b2,b3] and c2=[b2,b3], then c2 is a suffix of c1.
    c1=[0..45s], c2=[10..45s], IoU=(45-10)/(45-0)=35/45=0.78>0.5.

    The only way IoU<0.5 with shared last beat: c1 is very long and c2 is a small suffix.
    E.g. c1=[0..60s], c2=[40..60s] (20s min). IoU=20/60=0.33<0.5. ✓
    """
    # Beat layout: b1[0-10], b2[10-20], b3[20-40], b4[40-60]
    beats = [
        {"beat_id": "b1", "kind": "ACTION",   "start_s":  0.0,  "end_s": 10.0},
        {"beat_id": "b2", "kind": "DIALOGUE",  "start_s": 10.0,  "end_s": 20.0},
        {"beat_id": "b3", "kind": "DIALOGUE",  "start_s": 20.0,  "end_s": 40.0},
        {"beat_id": "b4", "kind": "SFX",       "start_s": 40.0,  "end_s": 60.0},
    ]

    # c1 spans the full 60s, last beat = b4 (scores higher)
    c1 = {
        "candidate_id": "cand_anchor_high",
        "beat_ids": ["b1", "b2", "b3", "b4"],
        "start_s": 0.0,
        "end_s": 60.0,
        "duration_s": 60.0,
        "hook": 2,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 9.0,
    }
    # c2 is a suffix [b3,b4] = 20s duration, last beat=b4 (same as c1)
    # IoU = (60-40) / (60-0) = 20/60 = 0.333 < 0.5 → IoU gate does NOT fire
    c2 = {
        "candidate_id": "cand_anchor_dup_last",
        "beat_ids": ["b3", "b4"],
        "start_s": 40.0,
        "end_s": 60.0,
        "duration_s": 20.0,
        "hook": 2,
        "cliff": 2,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 6.0,
    }

    _, survivors, fail_counts = evaluate_window_gates([c1, c2], beats, max_iou=0.5)
    assert len(survivors) == 1
    assert survivors[0]["candidate_id"] == "cand_anchor_high"
    assert fail_counts["DUPLICATE_ANCHOR"] == 1
    assert c2["first_failed_gate"] == "DUPLICATE_ANCHOR"


def test_gate_10_duplicate_contained():
    """Verify DUPLICATE_CONTAINED rejects candidate when overlap / shorter_duration > 0.6.

    Design:
    c1 covers [0..50s] (beats b1, b2, b3, b4), duration=50s, score=9.0.
    c2 covers [10..35s] (beats b2, b3), duration=25s, score=8.0.
    First beats differ (b1 vs b2).
    Last beats differ (b4 vs b3).
    IoU = 25 / 50 = 0.50 <= 0.5 (DUPLICATE_OVERLAP does not trigger).
    Overlap / shorter = 25 / 25 = 1.0 > 0.6 (DUPLICATE_CONTAINED triggers).
    """
    beats = [
        {"beat_id": "b1", "kind": "ACTION", "start_s": 0.0, "end_s": 10.0},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 10.0, "end_s": 20.0},
        {"beat_id": "b3", "kind": "DIALOGUE", "start_s": 20.0, "end_s": 35.0},
        {"beat_id": "b4", "kind": "SFX", "start_s": 35.0, "end_s": 50.0},
    ]

    c1 = {
        "candidate_id": "cand_high",
        "beat_ids": ["b1", "b2", "b3", "b4"],
        "start_s": 0.0,
        "end_s": 50.0,
        "duration_s": 50.0,
        "hook": 2,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 9.0,
    }
    c2 = {
        "candidate_id": "cand_contained",
        "beat_ids": ["b2", "b3"],
        "start_s": 10.0,
        "end_s": 35.0,
        "duration_s": 25.0,
        "hook": 2,
        "cliff": 2,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 8.0,
    }

    _, survivors, fail_counts = evaluate_window_gates([c1, c2], beats, max_iou=0.5, max_containment=0.6)
    assert len(survivors) == 1
    assert survivors[0]["candidate_id"] == "cand_high"
    assert fail_counts["DUPLICATE_CONTAINED"] == 1
    assert c2["first_failed_gate"] == "DUPLICATE_CONTAINED"


def test_soft_gates_mark_multiple_narrative_weaknesses_as_low_confidence():
    """Verify soft gates retain candidates and flag multiple narrative weaknesses as low_confidence."""
    beats = make_sample_episode_beats()
    cand_weak = make_clean_candidate()
    cand_weak["hook"] = 1
    cand_weak["cliff"] = 1

    _, survivors, fail_counts = evaluate_window_gates([cand_weak], beats, min_hook=2, min_cliff=2)
    assert len(survivors) == 1
    admitted = survivors[0]
    assert admitted["passed_all_gates"] is True
    assert admitted["low_confidence"] is True
    assert fail_counts["HOOK_TOO_WEAK"] == 1
    assert fail_counts["WEAK_CLIFFHANGER"] == 1


def test_hard_gates_strictly_reject_structural_and_straddle_failures():
    """Verify hard gates strictly reject structural (duration) and straddle cuts (0 survivors)."""
    beats = make_sample_episode_beats()

    # Window failing CUT_STRADDLES_SPOKEN
    cand_straddle = make_clean_candidate()
    cand_straddle["cut_straddles_spoken"] = True

    # Window failing DURATION
    cand_dur = make_clean_candidate()
    cand_dur["duration_s"] = 10.0

    _, survivors, fail_counts = evaluate_window_gates(
        [cand_straddle, cand_dur], beats
    )
    assert len(survivors) == 0, f"Expected 0 survivors from invalid structural candidates, got {len(survivors)}"
    assert fail_counts["CUT_STRADDLES_SPOKEN"] == 1
    assert fail_counts["DURATION"] == 1


def test_boundary_refinement_persists_raw_boundaries_and_ops():
    """Verify refinement on survivors records start_s_raw, end_s_raw, ops, and maintains duration clamp."""
    beats = [
        {"beat_id": "b0", "kind": "CAMERA", "start_s": 0.0, "end_s": 3.0},
        {"beat_id": "b1", "kind": "DIALOGUE", "start_s": 3.0, "end_s": 25.0},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 35.0},
    ]
    cand = {
        "candidate_id": "c_refine_test",
        "beat_ids": ["b1", "b2"],
        "start_s": 0.0,
        "end_s": 35.0,
        "duration_s": 35.0,
        "hook": 3,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 8.0,
    }
    all_eval, survivors, _ = evaluate_window_gates([cand], beats, refine=True)
    assert len(survivors) == 1
    surv = survivors[0]
    assert surv["start_s_raw"] == 0.0
    assert surv["end_s_raw"] == 35.0
    assert surv["start_s"] == 3.0
    assert surv["end_s"] == 35.6
    assert surv["duration_s"] == 32.6
    assert any("trim_lead_silence" in op for op in surv["refine_ops"])
    assert any("extend_to_last_beat" in op for op in surv["refine_ops"])
    assert surv["score"] == 8.0  # selection score preserved
    assert "score_refined" in surv
    assert 20.0 <= surv["duration_s"] <= 60.0


def test_boundary_refinement_does_not_reorder_survivors():
    """Verify survivors maintain ranking freeze even if refined score or duration changes."""
    beats = [
        {"beat_id": "b1", "kind": "DIALOGUE", "start_s": 0.0, "end_s": 25.0},
        {"beat_id": "b2", "kind": "DIALOGUE", "start_s": 25.0, "end_s": 50.0},
    ]
    c1 = {
        "candidate_id": "rank_1",
        "beat_ids": ["b1"],
        "start_s": 0.0,
        "end_s": 25.0,
        "duration_s": 25.0,
        "hook": 3,
        "cliff": 3,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 10.0,
    }
    c2 = {
        "candidate_id": "rank_2",
        "beat_ids": ["b2"],
        "start_s": 25.0,
        "end_s": 50.0,
        "duration_s": 25.0,
        "hook": 2,
        "cliff": 2,
        "resolved_end": False,
        "cut_straddles_spoken": False,
        "score": 7.0,
    }
    _, survivors, _ = evaluate_window_gates([c1, c2], beats, refine=True)
    assert [s["candidate_id"] for s in survivors] == ["rank_1", "rank_2"]



