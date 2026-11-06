"""What reaches the publishable list, and in what order, after the judge."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as runmod


def beats():
    return [{"beat_id": f"b{i}", "kind": "DIALOGUE", "start_s": i * 3.0, "end_s": i * 3.0 + 2.9, "text": "x"} for i in range(40)]


def judged(cid, s3a, passed=True, reason=None, error=None, low=False, timed=True, hook=3, cliff=3):
    c = {"candidate_id": cid, "episode": "episode-901", "s3a_score": s3a, "score": s3a,
         "passed_judge_gates": passed, "judge_gate_reason": reason, "low_confidence": low,
         "hook": hook, "cliff": cliff, "resolved_after": False, "beat_ids": ["b1", "b2"]}
    if timed:
        c.update({"start_s": 3.0, "end_s": 30.0, "duration_s": 27.0})
    if error:
        c["judge_error"] = error
    return c


@pytest.fixture
def dirs(tmp_path):
    for d in ("parsed", "cand", "judged", "out"):
        (tmp_path / d).mkdir()
    (tmp_path / "parsed" / "episode-901.json").write_text(json.dumps({"episode": "episode-901", "beats": beats()}))
    runmod.JUDGE_REJECTED.clear()
    return tmp_path


def promos(dirs, cands):
    (dirs / "judged" / "episode-901.json").write_text(json.dumps({"tier": "primary", "judged": cands}))
    return runmod.build_per_episode_promos(["episode-901"], dirs / "parsed", dirs / "cand", dirs / "judged", dirs / "out")


def test_judge_and_s2_agreed_resolved_is_removed_and_recorded(dirs):
    out = promos(dirs, [judged("a", 9.0, passed=False, reason="JUDGE_S2_MUTUAL_RESOLVED"), judged("b", 5.0)])
    assert [p["candidate_id"] for p in out] == ["b"]
    assert runmod.JUDGE_REJECTED == [{"episode": "episode-901", "candidate_id": "a", "reason": "JUDGE_S2_MUTUAL_RESOLVED"}]


def test_judge_failure_is_not_a_verdict(dirs):
    out = promos(dirs, [judged("a", 9.0, passed=False, reason="SCHEMA_PARSE_FAILED", error="SCHEMA_PARSE_FAILED"), judged("b", 5.0)])
    assert [p["candidate_id"] for p in out] == ["b", "a"]  # kept, but behind confident clips
    assert out[1]["low_confidence"] is True
    assert runmod.JUDGE_REJECTED == []


def test_untimed_record_never_becomes_a_zero_length_promo(dirs):
    out = promos(dirs, [judged("a", 9.0, error="MISSING_FROM_BATCH_OUTPUT", timed=False), judged("b", 5.0)])
    assert [p["candidate_id"] for p in out] == ["b"]
    assert all(p["duration_s"] > 0 for p in out)


def test_order_is_selection_score_then_id(dirs):
    out = promos(dirs, [judged("c", 5.0), judged("a", 7.0), judged("b", 5.0), judged("z", 9.0, low=True)])
    assert [p["candidate_id"] for p in out] == ["a", "b", "c", "z"]


def test_judge_scores_do_not_decide_show_level_membership(dirs, monkeypatch):
    monkeypatch.setattr(runmod, "get_gateway", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    pool = []
    for i in range(20):
        p = promos(dirs, [judged(f"e{i}", float(i), hook=0 if i >= 10 else 3, cliff=0 if i >= 10 else 3)])[0]
        p["episode"] = f"episode-{900 + i}"
        pool.append(p)
    top = runmod.build_show_level_top15(pool, dirs / "out", no_llm=False, rank_mode="s3a")
    # the ten highest S3a scores are exactly the ones the judge scored hook=0, cliff=0
    assert [p["candidate_id"] for p in top[:10]] == [f"e{i}" for i in range(19, 9, -1)]
    assert len(top) == 15
