import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search import baseline_last_n, enumerate_candidates, search_episode, straddle_flags

ROOT = Path(__file__).resolve().parents[1]


def mk(bid, kind, s, e, text="", spk=None):
    return {"beat_id": bid, "kind": kind, "start_s": s, "end_s": e, "text": text, "speaker_norm": spk}


def build(n=24, step=3.0):
    beats, ann, t = [], {}, 0.0
    for i in range(1, n + 1):
        bid = f"b{i:03d}"
        kind = "CAMERA" if i % 6 == 0 else "DIALOGUE"
        beats.append(mk(bid, kind, round(t, 2), round(t + step, 2), "A line.", f"SPK{i % 3}"))
        ann[bid] = {"h": 3 if i == 4 else 1, "q": 3 if i == 14 else 1, "r": False, "n": 0, "t": 2}
        t += step
    parsed = {"episode": "episode-999", "episode_end_s": round(t, 2), "beats": beats}
    return parsed, {"episode": "episode-999", "annotations": ann}


def test_every_candidate_respects_the_duration_window():
    parsed, ann = build()
    cands = enumerate_candidates(parsed, ann)
    assert cands
    assert all(20.0 - 1e-6 <= c["duration_s"] <= 60.0 + 1e-6 for c in cands)


def test_candidates_are_scored_and_carry_provenance():
    parsed, ann = build()
    c = enumerate_candidates(parsed, ann)[0]
    assert "breakdown" in c and "contrib" in c["breakdown"]
    assert abs(sum(c["breakdown"]["contrib"].values()) - c["score"]) < 1e-6
    assert c["features"]["beat_count"] >= 1


def test_search_is_deterministic():
    parsed, ann = build()
    a = search_episode(parsed, ann)
    b = search_episode(parsed, ann)
    assert [x["candidate_id"] for x in a] == [x["candidate_id"] for x in b]
    assert [x["start_s"] for x in a] == [x["start_s"] for x in b]


def test_refinement_does_not_reorder_the_picks():
    parsed, ann = build()
    plain = [c["candidate_id"] for c in search_episode(parsed, ann, refine=False)]
    polished = [c["candidate_id"] for c in search_episode(parsed, ann, refine=True)]
    assert plain == polished


def test_selected_windows_are_not_near_duplicates():
    parsed, ann = build()
    picks = search_episode(parsed, ann, top_k=3)
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            a, b = picks[i], picks[j]
            inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
            union = (a["end_s"] - a["start_s"]) + (b["end_s"] - b["start_s"]) - inter
            assert inter / union <= 0.5 + 1e-6


def test_straddle_detection():
    beats = [mk("b1", "DIALOGUE", 0.0, 10.0), mk("b2", "DIALOGUE", 8.0, 15.0)]
    spoken, action = straddle_flags(beats, 0.0, 10.0, {"b1"})
    assert spoken is True


def test_baseline_last_n_sits_at_the_tail():
    parsed, _ = build()
    b = baseline_last_n(parsed, 45.0)[0]
    assert b["end_s"] == parsed["episode_end_s"]
    assert round(b["duration_s"], 1) == 45.0


def test_episode_with_no_annotations_still_returns_candidates():
    parsed, _ = build()
    out = search_episode(parsed, {"annotations": {}})
    assert out and all(c["score"] is not None for c in out)


def test_fixtures_and_eval_run_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/make_fixtures.py"), "--out", td, "--episodes", "4"],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert r.returncode == 0, r.stderr
        gold = Path(td) / "gold_synthetic.csv"
        assert gold.exists() and len(gold.read_text().splitlines()) >= 2
        out = Path(td) / "eval.json"
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "scripts/eval_v2.py"), "--parsed", f"{td}/parsed",
             "--annotated", f"{td}/annotated", "--gold", str(gold), "--out", str(out), "--boot", "200"],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert r2.returncode == 0, r2.stderr
        data = json.loads(out.read_text())
        assert data["table"] and "paired_vs_baseline" in data
