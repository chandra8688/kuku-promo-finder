"""Production, evaluation and tuning must run the same procedure.

Before these tests existed, run.py scored with config/scoring.yaml while the
reported metrics used config/scoring_v2.yaml, and the two picked the same top-1
in 1 of 12 fixture episodes.
"""

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import engine
from src.candidates import process_episode_candidates
from src.score import DEFAULT_WEIGHTS, rank_key
from src.search import search_episode
from src.windows import load_v2_config, scoring_provenance


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_fixtures.py"), "--out", str(d), "--episodes", "6"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    return d


def episodes(fx):
    return sorted(p.stem for p in (fx / "parsed").glob("*.json"))


def load(fx, ep):
    return (
        json.loads((fx / "parsed" / f"{ep}.json").read_text()),
        json.loads((fx / "annotated" / f"{ep}.json").read_text()),
    )


def production(fx, ep):
    with redirect_stdout(io.StringIO()):
        process_episode_candidates(ep, fx / "parsed", fx / "annotated", fx / "cand", gold_file=fx / "no_gold.csv")
    return json.loads((fx / "cand" / f"{ep}.json").read_text())


def expected_weights():
    w = dict(DEFAULT_WEIGHTS)
    w.update(load_v2_config()["weights"])
    return w


def test_production_scores_with_the_fitted_v2_weights(fx):
    want = expected_weights()
    for ep in episodes(fx):
        d = production(fx, ep)
        assert d["scoring_provenance"]["effective_weights"] == {k: float(v) for k, v in sorted(want.items())}
        for c in d["candidates"]:
            assert c["breakdown"]["weights"] == {k: want.get(k, 0.0) for k in c["breakdown"]["weights"]}


def test_context_need_is_never_rewarded_in_production(fx):
    for ep in episodes(fx):
        for c in production(fx, ep)["candidates"]:
            assert c["breakdown"]["weights"]["context"] <= 0.0


def test_provenance_hash_tracks_the_config_file():
    a = scoring_provenance()
    assert len(a["config_sha256"]) == 16
    assert a == scoring_provenance()


def test_production_and_evaluation_pick_identical_windows_and_scores(fx):
    for ep in episodes(fx):
        d = production(fx, ep)
        prod = sorted([c for c in d["candidates"] if c.get("passed_all_gates")], key=rank_key)[:3]
        p, a = load(fx, ep)
        ev = search_episode(p, a, top_k=3)
        assert [(c["candidate_id"], c["start_s"], c["end_s"], c["score"]) for c in prod] == \
               [(c["candidate_id"], c["start_s"], c["end_s"], c["score"]) for c in ev], ep


def test_tuner_fast_path_equals_the_engine(fx):
    w = expected_weights()
    w.update({"hook": 2.5, "cliff": 1.0, "tail_pos": 0.0, "context": -2.0})
    for ep in episodes(fx):
        p, a = load(fx, ep)
        wins = engine.enumerate(p, a)
        _, fast, _ = engine.select(engine.rescore(wins, w), p, a, weights=w, top_k=3)
        _, _, full, _ = engine.run_episode(p, a, weights=w, top_k=3)
        key = lambda c: (c["candidate_id"], c["start_s"], c["end_s"], c["score"])
        assert [key(c) for c in sorted(fast, key=rank_key)[:3]] == [key(c) for c in sorted(full, key=rank_key)[:3]], ep


def test_rescore_does_not_mutate_the_cached_windows(fx):
    ep = episodes(fx)[0]
    p, a = load(fx, ep)
    wins = engine.enumerate(p, a)
    before = [(w["score"], w["start_s"], w["end_s"]) for w in wins]
    engine.select(engine.rescore(wins, {"hook": 3.0}), p, a, weights={"hook": 3.0})
    assert before == [(w["score"], w["start_s"], w["end_s"]) for w in wins]
