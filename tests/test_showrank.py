"""Unit tests for show-level ranking (src/showrank.py)."""

from src import showrank as sr

CFG = sr.load_config()


def promo(cid, ep, jh, jc, ah=3, ac=3, clarity=2, payoff=2, judged=True, s3a=10.0, **kw):
    p = {
        "candidate_id": cid,
        "episode": ep,
        "scores": {
            "hook": jh, "cliff": jc, "clarity": clarity, "payoff": payoff, "s3a_score": s3a,
            "judged": judged, "resolved_after": False, "annotator": {"hook": ah, "cliff": ac},
        },
        "low_confidence": False,
    }
    p.update(kw)
    return p


def test_funnel_bonus_full_then_linear_then_zero():
    assert sr.funnel_bonus("episode-001", CFG) == 1.0
    assert sr.funnel_bonus("episode-010", CFG) == 1.0
    assert sr.funnel_bonus("episode-020", CFG) == 0.5
    assert sr.funnel_bonus("episode-030", CFG) == 0.0


def test_agreed_score_is_the_lower_of_the_two_models():
    ag = sr.agreed(promo("a", "episode-005", jh=1, jc=3, ah=3, ac=2))
    assert (ag["hook"], ag["cliff"], ag["single_model"]) == (1, 2, False)


def test_unjudged_promo_uses_annotator_only():
    ag = sr.agreed(promo("a", "episode-005", jh=None, jc=None, ah=2.7, ac=3, judged=False))
    assert (ag["hook"], ag["cliff"], ag["single_model"]) == (2.7, 3, True)


def test_gate_reasons():
    assert sr.gate(promo("a", "episode-040", 3, 3), CFG) is None
    assert sr.gate(promo("a", "episode-040", 1, 3), CFG) == "hook_not_agreed"
    assert sr.gate(promo("a", "episode-040", 3, 1), CFG) == "cliffhanger_not_agreed"
    assert sr.gate(promo("a", "episode-040", 3, 3, low_confidence=True), CFG) == "low_confidence"
    p = promo("a", "episode-040", 3, 3)
    p["scores"]["annotator"]["resolved_last"] = True
    assert sr.gate(p, CFG) == "last_beat_resolves"


def test_early_episode_wins_a_tie_on_quality():
    top = sr.rank_show([promo("late", "episode-040", 3, 3, s3a=20), promo("early", "episode-002", 3, 3, s3a=5)], CFG)
    assert [p["candidate_id"] for p in top] == ["early", "late"]
    assert "leads straight into the first episodes" in top[0]["rank_reason"]


def test_disagreement_ranks_below_agreement_even_with_higher_s3a():
    top = sr.rank_show([promo("loud", "episode-040", 0, 3, s3a=99), promo("agreed", "episode-041", 2, 2, s3a=1)], CFG)
    assert [p["candidate_id"] for p in top] == ["agreed", "loud"]
    assert top[1]["below_bar"] == "hook_not_agreed"


def test_max_two_per_episode():
    pool = [promo(f"c{i}", "episode-001", 3, 3, s3a=float(i)) for i in range(5)]
    assert len(sr.rank_show(pool, CFG)) == 2


def test_wave1_spreads_hook_types_within_score_gap():
    pool = [promo(f"v{i}", f"episode-{40 + i:03d}", 3, 3, s3a=50 - i) for i in range(6)]
    info = {f"v{i}": {"hook_type": "physical_violence", "why_it_works": "x"} for i in range(6)}
    pool.append(promo("r", "episode-050", 3, 3, s3a=1))
    pool.append(promo("s", "episode-051", 3, 3, s3a=0))
    info["r"] = {"hook_type": "identity_reveal", "why_it_works": "y"}
    info["s"] = {"hook_type": "romantic_tension", "why_it_works": "z"}
    top = sr.rank_show(pool, CFG, finalist_info=info)
    wave1_types = {p["hook_type"] for p in top if p["wave"] == 1}
    assert len(wave1_types) >= 3
    assert top[0]["why"] == "x"


def test_wave1_keeps_order_without_hook_types():
    pool = [promo(f"v{i}", f"episode-{40 + i:03d}", 3, 3, s3a=50 - i) for i in range(6)]
    assert [p["candidate_id"] for p in sr.rank_show(pool, CFG)][:5] == [f"v{i}" for i in range(5)]
