import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labels import format_cell, load_labels, match_label, precision_at_k

HEADER = "candidate_id,episode,start_s,end_s,decision,reason_code\n"


def write(tmp_path, rows):
    p = tmp_path / "labels.csv"
    p.write_text(HEADER + "".join(rows), encoding="utf-8")
    return p


def promo(cid, ep="episode-003", s=10.0, e=40.0):
    return {"candidate_id": cid, "episode": ep, "start_s": s, "end_s": e, "duration_s": e - s}


def test_missing_file_is_empty_not_an_error():
    assert load_labels("/nonexistent/labels.csv") == {}


def test_load_and_skip_malformed_rows(tmp_path):
    p = write(tmp_path, [
        "c1,episode-003,10,40,approve,none\n",
        "c2,episode-003,50,80,reject,weak_hook\n",
        ",episode-003,1,2,approve,none\n",
        "c4,episode-003,1,2,skip,none\n",
    ])
    labels = load_labels(p)
    assert set(labels) == {"c1", "c2"}
    assert labels["c2"]["reason_code"] == "weak_hook"


def test_later_row_overrides_earlier(tmp_path):
    p = write(tmp_path, ["c1,episode-003,10,40,reject,confusing\n", "c1,episode-003,10,40,approve,none\n"])
    assert load_labels(p)["c1"]["decision"] == "approve"


def test_unlabelled_clip_is_not_counted_as_approved(tmp_path):
    p = write(tmp_path, ["c1,episode-003,10,40,approve,none\n"])
    res = precision_at_k([promo("c1"), promo("c2", s=60, e=95)], load_labels(p), k=2, n_boot=200)
    assert res["n_labelled"] == 1
    assert res["n_approved"] == 1
    assert res["complete"] is False
    assert res["unlabelled"] == ["c2"]


def test_no_labels_gives_no_precision(tmp_path):
    res = precision_at_k([promo("c1")], {}, k=1, n_boot=200)
    assert res["precision"] is None
    assert format_cell(res) == "not labelled"


def test_rejections_lower_precision(tmp_path):
    p = write(tmp_path, [
        "c1,episode-003,10,40,approve,none\n",
        "c2,episode-003,50,80,reject,resolved_ending\n",
        "c3,episode-003,90,120,approve,none\n",
    ])
    res = precision_at_k(
        [promo("c1"), promo("c2", s=50, e=80), promo("c3", s=90, e=120)],
        load_labels(p), k=3, n_boot=400,
    )
    assert res["n_approved"] == 2 and res["n_labelled"] == 3
    assert res["precision"] == round(2 / 3, 4)
    assert res["complete"] is True
    assert res["rejected"][0]["reason_code"] == "resolved_ending"
    assert res["ci_low"] <= res["precision"] <= res["ci_high"]


def test_window_fallback_match_when_candidate_id_changed(tmp_path):
    p = write(tmp_path, ["old-id,episode-003,10.0,40.0,approve,none\n"])
    labels = load_labels(p)
    assert match_label(promo("new-id", s=10.1, e=40.0), labels) is not None
    assert match_label(promo("new-id", s=200.0, e=240.0), labels) is None


def test_window_fallback_respects_episode(tmp_path):
    p = write(tmp_path, ["old-id,episode-003,10.0,40.0,approve,none\n"])
    assert match_label(promo("x", ep="episode-013", s=10.0, e=40.0), load_labels(p)) is None


def test_partial_coverage_is_announced_in_the_cell(tmp_path):
    p = write(tmp_path, ["c1,episode-003,10,40,approve,none\n"])
    cell = format_cell(precision_at_k([promo("c1"), promo("c2", s=60, e=95)], load_labels(p), k=2, n_boot=200))
    assert "partial" in cell and "1 of 2" in cell


def test_precision_is_reproducible(tmp_path):
    p = write(tmp_path, ["c1,episode-003,10,40,approve,none\n", "c2,episode-003,50,80,reject,other\n"])
    promos = [promo("c1"), promo("c2", s=50, e=80)]
    a = precision_at_k(promos, load_labels(p), k=2, n_boot=300)
    b = precision_at_k(promos, load_labels(p), k=2, n_boot=300)
    assert a == b
