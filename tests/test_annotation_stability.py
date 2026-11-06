import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / args[0])] + args[1:], capture_output=True, text=True, cwd=cwd)


def fixtures(tmp):
    r = run(["make_fixtures.py", "--out", str(tmp / "fx"), "--episodes", "4"], tmp)
    assert r.returncode == 0, r.stderr
    return tmp / "fx"


def test_identical_runs_agree_completely(tmp_path):
    fx = fixtures(tmp_path)
    out = tmp_path / "s.json"
    r = run(["annotation_stability.py", "--parsed", str(fx / "parsed"), "--a", str(fx / "annotated"),
             "--b", str(fx / "annotated"), "--out", str(out)], tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["episodes"] == 4
    assert all(v == 1.0 for v in d["exact_agreement"].values())
    assert d["top1_same_id"] == d["top1_same_clip_iou50"] == 4
    assert d["mean_top3_overlap"] >= 1


def test_perturbed_run_lowers_agreement_on_the_changed_field(tmp_path):
    fx = fixtures(tmp_path)
    b = tmp_path / "b"
    b.mkdir()
    for f in (fx / "annotated").glob("*.json"):
        d = json.loads(f.read_text())
        for i, v in enumerate(d["annotations"].values()):
            if i % 2 == 0:
                v["h"] = (int(v["h"]) + 1) % 4
        (b / f.name).write_text(json.dumps(d))
    out = tmp_path / "s.json"
    r = run(["annotation_stability.py", "--parsed", str(fx / "parsed"), "--a", str(fx / "annotated"),
             "--b", str(b), "--out", str(out)], tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["exact_agreement"]["h"] < 0.6
    assert d["exact_agreement"]["q"] == 1.0
