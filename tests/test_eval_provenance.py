import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_v2.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_public_reference_set_is_reported_as_ai_reviewed():
    prov = ev.provenance(ev.load_gold(str(ROOT / "gold" / "gold_public.csv")))
    assert prov["all_human"] is False
    assert prov["counts"].get("ai_reviewed") == prov["n"]
    assert "not verified by a person" in ev.provenance_line(prov)


def test_dev_reference_set_is_human():
    prov = ev.provenance(ev.load_gold(str(ROOT / "gold" / "gold_dev.csv")))
    assert prov["all_human"] is True


def test_require_human_refuses_ai_reviewed_references(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_v2.py"), "--parsed", str(tmp_path), "--gold",
         str(ROOT / "gold/gold_public.csv"), "--require-human", "--out", str(tmp_path / "e.json")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 2
    assert "REFUSING" in r.stderr


def test_eval_never_writes_the_report_unless_asked(tmp_path):
    fx = tmp_path / "fx"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_fixtures.py"), "--out", str(fx), "--episodes", "3"],
                   capture_output=True, text=True, cwd=tmp_path, check=True)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_v2.py"), "--parsed", str(fx / "parsed"), "--annotated",
         str(fx / "annotated"), "--gold", str(fx / "gold_synthetic.csv"), "--out", str(tmp_path / "e.json"), "--boot", "100"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "eval_report.md").exists()
