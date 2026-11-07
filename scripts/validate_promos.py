"""Check every output promo against the brief's hard rules. Numbers only.

  1. 20-60 s long, inside its episode
  2. something in the first 3 s scored as a hook (annotator h >= 2, or a strong
     sound/action cue) - the brief's "opens on a hook"
  3. no cut inside a spoken line (start/end never fall mid-dialogue)
  4. unresolved ending: the annotator does not mark the final beat as resolving,
     and the judge does not say the moments right after the clip resolve it

Usage: python -m scripts.validate_promos --run outputs_v2 [--file promos_ranked.json]
Exit code 1 if any promo fails any rule. Rules 2 and 4 need model scores, so a
--no-llm run reports them as "not checked" rather than passing them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.features import cue_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs_v2")
    ap.add_argument("--file", default="promos_ranked.json")
    a = ap.parse_args()
    run = Path(a.run)
    data = json.loads((run / a.file).read_text(encoding="utf-8"))
    promos = data["ranked_promos"] if isinstance(data, dict) else data
    parsed, ann = {}, {}
    unscored = 0  # promos from episodes with no annotator scores (--no-llm run)
    fails = {"duration": [], "outside_episode": [], "no_hook_in_3s": [], "cuts_mid_line": [], "ends_resolved": []}
    for p in promos:
        ep = p["episode"]
        if ep not in parsed:
            parsed[ep] = json.loads((run / "parsed" / f"{ep}.json").read_text(encoding="utf-8"))
            af = run / "annotated" / f"{ep}.json"
            ann[ep] = json.loads(af.read_text(encoding="utf-8")).get("annotations", {}) if af.exists() else {}
        beats, A = parsed[ep]["beats"], ann[ep]
        s, e, cid = p["start_s"], p["end_s"], p["candidate_id"]
        if not 20.0 - 1e-6 <= e - s <= 60.0 + 1e-6:
            fails["duration"].append(cid)
        if s < 0 or e > parsed[ep]["duration_max_end_s"] + 0.01:
            fails["outside_episode"].append(cid)
        early = [b for b in beats if b["start_s"] < s + 3.0 and b["end_s"] > s]
        if not A:
            unscored += 1
        elif not any((A.get(b["beat_id"]) or {}).get("h", 0) >= 2 or
                   (b["kind"] != "DIALOGUE" and cue_score(b.get("text") or "") >= 2) for b in early):
            fails["no_hook_in_3s"].append(cid)
        for b in beats:
            if b["kind"] == "DIALOGUE" and (b["start_s"] + 0.05 < s < b["end_s"] - 0.05 or b["start_s"] + 0.05 < e < b["end_s"] - 0.05):
                fails["cuts_mid_line"].append(cid)
                break
        inside = [b for b in beats if b["kind"] != "CAMERA" and b["start_s"] < e and b["end_s"] > s]
        if (inside and (A.get(inside[-1]["beat_id"]) or {}).get("r")) or (p.get("scores") or {}).get("resolved_after"):
            fails["ends_resolved"].append(cid)
    n = len(promos)
    report = {"promos_checked": n, **{k: f"{n - len(v)}/{n} pass" for k, v in fails.items()},
              "failing_ids": {k: v for k, v in fails.items() if v}}
    if unscored:
        report["no_hook_in_3s"] = f"not checked for {unscored}/{n} (no annotator scores: --no-llm run)"
        report["ends_resolved"] = f"not checked for {unscored}/{n} (no annotator scores: --no-llm run)"
    print(json.dumps(report, indent=2))
    return 1 if any(fails.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
