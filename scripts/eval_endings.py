"""Do the tool's cliffhanger scores match what the writers call a cliffhanger?

Each episode's CONTINUITY NOTES describe how the previous episode ended, and
the writers use the word "cliffhanger" for some endings and not for others
("Follows Episode 6, where ... found refuge"). That gives one independent label
per episode boundary, written before this tool existed.

Note on independence: the annotator of episode N-1 never sees episode N's notes.

Usage: python scripts/eval_endings.py --run outputs_v2 [--json eval_endings.json]
Prints numbers only (no screenplay text), so the output can be committed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def ep_num(s: str) -> int:
    return int(re.findall(r"\d+", s)[0])


def auc(pos, neg):
    """P(score of a random positive > random negative), ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs_v2")
    ap.add_argument("--tail-s", type=float, default=6.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)

    parsed = {ep_num(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in (run / "parsed").glob("*.json")}
    ann = {ep_num(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in (run / "annotated").glob("*.json")}

    rows = []
    for n in sorted(parsed):
        if n - 1 not in parsed or n - 1 not in ann:
            continue
        note = (parsed[n].get("metadata", {}).get("continuity_notes") or "").lower()
        if not note:
            continue
        prev, prev_ann = parsed[n - 1], ann[n - 1]
        end = prev.get("duration_max_end_s") or max(b["end_s"] for b in prev["beats"])
        tail = [b for b in prev["beats"] if b["kind"] != "CAMERA" and b["end_s"] > end - a.tail_s]
        qs = [(prev_ann.get("annotations", {}).get(b["beat_id"]) or {}).get("q") for b in tail]
        qs = [q for q in qs if q is not None]
        rows.append({
            "episode": n - 1,
            "writer_cliffhanger": "cliffhanger" in note,
            "tool_ends_on_cliffhanger": bool(prev_ann.get("ends_on_cliffhanger")),
            "tail_max_q": max(qs) if qs else 0,
        })

    pos = [r for r in rows if r["writer_cliffhanger"]]
    neg = [r for r in rows if not r["writer_cliffhanger"]]
    tp = sum(r["tool_ends_on_cliffhanger"] for r in pos)
    fp = sum(r["tool_ends_on_cliffhanger"] for r in neg)
    res = {
        "episode_boundaries": len(rows),
        "writer_cliffhanger": len(pos),
        "writer_other_ending": len(neg),
        "tool_flag": {
            "recall_on_writer_cliffhangers": f"{tp}/{len(pos)}",
            "flagged_calm_endings_as_cliffhangers": f"{fp}/{len(neg)}",
            "accuracy": round((tp + len(neg) - fp) / len(rows), 3) if rows else None,
            "always_yes_baseline_accuracy": round(len(pos) / len(rows), 3) if rows else None,
        },
        "tail_open_question_q": {
            "mean_on_writer_cliffhangers": round(mean(r["tail_max_q"] for r in pos), 2) if pos else None,
            "mean_on_other_endings": round(mean(r["tail_max_q"] for r in neg), 2) if neg else None,
            "auc": auc([r["tail_max_q"] for r in pos], [r["tail_max_q"] for r in neg]),
        },
        "misses": [r["episode"] for r in pos if not r["tool_ends_on_cliffhanger"]],
        "false_alarms": [r["episode"] for r in neg if r["tool_ends_on_cliffhanger"]],
    }
    print(json.dumps(res, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": res, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
