"""Blind rating: the tool's top promos vs a last-45-seconds baseline, shuffled.

make  -> writes clips.md (script text of each clip, labelled C01..), ratings.csv
         (blank scores) and key.json (which system made which clip) to --out.
         --out must be a private folder: it contains screenplay text.
score -> reads ratings.csv + key.json and prints numbers only (safe to commit).

Usage:
  python scripts/blind_sheet.py make  --run outputs_v2 --out _backups/blind_rating
  python scripts/blind_sheet.py score --out _backups/blind_rating --json eval/blind_rating.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

FIELDS = ["clip", "hook_1to5", "cliffhanger_1to5", "standalone_1to5", "would_post_y_n", "note"]


def render(beats, start, end):
    out = []
    for b in beats:
        if b["kind"] == "CAMERA" or b["end_s"] <= start or b["start_s"] >= end:
            continue
        t = max(b["start_s"] - start, 0.0)
        text = (b.get("text") or "").strip()
        if b["kind"] == "DIALOGUE":
            emo = f" ({b['emotion_label'].lower()})" if b.get("emotion_label") else ""
            out.append(f"`{t:5.1f}s` **{b.get('speaker_norm') or '?'}**{emo}: {text}")
        elif b["kind"] == "SFX":
            out.append(f"`{t:5.1f}s` *[sound]* {text}")
        else:
            out.append(f"`{t:5.1f}s` *[action]* {text}")
    return "\n\n".join(out)


def make(run: Path, out: Path, n: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ranked = json.loads((run / "promos_ranked.json").read_text(encoding="utf-8"))["ranked_promos"][:n]
    parsed = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run / "parsed").glob("*.json")}
    rng = random.Random(seed)
    clips = [{"system": "tool", "episode": p["episode"], "start": p["start_s"], "end": p["end_s"]} for p in ranked]
    # Baseline episodes must differ from the tool's, or near-identical clips
    # appear side by side and give away which system made which.
    tool_eps = {c["episode"] for c in clips}
    for ep in rng.sample(sorted(set(parsed) - tool_eps), n):
        end = parsed[ep]["duration_max_end_s"]
        clips.append({"system": "last45", "episode": ep, "start": max(end - 45.0, 0.0), "end": end})
    rng.shuffle(clips)
    md = ["# Blind rating", "",
          "Rate each clip as a cold viewer scrolling TikTok. Times are seconds from the clip start.",
          "Fill ratings.csv: hook (first 3 s), cliffhanger (ending), standalone (makes sense alone), 1-5; would_post y/n.",
          "Do not open key.json until you have finished.", ""]
    key = {}
    for i, c in enumerate(clips, start=1):
        cid = f"C{i:02d}"
        key[cid] = c
        md += [f"## {cid}  ({c['end'] - c['start']:.0f} s)", "", render(parsed[c["episode"]]["beats"], c["start"], c["end"]), ""]
    (out / "clips.md").write_text("\n".join(md), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for cid in key:
            w.writerow({"clip": cid})
    print(f"Wrote {len(clips)} clips to {out}")


def score(out: Path, json_path: str | None) -> None:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(out / "ratings.csv", encoding="utf-8")) if r.get("hook_1to5")]
    res = {}
    for system in ("tool", "last45"):
        rs = [r for r in rows if key[r["clip"]]["system"] == system]
        if not rs:
            continue
        res[system] = {
            "rated": len(rs),
            "hook": round(mean(float(r["hook_1to5"]) for r in rs), 2),
            "cliffhanger": round(mean(float(r["cliffhanger_1to5"]) for r in rs), 2),
            "standalone": round(mean(float(r["standalone_1to5"]) for r in rs), 2),
            "would_post": f"{sum(r['would_post_y_n'].strip().lower().startswith('y') for r in rs)}/{len(rs)}",
        }
    print(json.dumps(res, indent=2))
    if json_path:
        Path(json_path).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--run", default="outputs_v2")
    ap.add_argument("--out", default="_backups/blind_rating")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    make(Path(a.run), Path(a.out), a.n, a.seed) if a.cmd == "make" else score(Path(a.out), a.json)
