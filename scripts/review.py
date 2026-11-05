#!/usr/bin/env python3
"""Terminal Promo Reviewer (scripts/review.py).

Reviews candidate promos, records human judgments without screenplay text,
and logs decisions to labels.csv. Under 150 lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REASONS = {
    "1": "weak_hook",
    "2": "resolved_ending",
    "3": "confusing",
    "4": "boundary_off",
    "5": "duplicate",
    "6": "other",
}


def load_promos(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"Error: Promos file not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("promos", data.get("ranked_promos", data.get("rankings", [])))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal promo reviewer")
    parser.add_argument("--promos", default="outputs/promos_per_episode.json", help="Path to promos JSON")
    parser.add_argument("--out", default="labels.csv", help="Output labels CSV file")
    args = parser.parse_args()

    promos = load_promos(Path(args.promos))
    out_path = Path(args.out)

    fieldnames = ["candidate_id", "episode", "start_s", "end_s", "decision", "reason_code"]
    existing_cids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_cids.add(r.get("candidate_id"))

    file_exists = out_path.exists() and out_path.stat().st_size > 0
    csv_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    decisions = {"approve": 0, "reject": 0}
    rejection_reasons: Dict[str, int] = {v: 0 for v in REASONS.values()}

    print(f"\nLoaded {len(promos)} promos. Reviewing (Press 'q' at prompt to quit)...")
    try:
        for idx, p in enumerate(promos, start=1):
            cid = p.get("candidate_id", f"cand_{idx}")
            if cid in existing_cids:
                continue

            ep = p.get("episode", "unknown")
            st = p.get("start_s", 0.0)
            en = p.get("end_s", 0.0)
            dur = p.get("duration_s", round(en - st, 2))
            scores = p.get("scores", {})
            why = p.get("why", "N/A")

            print("\n" + "=" * 70)
            print(f"PROMO [{idx}/{len(promos)}]: {cid} | Episode: {ep}")
            print(f"Timing: {st:.2f}s - {en:.2f}s ({dur:.2f}s)")
            print(f"Scores: hook={scores.get('hook')} cliff={scores.get('cliff')} clarity={scores.get('clarity')} s3a={scores.get('s3a_score')}")
            print(f"Note  : {why}")
            print("=" * 70)

            while True:
                choice = input("[A]pprove / [R]eject / [S]kip / [Q]uit: ").strip().lower()
                if choice in ("a", "approve"):
                    writer.writerow({
                        "candidate_id": cid, "episode": ep, "start_s": st, "end_s": en,
                        "decision": "approve", "reason_code": "none",
                    })
                    csv_file.flush()
                    decisions["approve"] += 1
                    break
                elif choice in ("r", "reject"):
                    print("Select rejection reason:")
                    for k, v in REASONS.items():
                        print(f"  [{k}] {v}")
                    rc_choice = input("Reason code (1-6): ").strip()
                    rc = REASONS.get(rc_choice, "other")
                    writer.writerow({
                        "candidate_id": cid, "episode": ep, "start_s": st, "end_s": en,
                        "decision": "reject", "reason_code": rc,
                    })
                    csv_file.flush()
                    decisions["reject"] += 1
                    rejection_reasons[rc] += 1
                    break
                elif choice in ("s", "skip"):
                    break
                elif choice in ("q", "quit"):
                    return
                else:
                    print("Invalid input. Please enter 'a', 'r', 's', or 'q'.")
    finally:
        csv_file.close()
        total = decisions["approve"] + decisions["reject"]
        print("\n" + "=" * 70)
        print(f"REVIEW SUMMARY: {total} reviewed ({decisions['approve']} approved, {decisions['reject']} rejected)")
        if decisions["reject"] > 0:
            print("Rejection reasons breakdown:")
            for code, count in rejection_reasons.items():
                if count > 0:
                    print(f"  - {code:<18}: {count}")
        print(f"Saved judgments to {out_path} (no screenplay text).")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
