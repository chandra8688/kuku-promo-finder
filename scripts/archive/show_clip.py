"""Terminal-only clip inspector for human gold audit (scripts/show_clip.py).

Prints the clip's beats and the 12s after to the TERMINAL ONLY, with local indices.
Writes nothing to any files or logs. Screenplay text is confidential.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_clip.py <gold_id>")
        sys.exit(1)

    gold_id = sys.argv[1].strip()

    # Search in gold/audit.csv or gold/gold.csv
    row = None
    audit_path = Path("gold/audit.csv")
    gold_path = Path("gold/gold.csv")

    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("gold_id", "").strip() == gold_id:
                    row = r
                    break

    if row is None and gold_path.exists():
        with open(gold_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("gold_id", "").strip() == gold_id:
                    row = {
                        "gold_id": r.get("gold_id"),
                        "episode": r.get("episode"),
                        "approved_start_s": r.get("start_s"),
                        "approved_end_s": r.get("end_s"),
                        "decision": "accepted_in_gold",
                        "note": "",
                    }
                    break

    if row is None:
        print(f"Error: Gold ID '{gold_id}' not found in gold/audit.csv or gold/gold.csv.")
        sys.exit(1)

    ep_raw = row.get("episode", "").strip()
    try:
        ep_num = int(ep_raw)
        ep_id = f"episode-{ep_num:03d}"
    except Exception:
        ep_id = ep_raw

    parsed_file = Path(f"data/parsed/{ep_id}.json")
    if not parsed_file.exists():
        print(f"Error: Parsed file '{parsed_file}' does not exist.")
        sys.exit(1)

    with open(parsed_file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    start_s = float(row.get("approved_start_s", 0.0))
    end_s = float(row.get("approved_end_s", 0.0))
    dur_s = round(end_s - start_s, 2)
    decision = row.get("decision", "unknown")
    note = row.get("note", "")

    all_beats = parsed_data.get("beats", [])
    ep_final_end = max((b.get("end_s", 0.0) for b in all_beats), default=0.0)

    # Filter clip beats
    clip_beats = [
        b for b in all_beats
        if b.get("kind") != "CAMERA"
        and (b.get("start_s", 0.0) >= start_s - 0.05)
        and (b.get("end_s", 0.0) <= end_s + 0.05)
    ]
    if not clip_beats:
        clip_beats = [
            b for b in all_beats
            if b.get("kind") != "CAMERA"
            and (b.get("end_s", 0.0) > start_s)
            and (b.get("start_s", 0.0) < end_s)
        ]

    # Filter 12s after beats
    after_end = end_s + 12.0
    after_beats = [
        b for b in all_beats
        if b.get("kind") != "CAMERA"
        and (b.get("end_s", 0.0) > end_s)
        and (b.get("start_s", 0.0) < after_end)
    ]

    # Print to stdout only
    print("\n" + "=" * 80)
    print("NOTICE: This script outputs to terminal stdout only.")
    print("It writes nothing to any files or logs. Screenplay text is confidential.")
    print("=" * 80)
    print(f"GOLD ID   : {gold_id}")
    print(f"EPISODE   : {ep_id}")
    print(f"TIMING    : {start_s:.2f}s - {end_s:.2f}s ({dur_s:.2f}s)")
    print(f"DECISION  : {decision}")
    if note:
        print(f"NOTE      : {note}")
    print("-" * 80)
    print(f"--- CLIP BEATS ({len(clip_beats)} beats, local indices 1..{len(clip_beats)}) ---")
    for idx, b in enumerate(clip_beats, start=1):
        st = b.get("start_s", 0.0)
        en = b.get("end_s", 0.0)
        kind = b.get("kind", "")
        spk = b.get("speaker_norm") or "-"
        text = b.get("text", "")
        print(f"[{idx:>2}] ({st:>6.2f}s - {en:>6.2f}s, {kind:<8}, {spk}): {text}")

    print("-" * 80)
    print(f"--- 12s AFTER CLIP ({end_s:.2f}s .. {after_end:.2f}s) ---")
    if not after_beats:
        if end_s >= ep_final_end - 0.05:
            print("  [END OF EPISODE reached directly at clip end cut]")
        else:
            print("  [No non-camera beats in the 12s following clip end]")
    else:
        for idx, b in enumerate(after_beats, start=1):
            st = b.get("start_s", 0.0)
            en = b.get("end_s", 0.0)
            kind = b.get("kind", "")
            spk = b.get("speaker_norm") or "-"
            text = b.get("text", "")
            print(f"[+{idx}] ({st:>6.2f}s - {en:>6.2f}s, {kind:<8}, {spk}): {text}")

        if after_end >= ep_final_end - 0.05:
            print("  [END OF EPISODE reached within after window]")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

