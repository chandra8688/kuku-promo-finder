"""Validation and Ingest for Human Gold Standard Set (scripts/validate_gold.py).

Uses data/parsed only. For each non-rejected row in gold/audit.csv, validates:
- duration is between 20.0 and 60.0 s;
- start is within 0.05 s of a non-camera beat's start_s;
- end is within 0.05 s of a spoken beat's end_s (Rule 1), or later by at most 0.5 s (Rule 2),
  OR equals the episode's final end_s (Rule 3);
- no DIALOGUE or SFX beat straddles the start or the end;
- for the 12 s after the end: number of spoken beats and whether the episode ends within that time.

Builds gold/gold.csv and gold/gold_public.csv for all passing rows.
Never changes times automatically.
Never prints screenplay text or character names.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def validate_boundary_rules(
    approved_start_s: float,
    approved_end_s: float,
    beats: List[Dict[str, Any]],
    episode_final_end_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate boundary and duration rules against parsed episode beats.
    
    Returns a dict with validation results, matched rules, and nearest valid boundaries.
    """
    if episode_final_end_s is None:
        episode_final_end_s = max((b.get("end_s", 0.0) for b in beats), default=0.0)

    duration_s = round(approved_end_s - approved_start_s, 2)
    duration_ok = 20.0 <= duration_s <= 60.0

    non_cam_beats = [b for b in beats if b.get("kind") != "CAMERA"]
    spoken_beats = [b for b in beats if b.get("kind") == "DIALOGUE"]

    # 1. Start boundary check: within 0.05s of any non-camera beat start_s
    matching_starts = [
        b for b in non_cam_beats
        if abs(b.get("start_s", 0.0) - approved_start_s) <= 0.05 + 1e-4
    ]
    start_ok = len(matching_starts) > 0
    nearest_start_b = min(
        non_cam_beats,
        key=lambda b: abs(b.get("start_s", 0.0) - approved_start_s),
        default=None,
    )
    nearest_start_s = nearest_start_b.get("start_s", 0.0) if nearest_start_b else None

    # 2. End boundary check:
    # Rule 1: within 0.05s of spoken beat's end_s
    rule1_matches = [
        b for b in spoken_beats
        if abs(b.get("end_s", 0.0) - approved_end_s) <= 0.05 + 1e-4
    ]
    # Rule 2: later by at most 0.5s from spoken beat's end_s (0.0 < approved_end - beat_end <= 0.50)
    rule2_matches = [
        b for b in spoken_beats
        if 0.0 < (approved_end_s - b.get("end_s", 0.0)) <= 0.50 + 1e-4
    ]
    # Rule 3: equals episode's final end_s
    rule3_match = abs(approved_end_s - episode_final_end_s) <= 0.05 + 1e-4

    end_rule_matched: Optional[str] = None
    if rule1_matches:
        end_rule_matched = "RULE_1_SPOKEN_END_EXACT"
    elif rule2_matches:
        end_rule_matched = "RULE_2_SPOKEN_END_LATER_LE_0_5S"
    elif rule3_match:
        end_rule_matched = "RULE_3_EPISODE_FINAL_END"

    end_ok = end_rule_matched is not None

    nearest_spoken_b = min(
        spoken_beats,
        key=lambda b: abs(b.get("end_s", 0.0) - approved_end_s),
        default=None,
    )
    nearest_end_s = nearest_spoken_b.get("end_s", 0.0) if nearest_spoken_b else episode_final_end_s

    # 3. Straddle check: no DIALOGUE or SFX beat straddles start or end
    # b_start + 0.01 < cut < b_end - 0.01
    eps = 0.01
    straddles_start = [
        b.get("beat_id") for b in beats
        if b.get("kind") in ("DIALOGUE", "SFX")
        and (b.get("start_s", 0.0) + eps < approved_start_s < b.get("end_s", 0.0) - eps)
    ]
    straddles_end = [
        b.get("beat_id") for b in beats
        if b.get("kind") in ("DIALOGUE", "SFX")
        and (b.get("start_s", 0.0) + eps < approved_end_s < b.get("end_s", 0.0) - eps)
    ]
    straddle_ok = (len(straddles_start) == 0) and (len(straddles_end) == 0)

    # 4. 12s after the end: number of spoken beats and whether episode ends
    after_window_end = approved_end_s + 12.0
    after_spoken_beats = [
        b for b in spoken_beats
        if approved_end_s < b.get("end_s", 0.0) and b.get("start_s", 0.0) < after_window_end
    ]
    episode_ends_in_12s = episode_final_end_s <= after_window_end + 1e-4

    failures = []
    if not duration_ok:
        failures.append(f"DURATION_OUT_OF_BOUNDS({duration_s}s)")
    if not start_ok:
        failures.append(f"START_NOT_ALIGNED(nearest={nearest_start_s}s)")
    if not end_ok:
        failures.append(f"END_NOT_ALIGNED(nearest={nearest_end_s}s)")
    if not straddle_ok:
        failures.append(f"STRADDLE_DETECTED(start={straddles_start}, end={straddles_end})")

    passed = len(failures) == 0

    return {
        "passed": passed,
        "failures": failures,
        "duration_s": duration_s,
        "duration_ok": duration_ok,
        "start_ok": start_ok,
        "nearest_start_s": nearest_start_s,
        "end_ok": end_ok,
        "end_rule_matched": end_rule_matched,
        "nearest_end_s": nearest_end_s,
        "straddle_ok": straddle_ok,
        "straddles_start": straddles_start,
        "straddles_end": straddles_end,
        "after_12s_spoken_count": len(after_spoken_beats),
        "after_12s_episode_ends": episode_ends_in_12s,
        "episode_final_end_s": episode_final_end_s,
    }


def validate_gold_audit(
    audit_file: str | Path = "gold/audit.csv",
    parsed_dir: str | Path = "data/parsed",
) -> List[Dict[str, Any]]:
    """Validate all non-rejected rows from gold/audit.csv."""
    audit_p = Path(audit_file)
    parsed_p = Path(parsed_dir)

    if not audit_p.exists():
        raise FileNotFoundError(f"Missing audit file: {audit_p}")

    with open(audit_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []

    for r in rows:
        decision = r.get("decision", "").strip().lower()
        if decision == "reject":
            results.append({
                "row": r,
                "gold_id": r.get("gold_id"),
                "episode": r.get("episode"),
                "excluded": True,
                "exclude_reason": "REJECTED_IN_AUDIT",
                "validation": None,
            })
            continue

        gold_id = r.get("gold_id", "").strip()
        ep_raw = r.get("episode", "").strip()
        try:
            ep_num = int(ep_raw)
            ep_id = f"episode-{ep_num:03d}"
        except Exception:
            ep_id = ep_raw

        approved_start_s = float(r.get("approved_start_s", 0.0))
        approved_end_s = float(r.get("approved_end_s", 0.0))

        ep_parsed_file = parsed_p / f"{ep_id}.json"
        if not ep_parsed_file.exists():
            results.append({
                "row": r,
                "gold_id": gold_id,
                "episode": ep_raw,
                "excluded": True,
                "exclude_reason": f"PARSED_FILE_NOT_FOUND({ep_parsed_file})",
                "validation": None,
            })
            continue

        with open(ep_parsed_file, "r", encoding="utf-8") as pf:
            ep_data = json.load(pf)

        beats = ep_data.get("beats", [])
        ep_final_end = max((b.get("end_s", 0.0) for b in beats), default=0.0)

        v_res = validate_boundary_rules(
            approved_start_s=approved_start_s,
            approved_end_s=approved_end_s,
            beats=beats,
            episode_final_end_s=ep_final_end,
        )

        results.append({
            "row": r,
            "gold_id": gold_id,
            "episode": ep_raw,
            "excluded": not v_res["passed"],
            "exclude_reason": "; ".join(v_res["failures"]) if not v_res["passed"] else None,
            "validation": v_res,
        })

    return results


def print_validation_report(results: List[Dict[str, Any]]) -> None:
    """Print a clean PASS/FAIL table (counts and times only, no text)."""
    print("\n" + "=" * 100)
    print("GOLD SET VALIDATION REPORT (scripts/validate_gold.py)")
    print("=" * 100)
    header = (
        f"{'Gold ID':<12} | {'Ep':<3} | {'Times (s)':<16} | {'Dur(s)':<6} | "
        f"{'Start':<5} | {'End Rule':<28} | {'Straddle':<8} | {'+12s Spk':<8} | {'Status':<6}"
    )
    print(header)
    print("-" * 100)

    for res in results:
        gid = res["gold_id"]
        ep = res["episode"]
        r = res["row"]
        v = res.get("validation")

        if v is None:
            reason = res.get("exclude_reason", "EXCLUDED")
            print(f"{gid:<12} | {ep:<3} | {'EXCLUDED':<16} | {'-':<6} | {'-':<5} | {reason:<28} | {'-':<8} | {'-':<8} | FAIL")
            continue

        times_str = f"{float(r['approved_start_s']):.2f} - {float(r['approved_end_s']):.2f}"
        dur_str = f"{v['duration_s']:.1f}"
        st_str = "PASS" if v["start_ok"] else f"FAIL({v['nearest_start_s']}s)"
        rule_str = v["end_rule_matched"] or f"FAIL(near={v['nearest_end_s']}s)"
        strad_str = "PASS" if v["straddle_ok"] else "FAIL"
        after_str = f"{v['after_12s_spoken_count']} spk{' (END)' if v['after_12s_episode_ends'] else ''}"
        status_str = "PASS" if v["passed"] else "FAIL"

        print(
            f"{gid:<12} | {ep:<3} | {times_str:<16} | {dur_str:<6} | "
            f"{st_str:<5} | {rule_str:<28} | {strad_str:<8} | {after_str:<8} | {status_str:<6}"
        )

    print("-" * 100)
    total_evaluated = len(results)
    total_passed = sum(1 for r in results if r.get("validation") and r["validation"]["passed"])
    total_excluded = total_evaluated - total_passed
    print(f"Summary: {total_passed}/{total_evaluated} passed, {total_excluded} excluded.")
    print("=" * 100 + "\n")


def build_gold_files(
    results: List[Dict[str, Any]],
    gold_csv_path: str | Path = "gold/gold.csv",
    gold_public_csv_path: str | Path = "gold/gold_public.csv",
) -> Tuple[int, List[str]]:
    """Build gold/gold.csv and gold/gold_public.csv for passing rows.
    
    Status mapping:
    - human_confirmed empty -> status: "ai_reviewed", source: "ai_proposed_ai_reviewed"
    - human_confirmed="yes" -> status: "human_audited", source: "human_audited"
    """
    gold_p = Path(gold_csv_path)
    public_p = Path(gold_public_csv_path)

    gold_p.parent.mkdir(parents=True, exist_ok=True)
    public_p.parent.mkdir(parents=True, exist_ok=True)

    gold_rows = []
    public_rows = []
    excluded_rows = []

    for res in results:
        v = res.get("validation")
        r = res["row"]
        gid = res["gold_id"]
        ep = res["episode"]

        if v is None or not v["passed"]:
            excluded_rows.append(f"{gid} (Ep {ep}): {res.get('exclude_reason', 'Validation failed')}")
            continue

        hc = r.get("human_confirmed", "").strip().lower()
        if hc in ("yes", "true", "1"):
            status = "human_audited"
            source = "human_audited"
            hc_val = "yes"
        else:
            status = "ai_reviewed"
            source = "ai_proposed_ai_reviewed"
            hc_val = ""

        st_s = f"{float(r['approved_start_s']):.2f}"
        en_s = f"{float(r['approved_end_s']):.2f}"

        gold_rows.append({
            "gold_id": gid,
            "episode": ep,
            "seg_order": "1",
            "start_s": st_s,
            "end_s": en_s,
            "status": status,
            "source": source,
            "human_confirmed": hc_val,
        })

        public_rows.append({
            "gold_id": gid,
            "episode": ep,
            "seg_order": "1",
            "start_s": st_s,
            "end_s": en_s,
            "status": status,
        })

    # Write gold/gold.csv
    with open(gold_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "gold_id",
            "episode",
            "seg_order",
            "start_s",
            "end_s",
            "status",
            "source",
            "human_confirmed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_rows)

    # Write gold/gold_public.csv
    with open(public_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["gold_id", "episode", "seg_order", "start_s", "end_s", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)

    return len(gold_rows), excluded_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gold candidate boundaries and build gold files.")
    parser.add_argument("--audit-file", default="gold/audit.csv", help="Path to gold audit.csv")
    parser.add_argument("--parsed-dir", default="data/parsed", help="Path to data/parsed directory")
    parser.add_argument("--build-gold", action="store_true", default=True, help="Build gold/gold.csv and gold/gold_public.csv")
    args = parser.parse_args()

    results = validate_gold_audit(audit_file=args.audit_file, parsed_dir=args.parsed_dir)
    print_validation_report(results)

    if args.build_gold:
        n_written, excluded = build_gold_files(results)
        print(f"Built gold files with {n_written} verified rows.")
        if excluded:
            print("Excluded rows:")
            for e in excluded:
                print(f"  - {e}")
        else:
            print("Excluded rows: None (all audit candidates passed validation).")


if __name__ == "__main__":
    main()

