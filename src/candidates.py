"""Candidate generation and gate evaluation runner (src/candidates.py).

Integrates:
- S3a: Window enumeration and scoring (src/windows.py)
- S4: Hard gate filtering (src/gates.py)
- Generates data/candidates/<ep>.json (gitignored)
- Generates candidates_report.json (counts only, no screenplay text)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.reserved import is_episode_blocked
from src.windows import load_scoring_config, load_v2_config, scoring_provenance


def process_episode_candidates(
    ep_id: str,
    parsed_dir: str | Path = "data/parsed",
    annotated_dir: str | Path = "data/annotated",
    candidates_dir: str | Path = "data/candidates",
    gold_file: str | Path = "gold/gold.csv",
    scoring_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single episode through window enumeration and hard gates."""
    # Check reserved episodes rule
    if is_episode_blocked(ep_id, gold_file=gold_file):
        return None

    parsed_path = Path(parsed_dir) / f"{ep_id}.json"
    annotated_path = Path(annotated_dir) / f"{ep_id}.json"

    if not parsed_path.exists():
        print(f"Parsed file not found for {ep_id}: {parsed_path}")
        return None
    if not annotated_path.exists():
        print(f"Annotated file not found for {ep_id}: {annotated_path}")
        return None

    with open(parsed_path, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)
    with open(annotated_path, "r", encoding="utf-8") as f:
        annotated_data = json.load(f)

    if scoring_config is None:
        scoring_config = load_scoring_config()

    # Weights come from scoring_v2.yaml and nowhere else. scoring.yaml still
    # supplies thresholds, but its v1 weight block must never reach the v2 scorer:
    # there context is a positive weight, so it would reward context need.
    weights = dict(load_v2_config().get("weights") or {})
    top_k = scoring_config.get("thresholds", {}).get("top_k", 3)

    # S3a + S4 through the shared engine, so this is the exact procedure that
    # eval_v2.py and tune_weights.py measure
    from src.engine import run_episode

    windows, all_evaluated, survivors, gate_fail_counts = run_episode(
        parsed_data, annotated_data, weights=weights, top_k=top_k
    )

    fallback_count = sum(1 for w in survivors if w.get("low_confidence", False))

    # Save to data/candidates/<ep>.json
    cand_path = Path(candidates_dir)
    cand_path.mkdir(parents=True, exist_ok=True)
    out_file = cand_path / f"{ep_id}.json"

    ep_payload = {
        "episode": ep_id,
        "total_windows_enumerated": len(windows),
        "total_survivors": len(survivors),
        "fallback_survivors": fallback_count,
        "gate_fail_counts": gate_fail_counts,
        "scoring_provenance": scoring_provenance(),
        "annotation_partial": bool(annotated_data.get("partial", False)) if isinstance(annotated_data, dict) else False,
        "candidates": all_evaluated,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ep_payload, f, indent=2)

    return {
        "episode": ep_id,
        "windows_enumerated": len(windows),
        "survivors_count": len(survivors),
        "fallback_count": fallback_count,
        "top_k_survivors": [w for w in survivors if w.get("selected_top_k", False)],
        "gate_fail_counts": gate_fail_counts,
        "all_windows": all_evaluated,
    }


def run_candidate_pipeline(
    episodes: List[str],
    parsed_dir: str | Path = "data/parsed",
    annotated_dir: str | Path = "data/annotated",
    candidates_dir: str | Path = "data/candidates",
    report_file: str | Path = "candidates_report.json",
    gold_file: str | Path = "gold/gold.csv",
) -> Dict[str, Any]:
    """Run window search and hard gates for specified episodes."""
    scoring_config = load_scoring_config()
    results: List[Dict[str, Any]] = []

    overall_gate_counts: Dict[str, int] = {}
    total_enumerated = 0
    total_survivors = 0
    zero_survivor_episodes: List[str] = []

    for ep in episodes:
        res = process_episode_candidates(
            ep_id=ep,
            parsed_dir=parsed_dir,
            annotated_dir=annotated_dir,
            candidates_dir=candidates_dir,
            gold_file=gold_file,
            scoring_config=scoring_config,
        )
        if res is None:
            continue

        results.append(res)
        total_enumerated += res["windows_enumerated"]
        total_survivors += res["survivors_count"]

        if res["survivors_count"] == 0:
            zero_survivor_episodes.append(res["episode"])

        for gate, cnt in res["gate_fail_counts"].items():
            overall_gate_counts[gate] = overall_gate_counts.get(gate, 0) + cnt

    summary_report = {
        "overall": {
            "episodes_processed": len(results),
            "total_windows_enumerated": total_enumerated,
            "total_survivors": total_survivors,
            "gate_fail_counts": overall_gate_counts,
            "zero_survivor_episodes": zero_survivor_episodes,
        },
        "episodes": [
            {
                "episode": r["episode"],
                "windows_enumerated": r["windows_enumerated"],
                "survivors_count": r["survivors_count"],
                "gate_fail_counts": r["gate_fail_counts"],
            }
            for r in results
        ],
    }

    # Write report file (counts only, no screenplay text)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    return {"summary": summary_report, "details": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3a Window Search and S4 Hard Gates.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["episode-001", "episode-002", "episode-005"],
        help="Episodes to evaluate",
    )
    parser.add_argument(
        "--report-file",
        default="candidates_report.json",
        help="Path for summary report output",
    )
    args = parser.parse_args()

    run_candidate_pipeline(episodes=args.episodes, report_file=args.report_file)


if __name__ == "__main__":
    main()

