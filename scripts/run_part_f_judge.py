"""Part F: Full Pipeline Judge Run on All 40 Non-Reserved Episodes.

Runs Judge v2 in per_episode mode on all 40 non-reserved episodes, picks only, one order.
Evaluates judge gates and applies the configured ranking decision rule (s3a_score_with_judge_gate).
Reports calls, tokens, latency, and rank-1 windows per episode as:
[start_s-end_s, duration, low_confidence]
Never prints screenplay text, loglines, continuity notes, or character names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.judge import run_judge_pipeline
from src.reserved import is_episode_blocked


def main() -> None:
    # All 50 episodes; reserved episodes are filtered out automatically
    all_episodes = [f"episode-{i:03d}" for i in range(1, 51)]
    active_episodes = [ep for ep in all_episodes if not is_episode_blocked(ep)]

    print(f"Running Part F Judge v2 (per_episode, picks only) across {len(active_episodes)} non-reserved episodes...")

    result = run_judge_pipeline(
        episodes=active_episodes,
        role="judge",
        mode="per_episode",
        prompt_version="judge_v2",
        enable_cache=True,
        allow_variants=False,
        report_file="data/part_f_judged_report.json",
    )

    # Gather telemetry from the run
    ep_results = result.get("episodes", [])
    
    total_calls = len(ep_results)
    cached_calls = 0
    non_cached_calls = 0
    non_cached_tokens_in = 0
    non_cached_tokens_out = 0
    non_cached_latency_s = 0.0

    rank1_windows = []

    for ep_payload in ep_results:
        ep_id = ep_payload.get("episode")
        judged = ep_payload.get("judged", [])

        # Telemetry is stored on each judged candidate from batch_res
        if judged:
            first_c = judged[0]
            is_cached = first_c.get("llm_cached", False)
            if is_cached:
                cached_calls += 1
            else:
                non_cached_calls += 1
                non_cached_tokens_in += first_c.get("llm_input_tokens", 0)
                non_cached_tokens_out += first_c.get("llm_output_tokens", 0)
                non_cached_latency_s += first_c.get("llm_latency_s", 0.0)

            # Rank 1 window
            # First candidate in judged list is rank 1
            r1 = judged[0]
            start_s = r1.get("start_s", 0.0)
            end_s = r1.get("end_s", 0.0)
            duration_s = r1.get("duration_s", round(end_s - start_s, 2))
            low_conf = r1.get("low_confidence", False)
            passed_judge = r1.get("passed_judge_gates", False)
            rank1_windows.append({
                "episode": ep_id,
                "window": f"[{start_s:.1f}s-{end_s:.1f}s, {duration_s:.1f}s, low_confidence={low_conf}]",
                "passed_judge_gates": passed_judge,
                "s3a_score": r1.get("s3a_score", 0.0),
            })

    print("\n" + "=" * 80)
    print("PART F: FULL PIPELINE EXECUTION REPORT")
    print("=" * 80)
    print(f"Episodes processed      : {len(active_episodes)}")
    print(f"Total judge calls        : {total_calls} (cached: {cached_calls}, non-cached: {non_cached_calls})")
    print(f"Non-cached input tokens  : {non_cached_tokens_in:,}")
    print(f"Non-cached output tokens : {non_cached_tokens_out:,}")
    print(f"Non-cached total latency : {non_cached_latency_s:.2f}s (mean: {non_cached_latency_s/max(1, non_cached_calls):.2f}s/call)")
    print("\nRank-1 Candidate Windows Per Episode:")
    for rw in rank1_windows:
        print(f"  {rw['episode']}: {rw['window']}")

    # Save summary json (counts and numbers only)
    out_summary = Path("data/part_f_rank1_summary.json")
    out_summary.write_text(json.dumps({
        "total_episodes": len(active_episodes),
        "total_calls": total_calls,
        "cached_calls": cached_calls,
        "non_cached_calls": non_cached_calls,
        "non_cached_tokens_in": non_cached_tokens_in,
        "non_cached_tokens_out": non_cached_tokens_out,
        "non_cached_latency_s": round(non_cached_latency_s, 2),
        "rank1_windows": rank1_windows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

