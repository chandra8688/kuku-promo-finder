"""Primary Pipeline Run on 10 Reserved Episodes (scripts/run_primary.py).

Runs:
1. S2 Annotation (cheap role, cache enabled, run label "primary")
2. S3a/S4 Candidate Generation & Gate Evaluation (frozen scoring config)
3. S5 Judge v2 (per_episode mode, judge role, picks only, frozen ranking rule)

Reports calls, non-cached tokens, and latency.
Never prints screenplay text, loglines, or character names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.judge import run_judge_pipeline
from src.reserved import RESERVED_EPISODES, is_episode_blocked

RESERVED_LIST = sorted(list(RESERVED_EPISODES))


def main() -> None:
    print(f"Starting Primary Run on {len(RESERVED_LIST)} reserved episodes: {RESERVED_LIST}")

    # Stage 2: Annotation
    print("\n[Primary] Stage 2: Annotation (cheap role)...")
    s2_report = run_annotation_pipeline(
        episodes=RESERVED_LIST,
        role="cheap",
        max_workers=2,
        enable_cache=True,
        run_id="primary",
        report_file="data/primary_s2_report.json",
    )

    # Stage 3a / 4: Candidates & Gates
    print("\n[Primary] Stage 3a/4: Window Enumeration & Gate Filtering...")
    s3_results = []
    for ep in RESERVED_LIST:
        res = process_episode_candidates(ep)
        if res:
            s3_results.append(res)
    print(f"[Primary] Completed S3a/S4 for {len(s3_results)} episodes.")

    # Stage 5: Judge
    print("\n[Primary] Stage 5: Judge v2 (judge role, per_episode mode)...")
    judge_result = run_judge_pipeline(
        episodes=RESERVED_LIST,
        role="judge",
        max_workers=2,
        enable_cache=True,
        mode="per_episode",
        prompt_version="judge_v2",
        allow_variants=False,
        report_file="data/primary_judge_report.json",
    )

    # Compute telemetry
    s2_overall = s2_report.get("overall", {})
    s2_calls = s2_overall.get("total_calls", 0)
    s2_cached = s2_overall.get("cached_calls", 0)
    s2_non_cached = s2_overall.get("non_cached_calls", 0)
    s2_in = s2_overall.get("non_cached_input_tokens", 0)
    s2_out = s2_overall.get("non_cached_output_tokens", 0)
    s2_lat = s2_overall.get("non_cached_latency_s", 0.0)

    # Judge telemetry from candidate records
    judge_eps = judge_result.get("episodes", [])
    j_calls = len(judge_eps)
    j_cached = 0
    j_non_cached = 0
    j_in = 0
    j_out = 0
    j_lat = 0.0

    rank1_windows = []

    for ep_payload in judge_eps:
        ep_id = ep_payload.get("episode")
        judged = ep_payload.get("judged", [])
        if judged:
            first_c = judged[0]
            is_c = first_c.get("llm_cached", False)
            if is_c:
                j_cached += 1
            else:
                j_non_cached += 1
                j_in += first_c.get("llm_input_tokens", 0)
                j_out += first_c.get("llm_output_tokens", 0)
                j_lat += first_c.get("llm_latency_s", 0.0)

            st = first_c.get("start_s", 0.0)
            en = first_c.get("end_s", 0.0)
            dur = first_c.get("duration_s", round(en - st, 2))
            low_c = first_c.get("low_confidence", False)
            rank1_windows.append({
                "episode": ep_id,
                "window": f"[{st:.1f}s-{en:.1f}s, {dur:.1f}s, low_confidence={low_c}]",
                "start_s": st,
                "end_s": en,
                "duration_s": dur,
                "low_confidence": low_c,
                "candidate_id": first_c.get("candidate_id"),
            })

    total_calls = s2_calls + j_calls
    total_non_cached_calls = s2_non_cached + j_non_cached
    total_in = s2_in + j_in
    total_out = s2_out + j_out
    total_lat = s2_lat + j_lat

    print("\n" + "=" * 80)
    print("PRIMARY RUN REPORT (10 Reserved Episodes)")
    print("=" * 80)
    print(f"Episodes processed       : {len(RESERVED_LIST)}")
    print(f"S2 Annotate calls        : {s2_calls} (cached: {s2_cached}, non-cached: {s2_non_cached})")
    print(f"S2 Non-cached tokens in  : {s2_in:,}")
    print(f"S2 Non-cached tokens out : {s2_out:,}")
    print(f"S2 Non-cached latency    : {s2_lat:.2f}s")
    print(f"S5 Judge calls           : {j_calls} (cached: {j_cached}, non-cached: {j_non_cached})")
    print(f"S5 Non-cached tokens in  : {j_in:,}")
    print(f"S5 Non-cached tokens out : {j_out:,}")
    print(f"S5 Non-cached latency    : {j_lat:.2f}s")
    print("-" * 80)
    print(f"TOTAL calls              : {total_calls} (non-cached: {total_non_cached_calls})")
    print(f"TOTAL non-cached tokens  : In={total_in:,}, Out={total_out:,} (Total={total_in+total_out:,})")
    print(f"TOTAL non-cached latency : {total_lat:.2f}s")
    print("\nRank-1 Windows per Reserved Episode:")
    for rw in rank1_windows:
        print(f"  {rw['episode']}: {rw['window']}")

    # Save summary json (counts and numbers only)
    out_summary = Path("data/primary_run_summary.json")
    out_summary.write_text(json.dumps({
        "episodes": RESERVED_LIST,
        "s2": {
            "calls": s2_calls,
            "cached": s2_cached,
            "non_cached": s2_non_cached,
            "tokens_in": s2_in,
            "tokens_out": s2_out,
            "latency_s": round(s2_lat, 2),
        },
        "judge": {
            "calls": j_calls,
            "cached": j_cached,
            "non_cached": j_non_cached,
            "tokens_in": j_in,
            "tokens_out": j_out,
            "latency_s": round(j_lat, 2),
        },
        "total": {
            "calls": total_calls,
            "non_cached_calls": total_non_cached_calls,
            "tokens_in": total_in,
            "tokens_out": total_out,
            "latency_s": round(total_lat, 2),
        },
        "rank1_windows": rank1_windows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

