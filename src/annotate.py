"""Stage 2: Beat Annotation (src/annotate.py).

Performs LLM-based narrative beat annotation on non-camera beats:
{hook_potential (0-3), open_question_after (0-3), resolves_tension (bool),
 needs_context (0-3), intensity (0-3), emotion (str)}.
Includes episode summary, next_picks_up_immediately, and ends_on_cliffhanger.

Code maps local indices (1..N) to stable beat_ids.
The LLM never sees timestamps or beat_ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.llm import LLMResult, LLMQuotaExhaustedError, call, call_many, check_stage_quota, get_gateway
from src.reserved import RESERVED_EPISODES, is_episode_blocked


RESERVED_GOLD_EPISODES = {
    "episode-003",
    "episode-008",
    "episode-013",
    "episode-018",
    "episode-023",
    "episode-028",
    "episode-033",
    "episode-038",
    "episode-043",
    "episode-048",
}


class BeatAnnotation(BaseModel):
    i: int = Field(description="Local beat index (1..N)")
    h: int = Field(description="Hook potential (0-3)")
    q: int = Field(description="Open question after beat (0-3)")
    r: bool = Field(description="Resolves prior tension")
    n: int = Field(description="Needs prior context (0-3)")
    t: int = Field(description="Dramatic intensity (0-3)")
    e: str = Field(description="Primary emotion (single lowercase word)")


class EpisodeAnnotation(BaseModel):
    summary: str = Field(description="Episode dramatic summary (max 40 words)")
    next_picks_up_immediately: bool = Field(
        description="Whether narrative continues seamlessly into next episode"
    )
    ends_on_cliffhanger: bool = Field(
        description="Whether episode ends on unresolved high tension"
    )
    beats: List[BeatAnnotation] = Field(description="List of beat annotations for 1..N")


def load_prompt_rubric(prompt_path: str = "prompts/s2_v1.md") -> str:
    """Load rubric text from prompt file."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_episode_input(
    episode_data: Dict[str, Any]
) -> Tuple[str, Dict[int, str], List[Dict[str, Any]]]:
    """Format non-camera beats into sequential prompt input.

    Returns:
        (prompt_text, index_to_beat_id_map, non_camera_beats)
    """
    logline = episode_data.get("logline", "")
    continuity = episode_data.get("continuity_notes", "")
    all_beats = episode_data.get("beats", [])

    # Filter to non-camera beats only
    non_camera_beats = [b for b in all_beats if b.get("kind") != "CAMERA"]

    index_to_beat_id: Dict[int, str] = {}
    lines = [
        f"EPISODE LOGLINE: {logline}",
        f"CONTINUITY NOTES: {continuity}",
        "\nSEQUENTIAL DRAMATIC BEATS:",
    ]

    for idx, b in enumerate(non_camera_beats, start=1):
        index_to_beat_id[idx] = b["beat_id"]
        scene = b.get("scene", 1)
        kind = b.get("kind", "DIALOGUE")
        speaker_raw = b.get("speaker_raw") or b.get("speaker_norm") or ""
        text = b.get("text", "")
        cam_ctx = b.get("camera_context")
        closing_cam = b.get("closing_camera_context")

        spk_part = f" {speaker_raw}:" if speaker_raw else ":"
        line_str = f"[{idx}] (Scene {scene}) {kind}{spk_part} {text}"
        if cam_ctx:
            line_str += f" [cam: {cam_ctx}]"
        if closing_cam:
            line_str += f" [closing_cam: {closing_cam}]"

        lines.append(line_str)

    prompt_text = "\n".join(lines)
    return prompt_text, index_to_beat_id, non_camera_beats


def annotate_episode(
    episode_data: Dict[str, Any],
    role: str = "cheap",
    system_prompt: Optional[str] = None,
    prompt_version: str = "s2_v1",
    enable_cache: bool = True,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Annotate a single episode with validation, repair retry, and partial fill.

    Returns:
        (annotated_episode_dict, episode_telemetry_dict)
    """
    if system_prompt is None:
        system_prompt = load_prompt_rubric()

    ep_id = episode_data.get("episode", "unknown")
    user_prompt, idx_to_beat_id, non_cam_beats = build_episode_input(episode_data)
    expected_indices = set(range(1, len(non_cam_beats) + 1))
    total_expected = len(expected_indices)

    # Initial LLM call
    res: LLMResult = call(
        role=role,
        system=system_prompt,
        user=user_prompt,
        schema=EpisodeAnnotation,
        stage="s2_annotate",
        prompt_version=prompt_version,
        enable_cache=enable_cache,
        run_id=run_id,
    )

    calls_count = res.attempts
    in_tokens = res.input_tokens
    out_tokens = res.output_tokens
    thinking_tokens = res.thinking_tokens
    latency_s = res.latency_s
    model_name = res.model

    parsed: Optional[EpisodeAnnotation] = res.parsed

    # Verify all expected indices are present
    received_indices: Set[int] = set()
    if parsed and parsed.beats:
        received_indices = {b.i for b in parsed.beats}

    missing = expected_indices - received_indices
    is_partial = False

    # Perform exactly ONE repair retry if indices are missing or extra
    if missing or (parsed and len(parsed.beats) != total_expected):
        repair_msg = (
            f"Index Verification Error: Expected every index from 1 to {total_expected} exactly once.\n"
            f"Missing indices: {sorted(list(missing))}.\n"
            f"Total received: {len(received_indices)}.\n"
            f"Please return the full valid JSON with all indices from 1 to {total_expected}."
        )

        try:
            repair_res = call(
                role=role,
                system=system_prompt,
                user=f"{user_prompt}\n\n{repair_msg}",
                schema=EpisodeAnnotation,
                stage="s2_annotate_repair",
                prompt_version=prompt_version,
                enable_cache=enable_cache,
                run_id=run_id,
            )
            calls_count += repair_res.attempts
            in_tokens += repair_res.input_tokens
            out_tokens += repair_res.output_tokens
            thinking_tokens += repair_res.thinking_tokens
            latency_s += repair_res.latency_s

            if repair_res.parsed and repair_res.parsed.beats:
                parsed = repair_res.parsed
                received_indices = {b.i for b in parsed.beats}
                missing = expected_indices - received_indices
        except Exception:
            pass

    # Map annotations to beat_ids; fill missing indices with null
    annotations_map: Dict[str, Optional[Dict[str, Any]]] = {}
    parsed_dict: Dict[int, BeatAnnotation] = {}
    if parsed and parsed.beats:
        for b in parsed.beats:
            if b.i in expected_indices:
                parsed_dict[b.i] = b

    for idx in range(1, total_expected + 1):
        beat_id = idx_to_beat_id[idx]
        if idx in parsed_dict:
            b_obj = parsed_dict[idx]
            annotations_map[beat_id] = {
                "h": max(0, min(3, b_obj.h)),
                "q": max(0, min(3, b_obj.q)),
                "r": bool(b_obj.r),
                "n": max(0, min(3, b_obj.n)),
                "t": max(0, min(3, b_obj.t)),
                "e": b_obj.e.strip().lower(),
            }
        else:
            # Null fill for persistent missing index
            annotations_map[beat_id] = {
                "h": None,
                "q": None,
                "r": None,
                "n": None,
                "t": None,
                "e": None,
            }
            is_partial = True

    summary = parsed.summary if parsed else "Annotation failed"
    next_picks = parsed.next_picks_up_immediately if parsed else False
    ends_cliff = parsed.ends_on_cliffhanger if parsed else False

    annotated_output = {
        "episode": ep_id,
        "summary": summary,
        "next_picks_up_immediately": next_picks,
        "ends_on_cliffhanger": ends_cliff,
        "prompt_version": prompt_version,
        "model": model_name,
        "partial": is_partial,
        "annotations": annotations_map,
    }

    telemetry = {
        "episode": ep_id,
        "calls": calls_count,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "latency_s": round(latency_s, 3),
        "partial": is_partial,
    }

    return annotated_output, telemetry


def run_annotation_pipeline(
    episodes: Optional[List[str]] = None,
    parsed_dir: str | Path = "data/parsed",
    annotated_dir: str | Path = "data/annotated",
    report_file: str | Path = "annotate_report.json",
    role: str = "cheap",
    max_workers: int = 4,
    enable_cache: bool = True,
    run_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute S2 beat annotation across selected episodes."""
    parsed_path = Path(parsed_dir)
    annotated_path = Path(annotated_dir)
    annotated_path.mkdir(parents=True, exist_ok=True)

    if episodes:
        target_files = []
        for ep in episodes:
            candidate = parsed_path / f"{ep}.json"
            if candidate.exists():
                target_files.append(candidate)
            else:
                matches = list(parsed_path.glob(f"*{ep}*.json"))
                if matches:
                    target_files.append(matches[0])
    else:
        target_files = sorted(parsed_path.glob("*.json"))

    # Protect reserved gold set episodes from processing until unlocked
    filtered_targets = []
    for tf in target_files:
        if is_episode_blocked(tf.stem):
            continue
        filtered_targets.append(tf)
    target_files = filtered_targets

    if not target_files:
        print(f"No parsed episode files found to annotate in {parsed_dir}")
        return {}

    # Check 24h quota limit before starting (refuses unless force=True)
    check_stage_quota(role=role, required_calls=len(target_files), force=force)

    system_prompt = load_prompt_rubric()
    episodes_data: List[Dict[str, Any]] = []
    for tf in target_files:
        with open(tf, "r", encoding="utf-8") as f:
            episodes_data.append(json.load(f))

    # Concurrently execute annotations using gateway workers
    requests = []
    index_maps = []
    expected_counts = []

    for ep_data in episodes_data:
        user_prompt, idx_to_beat_id, non_cam_beats = build_episode_input(ep_data)
        requests.append(
            {
                "role": role,
                "system": system_prompt,
                "user": user_prompt,
                "schema": EpisodeAnnotation,
                "stage": "s2_annotate",
                "prompt_version": "s2_v1",
                "enable_cache": enable_cache,
                "run_id": run_id,
            }
        )
        index_maps.append(idx_to_beat_id)
        expected_counts.append(len(non_cam_beats))

    print(f"Running S2 annotation on {len(requests)} episode(s) using role='{role}' (workers={max_workers})...")
    results = call_many(requests, max_workers=max_workers, run_id=run_id)

    # Process and validate outputs
    annotated_reports: List[Dict[str, Any]] = []
    score_h = {0: 0, 1: 0, 2: 0, 3: 0}
    score_q = {0: 0, 1: 0, 2: 0, 3: 0}
    score_n = {0: 0, 1: 0, 2: 0, 3: 0}
    score_t = {0: 0, 1: 0, 2: 0, 3: 0}
    r_true_count = 0
    total_annotated_beats = 0
    next_picks_true = 0
    ends_cliff_true = 0

    total_in_tokens = 0
    total_out_tokens = 0
    total_thinking_tokens = 0
    total_latency_s = 0.0
    total_calls = 0

    for idx, (ep_data, res) in enumerate(zip(episodes_data, results)):
        ep_id = ep_data.get("episode", f"ep_{idx}")
        idx_to_beat_id = index_maps[idx]
        total_expected = expected_counts[idx]
        expected_indices = set(range(1, total_expected + 1))

        calls_count = res.attempts
        in_tokens = res.input_tokens
        out_tokens = res.output_tokens
        thinking_tokens = res.thinking_tokens
        latency_s = res.latency_s

        parsed: Optional[EpisodeAnnotation] = res.parsed
        received_indices = {b.i for b in parsed.beats} if (parsed and parsed.beats) else set()
        missing = expected_indices - received_indices
        is_partial = False

        # One repair retry on missing indices
        if missing or (parsed and len(parsed.beats) != total_expected):
            user_prompt, _, _ = build_episode_input(ep_data)
            repair_msg = (
                f"Index Error: Expected indices 1 to {total_expected} exactly once. "
                f"Missing: {sorted(list(missing))}. Return full valid JSON."
            )
            try:
                rep_res = call(
                    role=role,
                    system=system_prompt,
                    user=f"{user_prompt}\n\n{repair_msg}",
                    schema=EpisodeAnnotation,
                    stage="s2_annotate_repair",
                    prompt_version="s2_v1",
                    enable_cache=enable_cache,
                    run_id=run_id,
                )
                calls_count += rep_res.attempts
                in_tokens += rep_res.input_tokens
                out_tokens += rep_res.output_tokens
                thinking_tokens += rep_res.thinking_tokens
                latency_s += rep_res.latency_s

                if rep_res.parsed and rep_res.parsed.beats:
                    parsed = rep_res.parsed
                    received_indices = {b.i for b in parsed.beats}
            except Exception:
                pass

        # Build beat_id annotation mapping
        annotations_map: Dict[str, Optional[Dict[str, Any]]] = {}
        parsed_dict = {b.i: b for b in parsed.beats} if (parsed and parsed.beats) else {}

        for b_idx in range(1, total_expected + 1):
            b_id = idx_to_beat_id[b_idx]
            if b_idx in parsed_dict:
                b_obj = parsed_dict[b_idx]
                h_val = max(0, min(3, b_obj.h))
                q_val = max(0, min(3, b_obj.q))
                r_val = bool(b_obj.r)
                n_val = max(0, min(3, b_obj.n))
                t_val = max(0, min(3, b_obj.t))
                e_val = b_obj.e.strip().lower()

                annotations_map[b_id] = {
                    "h": h_val,
                    "q": q_val,
                    "r": r_val,
                    "n": n_val,
                    "t": t_val,
                    "e": e_val,
                }

                score_h[h_val] = score_h.get(h_val, 0) + 1
                score_q[q_val] = score_q.get(q_val, 0) + 1
                score_n[n_val] = score_n.get(n_val, 0) + 1
                score_t[t_val] = score_t.get(t_val, 0) + 1
                if r_val:
                    r_true_count += 1
                total_annotated_beats += 1
            else:
                annotations_map[b_id] = {
                    "h": None,
                    "q": None,
                    "r": None,
                    "n": None,
                    "t": None,
                    "e": None,
                }
                is_partial = True

        summary = parsed.summary if parsed else "Annotation failed"
        next_picks = bool(parsed.next_picks_up_immediately) if parsed else False
        ends_cliff = bool(parsed.ends_on_cliffhanger) if parsed else False

        if next_picks:
            next_picks_true += 1
        if ends_cliff:
            ends_cliff_true += 1

        total_in_tokens += in_tokens
        total_out_tokens += out_tokens
        total_thinking_tokens += thinking_tokens
        total_latency_s += latency_s
        total_calls += calls_count

        # Write output file data/annotated/<episode>.json
        ep_output = {
            "episode": ep_id,
            "summary": summary,
            "next_picks_up_immediately": next_picks,
            "ends_on_cliffhanger": ends_cliff,
            "prompt_version": "s2_v1",
            "model": res.model,
            "partial": is_partial,
            "annotations": annotations_map,
        }
        with open(annotated_path / f"{ep_id}.json", "w", encoding="utf-8") as f:
            json.dump(ep_output, f, indent=2)

        annotated_reports.append(
            {
                "episode": ep_id,
                "calls": calls_count,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "thinking_tokens": thinking_tokens,
                "latency_s": round(latency_s, 2),
                "cached": bool(res.cached),
                "partial": is_partial,
            }
        )

    ep_count = len(annotated_reports)
    cached_calls = sum(1 for r in annotated_reports if r.get("cached"))
    non_cached_calls = sum(1 for r in annotated_reports if not r.get("cached"))
    non_cached_in = sum(r["input_tokens"] for r in annotated_reports if not r.get("cached"))
    non_cached_out = sum(r["output_tokens"] for r in annotated_reports if not r.get("cached"))
    non_cached_lat = sum(r["latency_s"] for r in annotated_reports if not r.get("cached"))

    overall_summary = {
        "overall": {
            "total_episodes_annotated": ep_count,
            "total_calls": total_calls,
            "cached_calls": cached_calls,
            "non_cached_calls": non_cached_calls,
            "total_input_tokens": total_in_tokens,
            "total_output_tokens": total_out_tokens,
            "non_cached_input_tokens": non_cached_in,
            "non_cached_output_tokens": non_cached_out,
            "total_thinking_tokens": total_thinking_tokens,
            "total_latency_s": round(total_latency_s, 2),
            "non_cached_latency_s": round(non_cached_lat, 2),
            "score_distributions": {
                "h": score_h,
                "q": score_q,
                "n": score_n,
                "t": score_t,
            },
            "share_r_true": round(r_true_count / max(1, total_annotated_beats), 3),
            "share_next_picks_up_immediately": round(next_picks_true / max(1, ep_count), 3),
            "share_ends_on_cliffhanger": round(ends_cliff_true / max(1, ep_count), 3),
            "partial_episodes": [r["episode"] for r in annotated_reports if r["partial"]],
        },
        "episodes": annotated_reports,
    }

    # Write annotate_report.json (no screenplay text)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    return overall_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S2 Beat Annotation.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="List of episode IDs to annotate (e.g. episode-001 episode-002)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate all episodes found in data/parsed",
    )
    parser.add_argument(
        "--role",
        default="cheap",
        help="Model role to use (default: cheap)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent workers for call_many",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable gateway cache",
    )
    parser.add_argument(
        "--cache-key-extra",
        default=None,
        dest="cache_key_extra",
        help="Extra string appended to cache key (e.g. 'run2') to force cache miss",
    )
    parser.add_argument(
        "--report-file",
        default="annotate_report.json",
        help="Path for output report JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass daily quota limit check",
    )

    args = parser.parse_args()

    if not args.all and not args.episodes:
        print("Please specify --episodes <id1 id2...> or --all")
        sys.exit(1)

    episodes = None if args.all else args.episodes
    try:
        report = run_annotation_pipeline(
            episodes=episodes,
            role=args.role,
            max_workers=args.workers,
            enable_cache=not args.no_cache,
            run_id=args.cache_key_extra,
            report_file=args.report_file,
            force=args.force,
        )
    except LLMQuotaExhaustedError as e:
        print(f"\n[QUOTA_EXHAUSTED] {e}")
        print("Pipeline stopped cleanly. Previous results remain cached.")
        sys.exit(1)
    overall = report.get("overall", {})
    print("S2 Annotation complete.")
    print(f"Episodes annotated: {overall.get('total_episodes_annotated')}")
    print(f"Total tokens in: {overall.get('total_input_tokens')}, out: {overall.get('total_output_tokens')}")
    print(f"Score distributions: {overall.get('score_distributions')}")


if __name__ == "__main__":
    main()
