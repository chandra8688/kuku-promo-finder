#!/usr/bin/env python3
"""Unified CLI Entry Point for Promo Finder (run.py).

Executes the full end-to-end pipeline:
1. Parse raw screenplay text (.txt) -> validated beats and metadata (S1)
2. Beat-level narrative annotation (S2) or Tier-0 deterministic heuristic (--no-llm)
3. Window enumeration, structural and tension gates, and deduplication (S3a / S4)
4. Candidate evaluation, clarity assessment, and ranking (Judge v2)
5. Per-episode promo exports (promos_per_episode.json and .csv)
6. Show-level top 15 ranking with diversity cap of <= 2 per episode (promos_ranked.json and .csv)
7. Run manifest and execution telemetry (run_manifest.json)

Confidentiality: Screenplay dialogue, loglines, and character names are never printed to terminal or logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.judge import run_judge_pipeline
from src.llm import LLMQuotaExhaustedError, call, get_gateway
from src.parse import ParseError, ParseWarning, load_aliases, parse_file
from src.reserved import is_episode_blocked
from src.tier0 import generate_tier0_candidates_for_episode


class ListwiseCandidateRank(BaseModel):
    model_config = ConfigDict(extra="ignore")
    candidate_num: int = Field(description="Local candidate number (1..M)")
    rank: int = Field(description="Strictly unique rank 1..M (1 = best promo across entire show to post first)")
    why: str = Field(description="Short reason for rank (max 20 words)")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "candidate_num" not in data:
                for k in ("candidate", "num", "id", "candidate_id", "candidate_number"):
                    if k in data:
                        data["candidate_num"] = data[k]
                        break
        return data


class ListwiseRankingOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rankings: List[ListwiseCandidateRank] = Field(default_factory=list, description="Rankings for all candidates 1..M")

    @model_validator(mode="before")
    @classmethod
    def handle_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "rankings" not in data:
                for k in ("evaluations", "candidates", "results", "ranks", "items"):
                    if k in data:
                        data["rankings"] = data[k]
                        break
        elif isinstance(data, list):
            return {"rankings": data}
        return data


def format_seconds_to_mmss(s: float) -> str:
    """Format floating point seconds as mm:ss.ss."""
    m = int(s // 60)
    sec = s % 60
    return f"{m:02d}:{sec:05.2f}"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def build_per_episode_promos(
    episodes: List[str],
    parsed_dir: Path,
    candidates_dir: Path,
    judged_dir: Path,
    out_dir: Path,
    no_llm: bool = False,
) -> List[Dict[str, Any]]:
    """Format and export per-episode promos in JSON and CSV."""
    all_promos: List[Dict[str, Any]] = []

    for ep_id in episodes:
        p_file = parsed_dir / f"{ep_id}.json"
        if not p_file.exists():
            continue

        with open(p_file, "r", encoding="utf-8") as pf:
            p_data = json.load(pf)
        beats = p_data.get("beats", [])
        beat_lookup = {b["beat_id"]: b for b in beats}

        # Check judged file first, then candidate file
        j_file = judged_dir / f"{ep_id}.json"
        c_file = candidates_dir / f"{ep_id}.json"

        cands = []
        is_tier0_ep = no_llm
        tier_label = "tier0" if no_llm else "primary"

        if j_file.exists():
            with open(j_file, "r", encoding="utf-8") as jf:
                j_data = json.load(jf)
            cands = j_data.get("judged", [])
            tier_label = j_data.get("tier", tier_label)
        elif c_file.exists():
            with open(c_file, "r", encoding="utf-8") as cf:
                c_data = json.load(cf)
            cands = [c for c in c_data.get("candidates", []) if c.get("passed_all_gates")]
            tier_label = c_data.get("tier", "tier0_fallback")
            is_tier0_ep = True

        # Sort candidates: passed_judge_gates first, then s3a_score desc, then final_score desc
        cands.sort(
            key=lambda x: (
                1 if x.get("passed_judge_gates", True) else 0,
                x.get("s3a_score", x.get("score", 0.0)),
                x.get("final_score", 0),
            ),
            reverse=True,
        )

        for rank_idx, cand in enumerate(cands, start=1):
            st = float(cand.get("start_s", 0.0))
            en = float(cand.get("end_s", 0.0))
            dur = float(cand.get("duration_s", round(en - st, 2)))
            beat_ids = cand.get("beat_ids", [])

            h_idx = cand.get("hook_index", 1) or 1
            c_idx = cand.get("cliff_index", len(beat_ids)) or len(beat_ids)

            h_beat_id = beat_ids[h_idx - 1] if (1 <= h_idx <= len(beat_ids)) else (beat_ids[0] if beat_ids else "")
            c_beat_id = beat_ids[c_idx - 1] if (1 <= c_idx <= len(beat_ids)) else (beat_ids[-1] if beat_ids else "")

            hook_line = beat_lookup.get(h_beat_id, {}).get("text", "")
            cliff_line = beat_lookup.get(c_beat_id, {}).get("text", "")

            seg_sec = f"[{st:.2f}, {en:.2f}]"
            seg_mmss = f"{format_seconds_to_mmss(st)}-{format_seconds_to_mmss(en)}"

            scores = {
                "s3a_score": cand.get("s3a_score", cand.get("score", 0.0)),
                "judge_final_score": cand.get("final_score", 0),
                "hook": cand.get("hook", 0),
                "cliff": cand.get("cliff", 0),
                "clarity": cand.get("clarity", 0),
                "payoff": cand.get("payoff", 0),
                "resolved_after": cand.get("resolved_after", False),
            }

            cand_tier = cand.get("tier", tier_label)

            promo_entry = {
                "episode": ep_id,
                "rank_in_episode": rank_idx,
                "candidate_id": cand.get("candidate_id", f"{ep_id}_w{rank_idx:04d}"),
                "segments": [
                    {
                        "start_s": st,
                        "end_s": en,
                        "start_time": format_seconds_to_mmss(st),
                        "end_time": format_seconds_to_mmss(en),
                    }
                ],
                "segments_seconds": seg_sec,
                "segments_mmss": seg_mmss,
                "start_s": st,
                "end_s": en,
                "duration_s": dur,
                "hook_line": hook_line,
                "cliffhanger_line": cliff_line,
                "why": cand.get("why", "Heuristic proxy selection" if is_tier0_ep else ""),
                "scores": scores,
                "low_confidence": cand.get("low_confidence", False),
                "tier": cand_tier,
                "provenance": {
                    "annotator_model": "none" if no_llm else ("gemini-3.5-flash-lite" if cand_tier != "tier0_fallback" else "none"),
                    "annotator_prompt": "tier0" if is_tier0_ep else "s2_v1",
                    "judge_model": "none" if is_tier0_ep else "gemini-3.1-flash-lite",
                    "judge_prompt": "tier0" if is_tier0_ep else "judge_v2",
                },
            }
            all_promos.append(promo_entry)

    # Export promos_per_episode.json
    json_path = out_dir / "promos_per_episode.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_promos, f, indent=2)

    # Export promos_per_episode.csv
    csv_path = out_dir / "promos_per_episode.csv"
    csv_fields = [
        "episode",
        "rank_in_episode",
        "candidate_id",
        "segments_seconds",
        "segments_mmss",
        "duration_s",
        "s3a_score",
        "judge_final_score",
        "hook",
        "cliff",
        "clarity",
        "payoff",
        "resolved_after",
        "low_confidence",
        "tier",
        "why",
        "hook_line",
        "cliffhanger_line",
        "annotator_model",
        "judge_model",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for p in all_promos:
            writer.writerow({
                "episode": p["episode"],
                "rank_in_episode": p["rank_in_episode"],
                "candidate_id": p["candidate_id"],
                "segments_seconds": p["segments_seconds"],
                "segments_mmss": p["segments_mmss"],
                "duration_s": p["duration_s"],
                "s3a_score": p["scores"]["s3a_score"],
                "judge_final_score": p["scores"]["judge_final_score"],
                "hook": p["scores"]["hook"],
                "cliff": p["scores"]["cliff"],
                "clarity": p["scores"]["clarity"],
                "payoff": p["scores"]["payoff"],
                "resolved_after": p["scores"]["resolved_after"],
                "low_confidence": p["low_confidence"],
                "tier": p["tier"],
                "why": p["why"],
                "hook_line": p["hook_line"],
                "cliffhanger_line": p["cliffhanger_line"],
                "annotator_model": p["provenance"]["annotator_model"],
                "judge_model": p["provenance"]["judge_model"],
            })

    return all_promos


def build_show_level_top15(
    promos: List[Dict[str, Any]],
    out_dir: Path,
    no_llm: bool = False,
    enable_cache: bool = True,
    judge_role: str = "judge",
) -> List[Dict[str, Any]]:
    """Select up to 15 show-level promos with diversity cap of <= 2 per episode."""
    # Apply diversity cap of at most 2 per episode
    sorted_p = sorted(
        promos,
        key=lambda x: (
            1 if x["scores"]["hook"] >= 2 and x["scores"]["cliff"] >= 2 and not x["scores"]["resolved_after"] else 0,
            x["scores"]["s3a_score"],
            x["scores"]["judge_final_score"],
        ),
        reverse=True,
    )

    top_pool: List[Dict[str, Any]] = []
    ep_counts: Dict[str, int] = {}
    for p in sorted_p:
        ep = p["episode"]
        if ep_counts.get(ep, 0) < 2:
            top_pool.append(p)
            ep_counts[ep] = ep_counts.get(ep, 0) + 1
        if len(top_pool) == 15:
            break

    m_count = len(top_pool)
    if m_count == 0:
        ranked_promos = []
    elif no_llm or m_count == 1:
        ranked_promos = list(top_pool)
        for r_idx, rp in enumerate(ranked_promos, start=1):
            rp["show_rank"] = r_idx
    else:
        # Two listwise calls in opposite orders using judge role
        order1 = list(top_pool)
        order2 = list(reversed(top_pool))

        def make_listwise_prompt(cands_list: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
            parts = [
                f"You are the chief promo editor. Below are {len(cands_list)} top candidate promo clips from across a microdrama series.",
                f"Compare them directly and rank them from 1 to {len(cands_list)} (1 = best promo across the whole show to post first).",
                f"Use strictly unique integer ranks from 1 to {len(cands_list)} with NO ties.",
                "",
            ]
            meta = []
            for idx, c in enumerate(cands_list, start=1):
                parts.append(f"=== CANDIDATE {idx} ===")
                parts.append(f"Episode: {c['episode']} | Duration: {c['duration_s']:.1f}s")
                parts.append(f"Hook beat line: \"{c['hook_line']}\"")
                parts.append(f"Ending cliffhanger line: \"{c['cliffhanger_line']}\"")
                parts.append(f"Evaluation note: {c['why']}")
                parts.append("")
                meta.append({"num": idx, "cand": c})
            parts.append("Output strictly valid compact JSON with a list of rankings: [{'candidate_num': i, 'rank': r, 'why': '...'}].")
            return "\n".join(parts), meta

        sys_prompt = "You are an expert microdrama promo editor. Return strictly valid JSON."
        p1, meta1 = make_listwise_prompt(order1)
        p2, meta2 = make_listwise_prompt(order2)

        try:
            res1 = call(
                role=judge_role,
                system=sys_prompt,
                user=p1,
                schema=ListwiseRankingOutput,
                stage="show_listwise_fwd",
                enable_cache=enable_cache,
            )
            res2 = call(
                role=judge_role,
                system=sys_prompt,
                user=p2,
                schema=ListwiseRankingOutput,
                stage="show_listwise_rev",
                enable_cache=enable_cache,
            )
            out1: ListwiseRankingOutput = res1.parsed or ListwiseRankingOutput()
            out2: ListwiseRankingOutput = res2.parsed or ListwiseRankingOutput()

            map1 = {item.candidate_num: item.rank for item in out1.rankings}
            map2 = {item.candidate_num: item.rank for item in out2.rankings}

            avg_ranks = []
            for idx, c in enumerate(top_pool, start=1):
                cid = c["candidate_id"]
                r1 = map1.get(idx, idx)
                # In order2, index of candidate is (m_count - idx + 1)
                idx2 = m_count - idx + 1
                r2 = map2.get(idx2, idx)
                mean_r = (r1 + r2) / 2.0
                avg_ranks.append((mean_r, c["scores"]["s3a_score"], c))

            avg_ranks.sort(key=lambda x: (x[0], -x[1]))
            ranked_promos = []
            for final_r, (_, _, c) in enumerate(avg_ranks, start=1):
                c_copy = dict(c)
                c_copy["show_rank"] = final_r
                ranked_promos.append(c_copy)
        except Exception as e:
            print(f"[Warning] Listwise show ranking failed: {e}. Falling back to deterministic order.")
            ranked_promos = list(top_pool)
            for r_idx, rp in enumerate(ranked_promos, start=1):
                rp["show_rank"] = r_idx

    # Export promos_ranked.json
    with open(out_dir / "promos_ranked.json", "w", encoding="utf-8") as f:
        json.dump(ranked_promos, f, indent=2)

    # Export promos_ranked.csv
    csv_path = out_dir / "promos_ranked.csv"
    csv_fields = [
        "show_rank",
        "episode",
        "rank_in_episode",
        "candidate_id",
        "segments_seconds",
        "segments_mmss",
        "duration_s",
        "s3a_score",
        "judge_final_score",
        "hook",
        "cliff",
        "clarity",
        "payoff",
        "resolved_after",
        "low_confidence",
        "tier",
        "why",
        "hook_line",
        "cliffhanger_line",
        "annotator_model",
        "judge_model",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for p in ranked_promos:
            writer.writerow({
                "show_rank": p.get("show_rank", 1),
                "episode": p["episode"],
                "rank_in_episode": p["rank_in_episode"],
                "candidate_id": p["candidate_id"],
                "segments_seconds": p["segments_seconds"],
                "segments_mmss": p["segments_mmss"],
                "duration_s": p["duration_s"],
                "s3a_score": p["scores"]["s3a_score"],
                "judge_final_score": p["scores"]["judge_final_score"],
                "hook": p["scores"]["hook"],
                "cliff": p["scores"]["cliff"],
                "clarity": p["scores"]["clarity"],
                "payoff": p["scores"]["payoff"],
                "resolved_after": p["scores"]["resolved_after"],
                "low_confidence": p["low_confidence"],
                "tier": p["tier"],
                "why": p["why"],
                "hook_line": p["hook_line"],
                "cliffhanger_line": p["cliffhanger_line"],
                "annotator_model": p["provenance"]["annotator_model"],
                "judge_model": p["provenance"]["judge_model"],
            })

    return ranked_promos


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified promo finder CLI")
    parser.add_argument("--input", required=True, help="Folder containing raw screenplay .txt files")
    parser.add_argument("--out", required=True, help="Folder to save output promos, reports, and run manifest")
    parser.add_argument("--no-llm", action="store_true", help="Force Tier-0 heuristic mode with zero model calls")
    parser.add_argument("--episodes", nargs="+", help="Specific episode IDs to process (e.g. episode-001)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass LLM gateway cache")
    parser.add_argument("--strict", action="store_true", help="Raise error on unknown headers instead of skipping")
    parser.add_argument("--enforce-gold-lock", action="store_true", help="Enforce reserved episode locking on input")
    parser.add_argument("--workers", type=int, default=2, help="Worker threads for LLM calls")
    args = parser.parse_args()

    start_time = time.time()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    # 1. Discover screenplay files
    txt_files = sorted(input_path.glob("*.txt"))
    if args.episodes:
        req_stems = set(args.episodes)
        txt_files = [f for f in txt_files if f.stem in req_stems or any(ep in f.stem for ep in req_stems)]

    if not txt_files:
        print(f"No screenplay (.txt) files found in '{args.input}'.")
        sys.exit(1)

    # Filter reserved episodes if lock applies
    target_files = []
    for tf in txt_files:
        if is_episode_blocked(
            tf.stem,
            input_dir=args.input,
            enforce_gold_lock=args.enforce_gold_lock,
        ):
            continue
        target_files.append(tf)

    if not target_files:
        print("All target episodes were blocked by gold reserved locks.")
        sys.exit(1)

    ep_count = len(target_files)

    # 2. Pre-flight Quota Check
    gw = get_gateway()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 80)
    print(f"PROMO-FINDER RUN: {ep_count} episode(s) -> {out_path}")
    print(f"Mode: {'TIER-0 (NO-LLM)' if args.no_llm else 'PIPELINE (LLM)'} | Cache: {'DISABLED' if args.no_cache else 'ENABLED'}")
    print("=" * 80)

    if not args.no_llm:
        rem_cheap = gw.get_remaining_quota("cheap")
        rem_judge = gw.get_remaining_quota("judge")
        print(f"Pre-flight 24h Quota: cheap remaining={rem_cheap}, judge remaining={rem_judge}")
        try:
            gw.check_stage_quota("cheap", required_calls=ep_count)
            gw.check_stage_quota("judge", required_calls=ep_count + 2)
            print("[Pre-flight] Quota verified for cheap and judge roles.")
        except LLMQuotaExhaustedError as qe:
            print(f"[PRE-FLIGHT BLOCKED] {qe}")
            sys.exit(1)

    # Subdirectories for run artifacts
    parsed_dir = out_path / "parsed"
    annotated_dir = out_path / "annotated"
    candidates_dir = out_path / "candidates"
    judged_dir = out_path / "judged"

    parsed_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    judged_dir.mkdir(parents=True, exist_ok=True)

    # 3. Step 1: Parse
    print(f"\n[1/5] Parsing {len(target_files)} screenplay file(s)...")
    aliases = load_aliases("config/aliases.yaml")
    all_warnings: List[ParseWarning] = []
    episodes_parsed: List[str] = []

    for tf in target_files:
        ep_id = tf.stem
        try:
            beats, warnings, dur_max, dur_last, meta = parse_file(
                tf, aliases=aliases, episode_id=ep_id, strict=args.strict
            )
            all_warnings.extend(warnings)
            episodes_parsed.append(ep_id)

            payload = {
                "episode": ep_id,
                "duration_max_end_s": dur_max,
                "duration_last_beat_s": dur_last,
                "metadata": meta,
                "beats": [
                    {
                        "beat_id": b.beat_id,
                        "scene": b.scene,
                        "kind": b.kind,
                        "speaker_raw": b.speaker_raw,
                        "speaker_norm": b.speaker_norm,
                        "emotion_label": b.emotion_label,
                        "start_s": b.start_s,
                        "end_s": b.end_s,
                        "duration_s": b.duration_s,
                        "gap_prev_spoken_s": b.gap_prev_spoken_s,
                        "camera_context": b.camera_context,
                        "closing_camera_context": b.closing_camera_context,
                        "text": b.text,
                    }
                    for b in beats
                ],
            }
            with open(parsed_dir / f"{ep_id}.json", "w", encoding="utf-8") as pf:
                json.dump(payload, pf, indent=2)
        except ParseError as pe:
            print(f"Parse error in {tf.name}: {pe}")
            if args.strict:
                sys.exit(1)

    # 4. Step 2: S2 Annotation / Tier-0 Generation
    print(f"\n[2/5] Running S2 Annotation on {len(episodes_parsed)} episode(s)...")
    ann_report = run_annotation_pipeline(
        episodes=episodes_parsed,
        parsed_dir=parsed_dir,
        annotated_dir=annotated_dir,
        candidates_dir=candidates_dir,
        report_file=out_path / "annotate_report.json",
        role="cheap",
        max_workers=args.workers,
        enable_cache=not args.no_cache,
        run_id=run_id,
        force=True,
        no_llm=args.no_llm,
    )

    # 5. Step 3: S3a/S4 Candidate Generation & Gates
    print(f"\n[3/5] Running S3a/S4 Candidate Generation & Gates...")
    for ep_id in episodes_parsed:
        # If Tier-0 was run, candidates are already generated in candidates_dir
        if not (candidates_dir / f"{ep_id}.json").exists():
            process_episode_candidates(
                ep_id=ep_id,
                parsed_dir=parsed_dir,
                annotated_dir=annotated_dir,
                candidates_dir=candidates_dir,
            )

    # 6. Step 4: Stage 5 Judge
    if not args.no_llm:
        print(f"\n[4/5] Running Judge v2 on candidates...")
        # Check if any episodes were downgraded to Tier-0 fallback; judge only non-downgraded ones
        episodes_to_judge = []
        for ep_id in episodes_parsed:
            ann_file = annotated_dir / f"{ep_id}.json"
            if ann_file.exists():
                with open(ann_file, "r", encoding="utf-8") as af:
                    adata = json.load(af)
                if not adata.get("downgraded_to_tier0", False) and not adata.get("is_tier0", False):
                    episodes_to_judge.append(ep_id)

        if episodes_to_judge:
            run_judge_pipeline(
                episodes=episodes_to_judge,
                parsed_dir=parsed_dir,
                role="judge",
                max_workers=args.workers,
                enable_cache=not args.no_cache,
                run_id=run_id,
                mode="per_episode",
                prompt_version="judge_v2",
                annotated_dir=annotated_dir,
                candidates_dir=candidates_dir,
                judged_dir=judged_dir,
                report_file=out_path / "judge_report.json",
                force=True,
            )
    else:
        print(f"\n[4/5] Skipping Judge (Tier-0 no-llm mode)...")

    # 7. Step 5: Per-Episode Promos
    print(f"\n[5/5] Building final promo outputs...")
    per_ep_promos = build_per_episode_promos(
        episodes=episodes_parsed,
        parsed_dir=parsed_dir,
        candidates_dir=candidates_dir,
        judged_dir=judged_dir,
        out_dir=out_path,
        no_llm=args.no_llm,
    )

    # 8. Step 6: Show-Level Top 15 Ranking
    ranked_top15 = build_show_level_top15(
        promos=per_ep_promos,
        out_dir=out_path,
        no_llm=args.no_llm,
        enable_cache=not args.no_cache,
        judge_role="judge",
    )

    elapsed_s = round(time.time() - start_time, 2)

    # 9. Step 7: Telemetry & Run Manifest
    # Calculate calls and tokens from calls.jsonl if present
    calls_file = Path("runs") / run_id / "calls.jsonl"
    total_calls = 0
    cached_calls = 0
    in_tokens = 0
    out_tokens = 0
    if calls_file.exists():
        for line in calls_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    c = json.loads(line)
                    total_calls += 1
                    if c.get("cached"):
                        cached_calls += 1
                    in_tokens += c.get("tokens_in", 0)
                    out_tokens += c.get("tokens_out", 0)
                except Exception:
                    pass

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "input_directory": str(input_path),
        "output_directory": str(out_path),
        "mode": "tier0_no_llm" if args.no_llm else "pipeline_llm",
        "cache_enabled": not args.no_cache,
        "episodes_processed": len(episodes_parsed),
        "total_promos_found": len(per_ep_promos),
        "top_ranked_promos": len(ranked_top15),
        "total_calls": total_calls,
        "cached_calls": cached_calls,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "warnings_count": len(all_warnings),
        "elapsed_seconds": elapsed_s,
    }

    manifest_path = out_path / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("PROMO-FINDER RUN SUMMARY")
    print("=" * 80)
    print(f"Episodes Processed : {len(episodes_parsed)}")
    print(f"Total Promos Found : {len(per_ep_promos)} (exported to promos_per_episode.csv/.json)")
    print(f"Top Promos Ranked  : {len(ranked_top15)} (exported to promos_ranked.csv/.json)")
    print(f"LLM Network Calls  : {total_calls} ({cached_calls} cached, {total_calls - cached_calls} live)")
    print(f"Tokens Processed   : In: {in_tokens:,} | Out: {out_tokens:,}")
    print(f"Parser Warnings    : {len(all_warnings)}")
    print(f"Elapsed Time       : {elapsed_s}s")
    print(f"Run Manifest       : {manifest_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
