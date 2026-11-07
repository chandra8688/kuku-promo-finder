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
from typing import Any, Dict, List, Optional

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.judge import run_judge_pipeline
from src.llm import LLMQuotaExhaustedError, get_gateway
from src.parse import ParseError, ParseWarning, load_aliases, parse_file
from src.reserved import is_episode_blocked
from src.speaker import auto_alias_map
from src.finalists import run_finalists
from src.showrank import load_config as load_show_rank_config, rank_show
from src.tier0 import generate_tier0_candidates_for_episode


def role_model(role: str) -> str:
    """Model name for a role, read from config/models.yaml rather than hardcoded,
    so promo provenance cannot drift from the model that actually ran."""
    import yaml

    p = Path(__file__).resolve().parent / "config" / "models.yaml"
    try:
        roles = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("roles", {})
        return str(roles.get(role, {}).get("model_name") or "unknown")
    except Exception:
        return "unknown"


JUDGE_VERDICT_REJECTS = {
    "JUDGE_S2_MUTUAL_RESOLVED",
    "JUDGE_RESOLVED_AFTER",
    "JUDGE_HOOK_TOO_WEAK",
    "JUDGE_CLIFFHANGER_TOO_WEAK",
    "JUDGE_CLARITY_TOO_LOW",
}
JUDGE_REJECTED: List[Dict[str, Any]] = []


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
            # A V2 candidates file (it carries scoring_provenance) with no judged file
            # is V2 output the judge never saw, not a Tier-0 fallback.
            tier_label = c_data.get("tier") or ("v2_unjudged" if c_data.get("scoring_provenance") else "tier0_fallback")
            is_tier0_ep = str(tier_label).startswith("tier0")
        judged_ep = j_file.exists() and not is_tier0_ep
        # The annotator's own window scores (S3a features) travel with each promo,
        # so show-level ranking can require both models to agree.
        feat_by_id: Dict[str, Dict[str, Any]] = {}
        if c_file.exists():
            with open(c_file, "r", encoding="utf-8") as cf2:
                for c0 in json.load(cf2).get("candidates", []):
                    feat_by_id[c0.get("candidate_id")] = c0.get("features") or {}

        # A judge verdict that the clip resolves (agreed with S2) removes it from
        # the publishable pool; it is recorded, not ranked lower. A judge failure
        # (parse error, missing from batch) is not a verdict, so the clip stays
        # and is marked low-confidence instead.
        kept = []
        for c in cands:
            if not c.get("passed_judge_gates", True) and c.get("judge_gate_reason") in JUDGE_VERDICT_REJECTS:
                JUDGE_REJECTED.append({
                    "episode": ep_id,
                    "candidate_id": c.get("candidate_id"),
                    "reason": c.get("judge_gate_reason"),
                })
                continue
            if c.get("judge_error"):
                c["low_confidence"] = True
            if c.get("start_s") is None or c.get("end_s") is None:
                continue
            kept.append(c)
        cands = kept
        # Same total order as src/score.rank_key on the selection score
        cands.sort(
            key=lambda x: (
                1 if x.get("low_confidence", False) else 0,
                -float(x.get("s3a_score", x.get("score", 0.0))),
                str(x.get("candidate_id", "")),
            )
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

            feats = (feat_by_id.get(cand.get("candidate_id")) or feat_by_id.get(cand.get("parent_id"))
                     or cand.get("features") or {})
            if not feats and not judged_ep:
                feats = {"hook": cand.get("hook"), "cliff": cand.get("cliff"), "context": cand.get("context")}
            scores = {
                "s3a_score": cand.get("s3a_score", cand.get("score", 0.0)),
                "judge_final_score": cand.get("final_score", 0),
                "hook": cand.get("hook", 0),
                "cliff": cand.get("cliff", 0),
                "clarity": cand.get("clarity", 0),
                "payoff": cand.get("payoff", 0),
                "resolved_after": cand.get("resolved_after", False),
                "judged": bool(judged_ep and not cand.get("judge_error")),
                "annotator": {k: feats.get(k) for k in ("hook", "cliff", "context", "hook_kind", "resolved_last")},
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
                "beat_ids": beat_ids,
                "hook_line": hook_line,
                "cliffhanger_line": cliff_line,
                "why": cand.get("why", "Heuristic proxy selection" if is_tier0_ep else ""),
                "scores": scores,
                "low_confidence": cand.get("low_confidence", False),
                "tier": cand_tier,
                "provenance": {
                    "annotator_model": "none" if (no_llm or str(cand_tier).startswith("tier0")) else role_model("cheap"),
                    "annotator_prompt": "tier0" if is_tier0_ep else "s2_v1",
                    "judge_model": role_model("judge") if judged_ep else "none",
                    "judge_prompt": "judge_v2" if judged_ep else "none",
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
                "rank_in_episode": p.get("rank_in_episode", 1),
                "candidate_id": p["candidate_id"],
                "segments_seconds": p["segments_seconds"],
                "segments_mmss": p["segments_mmss"],
                "duration_s": p["duration_s"],
                "s3a_score": p.get("scores", {}).get("s3a_score", 0.0),
                "judge_final_score": p.get("scores", {}).get("judge_final_score", 0),
                "hook": p.get("scores", {}).get("hook", 0),
                "cliff": p.get("scores", {}).get("cliff", 0),
                "clarity": p.get("scores", {}).get("clarity", 0),
                "payoff": p.get("scores", {}).get("payoff", 0),
                "resolved_after": p.get("scores", {}).get("resolved_after", False),
                "low_confidence": p["low_confidence"],
                "tier": p["tier"],
                "why": p["why"],
                "hook_line": p["hook_line"],
                "cliffhanger_line": p["cliffhanger_line"],
                "annotator_model": p["provenance"]["annotator_model"],
                "judge_model": p["provenance"]["judge_model"],
            })

    return all_promos


HEADER_TEXT = (
    "Show ranking: a promo leads the list only if the annotator and the judge both rate its hook "
    "and cliffhanger at least 2/3 and it does not resolve itself; then post score = agreed hook + "
    "agreed cliffhanger + clarity + payoff + early-episode bonus (config/show_rank.yaml)."
)


def get_episode_order_num(ep_str: str) -> int:
    try:
        return int(ep_str.split("-")[-1])
    except Exception:
        digits = "".join(ch for ch in str(ep_str) if ch.isdigit())
        return int(digits) if digits else 0


RANKED_CSV_FIELDS = [
    "show_rank", "wave", "episode", "rank_in_episode", "candidate_id", "segments_seconds",
    "segments_mmss", "duration_s", "post_score", "agreed_hook", "agreed_cliff", "clarity",
    "payoff", "funnel_bonus", "below_bar", "hook_type", "why", "rank_reason", "hook_line",
    "cliffhanger_line", "s3a_score", "judge_hook", "judge_cliff", "annotator_hook",
    "annotator_cliff", "judge_note", "low_confidence", "tier", "annotator_model", "judge_model",
]


def build_show_level_top15(
    promos: List[Dict[str, Any]],
    out_dir: Path,
    parsed_dir: Optional[Path] = None,
    no_llm: bool = False,
    enable_cache: bool = True,
    finalists: bool = True,
) -> List[Dict[str, Any]]:
    """Rank per-episode promos across the show (src/showrank.py) and export them."""
    cfg = load_show_rank_config()
    for p in promos:
        # The judge's 'why' is a critique; keep it, but the promo's 'why' becomes
        # the reason it works (finalists call) or its rank reason.
        p.setdefault("judge_note", p.get("why", ""))

    finalist_info: Dict[str, Dict[str, str]] = {}
    fcfg = cfg.get("finalists", {})
    if finalists and not no_llm and fcfg.get("enabled", True) and parsed_dir is not None:
        # Wider pool than the final list, so wave-1 spreading has options.
        pre_cfg = dict(cfg, top_n=int(fcfg.get("max_candidates", 25)))
        pool = [p for p in rank_show(promos, pre_cfg) if p.get("below_bar") is None]
        finalist_info = run_finalists(
            pool,
            parsed_dir=parsed_dir,
            role=str(fcfg.get("role", "strong")),
            max_candidates=int(fcfg.get("max_candidates", 25)),
            enable_cache=enable_cache,
            fallback_roles=list(fcfg.get("fallback_roles", ["judge"])),
        )

    ranked_promos = rank_show(promos, cfg, finalist_info=finalist_info)

    with open(out_dir / "promos_ranked.json", "w", encoding="utf-8") as f:
        json.dump({
            "header": HEADER_TEXT,
            "show_rank_config": cfg,
            "finalists_labelled": len(finalist_info),
            "ranked_promos": ranked_promos,
        }, f, indent=2)

    with open(out_dir / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# {HEADER_TEXT}\n")
        writer = csv.DictWriter(f, fieldnames=RANKED_CSV_FIELDS)
        writer.writeheader()
        for p in ranked_promos:
            sc = p.get("scores", {})
            an = sc.get("annotator", {}) or {}
            writer.writerow({
                "show_rank": p["show_rank"],
                "wave": p["wave"],
                "episode": p["episode"],
                "rank_in_episode": p.get("rank_in_episode", 1),
                "candidate_id": p["candidate_id"],
                "segments_seconds": p["segments_seconds"],
                "segments_mmss": p["segments_mmss"],
                "duration_s": p["duration_s"],
                "post_score": p["post_score"],
                "agreed_hook": p["agreed_hook"],
                "agreed_cliff": p["agreed_cliff"],
                "clarity": sc.get("clarity", 0),
                "payoff": sc.get("payoff", 0),
                "funnel_bonus": p["funnel_bonus"],
                "below_bar": p["below_bar"] or "",
                "hook_type": p.get("hook_type") or "",
                "why": p["why"],
                "rank_reason": p["rank_reason"],
                "hook_line": p["hook_line"],
                "cliffhanger_line": p["cliffhanger_line"],
                "s3a_score": sc.get("s3a_score", 0.0),
                "judge_hook": sc.get("hook", "") if sc.get("judged") else "",
                "judge_cliff": sc.get("cliff", "") if sc.get("judged") else "",
                "annotator_hook": an.get("hook", ""),
                "annotator_cliff": an.get("cliff", ""),
                "judge_note": p.get("judge_note", ""),
                "low_confidence": p["low_confidence"],
                "tier": p["tier"],
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
    parser.add_argument("--no-finalists", action="store_true", help="Skip the finalists call (hook types and why-it-works lines)")
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
        try:
            beats, warnings, dur_max, dur_last, meta = parse_file(
                tf, aliases=aliases, episode_id=None, strict=args.strict
            )
            ep_id = meta.get("episode") or tf.stem
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

    # Same person under two names (MAYA / LADY MAYA): add speaker_key for
    # speaker counting. speaker_norm, which prompts show, is left untouched.
    speaker_names = set()
    for ep_id in episodes_parsed:
        data = json.loads((parsed_dir / f"{ep_id}.json").read_text(encoding="utf-8"))
        speaker_names.update(b.get("speaker_norm") for b in data["beats"] if b.get("speaker_norm"))
    auto_aliases = auto_alias_map(speaker_names)
    for ep_id in episodes_parsed:
        pth = parsed_dir / f"{ep_id}.json"
        data = json.loads(pth.read_text(encoding="utf-8"))
        for b in data["beats"]:
            if b.get("speaker_norm"):
                b["speaker_key"] = auto_aliases.get(b["speaker_norm"], b["speaker_norm"])
        pth.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out_path / "aliases_auto.json").write_text(json.dumps(auto_aliases, indent=2), encoding="utf-8")
    print(f"Merged {len(auto_aliases)} speaker name variant(s) (see aliases_auto.json).")

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
    from src.windows import scoring_provenance

    cfg_hash = scoring_provenance()["config_sha256"]
    for ep_id in episodes_parsed:
        # Tier-0 files are kept as-is. Anything else is regenerated unless it was
        # scored with the current scoring_v2.yaml, so a config change can never be
        # silently ignored by reusing stale candidate files.
        cf = candidates_dir / f"{ep_id}.json"
        stale = True
        if cf.exists():
            try:
                prev = json.loads(cf.read_text(encoding="utf-8"))
                is_tier0 = str(prev.get("tier", "")).startswith("tier0")
                same_cfg = prev.get("scoring_provenance", {}).get("config_sha256") == cfg_hash
                stale = not (is_tier0 or same_cfg)
            except Exception:
                stale = True
        if stale:
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
        parsed_dir=parsed_dir,
        no_llm=args.no_llm,
        enable_cache=not args.no_cache,
        finalists=not args.no_finalists,
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

    from src.windows import scoring_provenance

    ann_status = {"complete": 0, "partial": 0, "missing": 0}
    for ep_id in episodes_parsed:
        af = annotated_dir / f"{ep_id}.json"
        if not af.exists():
            ann_status["missing"] += 1
            continue
        try:
            ann_status["partial" if json.loads(af.read_text(encoding="utf-8")).get("partial") else "complete"] += 1
        except Exception:
            ann_status["missing"] += 1

    with open(out_path / "judge_rejected.json", "w", encoding="utf-8") as f:
        json.dump(JUDGE_REJECTED, f, indent=2)

    manifest = {
        "run_id": run_id,
        "scorer": "tier0_heuristic" if args.no_llm else "v2",
        "scoring_provenance": None if args.no_llm else scoring_provenance(),
        "annotation_status": ann_status,
        "judge_rejected_count": len(JUDGE_REJECTED),
        "top15_shortfall": max(0, 15 - len(ranked_top15)),
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
