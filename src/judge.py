"""Stage 5: Combined Judge (src/judge.py).

Evaluates each candidate window from S4 survivors using a combined LLM judge that
assesses hook, cliffhanger, cold-viewer clarity, and promo payoff in a single call.

Design rules (AGENTS.md):
- LLM never sees beat IDs or timestamps. It sees only local indices 1..N.
- All LLM calls route through src/llm.py exclusively.
- Look-ahead window: up to 12s of non-camera beats AFTER the clip end cut.
- If resolved_after=True: generate up to 3 earlier-cut variants and judge the best.
- Final score = hook + cliff + clarity + payoff (sum of four judge fields).
- S3a score kept as numeric tie-breaker only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.llm import LLMResult, call, call_many
from src.reserved import is_episode_blocked
from src.gates import evaluate_window_gates, GATE_ORDER

# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

LOOK_AHEAD_SECONDS = 12.0
PROMPT_PATH = Path("prompts/judge_v1.md")


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hook: int = Field(description="Hook strength of first beat (0-3)")
    cliff: int = Field(description="Cliffhanger strength of last beat (0-3)")
    resolved_after: bool = Field(description="Whether AFTER window resolves the tension")
    clarity: bool = Field(default=False, description="NOT USED — kept for schema compat")
    payoff: int = Field(description="Promo arc payoff (0-3)")
    hook_index: int = Field(description="Local index of strongest hook beat (1..N)")
    cliff_index: int = Field(description="Local index of strongest cliff beat (1..N)")
    why: str = Field(description="Short reason for scores (max 25 words)")


class JudgeOutputFull(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hook: int = Field(description="Hook strength of first beat (0-3)")
    cliff: int = Field(description="Cliffhanger strength of last beat (0-3)")
    resolved_after: bool = Field(description="Whether AFTER window resolves the tension")
    clarity: int = Field(description="Cold-viewer stand-alone clarity (0-3)")
    payoff: int = Field(description="Promo arc payoff (0-3)")
    hook_index: int = Field(description="Local index of strongest hook beat (1..N)")
    cliff_index: int = Field(description="Local index of strongest cliff beat (1..N)")
    why: str = Field(description="Short reason for scores (max 25 words)")


# ---------------------------------------------------------------------------
# Prompt Loader
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT: Optional[str] = None


def load_judge_system_prompt(prompt_path: str | Path = PROMPT_PATH) -> str:
    """Load the judge rubric from prompts/judge_v1.md."""
    global _JUDGE_SYSTEM_PROMPT
    if _JUDGE_SYSTEM_PROMPT is not None:
        return _JUDGE_SYSTEM_PROMPT
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Judge prompt not found: {p}")
    _JUDGE_SYSTEM_PROMPT = p.read_text(encoding="utf-8")
    return _JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def _format_beat_line(local_idx: int, beat: Dict[str, Any], cam_context: Optional[str]) -> str:
    """Format a single beat as a local-index prompt line. No beat IDs, no timestamps."""
    kind = beat.get("kind", "")
    speaker = beat.get("speaker_norm") or beat.get("speaker_raw") or ""
    emotion = beat.get("emotion_label") or ""
    text = beat.get("text", "") or ""

    # Strip any raw timestamps that might have leaked into text (defensive)
    text = re.sub(r"\d{1,2}:\d{2}(?:\.\d+)?", "", text).strip()

    if kind == "DIALOGUE":
        cam = f" [cam: {cam_context}]" if cam_context else ""
        emo_part = f" ({emotion})" if emotion else ""
        speaker_part = f"{speaker}{emo_part}: " if speaker else ""
        return f"[{local_idx}] DIALOGUE {speaker_part}{text}{cam}"
    elif kind == "ACTION":
        cam = f" [cam: {cam_context}]" if cam_context else ""
        return f"[{local_idx}] ACTION: {text}{cam}"
    elif kind == "SFX":
        return f"[{local_idx}] SFX: {text}"
    else:
        return f"[{local_idx}] {kind}: {text}"


def build_judge_prompt(
    candidate: Dict[str, Any],
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
) -> Tuple[str, int, bool]:
    """Build user-facing judge prompt for a candidate.

    Returns:
        (prompt_text, N_clip_beats, after_available)
        N_clip_beats: number of beats in the clip (local indices 1..N)
        after_available: True if AFTER window is non-empty
    """
    all_beats: List[Dict[str, Any]] = parsed_data.get("beats", [])
    beat_lookup = {b["beat_id"]: b for b in all_beats}

    clip_beat_ids: List[str] = candidate.get("beat_ids", [])
    clip_end_s: float = candidate.get("end_s", 0.0)

    # Build clip lines (local index 1..N, non-camera beats only by construction)
    clip_lines: List[str] = []
    for local_idx, beat_id in enumerate(clip_beat_ids, start=1):
        beat = beat_lookup.get(beat_id, {})
        cam_ctx = beat.get("camera_context") or beat.get("closing_camera_context")
        line = _format_beat_line(local_idx, beat, cam_ctx)
        clip_lines.append(line)

    N = len(clip_lines)

    # Build AFTER window: non-camera beats with start_s in [clip_end_s, clip_end_s + 12s)
    after_beats = [
        b for b in all_beats
        if b.get("kind") != "CAMERA"
        and b.get("start_s", 0.0) >= clip_end_s - 0.01
        and b.get("start_s", 0.0) < clip_end_s + LOOK_AHEAD_SECONDS
        and b["beat_id"] not in set(clip_beat_ids)
    ]
    after_beats.sort(key=lambda b: b.get("start_s", 0.0))

    after_lines: List[str] = []
    for rel_idx, beat in enumerate(after_beats, start=N + 1):
        cam_ctx = beat.get("camera_context") or beat.get("closing_camera_context")
        line = _format_beat_line(rel_idx, beat, cam_ctx)
        after_lines.append(line)

    after_available = bool(after_lines)

    # Compose prompt — no beat IDs, no timestamps in this text
    prompt_parts = ["CLIP:"]
    prompt_parts.extend(clip_lines)
    prompt_parts.append("")
    if after_lines:
        prompt_parts.append("AFTER (next 12s):")
        prompt_parts.extend(after_lines)
    else:
        prompt_parts.append("AFTER: [END OF EPISODE]")

    # Verify: no beat IDs or raw timestamps in the prompt
    prompt_text = "\n".join(prompt_parts)
    _verify_no_beat_ids_or_timestamps(prompt_text, clip_beat_ids)

    return prompt_text, N, after_available


def _verify_no_beat_ids_or_timestamps(prompt_text: str, beat_ids: List[str]) -> None:
    """Assert that no beat IDs or mm:ss timestamps appear in the prompt text."""
    # Check beat IDs
    for bid in beat_ids:
        if bid in prompt_text:
            raise ValueError(f"Beat ID leaked into judge prompt: {bid!r}")
    # Check mm:ss timestamp pattern (must not appear)
    if re.search(r"\b\d{1,3}:\d{2}(?:\.\d+)?\b", prompt_text):
        raise ValueError("Timestamp (mm:ss) leaked into judge prompt")


# ---------------------------------------------------------------------------
# Judge Gates
# ---------------------------------------------------------------------------

def apply_judge_gates(
    judge_out: JudgeOutputFull,
    min_hook: int = 2,
    min_cliff: int = 2,
    min_clarity: int = 1,
) -> Tuple[bool, Optional[str]]:
    """Apply post-judge hard gates.

    Returns:
        (passed, reason_code or None)
    """
    if judge_out.hook < min_hook:
        return False, "JUDGE_HOOK_TOO_WEAK"
    if judge_out.cliff < min_cliff:
        return False, "JUDGE_CLIFFHANGER_TOO_WEAK"
    if judge_out.clarity < min_clarity:
        return False, "JUDGE_CLARITY_TOO_LOW"
    if judge_out.resolved_after:
        return False, "JUDGE_RESOLVED_AFTER"
    return True, None


# ---------------------------------------------------------------------------
# Variant Generator (B4)
# ---------------------------------------------------------------------------

def generate_earlier_cut_variants(
    candidate: Dict[str, Any],
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
    min_duration_s: float = 20.0,
    max_duration_s: float = 60.0,
    max_variants: int = 3,
) -> List[Dict[str, Any]]:
    """Generate up to max_variants earlier-cut candidates when resolved_after=True.

    Each variant ends on an earlier non-camera beat (working backwards from the
    second-to-last beat in the window). Checks straddle rule and duration gate.

    Returns list of valid variant dicts with parent_id set.
    """
    from src.windows import check_straddle_cuts

    all_beats: List[Dict[str, Any]] = parsed_data.get("beats", [])
    beat_lookup = {b["beat_id"]: b for b in all_beats}

    clip_beat_ids: List[str] = candidate.get("beat_ids", [])
    if len(clip_beat_ids) < 2:
        return []

    parent_id = candidate.get("candidate_id", "unknown")
    start_s = candidate.get("start_s", 0.0)

    variants: List[Dict[str, Any]] = []

    # Try ending on beat at index (len-2), (len-3), … down to index 0
    for new_last_idx in range(len(clip_beat_ids) - 2, -1, -1):
        if len(variants) >= max_variants:
            break

        new_beat_ids = clip_beat_ids[: new_last_idx + 1]
        last_beat_id = new_beat_ids[-1]
        last_beat = beat_lookup.get(last_beat_id, {})
        new_end_s = last_beat.get("end_s", 0.0)
        duration_s = new_end_s - start_s

        # Duration gate
        if not (min_duration_s <= duration_s <= max_duration_s):
            continue

        # Straddle rule
        window_beat_ids_set = set(new_beat_ids)
        straddles_spoken, _ = check_straddle_cuts(
            start_cut=start_s,
            end_cut=new_end_s,
            all_beats=all_beats,
            window_beat_ids=window_beat_ids_set,
        )
        if straddles_spoken:
            continue

        # Reconstruct variant dict (preserve S3a scores from original candidate)
        # Annotations for the new last beat
        annotations = annotated_data.get("annotations", {})
        last_ann = annotations.get(last_beat_id) or {}
        resolved_end = bool(last_ann.get("r", False))
        cliff = last_ann.get("q", 0) or 0

        variant = {
            "candidate_id": f"{parent_id}_var{new_last_idx}",
            "parent_id": parent_id,
            "episode": candidate.get("episode"),
            "beat_ids": new_beat_ids,
            "start_s": round(start_s, 2),
            "end_s": round(new_end_s, 2),
            "duration_s": round(duration_s, 2),
            "hook": candidate.get("hook", 0),
            "cliff": cliff,
            "resolved_end": resolved_end,
            "context": candidate.get("context", 0),
            "intensity": candidate.get("intensity", 0.0),
            "score": candidate.get("score", 0.0),
            "cut_straddles_spoken": False,
            "gate_results": {},
            "passed_all_gates": False,
            "is_variant": True,
        }
        variants.append(variant)

    return variants


# ---------------------------------------------------------------------------
# Core Judge Call
# ---------------------------------------------------------------------------

def judge_candidate(
    candidate: Dict[str, Any],
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
    role: str = "judge",
    stage: str = "s5_judge",
    prompt_version: str = "judge_v1",
    enable_cache: bool = True,
    cache_key_extra: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the LLM judge on a single candidate and apply judge gates.

    Returns a result dict with judge_output, gate results, final_score,
    and the LLMResult telemetry fields.
    Never prints screenplay text.
    """
    system_prompt = load_judge_system_prompt()
    prompt_text, n_beats, after_available = build_judge_prompt(
        candidate, parsed_data, annotated_data
    )

    result: LLMResult = call(
        role=role,
        system=system_prompt,
        user=prompt_text,
        schema=JudgeOutputFull,
        stage=stage,
        prompt_version=prompt_version,
        enable_cache=enable_cache,
        cache_key_extra=cache_key_extra,
    )

    judge_out: Optional[JudgeOutputFull] = result.parsed
    if judge_out is None:
        # Schema parse failed even after repair — record failure, return sentinel
        return {
            "candidate_id": candidate.get("candidate_id"),
            "judge_error": "SCHEMA_PARSE_FAILED",
            "passed_judge_gates": False,
            "judge_gate_reason": "SCHEMA_PARSE_FAILED",
            "final_score": 0,
            "s3a_score": candidate.get("score", 0.0),
            "llm_input_tokens": result.input_tokens,
            "llm_output_tokens": result.output_tokens,
            "llm_thinking_tokens": result.thinking_tokens,
            "llm_latency_s": result.latency_s,
            "llm_cached": result.cached,
            "llm_attempts": result.attempts,
            "n_clip_beats": n_beats,
            "after_available": after_available,
        }

    passed, gate_reason = apply_judge_gates(judge_out)
    final_score = judge_out.hook + judge_out.cliff + judge_out.clarity + judge_out.payoff

    return {
        "candidate_id": candidate.get("candidate_id"),
        "episode": candidate.get("episode"),
        "start_s": candidate.get("start_s"),
        "end_s": candidate.get("end_s"),
        "duration_s": candidate.get("duration_s"),
        "beat_ids": candidate.get("beat_ids"),
        "parent_id": candidate.get("parent_id"),
        "is_variant": candidate.get("is_variant", False),
        # Judge scores
        "hook": judge_out.hook,
        "cliff": judge_out.cliff,
        "resolved_after": judge_out.resolved_after,
        "clarity": judge_out.clarity,
        "payoff": judge_out.payoff,
        "hook_index": judge_out.hook_index,
        "cliff_index": judge_out.cliff_index,
        # why field NOT stored (may contain screenplay references)
        # Gate results
        "passed_judge_gates": passed,
        "judge_gate_reason": gate_reason,
        "final_score": final_score,
        "s3a_score": candidate.get("score", 0.0),
        # Telemetry
        "llm_input_tokens": result.input_tokens,
        "llm_output_tokens": result.output_tokens,
        "llm_thinking_tokens": result.thinking_tokens,
        "llm_latency_s": result.latency_s,
        "llm_cached": result.cached,
        "llm_attempts": result.attempts,
        "n_clip_beats": n_beats,
        "after_available": after_available,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_judge_pipeline(
    episodes: List[str],
    parsed_dir: str | Path = "data/parsed",
    annotated_dir: str | Path = "data/annotated",
    candidates_dir: str | Path = "data/candidates",
    judged_dir: str | Path = "data/judged",
    gold_file: str | Path = "gold/gold.csv",
    role: str = "judge",
    max_workers: int = 2,
    enable_cache: bool = True,
    cache_key_extra: Optional[str] = None,
    report_file: str | Path = "judged_report.json",
    thinking_budget_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the combined judge across all candidates for the given episodes.

    Flow:
    1. Load S4 survivors from data/candidates/<ep>.json
    2. Build judge prompts (all with local indices, no beat IDs, no timestamps)
    3. call_many() for concurrency (workers <= max_workers)
    4. For resolved_after=True cases: generate & judge up to 3 earlier cuts
    5. Apply judge gates; compute final_score
    6. Save data/judged/<ep>.json (gitignored)
    7. Write judged_report.json (gitignored, counts only)
    """
    parsed_path = Path(parsed_dir)
    annotated_path = Path(annotated_dir)
    candidates_path = Path(candidates_dir)
    judged_path = Path(judged_dir)
    judged_path.mkdir(parents=True, exist_ok=True)

    system_prompt = load_judge_system_prompt()
    all_ep_results: List[Dict[str, Any]] = []

    for ep_id in episodes:
        if is_episode_blocked(ep_id, gold_file=gold_file):
            print(f"[judge] Skipping reserved episode: {ep_id}")
            continue

        cand_file = candidates_path / f"{ep_id}.json"
        parsed_file = parsed_path / f"{ep_id}.json"
        ann_file = annotated_path / f"{ep_id}.json"

        for f, label in [(cand_file, "candidates"), (parsed_file, "parsed"), (ann_file, "annotated")]:
            if not f.exists():
                print(f"[judge] Missing {label} file for {ep_id}: {f}")
                continue

        with open(cand_file, "r", encoding="utf-8") as f:
            cand_data = json.load(f)
        with open(parsed_file, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)
        with open(ann_file, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        # Only judge S4 survivors
        survivors = [c for c in cand_data.get("candidates", []) if c.get("passed_all_gates")]
        if not survivors:
            print(f"[judge] {ep_id}: no S4 survivors to judge.")
            continue

        print(f"[judge] {ep_id}: judging {len(survivors)} survivor(s) ...")

        # Build batch requests
        requests = []
        candidate_refs = []

        for cand in survivors:
            try:
                prompt_text, n_beats, after_available = build_judge_prompt(
                    cand, parsed_data, ann_data
                )
            except Exception as e:
                print(f"[judge] {ep_id}/{cand.get('candidate_id')}: prompt build failed: {e}")
                continue

            req: Dict[str, Any] = {
                "role": role,
                "system": system_prompt,
                "user": prompt_text,
                "schema": JudgeOutputFull,
                "stage": "s5_judge",
                "prompt_version": "judge_v1",
                "enable_cache": enable_cache,
                "cache_key_extra": cache_key_extra,
            }
            requests.append(req)
            candidate_refs.append((cand, n_beats, after_available))

        if not requests:
            continue

        results = call_many(requests, max_workers=max_workers)

        judged_candidates: List[Dict[str, Any]] = []
        variant_requests: List[Tuple[Dict[str, Any], int, bool]] = []  # (variant_cand, ...)

        for (cand, n_beats, after_avail), llm_res in zip(candidate_refs, results):
            judge_out: Optional[JudgeOutputFull] = llm_res.parsed

            if judge_out is None:
                judged_candidates.append({
                    "candidate_id": cand.get("candidate_id"),
                    "judge_error": "SCHEMA_PARSE_FAILED",
                    "passed_judge_gates": False,
                    "judge_gate_reason": "SCHEMA_PARSE_FAILED",
                    "final_score": 0,
                    "s3a_score": cand.get("score", 0.0),
                    "llm_input_tokens": llm_res.input_tokens,
                    "llm_output_tokens": llm_res.output_tokens,
                    "llm_thinking_tokens": llm_res.thinking_tokens,
                    "llm_latency_s": llm_res.latency_s,
                    "llm_cached": llm_res.cached,
                    "llm_attempts": llm_res.attempts,
                    "n_clip_beats": n_beats,
                    "after_available": after_avail,
                })
                continue

            passed, gate_reason = apply_judge_gates(judge_out)
            final_score = judge_out.hook + judge_out.cliff + judge_out.clarity + judge_out.payoff

            rec: Dict[str, Any] = {
                "candidate_id": cand.get("candidate_id"),
                "episode": cand.get("episode"),
                "start_s": cand.get("start_s"),
                "end_s": cand.get("end_s"),
                "duration_s": cand.get("duration_s"),
                "beat_ids": cand.get("beat_ids"),
                "parent_id": None,
                "is_variant": False,
                "hook": judge_out.hook,
                "cliff": judge_out.cliff,
                "resolved_after": judge_out.resolved_after,
                "clarity": judge_out.clarity,
                "payoff": judge_out.payoff,
                "hook_index": judge_out.hook_index,
                "cliff_index": judge_out.cliff_index,
                "passed_judge_gates": passed,
                "judge_gate_reason": gate_reason,
                "final_score": final_score,
                "s3a_score": cand.get("score", 0.0),
                "llm_input_tokens": llm_res.input_tokens,
                "llm_output_tokens": llm_res.output_tokens,
                "llm_thinking_tokens": llm_res.thinking_tokens,
                "llm_latency_s": llm_res.latency_s,
                "llm_cached": llm_res.cached,
                "llm_attempts": llm_res.attempts,
                "n_clip_beats": n_beats,
                "after_available": after_avail,
            }
            judged_candidates.append(rec)

            # B4: If tension resolved immediately after → generate earlier cuts
            if judge_out.resolved_after:
                variants = generate_earlier_cut_variants(cand, parsed_data, ann_data)
                for var in variants:
                    variant_requests.append((var, cand, parsed_data, ann_data))

        # Judge any variants (second pass, sequential to stay within rate limits)
        if variant_requests:
            print(f"[judge] {ep_id}: judging {len(variant_requests)} variant(s) from resolved_after=True cases ...")
            var_batch = []
            var_refs = []
            for (var_cand, orig_cand, p_data, a_data) in variant_requests:
                try:
                    var_prompt, var_n, var_after = build_judge_prompt(var_cand, p_data, a_data)
                except Exception:
                    continue
                var_batch.append({
                    "role": role,
                    "system": system_prompt,
                    "user": var_prompt,
                    "schema": JudgeOutputFull,
                    "stage": "s5_judge_variant",
                    "prompt_version": "judge_v1",
                    "enable_cache": enable_cache,
                    "cache_key_extra": cache_key_extra,
                })
                var_refs.append((var_cand, var_n, var_after))

            if var_batch:
                var_results = call_many(var_batch, max_workers=max_workers)
                for (var_cand, var_n, var_after), var_res in zip(var_refs, var_results):
                    var_out: Optional[JudgeOutputFull] = var_res.parsed
                    if var_out is None:
                        continue
                    var_passed, var_gate_reason = apply_judge_gates(var_out)
                    var_score = var_out.hook + var_out.cliff + var_out.clarity + var_out.payoff
                    judged_candidates.append({
                        "candidate_id": var_cand.get("candidate_id"),
                        "episode": var_cand.get("episode"),
                        "start_s": var_cand.get("start_s"),
                        "end_s": var_cand.get("end_s"),
                        "duration_s": var_cand.get("duration_s"),
                        "beat_ids": var_cand.get("beat_ids"),
                        "parent_id": var_cand.get("parent_id"),
                        "is_variant": True,
                        "hook": var_out.hook,
                        "cliff": var_out.cliff,
                        "resolved_after": var_out.resolved_after,
                        "clarity": var_out.clarity,
                        "payoff": var_out.payoff,
                        "hook_index": var_out.hook_index,
                        "cliff_index": var_out.cliff_index,
                        "passed_judge_gates": var_passed,
                        "judge_gate_reason": var_gate_reason,
                        "final_score": var_score,
                        "s3a_score": var_cand.get("score", 0.0),
                        "llm_input_tokens": var_res.input_tokens,
                        "llm_output_tokens": var_res.output_tokens,
                        "llm_thinking_tokens": var_res.thinking_tokens,
                        "llm_latency_s": var_res.latency_s,
                        "llm_cached": var_res.cached,
                        "llm_attempts": var_res.attempts,
                        "n_clip_beats": var_n,
                        "after_available": var_after,
                    })

        # Rank by final_score desc, then s3a_score as tie-breaker
        judged_candidates.sort(
            key=lambda x: (x.get("final_score", 0), x.get("s3a_score", 0.0)),
            reverse=True,
        )

        # Save per-episode judged output (gitignored)
        ep_payload = {
            "episode": ep_id,
            "total_candidates_judged": len(survivors),
            "total_variants_generated": len(variant_requests),
            "judged": judged_candidates,
        }
        out_file = judged_path / f"{ep_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ep_payload, f, indent=2)

        all_ep_results.append(ep_payload)
        print(f"[judge] {ep_id}: done. Wrote {out_file}")

    # Build summary report (counts only, no screenplay text)
    total_judged = sum(r["total_candidates_judged"] for r in all_ep_results)
    total_variants = sum(r["total_variants_generated"] for r in all_ep_results)
    total_passed = sum(
        1 for r in all_ep_results
        for c in r["judged"]
        if c.get("passed_judge_gates")
    )

    summary: Dict[str, Any] = {
        "episodes_processed": len(all_ep_results),
        "total_s4_survivors_judged": total_judged,
        "total_variants_generated": total_variants,
        "total_passed_judge_gates": total_passed,
        "episodes": [
            {
                "episode": r["episode"],
                "candidates_judged": r["total_candidates_judged"],
                "variants_generated": r["total_variants_generated"],
                "passed_judge_gates": sum(
                    1 for c in r["judged"] if c.get("passed_judge_gates")
                ),
            }
            for r in all_ep_results
        ],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "episodes": all_ep_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run S5 Combined Judge.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["episode-001", "episode-002", "episode-005"],
        help="Episodes to judge",
    )
    parser.add_argument("--role", default="judge", help="Model role (default: judge)")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent workers")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument(
        "--cache-key-extra",
        default=None,
        dest="cache_key_extra",
        help="Extra cache key suffix to force cache miss (e.g. 'run2')",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        dest="thinking_budget",
        help="Override thinking_budget for judge role (e.g. 1024 for ablation)",
    )
    parser.add_argument(
        "--report-file",
        default="judged_report.json",
        help="Path for judged summary report",
    )
    args = parser.parse_args()

    # Optional: override thinking budget for ablation runs
    if args.thinking_budget is not None:
        from src.llm import get_gateway
        gw = get_gateway()
        if "judge" in gw.roles_config:
            gw.roles_config["judge"]["thinking_budget"] = args.thinking_budget
            print(f"[judge] thinking_budget overridden to {args.thinking_budget} for this run.")

    result = run_judge_pipeline(
        episodes=args.episodes,
        role=args.role,
        max_workers=args.workers,
        enable_cache=not args.no_cache,
        cache_key_extra=args.cache_key_extra,
        report_file=args.report_file,
    )
    summary = result.get("summary", {})
    print(f"\nS5 Judge complete.")
    print(f"Episodes processed : {summary.get('episodes_processed')}")
    print(f"Candidates judged  : {summary.get('total_s4_survivors_judged')}")
    print(f"Variants generated : {summary.get('total_variants_generated')}")
    print(f"Passed judge gates : {summary.get('total_passed_judge_gates')}")


if __name__ == "__main__":
    main()
