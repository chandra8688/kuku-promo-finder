"""Stage 5: Combined Judge (src/judge.py).

Evaluates each candidate window from S4 survivors using a combined LLM judge that
assesses hook, cliffhanger, cold-viewer clarity, and promo payoff in a single call.

Design rules:
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

import yaml

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.llm import LLMResult, LLMQuotaExhaustedError, call, call_many, check_stage_quota
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


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_num: int = Field(description="Local candidate number (1..M)")
    rank: int = Field(default=1, description="Unique integer rank 1..M (1 = best promo to post first, no ties)")
    hook: int = Field(description="Hook strength of first beat (0-3)")
    cliff: int = Field(description="Cliffhanger strength of last beat (0-3)")
    resolved_after: bool = Field(description="Whether AFTER window resolves the tension")
    clarity: int = Field(description="Cold-viewer stand-alone clarity (0-3)")
    payoff: int = Field(description="Promo arc payoff (0-3)")
    hook_index: int = Field(description="Local index of strongest hook beat (1..N)")
    cliff_index: int = Field(description="Local index of strongest cliff beat (1..N)")
    why: str = Field(description="Short reason for scores (max 25 words)")

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "candidate_num" not in data:
                for k in ("candidate", "num", "id", "candidate_id", "candidate_number"):
                    if k in data:
                        data["candidate_num"] = data[k]
                        break
        return data


class EpisodeBatchJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evaluations: List[CandidateEvaluation] = Field(
        default_factory=list,
        description="Evaluations for each candidate 1..M with unique ranks 1..M",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_batch_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evaluations" not in data:
                for k in ("candidates", "results", "evals", "items"):
                    if k in data:
                        data["evaluations"] = data[k]
                        break
        elif isinstance(data, list):
            return {"evaluations": data}
        return data


def validate_unique_ranks(evaluations: List[CandidateEvaluation], expected_m: int) -> Tuple[bool, Optional[str]]:
    """Validate that candidate evaluations have unique ranks 1..M without ties."""
    if len(evaluations) != expected_m:
        return False, f"Expected {expected_m} evaluations, got {len(evaluations)}"
    ranks = [e.rank for e in evaluations]
    expected_set = set(range(1, expected_m + 1))
    actual_set = set(ranks)
    if actual_set != expected_set:
        duplicates = [r for r in set(ranks) if ranks.count(r) > 1]
        missing = sorted(list(expected_set - actual_set))
        return False, f"Ranks must be a unique permutation of 1..{expected_m}. Duplicates: {duplicates}, Missing: {missing}"
    return True, None


# ---------------------------------------------------------------------------
# Prompt Loader
# ---------------------------------------------------------------------------

_PROMPT_CACHE: Dict[str, str] = {}


def load_judge_system_prompt(prompt_path: str | Path = PROMPT_PATH) -> str:
    """Load the judge rubric from prompt file."""
    path_str = str(prompt_path)
    if path_str in _PROMPT_CACHE:
        return _PROMPT_CACHE[path_str]
    p = Path(prompt_path)
    if not p.exists():
        if "v2" in path_str and Path("prompts/judge_v1.md").exists():
            p = Path("prompts/judge_v1.md")
        elif "v1" in path_str and Path("prompts/judge_v2.md").exists():
            p = Path("prompts/judge_v2.md")
        else:
            raise FileNotFoundError(f"Judge prompt not found: {p}")
    text = p.read_text(encoding="utf-8")
    _PROMPT_CACHE[path_str] = text
    return text


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


def build_episode_batch_judge_prompt(
    candidates: List[Dict[str, Any]],
    parsed_data: Dict[str, Any],
    annotated_data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a unified prompt evaluating all candidate windows in one episode.

    Returns:
        (prompt_text, candidate_metadata_list)
    """
    parts = [
        "You are evaluating multiple candidate promo clips from the same episode.",
        "Evaluate each candidate independently using the scoring rubric.",
        "Return an evaluation object for each candidate numbered 1 through M.\n",
    ]
    meta_list = []

    for idx, cand in enumerate(candidates, start=1):
        prompt_text, n_beats, after_avail = build_judge_prompt(cand, parsed_data, annotated_data)
        parts.append(f"=== CANDIDATE {idx} ===")
        parts.append(prompt_text)
        parts.append("")
        meta_list.append({
            "candidate_num": idx,
            "candidate": cand,
            "n_clip_beats": n_beats,
            "after_available": after_avail,
        })

    return "\n".join(parts), meta_list


# ---------------------------------------------------------------------------
# Judge Gates & Scoring Helpers
# ---------------------------------------------------------------------------

def load_judge_scoring_config(config_path: str | Path = "config/scoring_v2.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        p = Path("config/scoring.yaml")
    if not p.exists():
        return {"mode": "score", "weight": 0.5, "reject_only_on_agreement": True}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("judge", {"mode": "score", "weight": 0.5, "reject_only_on_agreement": True})


CARRY_KEYS = (
    "candidate_id", "episode", "start_s", "end_s", "duration_s", "beat_ids",
    "start_s_raw", "end_s_raw", "refine_ops", "score_refined", "breakdown",
    "soft_gate_flags", "low_confidence", "unstable",
)


def _carry(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that must survive judging unchanged.

    Without this a judge parse failure produced a promo with no timings
    (00:00-00:00, 0s), and a successful judgement dropped low_confidence, so the
    "low-confidence clips rank last" rule silently stopped applying in LLM mode.
    """
    return {k: candidate.get(k) for k in CARRY_KEYS if k in candidate}


def check_candidate_s2_resolved(candidate: Dict[str, Any], annotated_data: Optional[Dict[str, Any]] = None) -> bool:
    if candidate.get("resolved_end"):
        return True
    feats = candidate.get("features")
    if isinstance(feats, dict) and (feats.get("resolved_tail") or feats.get("resolved_last")):
        return True
    if annotated_data:
        ann = annotated_data.get("annotations", {})
        b_ids = candidate.get("beat_ids", [])
        if b_ids:
            last_bid = b_ids[-1]
            if ann.get(last_bid, {}).get("r", False):
                return True
    return False


def apply_judge_gates(
    judge_out: JudgeOutputFull,
    min_hook: int = 2,
    min_cliff: int = 2,
    min_clarity: int = 1,
    mode: str = "gate",
    s2_resolved: bool = False,
    reject_only_on_agreement: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Apply judge gates.

    Modes:
      - 'gate': hard rejects on hook < min_hook, cliff < min_cliff,
        clarity < min_clarity, or resolved_after=True.
      - 'score': only rejects on resolved_after=True, and only when S2 also says
        the tail resolves if reject_only_on_agreement=True.
    """
    if mode == "score":
        if judge_out.resolved_after:
            if reject_only_on_agreement:
                if s2_resolved:
                    return False, "JUDGE_S2_MUTUAL_RESOLVED"
            else:
                return False, "JUDGE_RESOLVED_AFTER"
        return True, None

    # mode == 'gate'
    if judge_out.hook < min_hook:
        return False, "JUDGE_HOOK_TOO_WEAK"
    if judge_out.cliff < min_cliff:
        return False, "JUDGE_CLIFFHANGER_TOO_WEAK"
    if judge_out.clarity < min_clarity:
        return False, "JUDGE_CLARITY_TOO_LOW"
    if judge_out.resolved_after:
        if reject_only_on_agreement:
            if s2_resolved:
                return False, "JUDGE_S2_MUTUAL_RESOLVED"
        else:
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
    judge_mode: Optional[str] = None,
    judge_weight: Optional[float] = None,
    reject_only_on_agreement: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the LLM judge on a single candidate and apply judge scoring/gates.

    Returns a result dict with judge_output, gate results, composite_score, final_score,
    and the LLMResult telemetry fields.
    Never prints screenplay text.
    """
    j_cfg = load_judge_scoring_config()
    j_mode = judge_mode or j_cfg.get("mode", "score")
    j_weight = float(judge_weight if judge_weight is not None else j_cfg.get("weight", 0.5))
    j_agree = reject_only_on_agreement if reject_only_on_agreement is not None else j_cfg.get("reject_only_on_agreement", True)

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
    # selection score, not score_refined: ranking is frozen before refinement
    s3a_score = float(candidate.get("score", 0.0))
    if judge_out is None:
        # Schema parse failed even after repair — record failure, return sentinel
        return {
            **_carry(candidate),
            "low_confidence": True,
            "judge_error": "SCHEMA_PARSE_FAILED",
            "passed_judge_gates": False,
            "judge_gate_reason": "SCHEMA_PARSE_FAILED",
            "judge_composite": 0,
            "composite_score": s3a_score,
            "final_score": 0,
            "s3a_score": s3a_score,
            "score": s3a_score,
            "llm_input_tokens": result.input_tokens,
            "llm_output_tokens": result.output_tokens,
            "llm_thinking_tokens": result.thinking_tokens,
            "llm_latency_s": result.latency_s,
            "llm_cached": result.cached,
            "llm_attempts": result.attempts,
            "n_clip_beats": n_beats,
            "after_available": after_available,
        }

    s2_res = check_candidate_s2_resolved(candidate, annotated_data)
    passed, gate_reason = apply_judge_gates(
        judge_out,
        mode=j_mode,
        s2_resolved=s2_res,
        reject_only_on_agreement=j_agree,
    )
    judge_composite = judge_out.hook + judge_out.cliff + judge_out.clarity + judge_out.payoff
    composite_score = round(s3a_score + (j_weight * judge_composite), 3)

    return {
        **_carry(candidate),
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
        # Gate results & scoring
        "passed_judge_gates": passed,
        "judge_gate_reason": gate_reason,
        "judge_composite": judge_composite,
        "composite_score": composite_score,
        "final_score": judge_composite,
        "s3a_score": s3a_score,
        "score": composite_score if j_mode == "score" else s3a_score,
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
    run_id: Optional[str] = None,
    report_file: str | Path = "judged_report.json",
    thinking_budget_override: Optional[int] = None,
    mode: str = "per_candidate",
    force: bool = False,
    prompt_version: str = "judge_v2",
    allow_variants: bool = False,
    judge_mode: Optional[str] = None,
    judge_weight: Optional[float] = None,
    reject_only_on_agreement: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the combined judge across all candidates for the given episodes.

    Modes:
      - per_candidate: 1 LLM call per candidate window.
      - per_episode: 1 batched LLM call per episode evaluating all candidate windows.
    """
    j_cfg = load_judge_scoring_config()
    j_mode = judge_mode or j_cfg.get("mode", "score")
    j_weight = float(judge_weight if judge_weight is not None else j_cfg.get("weight", 0.5))
    j_agree = reject_only_on_agreement if reject_only_on_agreement is not None else j_cfg.get("reject_only_on_agreement", True)

    parsed_path = Path(parsed_dir)
    annotated_path = Path(annotated_dir)
    candidates_path = Path(candidates_dir)
    judged_path = Path(judged_dir)
    judged_path.mkdir(parents=True, exist_ok=True)

    if mode == "per_episode":
        prompt_path = Path("prompts/judge_v2.md")
    else:
        prompt_path = Path("prompts/judge_v1.md")
    system_prompt = load_judge_system_prompt(prompt_path)
    all_ep_results: List[Dict[str, Any]] = []

    # Pre-count required calls across active episodes for quota check
    active_episodes = [ep for ep in episodes if not is_episode_blocked(ep, gold_file=gold_file)]
    total_calls_needed = 0
    episodes_survivors: Dict[str, List[Dict[str, Any]]] = {}

    for ep_id in active_episodes:
        cand_file = candidates_path / f"{ep_id}.json"
        if cand_file.exists():
            try:
                with open(cand_file, "r", encoding="utf-8") as f:
                    cand_data = json.load(f)
                survs = [c for c in cand_data.get("candidates", []) if c.get("passed_all_gates")]
                episodes_survivors[ep_id] = survs
                if survs:
                    total_calls_needed += 1 if mode == "per_episode" else len(survs)
            except Exception:
                pass

    # Check 24h quota limit (refuses unless force=True)
    check_stage_quota(role=role, required_calls=total_calls_needed, force=force)

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

        print(f"[judge] {ep_id}: judging {len(survivors)} survivor(s) [mode={mode}] ...")

        judged_candidates: List[Dict[str, Any]] = []
        variant_requests: List[Tuple[Dict[str, Any], int, bool]] = []

        try:
            if mode == "per_episode":
                # Single call evaluating all candidates in this episode
                batch_prompt, meta_list = build_episode_batch_judge_prompt(
                    survivors, parsed_data, ann_data
                )
                batch_res = call(
                    role=role,
                    system=system_prompt,
                    user=batch_prompt,
                    schema=EpisodeBatchJudgeOutput,
                    stage="s5_judge_episode_batch",
                    prompt_version=prompt_version,
                    enable_cache=enable_cache,
                    cache_key_extra=cache_key_extra,
                    run_id=run_id,
                )

                batch_out: Optional[EpisodeBatchJudgeOutput] = batch_res.parsed

                # Rank validation & one repair retry
                is_valid_rank = True
                rank_err = None
                if batch_out and batch_out.evaluations:
                    is_valid_rank, rank_err = validate_unique_ranks(batch_out.evaluations, len(meta_list))
                else:
                    is_valid_rank = False
                    rank_err = "Empty or missing batch evaluations"

                if not is_valid_rank:
                    repair_prompt = (
                        f"{batch_prompt}\n\n"
                        f"RANK VALIDATION ERROR: {rank_err}\n"
                        f"Please output corrected evaluations for all {len(meta_list)} candidates where each candidate receives a strictly unique integer rank from 1 to {len(meta_list)} with NO duplicate ranks and NO ties."
                    )
                    rep_res = call(
                        role=role,
                        system=system_prompt,
                        user=repair_prompt,
                        schema=EpisodeBatchJudgeOutput,
                        stage="s5_judge_repair",
                        prompt_version=prompt_version,
                        enable_cache=enable_cache,
                        cache_key_extra=cache_key_extra,
                        run_id=run_id,
                    )
                    if rep_res.parsed and rep_res.parsed.evaluations:
                        rep_valid, _ = validate_unique_ranks(rep_res.parsed.evaluations, len(meta_list))
                        if rep_valid:
                            batch_out = rep_res.parsed

                eval_map = {}
                if batch_out and batch_out.evaluations:
                    eval_map = {ev.candidate_num: ev for ev in batch_out.evaluations}

                for meta in meta_list:
                    c_num = meta["candidate_num"]
                    cand = meta["candidate"]
                    n_beats = meta["n_clip_beats"]
                    after_avail = meta["after_available"]
                    ev = eval_map.get(c_num)

                    if ev is None:
                        judged_candidates.append({
                            **_carry(cand),
                            "low_confidence": True,
                            "judge_error": "MISSING_FROM_BATCH_OUTPUT",
                            "passed_judge_gates": False,
                            "judge_gate_reason": "MISSING_FROM_BATCH_OUTPUT",
                            "judge_composite": 0,
                            "composite_score": float(cand.get("score", 0.0)),
                            "final_score": 0,
                            "rank": None,
                            "s3a_score": float(cand.get("score", 0.0)),
                            "score": float(cand.get("score", 0.0)),
                            "llm_input_tokens": batch_res.input_tokens,
                            "llm_output_tokens": batch_res.output_tokens,
                            "llm_thinking_tokens": batch_res.thinking_tokens,
                            "llm_latency_s": batch_res.latency_s,
                            "llm_cached": batch_res.cached,
                            "llm_attempts": batch_res.attempts,
                            "n_clip_beats": n_beats,
                            "after_available": after_avail,
                        })
                        continue

                    s2_res = check_candidate_s2_resolved(cand, ann_data)
                    passed, gate_reason = apply_judge_gates(
                        ev,
                        mode=j_mode,
                        s2_resolved=s2_res,
                        reject_only_on_agreement=j_agree,
                    )
                    judge_composite = ev.hook + ev.cliff + ev.clarity + ev.payoff
                    s3a_score = float(cand.get("score", 0.0))
                    composite_score = round(s3a_score + (j_weight * judge_composite), 3)

                    rec = {
                        **_carry(cand),
                        "candidate_id": cand.get("candidate_id"),
                        "episode": cand.get("episode"),
                        "start_s": cand.get("start_s"),
                        "end_s": cand.get("end_s"),
                        "duration_s": cand.get("duration_s"),
                        "beat_ids": cand.get("beat_ids"),
                        "parent_id": None,
                        "is_variant": False,
                        "low_confidence": cand.get("low_confidence", False),
                        "rank": ev.rank,
                        "hook": ev.hook,
                        "cliff": ev.cliff,
                        "resolved_after": ev.resolved_after,
                        "clarity": ev.clarity,
                        "payoff": ev.payoff,
                        "hook_index": ev.hook_index,
                        "cliff_index": ev.cliff_index,
                        "why": ev.why,
                        "passed_judge_gates": passed,
                        "judge_gate_reason": gate_reason,
                        "judge_composite": judge_composite,
                        "composite_score": composite_score,
                        "final_score": judge_composite,
                        "s3a_score": s3a_score,
                        "score": composite_score if j_mode == "score" else s3a_score,
                        "llm_input_tokens": batch_res.input_tokens,
                        "llm_output_tokens": batch_res.output_tokens,
                        "llm_thinking_tokens": batch_res.thinking_tokens,
                        "llm_latency_s": batch_res.latency_s,
                        "llm_cached": batch_res.cached,
                        "llm_attempts": batch_res.attempts,
                        "n_clip_beats": n_beats,
                        "after_available": after_avail,
                    }
                    judged_candidates.append(rec)

                    if allow_variants and ev.resolved_after:
                        vars_gen = generate_earlier_cut_variants(cand, parsed_data, ann_data)
                        for v in vars_gen:
                            variant_requests.append((v, cand, parsed_data, ann_data))

            else:
                # Default: per_candidate mode
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
                        "run_id": run_id,
                    }
                    requests.append(req)
                    candidate_refs.append((cand, n_beats, after_available))

                if not requests:
                    continue

                results = call_many(requests, max_workers=max_workers)

                for (cand, n_beats, after_avail), llm_res in zip(candidate_refs, results):
                    judge_out: Optional[JudgeOutputFull] = llm_res.parsed

                    if judge_out is None:
                        judged_candidates.append({
                            "candidate_id": cand.get("candidate_id"),
                            "judge_error": "SCHEMA_PARSE_FAILED",
                            "passed_judge_gates": False,
                            "judge_gate_reason": "SCHEMA_PARSE_FAILED",
                            "judge_composite": 0,
                            "composite_score": float(cand.get("score", 0.0)),
                            "final_score": 0,
                            "s3a_score": float(cand.get("score", 0.0)),
                            "score": float(cand.get("score", 0.0)),
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

                    s2_res = check_candidate_s2_resolved(cand, ann_data)
                    passed, gate_reason = apply_judge_gates(
                        judge_out,
                        mode=j_mode,
                        s2_resolved=s2_res,
                        reject_only_on_agreement=j_agree,
                    )
                    judge_composite = judge_out.hook + judge_out.cliff + judge_out.clarity + judge_out.payoff
                    s3a_score = float(cand.get("score", 0.0))
                    composite_score = round(s3a_score + (j_weight * judge_composite), 3)

                    rec = {
                        **_carry(cand),
                        "candidate_id": cand.get("candidate_id"),
                        "episode": cand.get("episode"),
                        "start_s": cand.get("start_s"),
                        "end_s": cand.get("end_s"),
                        "duration_s": cand.get("duration_s"),
                        "beat_ids": cand.get("beat_ids"),
                        "parent_id": None,
                        "is_variant": False,
                        "low_confidence": cand.get("low_confidence", False),
                        "hook": judge_out.hook,
                        "cliff": judge_out.cliff,
                        "resolved_after": judge_out.resolved_after,
                        "clarity": judge_out.clarity,
                        "payoff": judge_out.payoff,
                        "hook_index": judge_out.hook_index,
                        "cliff_index": judge_out.cliff_index,
                        "why": judge_out.why,
                        "passed_judge_gates": passed,
                        "judge_gate_reason": gate_reason,
                        "judge_composite": judge_composite,
                        "composite_score": composite_score,
                        "final_score": judge_composite,
                        "s3a_score": s3a_score,
                        "score": composite_score if j_mode == "score" else s3a_score,
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

                    if judge_out.resolved_after:
                        vars_gen = generate_earlier_cut_variants(cand, parsed_data, ann_data)
                        for v in vars_gen:
                            variant_requests.append((v, cand, parsed_data, ann_data))

        except LLMQuotaExhaustedError as e:
            print(f"\n[QUOTA_EXHAUSTED] {e}")
            print(f"Pipeline stopped cleanly for {ep_id}. Finished calls are cached.")
            break

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
                var_refs.append((var_cand, var_n, var_after, a_data))

            if var_batch:
                try:
                    var_results = call_many(var_batch, max_workers=max_workers)
                    for (var_cand, var_n, var_after, a_data), var_res in zip(var_refs, var_results):
                        var_out: Optional[JudgeOutputFull] = var_res.parsed
                        if var_out is None:
                            continue
                        var_s2 = check_candidate_s2_resolved(var_cand, a_data)
                        var_passed, var_gate_reason = apply_judge_gates(
                            var_out,
                            mode=j_mode,
                            s2_resolved=var_s2,
                            reject_only_on_agreement=j_agree,
                        )
                        var_composite = var_out.hook + var_out.cliff + var_out.clarity + var_out.payoff
                        var_s3a = float(var_cand.get("score", 0.0))
                        var_comp_score = round(var_s3a + (j_weight * var_composite), 3)
                        judged_candidates.append({
                            **_carry(var_cand),
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
                            "judge_composite": var_composite,
                            "composite_score": var_comp_score,
                            "final_score": var_composite,
                            "s3a_score": var_s3a,
                            "score": var_comp_score if j_mode == "score" else var_s3a,
                            "llm_input_tokens": var_res.input_tokens,
                            "llm_output_tokens": var_res.output_tokens,
                            "llm_thinking_tokens": var_res.thinking_tokens,
                            "llm_latency_s": var_res.latency_s,
                            "llm_cached": var_res.cached,
                            "llm_attempts": var_res.attempts,
                            "n_clip_beats": var_n,
                            "after_available": var_after,
                        })
                except LLMQuotaExhaustedError as e:
                    print(f"[QUOTA_EXHAUSTED during variant evaluation] {e}")

        # Sort according to configured ranking decision rule
        ranking_rule = "s3a_score_with_judge_gate"
        try:
            with open("config/scoring.yaml", "r", encoding="utf-8") as sf:
                scfg = yaml.safe_load(sf) or {}
                ranking_rule = scfg.get("ranking_decision", {}).get("rule", ranking_rule)
        except Exception:
            pass

        if j_mode == "score":
            judged_candidates.sort(
                key=lambda x: (
                    1 if x.get("passed_judge_gates", False) else 0,
                    x.get("composite_score", x.get("final_score", 0.0)),
                    x.get("s3a_score", 0.0),
                ),
                reverse=True,
            )
        elif ranking_rule == "judge_rank":
            judged_candidates.sort(
                key=lambda x: (
                    0 if x.get("passed_judge_gates", False) else 1,
                    x.get("rank") if x.get("rank") is not None else 999,
                    -x.get("s3a_score", 0.0),
                )
            )
        else:
            # s3a_score with judge as pass/fail gate only
            judged_candidates.sort(
                key=lambda x: (
                    1 if x.get("passed_judge_gates", False) else 0,
                    x.get("s3a_score", 0.0),
                    x.get("judge_composite", x.get("final_score", 0)),
                ),
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
    parser.add_argument(
        "--mode",
        choices=["per_candidate", "per_episode"],
        default="per_candidate",
        help="Judge evaluation mode: per_candidate or per_episode",
    )
    parser.add_argument(
        "--allow-variants",
        action="store_true",
        help="Generate earlier-cut variants when resolved_after is True",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass daily quota limit check",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["score", "gate"],
        default=None,
        dest="judge_mode",
        help="Judge mode: score (fold into final score) or gate (pass/fail hard gate)",
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=None,
        dest="judge_weight",
        help="Weight to fold judge composite score into final score (default: 0.5)",
    )
    parser.add_argument(
        "--reject-only-on-agreement",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="reject_only_on_agreement",
        help="Only reject resolved_after if S2 also flags resolution (default: True)",
    )
    args = parser.parse_args()

    # Optional: override thinking budget for ablation runs
    if args.thinking_budget is not None:
        from src.llm import get_gateway
        gw = get_gateway()
        if "judge" in gw.roles_config:
            gw.roles_config["judge"]["thinking_budget"] = args.thinking_budget
            print(f"[judge] thinking_budget overridden to {args.thinking_budget} for this run.")

    try:
        result = run_judge_pipeline(
            episodes=args.episodes,
            role=args.role,
            max_workers=args.workers,
            enable_cache=not args.no_cache,
            cache_key_extra=args.cache_key_extra,
            report_file=args.report_file,
            mode=args.mode,
            force=args.force,
            allow_variants=args.allow_variants,
            judge_mode=args.judge_mode,
            judge_weight=args.judge_weight,
            reject_only_on_agreement=args.reject_only_on_agreement,
        )
    except LLMQuotaExhaustedError as e:
        print(f"\n[QUOTA_EXHAUSTED] {e}")
        print("Pipeline stopped cleanly. Previous results remain cached.")
        sys.exit(1)

    summary = result.get("summary", {})
    print(f"\nS5 Judge complete.")
    print(f"Episodes processed : {summary.get('episodes_processed')}")
    print(f"Candidates judged  : {summary.get('total_s4_survivors_judged')}")
    print(f"Variants generated : {summary.get('total_variants_generated')}")
    print(f"Passed judge gates : {summary.get('total_passed_judge_gates')}")


if __name__ == "__main__":
    main()
