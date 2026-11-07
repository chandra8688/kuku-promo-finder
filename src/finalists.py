"""One model call over the show's finalists: hook type and a "why it works" line.

The judge scores clips one episode at a time and writes critiques. The brief
asks for a line on why each promo works, and wave 1 needs hook types to spread
across. This stage sends the top finalists (script text only: no beat IDs, no
timestamps) to one role in a single call. It is optional: if the call fails or
no model is available, ranking falls back to code-built reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.judge import _format_beat_line
from src.llm import call

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "finalist_v1.md"
HOOK_TYPES = {
    "physical_violence", "humiliation_betrayal", "shocking_line", "identity_reveal",
    "danger_threat", "romantic_tension", "power_reversal", "other",
}


class FinalistClip(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n: int
    hook_type: str = "other"
    why_it_works: str = ""

    @field_validator("hook_type", mode="before")
    @classmethod
    def norm_type(cls, v: Any) -> str:
        v = str(v or "other").strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in HOOK_TYPES else "other"


class FinalistOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    clips: List[FinalistClip] = Field(default_factory=list)


def clip_text(promo: Dict[str, Any], beats_by_id: Dict[str, Dict[str, Any]], max_lines: int = 40) -> str:
    lines = []
    for b_id in promo.get("beat_ids", [])[:max_lines]:
        b = beats_by_id.get(b_id)
        if b and b.get("kind") != "CAMERA":
            lines.append(_format_beat_line(len(lines) + 1, b, None))
    return "\n".join(lines)


def build_prompt(finalists: List[Dict[str, Any]], parsed_dir: Path) -> str:
    cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    parts = [f"There are {len(finalists)} clips.", ""]
    for n, p in enumerate(finalists, start=1):
        ep = p["episode"]
        if ep not in cache:
            data = json.loads((parsed_dir / f"{ep}.json").read_text(encoding="utf-8"))
            cache[ep] = {b["beat_id"]: b for b in data.get("beats", [])}
        parts.append(f"=== CLIP {n} ({p['duration_s']:.0f} s) ===")
        parts.append(clip_text(p, cache[ep]))
        parts.append("")
    return "\n".join(parts)


def run_finalists(
    promos: List[Dict[str, Any]],
    parsed_dir: Path,
    role: str = "strong",
    max_candidates: int = 25,
    enable_cache: bool = True,
    fallback_roles: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return {candidate_id: {hook_type, why_it_works}}; empty on any failure."""
    finalists = promos[:max_candidates]
    if not finalists:
        return {}
    system = PROMPT_PATH.read_text(encoding="utf-8")
    user = build_prompt(finalists, parsed_dir)
    res = None
    # A busy model (503) should not cost the run its reasons: try the next role.
    for r in [role] + list(fallback_roles or []):
        try:
            res = call(role=r, system=system, user=user, schema=FinalistOutput,
                       stage="finalists", prompt_version="finalist_v1", enable_cache=enable_cache)
            if r != role:
                print(f"[finalists] '{role}' unavailable; used fallback role '{r}'.")
            break
        except Exception as e:  # quota, overload, network, auth
            print(f"[finalists] role '{r}' failed ({type(e).__name__}).")
    if res is None:
        print("[finalists] skipped; using code-built reasons.")
        return {}
    out: FinalistOutput = res.parsed or FinalistOutput()
    info: Dict[str, Dict[str, str]] = {}
    for clip in out.clips:
        if 1 <= clip.n <= len(finalists):
            info[finalists[clip.n - 1]["candidate_id"]] = {
                "hook_type": clip.hook_type,
                "why_it_works": clip.why_it_works.strip(),
            }
    print(f"[finalists] labelled {len(info)}/{len(finalists)} finalists.")
    return info
