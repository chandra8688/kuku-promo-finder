"""Build Final Outputs for All 50 Episodes (scripts/build_final_outputs.py).

Implements Task 6:
1. Per-episode promos from frozen pipeline (outputs/promos_per_episode.csv and .json)
2. Show-level ranking for top 15 promos with diversity cap of at most 2 per episode,
   evaluated via two listwise LLM calls in opposite orders, averaged, tie-broken by S3a score
   (outputs/promos_ranked.csv and .json).
3. Run manifest (outputs/run_manifest.json)

Outputs directory is strictly gitignored.
Never prints screenplay text, loglines, or character names to terminal or chat.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.llm import LLMResult, call, get_gateway
from src.reserved import is_freeze_tag_present


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
    m = int(s // 60)
    sec = s % 60
    return f"{m:02d}:{sec:05.2f}"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def get_file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Per-Episode Promos
# ---------------------------------------------------------------------------

def build_per_episode_promos(
    judged_dir: str | Path = "data/judged",
    parsed_dir: str | Path = "data/parsed",
    outputs_dir: str | Path = "outputs",
) -> List[Dict[str, Any]]:
    j_path = Path(judged_dir)
    p_path = Path(parsed_dir)
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_promos = []

    for ep_num in range(1, 51):
        ep_id = f"episode-{ep_num:03d}"
        j_file = j_path / f"{ep_id}.json"
        p_file = p_path / f"{ep_id}.json"

        if not j_file.exists() or not p_file.exists():
            continue

        with open(j_file, "r", encoding="utf-8") as jf:
            j_data = json.load(jf)
        with open(p_file, "r", encoding="utf-8") as pf:
            p_data = json.load(pf)

        beats = p_data.get("beats", [])
        beat_lookup = {b["beat_id"]: b for b in beats}

        judged_cands = j_data.get("judged", [])

        # Sort by frozen ranking rule: passed_judge_gates first, then s3a_score desc, then final_score desc
        judged_cands.sort(
            key=lambda x: (
                1 if x.get("passed_judge_gates", False) else 0,
                x.get("s3a_score", 0.0),
                x.get("final_score", 0),
            ),
            reverse=True,
        )

        for rank_idx, cand in enumerate(judged_cands, start=1):
            st = cand.get("start_s", 0.0)
            en = cand.get("end_s", 0.0)
            dur = cand.get("duration_s", round(en - st, 2))
            beat_ids = cand.get("beat_ids", [])

            # Extract hook and cliffhanger lines from parsed beats using local indices
            h_idx = cand.get("hook_index", 1) or 1
            c_idx = cand.get("cliff_index", len(beat_ids)) or len(beat_ids)

            h_beat_id = beat_ids[h_idx - 1] if (1 <= h_idx <= len(beat_ids)) else (beat_ids[0] if beat_ids else "")
            c_beat_id = beat_ids[c_idx - 1] if (1 <= c_idx <= len(beat_ids)) else (beat_ids[-1] if beat_ids else "")

            hook_line = beat_lookup.get(h_beat_id, {}).get("text", "")
            cliff_line = beat_lookup.get(c_beat_id, {}).get("text", "")

            # Format timing
            seg_sec = f"[{st:.2f}, {en:.2f}]"
            seg_mmss = f"{format_seconds_to_mmss(st)}-{format_seconds_to_mmss(en)}"

            promo_entry = {
                "episode": ep_id,
                "rank_in_episode": rank_idx,
                "candidate_id": cand.get("candidate_id"),
                "segments_seconds": seg_sec,
                "segments_mmss": seg_mmss,
                "start_s": st,
                "end_s": en,
                "duration_s": dur,
                "hook_text": hook_line,
                "cliffhanger_text": cliff_line,
                "why": cand.get("why", ""),
                "scores": {
                    "s3a_score": cand.get("s3a_score", 0.0),
                    "judge_final_score": cand.get("final_score", 0),
                    "hook": cand.get("hook", 0),
                    "cliff": cand.get("cliff", 0),
                    "clarity": cand.get("clarity", 0),
                    "payoff": cand.get("payoff", 0),
                    "resolved_after": cand.get("resolved_after", False),
                },
                "low_confidence": cand.get("low_confidence", False),
                "provenance": {
                    "annotator_model": "gemini-3.5-flash-lite",
                    "annotator_prompt": "s2_v1",
                    "judge_model": "gemini-3.1-flash-lite",
                    "judge_prompt": "judge_v2",
                    "run_label": "primary",
                },
            }
            all_promos.append(promo_entry)

    # Save to JSON
    with open(out_path / "promos_per_episode.json", "w", encoding="utf-8") as f:
        json.dump(all_promos, f, indent=2)

    # Save to CSV
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
        "why",
        "hook_text",
        "cliffhanger_text",
        "annotator_model",
        "judge_model",
    ]
    with open(out_path / "promos_per_episode.csv", "w", newline="", encoding="utf-8") as f:
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
                "why": p["why"],
                "hook_text": p["hook_text"],
                "cliffhanger_text": p["cliffhanger_text"],
                "annotator_model": p["provenance"]["annotator_model"],
                "judge_model": p["provenance"]["judge_model"],
            })

    return all_promos


# ---------------------------------------------------------------------------
# 2. Show-Level Ranking (Top 15 Promos)
# ---------------------------------------------------------------------------

def select_top15_promos_with_diversity(promos: List[Dict[str, Any]], max_per_episode: int = 2) -> List[Dict[str, Any]]:
    # Filter to episode rank 1 or passed judge gates, sort by frozen rule
    sorted_p = sorted(
        promos,
        key=lambda x: (
            1 if x["scores"]["hook"] >= 2 and x["scores"]["cliff"] >= 2 and not x["scores"]["resolved_after"] else 0,
            x["scores"]["s3a_score"],
            x["scores"]["judge_final_score"],
        ),
        reverse=True,
    )

    selected = []
    ep_counts: Dict[str, int] = {}
    for p in sorted_p:
        ep = p["episode"]
        if ep_counts.get(ep, 0) < max_per_episode:
            selected.append(p)
            ep_counts[ep] = ep_counts.get(ep, 0) + 1
        if len(selected) == 15:
            break
    return selected


def run_listwise_show_ranking(
    top15_promos: List[Dict[str, Any]],
    parsed_dir: str | Path = "data/parsed",
    outputs_dir: str | Path = "outputs",
) -> Tuple[List[Dict[str, Any]], str]:
    """Rank top 15 promos with two listwise LLM calls in opposite orders."""
    gw = get_gateway()
    rem_strong = gw.get_remaining_quota("strong")
    # Use strong if reachable and quota > 0, else role judge
    if rem_strong is not None and rem_strong >= 2:
        listwise_role = "strong"
    else:
        listwise_role = "judge"

    print(f"\n[Show-Level Ranking] Using role '{listwise_role}' for listwise calls (strong quota remaining: {rem_strong})...")

    # Build prompt representations for candidates 1..15
    # Forward order (Order 1)
    order1_cands = list(top15_promos)
    # Reverse order (Order 2)
    order2_cands = list(reversed(top15_promos))

    def build_listwise_user_prompt(cands_list: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        parts = []
        parts.append("You are the chief promo editor. Below are 15 top candidate promo clips from across a 50-episode microdrama series.")
        parts.append("Compare them directly and rank them from 1 to 15 (1 = best promo across the whole show to post first to drive viewers into the series).")
        parts.append("Use strictly unique integer ranks from 1 to 15 with NO ties.")
        parts.append("")

        meta_list = []
        for idx, c in enumerate(cands_list, start=1):
            ep = c["episode"]
            dur = c["duration_s"]
            hook = c["hook_text"]
            cliff = c["cliffhanger_text"]
            why = c["why"]

            parts.append(f"=== CANDIDATE {idx} ===")
            parts.append(f"Episode: {ep} | Duration: {dur:.1f}s")
            parts.append(f"Hook beat line: \"{hook}\"")
            parts.append(f"Ending cliffhanger line: \"{cliff}\"")
            parts.append(f"Evaluation note: {why}")
            parts.append("")
            meta_list.append({"candidate_num": idx, "candidate": c})

        parts.append("Output strictly valid compact JSON with a list of rankings: [{'candidate_num': i, 'rank': r, 'why': '...'}].")
        return "\n".join(parts), meta_list

    sys_prompt = (
        "You are an expert microdrama promo editor. Rank candidate promos by their viral potential, "
        "hook urgency, and unresolved tension. Return strictly valid JSON."
    )

    prompt1, meta1 = build_listwise_user_prompt(order1_cands)
    prompt2, meta2 = build_listwise_user_prompt(order2_cands)

    res1 = call(
        role=listwise_role,
        system=sys_prompt,
        user=prompt1,
        schema=ListwiseRankingOutput,
        stage="listwise_order1",
        enable_cache=True,
    )
    res2 = call(
        role=listwise_role,
        system=sys_prompt,
        user=prompt2,
        schema=ListwiseRankingOutput,
        stage="listwise_order2",
        enable_cache=True,
    )

    out1: ListwiseRankingOutput = res1.parsed or ListwiseRankingOutput()
    out2: ListwiseRankingOutput = res2.parsed or ListwiseRankingOutput()

    map1 = {item.candidate_num: item.rank for item in out1.rankings}
    map2 = {item.candidate_num: item.rank for item in out2.rankings}

    # Map back to candidate_id
    ranks_order1: Dict[str, int] = {}
    for m in meta1:
        cid = m["candidate"]["candidate_id"]
        c_num = m["candidate_num"]
        ranks_order1[cid] = map1.get(c_num, c_num)

    ranks_order2: Dict[str, int] = {}
    for m in meta2:
        cid = m["candidate"]["candidate_id"]
        c_num = m["candidate_num"]
        ranks_order2[cid] = map2.get(c_num, c_num)

    # Average ranks and tie-break by S3a score
    ranked_promos = []
    for c in top15_promos:
        cid = c["candidate_id"]
        r1 = ranks_order1.get(cid, 8)
        r2 = ranks_order2.get(cid, 8)
        avg_r = round((r1 + r2) / 2.0, 2)
        ep_num = int(c["episode"].split("-")[-1])

        ranked_promos.append({
            "candidate_id": cid,
            "episode": c["episode"],
            "landing_episode": ep_num,  # Promo's own episode number (viewers land on early episodes)
            "start_s": c["start_s"],
            "end_s": c["end_s"],
            "duration_s": c["duration_s"],
            "segments_seconds": c["segments_seconds"],
            "segments_mmss": c["segments_mmss"],
            "avg_listwise_rank": avg_r,
            "listwise_rank_order1": r1,
            "listwise_rank_order2": r2,
            "s3a_score": c["scores"]["s3a_score"],
            "judge_final_score": c["scores"]["judge_final_score"],
            "post_first_reason": c["why"],
            "hook_text": c["hook_text"],
            "cliffhanger_text": c["cliffhanger_text"],
        })

    # Sort by avg_listwise_rank asc, then s3a_score desc, then judge_final_score desc
    ranked_promos.sort(key=lambda x: (x["avg_listwise_rank"], -x["s3a_score"], -x["judge_final_score"]))

    for idx, p in enumerate(ranked_promos, start=1):
        p["overall_rank"] = idx

    out_p = Path(outputs_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(out_p / "promos_ranked.json", "w", encoding="utf-8") as f:
        json.dump({
            "assumption": "Viewers land on the early free episodes; landing_episode is set to the promo's own episode number.",
            "listwise_role": listwise_role,
            "ranked_promos": ranked_promos,
        }, f, indent=2)

    # Write CSV
    csv_path = out_p / "promos_ranked.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Assumption: Viewers land on the early free episodes; landing_episode is set to the promo's own episode number.\n")
        writer = csv.DictWriter(f, fieldnames=[
            "overall_rank",
            "episode",
            "candidate_id",
            "segments_seconds",
            "segments_mmss",
            "duration_s",
            "avg_listwise_rank",
            "listwise_rank_order1",
            "listwise_rank_order2",
            "s3a_score",
            "judge_final_score",
            "post_first_reason",
            "landing_episode",
        ])
        writer.writeheader()
        for p in ranked_promos:
            writer.writerow({
                "overall_rank": p["overall_rank"],
                "episode": p["episode"],
                "candidate_id": p["candidate_id"],
                "segments_seconds": p["segments_seconds"],
                "segments_mmss": p["segments_mmss"],
                "duration_s": p["duration_s"],
                "avg_listwise_rank": p["avg_listwise_rank"],
                "listwise_rank_order1": p["listwise_rank_order1"],
                "listwise_rank_order2": p["listwise_rank_order2"],
                "s3a_score": p["s3a_score"],
                "judge_final_score": p["judge_final_score"],
                "post_first_reason": p["post_first_reason"],
                "landing_episode": p["landing_episode"],
            })

    return ranked_promos, listwise_role


# ---------------------------------------------------------------------------
# 3. Run Manifest
# ---------------------------------------------------------------------------

def build_run_manifest(outputs_dir: str | Path = "outputs") -> Dict[str, Any]:
    gw = get_gateway()
    ledger_records = []
    usage_file = Path("runs/usage.json")
    if usage_file.exists():
        with open(usage_file, "r", encoding="utf-8") as uf:
            u_data = json.load(uf)
            ledger_records = u_data.get("records", [])

    total_calls = len(ledger_records)
    cached_calls = sum(1 for r in ledger_records if r.get("cached", False))
    non_cached_calls = total_calls - cached_calls
    total_tin = sum(r.get("tokens_in", 0) for r in ledger_records)
    total_tout = sum(r.get("tokens_out", 0) for r in ledger_records)
    total_tthink = sum(r.get("thinking_tokens", 0) for r in ledger_records)

    non_cached_by_stage_model: Dict[str, int] = {}
    for r in ledger_records:
        if not r.get("cached", False):
            key = f"{r.get('stage')}:{r.get('model_version_returned') or r.get('model')}"
            non_cached_by_stage_model[key] = non_cached_by_stage_model.get(key, 0) + 1

    manifest = {
        "models": gw.roles_config,
        "git_commit": get_git_commit(),
        "git_tag": "config-frozen-before-gold",
        "freeze_tag_present": is_freeze_tag_present("config-frozen-before-gold"),
        "config_hashes": {
            "scoring_yaml": get_file_sha256("config/scoring.yaml"),
            "models_yaml": get_file_sha256("config/models.yaml"),
        },
        "prompt_versions": {
            "s2": "s2_v1",
            "judge": "judge_v2",
        },
        "token_totals": {
            "total_input_tokens": total_tin,
            "total_output_tokens": total_tout,
            "total_thinking_tokens": total_tthink,
            "total_all_tokens": total_tin + total_tout + total_tthink,
        },
        "call_counts": {
            "total_calls": total_calls,
            "cached_calls": cached_calls,
            "non_cached_calls": non_cached_calls,
            "non_cached_by_stage_model": non_cached_by_stage_model,
        },
        "privacy_note": (
            "Screenplay text and extracted candidates are confidential under Rule 1. "
            "Data directories (data/, outputs/, .env) are strictly gitignored. "
            "All model calls were routed through src/llm.py exclusively with centralized token tracking."
        ),
    }

    out_p = Path(outputs_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "run_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    return manifest


def main() -> None:
    print("\n[Final Outputs] 1. Building per-episode promos for all 50 episodes...")
    all_promos = build_per_episode_promos()
    print(f"[Final Outputs] Wrote {len(all_promos)} promos to outputs/promos_per_episode.json and .csv")

    print("\n[Final Outputs] 2. Selecting top 15 promos with diversity cap (max 2/ep)...")
    top15 = select_top15_promos_with_diversity(all_promos, max_per_episode=2)
    ranked, role_used = run_listwise_show_ranking(top15)
    print(f"[Final Outputs] Wrote {len(ranked)} show-ranked promos (role={role_used}) to outputs/promos_ranked.json and .csv")

    print("\n[Final Outputs] 3. Building run manifest...")
    manifest = build_run_manifest()
    print("Wrote run manifest to outputs/run_manifest.json")


if __name__ == "__main__":
    main()
