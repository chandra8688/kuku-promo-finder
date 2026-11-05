"""Pre-flight planning and quota verification tool (src/plan.py).

Calculates model call requirements per stage without making any LLM calls.
Validates requirements against rolling 24-hour quota from runs/usage.json.
Refuses to start if quota would be exceeded, unless --force is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.llm import ModelGateway, get_gateway
from src.reserved import is_episode_blocked


def count_stage_calls(
    stage: str,
    episodes: List[str],
    candidates_dir: str | Path = "data/candidates",
    parsed_dir: str | Path = "data/parsed",
    judge_mode: str = "per_candidate",
) -> Dict[str, Any]:
    """Calculate required calls and role for a stage across specified episodes."""
    stage_lower = stage.lower().strip()
    cands_path = Path(candidates_dir)
    parsed_path = Path(parsed_dir)

    # Filter out reserved episodes
    active_episodes = [ep for ep in episodes if not is_episode_blocked(ep)]

    if stage_lower in ("s2", "annotate", "s2_annotate"):
        role = "cheap"
        # 1 call per episode
        calls_needed = len(active_episodes)
        return {
            "stage": "s2_annotate",
            "role": role,
            "episodes_count": len(active_episodes),
            "calls_needed": calls_needed,
            "details": f"{calls_needed} episode(s), 1 call each",
        }

    elif stage_lower in ("s5", "judge", "s5_judge"):
        role = "judge"
        if judge_mode == "per_episode":
            calls_needed = len(active_episodes)
            return {
                "stage": "s5_judge",
                "role": role,
                "episodes_count": len(active_episodes),
                "calls_needed": calls_needed,
                "details": f"{calls_needed} episode(s) in batched per_episode mode",
            }
        else:
            # per_candidate mode: count survivors or estimate from dev average (3.0/ep)
            total_survivors = 0
            missing_eps = 0
            for ep in active_episodes:
                cand_file = cands_path / f"{ep}.json"
                if cand_file.exists():
                    try:
                        with open(cand_file, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        survivors = [c for c in d.get("candidates", []) if c.get("passed_all_gates")]
                        total_survivors += len(survivors)
                    except Exception:
                        missing_eps += 1
                else:
                    missing_eps += 1

            if missing_eps > 0:
                # Estimate ungenerated episodes at dev average of 3.0 survivors/ep
                est_additional = round(missing_eps * 3.0)
                total_survivors += est_additional
                details = f"{total_survivors} survivors ({len(active_episodes) - missing_eps} exact + {missing_eps} estimated @ 3.0/ep)"
            else:
                details = f"{total_survivors} survivor(s) across {len(active_episodes)} episode(s)"

            return {
                "stage": "s5_judge",
                "role": role,
                "episodes_count": len(active_episodes),
                "calls_needed": total_survivors,
                "details": details,
            }

    elif stage_lower in ("cold_viewer", "cold"):
        role = "cheap"
        calls_needed = len(active_episodes) * 2
        return {
            "stage": "cold_viewer",
            "role": role,
            "episodes_count": len(active_episodes),
            "calls_needed": calls_needed,
            "details": f"{len(active_episodes)} episode(s) x top 2 candidates = {calls_needed} calls",
        }

    elif stage_lower in ("listwise", "ranking", "listwise_ranking"):
        role = "strong"
        calls_needed = 2 if len(active_episodes) > 0 else 0
        return {
            "stage": "listwise",
            "role": role,
            "episodes_count": len(active_episodes),
            "calls_needed": calls_needed,
            "details": f"Global listwise ranking = 2 calls total (strong role, two orders)",
        }

    elif stage_lower in ("stability_s2", "stability_annotate"):
        role = "cheap"
        ep_subset = min(10, len(active_episodes))
        calls_needed = ep_subset * 3
        return {
            "stage": "stability_s2",
            "role": role,
            "episodes_count": ep_subset,
            "calls_needed": calls_needed,
            "details": f"{ep_subset} episode(s) x 3 reruns = {calls_needed} calls",
        }

    elif stage_lower in ("stability_judge",):
        role = "judge"
        ep_subset = min(10, len(active_episodes))
        # 3 reruns x ~3 survivors/ep = 90 calls
        calls_needed = round(ep_subset * 3.0 * 3)
        return {
            "stage": "stability_judge",
            "role": role,
            "episodes_count": ep_subset,
            "calls_needed": calls_needed,
            "details": f"{ep_subset} episode(s) x 3.0 survivors x 3 reruns = {calls_needed} calls",
        }
    else:
        # Unknown or no-LLM stage (e.g. s1, s3a, s4, tier0)
        return {
            "stage": stage,
            "role": "none",
            "episodes_count": len(active_episodes),
            "calls_needed": 0,
            "details": "Deterministic / no LLM calls required",
        }


def plan_execution(
    episodes: List[str],
    stages: List[str],
    judge_mode: str = "per_candidate",
    force: bool = False,
    models_config_path: str = "config/models.yaml",
) -> bool:
    """Pre-flight plan check. Prints table and returns True if feasible or forced."""
    gw = get_gateway()
    ledger = gw.usage_ledger

    # Load models config to inspect roles and limits
    try:
        with open(models_config_path, "r", encoding="utf-8") as f:
            mcfg = yaml.safe_load(f) or {}
        roles_cfg = mcfg.get("roles", {})
    except Exception:
        roles_cfg = {}

    print("\n" + "=" * 80)
    print("PROMO-FINDER PRE-FLIGHT EXECUTION PLAN")
    print(f"Current Day (LA): {ledger.get_current_la_day()} | Next Quota Reset: {ledger.get_next_reset_ist()}")
    print("=" * 80)

    all_feasible = True
    stage_plans = []
    active_episodes = [ep for ep in episodes if not is_episode_blocked(ep)]

    for st in stages:
        plan_info = count_stage_calls(st, active_episodes, judge_mode=judge_mode)
        role = plan_info["role"]
        calls = plan_info["calls_needed"]

        if role != "none":
            rcfg = roles_cfg.get(role, {})
            model = rcfg.get("model_name", "unknown")
            rpd = rcfg.get("rpd_limit")
            rpm = rcfg.get("rpm_limit", 15)
            est_minutes = round(calls / max(1, rpm), 1)
            stats = ledger.get_stats_24h(model)
            used_24h = stats.get("total_calls", 0)
            remaining = ledger.get_remaining_daily_quota(model, rpd)

            if remaining is not None:
                feasible = calls <= remaining
                status = "OK" if feasible else "QUOTA_EXCEEDED"
                if not feasible:
                    all_feasible = False
            else:
                feasible = True
                status = "OK (no limit)"

            plan_info.update({
                "model": model,
                "rpd_limit": rpd if rpd is not None else "unlimited",
                "est_minutes": est_minutes,
                "used_24h": used_24h,
                "remaining_quota": remaining if remaining is not None else "unlimited",
                "status": status,
                "feasible": feasible,
            })
        else:
            plan_info.update({
                "model": "none",
                "rpd_limit": "n/a",
                "est_minutes": 0.0,
                "used_24h": 0,
                "remaining_quota": "n/a",
                "status": "OK (0 calls)",
                "feasible": True,
            })

        stage_plans.append(plan_info)

    # Print summary table
    header = f"{'Stage':<16} | {'Role':<6} | {'Model':<22} | {'Calls':<6} | {'Est Min':<8} | {'24h Used':<9} | {'Daily Limit':<11} | {'Remaining':<9} | {'Status':<14}"
    print(header)
    print("-" * len(header))

    for p in stage_plans:
        row = (
            f"{p['stage']:<16} | "
            f"{p['role']:<6} | "
            f"{p['model']:<22} | "
            f"{p['calls_needed']:<6} | "
            f"{str(p['est_minutes'])+'m':<8} | "
            f"{str(p['used_24h']):<9} | "
            f"{str(p['rpd_limit']):<11} | "
            f"{str(p['remaining_quota']):<9} | "
            f"{p['status']:<14}"
        )
        print(row)

    print("-" * len(header))

    if not all_feasible:
        if force:
            print("\n[WARNING] Quota check failed, but --force flag was provided. Execution allowed.\n")
            return True
        else:
            print("\n[ERROR] Required calls exceed remaining rolling 24h quota. Refusing to start.")
            print("Use --force to bypass this check if you have quota or use cached results.\n")
            return False
    else:
        print("\n[SUCCESS] All stages feasible within current 24h quota limits.\n")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight execution planning and quota check.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["episode-001", "episode-002", "episode-005"],
        help="Episodes to evaluate",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["s2", "s5"],
        help="Stages to plan (e.g. s2 s5)",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["per_candidate", "per_episode"],
        default="per_candidate",
        help="Judge mode: per_candidate or per_episode",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Plan entire pipeline (S2, judge, cold viewer, listwise ranking, stability reruns)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass quota check failure",
    )
    args = parser.parse_args()

    stages = args.stages
    if args.full_pipeline:
        stages = ["s2", "s5_judge", "cold_viewer", "listwise", "stability_s2", "stability_judge"]

    feasible = plan_execution(
        episodes=args.episodes,
        stages=stages,
        judge_mode=args.judge_mode,
        force=args.force,
    )

    sys.exit(0 if feasible else 1)


if __name__ == "__main__":
    main()

