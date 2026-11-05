"""Part D: Validity Test on First 10 Non-Reserved Episodes.

Tests judge role (gemini-3.1-flash-lite) and alt role (gemini-3.5-flash-lite)
using Judge v2 per_episode batched mode with strict unique ranks.

Candidate pool per episode:
- Selected picks (S4 survivors)
- Controls: up to 2 RESOLVED_ENDING-only rejected windows, 1 B1 baseline, 1 B2 baseline.

Evaluated in two opposite orders (Order A = shuffled seed 42, Order B = reversed Order A).
Reports counts only:
- pick-vs-control pairs won (n of N) per order and pooled
- episodes where top-1 agrees across the two orders
- rank correlation between the two orders
- agreement between judge and alt roles
Never prints screenplay text or character names!
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baselines import generate_b1_baseline, generate_b2_baseline, load_episode_durations
from src.judge import (
    CandidateEvaluation,
    EpisodeBatchJudgeOutput,
    build_episode_batch_judge_prompt,
    load_judge_system_prompt,
    validate_unique_ranks,
)
from src.llm import call

# First 10 non-reserved episodes
VALIDITY_EPISODES = [
    "episode-001",
    "episode-002",
    "episode-004",
    "episode-005",
    "episode-006",
    "episode-007",
    "episode-009",
    "episode-010",
    "episode-011",
    "episode-012",
]


def populate_beats(start_s: float, end_s: float, all_beats: List[Dict[str, Any]]) -> List[str]:
    b_ids = [
        b["beat_id"]
        for b in all_beats
        if b.get("kind") != "CAMERA"
        and b.get("start_s", 0.0) >= start_s - 0.05
        and b.get("end_s", 0.0) <= end_s + 0.05
    ]
    if not b_ids:
        b_ids = [
            b["beat_id"]
            for b in all_beats
            if b.get("kind") != "CAMERA"
            and b.get("end_s", 0.0) > start_s
            and b.get("start_s", 0.0) < end_s
        ]
    return b_ids


def build_episode_pool(ep_id: str, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (order_a_candidates, order_b_candidates) with metadata stripped for prompt."""
    parsed_p = Path(f"data/parsed/{ep_id}.json")
    cand_p = Path(f"data/candidates/{ep_id}.json")
    parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
    cand_data = json.loads(cand_p.read_text(encoding="utf-8"))
    all_beats = parsed_data.get("beats", [])

    durations = load_episode_durations("data/parsed")
    b1_dict = generate_b1_baseline(durations)
    b2_dict = generate_b2_baseline(durations, seeds=[0], windows_per_episode=1)[0]

    # 1. Selected picks (S4 survivors, top_k selected)
    all_cands = cand_data.get("candidates", [])
    picks = [c for c in all_cands if c.get("passed_all_gates")]
    # If more than 3, prioritize selected_top_k
    top_picks = [c for c in picks if c.get("selected_top_k", False)]
    if not top_picks:
        top_picks = picks[:3]

    pool = []
    for p in top_picks:
        item = dict(p)
        item["is_pick"] = True
        pool.append(item)

    # 2. Controls: up to 2 RESOLVED_ENDING-only windows
    res_cands = [c for c in all_cands if c.get("first_failed_gate") == "RESOLVED_ENDING"]
    res_cands.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    for rc in res_cands[:2]:
        item = dict(rc)
        item["is_pick"] = False
        pool.append(item)

    # 3. Control B1
    b1_win = b1_dict.get(ep_id, [{}])[0]
    b1_beats = populate_beats(b1_win["start_s"], b1_win["end_s"], all_beats)
    pool.append({
        "candidate_id": f"{ep_id}_ctrl_b1",
        "episode": ep_id,
        "start_s": b1_win["start_s"],
        "end_s": b1_win["end_s"],
        "duration_s": b1_win["duration_s"],
        "beat_ids": b1_beats,
        "is_pick": False,
    })

    # 4. Control B2
    b2_win = b2_dict.get(ep_id, [{}])[0]
    b2_beats = populate_beats(b2_win["start_s"], b2_win["end_s"], all_beats)
    pool.append({
        "candidate_id": f"{ep_id}_ctrl_b2",
        "episode": ep_id,
        "start_s": b2_win["start_s"],
        "end_s": b2_win["end_s"],
        "duration_s": b2_win["duration_s"],
        "beat_ids": b2_beats,
        "is_pick": False,
    })

    # Shuffle with fixed seed
    rng = random.Random(seed)
    order_a = list(pool)
    rng.shuffle(order_a)

    # Order B is exact reverse of Order A
    order_b = list(reversed(order_a))

    return order_a, order_b


def run_batch_eval(
    ep_id: str,
    candidates: List[Dict[str, Any]],
    role: str,
    order_name: str,
) -> Dict[str, int]:
    """Run per_episode Judge v2 on candidates and return candidate_id -> rank mapping."""
    parsed_p = Path(f"data/parsed/{ep_id}.json")
    ann_p = Path(f"data/annotated/{ep_id}.json")
    parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
    ann_data = json.loads(ann_p.read_text(encoding="utf-8"))

    prompt_file = Path("prompts/judge_v2.md")
    system_prompt = load_judge_system_prompt(prompt_file)

    batch_prompt, meta_list = build_episode_batch_judge_prompt(candidates, parsed_data, ann_data)

    stage_name = f"s5_validity_{role}_{order_name}"
    res = call(
        role=role,
        system=system_prompt,
        user=batch_prompt,
        schema=EpisodeBatchJudgeOutput,
        stage=stage_name,
        prompt_version="judge_v2",
        enable_cache=True,
    )

    batch_out: EpisodeBatchJudgeOutput = res.parsed
    is_valid, err = validate_unique_ranks(batch_out.evaluations, len(candidates)) if batch_out else (False, "unparsed")

    if not is_valid:
        repair_msg = (
            f"{batch_prompt}\n\n"
            f"RANK VALIDATION ERROR: {err}\n"
            f"Please output corrected evaluations for all {len(candidates)} candidates where each candidate receives a strictly unique integer rank from 1 to {len(candidates)} with NO ties."
        )
        rep_res = call(
            role=role,
            system=system_prompt,
            user=repair_msg,
            schema=EpisodeBatchJudgeOutput,
            stage=f"{stage_name}_repair",
            prompt_version="judge_v2",
            enable_cache=True,
        )
        if rep_res.parsed and rep_res.parsed.evaluations:
            rep_valid, _ = validate_unique_ranks(rep_res.parsed.evaluations, len(candidates))
            if rep_valid:
                batch_out = rep_res.parsed

    eval_map = {ev.candidate_num: ev.rank for ev in batch_out.evaluations} if (batch_out and batch_out.evaluations) else {}

    ranks: Dict[str, int] = {}
    for meta in meta_list:
        cid = meta["candidate"]["candidate_id"]
        c_num = meta["candidate_num"]
        ranks[cid] = eval_map.get(c_num, 999)

    return ranks


def compute_kendall_tau(ranks_a: Dict[str, int], ranks_b: Dict[str, int]) -> float:
    items = list(ranks_a.keys())
    n = len(items)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            c1 = items[i]
            c2 = items[j]
            ra1, ra2 = ranks_a[c1], ranks_a[c2]
            rb1, rb2 = ranks_b[c1], ranks_b[c2]
            diff_a = ra1 - ra2
            diff_b = rb1 - rb2
            if (diff_a > 0 and diff_b > 0) or (diff_a < 0 and diff_b < 0):
                concordant += 1
            elif (diff_a > 0 and diff_b < 0) or (diff_a < 0 and diff_b > 0):
                discordant += 1
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0


def evaluate_validity_for_role(role: str) -> Dict[str, Any]:
    print(f"\n[Part D] Running Validity Evaluation for role='{role}' on 10 episodes x 2 orders...")
    
    order_a_wins = 0
    order_a_pairs = 0
    order_b_wins = 0
    order_b_pairs = 0

    top1_agreements = 0
    correlations = []
    ep_results = {}

    for ep in VALIDITY_EPISODES:
        order_a_cands, order_b_cands = build_episode_pool(ep, seed=42)
        picks_ids = {c["candidate_id"] for c in order_a_cands if c.get("is_pick")}
        ctrl_ids = {c["candidate_id"] for c in order_a_cands if not c.get("is_pick")}

        ranks_a = run_batch_eval(ep, order_a_cands, role=role, order_name="order_a")
        ranks_b = run_batch_eval(ep, order_b_cands, role=role, order_name="order_b")

        # Order A pair wins
        ep_a_wins = 0
        ep_a_pairs = 0
        for p in picks_ids:
            for c in ctrl_ids:
                ep_a_pairs += 1
                if ranks_a.get(p, 999) < ranks_a.get(c, 999):
                    ep_a_wins += 1
        order_a_wins += ep_a_wins
        order_a_pairs += ep_a_pairs

        # Order B pair wins
        ep_b_wins = 0
        ep_b_pairs = 0
        for p in picks_ids:
            for c in ctrl_ids:
                ep_b_pairs += 1
                if ranks_b.get(p, 999) < ranks_b.get(c, 999):
                    ep_b_wins += 1
        order_b_wins += ep_b_wins
        order_b_pairs += ep_b_pairs

        # Top-1 agreement
        top1_a = min(ranks_a, key=ranks_a.get) if ranks_a else None
        top1_b = min(ranks_b, key=ranks_b.get) if ranks_b else None
        t1_agree = (top1_a == top1_b)
        if t1_agree:
            top1_agreements += 1

        # Rank correlation between orders
        tau = compute_kendall_tau(ranks_a, ranks_b)
        correlations.append(tau)

        ep_results[ep] = {
            "order_a_wins": f"{ep_a_wins}/{ep_a_pairs}",
            "order_b_wins": f"{ep_b_wins}/{ep_b_pairs}",
            "top1_a": top1_a,
            "top1_b": top1_b,
            "top1_agree": t1_agree,
            "tau": round(tau, 3),
            "ranks_a": ranks_a,
            "ranks_b": ranks_b,
        }

    pooled_wins = order_a_wins + order_b_wins
    pooled_pairs = order_a_pairs + order_b_pairs
    mean_tau = sum(correlations) / len(correlations) if correlations else 0.0

    return {
        "role": role,
        "order_a_wins": order_a_wins,
        "order_a_pairs": order_a_pairs,
        "order_a_win_rate": round(order_a_wins / max(1, order_a_pairs), 4),
        "order_b_wins": order_b_wins,
        "order_b_pairs": order_b_pairs,
        "order_b_win_rate": round(order_b_wins / max(1, order_b_pairs), 4),
        "pooled_wins": pooled_wins,
        "pooled_pairs": pooled_pairs,
        "pooled_win_rate": round(pooled_wins / max(1, pooled_pairs), 4),
        "top1_agreements": top1_agreements,
        "total_episodes": len(VALIDITY_EPISODES),
        "mean_rank_correlation": round(mean_tau, 4),
        "episodes": ep_results,
    }


def main():
    judge_res = evaluate_validity_for_role(role="judge")
    alt_res = evaluate_validity_for_role(role="alt")

    # Agreement between roles
    # Compare top-1 across roles on Order A
    top1_role_agree_a = 0
    top1_role_agree_b = 0
    role_taus_a = []
    role_taus_b = []

    for ep in VALIDITY_EPISODES:
        j_data = judge_res["episodes"][ep]
        a_data = alt_res["episodes"][ep]

        if j_data["top1_a"] == a_data["top1_a"]:
            top1_role_agree_a += 1
        if j_data["top1_b"] == a_data["top1_b"]:
            top1_role_agree_b += 1

        tau_a = compute_kendall_tau(j_data["ranks_a"], a_data["ranks_a"])
        tau_b = compute_kendall_tau(j_data["ranks_b"], a_data["ranks_b"])
        role_taus_a.append(tau_a)
        role_taus_b.append(tau_b)

    mean_role_tau = (sum(role_taus_a) + sum(role_taus_b)) / (len(role_taus_a) + len(role_taus_b))

    print("\n" + "=" * 80)
    print("PART D: VALIDITY TEST REPORT (10 Dev Episodes)")
    print("=" * 80)
    print(f"\n1. Primary Judge Role ({judge_res['role']}):")
    print(f"   Order A: Pick-vs-control pairs won: {judge_res['order_a_wins']}/{judge_res['order_a_pairs']} ({judge_res['order_a_win_rate']*100:.1f}%)")
    print(f"   Order B: Pick-vs-control pairs won: {judge_res['order_b_wins']}/{judge_res['order_b_pairs']} ({judge_res['order_b_win_rate']*100:.1f}%)")
    print(f"   Pooled : Pick-vs-control pairs won: {judge_res['pooled_wins']}/{judge_res['pooled_pairs']} ({judge_res['pooled_win_rate']*100:.1f}%)")
    print(f"   Top-1 Agreement Across Orders: {judge_res['top1_agreements']}/{judge_res['total_episodes']} episodes")
    print(f"   Mean Rank Correlation Across Orders: {judge_res['mean_rank_correlation']:.3f}")

    print(f"\n2. Alt Role ({alt_res['role']}):")
    print(f"   Order A: Pick-vs-control pairs won: {alt_res['order_a_wins']}/{alt_res['order_a_pairs']} ({alt_res['order_a_win_rate']*100:.1f}%)")
    print(f"   Order B: Pick-vs-control pairs won: {alt_res['order_b_wins']}/{alt_res['order_b_pairs']} ({alt_res['order_b_win_rate']*100:.1f}%)")
    print(f"   Pooled : Pick-vs-control pairs won: {alt_res['pooled_wins']}/{alt_res['pooled_pairs']} ({alt_res['pooled_win_rate']*100:.1f}%)")
    print(f"   Top-1 Agreement Across Orders: {alt_res['top1_agreements']}/{alt_res['total_episodes']} episodes")
    print(f"   Mean Rank Correlation Across Orders: {alt_res['mean_rank_correlation']:.3f}")

    print(f"\n3. Inter-Role Agreement (judge vs alt):")
    print(f"   Top-1 Agreement (Order A): {top1_role_agree_a}/{len(VALIDITY_EPISODES)} episodes")
    print(f"   Top-1 Agreement (Order B): {top1_role_agree_b}/{len(VALIDITY_EPISODES)} episodes")
    print(f"   Mean Rank Correlation (judge vs alt): {mean_role_tau:.3f}")

    out_file = Path("data/validity_test_results.json")
    out_file.write_text(json.dumps({
        "judge": judge_res,
        "alt": alt_res,
        "inter_role": {
            "top1_agree_order_a": top1_role_agree_a,
            "top1_agree_order_b": top1_role_agree_b,
            "mean_correlation": round(mean_role_tau, 4),
        }
    }, indent=2), encoding="utf-8")
    print(f"\nWrote full report to {out_file}")


if __name__ == "__main__":
    main()

