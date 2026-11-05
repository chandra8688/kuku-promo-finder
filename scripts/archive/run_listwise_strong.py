"""Listwise Re-Ranking with Strong Role (gemini-3.6-flash) (scripts/run_listwise_strong.py).

Executes listwise ranking forward and reverse with strong role under run label 'listwise_strong'.
Computes Kendall tau vs judge ranking and top-5 overlap.
Outputs saved to posthoc/listwise_strong.json (POST-HOC).
Never prints screenplay text, loglines, or character names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from scipy.stats import kendalltau

from src.llm import call
from scripts.build_final_outputs import ListwiseRankingOutput


def main() -> None:
    # 1. Load the 15 promos from outputs/promos_ranked.json
    with open("outputs/promos_ranked.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    ranked_promos = data["ranked_promos"]
    assert len(ranked_promos) == 15, f"Expected 15 promos, got {len(ranked_promos)}"

    # Order 1 (forward) and Order 2 (reverse)
    order1_cands = list(ranked_promos)
    order2_cands = list(reversed(ranked_promos))

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
            hook = c.get("hook_text", "")
            cliff = c.get("cliffhanger_text", "")
            why = c.get("why", "") or c.get("post_first_reason", "")

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

    print("[listwise_strong] Executing forward call (Order 1) with role='strong'...")
    res1 = call(
        role="strong",
        system=sys_prompt,
        user=prompt1,
        schema=ListwiseRankingOutput,
        stage="listwise_order1",
        run_id="listwise_strong",
        enable_cache=False,
    )

    print("[listwise_strong] Executing reverse call (Order 2) with role='strong'...")
    res2 = call(
        role="strong",
        system=sys_prompt,
        user=prompt2,
        schema=ListwiseRankingOutput,
        stage="listwise_order2",
        run_id="listwise_strong",
        enable_cache=False,
    )

    out1: ListwiseRankingOutput = res1.parsed or ListwiseRankingOutput()
    out2: ListwiseRankingOutput = res2.parsed or ListwiseRankingOutput()

    # Map candidate_id to rank in order1
    rank_map1 = {}
    for item in out1.rankings:
        c_num = item.candidate_num
        if 1 <= c_num <= len(meta1):
            cid = meta1[c_num - 1]["candidate"]["candidate_id"]
            rank_map1[cid] = item.rank

    # Map candidate_id to rank in order2
    rank_map2 = {}
    for item in out2.rankings:
        c_num = item.candidate_num
        if 1 <= c_num <= len(meta2):
            cid = meta2[c_num - 1]["candidate"]["candidate_id"]
            rank_map2[cid] = item.rank

    # Compute averaged rank for strong role
    strong_scores = []
    for p in ranked_promos:
        cid = p["candidate_id"]
        r1 = rank_map1.get(cid, 8.0)
        r2 = rank_map2.get(cid, 8.0)
        avg_r = (r1 + r2) / 2.0
        strong_scores.append((cid, p, r1, r2, avg_r))

    strong_scores.sort(key=lambda x: (x[4], x[1]["s3a_score"]))

    # Assign final strong ranks 1..15
    judge_ranks = []
    strong_ranks = []
    strong_ranked_list = []

    for final_rank, (cid, p, r1, r2, avg_r) in enumerate(strong_scores, start=1):
        judge_r = p["overall_rank"]
        judge_ranks.append(judge_r)
        strong_ranks.append(final_rank)
        strong_ranked_list.append({
            "strong_rank": final_rank,
            "judge_rank": judge_r,
            "candidate_id": cid,
            "episode": p["episode"],
            "duration_s": p["duration_s"],
            "avg_rank": avg_r,
            "order1_rank": r1,
            "order2_rank": r2,
        })

    # Kendall tau
    tau, p_value = kendalltau(judge_ranks, strong_ranks)

    # Top-5 overlap
    judge_top5 = {p["candidate_id"] for p in ranked_promos if p["overall_rank"] <= 5}
    strong_top5 = {item["candidate_id"] for item in strong_ranked_list if item["strong_rank"] <= 5}
    top5_overlap = len(judge_top5 & strong_top5)

    posthoc_dir = Path("posthoc")
    posthoc_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "run_id": "listwise_strong",
        "model": res1.model,
        "calls": 2,
        "order1_tokens_in": res1.input_tokens,
        "order1_tokens_out": res1.output_tokens,
        "order1_latency_s": res1.latency_s,
        "order2_tokens_in": res2.input_tokens,
        "order2_tokens_out": res2.output_tokens,
        "order2_latency_s": res2.latency_s,
        "kendall_tau": round(float(tau), 4),
        "kendall_p_value": round(float(p_value), 4),
        "top5_overlap": f"{top5_overlap} of 5",
        "judge_top5_cands": sorted(list(judge_top5)),
        "strong_top5_cands": sorted(list(strong_top5)),
        "strong_ranked_promos": strong_ranked_list,
    }

    out_json = posthoc_dir / "listwise_strong.json"
    out_json.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"\n[listwise_strong] Completed successfully.")
    print(f"Kendall tau: {tau:.4f} (p={p_value:.4f})")
    print(f"Top-5 overlap: {top5_overlap} of 5")
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
