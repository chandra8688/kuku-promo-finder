"""Part G: Judge Model Swap (role='alt', gemini-3.5-flash-lite).

Runs the 9 selected candidate survivors + 12 Part F controls through role 'alt'.
Compares against primary judge ('gemini-3.1-flash-lite').

Reports:
- Pass/fail agreement
- Per-field exact agreement (hook, cliff, clarity, payoff, resolved_after)
- Mean absolute difference (MAD) for each field and final_score
- Controls vs selected picks comparison for both models.
Never prints screenplay text or character names!
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.run_part_f_controls import collect_controls
from src.judge import judge_candidate

DEV_EPISODES = ["episode-001", "episode-002", "episode-005"]


def run_model_swap():
    # 1. Load Part E candidate survivors from data/judged
    cand_primary: Dict[str, Dict[str, Any]] = {}
    for ep in DEV_EPISODES:
        jp = Path(f"data/judged/{ep}.json")
        if jp.exists():
            d = json.loads(jp.read_text(encoding="utf-8"))
            for j in d.get("judged", []):
                cand_primary[j["candidate_id"]] = j

    # 2. Load Part F controls from data/judged_controls_judge.json
    ctrl_p = Path("data/judged_controls_judge.json")
    ctrl_primary_list = json.loads(ctrl_p.read_text(encoding="utf-8")) if ctrl_p.exists() else []
    ctrl_primary: Dict[str, Dict[str, Any]] = {c["candidate_id"]: c for c in ctrl_primary_list}

    print(f"[Part G] Loaded {len(cand_primary)} candidate picks and {len(ctrl_primary)} controls from primary judge.")

    # 3. Run candidates on role 'alt'
    alt_cand_results: List[Dict[str, Any]] = []
    for ep in DEV_EPISODES:
        parsed_p = Path(f"data/parsed/{ep}.json")
        ann_p = Path(f"data/annotated/{ep}.json")
        cand_p = Path(f"data/candidates/{ep}.json")
        parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
        ann_data = json.loads(ann_p.read_text(encoding="utf-8"))
        cand_data = json.loads(cand_p.read_text(encoding="utf-8"))
        cand_map = {c["candidate_id"]: c for c in cand_data.get("candidates", [])}

        for cid, prim in cand_primary.items():
            if prim.get("episode") == ep:
                cand = cand_map.get(cid)
                if not cand:
                    continue
                res = judge_candidate(
                    candidate=cand,
                    parsed_data=parsed_data,
                    annotated_data=ann_data,
                    role="alt",
                    stage="s5_judge_swap_candidates",
                    enable_cache=True,
                )
                alt_cand_results.append(res)

    # 4. Run controls on role 'alt'
    controls_by_ep = collect_controls()
    alt_ctrl_results: List[Dict[str, Any]] = []
    for ep, cand_list in controls_by_ep.items():
        parsed_p = Path(f"data/parsed/{ep}.json")
        ann_p = Path(f"data/annotated/{ep}.json")
        parsed_data = json.loads(parsed_p.read_text(encoding="utf-8"))
        ann_data = json.loads(ann_p.read_text(encoding="utf-8"))

        for cand in cand_list:
            res = judge_candidate(
                candidate=cand,
                parsed_data=parsed_data,
                annotated_data=ann_data,
                role="alt",
                stage="s5_judge_swap_controls",
                enable_cache=True,
            )
            res["control_type"] = cand.get("control_type")
            res["control_name"] = cand.get("control_name")
            alt_ctrl_results.append(res)

    # Save alt results
    Path("data/judged_swap_candidates.json").write_text(json.dumps(alt_cand_results, indent=2), encoding="utf-8")
    Path("data/judged_swap_controls.json").write_text(json.dumps(alt_ctrl_results, indent=2), encoding="utf-8")

    # 5. Analysis & Comparison
    alt_cand_map = {c["candidate_id"]: c for c in alt_cand_results}
    alt_ctrl_map = {c["candidate_id"]: c for c in alt_ctrl_results}

    all_pairs = []
    for cid, prim in cand_primary.items():
        if cid in alt_cand_map:
            all_pairs.append(("candidate", prim, alt_cand_map[cid]))
    for cid, prim in ctrl_primary.items():
        if cid in alt_ctrl_map:
            all_pairs.append(("control", prim, alt_ctrl_map[cid]))

    N = len(all_pairs)
    print(f"\n[Part G Analysis] Comparing {N} total items (9 candidates + 12 controls) between judge and alt:")

    pass_agree = sum(1 for _, p, a in all_pairs if p.get("passed_judge_gates") == a.get("passed_judge_gates"))
    hook_agree = sum(1 for _, p, a in all_pairs if p.get("hook") == a.get("hook"))
    cliff_agree = sum(1 for _, p, a in all_pairs if p.get("cliff") == a.get("cliff"))
    clarity_agree = sum(1 for _, p, a in all_pairs if p.get("clarity") == a.get("clarity"))
    payoff_agree = sum(1 for _, p, a in all_pairs if p.get("payoff") == a.get("payoff"))
    ra_agree = sum(1 for _, p, a in all_pairs if p.get("resolved_after") == a.get("resolved_after"))

    hook_mad = sum(abs(p.get("hook", 0) - a.get("hook", 0)) for _, p, a in all_pairs) / N
    cliff_mad = sum(abs(p.get("cliff", 0) - a.get("cliff", 0)) for _, p, a in all_pairs) / N
    clarity_mad = sum(abs(p.get("clarity", 0) - a.get("clarity", 0)) for _, p, a in all_pairs) / N
    payoff_mad = sum(abs(p.get("payoff", 0) - a.get("payoff", 0)) for _, p, a in all_pairs) / N
    score_mad = sum(abs(p.get("final_score", 0) - a.get("final_score", 0)) for _, p, a in all_pairs) / N

    print("\n--- AGREEMENT METRICS ---")
    print(f"Pass/Fail Gate Agreement : {pass_agree}/{N} ({pass_agree/N*100:.1f}%)")
    print(f"Hook Exact Agreement      : {hook_agree}/{N} ({hook_agree/N*100:.1f}%) | MAD: {hook_mad:.3f}")
    print(f"Cliff Exact Agreement     : {cliff_agree}/{N} ({cliff_agree/N*100:.1f}%) | MAD: {cliff_mad:.3f}")
    print(f"Clarity Exact Agreement   : {clarity_agree}/{N} ({clarity_agree/N*100:.1f}%) | MAD: {clarity_mad:.3f}")
    print(f"Payoff Exact Agreement    : {payoff_agree}/{N} ({payoff_agree/N*100:.1f}%) | MAD: {payoff_mad:.3f}")
    print(f"Resolved_After Agreement  : {ra_agree}/{N} ({ra_agree/N*100:.1f}%)")
    print(f"Final Score MAD           : {score_mad:.3f}")

    # Controls vs Picks Comparison
    print("\n--- CONTROLS VS SELECTED PICKS COMPARISON ---")
    cand_pairs = [p for p in all_pairs if p[0] == "candidate"]
    ctrl_pairs = [p for p in all_pairs if p[0] == "control"]

    for label, pairs in [("Selected Picks (9)", cand_pairs), ("Controls (12)", ctrl_pairs)]:
        p_hook = sum(p.get("hook", 0) for _, p, _ in pairs) / len(pairs)
        p_cliff = sum(p.get("cliff", 0) for _, p, _ in pairs) / len(pairs)
        p_clarity = sum(p.get("clarity", 0) for _, p, _ in pairs) / len(pairs)
        p_payoff = sum(p.get("payoff", 0) for _, p, _ in pairs) / len(pairs)
        p_score = sum(p.get("final_score", 0) for _, p, _ in pairs) / len(pairs)
        p_ra = sum(1 for _, p, _ in pairs if p.get("resolved_after")) / len(pairs)

        a_hook = sum(a.get("hook", 0) for _, _, a in pairs) / len(pairs)
        a_cliff = sum(a.get("cliff", 0) for _, _, a in pairs) / len(pairs)
        a_clarity = sum(a.get("clarity", 0) for _, _, a in pairs) / len(pairs)
        a_payoff = sum(a.get("payoff", 0) for _, _, a in pairs) / len(pairs)
        a_score = sum(a.get("final_score", 0) for _, _, a in pairs) / len(pairs)
        a_ra = sum(1 for _, _, a in pairs if a.get("resolved_after")) / len(pairs)

        print(f"\n{label}:")
        print(f"  Primary Judge (3.1-flash-lite) -> Hook: {p_hook:.2f}, Cliff: {p_cliff:.2f}, Clarity: {p_clarity:.2f}, Payoff: {p_payoff:.2f}, Score: {p_score:.2f}, Resolved_After Rate: {p_ra*100:.1f}%")
        print(f"  Alt Model     (3.5-flash-lite) -> Hook: {a_hook:.2f}, Cliff: {a_cliff:.2f}, Clarity: {a_clarity:.2f}, Payoff: {a_payoff:.2f}, Score: {a_score:.2f}, Resolved_After Rate: {a_ra*100:.1f}%")


if __name__ == "__main__":
    run_model_swap()

