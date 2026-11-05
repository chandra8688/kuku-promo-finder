"""Run sensitivity analysis on dev episodes (scripts/run_sensitivity_dev.py).

Evaluates 28 weight perturbations on dev episodes (episode-001, episode-002, episode-005).
Reports:
- pool size per episode
- set overlap (IoU >= 0.5)
- top-1 stability
- Kendall tau over top 5
Never prints screenplay text or character names!
"""

import json
from pathlib import Path
from src.sensitivity import run_weight_sensitivity_analysis, run_extreme_weight_check

DEV_EPISODES = ["episode-001", "episode-002", "episode-005"]

episodes_windows = {}
for ep in DEV_EPISODES:
    cand_p = Path(f"data/candidates/{ep}.json")
    if cand_p.exists():
        d = json.loads(cand_p.read_text(encoding="utf-8"))
        episodes_windows[ep] = d.get("candidates", [])

print("=" * 80)
print("SENSITIVITY ANALYSIS ON DEV EPISODES (STRUCTURAL POOL ONLY)")
print("=" * 80)
res_struct = run_weight_sensitivity_analysis(
    episodes_windows=episodes_windows,
    seed=42,
    top_k=3,
    max_iou=0.5,
    structural_gates_only=True,
)
print(f"Pool sizes: {res_struct['pool_sizes']}")
print(f"Distinct (hook, cliff) combos: {res_struct['distinct_hook_cliff_combos']}")
print(f"Overall Mean Set Overlap (IoU >= 0.5): {res_struct['overall_mean_overlap']}")
print(f"Per-episode Mean Overlap: {res_struct['per_episode_mean_overlap']}")
print(f"Overall Top-1 Stability: {res_struct['overall_top1_stability']}")
print(f"Per-episode Top-1 Stability: {res_struct['per_episode_top1_stability']}")
print(f"Overall Kendall Tau (top 5): {res_struct['overall_kendall_tau_top5']}")
print(f"Per-episode Kendall Tau (top 5): {res_struct['per_episode_kendall_tau_top5']}")

print("\n" + "=" * 80)
print("EXTREME WEIGHT CHECK (100 vs 0 weights)")
print("=" * 80)
res_extreme = run_extreme_weight_check(
    episodes_windows=episodes_windows,
    top_k=3,
    max_iou=0.5,
    structural_gates_only=True,
)
print(f"Extreme weight check confirmed sensitivity: {res_extreme['sensitivity_confirmed']}")
for ep, vals in res_extreme['per_episode'].items():
    print(f"  {ep}: {vals}")

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS ON DEV EPISODES (ALL GATES / SURVIVORS)")
print("=" * 80)
res_all = run_weight_sensitivity_analysis(
    episodes_windows=episodes_windows,
    seed=42,
    top_k=3,
    max_iou=0.5,
    structural_gates_only=False,
)
print(f"Pool sizes: {res_all['pool_sizes']}")
print(f"Distinct (hook, cliff) combos: {res_all['distinct_hook_cliff_combos']}")
print(f"Overall Mean Set Overlap (IoU >= 0.5): {res_all['overall_mean_overlap']}")
print(f"Per-episode Mean Overlap: {res_all['per_episode_mean_overlap']}")
print(f"Overall Top-1 Stability: {res_all['overall_top1_stability']}")
print(f"Per-episode Top-1 Stability: {res_all['per_episode_top1_stability']}")
print(f"Overall Kendall Tau (top 5): {res_all['overall_kendall_tau_top5']}")
print(f"Per-episode Kendall Tau (top 5): {res_all['per_episode_kendall_tau_top5']}")

