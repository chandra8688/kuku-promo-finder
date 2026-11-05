"""Stability Evaluation across 3 Pipeline Runs on 10 Reserved Episodes (scripts/run_stability.py).

Runs S2 -> S3a/S4 -> Judge v2 two more times with cache_key_extra "run2" and "run3".
Compares:
- Top-1 window agreement between runs (IoU >= 0.5; n of 10 episodes, per pair)
- Mean top-3 set overlap (mean IoU >= 0.5 set overlap per episode)
- S2 per-field exact agreement across runs (h, q, r, n, t)
- Gold hits per run (n of 7 gold windows recalled, IoU >= 0.5)

Primary run is the headline; does not pick the best run.
Never prints screenplay text, loglines, or character names.
Reports raw counts (n of N), never percentages.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.judge import run_judge_pipeline
from src.reserved import RESERVED_EPISODES
from src.windows import compute_interval_iou

RESERVED_LIST = sorted(list(RESERVED_EPISODES))
GOLD_EPISODES = [3, 13, 23, 33, 43]
GOLD_EP_STRS = [f"episode-{ep:03d}" for ep in GOLD_EPISODES]


def run_pipeline_iteration(run_label: str) -> None:
    """Run S2 -> S3a/S4 -> Judge for a stability iteration."""
    ann_dir = Path(f"data/stability/{run_label}/annotated")
    cand_dir = Path(f"data/stability/{run_label}/candidates")
    judged_dir = Path(f"data/stability/{run_label}/judged")

    ann_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    judged_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stability] Running {run_label}: Stage 2 Annotation (cheap role)...")
    run_annotation_pipeline(
        episodes=RESERVED_LIST,
        role="cheap",
        max_workers=2,
        enable_cache=False,
        cache_key_extra=run_label,
        run_id=run_label,
        annotated_dir=ann_dir,
        report_file=f"data/stability/{run_label}/annotate_report.json",
    )

    print(f"\n[Stability] Running {run_label}: Stage 3a/4 Window Search & Gates...")
    for ep in RESERVED_LIST:
        process_episode_candidates(
            ep_id=ep,
            annotated_dir=ann_dir,
            candidates_dir=cand_dir,
        )

    print(f"\n[Stability] Running {run_label}: Stage 5 Judge v2 (judge role)...")
    run_judge_pipeline(
        episodes=RESERVED_LIST,
        role="judge",
        max_workers=2,
        enable_cache=False,
        cache_key_extra=run_label,
        run_id=run_label,
        mode="per_episode",
        prompt_version="judge_v2",
        allow_variants=False,
        annotated_dir=ann_dir,
        candidates_dir=cand_dir,
        judged_dir=judged_dir,
        report_file=f"data/stability/{run_label}/judge_report.json",
    )
    print(f"[Stability] Finished {run_label}.")


def load_judged_windows_for_run(judged_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    res = {}
    for ep in RESERVED_LIST:
        f = judged_dir / f"{ep}.json"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            res[ep] = d.get("judged", [])
        else:
            res[ep] = []
    return res


def load_annotations_for_run(ann_dir: Path) -> Dict[str, Dict[str, Any]]:
    res = {}
    for ep in RESERVED_LIST:
        f = ann_dir / f"{ep}.json"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            res[ep] = d.get("annotations", {})
        else:
            res[ep] = {}
    return res


def compute_stability_metrics() -> Dict[str, Any]:
    # 1. Load judged outputs for primary, run2, run3
    primary_judged = load_judged_windows_for_run(Path("data/judged"))
    run2_judged = load_judged_windows_for_run(Path("data/stability/run2/judged"))
    run3_judged = load_judged_windows_for_run(Path("data/stability/run3/judged"))

    runs = {
        "primary": primary_judged,
        "run2": run2_judged,
        "run3": run3_judged,
    }

    pairs = [
        ("primary", "run2"),
        ("primary", "run3"),
        ("run2", "run3"),
    ]

    # Top-1 Agreement (IoU >= 0.5)
    top1_agreement = {}
    top3_overlap = {}

    for (rA, rB) in pairs:
        agreed_top1 = 0
        overlap_ratios = []

        for ep in RESERVED_LIST:
            wA = runs[rA].get(ep, [])
            wB = runs[rB].get(ep, [])

            # Top 1
            if wA and wB:
                t1A = wA[0]
                t1B = wB[0]
                iou1 = compute_interval_iou(t1A["start_s"], t1A["end_s"], t1B["start_s"], t1B["end_s"])
                if iou1 >= 0.5:
                    agreed_top1 += 1

            # Top 3 set overlap
            top3A = wA[:3]
            top3B = wB[:3]
            matched_count = 0
            for itemA in top3A:
                for itemB in top3B:
                    if compute_interval_iou(itemA["start_s"], itemA["end_s"], itemB["start_s"], itemB["end_s"]) >= 0.5:
                        matched_count += 1
                        break
            denom = max(1, min(len(top3A), len(top3B))) if (top3A or top3B) else 1
            overlap_ratios.append(matched_count / denom)

        top1_agreement[f"{rA}_vs_{rB}"] = f"{agreed_top1} of {len(RESERVED_LIST)}"
        top3_overlap[f"{rA}_vs_{rB}"] = round(statistics.mean(overlap_ratios), 3)

    # 2. S2 per-field exact agreement across runs
    primary_ann = load_annotations_for_run(Path("data/annotated"))
    run2_ann = load_annotations_for_run(Path("data/stability/run2/annotated"))
    run3_ann = load_annotations_for_run(Path("data/stability/run3/annotated"))

    ann_runs = {
        "primary": primary_ann,
        "run2": run2_ann,
        "run3": run3_ann,
    }

    s2_agreement = {}
    for (rA, rB) in pairs:
        field_matches = {"h": 0, "q": 0, "r": 0, "n": 0, "t": 0}
        total_beats = 0

        for ep in RESERVED_LIST:
            beatsA = ann_runs[rA].get(ep, {})
            beatsB = ann_runs[rB].get(ep, {})
            common_beats = set(beatsA.keys()) & set(beatsB.keys())
            total_beats += len(common_beats)

            for b in common_beats:
                for fld in ("h", "q", "r", "n", "t"):
                    if beatsA[b].get(fld) == beatsB[b].get(fld):
                        field_matches[fld] += 1

        s2_agreement[f"{rA}_vs_{rB}"] = {
            fld: f"{field_matches[fld]} of {total_beats}"
            for fld in ("h", "q", "r", "n", "t")
        }
        s2_agreement[f"{rA}_vs_{rB}"]["total_evaluated_beats"] = total_beats

    # 3. Gold hits per run (IoU >= 0.5)
    with open("gold/gold.csv", "r", encoding="utf-8") as gf:
        gold_windows = list(csv.DictReader(gf))

    gold_hits = {}
    for r_name, r_judged in runs.items():
        recalled_gold = set()
        for g in gold_windows:
            ep_str = f"episode-{int(g['episode']):03d}"
            cands = r_judged.get(ep_str, [])[:3]
            g_st = float(g["start_s"])
            g_en = float(g["end_s"])
            for c in cands:
                if compute_interval_iou(c["start_s"], c["end_s"], g_st, g_en) >= 0.5:
                    recalled_gold.add(g["gold_id"])
                    break
        gold_hits[r_name] = f"{len(recalled_gold)} of {len(gold_windows)}"

    return {
        "top1_agreement": top1_agreement,
        "top3_mean_overlap": top3_overlap,
        "s2_field_agreement": s2_agreement,
        "gold_hits": gold_hits,
    }


def main() -> None:
    # Run iteration 2
    run_pipeline_iteration("run2")

    # Run iteration 3
    run_pipeline_iteration("run3")

    # Compute metrics
    metrics = compute_stability_metrics()

    print("\n" + "=" * 80)
    print("STABILITY REPORT (3 Pipeline Runs: primary, run2, run3)")
    print("=" * 80)
    print("\n1. Top-1 Window Agreement (IoU >= 0.5):")
    for pair, count_str in metrics["top1_agreement"].items():
        print(f"   {pair:<18}: {count_str}")

    print("\n2. Mean Top-3 Set Overlap (IoU >= 0.5):")
    for pair, ov in metrics["top3_mean_overlap"].items():
        print(f"   {pair:<18}: {ov}")

    print("\n3. S2 Per-Field Exact Agreement Across Runs:")
    for pair, fld_data in metrics["s2_field_agreement"].items():
        print(f"   [{pair}] (total beats: {fld_data['total_evaluated_beats']})")
        print(f"     h (hook)     : {fld_data['h']}")
        print(f"     q (cliff)    : {fld_data['q']}")
        print(f"     r (resolved) : {fld_data['r']}")
        print(f"     n (context)  : {fld_data['n']}")
        print(f"     t (intensity): {fld_data['t']}")

    print("\n4. Gold Hits Per Run (n of 7 gold windows recalled, IoU >= 0.5):")
    for r_name, hits in metrics["gold_hits"].items():
        print(f"   {r_name:<10}: {hits}")

    out_file = Path("data/stability_report.json")
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote stability results to {out_file}")


if __name__ == "__main__":
    main()
