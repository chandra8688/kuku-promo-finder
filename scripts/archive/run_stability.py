"""Stability Evaluation across 3 Pipeline Runs on 10 Reserved Episodes (scripts/run_stability.py).

POST-FREEZE robustness update:
- Capped retries on 503/UNAVAILABLE at 3 attempts with <= 30s backoff.
- Per-role fallback chain (cheap -> judge model).
- Resumable: skips items already written to disk.
- Runs on 5 gold episodes first, then other 5 reserved episodes.
- Asserts zero cached calls across runs.
- Reports raw counts (n of N) over completed episodes only, never filling in missing episodes.
- Never prints screenplay text, loglines, or character names.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.consistency import aggregate_annotations, annotation_agreement, run_stability
from src.judge import run_judge_pipeline
from src.windows import compute_interval_iou

GOLD_EP_STRS = ["episode-003", "episode-013", "episode-023", "episode-033", "episode-043"]
OTHER_EP_STRS = ["episode-008", "episode-018", "episode-028", "episode-038", "episode-048"]
ORDERED_EPISODES = GOLD_EP_STRS + OTHER_EP_STRS


def run_pipeline_iteration(run_label: str) -> None:
    """Run S2 -> S3a/S4 -> Judge for a stability iteration in a resumable manner."""
    ann_dir = Path(f"data/stability/{run_label}/annotated")
    cand_dir = Path(f"data/stability/{run_label}/candidates")
    judged_dir = Path(f"data/stability/{run_label}/judged")

    ann_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    judged_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage 2 Annotation
    episodes_to_annotate = [ep for ep in ORDERED_EPISODES if not (ann_dir / f"{ep}.json").exists()]
    if episodes_to_annotate:
        print(f"\n[Stability] Running {run_label}: Stage 2 Annotation on {len(episodes_to_annotate)} episode(s)...")
        gold_chunk = [ep for ep in episodes_to_annotate if ep in GOLD_EP_STRS]
        other_chunk = [ep for ep in episodes_to_annotate if ep not in GOLD_EP_STRS]

        for chunk, label in [(gold_chunk, "gold"), (other_chunk, "other reserved")]:
            if not chunk:
                continue
            print(f"[Stability {run_label}] Annotating {label} chunk: {chunk}")
            run_annotation_pipeline(
                episodes=chunk,
                role="cheap",
                max_workers=2,
                enable_cache=False,
                cache_key_extra=run_label,
                run_id=run_label,
                annotated_dir=ann_dir,
                candidates_dir=cand_dir,
                report_file=f"data/stability/{run_label}/annotate_report.json",
                force=True,
            )
    else:
        print(f"\n[Stability] {run_label}: Stage 2 Annotation already complete (resuming).")

    # 2. Stage 3a/4 Window Search & Gates
    print(f"\n[Stability] Running {run_label}: Stage 3a/4 Window Search & Gates...")
    for ep in ORDERED_EPISODES:
        ann_file = ann_dir / f"{ep}.json"
        if ann_file.exists():
            process_episode_candidates(
                ep_id=ep,
                annotated_dir=ann_dir,
                candidates_dir=cand_dir,
            )

    # 3. Stage 5 Judge v2
    episodes_to_judge = [
        ep for ep in ORDERED_EPISODES
        if (cand_dir / f"{ep}.json").exists() and not (judged_dir / f"{ep}.json").exists()
    ]
    if episodes_to_judge:
        print(f"\n[Stability] Running {run_label}: Stage 5 Judge on {len(episodes_to_judge)} episode(s)...")
        gold_judge = [ep for ep in episodes_to_judge if ep in GOLD_EP_STRS]
        other_judge = [ep for ep in episodes_to_judge if ep not in GOLD_EP_STRS]

        for chunk, label in [(gold_judge, "gold"), (other_judge, "other reserved")]:
            if not chunk:
                continue
            print(f"[Stability {run_label}] Judging {label} chunk: {chunk}")
            run_judge_pipeline(
                episodes=chunk,
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
                force=True,
            )
    else:
        print(f"\n[Stability] {run_label}: Stage 5 Judge already complete (resuming).")

    # 4. Assert zero cached calls
    calls_file = Path(f"runs/{run_label}/calls.jsonl")
    if calls_file.exists():
        calls = [json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        cached_count = sum(1 for c in calls if c.get("cached"))
        assert cached_count == 0, f"Integrity error: {cached_count} calls were cached in {run_label}"
        print(f"[Stability {run_label}] Verified {len(calls)} calls: exactly 0 cached calls.")


def load_judged_windows_for_run(judged_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    res = {}
    for ep in ORDERED_EPISODES:
        f = judged_dir / f"{ep}.json"
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                res[ep] = d.get("judged", [])
            except Exception:
                res[ep] = []
        else:
            res[ep] = []
    return res


def load_annotations_for_run(ann_dir: Path) -> Dict[str, Dict[str, Any]]:
    res = {}
    for ep in ORDERED_EPISODES:
        f = ann_dir / f"{ep}.json"
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                res[ep] = d.get("annotations", {})
            except Exception:
                res[ep] = {}
        else:
            res[ep] = {}
    return res


def is_genuine_model_ep(ann_path: Path, ep: str) -> bool:
    """Check if episode completed annotation using genuine model without fallback."""
    f = ann_path / f"{ep}.json"
    if not f.exists():
        return False
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("status") == "failed" or d.get("downgraded_to_tier0") or d.get("is_tier0") or d.get("tier") == "tier0_fallback":
            return False
        return bool(d.get("annotations"))
    except Exception:
        return False


def compute_stability_metrics() -> Dict[str, Any]:
    primary_judged = load_judged_windows_for_run(Path("data/judged"))
    run2_judged = load_judged_windows_for_run(Path("data/stability/run2/judged"))
    run3_judged = load_judged_windows_for_run(Path("data/stability/run3/judged"))

    ann_dirs = {
        "primary": Path("data/annotated"),
        "run2": Path("data/stability/run2/annotated"),
        "run3": Path("data/stability/run3/annotated"),
    }

    runs = {
        "primary": primary_judged,
        "run2": run2_judged,
        "run3": run3_judged,
    }

    # Strict check: count genuine model runs (excluding silent fallbacks/failures)
    completed_counts = {
        r_name: sum(1 for ep in ORDERED_EPISODES if is_genuine_model_ep(ann_dirs[r_name], ep) and len(r_judged.get(ep, [])) > 0)
        for r_name, r_judged in runs.items()
    }
    completed_ratios = {
        r_name: f"{count} of {len(ORDERED_EPISODES)}"
        for r_name, count in completed_counts.items()
    }

    pairs = [
        ("primary", "run2"),
        ("primary", "run3"),
        ("run2", "run3"),
    ]

    top1_agreement = {}
    top3_overlap = {}

    for (rA, rB) in pairs:
        # Comparable episodes ONLY: genuine frozen role model execution in BOTH runs
        common_eps = [
            ep for ep in ORDERED_EPISODES
            if is_genuine_model_ep(ann_dirs[rA], ep) and is_genuine_model_ep(ann_dirs[rB], ep)
            and len(runs[rA].get(ep, [])) > 0 and len(runs[rB].get(ep, [])) > 0
        ]
        if not common_eps:
            top1_agreement[f"{rA}_vs_{rB}"] = f"0 of 0 (no comparable episodes)"
            top3_overlap[f"{rA}_vs_{rB}"] = "n/a"
            continue

        agreed_top1 = 0
        overlap_ratios = []

        for ep in common_eps:
            wA = runs[rA][ep]
            wB = runs[rB][ep]

            # Top 1 IoU >= 0.5
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

        top1_agreement[f"{rA}_vs_{rB}"] = f"{agreed_top1} of {len(common_eps)} comparable episodes ({agreed_top1/len(common_eps)*100:.1f}%)"
        top3_overlap[f"{rA}_vs_{rB}"] = f"{round(statistics.mean(overlap_ratios), 3)} (over {len(common_eps)} comparable episodes)"

    # S2 per-field exact agreement using src.consistency.annotation_agreement
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
        common_eps = [
            ep for ep in ORDERED_EPISODES
            if is_genuine_model_ep(ann_dirs[rA], ep) and is_genuine_model_ep(ann_dirs[rB], ep)
        ]

        # Combine beat maps across comparable episodes
        mapA = {}
        mapB = {}
        for ep in common_eps:
            mapA.update({f"{ep}_{k}": v for k, v in ann_runs[rA].get(ep, {}).items()})
            mapB.update({f"{ep}_{k}": v for k, v in ann_runs[rB].get(ep, {}).items()})

        agr = annotation_agreement([mapA, mapB])
        s2_agreement[f"{rA}_vs_{rB}"] = {
            "comparable_episodes": len(common_eps),
            "total_evaluated_beats": agr.get("comparable_beats", 0),
            "h": f"{agr.get('exact_agreement', {}).get('h', 0.0)*100:.1f}%",
            "q": f"{agr.get('exact_agreement', {}).get('q', 0.0)*100:.1f}%",
            "r": f"{agr.get('r_agreement', 0.0)*100:.1f}%",
            "n": f"{agr.get('exact_agreement', {}).get('n', 0.0)*100:.1f}%",
            "t": f"{agr.get('exact_agreement', {}).get('t', 0.0)*100:.1f}%",
        }

    # Gold hits
    gold_file = Path("gold/gold.csv")
    gold_hits = {}
    if gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as gf:
            gold_windows = list(csv.DictReader(gf))

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
        "episodes_completed": completed_ratios,
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
    print("\nEpisodes Completed Per Run:")
    for r_name, count_str in metrics["episodes_completed"].items():
        print(f"   {r_name:<10}: {count_str}")

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
    for r_name, hits in metrics.get("gold_hits", {}).items():
        print(f"   {r_name:<10}: {hits}")

    out_file = Path("data/stability_report.json")
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote stability results to {out_file}")


if __name__ == "__main__":
    main()
