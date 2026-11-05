"""Model Swap Evaluation on 10 Reserved Episodes (scripts/run_model_swap.py).

Reruns S2 annotation using judge-role model (gemini-3.1-flash-lite) as the annotator,
cache_key_extra "swap_s2", then runs S3a/S4 and Judge v2 ranking.

Reports:
- S2 per-field exact agreement vs primary run
- Top-1 window agreement vs primary run (IoU >= 0.5; n of 10)
- Gold hits vs primary (n of 7 gold windows recalled, IoU >= 0.5)
- Notes that both models are from the same Gemini Flash-Lite family.

Never prints screenplay text, loglines, or character names.
Reports raw counts (n of N), never percentages.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.annotate import run_annotation_pipeline
from src.candidates import process_episode_candidates
from src.judge import run_judge_pipeline
from src.reserved import RESERVED_EPISODES
from src.windows import compute_interval_iou

RESERVED_LIST = sorted(list(RESERVED_EPISODES))


def main() -> None:
    ann_dir = Path("data/model_swap/annotated")
    cand_dir = Path("data/model_swap/candidates")
    judged_dir = Path("data/model_swap/judged")

    ann_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    judged_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Model Swap] Stage 2: Annotation with judge-role model (gemini-3.1-flash-lite)...")
    from src.llm import get_gateway
    gw = get_gateway()
    orig_judge_max_tokens = gw.roles_config.get("judge", {}).get("max_output_tokens", 2048)
    # In-memory token expansion for annotation (4096 tokens needed for 50+ beat JSON)
    gw.roles_config["judge"]["max_output_tokens"] = 4096

    try:
        run_annotation_pipeline(
            episodes=RESERVED_LIST,
            role="judge",  # swap annotator to gemini-3.1-flash-lite
            max_workers=2,
            enable_cache=True,
            run_id="swap_s2",
            annotated_dir=ann_dir,
            report_file="data/model_swap/annotate_report.json",
        )
    finally:
        gw.roles_config["judge"]["max_output_tokens"] = orig_judge_max_tokens

    print("\n[Model Swap] Stage 3a/4: Window Search & Gates...")
    for ep in RESERVED_LIST:
        process_episode_candidates(
            ep_id=ep,
            annotated_dir=ann_dir,
            candidates_dir=cand_dir,
        )

    print("\n[Model Swap] Stage 5: Judge v2 Ranking...")
    run_judge_pipeline(
        episodes=RESERVED_LIST,
        role="judge",
        max_workers=2,
        enable_cache=True,
        cache_key_extra="swap_s2",
        mode="per_episode",
        prompt_version="judge_v2",
        allow_variants=False,
        annotated_dir=ann_dir,
        candidates_dir=cand_dir,
        judged_dir=judged_dir,
        report_file="data/model_swap/judge_report.json",
    )

    # 1. Compare S2 annotations (primary vs swap)
    primary_ann = {}
    swap_ann = {}
    for ep in RESERVED_LIST:
        with open(f"data/annotated/{ep}.json", "r", encoding="utf-8") as f:
            primary_ann[ep] = json.load(f).get("annotations", {})
        with open(ann_dir / f"{ep}.json", "r", encoding="utf-8") as f:
            swap_ann[ep] = json.load(f).get("annotations", {})

    field_matches = {"h": 0, "q": 0, "r": 0, "n": 0, "t": 0}
    total_beats = 0

    for ep in RESERVED_LIST:
        p_beats = primary_ann.get(ep, {})
        s_beats = swap_ann.get(ep, {})
        common = set(p_beats.keys()) & set(s_beats.keys())
        total_beats += len(common)

        for b in common:
            for fld in ("h", "q", "r", "n", "t"):
                if p_beats[b].get(fld) == s_beats[b].get(fld):
                    field_matches[fld] += 1

    s2_agreement = {
        fld: f"{field_matches[fld]} of {total_beats}"
        for fld in ("h", "q", "r", "n", "t")
    }

    # 2. Compare Top-1 window agreement (primary vs swap)
    primary_judged = {}
    swap_judged = {}
    for ep in RESERVED_LIST:
        with open(f"data/judged/{ep}.json", "r", encoding="utf-8") as f:
            primary_judged[ep] = json.load(f).get("judged", [])
        with open(judged_dir / f"{ep}.json", "r", encoding="utf-8") as f:
            swap_judged[ep] = json.load(f).get("judged", [])

    agreed_top1 = 0
    top1_pairs = []
    for ep in RESERVED_LIST:
        pw = primary_judged.get(ep, [])
        sw = swap_judged.get(ep, [])
        if pw and sw:
            iou = compute_interval_iou(pw[0]["start_s"], pw[0]["end_s"], sw[0]["start_s"], sw[0]["end_s"])
            match = iou >= 0.5
            if match:
                agreed_top1 += 1
            top1_pairs.append({
                "episode": ep,
                "primary": [pw[0]["start_s"], pw[0]["end_s"]],
                "swap": [sw[0]["start_s"], sw[0]["end_s"]],
                "iou": round(iou, 3),
                "matched": match,
            })

    # 3. Gold hits (primary vs swap)
    with open("gold/gold.csv", "r", encoding="utf-8") as gf:
        gold_windows = list(csv.DictReader(gf))

    primary_gold_recalled = set()
    swap_gold_recalled = set()

    for g in gold_windows:
        ep_str = f"episode-{int(g['episode']):03d}"
        g_st = float(g["start_s"])
        g_en = float(g["end_s"])

        for c in primary_judged.get(ep_str, [])[:3]:
            if compute_interval_iou(c["start_s"], c["end_s"], g_st, g_en) >= 0.5:
                primary_gold_recalled.add(g["gold_id"])
                break

        for c in swap_judged.get(ep_str, [])[:3]:
            if compute_interval_iou(c["start_s"], c["end_s"], g_st, g_en) >= 0.5:
                swap_gold_recalled.add(g["gold_id"])
                break

    report_payload = {
        "model_family_note": "Both models are from the same family (Google Gemini Flash-Lite series: gemini-3.5-flash-lite vs gemini-3.1-flash-lite).",
        "total_evaluated_beats": total_beats,
        "s2_field_agreement": s2_agreement,
        "top1_agreement": f"{agreed_top1} of {len(RESERVED_LIST)}",
        "top1_details": top1_pairs,
        "gold_hits": {
            "primary": f"{len(primary_gold_recalled)} of {len(gold_windows)}",
            "swap": f"{len(swap_gold_recalled)} of {len(gold_windows)}",
        },
    }

    print("\n" + "=" * 80)
    print("MODEL SWAP REPORT (gemini-3.1-flash-lite vs gemini-3.5-flash-lite)")
    print("=" * 80)
    print(f"Family Note           : {report_payload['model_family_note']}")
    print(f"\n1. S2 Per-Field Exact Agreement (total beats: {total_beats}):")
    for fld, score_str in s2_agreement.items():
        print(f"   {fld:<10}: {score_str}")
    print(f"\n2. Top-1 Window Agreement (IoU >= 0.5): {agreed_top1} of {len(RESERVED_LIST)}")
    print(f"3. Gold Hits Comparison (IoU >= 0.5):")
    print(f"   Primary Run (3.5-flash-lite): {report_payload['gold_hits']['primary']}")
    print(f"   Swap Run    (3.1-flash-lite): {report_payload['gold_hits']['swap']}")

    out_file = Path("data/model_swap_report.json")
    out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nWrote model swap report to {out_file}")


if __name__ == "__main__":
    main()

