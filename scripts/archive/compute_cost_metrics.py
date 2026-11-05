"""Cost and Token Consumption Analysis (scripts/compute_cost_metrics.py).

Calculates from non-cached calls only:
- Tokens and latency per episode per stage (S2 annotation, S5 judge)
- Tokens and latency per 50-episode show
- Formula: N shows per month x tokens per show (N as parameter)
- Strictly no dollar prices or proprietary volume figures.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    calls = []
    for f in sorted(glob.glob("runs/*/calls.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    calls.append(json.loads(line))
                except Exception:
                    pass

    # Filter to non-cached calls from primary pipeline stages
    s2_calls = [
        c for c in calls
        if not c.get("cached", False) and c.get("stage") in ("s2_annotate", "s2_annotate_repair")
    ]
    s5_calls = [
        c for c in calls
        if not c.get("cached", False) and c.get("stage") in ("s5_judge_episode_batch", "s5_judge")
    ]

    # Metrics per episode for S2 (from 50 non-cached S2 primary calls)
    s2_tin = sum(c.get("tokens_in", 0) for c in s2_calls[:50])
    s2_tout = sum(c.get("tokens_out", 0) for c in s2_calls[:50])
    s2_lat = sum(c.get("latency_s", 0.0) for c in s2_calls[:50])
    s2_n = len(s2_calls[:50]) or 1

    s2_per_ep_tin = round(s2_tin / s2_n)
    s2_per_ep_tout = round(s2_tout / s2_n)
    s2_per_ep_tot = s2_per_ep_tin + s2_per_ep_tout
    s2_per_ep_lat = round(s2_lat / s2_n, 2)

    # Metrics per episode for S5 Judge (from 50 non-cached primary judge calls)
    s5_tin = sum(c.get("tokens_in", 0) for c in s5_calls[:50])
    s5_tout = sum(c.get("tokens_out", 0) for c in s5_calls[:50])
    s5_lat = sum(c.get("latency_s", 0.0) for c in s5_calls[:50])
    s5_n = len(s5_calls[:50]) or 1

    s5_per_ep_tin = round(s5_tin / s5_n)
    s5_per_ep_tout = round(s5_tout / s5_n)
    s5_per_ep_tot = s5_per_ep_tin + s5_per_ep_tout
    s5_per_ep_lat = round(s5_lat / s5_n, 2)

    # Full 50-episode show (S2 + S5 + listwise 2 calls)
    listwise_calls = [
        c for c in calls
        if not c.get("cached", False) and c.get("stage") in ("listwise_order1", "listwise_order2")
    ]
    lw_tin = sum(c.get("tokens_in", 0) for c in listwise_calls)
    lw_tout = sum(c.get("tokens_out", 0) for c in listwise_calls)
    lw_lat = sum(c.get("latency_s", 0.0) for c in listwise_calls)

    show_tin = s2_per_ep_tin * 50 + s5_per_ep_tin * 50 + lw_tin
    show_tout = s2_per_ep_tout * 50 + s5_per_ep_tout * 50 + lw_tout
    show_tot_tokens = show_tin + show_tout
    show_lat = s2_per_ep_lat * 50 + s5_per_ep_lat * 50 + lw_lat

    payload = {
        "per_episode": {
            "s2_annotate": {
                "tokens_in": s2_per_ep_tin,
                "tokens_out": s2_per_ep_tout,
                "tokens_total": s2_per_ep_tot,
                "latency_s": s2_per_ep_lat,
            },
            "s5_judge": {
                "tokens_in": s5_per_ep_tin,
                "tokens_out": s5_per_ep_tout,
                "tokens_total": s5_per_ep_tot,
                "latency_s": s5_per_ep_lat,
            },
            "combined_per_episode": {
                "tokens_in": s2_per_ep_tin + s5_per_ep_tin,
                "tokens_out": s2_per_ep_tout + s5_per_ep_tout,
                "tokens_total": s2_per_ep_tot + s5_per_ep_tot,
                "latency_s": round(s2_per_ep_lat + s5_per_ep_lat, 2),
            },
        },
        "per_50_episode_show": {
            "s2_tokens": s2_per_ep_tot * 50,
            "s5_tokens": s5_per_ep_tot * 50,
            "listwise_tokens": lw_tin + lw_tout,
            "total_tokens_in": show_tin,
            "total_tokens_out": show_tout,
            "total_tokens": show_tot_tokens,
            "total_latency_s": round(show_lat, 2),
            "total_latency_min": round(show_lat / 60, 2),
        },
        "scaling_formula": f"Tokens_per_month = N * {show_tot_tokens:,} tokens (where N = number of 50-episode shows processed per month)",
    }

    print("\n" + "=" * 80)
    print("EMPIRICAL COST & TOKEN CONSUMPTION REPORT (Non-Cached Calls)")
    print("=" * 80)
    print("1. Per-Episode Telemetry:")
    print(f"   Stage 2 (Annotate): In={s2_per_ep_tin:,}, Out={s2_per_ep_tout:,}, Total={s2_per_ep_tot:,} tokens | Latency: {s2_per_ep_lat:.2f}s")
    print(f"   Stage 5 (Judge)   : In={s5_per_ep_tin:,}, Out={s5_per_ep_tout:,}, Total={s5_per_ep_tot:,} tokens | Latency: {s5_per_ep_lat:.2f}s")
    print(f"   Episode Total     : In={s2_per_ep_tin+s5_per_ep_tin:,}, Out={s2_per_ep_tout+s5_per_ep_tout:,}, Total={s2_per_ep_tot+s5_per_ep_tot:,} tokens | Latency: {s2_per_ep_lat+s5_per_ep_lat:.2f}s")

    print("\n2. Per 50-Episode Show (Full Pipeline):")
    print(f"   Total Input Tokens : {show_tin:,}")
    print(f"   Total Output Tokens: {show_tout:,}")
    print(f"   Total Tokens/Show  : {show_tot_tokens:,}")
    print(f"   Total Latency/Show : {show_lat:.2f}s (~{show_lat/60:.1f} minutes)")

    print("\n3. Production Scaling Formula:")
    print(f"   Formula: N shows/month x {show_tot_tokens:,} tokens/show")
    print("   (N is the parameter representing number of 50-episode shows per month)")

    out_file = Path("data/cost_metrics.json")
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

