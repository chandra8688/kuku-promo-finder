"""Coordinate-ascent weight fit on DEV episodes only.

Refuses to run if any dev episode is in the reserved/gold list. Tuning on the
gold episodes and then reporting gold numbers is the one mistake that would
invalidate the whole evaluation, so the protection check is hard-coded rather than optional.

    python scripts/tune_weights.py --parsed data/parsed --annotated data/annotated \
        --gold gold/gold_dev.csv --out config/scoring_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalkit import system_metrics
from src.score import DEFAULT_WEIGHTS
from src.search import search_episode

RESERVED_DEFAULT = ["episode-003", "episode-013", "episode-023", "episode-033", "episode-043"]
GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8, 1.0, 1.4, 2.0, 2.5, 3.0]


def norm_ep(v) -> str:
    s = str(v).strip()
    return f"episode-{int(s):03d}" if s.isdigit() else s.lower()


def load_gold(path: str) -> dict:
    import csv

    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(norm_ep(row["episode"]), []).append(
                {"gold_id": row.get("gold_id"), "start_s": float(row["start_s"]), "end_s": float(row["end_s"])}
            )
    return out


def objective(m: dict, w_top1: float, w_recall: float, w_err: float) -> float:
    err = m["median_end_err"] if m["median_end_err"] is not None else 30.0
    return w_top1 * m["top1_rate"] + w_recall * m["gold_recall"] - w_err * (err / 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--gold", required=True, help="DEV gold csv, not the reserved gold")
    ap.add_argument("--reserved", nargs="*", default=RESERVED_DEFAULT)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--w-top1", type=float, default=1.0)
    ap.add_argument("--w-recall", type=float, default=0.5)
    ap.add_argument("--w-err", type=float, default=0.05)
    ap.add_argument("--out", default="config/scoring_v2.yaml")
    a = ap.parse_args()

    gold = load_gold(a.gold)
    reserved = {norm_ep(x) for x in a.reserved}
    leaked = sorted(set(gold) & reserved)
    if leaked:
        print(f"REFUSING: dev gold contains reserved episodes {leaked}.", file=sys.stderr)
        print("Tune on held-out dev episodes only. Reserved episodes stay untouched.", file=sys.stderr)
        sys.exit(2)

    eps, parsed, ann = [], {}, {}
    for ep in sorted(gold):
        pf, af = Path(a.parsed) / f"{ep}.json", Path(a.annotated) / f"{ep}.json"
        if not pf.exists():
            continue
        parsed[ep] = json.loads(pf.read_text(encoding="utf-8"))
        ann[ep] = json.loads(af.read_text(encoding="utf-8")) if af.exists() else {"annotations": {}}
        eps.append(ep)
    if not eps:
        print("no usable dev episodes", file=sys.stderr)
        sys.exit(1)
    gold = {e: gold[e] for e in eps}
    print(f"tuning on {len(eps)} dev episodes, {sum(len(v) for v in gold.values())} windows")

    def evaluate(w):
        preds = {e: search_episode(parsed[e], ann[e], weights=w, top_k=a.k) for e in eps}
        return system_metrics(preds, gold, a.iou, (1, 3))

    best = dict(DEFAULT_WEIGHTS)
    best_m = evaluate(best)
    best_obj = objective(best_m, a.w_top1, a.w_recall, a.w_err)
    print(f"start obj={best_obj:.4f} top1={best_m['top1']}/{len(eps)} recall={best_m['gold_recall']:.2f}")

    for r in range(a.rounds):
        for key in sorted(best):
            cur = best[key]
            local_best, local_obj = cur, best_obj
            for v in GRID:
                if v == cur:
                    continue
                trial = dict(best)
                trial[key] = v
                obj = objective(evaluate(trial), a.w_top1, a.w_recall, a.w_err)
                if obj > local_obj + 1e-9:
                    local_best, local_obj = v, obj
            if local_best != cur:
                best[key] = local_best
                best_obj = local_obj
                print(f"  round {r+1}: {key} {cur} -> {local_best}  obj={best_obj:.4f}")
    final = evaluate(best)
    print(f"final obj={best_obj:.4f} top1={final['top1']}/{len(eps)} recall={final['gold_recall']:.2f} "
          f"med_end_err={final['median_end_err']}")

    lines = [
        "# Fitted on DEV episodes only. Reserved/gold episodes were not used.",
        f"# dev episodes: {', '.join(eps)}",
        f"# dev top1: {final['top1']}/{len(eps)}  dev recall: {final['gold_recall']:.3f}",
        "weights:",
    ]
    lines += [f"  {k}: {v}" for k, v in sorted(best.items())]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
