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
from src.score import DEFAULT_WEIGHTS, rank_key
from src.search import search_episode

RESERVED_DEFAULT = ["episode-003", "episode-013", "episode-023", "episode-033", "episode-043"]
GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8, 1.0, 1.4, 2.0, 2.5, 3.0]

# Weight bounds come from the brief, not from the data.
#
# Signs: a promo must end unresolved and must not slice a beat in half, so those
# terms can only ever subtract. Unconstrained coordinate ascent on 25 dev windows
# happily flips them and ships a scorer that pays a bonus for resolving the
# cliffhanger, which is the opposite of criterion 3.
#
# Floors: "hook in the first 3 seconds" and "unresolved cliffhanger" are two of
# the four stated criteria, so neither may be driven to zero by a fit over a few
# dozen windows. A term at 0.0 has been deleted from the model, and deleting a
# criterion is a decision for the brief to make, not for coordinate ascent.
#
# (lo, hi); None means unbounded on that side.
WEIGHT_BOUNDS = {
    "hook": (1.0, None),
    "cliff": (1.0, None),
    "intensity": (0.0, None),
    "peak_intensity": (0.0, None),
    "intensity_slope": (0.0, None),
    "tail_pos": (0.0, None),
    "hook_latency": (None, 0.0),
    "context": (None, 0.0),
    "resolved_tail": (None, 0.0),
    "resolved_last": (None, 0.0),
    "straddle_spoken": (None, 0.0),
    "straddle_action": (None, 0.0),
    "speaker_overload": (None, 0.0),
    "silence": (None, 0.0),
}


def allowed_grid(key: str, grid=GRID):
    lo, hi = WEIGHT_BOUNDS.get(key, (None, None))
    return [
        v for v in grid
        if (lo is None or v >= lo) and (hi is None or v <= hi)
    ]


def bound_violations(weights: dict) -> list:
    bad = []
    for k, v in sorted(weights.items()):
        lo, hi = WEIGHT_BOUNDS.get(k, (None, None))
        if lo is not None and v < lo:
            bad.append(f"{k}={v} must be >= {lo}")
        if hi is not None and v > hi:
            bad.append(f"{k}={v} must be <= {hi}")
    return bad


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
    print(f"tuning on {len(eps)} dev episodes, {sum(len(v) for v in gold.values())} windows", flush=True)

    # Features do not depend on weights, so enumerate once and only rescore per
    # trial. Selection, dedupe and refinement go through src/engine.py, the same
    # code run.py executes; the earlier inline copy skipped the shared-anchor
    # dedupe rule, so the old fit optimised a procedure that never shipped.
    from src import engine

    windows = {ep: engine.enumerate(parsed[ep], ann[ep]) for ep in eps}

    def evaluate(w):
        preds = {}
        for ep in eps:
            _, surv, _ = engine.select(engine.rescore(windows[ep], w), parsed[ep], ann[ep], weights=w, top_k=a.k)
            preds[ep] = sorted(surv, key=rank_key)[: a.k]
        return system_metrics(preds, gold, a.iou, (1, 3))

    best = dict(DEFAULT_WEIGHTS)
    pre = bound_violations(best)
    if pre:
        print("REFUSING: DEFAULT_WEIGHTS violate the weight bounds:", file=sys.stderr)
        for b in pre:
            print(f"  {b}", file=sys.stderr)
        sys.exit(3)
    best_m = evaluate(best)
    best_obj = objective(best_m, a.w_top1, a.w_recall, a.w_err)
    print(f"start obj={best_obj:.4f} top1={best_m['top1']}/{len(eps)} recall={best_m['gold_recall']:.2f}", flush=True)

    for r in range(a.rounds):
        for key in sorted(best):
            cur = best[key]
            local_best, local_obj = cur, best_obj
            for v in allowed_grid(key):
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
                print(f"  round {r+1}: {key} {cur} -> {local_best}  obj={best_obj:.4f}", flush=True)

    bad = bound_violations(best)
    if bad:
        print("REFUSING to write: fitted weights violate the weight bounds:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        sys.exit(3)

    # Final check with standard search_episode
    full_preds = {e: search_episode(parsed[e], ann[e], weights=best, top_k=a.k) for e in eps}
    final = system_metrics(full_preds, gold, a.iou, (1, 3))
    print(
        f"final obj={best_obj:.4f} top1={final['top1']}/{len(eps)} recall={final['gold_recall']:.2f} "
        f"med_end_err={final['median_end_err']}",
        flush=True,
    )

    import yaml

    existing = {}
    if Path(a.out).exists():
        try:
            existing = yaml.safe_load(Path(a.out).read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    existing["weights"] = {k: float(v) for k, v in sorted(best.items())}

    lines = [
        "# Fitted on DEV episodes only. Reserved/gold episodes were not used.",
        "# Bounded fit: penalty terms cannot go positive, and hook and cliff cannot be",
        "# driven below 1.0. Both constraints are priors from the brief's four criteria,",
        "# not tuned choices. See WEIGHT_BOUNDS in scripts/tune_weights.py.",
        f"# dev episodes: {', '.join(eps)}",
        f"# dev top1: {final['top1']}/{len(eps)}  dev recall: {final['gold_recall']:.3f}",
        "",
    ]

    for sec in ["features", "weights", "soft_gates", "hard_gates", "refine", "dedupe", "judge"]:
        if sec in existing and isinstance(existing[sec], dict):
            lines.append(f"{sec}:")
            for k, v in existing[sec].items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        lines.append(f"    {sub_k}: {sub_v}")
                elif isinstance(v, bool):
                    lines.append(f"  {k}: {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v}")
                elif v is None:
                    lines.append(f"  {k}: null")
                else:
                    lines.append(f'  {k}: "{v}"')
            lines.append("")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
