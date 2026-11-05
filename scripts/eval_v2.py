"""Single evaluation entrypoint. Prints the headline table with bootstrap CIs
and a paired comparison against the baseline, then writes the same thing to JSON.

Offline smoke run on fixtures:

    python scripts/make_fixtures.py --out fixtures --episodes 12
    python scripts/eval_v2.py --parsed fixtures/parsed --annotated fixtures/annotated \
        --gold fixtures/gold_synthetic.csv --out eval_v2.json

Real run (gold episodes only, nothing tuned on them):

    python scripts/eval_v2.py --parsed data/parsed --annotated data/annotated \
        --gold gold/gold_private.csv --systems data/judged=judged --out eval_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalkit import bootstrap_ci, fmt_ci, markdown_table, paired_bootstrap, system_metrics
from src.search import baseline_last_n, baseline_random, search_episode

RANK_KEYS = {
    "judged": lambda c: (0 if c.get("passed_judge_gates", True) else 1, -float(c.get("s3a_score", c.get("score", 0.0)))),
    "score": lambda c: (0 if c.get("passed_judge_gates", True) else 1, -float(c.get("composite_score", c.get("score", 0.0)))),
    "judge_composite": lambda c: (0 if c.get("passed_judge_gates", True) else 1, -float(c.get("judge_composite", c.get("final_score", 0.0)))),
    "rank": lambda c: int(c.get("rank", 999)),
}


def norm_ep(v) -> str:
    s = str(v).strip()
    if s.isdigit():
        return f"episode-{int(s):03d}"
    return s.lower()


def load_gold(path: str) -> dict:
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = norm_ep(row["episode"])
            out.setdefault(ep, []).append(
                {
                    "gold_id": row.get("gold_id"),
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "status": row.get("status", ""),
                }
            )
    return out


def load_dir(path: str, episodes, rank: str = "score", k: int = 5) -> dict:
    p = Path(path)
    key = RANK_KEYS.get(rank, RANK_KEYS["score"])
    out = {}
    for ep in episodes:
        f = p / f"{ep}.json"
        if not f.exists():
            out[ep] = []
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = None
        if isinstance(d, list):
            rows = d
        else:
            for name in ("judged", "candidates", "survivors", "tier0_candidates", "promos"):
                if isinstance(d.get(name), list):
                    rows = d[name]
                    break
        rows = rows or []
        if rows and "passed_all_gates" in rows[0]:
            rows = [r for r in rows if r.get("passed_all_gates")] or rows
        out[ep] = sorted(rows, key=key)[:k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--annotated", default=None, help="omit to evaluate baselines only")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--systems", nargs="*", default=[], help="NAME=dir or dir=rankkey entries, e.g. data/judged=judged")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--baseline-name", default="B1 last 45s")
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--out", default="eval_v2.json")
    a = ap.parse_args()

    gold = load_gold(a.gold)
    episodes = sorted(gold)
    parsed = {}
    for ep in episodes:
        f = Path(a.parsed) / f"{ep}.json"
        if f.exists():
            parsed[ep] = json.loads(f.read_text(encoding="utf-8"))
    missing = [e for e in episodes if e not in parsed]
    if missing:
        print(f"warning: no parsed file for {missing}", file=sys.stderr)
    episodes = [e for e in episodes if e in parsed]
    gold = {e: gold[e] for e in episodes}

    systems: dict = {}
    if a.annotated:
        v2 = {}
        for ep in episodes:
            f = Path(a.annotated) / f"{ep}.json"
            ann = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"annotations": {}}
            v2[ep] = search_episode(parsed[ep], ann, top_k=a.k)
        systems["V2 features+refine"] = v2
        systems["V2 no refine"] = {
            ep: search_episode(parsed[ep], json.loads((Path(a.annotated) / f"{ep}.json").read_text(encoding="utf-8")) if (Path(a.annotated) / f"{ep}.json").exists() else {"annotations": {}}, top_k=a.k, refine=False)
            for ep in episodes
        }

    for spec in a.systems:
        if "=" not in spec:
            print(f"skipping malformed --systems entry: {spec}", file=sys.stderr)
            continue
        where, rank = spec.split("=", 1)
        systems[f"{Path(where).name} ({rank})"] = load_dir(where, episodes, rank, a.k)

    systems[a.baseline_name] = {ep: baseline_last_n(parsed[ep]) for ep in episodes}

    rows, detail = [], {}
    per_ep_top1 = {}
    for name, preds in systems.items():
        m = system_metrics(preds, gold, a.iou, (1, 3, 5))
        t1 = {ep: float(m["per_episode"][ep]["top1_hit"]) for ep in episodes}
        rc = {ep: float(m["per_episode"][ep]["recall"]) for ep in episodes}
        per_ep_top1[name] = t1
        p1, l1, h1 = bootstrap_ci(t1, a.boot)
        pr, lr, hr = bootstrap_ci(rc, a.boot)
        rows.append(
            {
                "System": name,
                "Top-1": f"{m['top1']}/{m['n_episodes']}",
                "Top-1 CI": fmt_ci(p1, l1, h1),
                "Top-3": f"{m['top3']}/{m['n_episodes']}",
                "Recall": f"{m['gold_matched']}/{m['n_gold']}",
                "Recall CI": fmt_ci(pr, lr, hr),
                "Med start err": m["median_start_err"],
                "Med end err": m["median_end_err"],
            }
        )
        detail[name] = m

    # B2 random, averaged over seeds
    r_top1, r_rec = [], []
    for seed in range(a.random_seeds):
        preds = {ep: baseline_random(parsed[ep], seed=seed, k=a.k) for ep in episodes}
        m = system_metrics(preds, gold, a.iou, (1, 3, 5))
        r_top1.append(m["top1"])
        r_rec.append(m["gold_matched"])
    if r_top1:
        rows.append(
            {
                "System": f"B2 random ({a.random_seeds} seeds)",
                "Top-1": f"mean {statistics.mean(r_top1):.1f}/{len(episodes)}",
                "Top-1 CI": f"range [{min(r_top1)}, {max(r_top1)}]",
                "Top-3": "-",
                "Recall": f"mean {statistics.mean(r_rec):.1f}",
                "Recall CI": f"range [{min(r_rec)}, {max(r_rec)}]",
                "Med start err": "-",
                "Med end err": "-",
            }
        )

    cols = ["System", "Top-1", "Top-1 CI", "Top-3", "Recall", "Recall CI", "Med start err", "Med end err"]
    print(f"\nEpisodes: {len(episodes)}   Gold windows: {sum(len(v) for v in gold.values())}   IoU >= {a.iou}\n")
    print(markdown_table(rows, cols))

    paired = {}
    base = per_ep_top1.get(a.baseline_name, {})
    print("\nPaired bootstrap on top-1, each system vs the baseline:\n")
    prows = []
    for name, t1 in per_ep_top1.items():
        if name == a.baseline_name:
            continue
        r = paired_bootstrap(t1, base, a.boot)
        paired[name] = r
        prows.append(
            {
                "System": name,
                "vs": a.baseline_name,
                "mean delta": r.get("mean_delta"),
                "95% CI": f"[{r.get('ci_low')}, {r.get('ci_high')}]",
                "P(better)": r.get("p_a_better"),
                "significant": r.get("significant"),
            }
        )
    print(markdown_table(prows, ["System", "vs", "mean delta", "95% CI", "P(better)", "significant"]))
    if len(episodes) < 12:
        print(
            f"\nNOTE: {len(episodes)} episodes is too few to separate these systems. "
            "Widen the gold set before claiming any system is better."
        )

    Path(a.out).write_text(
        json.dumps(
            {
                "config": {"iou": a.iou, "k": a.k, "boot": a.boot, "episodes": episodes},
                "table": rows,
                "paired_vs_baseline": paired,
                "detail": detail,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
