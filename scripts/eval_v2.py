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
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalkit import bootstrap_ci, bootstrap_ratio_ci, fmt_ci, markdown_table, paired_bootstrap, system_metrics
from src.labels import format_cell, load_labels, precision_at_k
from src.search import baseline_last_n, baseline_random, search_episode
from src.tier0 import generate_tier0_candidates_for_episode

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


HUMAN_STATUSES = {"human", "ai_proposed_human_reviewed"}


def provenance(gold: dict) -> dict:
    counts: dict = {}
    for wins in gold.values():
        for w in wins:
            st = (w.get("status") or "unknown").strip() or "unknown"
            counts[st] = counts.get(st, 0) + 1
    n = sum(counts.values())
    human = sum(v for k, v in counts.items() if k in HUMAN_STATUSES)
    return {"counts": counts, "n": n, "human": human, "all_human": n > 0 and human == n}


def provenance_line(prov: dict) -> str:
    parts = ", ".join(f"{v} `{k}`" for k, v in sorted(prov["counts"].items()))
    if prov["all_human"]:
        return f"> **Reference provenance**: {parts}. Every window was labelled or verified by a person."
    return (
        f"> **Reference provenance**: {parts}. {prov['n'] - prov['human']} of {prov['n']} windows were not verified "
        "by a person, so these numbers measure agreement with an AI-assisted reference, not with human editors."
    )


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


def build_eval_report_markdown(
    table_rows: list,
    prows: list,
    detail: dict,
    gold: dict,
    gold_path: str,
    episodes: list,
    iou_thresh: float,
    show_promos_path: Optional[str] = None,
    labels_path: Optional[str] = None,
) -> str:
    """Generate comprehensive evaluation report markdown with honest baseline framing."""
    n_eps = len(episodes)
    n_gold = sum(len(v) for v in gold.values())
    is_reserved = n_eps == 5 and all(norm_ep(e) in ["episode-003", "episode-013", "episode-023", "episode-033", "episode-043"] for e in episodes)

    lines = [
        "# Promo Finder — Evaluation Report",
        "",
        "> [!IMPORTANT]",
        f"> **Evaluation Scope**: {n_eps} episodes, {n_gold} reference windows (IoU threshold $\\ge {iou_thresh}$).",
        provenance_line(provenance(gold)),
    ]
    if is_reserved:
        lines.extend([
            "> **Dataset Partition**: Reserved **Test Set** (`3, 13, 23, 33, 43`). Held out, frozen, and untouched.",
            "> **Development Set**: reference windows in `gold/gold_dev.csv`, disjoint from these episodes. Weights were fitted on it, so any dev-set number is in-sample.",
            "> **Baseline Rule**: Baseline B1 (last 45s), B2 (random), and B3 (Tier-0) are kept in all tables.",
        ])
    else:
        lines.extend([
            f"> **Dataset Source**: `{gold_path}`.",
            "> **Partition**: Development set. Weights were fitted on it, so these numbers are in-sample, not a test result.",
        ])
    lines.extend([
        "",
        "---",
        "",
        f"## 1. Primary Evaluation Table (IoU $\\ge {iou_thresh}$)",
        "",
        markdown_table(table_rows, ["System", "Top-1", "Top-1 CI", "Top-3", "Recall", "Recall CI", "Med start err", "Med end err"]),
        "",
        "---",
        "",
        "## 2. Paired Bootstrap Comparison vs Baseline (B1 Last 45s)",
        "",
        markdown_table(prows, ["System", "vs", "mean delta", "95% CI", "P(better)", "significant"]),
        "",
        "---",
        "",
        "## 3. Commercial Framing & Show-Level Precision@15",
        "",
        "**Why per-episode Top-1 is not the headline metric:**",
        "- The business objective is to decide **which promos to post first across the entire series**.",
        "- Baseline B1 (last 45s) outputs exactly one unranked, fixed-length candidate per episode with no score.",
        "- Consequently, **B1 cannot produce a show-level ranking** and cannot prioritize content across episodes.",
        "",
    ])

    # Show-level Precision@15
    show_table_rows = []
    labels = load_labels(labels_path) if labels_path else {}
    pipeline_res = None
    if show_promos_path and Path(show_promos_path).exists():
        try:
            sd = json.loads(Path(show_promos_path).read_text(encoding="utf-8"))
            ranked_promos = sd.get("ranked_promos", sd if isinstance(sd, list) else [])
            if ranked_promos:
                pipeline_res = precision_at_k(ranked_promos, labels, k=15)
                show_table_rows.append({
                    "System": "Pipeline (V2 features + refine)",
                    "Precision@15": format_cell(pipeline_res),
                    "Labelled By": "human review via scripts/review.py" if labels else "not labelled",
                    "Ordering Capability": "Global score ranking across 50 episodes with diversity cap <= 2",
                })
        except Exception as exc:
            show_table_rows.append({
                "System": "Pipeline (V2 features + refine)",
                "Precision@15": f"not measured ({type(exc).__name__})",
                "Labelled By": "-",
                "Ordering Capability": "Global score ranking across 50 episodes with diversity cap <= 2",
            })

    if not show_table_rows:
        show_table_rows.append({
            "System": "Pipeline (V2 features + refine)",
            "Precision@15": "not measured",
            "Labelled By": "-",
            "Ordering Capability": "Global score ranking across 50 episodes with diversity cap <= 2",
        })

    show_table_rows.extend([
        {
            "System": "B1 last 45s",
            "Precision@15": "n/a",
            "Labelled By": "-",
            "Ordering Capability": "Cannot rank across show (1 unranked clip per episode)",
        },
        {
            "System": "B3 Tier-0 (deterministic)",
            "Precision@15": "not labelled",
            "Labelled By": "-",
            "Ordering Capability": "Deterministic heuristic rule score ranking",
        },
    ])

    lines.extend([
        markdown_table(show_table_rows, ["System", "Precision@15", "Labelled By", "Ordering Capability"]),
        "",
        "---",
        "",
        "## 4. Match Details (V2 Features + Boundary Refinement)",
        "",
    ])

    v2_matches = []
    v2_detail = detail.get("V2 features+refine", {})
    per_ep = v2_detail.get("per_episode", {})
    for ep in episodes:
        ep_matches = per_ep.get(ep, {}).get("matches", [])
        for m in ep_matches:
            v2_matches.append({
                "Episode": ep,
                "Pred Rank": m.get("pred_rank"),
                "Gold ID": m.get("gold_id"),
                "Candidate Window": f"[{m['pred'][0]:.1f}s - {m['pred'][1]:.1f}s]",
                "Gold Window": f"[{m['gold'][0]:.1f}s - {m['gold'][1]:.1f}s]",
                "IoU": f"{m.get('iou', 0.0):.4f}",
                "Start Error": f"{m.get('start_err', 0.0):.2f}s",
                "End Error": f"{m.get('end_err', 0.0):.2f}s",
            })

    if v2_matches:
        lines.append(markdown_table(v2_matches, ["Episode", "Pred Rank", "Gold ID", "Candidate Window", "Gold Window", "IoU", "Start Error", "End Error"]))
    else:
        lines.append("*No matches recorded under current threshold.*")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--annotated", default=None, help="omit to evaluate baselines only")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--require-human", action="store_true",
                    help="refuse to run unless every reference window was labelled or verified by a person")
    ap.add_argument("--systems", nargs="*", default=[], help="NAME=dir or dir=rankkey entries, e.g. data/judged=judged")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--baseline-name", default="B1 last 45s")
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--show-promos", default=None, help="Show-level ranked promos JSON (default: outputs_final/, else outputs/)")
    ap.add_argument("--config", default="config/scoring_v2.yaml", help="Scoring config YAML with weights")
    ap.add_argument("--labels", default="labels.csv", help="Human review labels from scripts/review.py; precision@15 is computed only from these")
    ap.add_argument("--report", default=None, help="Write the markdown report here (opt-in: the old default overwrote eval_report.md with whatever was evaluated, fixtures included)")
    ap.add_argument("--out", default="eval_v2.json")
    a = ap.parse_args()

    if a.show_promos is None:
        a.show_promos = next((p for p in ("outputs_final/promos_ranked.json", "outputs/promos_ranked.json") if Path(p).exists()),
                             "outputs_final/promos_ranked.json")
    gold = load_gold(a.gold)
    prov = provenance(gold)
    if a.require_human and not prov["all_human"]:
        print(f"REFUSING: --require-human, but reference provenance is {prov['counts']}", file=sys.stderr)
        sys.exit(2)
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

    cfg_weights = None
    if a.config and Path(a.config).exists():
        import yaml
        try:
            cfg_data = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
            cfg_weights = cfg_data.get("weights")
        except Exception as e:
            print(f"warning: failed to load config {a.config}: {e}", file=sys.stderr)

    systems: dict = {}
    if a.annotated:
        v2 = {}
        for ep in episodes:
            f = Path(a.annotated) / f"{ep}.json"
            ann = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"annotations": {}}
            v2[ep] = search_episode(parsed[ep], ann, weights=cfg_weights, top_k=a.k)
        systems["V2 features+refine"] = v2
        systems["V2 no refine"] = {
            ep: search_episode(parsed[ep], json.loads((Path(a.annotated) / f"{ep}.json").read_text(encoding="utf-8")) if (Path(a.annotated) / f"{ep}.json").exists() else {"annotations": {}}, weights=cfg_weights, top_k=a.k, refine=False)
            for ep in episodes
        }

    for spec in a.systems:
        if "=" not in spec:
            print(f"skipping malformed --systems entry: {spec}", file=sys.stderr)
            continue
        where, rank = spec.split("=", 1)
        systems[f"{Path(where).name} ({rank})"] = load_dir(where, episodes, rank, a.k)

    systems[a.baseline_name] = {ep: baseline_last_n(parsed[ep]) for ep in episodes}

    # B3 Tier-0 (deterministic heuristic baseline)
    b3_preds = {}
    for ep in episodes:
        t0_res = generate_tier0_candidates_for_episode(parsed[ep])
        survs = [c for c in t0_res.get("candidates", []) if c.get("passed_all_gates")]
        survs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        b3_preds[ep] = survs[:a.k]
    systems["B3 Tier-0"] = b3_preds

    rows, detail = [], {}
    per_ep_top1 = {}
    for name, preds in systems.items():
        m = system_metrics(preds, gold, a.iou, (1, 3, 5))
        t1 = {ep: float(m["per_episode"][ep]["top1_hit"]) for ep in episodes}
        per_ep_top1[name] = t1
        p1, l1, h1 = bootstrap_ci(t1, a.boot)
        matched = {ep: float(len({mt["gold_id"] for mt in m["per_episode"][ep]["matches"]})) for ep in episodes}
        n_gold_ep = {ep: float(m["per_episode"][ep]["n_gold"]) for ep in episodes}
        pr, lr, hr = bootstrap_ratio_ci(matched, n_gold_ep, a.boot)
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
    print(f"\nEpisodes: {len(episodes)}   Reference windows: {sum(len(v) for v in gold.values())}   IoU >= {a.iou}")
    print(provenance_line(provenance(gold)).lstrip("> ") + "\n")
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

    # Show-level ranking evaluation output
    show_p = Path(a.show_promos) if a.show_promos else None
    if show_p and show_p.exists():
        try:
            sd = json.loads(show_p.read_text(encoding="utf-8"))
            ranked_promos = sd.get("ranked_promos", [])
            if ranked_promos:
                res = precision_at_k(ranked_promos, load_labels(a.labels), k=15)
                if res["n_labelled"]:
                    print(f"\nShow-Level Precision@15: {format_cell(res)}")
                    if res["rejected"]:
                        codes = ", ".join(sorted({r["reason_code"] for r in res["rejected"] if r["reason_code"]}))
                        print(f"  rejected reasons: {codes or 'unspecified'}")
                else:
                    print(f"\nShow-Level Precision@15: not measured — no labels at {a.labels}")
                    print(f"  label the top 15 first:  python scripts/review.py --promos {a.show_promos} --out {a.labels}")
                print("Note: Baseline B1 cannot produce a show-level ranking (returns 1 unranked clip per episode).")
        except Exception:
            pass

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

    if a.report:
        report_md = build_eval_report_markdown(
            table_rows=rows,
            prows=prows,
            detail=detail,
            gold=gold,
            gold_path=a.gold,
            episodes=episodes,
            iou_thresh=a.iou,
            show_promos_path=a.show_promos,
            labels_path=a.labels,
        )
        Path(a.report).write_text(report_md, encoding="utf-8")
        print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
