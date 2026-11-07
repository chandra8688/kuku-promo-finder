"""How much do two independent annotation runs disagree, and does it change the picks?

    python scripts/annotation_stability.py --parsed data/parsed \
        --a data/annotated --b outputs_final/annotated --out eval/annotation_stability.json

Compares the S2 annotations of the same episodes from two runs (same model, same
prompt) beat by beat, then runs the production engine on each and compares the
top picks. No model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consistency import annotation_agreement
from src.evalkit import iou
from src.search import search_episode

FIELDS = ("h", "q", "n", "t")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--a", required=True, help="annotated dir of run A")
    ap.add_argument("--b", required=True, help="annotated dir of run B")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", default="eval/annotation_stability.json")
    a = ap.parse_args()

    eps = sorted(p.stem for p in Path(a.a).glob("*.json") if (Path(a.b) / p.name).exists() and (Path(a.parsed) / p.name).exists())
    if not eps:
        print("no episodes present in all three dirs", file=sys.stderr)
        return 1

    beats = 0
    agree = {k: 0.0 for k in FIELDS + ("r",)}
    top1_id = top1_iou = 0
    overlap = 0
    per_ep = {}
    for ep in eps:
        parsed = load(Path(a.parsed) / f"{ep}.json")
        ann_a = load(Path(a.a) / f"{ep}.json")
        ann_b = load(Path(a.b) / f"{ep}.json")
        rep = annotation_agreement([ann_a.get("annotations", {}), ann_b.get("annotations", {})])
        n = rep.get("comparable_beats", 0)
        beats += n
        for k in FIELDS:
            agree[k] += rep["exact_agreement"][k] * n if n else 0.0
        agree["r"] += rep.get("r_agreement", 0.0) * n

        pa = search_episode(parsed, ann_a, top_k=a.top_k)
        pb = search_episode(parsed, ann_b, top_k=a.top_k)
        same_id = bool(pa and pb and pa[0]["candidate_id"] == pb[0]["candidate_id"])
        same_clip = bool(pa and pb and iou(pa[0]["start_s"], pa[0]["end_s"], pb[0]["start_s"], pb[0]["end_s"]) >= 0.5)
        ov = len({c["candidate_id"] for c in pa} & {c["candidate_id"] for c in pb})
        top1_id += same_id
        top1_iou += same_clip
        overlap += ov
        per_ep[ep] = {"comparable_beats": n, "top1_same_id": same_id, "top1_same_clip": same_clip, f"top{a.top_k}_overlap": ov}

    out = {
        "run_a": a.a,
        "run_b": a.b,
        "episodes": len(eps),
        "comparable_beats": beats,
        "exact_agreement": {k: round(v / beats, 4) if beats else 0.0 for k, v in agree.items()},
        "top1_same_id": top1_id,
        "top1_same_clip_iou50": top1_iou,
        f"mean_top{a.top_k}_overlap": round(overlap / len(eps), 2),
        "per_episode": per_ep,
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"episodes {len(eps)}, comparable beats {beats}")
    print("exact agreement: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in out["exact_agreement"].items()))
    print(f"top-1 same window id: {top1_id}/{len(eps)}")
    print(f"top-1 same clip (IoU >= 0.5): {top1_iou}/{len(eps)}")
    print(f"mean top-{a.top_k} overlap: {out[f'mean_top{a.top_k}_overlap']}/{a.top_k}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
