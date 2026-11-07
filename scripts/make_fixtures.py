"""Generate synthetic screenplays in the real input format, plus matching
annotations and a synthetic gold set.

Point of this: iterate on features / scoring / refinement / evaluation with zero
API calls and zero contact with data/raw. Nothing here is real screenplay text.

    python scripts/make_fixtures.py --out fixtures --episodes 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAMES = ["MAYA", "RUDRA", "KAVYA", "ARJUN", "DEVIKA", "SHANKAR"]
EMOTIONS = ["ANGRY", "WHISPERING", "BREATHLESS", "COLD", "PLEADING"]
TAGS = ["(O.S.)", "(V.O.)", "(CONT'D)"]
LINES = [
    "You were never supposed to find out about the letter.",
    "Say that again and I will walk out of this house tonight.",
    "The money was never mine to give away.",
    "Who told you where I was staying?",
    "I signed it. All of it. Yesterday.",
    "Do not touch that drawer.",
]
ACTIONS = [
    "She steps back and the chair scrapes across the floor.",
    "He grabs her wrist and the glass shatters on the tiles.",
    "The envelope slips out of her hand.",
]
CAMERAS = [
    "Slow push in on the closed door.",
    "Crash zoom to her face.",
    "Wide shot of the empty courtyard.",
]
SFX = ["A loud slap echoes.", "Thunder cracks overhead.", "A phone rings twice and stops."]


def ts(v: float) -> str:
    return f"({int(v // 60):02d}:{v % 60:05.2f})"


def gen_episode(ep_num: int, rng: random.Random, quirks: bool):
    beats, t = [], 0.0
    # cold open on camera so there is no speech in the first 3s (matches 66% of real episodes)
    beats.append(("CAMERA", None, rng.choice(CAMERAS), 2.4))
    t += 2.4
    n_beats = rng.randint(26, 40)
    # half the episodes end on the planted promo, like real microdrama tails;
    # the rest plant it mid-episode so a tail baseline cannot win by default
    if ep_num % 2 == 0:
        plant_at = max(n_beats - 13, 2)
    else:
        plant_at = rng.randint(6, max(7, n_beats - 20))
    for i in range(n_beats):
        if i == plant_at:
            beats.append(("SFX", None, "A loud slap echoes.", 1.2, "plant_start"))
            for j in range(rng.randint(8, 11)):
                spk = NAMES[(ep_num + j) % len(NAMES)]
                beats.append(("DIALOGUE", f"{spk} ({rng.choice(EMOTIONS)})", rng.choice(LINES), rng.uniform(2.2, 4.1)))
                if j == 3:
                    beats.append(("ACTION", None, rng.choice(ACTIONS), 1.8))
            beats.append(("DIALOGUE", f"{NAMES[ep_num % len(NAMES)]} (COLD)", "Who told you where I was staying?", 3.1, "plant_end"))
            continue
        kind = rng.choices(["DIALOGUE", "CAMERA", "ACTION", "SFX"], [0.42, 0.42, 0.11, 0.05])[0]
        hdr = None
        if kind == "DIALOGUE":
            spk = rng.choice(NAMES)
            if quirks and rng.random() < 0.18:
                hdr = f"{spk} {rng.choice(TAGS)} ({rng.choice(EMOTIONS)})"
            else:
                hdr = f"{spk} ({rng.choice(EMOTIONS)})"
        text = {"DIALOGUE": LINES, "CAMERA": CAMERAS, "ACTION": ACTIONS, "SFX": SFX}[kind]
        beats.append((kind, hdr, rng.choice(text), rng.uniform(1.2, 4.0)))

    rows, ann, t = [], {}, 0.0
    ep_id = f"episode-{ep_num:03d}"
    plant_start = plant_end = None
    for idx, b4 in enumerate(beats, start=1):
        kind, hdr, text, dur = b4[0], b4[1], b4[2], b4[3]
        mark = b4[4] if len(b4) > 4 else None
        bid = f"{ep_id}-b{idx:04d}"
        start = round(t, 2)
        end = round(t + dur, 2)
        # camera beats run concurrent with the next beat, like the real packs
        t = end if kind != "CAMERA" else round(t + dur * 0.4, 2)
        rows.append({"beat_id": bid, "kind": kind, "header": hdr, "text": text, "start_s": start, "end_s": end})
        strong = text.startswith("A loud slap") or text.startswith("Crash zoom")
        open_q = text.endswith("?")
        ann[bid] = {
            "h": 3 if strong else (2 if kind == "DIALOGUE" and idx % 5 == 0 else 1),
            "q": 3 if open_q else (2 if kind == "DIALOGUE" and idx % 4 == 0 else 1),
            "r": bool(text.startswith("I signed it")),
            "n": 0 if kind != "DIALOGUE" else (1 if idx % 7 == 0 else 0),
            "t": 3 if strong or open_q else (2 if kind in ("DIALOGUE", "ACTION") else 1),
        }
        if mark == "plant_start" and plant_start is None:
            plant_start = start
        if mark == "plant_end" and plant_start is not None:
            plant_end = end
    if plant_end is None and plant_start is not None:
        plant_end = min(plant_start + 42.0, rows[-1]["end_s"])
    return ep_id, rows, ann, (plant_start, plant_end)


def render(ep_id: str, ep_num: int, rows) -> str:
    out = [
        f"EPISODE NUMBER: {ep_num}",
        "EPISODE LOGLINE: Synthetic fixture episode for offline pipeline tests.",
        "CONTINUITY NOTES: Next episode picks up immediately.",
        "",
        "--- SCRIPT START ---",
        "SCENE 1",
        "Scene heading: INT. FIXTURE HOUSE - NIGHT",
        "Scene start: (00:00.00)",
        f"Scene end: {ts(rows[-1]['end_s'])}",
        "Actions and dialogue:",
    ]
    for b in rows:
        hdr = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SFX": "SOUND FX"}.get(b["kind"]) or b["header"]
        out.append(f"**{hdr}:**")
        out.append(f"Start: {ts(b['start_s'])} | End: {ts(b['end_s'])}")
        out.append(b["text"])
    out.append("--- SCRIPT END ---")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fixtures")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--start-num", type=int, default=901)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-quirks", action="store_true", help="skip double-parenthetical headers")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "parsed").mkdir(parents=True, exist_ok=True)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    gold = ["gold_id,episode,seg_order,start_s,end_s,status"]

    for i in range(a.episodes):
        num = a.start_num + i
        ep_id, rows, ann, (ps, pe) = gen_episode(num, rng, not a.no_quirks)
        (out / "raw" / f"{ep_id}-fixture.txt").write_text(render(ep_id, num, rows), encoding="utf-8")
        beats = [
            {
                "beat_id": b["beat_id"],
                "kind": b["kind"],
                "speaker_norm": (b["header"].split("(")[0].strip().upper() if b["header"] else None),
                "start_s": b["start_s"],
                "end_s": b["end_s"],
                "text": b["text"],
            }
            for b in rows
        ]
        (out / "parsed" / f"{ep_id}.json").write_text(
            json.dumps({"episode": ep_id, "episode_end_s": rows[-1]["end_s"], "beats": beats}, indent=2),
            encoding="utf-8",
        )
        (out / "annotated" / f"{ep_id}.json").write_text(
            json.dumps({"episode": ep_id, "annotations": ann}, indent=2), encoding="utf-8"
        )
        if ps is not None and pe is not None and pe - ps >= 20.0:
            gold.append(f"F{num}_REF001,{num},1,{ps:.2f},{pe:.2f},synthetic")

    (out / "gold_synthetic.csv").write_text("\n".join(gold) + "\n", encoding="utf-8")
    print(f"wrote {a.episodes} fixture episodes to {out}/raw, parsed, annotated")
    print(f"synthetic gold windows: {len(gold) - 1} -> {out}/gold_synthetic.csv")


if __name__ == "__main__":
    main()
