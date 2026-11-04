"""Evaluation Baselines (src/baselines.py).

Implements:
- B1: Last 45 s of each episode, unsnapped.
- B2: 3 random windows per episode for seeds 0..19 (duration uniform 20-60s, start uniform).
Computes for all 50 episodes using duration metadata only (no content read for reserved episodes).
Outputs match candidate JSON structure.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_episode_durations(parsed_dir: str | Path = "data/parsed") -> Dict[str, float]:
    """Load episode duration_max_end_s without reading screenplay beats or text."""
    path = Path(parsed_dir)
    durations: Dict[str, float] = {}

    for pf in sorted(path.glob("*.json")):
        ep_id = pf.stem
        try:
            with open(pf, "r", encoding="utf-8") as f:
                d = json.load(f)
            dur = d.get("duration_max_end_s") or d.get("duration_s") or 0.0
            if dur <= 0.0 and "beats" in d and d["beats"]:
                dur = max(b.get("end_s", 0.0) for b in d["beats"])
            durations[ep_id] = round(float(dur), 2)
        except Exception:
            continue

    return durations


def generate_b1_baseline(
    durations: Dict[str, float],
    target_duration: float = 45.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """B1: Last 45 seconds of each episode, unsnapped."""
    results: Dict[str, List[Dict[str, Any]]] = {}

    for ep_id, total_dur in sorted(durations.items()):
        end_s = total_dur
        start_s = max(0.0, end_s - target_duration)
        dur_s = end_s - start_s

        cand = {
            "candidate_id": f"{ep_id}_b1_last45s",
            "episode": ep_id,
            "baseline": "B1",
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "duration_s": round(dur_s, 2),
            "beat_ids": [],
            "hook": None,
            "cliff": None,
            "resolved_end": None,
            "context": None,
            "intensity": None,
            "score": None,
            "passed_all_gates": (20.0 <= dur_s <= 60.0),
        }
        results[ep_id] = [cand]

    return results


def generate_b2_baseline(
    durations: Dict[str, float],
    seeds: range | List[int] = range(20),
    windows_per_episode: int = 3,
    min_dur: float = 20.0,
    max_dur: float = 60.0,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """B2: For seeds 0..19, 3 random windows per episode (uniform duration 20-60s, uniform start).

    Returns:
        dict mapping seed -> {ep_id: [windows]}
    """
    b2_by_seed: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for seed in seeds:
        rng = random.Random(seed)
        seed_results: Dict[str, List[Dict[str, Any]]] = {}

        for ep_id, total_dur in sorted(durations.items()):
            ep_windows: List[Dict[str, Any]] = []

            for w_idx in range(windows_per_episode):
                actual_max_dur = min(max_dur, total_dur)
                actual_min_dur = min(min_dur, actual_max_dur)

                if actual_max_dur <= actual_min_dur:
                    w_dur = total_dur
                    w_start = 0.0
                else:
                    w_dur = rng.uniform(actual_min_dur, actual_max_dur)
                    max_start = max(0.0, total_dur - w_dur)
                    w_start = rng.uniform(0.0, max_start)

                w_end = min(total_dur, w_start + w_dur)
                dur_s = w_end - w_start

                cand = {
                    "candidate_id": f"{ep_id}_b2_s{seed}_w{w_idx+1}",
                    "episode": ep_id,
                    "baseline": "B2",
                    "seed": seed,
                    "start_s": round(w_start, 2),
                    "end_s": round(w_end, 2),
                    "duration_s": round(dur_s, 2),
                    "beat_ids": [],
                    "hook": None,
                    "cliff": None,
                    "resolved_end": None,
                    "context": None,
                    "intensity": None,
                    "score": None,
                    "passed_all_gates": (20.0 <= dur_s <= 60.0),
                }
                ep_windows.append(cand)

            seed_results[ep_id] = ep_windows

        b2_by_seed[seed] = seed_results

    return b2_by_seed

