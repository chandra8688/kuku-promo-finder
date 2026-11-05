"""Terminal-only window inspector for promo candidate review (scripts/show_window.py).

Prints the window's non-camera beats and the 12s after to the TERMINAL ONLY, with local indices.
Writes nothing to any files or logs. Screenplay text is confidential.

Usage:
    python scripts/show_window.py <episode_id> <start_s> <end_s>
Example:
    python scripts/show_window.py episode-003 58.3 78.9
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_episode_id(raw: str) -> str:
    """Normalize episode input to episode-XXX format."""
    s = raw.strip()
    if s.endswith(".json"):
        s = s[:-5]
    if s.startswith("episode-"):
        return s
    try:
        num = int(s)
        return f"episode-{num:03d}"
    except ValueError:
        return s


def render_window_display(
    episode_id: str,
    parsed_data: Dict[str, Any],
    start_s: float,
    end_s: float,
) -> str:
    """Format the window beats and 12s after lookahead as a terminal-ready string."""
    dur_s = round(end_s - start_s, 2)
    after_end = end_s + 12.0

    all_beats = parsed_data.get("beats", [])
    ep_final_end = max((b.get("end_s", 0.0) for b in all_beats), default=0.0)

    # Filter window non-camera beats
    window_beats = [
        b
        for b in all_beats
        if b.get("kind") != "CAMERA"
        and (b.get("start_s", 0.0) >= start_s - 0.05)
        and (b.get("end_s", 0.0) <= end_s + 0.05)
    ]
    if not window_beats:
        window_beats = [
            b
            for b in all_beats
            if b.get("kind") != "CAMERA"
            and (b.get("end_s", 0.0) > start_s)
            and (b.get("start_s", 0.0) < end_s)
        ]

    # Filter 12s after non-camera beats
    after_beats = [
        b
        for b in all_beats
        if b.get("kind") != "CAMERA"
        and (b.get("end_s", 0.0) > end_s)
        and (b.get("start_s", 0.0) < after_end)
    ]

    lines = []
    lines.append("=" * 80)
    lines.append("NOTICE: Terminal output only. Confidential screenplay inspection.")
    lines.append("This script writes nothing to any files or logs.")
    lines.append("=" * 80)
    lines.append(f"EPISODE   : {episode_id}")
    lines.append(f"WINDOW    : {start_s:.2f}s - {end_s:.2f}s ({dur_s:.2f}s)")
    lines.append(f"LOOKAHEAD : {end_s:.2f}s - {after_end:.2f}s (+12.0s)")
    lines.append("-" * 80)
    lines.append(f"--- WINDOW BEATS ({len(window_beats)} beats, local indices 1..{len(window_beats)}) ---")

    if not window_beats:
        lines.append("  [No non-camera beats found in specified interval]")
    else:
        for idx, b in enumerate(window_beats, start=1):
            st = b.get("start_s", 0.0)
            en = b.get("end_s", 0.0)
            kind = b.get("kind", "")
            spk = b.get("speaker_norm") or "-"
            text = b.get("text", "")
            lines.append(f"[{idx:>2}] ({st:>6.2f}s - {en:>6.2f}s, {kind:<8}, {spk}): {text}")

    lines.append("-" * 80)
    lines.append(f"--- 12s AFTER WINDOW ({end_s:.2f}s .. {after_end:.2f}s) ---")

    if not after_beats:
        if end_s >= ep_final_end - 0.05:
            lines.append("  [END OF EPISODE reached directly at window end cut]")
        else:
            lines.append("  [No non-camera beats in the 12s following window end]")
    else:
        for idx, b in enumerate(after_beats, start=1):
            st = b.get("start_s", 0.0)
            en = b.get("end_s", 0.0)
            kind = b.get("kind", "")
            spk = b.get("speaker_norm") or "-"
            text = b.get("text", "")
            lines.append(f"[+{idx}] ({st:>6.2f}s - {en:>6.2f}s, {kind:<8}, {spk}): {text}")

        if after_end >= ep_final_end - 0.05:
            lines.append("  [END OF EPISODE reached within after window]")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python scripts/show_window.py <episode_id> <start_s> <end_s>")
        print("Example: python scripts/show_window.py episode-003 58.3 78.9")
        sys.exit(1)

    ep_raw = sys.argv[1].strip()
    ep_id = normalize_episode_id(ep_raw)

    try:
        start_s = float(sys.argv[2])
        end_s = float(sys.argv[3])
    except ValueError as e:
        print(f"Error: start_s and end_s must be numeric values: {e}")
        sys.exit(1)

    parsed_file = Path(f"data/parsed/{ep_id}.json")
    if not parsed_file.exists():
        print(f"Error: Parsed file '{parsed_file}' does not exist.")
        sys.exit(1)

    with open(parsed_file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    output = render_window_display(
        episode_id=ep_id,
        parsed_data=parsed_data,
        start_s=start_s,
        end_s=end_s,
    )
    print(output)


if __name__ == "__main__":
    main()

