"""Confidentiality leak checker v2 (scripts/leak_check.py).

Two independent verification checks:
1. Shingle Leak Check:
   - Builds 8-word shingles from data/raw/*.txt (lowercased, punctuation stripped).
   - Ignores structural boilerplate:
     * Drops any shingle appearing in >= 10 of the raw files.
     * Drops any shingle with fewer than 5 alphabetic tokens (timestamps & numbers).
   - Reports counts of shingles ignored under each rule.

2. Name Denylist Check:
   - Builds denylist from data/parsed/*.json:
     * Words of 4+ letters from speaker_norm (parentheticals stripped).
     * Words of 4+ letters from capitalized runs in loglines.
     * Excludes common English vocabulary and project architectural keywords.
   - Scans tracked files (git ls-files) case-insensitively as whole words.

Reports file names and match counts only; never displays matched text or names.
Exits 0 if both checks pass with zero matches; exits 1 on any match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Common English vocabulary, prepositions, and Antigravity/project keywords
COMMON_EXCLUDED_WORDS = {
    # Generic titles and descriptors that appear in speaker headers but identify no one;
    # src/speaker.py must name titles like these to merge name variants.
    "lady", "lord", "guard", "mysterious", "stranger", "father", "police",
    "with", "while", "when", "after", "before", "from", "into", "then", "that",
    "this", "there", "their", "what", "which", "where", "more", "most", "some",
    "other", "have", "been", "were", "over", "down", "back", "first", "last",
    "about", "above", "again", "against", "also", "because", "could", "should",
    "would", "every", "each", "does", "doing", "done", "only", "same", "such",
    "than", "too", "very", "just", "made", "make", "makes", "much", "many",
    "scene", "camera", "action", "screen", "start", "test", "tests", "line", "lines",
    "dialogue", "open", "close", "ends", "part", "code", "file", "data", "read",
    "prompt", "stage", "model", "models", "role", "roles", "token", "tokens",
    "judge", "gold", "safe", "order", "true", "false", "none", "case", "high",
    "third", "will", "must", "even", "here", "give", "take", "time", "hand",
    "meeting", "night", "morning", "evening", "room", "hall", "house", "door",
}


def get_words(text: str) -> List[str]:
    """Lowercase and strip punctuation into sequential alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_filtered_shingles(
    raw_dir: str | Path = "data/raw",
    shingle_size: int = 8,
    file_frequency_threshold: int = 10,
    min_alphabetic_tokens: int = 5,
) -> Tuple[Set[str], Dict[str, int]]:
    """Build active 8-word shingles, filtering out structural boilerplate."""
    path = Path(raw_dir)
    if not path.exists():
        return set(), {}

    raw_files = list(path.glob("*.txt"))
    if not raw_files:
        return set(), {}

    shingle_file_count: Dict[str, int] = {}
    all_shingles: Set[str] = set()

    for rf in raw_files:
        seen_in_file: Set[str] = set()
        try:
            with open(rf, "r", encoding="utf-8", errors="ignore") as f:
                words = get_words(f.read())
            for i in range(len(words) - shingle_size + 1):
                sh = " ".join(words[i : i + shingle_size])
                seen_in_file.add(sh)
                all_shingles.add(sh)
        except Exception as e:
            print(f"Warning reading {rf}: {e}", file=sys.stderr)

        for sh in seen_in_file:
            shingle_file_count[sh] = shingle_file_count.get(sh, 0) + 1

    total_shingles = len(all_shingles)

    # Rule 1: Structural boilerplate appearing in >= 10 raw files
    rule1_dropped = {
        sh for sh, c in shingle_file_count.items() if c >= file_frequency_threshold
    }

    # Rule 2: Non-alphabetic shingles (numbers/timestamps with < 5 alpha tokens)
    rule2_dropped = {
        sh
        for sh in all_shingles
        if sum(1 for t in sh.split() if re.search(r"^[a-z]+$", t)) < min_alphabetic_tokens
    }

    total_dropped = rule1_dropped | rule2_dropped
    active_shingles = all_shingles - total_dropped

    stats = {
        "raw_files_scanned": len(raw_files),
        "total_unique_shingles": total_shingles,
        "rule1_boilerplate_dropped": len(rule1_dropped),
        "rule2_numbers_timestamps_dropped": len(rule2_dropped),
        "overlap_both_rules": len(rule1_dropped & rule2_dropped),
        "total_shingles_dropped": len(total_dropped),
        "active_shingles_count": len(active_shingles),
    }

    return active_shingles, stats


def build_name_denylist(
    parsed_dir: str | Path = "data/parsed",
    min_word_len: int = 4,
) -> Set[str]:
    """Build denylist of speaker and logline proper nouns."""
    path = Path(parsed_dir)
    if not path.exists():
        return set()

    denylist: Set[str] = set()
    for pf in path.glob("*.json"):
        try:
            with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                d = json.load(f)
        except Exception:
            continue

        # 1. speaker_norm words (parentheticals like emotion/direction stripped)
        for b in d.get("beats", []):
            spk = b.get("speaker_norm")
            if spk and isinstance(spk, str):
                name_clean = re.sub(r"\(.*?\)", "", spk).strip()
                for w in re.findall(r"[a-zA-Z]{4,}", name_clean):
                    w_lower = w.lower()
                    if w_lower not in COMMON_EXCLUDED_WORDS:
                        denylist.add(w_lower)

        # 2. Capitalized runs in logline
        logline = d.get("logline", "")
        if logline and isinstance(logline, str):
            for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", logline):
                for w in m.group(0).split():
                    w_lower = w.lower()
                    if len(w_lower) >= min_word_len and w_lower not in COMMON_EXCLUDED_WORDS:
                        denylist.add(w_lower)

    return denylist


def get_tracked_files() -> List[str]:
    """Return tracked files via git ls-files."""
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        return []


def run_checks(
    raw_dir: str | Path = "data/raw",
    parsed_dir: str | Path = "data/parsed",
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Run both shingle and name denylist checks on tracked files.

    Returns:
        (shingle_matches, name_matches, shingle_stats)
    """
    active_shingles, stats = build_filtered_shingles(raw_dir=raw_dir)
    name_denylist = build_name_denylist(parsed_dir=parsed_dir)

    tracked_files = get_tracked_files()
    shingle_matches: Dict[str, int] = {}
    name_matches: Dict[str, int] = {}

    for tf in tracked_files:
        p = Path(tf)
        if not p.is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            text_lower = content.lower()
            words = re.findall(r"[a-z0-9]+", text_lower)
        except Exception:
            continue

        # Check shingles
        if active_shingles and len(words) >= 8:
            sc = sum(
                1
                for i in range(len(words) - 7)
                if " ".join(words[i : i + 8]) in active_shingles
            )
            if sc > 0:
                shingle_matches[tf] = sc

        # Check name denylist
        if name_denylist:
            nc = 0
            for name in name_denylist:
                nc += len(re.findall(r"\b" + re.escape(name) + r"\b", text_lower))
            if nc > 0:
                name_matches[tf] = nc

    return shingle_matches, name_matches, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidentiality Leak Checker v2.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw screenplay files")
    parser.add_argument("--parsed-dir", default="data/parsed", help="Path to parsed episode JSONs")
    args = parser.parse_args()

    shingle_matches, name_matches, stats = run_checks(
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
    )

    print("=== LEAK CHECK v2 REPORT ===")
    if stats:
        print(f"Raw Files Scanned: {stats['raw_files_scanned']}")
        print(f"Total Unique 8-Word Shingles: {stats['total_unique_shingles']}")
        print(f"  - Dropped Rule 1 (appears in >= 10 files): {stats['rule1_boilerplate_dropped']}")
        print(f"  - Dropped Rule 2 (< 5 alphabetic tokens): {stats['rule2_numbers_timestamps_dropped']}")
        print(f"  - Total Shingles Dropped: {stats['total_shingles_dropped']}")
        print(f"Active Shingles Evaluated: {stats['active_shingles_count']}")
    print("")

    has_failure = False

    if shingle_matches:
        has_failure = True
        print("[FAIL] Shingle Leak Check: Matching raw shingles detected:")
        for tf, cnt in sorted(shingle_matches.items()):
            print(f"  {tf}: {cnt} matching shingle(s)")
    else:
        print("[PASS] Shingle Leak Check: Zero matching dialogue/narration shingles detected.")

    if name_matches:
        has_failure = True
        print("[FAIL] Name Denylist Check: Character/proper names detected:")
        for tf, cnt in sorted(name_matches.items()):
            print(f"  {tf}: {cnt} matching name(s)")
    else:
        print("[PASS] Name Denylist Check: Zero denylist names detected.")

    print("")
    if has_failure:
        print("RESULT: LEAK CHECK FAILED.")
        sys.exit(1)
    else:
        print("RESULT: LEAK CHECK PASSED (Clean Tree).")
        sys.exit(0)


if __name__ == "__main__":
    main()
