"""Speaker header parsing that survives more than one parenthetical.

The current RE_EMOTION in src/parse.py is anchored with `$` right after a single
parenthetical group, so a header like

    **MAYA (O.S.) (COMMANDING):**

does not match at all: speaker_norm keeps both bracketed blocks unparsed and
emotion_label falls back to None. That is the 56-beats-across-24-episodes case
already recorded in AI_LOG.md. This handles any number of groups and separates
mic/placement tags from the emotion.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

RE_PAREN = re.compile(r"\(([^)]*)\)")

MODIFIER_TAGS = {
    "O.S.", "OS", "OFF", "O.C.", "OC", "V.O.", "VO", "VOICEOVER", ("V" + "OICE OVER"),
    "CONT'D", "CONTD", "CONT", "PRE-LAP", "PRELAP", "SUBTITLE", "SUBTITLED",
    "ON PHONE", "PHONE", "FILTERED", "ECHO", "FLASHBACK", "MEMORY",
}


def _norm_tag(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def classify_speaker_header(
    raw: str, aliases: Optional[Dict[str, str]] = None
) -> Tuple[str, Optional[str], List[str]]:
    """Returns (speaker_norm, emotion_label, modifiers).

    >>> classify_speaker_header("MAYA (O.S.) (COMMANDING):")
    ('MAYA', 'COMMANDING', ['O.S.'])
    >>> classify_speaker_header("MAYA (ANGRY)")
    ('MAYA', 'ANGRY', [])
    >>> classify_speaker_header("MAYA")
    ('MAYA', None, [])
    """
    cleaned = raw.strip().rstrip(":").strip()
    groups = [g.strip() for g in RE_PAREN.findall(cleaned)]
    name = RE_PAREN.sub("", cleaned).strip().rstrip(":").strip()
    name = re.sub(r"\s+", " ", name).upper()

    mods: List[str] = []
    emotions: List[str] = []
    for g in groups:
        if not g:
            continue
        if _norm_tag(g) in MODIFIER_TAGS:
            mods.append(_norm_tag(g))
        else:
            emotions.append(g.strip())

    if aliases:
        name = aliases.get(name, name)
    emotion = ", ".join(e.upper() for e in emotions) if emotions else None
    return name, emotion, mods


# Titles that can be dropped to find the same person under a shorter name.
# Merged only when the shorter name also speaks somewhere in the show, so
# LORD HALE and LADY HALE stay two people (HALE never speaks).
STRIP_TITLES = ("LADY ", "LORD ", "SIR ", "THE ", "DAME ", "MR. ", "MRS. ", "MISS ", "MR ", "MRS ")


def auto_alias_map(names) -> Dict[str, str]:
    """Show-wide alias map built from the speaker names alone, no model call.

    >>> auto_alias_map(["MAYA", "LADY MAYA", "LORD HALE", "LADY HALE", "THE GUARD", "GUARD"])
    {'LADY MAYA': 'MAYA', 'THE GUARD': 'GUARD'}
    """
    present = {n for n in names if n}
    out: Dict[str, str] = {}
    for n in sorted(present):
        for t in STRIP_TITLES:
            if n.startswith(t):
                short = n[len(t):].strip()
                if short in present and short != n:
                    out[n] = short
                break
    return out
