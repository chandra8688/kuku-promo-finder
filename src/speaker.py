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
