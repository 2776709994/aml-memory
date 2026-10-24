"""Explicit retraction: "forget that", "no longer", "scratch that".

The service stores what it is told, which is right — a memory that silently rewrites
itself is worse than one that keeps history. But when a turn *explicitly* retracts an
earlier fact, the reader prompt's "prefer the most recent value when sources conflict"
rule is a coin flip if both rows come back. Marking the old row as superseded (never
deleting it) makes the newest truth the only candidate.

Deliberately conservative: one candidate per retraction, same speaker preferred, and a
high lexical-overlap threshold. This module only *proposes*; the store decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .bm25 import STOPWORDS as _stopwords
from .ingest import Unit

_FORGET = re.compile(
    r"\b(?:forget|delete|remove|erase|scratch|ignore|disregard|drop)\b"
    r"[^.?!]{0,40}?\b(?:that|this|it|my|the|about|what\s+i\s+said|everything)\b",
    re.I,
)
_SUPERSEDE = re.compile(
    r"\b(?:no\s+longer|not\s+anymore|not\s+any\s+more|never\s+mind|scratch\s+that|"
    r"instead|actually|correction|i\s+was\s+wrong|that'?s\s+wrong|i\s+meant)\b",
    re.I,
)
_DELETE_ALL = re.compile(r"\b(?:forget|delete|erase|ignore)\s+(?:everything|all)\b", re.I)
_HEADER = re.compile(r"^[^:()]{1,40}\([^)]*\):\s*")
# Words that belong to the *frame* of a retraction rather than to what it is about.
# Without dropping them, "forget my home address, that was the old place" scores worse
# against "my home address is 12 Main Street" than it should.
_FRAME = frozenset(
    """forget forgot forgets delete deletes remove removes erase scratch ignore disregard drop
    said say saying mentioned mention telling told about that this these those it its
    no longer anymore any more never mind instead actually correction correct wrong meant
    mean old new place thing things everything all please sorry rather already still just
    only really very much some any what i said earlier before previously""".split()
)


@dataclass(frozen=True)
class Retraction:
    kind: str          # "forget" | "supersede"
    text: str          # the utterance, for the lexical match
    global_scope: bool  # "forget everything"


def detect(text: str) -> Retraction | None:
    """Return the retraction carried by one turn, if any."""
    body = body_of(text)
    if not body:
        return None
    if _DELETE_ALL.search(body):
        return Retraction("forget", body, True)
    if _FORGET.search(body):
        return Retraction("forget", body, False)
    if _SUPERSEDE.search(body):
        return Retraction("supersede", body, False)
    return None


def body_of(text: str) -> str:
    """Strip the ``Speaker (date): `` header the ingest layer adds."""
    return _HEADER.sub("", text or "", count=1).strip()


def _tokens(text: str, *, drop_frame: bool = False) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9']+", (text or "").lower()):
        tok = raw.strip("'")
        if len(tok) < 2 or tok in _stopwords:
            continue
        if drop_frame and tok in _FRAME:
            continue
        out.add(tok)
    return out


def overlap(target: set[str], candidate: set[str]) -> float:
    """Containment of the *retraction's* subject in the candidate.

    Jaccard is wrong here: a retraction is short ("forget my home address") and the
    stored fact is long ("my home address is 12 Main Street, Bristol"), so Jaccard is
    low even on a perfect match while a long unrelated turn can score similarly.
    """
    if not target or not candidate:
        return 0.0
    shared = target & candidate
    if len(shared) < 2:  # one common word is not evidence of "about the same thing"
        return 0.0
    return len(shared) / len(target)


def pick_superseded(
    retraction: Retraction,
    earlier: Sequence[Unit],
    *,
    speaker: str = "",
    threshold: float = 0.45,
) -> Unit | None:
    """Best earlier turn the retraction is talking about, or None.

    ``earlier`` must already be restricted to the same user and to strictly older
    turns. Same-speaker candidates get a small edge: people retract their own facts far
    more often than their interlocutor's.
    """
    target = _tokens(retraction.text, drop_frame=True)
    if len(target) < 2:
        return None
    best: tuple[float, int, Unit] | None = None
    for unit in earlier:
        score = overlap(target, _tokens(body_of(unit.text)))
        if speaker and unit.speaker == speaker:
            score += 0.05
        if score < threshold:
            continue
        if best is None or score > best[0]:
            best = (score, unit.seq, unit)
    return best[2] if best else None


def apply(store, *, user_id: str, new_units: Iterable[Unit], threshold: float) -> int:
    """Mark earlier turns as superseded by the retractions in ``new_units``.

    Returns how many turns were superseded. Never raises on a single bad turn: losing
    one retraction is cheaper than failing an Add.
    """
    marked = 0
    units = [u for u in new_units if getattr(u, "id", None)]
    if not units:
        return 0
    for unit in units:
        found = detect(unit.text)
        if found is None:
            continue
        if found.global_scope:
            continue  # "forget everything" is a product decision, not a heuristic one
        earlier = store.earlier_units(user_id, before_seq=unit.seq, limit=600)
        victim = pick_superseded(found, earlier, speaker=unit.speaker, threshold=threshold)
        if victim is None or victim.id == unit.id:
            continue
        if store.mark_superseded([victim.id], by_id=unit.id):
            marked += 1
    return marked
