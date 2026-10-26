"""Turn an AML ``/add`` payload into retrievable, self-describing memory units.

Ingest is where most of the achievable score is decided.  A raw dialogue line such
as ``"yeah, that was fun"`` is useless to a frozen answer model; the same line
annotated with speaker and a resolved date often answers a question on its own.
So every unit is rendered as::

    [2023-05-07] Caroline: I went to the support group yesterday (6 May 2023).

and units keep ``prev_id`` / ``next_id`` links so retrieval can widen context
without paying for duplicated embeddings.
"""
from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Settings
from .llm import LLM
from .store import Unit
from .timeutil import anchor_relative_times, day_str, header_date, human_date, parse_datetime

_SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'\-]{0,28}?)\s*:\s+(?=\S)")
_IMAGE_MARK = "[image]"
_MAX_UNIT_CHARS = 1400
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Message:
    role: str
    content: str
    timestamp: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


def parse_content(content: Any) -> tuple[str, list[str]]:
    """Return (plain text, image urls) from a string or multimodal content array."""
    if isinstance(content, str):
        return content.strip(), []
    if isinstance(content, list):
        parts: list[str] = []
        images: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = str(item.get("type") or "").lower()
                if kind in {"image_url", "image"}:
                    raw = item.get("image_url")
                    if isinstance(raw, dict):
                        url = str(raw.get("url") or "")
                    else:
                        url = str(raw or "")
                    if url:
                        images.append(url)
                        parts.append(_IMAGE_MARK)
                else:
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
        return " ".join(part for part in parts if part).strip(), images
    if content is None:
        return "", []
    return str(content).strip(), []


def parse_messages(raw_messages: Sequence[Any]) -> list[Message]:
    parsed: list[Message] = []
    for raw in raw_messages:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if not isinstance(raw, dict):
            continue
        text, _images = parse_content(raw.get("content"))
        role = str(raw.get("role") or "user")
        parsed.append(Message(role=role, content=text, timestamp=raw.get("timestamp")))
    return parsed


def split_speaker(text: str, role: str) -> tuple[str, str]:
    """Prefer an explicit ``Name:`` prefix inside the content, else fall back to role."""
    match = _SPEAKER_PREFIX.match(text)
    if match:
        name = match.group(1).strip()
        rest = text[match.end():].strip()
        if rest and len(name) <= 30:
            return name, rest
    speaker = {"user": "User", "assistant": "Assistant", "system": "System"}.get(role.lower(), role or "Speaker")
    return speaker, text


def _split_long(text: str, limit: int = _MAX_UNIT_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        if len(buffer) + len(sentence) + 1 <= limit:
            buffer = f"{buffer} {sentence}".strip()
        else:
            if buffer:
                pieces.append(buffer)
            while len(sentence) > limit:
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces or [text[:limit]]


def _embed_text(mode: str, speaker: str, header: str, part_text: str) -> str | None:
    """Which string the embedder sees.

    The stored record must carry the speaker and the absolute date (the reader needs
    both), but the embedding does not have to: repeated speaker names and dates are
    the noisiest part of a conversational corpus. Modes:
      header  - embed the stored record verbatim (the long-standing behaviour)
      speaker - embed "Speaker: body" without the date
      body    - embed the body without speaker or date
    Returns None for `header`, meaning "just use Unit.text".
    """
    if mode == "header":
        return None
    body = part_text[len(header):] if part_text.startswith(header) else part_text
    return f"{speaker}: {body}" if mode == "speaker" else body


class Ingestor:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm

    def build_units(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: Sequence[Message],
    ) -> list[Unit]:
        parsed = [m for m in messages if m.content]
        if not parsed:
            return []
        mode = getattr(self.settings, "embed_text_mode", "header") or "header"

        # Session reference date: first explicit timestamp, else sentence-level cues.
        ref_dt = None
        for message in parsed:
            ref_dt = parse_datetime(message.timestamp)
            if ref_dt:
                break
        session_date = day_str(ref_dt)
        human = human_date(ref_dt)

        units: list[Unit] = []
        carry_dt = ref_dt
        for index, message in enumerate(parsed):
            speaker, body = split_speaker(message.content, message.role)
            explicit = parse_datetime(message.timestamp)
            # A message with no usable timestamp inherits the most recent date we
            # know, and can itself supply one through an embedded session header.
            # Falling back to the batch's first timestamp instead would pin every
            # turn to the session start, which is wrong across a multi-day chunk.
            message_dt = explicit or header_date(message.content) or carry_dt or ref_dt
            if message_dt:
                carry_dt = message_dt
            anchored = anchor_relative_times(body, message_dt or ref_dt)
            date_label = day_str(message_dt) or session_date
            readable = _readable_date(message_dt) if message_dt else ""
            # Speaker first: the answer prompts bucket memories per speaker, and a
            # frozen reader attributes every line much more reliably when the
            # speaker name leads the line.
            header = f"{speaker} ({readable}): " if readable else f"{speaker}: "
            text = f"{header}{anchored}".strip()
            for part_index, chunk in enumerate(_split_long(text)):
                unit_id = _unit_id(request_id, index, part_index)
                created_ms = int(message_dt.timestamp() * 1000) if message_dt else None
                # Every part repeats the header: a long reply split into pieces must
                # not leave anonymous fragments behind, or a fragment shown alone is
                # unattributable and undated.
                part_text = chunk if not part_index else f"{header}[part {part_index + 1}] {chunk}"
                units.append(
                    Unit(
                        id=unit_id,
                        user_id=user_id,
                        session_id=session_id,
                        seq=index * 100 + part_index,
                        kind="turn",
                        speaker=speaker,
                        created_ms=created_ms,
                        date_str=date_label,
                        text=part_text,
                        embed_text=_embed_text(mode, speaker, header, part_text),
                        prev_id=None,
                        next_id=None,
                        meta={
                            "role": message.role,
                            "turn": index,
                            "part": part_index,
                            "session_date": session_date,
                            "request_id": request_id,
                        },
                    )
                )
        for position, unit in enumerate(units):
            if position > 0 and units[position - 1].session_id == unit.session_id:
                unit.prev_id = units[position - 1].id
            if position + 1 < len(units) and units[position + 1].session_id == unit.session_id:
                unit.next_id = units[position + 1].id

        if self.settings.ingest_llm in {"session_summary", "summary", "facts", "all"} and self.llm.enabled:
            try:
                units.extend(self._summaries(units, request_id, user_id, session_id, human))
            except Exception:
                pass
        return units

    def _summaries(
        self,
        units: list[Unit],
        request_id: str,
        user_id: str,
        session_id: str,
        human: str,
    ) -> list[Unit]:
        transcript = "\n".join(unit.text for unit in units)
        if not transcript:
            return []
        budget = max(400, self.settings.session_summary_chars)
        if len(transcript) > budget * 8:
            transcript = transcript[: budget * 8]
        prompt = (
            "Summarize the durable, question-answerable content of this conversation excerpt.\n"
            "Keep: named entities, concrete facts (dates, places, numbers, counts), preferences, "
            "plans, changes of state, and anything stated about people's relationships or history.\n"
            "Drop: greetings, filler, and speculation. Write 4-8 dense bullet points, one fact each.\n\n"
            f"<conversation date=\"{human or 'unknown'}\">\n{transcript}\n</conversation>"
        )
        summary = self.llm.chat(
            [{"role": "system", "content": "You extract durable memory facts. Output only the bullets."},
             {"role": "user", "content": prompt}],
            max_tokens=500,
        )
        if not summary.strip():
            return []
        prefix = f"[{human}] " if human else ""
        return [
            Unit(
                id=_unit_id(request_id, 900, 0),
                user_id=user_id,
                session_id=session_id,
                seq=900,
                kind="summary",
                speaker="",
                created_ms=units[-1].created_ms if units else None,
                date_str=units[-1].date_str if units else "",
                text=f"{prefix}Session summary: {summary.strip()[:1500]}",
                prev_id=None,
                next_id=None,
                meta={"kind": "summary", "request_id": request_id},
            )
        ]


def _unit_id(request_id: str, index: int, part: int) -> str:
    digest = hashlib.sha1(f"{request_id}|{index}|{part}".encode("utf-8")).hexdigest()[:20]
    return f"m_{digest}"


def _readable_date(dt) -> str:
    if dt is None:
        return ""
    return f"{dt.day} {calendar.month_name[dt.month]} {dt.year}"
