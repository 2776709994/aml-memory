"""Dataset adapters: normalise public memory benchmarks into AML Add/Search shapes.

Only public datasets with released question/evidence annotations are supported here;
the AML private bundles are never touched.  The point is local tuning: for every
question we know which dialogue turns support the answer, so retrieval quality can
be measured without the platform's answer model or judge.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def to_epoch_ms(when: _dt.datetime) -> int:
    """Encode a naive wall-clock time as epoch ms without shifting the clock.

    ``datetime.timestamp()`` applies the host timezone and the service decodes epoch ms
    as naive UTC, so a 20:00 session would silently land on the next day.
    ``calendar.timegm`` keeps the dataset's wall clock intact end to end.
    """
    return calendar.timegm(when.replace(tzinfo=None).timetuple()) * 1000


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str = ""
    timestamp: _dt.datetime | None = None


@dataclass
class Session:
    index: int
    when: _dt.datetime | None
    turns: list[Turn]


@dataclass
class QA:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    category: str = ""
    options: list[str] = field(default_factory=list)
    adversarial: str = ""


@dataclass
class Instance:
    sample_id: str
    speakers: list[str]
    sessions: list[Session]
    qas: list[QA]


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_session_time(text: str) -> _dt.datetime | None:
    """'1:56 pm on 8 May, 2023' -> datetime(2023, 5, 8, 13, 56).

    Also accepts LongMemEval's ISO-ish form: '2023/05/30 (Tue) 23:40'.
    """
    if not text:
        return None
    raw = str(text)
    iso = re.search(
        r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:\s*\([A-Za-z]+\))?(?:\s*(\d{1,2}):(\d{2}))?",
        raw,
    )
    if iso:
        try:
            return _dt.datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)),
                                int(iso.group(4) or 0), int(iso.group(5) or 0))
        except ValueError:
            return None
    match = re.search(
        r"(?:(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*)?(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})",
        str(text),
        re.I,
    )
    if not match:
        return None
    hour = int(match.group(1) or 12)
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    month = _MONTHS.get(match.group(5).lower())
    if not month:
        return None
    try:
        return _dt.datetime(int(match.group(6)), month, int(match.group(4)), hour, minute)
    except ValueError:
        return None


def load_locomo(path: str | Path) -> list[Instance]:
    """LoCoMo-10: conversation `session_N` lists + `qa` with `evidence` dia_ids."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    instances: list[Instance] = []
    for sample in raw:
        sample_id = str(sample.get("sample_id") or sample.get("id") or "sample")
        conversation = sample.get("conversation") or {}
        speaker_a = conversation.get("speaker_a") or "Speaker A"
        speaker_b = conversation.get("speaker_b") or "Speaker B"

        session_keys = [
            key for key in conversation
            if re.fullmatch(r"session_\d+", key)
        ]
        session_keys.sort(key=lambda key: int(key.split("_")[1]))
        sessions: list[Session] = []
        for key in session_keys:
            number = int(key.split("_")[1])
            when = parse_session_time(conversation.get(f"{key}_date_time", ""))
            turns: list[Turn] = []
            for turn in conversation.get(key) or []:
                text = str(turn.get("text") or "").strip()
                caption = str(turn.get("blip_caption") or "").strip()
                if caption:
                    text = f"{text} [shares a photo: {caption}]".strip()
                if not text:
                    continue
                turns.append(
                    Turn(
                        speaker=str(turn.get("speaker") or speaker_a),
                        text=text,
                        dia_id=str(turn.get("dia_id") or ""),
                        timestamp=when,
                    )
                )
            sessions.append(Session(index=number, when=when, turns=turns))

        qas: list[QA] = []
        for item in sample.get("qa") or []:
            evidence = item.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            category = item.get("category")
            qas.append(
                QA(
                    question=str(item.get("question") or "").strip(),
                    answer=str(item.get("answer") or "").strip(),
                    evidence=[str(e) for e in evidence],
                    category=_locomo_category(category),
                    adversarial=str(item.get("adversarial_answer") or ""),
                )
            )
        instances.append(
            Instance(
                sample_id=sample_id,
                speakers=[speaker_a, speaker_b],
                sessions=sessions,
                qas=[qa for qa in qas if qa.question],
            )
        )
    return instances


def load_locomo_refined(path: str | Path) -> list[Instance]:
    """LoCoMo-Refined: `conversations.jsonl` + `questions.jsonl`.

    The refined release is the closest public analogue of the AML payload: turns carry
    an explicit `speaker` name and `role`, evidence comes pre-joined as `dia_id`s, and
    answers are lists.  `path` may be the directory or either of the two files.
    """
    target = Path(path)
    if target.is_dir():
        conv_path = target / "conversations.jsonl"
        qa_path = target / "questions.jsonl"
    elif target.name.startswith("questions"):
        conv_path = target.with_name("conversations.jsonl")
        qa_path = target
    else:
        conv_path = target
        qa_path = target.with_name("questions.jsonl")
    conversations = [
        json.loads(line) for line in conv_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_sample: dict[str, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(str(row.get("sample_id") or ""), []).append(row)

    instances: list[Instance] = []
    for conversation in conversations:
        sample_id = str(conversation.get("sample_id") or "sample")
        speaker_a = str(conversation.get("speaker_a") or "Speaker A")
        speaker_b = str(conversation.get("speaker_b") or "Speaker B")
        sessions: list[Session] = []
        for session in conversation.get("sessions") or []:
            when = parse_session_time(str(session.get("date_time") or ""))
            turns: list[Turn] = []
            for message in session.get("messages") or []:
                text = str(message.get("text") or "").strip()
                caption = str(message.get("blip_caption") or "").strip()
                if caption:
                    text = f"{text} [shares a photo: {caption}]".strip()
                if not text:
                    continue
                turns.append(
                    Turn(
                        speaker=str(message.get("speaker") or message.get("role") or speaker_a),
                        text=text,
                        dia_id=str(message.get("dia_id") or ""),
                        timestamp=when,
                    )
                )
            if turns:
                sessions.append(
                    Session(index=int(session.get("session_index") or len(sessions) + 1),
                            when=when, turns=turns)
                )

        qas: list[QA] = []
        for row in by_sample.get(sample_id, []):
            evidence = row.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            answer = row.get("answer")
            if isinstance(answer, list):
                # The official pipeline renders a list answer through `memory_text`,
                # which joins with newlines.  Joining with "; " instead would make the
                # judge read the list as one comma-list requiring EVERY item and mark
                # a correct single alternative as WRONG.
                answer = "\n".join(str(a) for a in answer)
            qas.append(
                QA(
                    question=str(row.get("question") or "").strip(),
                    answer=str(answer or "").strip(),
                    evidence=[str(e) for e in evidence],
                    category=_locomo_category(row.get("category")),
                )
            )
        instances.append(
            Instance(
                sample_id=sample_id,
                speakers=[speaker_a, speaker_b],
                sessions=sessions,
                qas=[qa for qa in qas if qa.question],
            )
        )
    return instances


def _locomo_category(category: Any) -> str:
    mapping = {
        1: "multihop",
        2: "temporal",
        3: "open_domain",
        4: "single_hop",
        5: "adversarial",
        "1": "multihop", "2": "temporal", "3": "open_domain", "4": "single_hop", "5": "adversarial",
    }
    return mapping.get(category, f"cat_{category}")


def load_longmemeval(path: str | Path) -> list[Instance]:
    """LongMemEval (oracle / S / M): one instance per question, one session per haystack entry.

    `answer_session_ids` is session-level ground truth, so each answer session is
    represented by its FIRST turn id: R@k then answers "did we reach the right
    session", which is the metric LongMemEval itself reports.
    """
    raw = Path(path).read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    instances: list[Instance] = []
    for row in rows:
        sessions_raw = row.get("haystack_sessions") or row.get("sessions") or []
        dates = row.get("haystack_dates") or []
        ids = row.get("haystack_session_ids") or []
        sessions: list[Session] = []
        first_turn: dict[str, str] = {}
        for index, session in enumerate(sessions_raw):
            when = parse_session_time(dates[index]) if index < len(dates) else None
            sid = str(ids[index]) if index < len(ids) else f"s{index}"
            turns: list[Turn] = []
            for turn_index, turn in enumerate(session):
                text = str(turn.get("content") or "")
                if not text.strip():
                    continue
                dia = f"{sid}:{turn_index}"
                if sid not in first_turn:
                    first_turn[sid] = dia
                turns.append(
                    Turn(speaker=str(turn.get("role") or "user"), text=text,
                         dia_id=dia, timestamp=when)
                )
            sessions.append(Session(index=index, when=when, turns=turns))
        question = str(row.get("question") or "").strip()
        answer = row.get("answer")
        if isinstance(answer, list):
            answer = "; ".join(str(a) for a in answer)
        evidence = [first_turn[str(e)] for e in row.get("answer_session_ids") or []
                    if str(e) in first_turn]
        instances.append(
            Instance(
                sample_id=str(row.get("question_id") or row.get("id") or f"lme-{len(instances)}"),
                speakers=["user", "assistant"],
                sessions=sessions,
                qas=[QA(question=question, answer=str(answer or ""), evidence=evidence,
                        category=str(row.get("question_type") or ""))] if question else [],
            )
        )
    return instances


def load_beam(path: str | Path) -> list[Instance]:
    """BEAM (github.com/mohammadtavakoli78/BEAM): 20 chats x 10 abilities x 2 questions.

    Each chat is three batches of turns; every turn carries its own ``time_anchor``
    ("March-15-2024") rather than a session-level date, so the timestamps here are
    per-turn.  The abilities map closely onto the leaderboard's capability axes
    (knowledge_update / contradiction_resolution -> governance, instruction_following
    -> rules, abstention -> epistemic safety, temporal_reasoning / event_ordering ->
    time), which makes this the closest public proxy for a Full run.
    """
    root = Path(path)
    convs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    instances: list[Instance] = []
    for conv in convs:
        chat_file = conv / "chat.json"
        prob_file = conv / "probing_questions" / "probing_questions.json"
        if not chat_file.exists() or not prob_file.exists():
            continue
        batches = json.loads(chat_file.read_text(encoding="utf-8"))
        sessions: list[Session] = []
        speakers: list[str] = []
        for batch in batches:
            turns: list[Turn] = []
            for group in batch.get("turns") or []:
                for raw in group:
                    role = str(raw.get("role") or "user")
                    text = str(raw.get("content") or "").strip()
                    if not text:
                        continue
                    speaker = "User" if role == "user" else "Assistant"
                    if speaker not in speakers:
                        speakers.append(speaker)
                    when = _beam_time(raw.get("time_anchor"))
                    turns.append(
                        Turn(
                            speaker=speaker,
                            text=text,
                            dia_id=str(raw.get("index") or len(turns)),
                            timestamp=when,
                        )
                    )
            if turns:
                sessions.append(Session(index=len(sessions) + 1, when=turns[0].timestamp, turns=turns))
        probes = json.loads(prob_file.read_text(encoding="utf-8"))
        qas: list[QA] = []
        for ability, items in (probes.items() if isinstance(probes, dict) else []):
            for item in items or []:
                question = str(item.get("question") or "").strip()
                if not question:
                    continue
                gold = ""
                for field_name in ("answer", "ideal_answer", "ideal_response", "ideal_summary"):
                    value = item.get(field_name)
                    if value:
                        gold = value if isinstance(value, str) else "; ".join(str(v) for v in value)
                        break
                rubric = item.get("rubric")
                if not gold and isinstance(rubric, list):
                    gold = "\n".join(str(part) for part in rubric)
                category = ability if isinstance(ability, str) else str(ability)
                qas.append(
                    QA(
                        question=question,
                        answer=str(gold),
                        evidence=[],
                        category=category,
                        adversarial=str(item.get("abstention_type") or ""),
                    )
                )
        if sessions and qas:
            instances.append(
                Instance(sample_id=f"beam-{conv.name}", speakers=speakers or ["User"], sessions=sessions, qas=qas)
            )
    return instances


def _beam_time(value: Any) -> _dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%B-%d-%Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return parse_session_time(text)


LOADERS = {
    "locomo": load_locomo,
    "locomo-refined": load_locomo_refined,
    "locomo_refined": load_locomo_refined,
    "longmemeval": load_longmemeval,
    "beam": load_beam,
}


def load(kind: str, path: str | Path) -> list[Instance]:
    return LOADERS[kind](path)


def iter_requests(instance: Instance, *, chunk_messages: int = 20,
                  speaker_prefix: bool = True) -> Iterator[dict]:
    """Yield AML `/add` payloads, chunked the way the platform chunks them.

    `speaker_prefix=True` renders each turn as ``"Name: text"``, mirroring the official
    LoCoMo-Refined flattening; the service splits the name back out, so stored units are
    labelled with the real speaker instead of a generic role.
    """
    for session in instance.sessions:
        turns = session.turns
        for start in range(0, len(turns), chunk_messages):
            chunk = turns[start:start + chunk_messages]
            base = session.when or _dt.datetime(2023, 1, 1)
            messages = []
            for offset, turn in enumerate(chunk):
                # LoCoMo only records a session-level timestamp; spread turns by minute.
                when = turn.timestamp or base
                stamp = when + _dt.timedelta(minutes=start + offset)
                text = f"{turn.speaker}: {turn.text}" if speaker_prefix else turn.text
                messages.append(
                    {
                        "role": "user" if turn.speaker == instance.speakers[0] else "assistant",
                        "content": text,
                        "timestamp": to_epoch_ms(stamp),
                    }
                )
                messages[-1]["_speaker"] = turn.speaker
                messages[-1]["_dia_id"] = turn.dia_id
            yield {
                "request_id": f"{instance.sample_id}:s{session.index}:c{start // chunk_messages}",
                "user_id": instance.sample_id,
                "session_id": f"{instance.sample_id}:s{session.index}",
                "messages": messages,
            }
