"""Date parsing plus relative-time anchoring.

Why this matters: benchmark gold answers are frequently expressed relative to the
moment of an utterance ("yesterday", "last year"), while the question is asked
long after.  We keep the original surface form AND append the resolved absolute
date, so the frozen answer model can reason either way without losing the
evidence wording.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import re

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def ms_to_datetime(ms: object) -> _dt.datetime | None:
    """Unix milliseconds (or seconds) -> naive UTC datetime."""
    if ms is None:
        return None
    try:
        value = float(ms)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 1e11:  # milliseconds
        value /= 1000.0
    try:
        # naive UTC on purpose: every other helper in this module works on naive
        # UTC datetimes, and mixing aware/naive values breaks date arithmetic.
        return _dt.datetime.fromtimestamp(value, _dt.timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        return None


def parse_datetime(value: object) -> _dt.datetime | None:
    """Best-effort timestamp parser: epoch ms/s, ISO-8601, or common date formats."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return ms_to_datetime(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{10}(\.\d+)?", text) or re.fullmatch(r"\d{13}", text):
        return ms_to_datetime(text)
    cleaned = text.replace("Z", "+00:00")
    try:
        parsed = _dt.datetime.fromisoformat(cleaned)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d %B, %Y", "%B %d, %Y",
        "%b %d, %Y", "%d %b %Y", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y %B %d",
        "%d %B %Y", "%B %d %Y",
    ):
        try:
            return _dt.datetime.strptime(text[: len(fmt) + 4].strip(), fmt)
        except ValueError:
            continue
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        try:
            return _dt.datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def day_str(dt: _dt.datetime | None) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d")


def human_date(dt: _dt.datetime | None) -> str:
    """'7 May 2023 (Sunday)' — readable by the answer model, cheap in tokens."""
    if dt is None:
        return ""
    return f"{dt.day} {calendar.month_name[dt.month]} {dt.year} ({calendar.day_name[dt.weekday()]})"


def _safe_date(year: int, month: int, day: int) -> _dt.datetime | None:
    try:
        return _dt.datetime(year, month, day)
    except ValueError:
        return None


def _shift_months(dt: _dt.datetime, months: int) -> _dt.datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


_REL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(day before yesterday)\b", re.I), "day_before_yesterday"),
    (re.compile(r"\b(yesterday|last night)( morning| afternoon| evening)?\b", re.I), "yesterday"),
    (re.compile(r"\btoday( morning| afternoon| evening)?\b", re.I), "today"),
    (re.compile(r"\btomorrow( morning| afternoon| evening)?\b", re.I), "tomorrow"),
    (re.compile(r"\bthe day after tomorrow\b", re.I), "day_after_tomorrow"),
    (re.compile(r"\b(this morning|this afternoon|this evening|tonight)\b", re.I), "today"),
    (re.compile(r"\b(last|past) (week|weekend)\b", re.I), "last_week"),
    (re.compile(r"\bthis (week|weekend)\b", re.I), "this_week"),
    (re.compile(r"\bnext (week|weekend)\b", re.I), "next_week"),
    (re.compile(r"\b(earlier this week)\b", re.I), "last_week"),
    (re.compile(r"\b(last|past) month\b", re.I), "last_month"),
    (re.compile(r"\bthis month\b", re.I), "this_month"),
    (re.compile(r"\bnext month\b", re.I), "next_month"),
    (re.compile(r"\b(last|past) year\b", re.I), "last_year"),
    (re.compile(r"\bthis year\b", re.I), "this_year"),
    (re.compile(r"\bnext year\b", re.I), "next_year"),
    (re.compile(r"\b(\d{1,2}) (day|days) ago\b", re.I), "n_days_ago"),
    (re.compile(r"\b(\d{1,2}) (week|weeks) ago\b", re.I), "n_weeks_ago"),
    (re.compile(r"\b(\d{1,2}) (month|months) ago\b", re.I), "n_months_ago"),
    (re.compile(r"\b(\d{1,2}) (year|years) ago\b", re.I), "n_years_ago"),
    (re.compile(r"\ba (day|week|month|year) ago\b", re.I), "one_ago"),
    (
        re.compile(
            r"\b(last|on|this|previous|past)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        ),
        "weekday",
    ),
]

# How precise the injected anchor should be.  Over-precise anchors are actively
# harmful: the judge compares at the gold answer's granularity, so a memory line
# that turns "this month" into "26 June 2023" invites a WRONG answer where the
# gold is "July 2023".
_GRANULARITY = {
    "yesterday": "day",
    "day_before_yesterday": "day",
    "today": "day",
    "tomorrow": "day",
    "day_after_tomorrow": "day",
    "n_days_ago": "day",
    "weekday": "day",
    "last_week": "week",
    "this_week": "week",
    "next_week": "week",
    "n_weeks_ago": "week",
    "last_month": "month",
    "this_month": "month",
    "next_month": "month",
    "n_months_ago": "month",
    "last_year": "year",
    "this_year": "year",
    "next_year": "year",
    "n_years_ago": "year",
}


def _resolve(kind: str, match: re.Match[str], ref: _dt.datetime) -> _dt.datetime | None:
    if kind == "yesterday":
        return ref - _dt.timedelta(days=1)
    if kind == "day_before_yesterday":
        return ref - _dt.timedelta(days=2)
    if kind == "today":
        return ref
    if kind == "tomorrow":
        return ref + _dt.timedelta(days=1)
    if kind == "day_after_tomorrow":
        return ref + _dt.timedelta(days=2)
    if kind == "last_week":
        return ref - _dt.timedelta(days=7)
    if kind == "this_week":
        return ref
    if kind == "next_week":
        return ref + _dt.timedelta(days=7)
    if kind == "last_month":
        return _shift_months(ref, -1)
    if kind == "this_month":
        return ref
    if kind == "next_month":
        return _shift_months(ref, 1)
    if kind == "last_year":
        return _shift_months(ref, -12)
    if kind == "this_year":
        return ref
    if kind == "next_year":
        return _shift_months(ref, 12)
    if kind == "n_days_ago":
        return ref - _dt.timedelta(days=int(match.group(1)))
    if kind == "n_weeks_ago":
        return ref - _dt.timedelta(weeks=int(match.group(1)))
    if kind == "n_months_ago":
        return _shift_months(ref, -int(match.group(1)))
    if kind == "n_years_ago":
        return _shift_months(ref, -12 * int(match.group(1)))
    if kind == "one_ago":
        unit = match.group(1).lower()
        if unit == "day":
            return ref - _dt.timedelta(days=1)
        if unit == "week":
            return ref - _dt.timedelta(weeks=1)
        if unit == "month":
            return _shift_months(ref, -1)
        return _shift_months(ref, -12)
    if kind == "weekday":
        word = match.group(2).lower()
        target = _WEEKDAYS.get(word)
        if target is None:
            return None
        delta = (ref.weekday() - target) % 7
        if delta == 0:
            delta = 7
        return ref - _dt.timedelta(days=delta)
    return None


def _anchor_label(kind: str, resolved: _dt.datetime) -> str:
    granularity = _GRANULARITY.get(kind, "day")
    if granularity == "day":
        return f"{resolved.day} {calendar.month_name[resolved.month]} {resolved.year}"
    if granularity == "week":
        return f"the week of {resolved.day} {calendar.month_name[resolved.month]} {resolved.year}"
    if granularity == "month":
        return f"{calendar.month_name[resolved.month]} {resolved.year}"
    return f"{resolved.year}"


def anchor_relative_times(text: str, ref: _dt.datetime | None, max_hits: int = 4) -> str:
    """Append resolved absolute dates to relative time expressions.

    ``"yesterday"`` becomes ``"yesterday (6 May 2023)"``, while coarser
    expressions stay coarse: ``"this month"`` becomes ``"this month (May 2023)"``.
    The original wording is preserved because the judge sometimes requires a
    relative-form answer.
    """
    if not text or ref is None:
        return text
    out = text
    hits = 0
    for pattern, kind in _REL_PATTERNS:
        def repl(match: re.Match[str], _kind: str = kind) -> str:
            nonlocal hits
            if hits >= max_hits:
                return match.group(0)
            literal = match.group(0)
            if "(" in out[match.end(): match.end() + 2]:
                return literal
            resolved = _resolve(_kind, match, ref)
            if resolved is None:
                return literal
            hits += 1
            return f"{literal} ({_anchor_label(_kind, resolved)})"

        out = pattern.sub(repl, out)
    return out



_MONTH_DAY = re.compile(
    r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b",
    re.I,
)


_HEADER_HINT = re.compile(r"(?:session\s*\d|date\s*[:\-–—]|timestamp|started\s+at|began\s+at)", re.I)
_DAY_FIRST = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s*(\d{4})\b")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\b", re.I)


def header_date(text: str) -> _dt.datetime | None:
    """Recover the session date from a header embedded in the message text.

    The platform may send `/add` messages without a `timestamp` while the text
    itself carries '--- Session 1 started at 1:56 pm on 8 May, 2023 ---'.  Losing
    that date costs every temporal question in the session, so it is worth
    recovering — but only from an explicit header, never from a date mentioned in
    passing ("I met her on 3 May" is an event, not the session's clock).
    """
    if not text or not _HEADER_HINT.search(text):
        return None
    for match in _DAY_FIRST.finditer(text):
        month = MONTHS.get(match.group(2).lower())
        if not month:
            continue
        resolved = _safe_date(int(match.group(3)), month, int(match.group(1)))
        if resolved is None:
            continue
        clock = None
        for candidate in _CLOCK.finditer(text[: match.end()]):
            clock = candidate
        if clock is not None:
            hour, minute = int(clock.group(1)), int(clock.group(2))
            meridiem = clock.group(3).lower()
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            if 0 <= hour < 24 and 0 <= minute < 60:
                resolved = resolved.replace(hour=hour, minute=minute)
        return resolved
    return None


def extract_dates(text: str, ref: _dt.datetime | None = None) -> list[_dt.datetime]:
    """Dates mentioned in a question — used for temporal filtering of candidates."""
    found: list[_dt.datetime] = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text or ""):
        dt = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if dt:
            found.append(dt)
    for match in _MONTH_DAY.finditer(text or ""):
        month = MONTHS.get(match.group(1).lower())
        if not month:
            continue
        year = int(match.group(3)) if match.group(3) else (ref.year if ref else None)
        if not year:
            continue
        dt = _safe_date(year, month, int(match.group(2)))
        if dt:
            found.append(dt)
    for match in re.finditer(r"\b(?:in|since|during|by)\s+(\d{4})\b", text or ""):
        dt = _safe_date(int(match.group(1)), 1, 1)
        if dt:
            found.append(dt)
    for match in re.finditer(r"\b(19|20)\d{2}\b", text or ""):
        year = int(match.group(0))
        if not any(d.year == year for d in found):
            dt = _safe_date(year, 1, 1)
            if dt:
                found.append(dt)
    return found
