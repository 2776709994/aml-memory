"""Unit tests for the two pieces of logic most likely to corrupt an answer.

1. Relative-time anchoring -- the judge compares dates at the gold granularity, so a
   too-precise anchor is as wrong as a missing one.
2. Turn rendering -- every returned memory line must name its speaker, because the
   answer prompt buckets memories per speaker and a bare "yeah, it was fun" is
   unusable to a frozen reader.
"""
from __future__ import annotations

import datetime as dt

from aml_memory.timeutil import anchor_relative_times, parse_datetime
from aml_memory.ingest import split_speaker


REF = dt.datetime(2023, 7, 26, 9, 30)


def test_yesterday_is_anchored_as_a_day():
    assert anchor_relative_times("I went yesterday.", REF) == "I went yesterday (25 July 2023)."


def test_week_and_month_use_coarse_granularity():
    assert "(the week of" in anchor_relative_times("I started last week.", REF)
    assert anchor_relative_times("I moved this month.", REF).endswith("(July 2023).")


def test_month_word_is_not_parsed_as_monday():
    """Regression: /mon[a-z]*/ matched "month" and shifted the anchor by a week."""
    anchored = anchor_relative_times("I will travel this month.", REF)
    assert "July 2023" in anchored


def test_last_year_is_a_year():
    assert anchor_relative_times("I graduated last year.", REF).endswith("(2022).")


def test_original_wording_is_preserved():
    text = "We met 3 weeks ago at the market."
    assert text.split()[1] == "met"
    assert "3 weeks ago" in anchor_relative_times(text, REF)


def test_parse_datetime_accepts_epoch_ms_and_iso():
    # Timestamps are treated as naive UTC (the platform's wire format is epoch ms;
    # interpreting it as UTC is the only choice that does not depend on the host's
    # timezone, and the day granularity we render is what the judge compares).
    stamp = int(dt.datetime(2023, 5, 8, 13, 56, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert parse_datetime(stamp) == dt.datetime(2023, 5, 8, 13, 56)
    assert parse_datetime("2023-05-08T13:56:00Z") == dt.datetime(2023, 5, 8, 13, 56)
    assert parse_datetime(None) is None


def test_speaker_prefix_wins_over_role():
    assert split_speaker("Caroline: I adopted a greyhound.", "assistant") == (
        "Caroline",
        "I adopted a greyhound.",
    )


def test_role_used_when_no_prefix():
    assert split_speaker("I adopted a greyhound.", "assistant") == (
        "Assistant",
        "I adopted a greyhound.",
    )


def test_url_is_not_mistaken_for_a_speaker():
    speaker, body = split_speaker("https://example.com/x: look at this", "user")
    assert speaker == "User"
    assert body.startswith("https://")
