"""Multiple-choice handling.

The platform sends `options` on MCQ suites (PersonaMem, ScriptMem).  The question on
its own ("what would you suggest?") carries almost no retrievable content, so the
option texts are added as one extra query variant — but only when the switch is on,
because the same trick also pulls in turns that merely resemble a distractor.
"""

from __future__ import annotations

import pytest

from aml_memory.config import load_settings
from aml_memory.query import QueryAnalyzer


def _analyzer(monkeypatch: pytest.MonkeyPatch, option_query: bool) -> QueryAnalyzer:
    monkeypatch.setenv("AML_OPTION_QUERY", "1" if option_query else "0")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    return QueryAnalyzer(load_settings(), llm=None)


OPTIONS = ["Go for a morning run", "Read a novel in the garden", "Call your sister"]


def test_options_become_a_query_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _analyzer(monkeypatch, True).analyze("What should I do this weekend?", OPTIONS)
    variants = plan.variants()
    assert len(variants) > 1
    assert any("Call your sister" in variant for variant in variants)


def test_the_switch_removes_the_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _analyzer(monkeypatch, False).analyze("What should I do this weekend?", OPTIONS)
    variants = plan.variants()
    assert not any("Call your sister" in variant for variant in variants)
    # The question itself always remains a variant, so retrieval is not crippled.
    assert plan.variants()[0] == "What should I do this weekend?"


def test_options_do_not_leak_into_the_question_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """An option listing a date must not make the plan think the question is temporal."""
    plan = _analyzer(monkeypatch, True).analyze(
        "Which plan fits me best?",
        ["Attend the fair on 7 May 2023", "Stay home and rest", "Visit the museum"],
    )
    assert plan.raw == "Which plan fits me best?"
    assert plan.source in {"rules", "rules+llm", "llm"}
