"""The reranker must degrade, not fail, and it must say so in its counters."""
from __future__ import annotations

from dataclasses import replace

import pytest

from aml_memory.config import load_settings
from aml_memory.rerank import Reranker


def _settings(**over) -> object:
    base = dict(
        rerank_provider="dashscope",
        rerank_api_key="sk-test",
        rerank_model="gte-rerank-v2",
        rerank_timeout_s=5,
    )
    base.update(over)
    return replace(load_settings(), **base)


def test_shorter_window_retry_saves_the_ranking(monkeypatch):
    reranker = Reranker(_settings())
    calls: list[int] = []

    def fake_http(query, docs):
        calls.append(len(docs))
        if len(docs) > 20:
            raise RuntimeError("read timeout")
        return [1.0 - 0.01 * i for i in range(len(docs))]

    monkeypatch.setattr(reranker, "_http_scores", fake_http)
    scores = reranker.score("q", [f"doc {i}" for i in range(60)])

    assert scores is not None and len(scores) == 60
    assert calls == [60, 20], calls
    # the 20 rescued documents keep their score, the untouched tail stays at 0.0
    assert scores[0] > 0.9
    assert scores[19] > 0.8
    assert all(value == 0.0 for value in scores[20:])
    assert reranker.stats()["retries"] == 1
    assert reranker.stats()["fallbacks"] == 0


def test_total_failure_falls_back_to_fusion_order(monkeypatch):
    reranker = Reranker(_settings())

    def always_boom(query, docs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(reranker, "_http_scores", always_boom)
    assert reranker.score("q", [f"doc {i}" for i in range(60)]) is None
    stats = reranker.stats()
    assert stats["failures"] == 1 and stats["fallbacks"] == 1
    assert stats["enabled"] == 1


def test_disabled_reranker_is_not_counted_as_a_failure():
    reranker = Reranker(_settings(rerank_provider="none"))
    assert reranker.score("q", ["doc"]) is None
    stats = reranker.stats()
    assert stats["calls"] == 0 and stats["failures"] == 0 and stats["enabled"] == 0


def test_single_document_is_not_retried(monkeypatch):
    reranker = Reranker(_settings())
    seen: list[int] = []

    def boom(query, docs):
        seen.append(len(docs))
        raise RuntimeError("boom")

    monkeypatch.setattr(reranker, "_http_scores", boom)
    assert reranker.score("q", ["only doc"]) is None
    assert seen == [1], seen
    assert reranker.stats()["fallbacks"] == 1
