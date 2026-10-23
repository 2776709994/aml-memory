"""Ranking knobs that do not depend on the corpus: role prior and exhaustive cap."""
from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


def _app(tmp_path, monkeypatch, **extra):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    monkeypatch.setenv("AML_DISABLE_AUTH", "1")
    for name, value in extra.items():
        monkeypatch.setenv(name, str(value))
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    return main.create_app(config.load_settings())


def _retriever(app_module_settings, prior: float):
    config = importlib.import_module("aml_memory.config")
    retrieve = importlib.import_module("aml_memory.retrieve")
    store_module = importlib.import_module("aml_memory.store")
    settings = config.load_settings()
    settings.assistant_prior = prior
    store = store_module.Store(settings.db_path, embed_model="hash")
    return retrieve.Retriever(settings, store, None, None, None), store_module


def test_assistant_turns_can_be_pushed_back(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        client.post("/add", json={
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": [
                {"role": "user", "content": "User: I practice yoga three times a week for my anxiety.",
                 "timestamp": 1683093600000},
                {"role": "assistant", "content": "Assistant: Yoga is a great idea for anxiety. Try three times a week.",
                 "timestamp": 1683093660000},
            ],
        })
    for prior, keep in ((1.0, True), (0.0, False)):
        config = importlib.import_module("aml_memory.config")
        retrieve = importlib.import_module("aml_memory.retrieve")
        store_module = importlib.import_module("aml_memory.store")
        settings = config.load_settings()
        settings.assistant_prior = prior
        store = store_module.Store(settings.db_path, embed_model="hash")
        retriever = retrieve.Retriever(settings, store, None, None, None)
        index = store.user_index("u1")
        plan = retriever.analyzer.analyze("how often do I practice yoga for anxiety")
        scores = {i: 1.0 for i in range(len(index.units))}
        boosted = retriever._boost(index, plan, scores)
        assistant_idx = [i for i, u in enumerate(index.units) if u.meta.get("role") == "assistant"]
        assert assistant_idx, [u.meta for u in index.units]
        for idx in assistant_idx:
            assert (boosted[idx] > 0) is keep, (prior, boosted)
