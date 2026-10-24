"""Explicit retraction: the old half of a changed fact must not reach the reader twice."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _fresh(tmp_path: Path, monkeypatch, **env):
    from aml_memory.config import load_settings
    from aml_memory.main import create_app

    monkeypatch.setenv("AML_SKIP_DOTENV", "1")
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AML_DISABLE_AUTH", "1")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    monkeypatch.setenv("RERANK_PROVIDER", "none")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return create_app(load_settings())


def _add(client, request_id, text, *, user="u1", ts=1_690_000_000_000):
    response = client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user,
            "session_id": "s1",
            "messages": [{"role": "user", "content": f"User: {text}", "timestamp": ts}],
        },
    )
    assert response.status_code == 200, response.text
    return response


def test_detect_finds_retractions_and_ignores_ordinary_sentences():
    from aml_memory.governance import detect

    assert detect("User (6 May 2023): Forget that, my address is different now.").kind == "forget"
    assert detect("User (6 May 2023): I no longer work at Northwind.").kind == "supersede"
    assert detect("User (6 May 2023): Please ignore what I said about the trip.") is not None
    assert detect("User (6 May 2023): I forgot my keys at the office.") is None  # past tense
    assert detect("User (6 May 2023): I bought a new bicycle.") is None


def test_global_forget_is_left_alone():
    from aml_memory.governance import detect

    found = detect("User (6 May 2023): forget everything")
    assert found is not None and found.global_scope is True


def test_a_retraction_retires_the_older_turn(tmp_path, monkeypatch):
    app = _fresh(tmp_path, monkeypatch, AML_FORGET_MATCHING=1)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _add(client, "r1", "My home address is 12 Main Street, Bristol.")
        before = client.post("/search", json={"query": "home address Bristol", "user_id": "u1", "top_k": 10})
        assert any("12 Main Street" in item["content"] for item in before.json()["data"])

        _add(client, "r2", "Forget my home address, that was the old place.", ts=1_690_000_100_000)
        after = client.post("/search", json={"query": "home address Bristol", "user_id": "u1", "top_k": 10})
        contents = [item["content"] for item in after.json()["data"]]
        assert not any("12 Main Street" in text for text in contents), contents
        assert any("Forget my home address" in text for text in contents), contents
        assert app.state.store.count_superseded() == 1
        assert app.state.store.stats()["superseded"] == 1


def test_retraction_matching_is_off_by_default(tmp_path, monkeypatch):
    app = _fresh(tmp_path, monkeypatch)  # AML_FORGET_MATCHING defaults to false
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _add(client, "r1", "My home address is 12 Main Street, Bristol.")
        _add(client, "r2", "Forget my home address, that was the old place.", ts=1_690_000_100_000)
        assert app.state.store.count_superseded() == 0


def test_an_unrelated_turn_retires_nothing(tmp_path, monkeypatch):
    app = _fresh(tmp_path, monkeypatch, AML_FORGET_MATCHING=1)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _add(client, "r1", "My home address is 12 Main Street, Bristol.")
        # A retraction that matches nothing must not delete something at random.
        _add(client, "r2", "Forget that I mentioned the marathon, it was cancelled.",
             ts=1_690_000_100_000)
        assert app.state.store.count_superseded() == 0


def test_retraction_never_crosses_users(tmp_path, monkeypatch):
    app = _fresh(tmp_path, monkeypatch, AML_FORGET_MATCHING=1)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _add(client, "r1", "My home address is 12 Main Street, Bristol.", user="alice")
        _add(client, "r2", "Forget my home address, that was the old place.",
             user="bob", ts=1_690_000_100_000)
        assert app.state.store.count_superseded() == 0
        found = client.post("/search", json={"query": "home address", "user_id": "alice", "top_k": 5})
        assert any("12 Main Street" in item["content"] for item in found.json()["data"])
