"""Contract tests: the three endpoints the AML platform calls, plus isolation rules.

These run with the offline `hash` embedder, so they need no network and no API key:

    pytest -q tests/
"""
from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    monkeypatch.setenv("AML_DISABLE_AUTH", "1")
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    app = main.create_app(config.load_settings())
    with TestClient(app) as test_client:
        yield test_client


def _add(client, *, request_id="r1", user_id="u1", session_id="s1", messages=None):
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages
        or [
            {"role": "user", "content": "Caroline: I adopted a greyhound named Pepper in May 2023.",
             "timestamp": 1683093600000},
            {"role": "assistant", "content": "Melanie: Lovely! I signed up for a pottery class yesterday.",
             "timestamp": 1683093660000},
        ],
    }
    return client.post("/add", json=payload)


def test_health_is_public_and_cheap(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_echoes_identifiers_exactly(client):
    response = _add(client, request_id="req-42", user_id="alice", session_id="sess-7")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "success": True,
        "request_id": "req-42",
        "user_id": "alice",
        "session_id": "sess-7",
    }


def test_add_is_immediately_searchable(client):
    _add(client)
    response = client.post("/search", json={"query": "What is the greyhound called?", "user_id": "u1", "top_k": 100})
    assert response.status_code == 200
    items = response.json()["data"]
    assert items, "memories must be retrievable right after /add returns"
    assert "Pepper" in " ".join(item["content"] for item in items)


def test_search_response_shape(client):
    _add(client)
    items = client.post("/search", json={"query": "pottery", "user_id": "u1", "top_k": 100}).json()["data"]
    assert len(items) <= 100
    for item in items:
        assert set(item) <= {"id", "content", "score", "created_at"}
        assert item["id"] and item["content"]


def test_add_is_idempotent_per_request_id(client):
    _add(client, request_id="dup")
    stats_before = client.get("/admin/stats").json()
    response = _add(client, request_id="dup")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert client.get("/admin/stats").json()["units"] == stats_before["units"]


def test_users_are_isolated(client):
    _add(client, request_id="a", user_id="alice")
    _add(client, request_id="b", user_id="bob", messages=[
        {"role": "user", "content": "Bob: my cat is called Tofu.", "timestamp": 1683093600000}
    ])
    alice = client.post("/search", json={"query": "cat name", "user_id": "alice", "top_k": 10}).json()["data"]
    bob = client.post("/search", json={"query": "cat name", "user_id": "bob", "top_k": 10}).json()["data"]
    assert all("Tofu" not in item["content"] for item in alice)
    assert any("Tofu" in item["content"] for item in bob)


def test_unknown_user_returns_empty_list(client):
    response = client.post("/search", json={"query": "anything", "user_id": "nobody", "top_k": 100})
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_frontend_style_payload_extras_are_ignored(client):
    payload = {
        "request_id": "r9",
        "user_id": "u9",
        "session_id": "s9",
        "app_id": "x",
        "async_mode": True,
        "metadata": {"a": 1},
        "messages": [{"role": "user", "content": "hi there", "timestamp": 1683093600000, "name": "Jo"}],
    }
    assert client.post("/add", json=payload).status_code == 200


def test_bad_payload_is_rejected_without_retry(client):
    response = client.post("/add", json={"user_id": "u1", "messages": []})
    assert 400 <= response.status_code < 500


def test_mcq_options_are_accepted(client):
    _add(client)
    response = client.post(
        "/search",
        json={"query": "Which pet does Caroline own?", "options": ["(a) a cat", "(b) a dog"],
              "user_id": "u1", "top_k": 50},
    )
    assert response.status_code == 200


def test_auth_required_when_key_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("AML_API_KEY", "s3cret")
    monkeypatch.setenv("AML_DISABLE_AUTH", "0")
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    with TestClient(main.create_app(config.load_settings())) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/search", json={"query": "x", "user_id": "u", "top_k": 5}).status_code == 401
        ok = client.post("/search", json={"query": "x", "user_id": "u", "top_k": 5},
                         headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        wrong = client.post("/search", json={"query": "x", "user_id": "u", "top_k": 5},
                            headers={"Authorization": "Bearer nope"})
        assert wrong.status_code in (401, 403)
