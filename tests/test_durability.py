"""Durability tests: the guarantees that a 1-2 day Full run depends on.

A Full run retries a timed-out Add with the *same* request_id (up to 32 times) and
may restart the process while doing so, so two invariants have to hold:

1. a request_id is claimed exactly once, even by concurrent retries;
2. the claim survives a process restart (it lives in SQLite, not in RAM).

They run with the offline `hash` embedder: no network, no API key.
"""
from __future__ import annotations

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient


def _fresh_app(tmp_path, monkeypatch, *, api_key: str = ""):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    monkeypatch.setenv("AML_DISABLE_AUTH", "1" if not api_key else "0")
    monkeypatch.setenv("AML_API_KEY", api_key)
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    return main.create_app(config.load_settings())


def _payload(request_id: str, text: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": "u1",
        "session_id": "s1",
        "messages": [{"role": "user", "content": text, "timestamp": 1683093600000}],
    }


def _count_units(client: TestClient, headers: dict | None = None) -> int:
    stats = client.get("/admin/stats", headers=headers or {}).json()
    assert "units" in stats, stats
    return int(stats["units"])


def test_replay_after_restart_is_idempotent(tmp_path, monkeypatch):
    app1 = _fresh_app(tmp_path, monkeypatch, api_key="secret")
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app1) as client:
        first = client.post("/add", json=_payload("req-mem", "Alice: my sister lives in Kyoto."),
                            headers=headers)
        assert first.status_code == 200
        before = _count_units(client, headers)
        assert before > 0

    # New process, same data dir: the client object and caches are gone.
    app2 = _fresh_app(tmp_path, monkeypatch, api_key="secret")
    with TestClient(app2) as client:
        replay = client.post("/add", json=_payload("req-mem", "Completely different text."),
                             headers=headers)
        assert replay.status_code == 200, replay.text
        assert replay.json() == {"success": True, "request_id": "req-mem",
                                 "user_id": "u1", "session_id": "s1"}
        assert _count_units(client, headers) == before, "replayed request wrote a second copy"
        hits = client.post("/search", json={"query": "sister Kyoto", "user_id": "u1",
                                            "top_k": 10}, headers=headers).json()["data"]
        assert hits, "memory did not survive the restart"
        assert any("Kyoto" in hit["content"] for hit in hits)
        assert not any("Completely different" in hit["content"] for hit in hits)


def test_concurrent_retries_write_once(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        payload = _payload("req-race", "Bob: I planted tomatoes last spring.")

        def fire(_: int) -> int:
            return client.post("/add", json=payload).status_code

        with ThreadPoolExecutor(max_workers=6) as pool:
            codes = list(pool.map(fire, range(6)))
        assert set(codes) == {200}, codes
        assert _count_units(client) == 1

        hits = client.post("/search", json={"query": "tomatoes", "user_id": "u1",
                                            "top_k": 10}).json()["data"]
        assert len(hits) == 1, f"duplicate memories returned: {len(hits)}"


def test_add_without_timestamp_is_accepted(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        payload = {
            "request_id": "req-nots",
            "user_id": "u1",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "Dana: I moved to Lisbon in March."}],
        }
        assert client.post("/add", json=payload).status_code == 200
        hits = client.post("/search", json={"query": "where does Dana live", "user_id": "u1",
                                            "top_k": 10}).json()["data"]
        assert hits and "Lisbon" in hits[0]["content"]


def test_streaming_interleave(tmp_path, monkeypatch):
    """Add and Search alternate in event order, as the streaming track does."""
    app = _fresh_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        for step in range(3):
            body = {"request_id": f"req-s{step}", "user_id": "u1", "session_id": "s1",
                    "messages": [{"role": "user",
                                  "content": f"Erin: day {step} note about topic{step}."}]}
            assert client.post("/add", json=body).status_code == 200
            hits = client.post("/search", json={"query": f"topic{step}", "user_id": "u1",
                                                "top_k": 5}).json()["data"]
            assert hits, f"step {step} memory not searchable immediately"
            assert f"topic{step}" in hits[0]["content"]


def test_search_never_returns_other_users(tmp_path, monkeypatch):
    app = _fresh_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        client.post("/add", json={"request_id": "req-a", "user_id": "ua", "session_id": "s",
                                  "messages": [{"role": "user", "content": "Frank: secret plan alpha."}]})
        client.post("/add", json={"request_id": "req-b", "user_id": "ub", "session_id": "s",
                                  "messages": [{"role": "user", "content": "Grace: secret plan beta."}]})
        hits = client.post("/search", json={"query": "secret plan", "user_id": "ua",
                                            "top_k": 10}).json()["data"]
        assert all("beta" not in hit["content"] for hit in hits)
        assert any("alpha" in hit["content"] for hit in hits)


@pytest.mark.parametrize("bad", [
    {"messages": []},
    {"request_id": "x"},
    {"user_id": "u1"},
])
def test_missing_required_fields_are_4xx(tmp_path, monkeypatch, bad):
    app = _fresh_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        assert 400 <= client.post("/add", json=bad).status_code < 500

