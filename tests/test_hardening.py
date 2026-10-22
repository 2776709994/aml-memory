"""Hardening tests: the failure modes that only appear under a real Full run.

 * a negative/absurd `top_k` must not slice the list from the wrong end,
 * concurrent cold-cache searches must build a user's index once, not once each,
 * `/health` must be served off the event loop, so a saturated worker pool cannot
   make the container healthcheck fail and restart us mid-run.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
import threading

import numpy as np


def _fresh_modules(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "64")
    monkeypatch.setenv("AML_DISABLE_AUTH", "1")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    return config.load_settings(), main


def test_top_k_is_clamped_before_the_slice(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    settings, main = _fresh_modules(tmp_path, monkeypatch)
    with TestClient(main.create_app(settings)) as client:
        for index in range(3):
            client.post("/add", json={
                "request_id": f"r{index}", "user_id": "u1", "session_id": "s1",
                "messages": [{"role": "user",
                              "content": f"Caroline: note {index} about the pottery class",
                              "timestamp": 1683000000000 + index}],
            })
        # A negative top_k must mean "no items", not "drop the best N from the tail".
        assert client.post("/search", json={"query": "pottery", "user_id": "u1",
                                           "top_k": -5}).json()["data"] == []
        assert client.post("/search", json={"query": "pottery", "user_id": "u1",
                                           "top_k": 0}).json()["data"] == []
        huge = client.post("/search", json={"query": "pottery", "user_id": "u1",
                                           "top_k": 100000}).json()["data"]
        assert 1 <= len(huge) <= settings.top_k_max


def test_concurrent_cold_cache_builds_the_index_once(tmp_path, monkeypatch):
    settings, _ = _fresh_modules(tmp_path, monkeypatch)
    from aml_memory.ingest import Ingestor, Message
    from aml_memory.store import Store

    store = Store(tmp_path / "cache.sqlite3", embed_model="hash", cache_indexes=4)
    ingestor = Ingestor(settings, None)
    units = []
    for index in range(5):
        units.extend(ingestor.build_units(
            request_id=f"r{index}", user_id="u1", session_id="s1",
            messages=[Message(role="user", content=f"Caroline: note {index}",
                              timestamp=1683000000000 + index)],
        ))
    store.insert_add(request_id="batch", user_id="u1", session_id="s1", units=units,
                     vectors=[np.full(64, 0.1, dtype=np.float32) for _ in units],
                     model="hash")

    calls: list[str] = []
    original = store._build_index

    def counting(user_id, version):
        calls.append(user_id)
        return original(user_id, version)

    monkeypatch.setattr(store, "_build_index", counting)

    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(store.user_index("u1"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(result is results[0] for result in results), "callers must share one index"
    assert len(calls) == 1, f"expected a single build, got {len(calls)}"


def test_health_is_async_so_a_busy_pool_cannot_starve_it(tmp_path, monkeypatch):
    settings, main = _fresh_modules(tmp_path, monkeypatch)
    app = main.create_app(settings)
    endpoint = None
    for route in app.routes:
        if getattr(route, "path", None) == settings.health_path:
            endpoint = route.endpoint
    assert endpoint is not None, "health route not registered"
    assert inspect.iscoroutinefunction(endpoint)
    assert asyncio.run(endpoint())["status"] == "ok"


def test_settings_find_a_dotenv_outside_the_working_directory(tmp_path, monkeypatch):
    """`uvicorn` started from $HOME used to silently drop every setting in `.env`:
    the service came up with no reranker and no embedding key, and still answered
    requests. `AML_ENV_FILE` pins the file; the package-adjacent `.env` is the
    fallback when the working directory has none."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("AML_TOP_K_MAX=42\nEMBED_PROVIDER=hash\n", encoding="utf-8")
    monkeypatch.delenv("AML_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("AML_TOP_K_MAX", raising=False)
    monkeypatch.setenv("AML_ENV_FILE", str(env_file))
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path / "data"))
    settings, _ = _fresh_modules(tmp_path, monkeypatch, AML_ENV_FILE=str(env_file))
    assert settings.top_k_max == 42


def test_embed_cache_is_capped_so_a_long_run_cannot_fill_the_disk(tmp_path, monkeypatch):
    """One row per embedded text is the largest table in the database: a Full run
    over millions of turns would grow it without bound. The cap keeps the newest
    rows, and a miss costs exactly one embedding call."""
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path / "cache"))
    from aml_memory.store import Store

    store = Store(tmp_path / "cache" / "aml.sqlite3", embed_model="hash", embed_cache_rows=100)
    vec = np.ones(64, dtype=np.float32) / 8.0
    store.cache_vectors("hash", [(f"h{i}", vec) for i in range(2500)])
    with store._conn() as conn:  # noqa: SLF001 - asserting on internal state on purpose
        count = conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
    assert count <= 110, f"cache should have been pruned to ~100 rows, got {count}"
    # the newest rows survive, so a retry of the last batch is still a hit
    assert "h2499" in store.load_cached_vectors("hash", ["h2499"])
    assert "h0" not in store.load_cached_vectors("hash", ["h0"])
