"""Assembly tests: item size, context widening, dedupe and the RAM guards.

These exercise behaviour the platform contract leaves to us but grades indirectly:
the reader sees a PREFIX of our list, so what sits inside one item matters as much
as its rank.
"""
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


SESSION = [
    "Caroline: I adopted a greyhound named Pepper on 7 May 2023.",
    "Melanie: Pepper is such a good name! I signed up for a pottery class yesterday.",
    "Caroline: The pottery studio on Elm Street is where I take Pepper for walks.",
    "Melanie: Nice. My sister also wanted a greyhound but she moved to Lisbon.",
    "Caroline: Lisbon in July is lovely, we went there before the adoption.",
    "Melanie: I will ask about the studio schedule next Monday.",
]


def _seed(client, turns=SESSION, session_id="s1", start_ms=1683093600000):
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text,
         "timestamp": start_ms + index * 60_000}
        for index, text in enumerate(turns)
    ]
    response = client.post("/add", json={"request_id": f"r-{session_id}", "user_id": "u1",
                                         "session_id": session_id, "messages": messages})
    assert response.status_code == 200, response.text


def test_items_stay_single_turns_without_widening(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_CONTEXT_RADIUS=0)
    with TestClient(app) as client:
        _seed(client)
        items = client.post("/search", json={"query": "pottery studio Elm Street",
                                             "user_id": "u1", "top_k": 100}).json()["data"]
    assert items
    for item in items:
        assert item["content"].count("\n") <= 2, item["content"]


def test_widening_pulls_neighbour_turns_into_the_head(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_CONTEXT_RADIUS=1, AML_CONTEXT_TOP=1,
               AML_CONTEXT_FILL=0,
               AML_EXPAND_NEIGHBORS=0, AML_MERGE_MAX_UNITS=1)
    with TestClient(app) as client:
        _seed(client)
        payload = client.post("/search", json={"query": "pottery studio Elm Street",
                                               "user_id": "u1", "top_k": 100}).json()["data"]
    head = payload[0]["content"]
    assert "\n" in head, f"top item was not widened: {head!r}"
    # the returned list must not repeat a turn that the head already absorbed.
    # Compare on distinctive phrases: stored turns carry an anchored date, so the
    # raw input string never appears verbatim.
    flat = "\n".join(item["content"] for item in payload)
    for phrase in ("greyhound named Pepper", "pottery studio on Elm Street",
                   "moved to Lisbon", "before the adoption", "studio schedule"):
        assert flat.count(phrase) == 1, f"{phrase!r} appears {flat.count(phrase)} times"


def test_widening_never_crosses_a_session_boundary(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_CONTEXT_RADIUS=2, AML_CONTEXT_TOP=100,
               AML_EXPAND_NEIGHBORS=0, AML_MERGE_MAX_UNITS=1)
    with TestClient(app) as client:
        _seed(client, turns=SESSION[:2], session_id="sA")
        _seed(client, turns=["Melanie: Different week entirely, I bought a violin."],
               session_id="sB", start_ms=1683093600000 + 10 * 86_400_000)
        items = client.post("/search", json={"query": "greyhound Pepper", "user_id": "u1",
                                             "top_k": 100}).json()["data"]
    for item in items:
        if "violin" in item["content"]:
            assert "Pepper" not in item["content"], "a window crossed the session boundary"


def test_duplicate_turns_are_deduped(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_CONTEXT_RADIUS=0)
    with TestClient(app) as client:
        _seed(client, turns=["Caroline: I adopted a greyhound named Pepper."] * 3, session_id="s1")
        items = client.post("/search", json={"query": "greyhound Pepper", "user_id": "u1",
                                             "top_k": 100}).json()["data"]
    assert len(items) == 1, [item["content"] for item in items]


def test_dedupe_prefers_the_newer_statement_when_asked(tmp_path, monkeypatch):
    """Update questions ("where do I live now?") hinge on the newer half of a fact.

    The default keeps the better-ranked half; AML_DEDUPE_PREFER=newer keeps the
    later statement, which is what "now" means.
    """
    old = "Caroline: I live in Boston now, and honestly I really love the neighbourhood so much."
    new = "Caroline: I live in Denver now, and honestly I really love the neighbourhood so much."
    for prefer, expected, dropped in (("score", "Boston", None), ("newer", "Denver", "Boston")):
        case = tmp_path / prefer
        case.mkdir()
        app = _app(case, monkeypatch, AML_CONTEXT_RADIUS=0, AML_DEDUPE_PREFER=prefer)
        with TestClient(app) as client:
            _seed(client, turns=[old], session_id="s1", start_ms=1683093600000)
            _seed(client, turns=[new], session_id="s2", start_ms=1691000000000)
            items = client.post("/search", json={"query": "where does Caroline live now",
                                                 "user_id": "u1", "top_k": 100}).json()["data"]
        text = " ".join(item["content"] for item in items)
        assert expected in text, (prefer, text)
        if dropped:
            assert dropped not in text, (prefer, text)


def test_exhaustive_cap_trims_list_answers(tmp_path, monkeypatch):
    """Counting questions are exhaustive by nature; the cap is a knob, not a constant."""
    turns = [f"Caroline: On trip {index} I visited town{index} with my sister." for index in range(30)]
    app = _app(tmp_path, monkeypatch, AML_CONTEXT_RADIUS=0, AML_EXHAUSTIVE_CAP=5)
    with TestClient(app) as client:
        _seed(client, turns=turns)
        items = client.post("/search", json={"query": "how many towns did I visit with my sister",
                                             "user_id": "u1", "top_k": 100}).json()["data"]
    assert 0 < len(items) <= 5, len(items)


def test_index_cache_evicts_by_row_budget(tmp_path, monkeypatch):
    """A count-only cache would let a few huge users exhaust the container."""
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "16")
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    store_module = importlib.import_module("aml_memory.store")
    settings = config.load_settings()
    store = store_module.Store(settings.db_path, embed_model="hash", cache_indexes=24, cache_rows=10)

    def seed(user_id: str, count: int) -> None:
        from aml_memory.store import Unit

        units = [
            Unit(id=f"{user_id}-{index}", user_id=user_id, session_id="s", seq=index, kind="turn",
                 speaker="Caroline", created_ms=0, date_str="2023-05-07", text=f"{user_id} turn {index}",
                 prev_id=None, next_id=None, meta={})
            for index in range(count)
        ]
        import numpy as np

        vectors = [np.ones(16, dtype=np.float32) / 4 for _ in units]
        store.insert_add(request_id=f"r-{user_id}", user_id=user_id, session_id="s",
                         units=units, vectors=vectors, model="hash")

    seed("a", 6)
    seed("b", 6)
    seed("c", 6)
    store.user_index("a")
    store.user_index("b")
    store.user_index("c")
    assert len(store._indexes) <= 2, list(store._indexes)
    assert "c" in store._indexes  # the user just loaded is never evicted
