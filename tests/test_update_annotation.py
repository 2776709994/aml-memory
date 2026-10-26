"""The update-annotation pass: a revision is marked stale where the reader can see it.

A governance question ("what was my status *before* the upgrade?") is unanswerable
when only the current value survives, and misleading when the stale value sits
unlabelled next to it.  These tests pin that behaviour: the older wording is appended
under the newer record, and nothing is annotated when both turns say the same thing.

The similarity gate is normally semantic, which the offline `hash` embedder used here
cannot express, so these tests set `AML_UPDATE_ANNOTATE_SIM` below zero: what is under
test is the value-difference gate and where the note lands, not the vector space.
"""
from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient


def _app(tmp_path, monkeypatch, **extra):
    monkeypatch.setenv("AML_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("AML_QUERY_LLM", "0")
    monkeypatch.setenv("AML_INGEST_LLM", "off")
    monkeypatch.setenv("AML_DISABLE_AUTH", "1")
    monkeypatch.setenv("AML_CONTEXT_RADIUS", "0")
    for name, value in extra.items():
        monkeypatch.setenv(name, str(value))
    for name in [m for m in list(sys.modules) if m.startswith("aml_memory")]:
        sys.modules.pop(name)
    config = importlib.import_module("aml_memory.config")
    main = importlib.import_module("aml_memory.main")
    return main.create_app(config.load_settings())


REVISION = [
    "Caroline: I am a Gold member on United Airlines since 2019.",
    "Caroline: I just got upgraded to Platinum status on United Airlines.",
]

RESTATEMENT = [
    "Caroline: I adopted a greyhound named Pepper in May 2023.",
    "Caroline: Pepper the greyhound is doing great, adopted in May 2023.",
]


def _seed(client, turns, session_id="s1", start_ms=1683093600000):
    messages = [
        {"role": "user", "content": text, "timestamp": start_ms + index * 86_400_000}
        for index, text in enumerate(turns)
    ]
    response = client.post("/add", json={"request_id": f"r-{session_id}", "user_id": "u1",
                                         "session_id": session_id, "messages": messages})
    assert response.status_code == 200, response.text


def _contents(client, query):
    response = client.post("/search", json={"query": query, "user_id": "u1", "top_k": 100})
    assert response.status_code == 200, response.text
    return [item["content"] for item in response.json()["data"]]


def test_annotation_is_off_unless_asked_for(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        _seed(client, REVISION)
        contents = _contents(client, "United Airlines status")
    assert contents
    assert not any("earlier note" in text for text in contents)


def test_a_revision_carries_the_superseded_wording(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_UPDATE_ANNOTATE_TOP=10, AML_UPDATE_ANNOTATE_SIM=-1)
    with TestClient(app) as client:
        _seed(client, REVISION)
        contents = _contents(client, "What was my United Airlines status before the upgrade?")
    annotated = [text for text in contents if "earlier note" in text]
    assert annotated, contents
    # The note hangs under the newer record and quotes the older value.
    assert all("Platinum" in text for text in annotated), annotated
    assert any("Gold" in text for text in annotated), annotated


def test_a_restatement_is_not_annotated(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, AML_UPDATE_ANNOTATE_TOP=10, AML_UPDATE_ANNOTATE_SIM=-1)
    with TestClient(app) as client:
        _seed(client, RESTATEMENT)
        contents = _contents(client, "Which dog did Caroline adopt?")
    assert contents
    assert not any("earlier note" in text for text in contents), contents
