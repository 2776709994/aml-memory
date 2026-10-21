"""Local smoke test: exercise /add + /search without any external API."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp(prefix="aml-smoke-"))
os.environ["AML_DATA_DIR"] = str(tmp)
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["AML_QUERY_LLM"] = "0"
os.environ["AML_DISABLE_AUTH"] = "1"
os.environ["AML_INGEST_LLM"] = "off"

from fastapi.testclient import TestClient  # noqa: E402

from aml_memory.main import create_app  # noqa: E402

CONVERSATION = [
    ("2023-05-07T10:00:00Z", "user", "Caroline: I went to the LGBTQ support group yesterday and it was empowering."),
    ("2023-05-07T10:01:00Z", "assistant", "Melanie: That sounds great. I adopted a puppy last week, a golden retriever named Rex."),
    ("2023-05-07T10:02:00Z", "user", "Caroline: I have been painting for six years now, mostly watercolors."),
    ("2023-05-08T09:00:00Z", "user", "Caroline: I am planning a trip to Norway next month with my sister."),
    ("2023-05-08T09:01:00Z", "assistant", "Melanie: I used to work as a nurse but I quit two years ago to study psychology."),
    ("2023-05-08T09:02:00Z", "user", "Caroline: My favourite book is 'The Left Hand of Darkness' by Ursula K. Le Guin."),
    ("2023-06-01T18:00:00Z", "assistant", "Melanie: I have three siblings, two brothers and one sister."),
    ("2023-06-01T18:01:00Z", "user", "Caroline: I moved to Berlin in 2019 and I still miss Lisbon."),
]

QUESTIONS = [
    "When did Caroline go to the support group?",
    "How many siblings does Melanie have?",
    "What is Caroline's favourite book?",
    "Where did Caroline move in 2019?",
    "What did Melanie adopt and what was it called?",
    "Why did Melanie quit her job?",
    "Which city does Caroline miss?",
    "What does Caroline do for a hobby and how long?",
]


def main() -> None:
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    print("health:", health.status_code, health.json())

    payload = {
        "request_id": "req-1",
        "user_id": "user-a",
        "session_id": "sess-1",
        "messages": [
            {"role": role, "content": content, "timestamp": ts}
            for ts, role, content in CONVERSATION
        ],
    }
    resp = client.post("/add", json=payload)
    print("add:", resp.status_code, resp.json())
    assert resp.status_code == 200 and resp.json()["request_id"] == "req-1"
    assert resp.json()["session_id"] == "sess-1"

    again = client.post("/add", json=payload)
    print("add replay:", again.status_code, again.json())

    stats = client.get("/admin/stats").json()
    print("stats:", stats)

    for question in QUESTIONS:
        out = client.post("/search", json={"query": question, "user_id": "user-a", "top_k": 5}).json()
        print("\nQ:", question)
        for item in out["data"]:
            print(f"  {item['score']:.4f}  {item['content'][:120]}")

    empty = client.post("/search", json={"query": "anything", "user_id": "nobody", "top_k": 10}).json()
    print("\nunknown user:", empty)
    assert empty["data"] == []

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
