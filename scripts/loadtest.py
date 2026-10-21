"""End-to-end load test against a live server: does this survive a Full run?

The platform's shape, reproduced against a real HTTP endpoint (not TestClient):

* Adds carry at most 20 messages (the platform chunks at 20 messages / 2,000 words)
  and arrive for many users concurrently;
* every Add must be searchable the moment it returns;
* the same request_id is retried while the first attempt is still in flight;
* a Full run lasts 1-2 days, so the process must keep latency flat as the corpus
  and the number of users grow.

    python scripts/loadtest.py --base http://127.0.0.1:8123 --users 10 --turns 500

Exit code is non-zero if any invariant breaks (dup items, >top_k items, memory
written but not searchable, retry writing twice).
"""
from __future__ import annotations

import argparse
import random
import statistics
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

WORDS = ("sourdough pottery marathon lisbon camera guitar seedling ferry violin "
         "harbour notebook kayak lantern orchard pumpkin bicycle telescope").split()


def _sentence(rng: random.Random, speaker: str) -> str:
    topic = rng.choice(WORDS)
    verb = rng.choice(["started", "quit", "bought", "visited", "fixed", "joined"])
    when = rng.choice(["yesterday", "last week", "in March", "two years ago", "last Monday"])
    return f"{speaker}: I {verb} a {topic} project {when} and it changed my routine."


def build_turns(rng: random.Random, speaker: str, count: int) -> list[dict]:
    now = int(time.time() * 1000)
    return [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": _sentence(rng, speaker),
         "timestamp": now - (count - index) * 60_000}
        for index in range(count)
    ]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8123")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--add-workers", type=int, default=8)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument("--searches", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    base = args.base.rstrip("/")
    rng = random.Random(args.seed)
    client = httpx.Client(base_url=base, timeout=180.0)

    health = client.get("/health")
    print(f"health {health.status_code} {health.text[:120]}")
    if health.status_code >= 300:
        return 2

    # ---- build the Add workload ------------------------------------------------
    adds: list[dict] = []
    for user in range(args.users):
        user_id = f"lt-user-{user:03d}"
        speaker = string.ascii_uppercase[user % 26] + "lice"
        turns = build_turns(rng, speaker, args.turns)
        for chunk_index in range(0, len(turns), args.chunk):
            chunk = turns[chunk_index:chunk_index + args.chunk]
            adds.append(
                {
                    "request_id": f"{user_id}-c{chunk_index // args.chunk:04d}",
                    "user_id": user_id,
                    "session_id": f"s{chunk_index // (args.chunk * 5)}",
                    "messages": chunk,
                }
            )
    rng.shuffle(adds)
    print(f"ingest: {len(adds)} add calls, {args.users * args.turns} turns, "
          f"{args.add_workers} concurrent")

    # ---- every Add must be searchable when it returns --------------------------
    def do_add(payload: dict) -> tuple[float, str]:
        started = time.perf_counter()
        response = client.post("/add", json=payload)
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            return elapsed, f"add {response.status_code}: {response.text[:200]}"
        body = response.json()
        if body.get("request_id") != payload["request_id"]:
            return elapsed, "add did not echo request_id"
        probe = client.post("/search", json={"query": "sourdough pottery marathon",
                                             "user_id": payload["user_id"],
                                             "top_k": 5})
        if probe.status_code != 200 or not probe.json().get("data"):
            return elapsed, "memory not searchable right after add"
        return elapsed, ""

    started = time.perf_counter()
    add_times: list[float] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.add_workers) as pool:
        for elapsed, error in pool.map(do_add, adds):
            add_times.append(elapsed)
            if error:
                errors.append(error)
    ingest_wall = time.perf_counter() - started
    print(f"ingest wall {ingest_wall:.1f}s  "
          f"add p50 {percentile(add_times, .5):.2f}s p95 {percentile(add_times, .95):.2f}s")

    # ---- retry storm: same request_id while the original is in flight ---------
    retry_payload = next(p for p in adds if p["user_id"].endswith("000"))
    before = client.post("/search", json={"query": "sourdough", "user_id": retry_payload["user_id"],
                                          "top_k": args.top_k}).json()["data"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(lambda _: client.post("/add", json=retry_payload).status_code,
                              range(8)))
    print(f"retry storm codes {codes}")
    if any(code != 200 for code in codes):
        errors.append(f"retry storm returned {codes}")

    # ---- search workload ------------------------------------------------------
    queries = ["What did Alice start yesterday?", "Which project did the user quit last week?",
               "When did they buy the camera?", "List everything about pottery.",
               "Did the user ever mention Lisbon before March?"]
    search_times: list[float] = []
    problems: list[str] = []

    def do_search(index: int) -> None:
        user_id = f"lt-user-{index % args.users:03d}"
        started = time.perf_counter()
        response = client.post("/search", json={"query": rng.choice(queries), "user_id": user_id,
                                                "top_k": args.top_k})
        search_times.append(time.perf_counter() - started)
        if response.status_code != 200:
            problems.append(f"search {response.status_code}")
            return
        items = response.json().get("data", [])
        if len(items) > args.top_k:
            problems.append(f"returned {len(items)} items for top_k={args.top_k}")
        ids = [item["id"] for item in items]
        if len(ids) != len(set(ids)):
            problems.append("duplicate ids in one response")
        for item in items:
            if "lt-user-" in item["content"] and user_id not in item["content"]:
                problems.append("leaked another user's memory")

    with ThreadPoolExecutor(max_workers=args.search_workers) as pool:
        list(pool.map(do_search, range(args.searches)))
    print(f"search n={len(search_times)} p50 {percentile(search_times, .5) * 1000:.0f}ms "
          f"p95 {percentile(search_times, .95) * 1000:.0f}ms "
          f"max {max(search_times) * 1000:.0f}ms")

    stats = client.get("/admin/stats")
    print(f"stats {stats.status_code} {stats.text[:200]}")

    errors.extend(sorted(set(problems))[:5])
    if errors:
        print("FAIL")
        for error in errors[:8]:
            print(f"  - {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
