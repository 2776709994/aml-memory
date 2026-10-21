#!/usr/bin/env python
"""Check a *deployed* endpoint against the leaderboard's API contract.

Run this against your public HTTPS URL after `docker compose up` and before you
register the endpoint with AML. It exercises exactly what the platform will do:
health without a token, a rejected token, an Add that must be searchable the
moment it returns, a replay of the same `request_id`, `top_k` clamping, an
unknown user, and a malformed payload.

    python scripts/verify_endpoint.py --base-url https://mem.example.com \
        --token "$AML_API_KEY"

Exit code 0 means every check passed. Nothing here is destructive: the probe
writes under a random `user_id`, which is the only isolation boundary the
service has.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid

import httpx

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"{PASS if ok else FAIL:4}  {name:44}  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. https://mem.example.com")
    ap.add_argument("--add-path", default="/add")
    ap.add_argument("--search-path", default="/search")
    ap.add_argument("--health-path", default="/health")
    ap.add_argument("--token", default="", help="the AML_API_KEY you configured")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    auth = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    client = httpx.Client(base_url=base, timeout=args.timeout)
    run = uuid.uuid4().hex[:8]
    user = f"probe-{run}"

    # 1. health must answer without credentials and must not leak configuration.
    try:
        t0 = time.perf_counter()
        r = client.get(args.health_path)
        ms = (time.perf_counter() - t0) * 1000
        body = r.text[:200]
        record("health without auth", 200 <= r.status_code < 300, f"{r.status_code} in {ms:.0f} ms {body}")
        leaked = any(word in r.text.lower() for word in ("api_key", "sk-", "token", "secret"))
        record("health leaks no configuration", not leaked, "" if not leaked else r.text[:120])
    except Exception as exc:  # noqa: BLE001
        record("health without auth", False, f"{type(exc).__name__}: {exc}")
        print("\nCannot reach the service — fix the URL / TLS / container first.")
        return 1

    # 2. a wrong token must be rejected (only meaningful when auth is on).
    if args.token:
        r = client.post(args.search_path, json={"query": "x", "user_id": user, "top_k": 3})
        record("search without token -> 401", r.status_code == 401, str(r.status_code))
        r = client.post(
            args.search_path,
            json={"query": "x", "user_id": user, "top_k": 3},
            headers={"Authorization": "Bearer wrong-token"},
        )
        record("search with wrong token -> 401", r.status_code == 401, str(r.status_code))

    # 3. Add, then search for a phrase that only exists in that payload.
    marker = f"zeta marker {run}"
    payload = {
        "request_id": f"probe-{run}-1",
        "user_id": user,
        "session_id": f"session-{run}",
        "messages": [
            {"role": "user", "content": f"My access badge number is {run}.",
             "timestamp": 1683504000000},  # 2023-05-08T00:00:00Z
            {"role": "assistant", "content": f"Noted. I will remember {marker}.",
             "timestamp": 1683504060000},
        ],
    }
    try:
        t0 = time.perf_counter()
        r = client.post(args.add_path, json=payload, headers=auth)
        add_ms = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200 and r.json().get("success") is True
        echoed = r.json() if r.status_code == 200 else {}
        record(
            "add returns 200 {success, request_id, user_id, session_id}",
            ok and echoed.get("request_id") == payload["request_id"]
            and echoed.get("user_id") == user and echoed.get("session_id") == payload["session_id"],
            f"{r.status_code} in {add_ms:.0f} ms {r.text[:120]}",
        )
    except Exception as exc:  # noqa: BLE001
        record("add returns 200", False, f"{type(exc).__name__}: {exc}")
        return 1

    # 4. The write must be searchable immediately, without a pause.
    def search(query: str, top_k: int = 100, extra: dict | None = None) -> tuple[httpx.Response, float]:
        body = {"query": query, "user_id": user, "top_k": top_k}
        body.update(extra or {})
        t0 = time.perf_counter()
        out = client.post(args.search_path, json=body, headers=auth)
        return out, (time.perf_counter() - t0) * 1000

    r, ms = search(marker)
    try:
        data = r.json().get("data", [])
    except Exception:  # noqa: BLE001
        data = []
    hit = any(marker in str(item.get("content", "")) for item in data)
    record("add is searchable immediately", r.status_code == 200 and hit, f"{len(data)} items, {ms:.0f} ms")
    if data:
        keys = set(data[0])
        record(
            "item shape is a subset of {id,content,score,created_at}",
            keys <= {"id", "content", "score", "created_at"} and bool(data[0].get("id")),
            ",".join(sorted(keys)),
        )

    # 5. Replaying the same request_id must be a no-op (the platform retries).
    r, _ = search(marker)
    before = {(item.get("id"), item.get("content")) for item in r.json().get("data", [])}
    r = client.post(args.add_path, json={**payload, "messages": payload["messages"] + [
        {"role": "user", "content": f"duplicate attempt {run}"},
    ]}, headers=auth)
    record("replayed request_id returns 200", r.status_code == 200, str(r.status_code))
    r, _ = search(marker)
    after = {(item.get("id"), item.get("content")) for item in r.json().get("data", [])}
    record("replay stored nothing new", before == after, f"{len(before)} -> {len(after)} items")

    # 6. top_k is a ceiling, never a floor.
    r, _ = search(marker, top_k=1)
    n = len(r.json().get("data", []))
    record("top_k is clamped (top_k=1 -> <=1 item)", n <= 1, f"{n} item(s)")
    r, _ = search(marker, top_k=1000)
    n = len(r.json().get("data", []))
    record("top_k>100 is clamped to 100", n <= 100, f"{n} item(s)")

    # 7. An unknown user is an empty list, not an error or someone else's memories.
    r = client.post(args.search_path,
                    json={"query": marker, "user_id": f"nobody-{run}", "top_k": 10}, headers=auth)
    record("unknown user -> {\"data\": []}", r.status_code == 200 and r.json() == {"data": []},
           r.text[:80])

    # 8. Malformed input must be a 4xx, never a 5xx (the platform treats 5xx as retryable).
    r = client.post(args.search_path, json={"user_id": user}, headers=auth)
    record("malformed search -> 4xx", 400 <= r.status_code < 500, str(r.status_code))
    r = client.post(args.add_path, json={"user_id": user, "messages": []}, headers=auth)
    record("empty add -> 4xx", 400 <= r.status_code < 500, str(r.status_code))

    # 9. Latency, measured the way the run will feel it.
    times = []
    for _ in range(5):
        _, ms = search(marker)
        times.append(ms)
    record(
        "search latency (p50 over 5 calls)",
        statistics.median(times) < 2000,
        f"p50 {statistics.median(times):.0f} ms, max {max(times):.0f} ms",
    )

    failed = [name for name, verdict, _ in results if verdict == FAIL]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: " + ", ".join(failed))
        return 2
    print(f"all {len(results)} checks passed - the endpoint is ready to register.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
