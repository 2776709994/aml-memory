#!/usr/bin/env python
"""Pre-Full rehearsal against a *deployed* endpoint.

`verify_endpoint.py` answers "does the contract hold?".  This one answers the
question that actually decides a Full run: "will it still hold for ~6 000
instances, ~300 M tokens and a reader that dies on the first 500%?"  It adds the
three things that only show up at scale:

* readiness  — the embedding pipeline answered (``/health`` → ``embedding``),
                the reranker is on, the version is the one we think it is;
* capacity   — Add and Search latency under concurrency, plus units/vectors/
                db_bytes/embed_cache growth, so a disk or vector gap surfaces now
                instead of at 80 % of a run that only gets one attempt;
* soak-lite  — one Add → immediate Search → re-Add cycle repeated, which is the
                platform's real access pattern (interleaved, not batched).

    python scripts/preflight_full.py --base-url https://mem.example.com \
        --token "$AML_API_KEY" --concurrency 16

Exit 0 means every hard check passed.  It writes only under random
``probe-pre-<hex>`` user ids, so it never touches real evaluation data.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import sys
import time
import uuid

import httpx

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool | None, detail: str = "") -> None:
    """ok=True → PASS, False → FAIL, None → WARN (informational, never fatal)."""
    verdict = WARN if ok is None else (PASS if ok else FAIL)
    results.append((name, verdict, detail))
    print(f"{verdict:4}  {name:44}  {detail}", flush=True)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--add-path", default="/add")
    ap.add_argument("--search-path", default="/search")
    ap.add_argument("--health-path", default="/health")
    ap.add_argument("--stats-path", default="/admin/stats")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=16, help="Add/Search workers to probe")
    ap.add_argument("--search-samples", type=int, default=60)
    ap.add_argument("--add-samples", type=int, default=12)
    ap.add_argument("--max-p95-ms", type=float, default=3000.0)
    ap.add_argument("--skip-capacity", action="store_true")
    ap.add_argument("--allow-no-rerank", action="store_true",
                    help="downgrade the reranker check to a warning (local/dev runs)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    run_id = uuid.uuid4().hex[:8]
    user = f"probe-pre-{run_id}"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    auth_headers = {**headers, "Content-Type": "application/json"}
    client = httpx.Client(timeout=args.timeout, headers={"User-Agent": "aml-preflight/1"})

    def post(path: str, payload: dict, **kwargs) -> httpx.Response:
        return client.post(base + path, json=payload, headers=auth_headers, **kwargs)

    def add(request_id: str, turns: list[str], session: str = "s1") -> httpx.Response:
        return post(
            args.add_path,
            {
                "request_id": request_id,
                "user_id": user,
                "session_id": session,
                "messages": [
                    {"role": "user" if index % 2 == 0 else "assistant", "content": text,
                     "timestamp": 1_690_000_000_000 + index * 60_000}
                    for index, text in enumerate(turns)
                ],
            },
        )

    def search(query: str, top_k: int = 100, who: str | None = None) -> httpx.Response:
        return post(args.search_path, {"query": query, "user_id": who or user, "top_k": top_k})

    turns = [
        "Caroline: I signed up for a pottery class in the town hall last Tuesday.",
        "Melanie: My sister adopted a greyhound called Pepper the week before.",
        "Caroline: I am thinking about a marathon in November, but my knee hurts.",
        "Melanie: I moved to Denver in March, so the commute is longer now.",
    ]

    print(f"# preflight {base} (probe user {user})\n")

    # Warm up before reading readiness: `embedding` only turns from "unknown" into
    # "ok"/"failing" once a real embedding call has happened, and a cold service would
    # otherwise look broken on the most important check of the run.
    print("## warm-up")
    try:
        warm = add(f"pre-{run_id}-warm", turns, session="warm")
        record("warm-up add accepted", warm.status_code == 200, f"{warm.status_code}")
    except Exception as exc:  # noqa: BLE001
        record("warm-up add accepted", False, f"{type(exc).__name__}: {exc}")

    print("\n## readiness")

    try:
        health = client.get(base + args.health_path, headers={"User-Agent": "aml-preflight/1"})
        record("health answers without a token", health.status_code // 100 == 2, f"{health.status_code}")
        body = health.json() if health.headers.get("content-type", "").startswith("application/json") else {}
        lowered = health.text.lower()
        leaked = [token for token in ("api_key", "sk-", "token", "secret") if token in lowered]
        record("health leaks no configuration", not leaked, ",".join(leaked))
        embedding = str(body.get("embedding", "unknown"))
        if embedding == "unknown":
            record("embedding pipeline is healthy", None,
                   "health.embedding=unknown (the warm-up add did not reach the embedder)")
        else:
            record(
                "embedding pipeline is healthy",
                embedding == "ok",
                f"health.embedding={embedding}"
                + ("" if embedding == "ok" else " → fix the provider key/quota before a Full run"),
            )
        version = str(body.get("version", "?"))
        record("version is reported", version not in {"", "?"}, version)
    except Exception as exc:  # noqa: BLE001
        record("health answers without a token", False, f"{type(exc).__name__}: {exc}")

    stats: dict = {}
    try:
        stats_resp = client.get(base + args.stats_path, headers=headers)
        record("stats readable with the token", stats_resp.status_code == 200, f"{stats_resp.status_code}")
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            units = int(stats.get("units", 0))
            vectors = int(stats.get("vectors", 0))
            missing = max(0, units - vectors)
            record(
                "every stored turn has a vector",
                missing == 0,
                f"units={units} vectors={vectors} missing={missing}"
                + ("" if missing == 0 else " → run scripts/backfill_vectors.py"),
            )
            db_bytes = int(stats.get("db_bytes", 0))
            record("storage footprint", None,
                   f"db_bytes={db_bytes / 1e6:.1f} MB embed_cache={stats.get('embed_cache')}")
            rerank = stats.get("rerank") or {}
            rerank_on = bool(rerank.get("enabled"))
            record(
                "reranker enabled",
                True if rerank_on else (None if args.allow_no_rerank else False),
                f"enabled={rerank.get('enabled')} fallbacks={rerank.get('fallbacks')}"
                + ("" if rerank_on else " → gte-rerank-v2 is worth ~0.17 R@1; enable it for Full"),
            )
    except Exception as exc:  # noqa: BLE001
        record("stats readable with the token", False, f"{type(exc).__name__}: {exc}")

    print("\n## contract under load")
    try:
        unauth = client.post(base + args.search_path, json={"query": "x", "user_id": user},
                             headers={"User-Agent": "aml-preflight/1"})
        record("search without a token is rejected", unauth.status_code == 401, f"{unauth.status_code}")

        latencies: list[float] = []
        adds: list[float] = []
        failures: list[str] = []
        for index in range(max(1, args.add_samples)):
            started = time.perf_counter()
            response = add(f"pre-{run_id}-c{index}", turns, session=f"s{index}")
            adds.append((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                failures.append(f"add {response.status_code}: {response.text[:80]}")
                continue
            echo = response.json()
            if echo.get("request_id") != f"pre-{run_id}-c{index}" or echo.get("user_id") != user:
                failures.append("add did not echo the ids")
            found = search("pottery class marathon greyhound")
            latencies.append((time.perf_counter() - started) * 1000)
            if found.status_code != 200 or not found.json().get("data"):
                failures.append(f"add not searchable on cycle {index}")
        record(
            "add → immediately searchable, repeated",
            not failures,
            failures[0] if failures else f"{len(adds)} cycles, add p50 {statistics.median(adds):.0f} ms",
        )

        replay_target = f"pre-{run_id}-c0"
        before = client.get(base + args.stats_path, headers=headers).json().get("units", 0)
        replayed = add(replay_target, turns, session="s0")
        after = client.get(base + args.stats_path, headers=headers).json().get("units", 0)
        record(
            "replayed request_id stores nothing new",
            200 <= replayed.status_code < 300 and after == before,
            f"{replayed.status_code} units {before} -> {after}",
        )

        one = search("pottery", top_k=1)
        record("top_k=1 returns at most one item", len(one.json().get("data", [])) <= 1,
               str(len(one.json().get("data", []))))
        many = search("pottery", top_k=5000)
        record("top_k is clamped to the maximum", len(many.json().get("data", [])) <= 100,
               str(len(many.json().get("data", []))))
        empty = search("pottery", who=f"probe-unknown-{run_id}")
        record("unknown user returns an empty list", empty.json().get("data") == [],
               str(empty.text[:60]))

        for label, payload in (
            ("malformed search is rejected", {"user_id": user}),
        ):
            code = post(args.search_path, payload).status_code
            record(label, 400 <= code < 500, f"{code}")

        # An empty chunk is accepted on purpose: the platform always sends at least one
        # message, while refusing a chunk it does consider valid would fail the whole
        # stage (and a Full run gets one real attempt).  A no-op 200 is the safe side of
        # that asymmetry, so this is informational, not a failure.
        empty_code = post(
            args.add_path,
            {"request_id": f"pre-{run_id}-empty", "user_id": user, "messages": []},
        ).status_code
        record("empty chunk accepted as a no-op", None, f"{empty_code} (by design)")

        if not args.skip_capacity:
            queries = ["pottery class", "marathon knee", "greyhound Pepper", "Denver commute",
                       "town hall", "sister adopted", "November race", "pottery teacher"]
            measured: list[float] = []

            def timed_search(index: int) -> tuple[float, bool]:
                started = time.perf_counter()
                response = search(queries[index % len(queries)])
                ok = response.status_code == 200 and bool(response.json().get("data"))
                return (time.perf_counter() - started) * 1000, ok

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                for elapsed, ok in pool.map(timed_search, range(max(1, args.search_samples))):
                    measured.append(elapsed)
                    if not ok:
                        failures.append("a concurrent search failed")
            p50, p95, worst = _pct(measured, 0.50), _pct(measured, 0.95), max(measured)
            record(
                f"search p95 under {args.concurrency}-way concurrency",
                p95 <= args.max_p95_ms,
                f"n={len(measured)} p50={p50:.0f} ms p95={p95:.0f} ms max={worst:.0f} ms",
            )
            record("ingest latency", None,
                   f"add p50={statistics.median(adds):.0f} ms p95={_pct(adds, 0.95):.0f} ms "
                   f"(per Add of {len(turns)} turns)")
    except Exception as exc:  # noqa: BLE001
        record("contract under load", False, f"{type(exc).__name__}: {exc}")

    failed = [name for name, verdict, _ in results if verdict == FAIL]
    print("\n## platform gate map")
    print("  1 smoke completed        → platform-side (needs a passing smoke run)")
    print("  2 interface contract     → checks above: health/auth/add echo/searchable/replay/top_k/4xx")
    print("  3 model rules            → text-embedding-v4 for vectors; generative N/A; reranker free")
    print("  4 capacity / limits      → latency + units/vectors/db_bytes above")
    print("  5 deployment information → endpoint + token registered in the platform form")
    print("  6 originality            → docs/METHOD.md, MIT, public repo at the pinned tag")
    print("  7 substantive submission → retrieval + write paths documented and measured")
    print("  8 no cheating            → no answers/options injected, no eval-data training")

    print()
    if failed:
        print(f"FAILED {len(failed)} check(s): " + ", ".join(failed))
        return 1
    print("all hard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
