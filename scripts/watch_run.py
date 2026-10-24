#!/usr/bin/env python3
"""Poll a deployed memory service while a long evaluation runs.

A Full run lasts hours to days and the two ways it dies are boring: the disk fills
up, or the embedding provider stops answering while the process stays healthy.  Both
are visible in `/admin/stats` and `/health`, so this just samples them on an interval
and prints one line per sample plus a warning when a trend turns bad.

Usage:
    python scripts/watch_run.py --base-url https://memory.example.com --token TOKEN
    python scripts/watch_run.py --base-url ... --token ... --interval 120 --samples 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

GB = 1024 ** 3


def fetch(url: str, token: str | None, timeout: float = 20.0) -> tuple[int, dict | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "aml-watch/1"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, None
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:  # noqa: BLE001 - a monitor must never crash the loop
        return 0, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--stats-path", default="/admin/stats")
    ap.add_argument("--health-path", default="/health")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--samples", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--disk-warn-gb", type=float, default=3.0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    print(f"# watching {base} every {args.interval:.0f}s"
          f"{' forever' if args.samples == 0 else f' for {args.samples} samples'}")
    print(f"{'time':8} {'units':>8} {'vectors':>8} {'miss':>5} {'db_GB':>6} {'free_GB':>7} "
          f"{'GB/h':>5} {'embed':>7} {'rerank_fb':>9}  note")

    previous: dict | None = None
    previous_at: float | None = None
    problems: list[str] = []
    sample = 0

    while args.samples == 0 or sample < args.samples:
        sample += 1
        status, stats = fetch(base + args.stats_path, args.token)
        _, health = fetch(base + args.health_path, None)
        now = time.time()
        stamp = time.strftime("%H:%M:%S")

        if status != 200 or not isinstance(stats, dict):
            print(f"{stamp:8} {'-':>8} {'-':>8} {'-':>5} {'-':>6} {'-':>7} {'-':>5} "
                  f"{'-':>7} {'-':>9}  stats HTTP {status}")
            problems.append(f"{stamp} stats HTTP {status}")
            time.sleep(args.interval)
            continue

        units = int(stats.get("units", 0))
        vectors = int(stats.get("vectors", 0))
        missing = max(0, units - vectors)
        db_gb = int(stats.get("db_bytes", 0)) / GB
        free_gb = int(stats.get("disk_free_bytes", 0)) / GB
        embed = str((health or {}).get("embedding", "?"))
        rerank = stats.get("rerank") or {}
        notes: list[str] = []

        rate = None
        if previous is not None and previous_at is not None and now > previous_at:
            delta = int(stats.get("db_bytes", 0)) - int(previous.get("db_bytes", 0))
            rate = delta / GB / ((now - previous_at) / 3600.0)

        if embed == "failing":
            notes.append("EMBEDDING FAILING")
            problems.append(f"{stamp} embedding failing")
        elif embed == "unknown":
            notes.append("embedding state unknown (no Add since start)")
        if missing:
            notes.append(f"{missing} turns without a vector")
        if free_gb and free_gb < args.disk_warn_gb:
            notes.append("DISK LOW")
            problems.append(f"{stamp} disk free {free_gb:.1f} GB")
        if rerank.get("fallbacks"):
            notes.append(f"rerank fallbacks={rerank.get('fallbacks')}")
        if units and previous is not None and units == previous.get("units"):
            notes.append("no new units since last sample")

        print(f"{stamp:8} {units:8d} {vectors:8d} {missing:5d} {db_gb:6.2f} {free_gb:7.2f} "
              f"{(rate if rate is not None else 0):5.2f} {embed:>7} "
              f"{rerank.get('fallbacks', 0):9d}  {'; '.join(notes)}")

        previous, previous_at = stats, now
        if args.samples == 0 or sample < args.samples:
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break

    if problems:
        print(f"\n{len(problems)} warning(s):")
        for line in problems[-10:]:
            print(" -", line)
        return 2
    print("\nno warnings observed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
