"""Diagnose a harness run: did the gold evidence reach the returned items at all?

Two different failures look identical in the aggregate table:
  * ranking failure  - the gold turn was never selected into the top-k
  * assembly failure - the turn text IS inside a returned block (pulled in as a
    neighbour or merged), but no returned id points at it, so id-based recall
    under-reports it.  The frozen answer model only sees text, so presence is the
    metric that matches reality; the id gap tells us whether we should keep or
    drop neighbour expansion.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evalharness import datasets

_SPACE = re.compile(r"\s+")


def norm(text: str) -> str:
    return _SPACE.sub(" ", text or "").strip().lower()


def gold_texts(instance: "datasets.Instance") -> dict[str, str]:
    out: dict[str, str] = {}
    for session in instance.sessions:
        for turn in session.turns:
            if turn.dia_id:
                out[turn.dia_id] = turn.text
    return out


def analyse(*, run_path: str, dataset: str, data_path: str, show: int = 5, category: str = "") -> None:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    instances = {instance.sample_id: instance for instance in datasets.load(dataset, data_path)}
    texts_by_sample = {sid: gold_texts(instance) for sid, instance in instances.items()}

    buckets: dict[str, dict] = {}
    failures: list[dict] = []
    for row in run["results"]:
        cat = row.get("category") or "unknown"
        if category and cat != category:
            continue
        blob = norm(" ".join(row.get("contents") or []))
        texts = texts_by_sample.get(row.get("sample_id"), {})
        gold = row.get("gold") or []
        present = 0
        for dia in gold:
            needle = norm(texts.get(dia, ""))[:80]
            if needle and needle in blob:
                present += 1
        bucket = buckets.setdefault(cat, {"n": 0, "id_hit": 0, "text_hit": 0, "full_id": 0, "full_text": 0, "graded": 0})
        bucket["n"] += 1
        if not gold:
            continue
        bucket["graded"] += 1
        if row.get("hit_ranks"):
            bucket["id_hit"] += 1
        if present:
            bucket["text_hit"] += 1
        if gold and len(row.get("hit_ranks") or []) >= len(gold):
            bucket["full_id"] += 1
        if gold and present >= len(gold):
            bucket["full_text"] += 1
        if present < len(gold):
            failures.append(
                {
                    "category": cat,
                    "sample_id": row.get("sample_id"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "gold": gold,
                    "missing": [d for d in gold if norm(texts.get(d, ""))[:80] not in blob],
                    "items": row.get("items"),
                    "top_preview": row.get("top_preview"),
                }
            )

    print(f"{'category':<14}{'n':>5}{'graded':>7}{'id_hit':>8}{'text_hit':>9}{'full_id':>8}{'full_text':>10}")
    print("-" * 61)
    for cat in sorted(buckets):
        b = buckets[cat]
        graded = max(1, b["graded"])
        print(
            f"{cat:<14}{b['n']:>5}{b['graded']:>7}{b['id_hit'] / graded:>8.3f}"
            f"{b['text_hit'] / graded:>9.3f}{b['full_id'] / graded:>8.3f}{b['full_text'] / graded:>10.3f}"
        )

    failures.sort(key=lambda row: len(row["missing"]), reverse=True)
    print(f"\n{len(failures)} questions still missing evidence text; showing {min(show, len(failures))}")
    for row in failures[:show]:
        print(f"\n[{row['category']}] {row['question']}")
        print(f"  gold answer : {row['answer']}")
        print(f"  missing dia : {row['missing']}")
        for line in (row.get("top_preview") or [])[:3]:
            print(f"  item> {line[:180]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", default="locomo")
    parser.add_argument("--path", required=True)
    parser.add_argument("--category", default="")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()
    analyse(run_path=args.run, dataset=args.dataset, data_path=args.path, show=args.show, category=args.category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
