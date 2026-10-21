"""Replay a public dataset through the real HTTP contract and score retrieval.

Why retrieval metrics instead of answer accuracy: the AML answer model (gpt-4o-mini)
and judge are frozen and not available offline, but every point of end-to-end score
depends on the <=100 memory items we return.  Recall of the gold evidence turns is a
strictly monotone proxy: an answer can only be produced from memory lines the answer
model actually sees.

Usage:
    python -m evalharness.runner --dataset locomo --path data/locomo/locomo10.json \
        --samples 3 --top-k 100 --out runs/locomo.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evalharness import datasets  # noqa: E402


def _unit_ids(request_id: str, index: int, parts: int = 4) -> list[str]:
    return [
        "m_" + hashlib.sha1(f"{request_id}|{index}|{part}".encode("utf-8")).hexdigest()[:20]
        for part in range(parts)
    ]


_PUNCT = re.compile(r"[^0-9a-z\u4e00-\u9fff ]+")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", str(text).lower()).split())


def answer_coverage(answer: str, contents: list[str]) -> tuple[bool, float]:
    """Can the gold answer be *read off* the returned memory text?

    Recall of gold evidence ids is only a proxy; the score is produced by a frozen
    reader that must be able to emit the exact gold string.  This checks the string
    itself (``ans_str``) and token coverage (``ans_tok``), which is what makes the
    date-anchoring work measurable.
    """
    alternatives = [part for part in str(answer).split(";") if part.strip()]
    if not alternatives:
        return False, 0.0
    haystack = _norm(" \n ".join(contents))
    strict = False
    best = 0.0
    for alternative in alternatives:
        needle = _norm(alternative).strip()
        if not needle:
            continue
        if needle in haystack:
            strict = True
        tokens = [token for token in needle.split() if len(token) > 2 or token.isdigit()]
        if tokens:
            present = sum(1 for token in tokens if token in haystack)
            best = max(best, present / len(tokens))
    return strict, best


@dataclass
class QAResult:
    sample_id: str
    category: str
    question: str
    answer: str
    gold: list[str]
    retrieved: list[str]
    hit_ranks: list[int]
    items: int
    chars: int
    elapsed_ms: float
    top_preview: list[str] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    ans_str: bool = False
    ans_tok: float = 0.0

    @property
    def recall(self) -> float:
        if not self.gold:
            return 0.0
        found = len({rank for rank in self.hit_ranks})
        return found / len(self.gold)

    @property
    def first_rank(self) -> int:
        return min(self.hit_ranks) if self.hit_ranks else 10 ** 6


def _make_app(settings):
    from aml_memory.main import create_app

    return create_app(settings)


def run(
    *,
    dataset: str,
    path: str,
    samples: int = 0,
    qas_per_sample: int = 0,
    top_k: int = 100,
    out: str = "",
    previews: int = 3,
    keep_content: bool = False,
) -> dict[str, Any]:
    from aml_memory.config import load_settings
    from fastapi.testclient import TestClient

    settings = load_settings()
    instances = datasets.load(dataset, path)
    if samples:
        instances = instances[:samples]

    client = TestClient(_make_app(settings))
    results: list[QAResult] = []

    for instance in instances:
        dia_to_units: dict[str, set[str]] = {}
        added = 0
        for payload in datasets.iter_requests(instance):
            response = client.post("/add", json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"add failed {response.status_code}: {response.text[:400]}")
            for index, message in enumerate(payload["messages"]):
                dia = message.get("_dia_id") or ""
                if not dia:
                    continue
                dia_to_units.setdefault(dia, set()).update(_unit_ids(payload["request_id"], index))
            added += len(payload["messages"])

        qas = instance.qas
        if qas_per_sample:
            qas = qas[:qas_per_sample]
        for qa in qas:
            started = time.perf_counter()
            response = client.post(
                "/search",
                json={"query": qa.question, "user_id": instance.sample_id, "top_k": top_k},
            )
            elapsed = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                raise RuntimeError(f"search failed {response.status_code}: {response.text[:400]}")
            items = response.json().get("data") or []

            gold_ids: set[str] = set()
            for dia in qa.evidence:
                gold_ids |= dia_to_units.get(dia, set())

            hit_ranks: list[int] = []
            for rank, item in enumerate(items, start=1):
                components = str(item.get("id") or "").split("+")
                if any(component in gold_ids for component in components):
                    hit_ranks.append(rank)

            item_contents = [str(item.get("content") or "") for item in items]
            ans_str, ans_tok = answer_coverage(qa.answer, item_contents)

            results.append(
                QAResult(
                    sample_id=instance.sample_id,
                    category=qa.category or "unknown",
                    question=qa.question,
                    answer=qa.answer,
                    gold=list(qa.evidence),
                    retrieved=[str(item.get("id")) for item in items],
                    hit_ranks=hit_ranks,
                    items=len(items),
                    chars=sum(len(str(item.get("content") or "")) for item in items),
                    elapsed_ms=elapsed,
                    top_preview=[
                        str(item.get("content") or "")[:200] for item in items[:previews]
                    ],
                    contents=item_contents if keep_content else [],
                    ans_str=ans_str,
                    ans_tok=ans_tok,
                )
            )
        print(f"  [{instance.sample_id}] added {added} turns, scored {len(qas)} questions", flush=True)

    report = summarise(results, top_k=top_k)
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "path": path,
                    "top_k": top_k,
                    "env": {k: ("***" if ("KEY" in k or "TOKEN" in k or "SECRET" in k) else v)
                            for k, v in os.environ.items()
                            if k.startswith(("EMBED_", "AML_", "RERANK"))},
                    "summary": report,
                    "results": [r.__dict__ | {"recall": r.recall, "first_rank": r.first_rank} for r in results],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return report


def summarise(results: list[QAResult], *, top_k: int) -> dict[str, Any]:
    if not results:
        return {"questions": 0}

    def metrics(subset: list[QAResult]) -> dict[str, Any]:
        if not subset:
            return {}
        graded = [r for r in subset if r.gold]
        row: dict[str, Any] = {"n": len(subset)}
        for k in (1, 5, 10, 20, 50, 100):
            if k > top_k:
                continue
            row[f"R@{k}"] = round(
                sum(1 for r in graded if any(rank <= k for rank in r.hit_ranks)) / len(graded), 4
            ) if graded else 0.0
        row["recall_all@k"] = round(
            sum(r.recall for r in graded) / len(graded), 4
        ) if graded else 0.0
        row["full@k"] = round(
            sum(1 for r in graded if r.gold and r.recall >= 0.999) / len(graded), 4
        ) if graded else 0.0
        row["mrr"] = round(
            sum(1.0 / r.first_rank for r in graded) / len(graded), 4
        ) if graded else 0.0
        row["items_avg"] = round(statistics.mean(r.items for r in subset), 1)
        row["chars_avg"] = round(statistics.mean(r.chars for r in subset), 1)
        row["ms_p50"] = round(statistics.median(r.elapsed_ms for r in subset), 1)
        answered = [r for r in subset if r.answer]
        row["ans_str"] = round(sum(1 for r in answered if r.ans_str) / len(answered), 4) if answered else 0.0
        row["ans_tok"] = round(statistics.mean(r.ans_tok for r in answered), 4) if answered else 0.0
        return row

    categories = sorted({r.category for r in results})
    report: dict[str, Any] = {
        "questions": len(results),
        "overall": metrics(results),
        "by_category": {category: metrics([r for r in results if r.category == category]) for category in categories},
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    header = f"{'slice':<18}{'n':>5}{'R@1':>8}{'R@10':>8}{'all':>8}{'full':>8}{'MRR':>8}{'ansS':>7}{'ansT':>7}{'items':>7}{'chars':>8}{'ms':>7}"
    print(header)
    print("-" * len(header))
    rows = [("overall", report["overall"])] + list(report["by_category"].items())
    for name, row in rows:
        if not row:
            continue
        print(
            f"{name:<18}{row['n']:>5}{row.get('R@1', 0):>8.3f}"
            f"{row.get('R@10', 0):>8.3f}"
            f"{row.get('recall_all@k', 0):>8.3f}{row.get('full@k', 0):>8.3f}{row.get('mrr', 0):>8.3f}"
            f"{row.get('ans_str', 0):>7.3f}{row.get('ans_tok', 0):>7.3f}"
            f"{row.get('items_avg', 0):>7.1f}{row.get('chars_avg', 0):>8.0f}{row.get('ms_p50', 0):>7.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=sorted(datasets.LOADERS))
    parser.add_argument("--path", required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--qas", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", default="")
    parser.add_argument("--keep-content", action="store_true")
    args = parser.parse_args(argv)

    report = run(
        dataset=args.dataset,
        path=args.path,
        samples=args.samples,
        qas_per_sample=args.qas,
        top_k=args.top_k,
        out=args.out,
        keep_content=args.keep_content,
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

