"""Grid over retrieval knobs, reusing one ingested database.

Adds are replayed (the service short-circuits a seen request_id) and vectors are
cached in sqlite, so after the first ingest every extra configuration pays only for
search -- a full 1,382-question sweep costs a couple of minutes instead of an
re-ingest.

    python -m evalharness.sweep                       # default grid, LoCoMo-Refined
    python -m evalharness.sweep base rr-minilm        # only named configs
    SWEEP_SAMPLES=2 python -m evalharness.sweep       # quicker subset

Environment knobs: SWEEP_DATASET, SWEEP_DATA, SWEEP_DB, SWEEP_SAMPLES, SWEEP_TOP_K,
SWEEP_EMBED_PROVIDER, SWEEP_EMBED_MODEL.

The embedder is inferred from the vectors already stored in SWEEP_DB: query and
document vectors must live in the SAME space, so a sweep over a text-embedding-v4
database must not silently embed queries with a local bge model (that mistake cost
one 40-minute run that produced meaningless numbers).
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT.parent / ".venv" / "Scripts" / "python.exe")

DEFAULT_DATA = ROOT.parent / "data" / "locomo_refined"
DEFAULT_DB = ROOT.parent / "runs" / "devdb-refined-bge"

MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_RR = "BAAI/bge-reranker-v2-m3"

CONFIGS: list[dict] = [
    {"name": "base", "env": {}},
    {"name": "bm25-only", "env": {"AML_DENSE_WEIGHT": "0"}},
    {"name": "dense-only", "env": {"AML_BM25_WEIGHT": "0"}},
    {"name": "bm25x1.5", "env": {"AML_BM25_WEIGHT": "1.5"}},
    {"name": "densex0.5", "env": {"AML_DENSE_WEIGHT": "0.5"}},
    {"name": "depth500k20", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20"}},
    {"name": "no-expand", "env": {"AML_EXPAND_NEIGHBORS": "0"}},
    {"name": "no-merge", "env": {"AML_MERGE_MAX_UNITS": "1"}},
    {"name": "floor09", "env": {"AML_SCORE_FLOOR": "0.09"}},
    {"name": "rr-minilm", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                  "RERANK_DEPTH": "60"}},
    # Rerank cost is proportional to the number of documents sent to the paid service
    # (it was 84% of one real bill). These sweep how far the depth can be cut before
    # precision collapses, so a Full run can be budgeted.
    {"name": "rr-minilm-d10", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                      "RERANK_DEPTH": "10"}},
    {"name": "rr-minilm-d20", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                      "RERANK_DEPTH": "20"}},
    {"name": "rr-minilm-d30", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                      "RERANK_DEPTH": "30"}},
    {"name": "rr-minilm-d40", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                      "RERANK_DEPTH": "40"}},
    {"name": "rr-minilm-bm25", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                       "RERANK_DEPTH": "60", "AML_BM25_WEIGHT": "1.5"}},
    {"name": "rr-bge", "env": {"RERANK_PROVIDER": "local", "RERANK_MODEL": BGE_RR,
                               "RERANK_DEPTH": "40"}},
    # stack the two independent levers: deeper candidate pool + cross-encoder
    {"name": "d5k20-rr60", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                   "RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                   "RERANK_DEPTH": "60"}},
    {"name": "d5k20-rr120", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                    "RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                    "RERANK_DEPTH": "120"}},
    {"name": "d5k20-rr60-big", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                       "RERANK_PROVIDER": "local", "RERANK_MODEL": MINILM,
                                       "RERANK_DEPTH": "60",
                                       "AML_TOTAL_CHAR_BUDGET": "200000"}},
    # hosted reranker: no torch in the container, multilingual, uses RERANK_API_KEY
    # that the caller exports (never stored in this file).
    {"name": "rr-api", "env": {"RERANK_PROVIDER": "dashscope", "RERANK_MODEL": "gte-rerank-v2",
                               "RERANK_DEPTH": "60"}},
    {"name": "d5k20-rr-api", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                     "RERANK_PROVIDER": "dashscope",
                                     "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60"}},
    # SiliconFlow Qwen3-Reranker-4B (rerankers are unrestricted by the rules). Rerank
    # is billed per query+document token, so these also measure how far the depth can be
    # cut: depth 20 costs a third of depth 60.
    {"name": "sf-d60", "env": {"RERANK_PROVIDER": "siliconflow",
                               "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
                               "RERANK_BASE_URL": "https://api.siliconflow.cn/v1/rerank",
                               "RERANK_DEPTH": "60", "RERANK_TIMEOUT_S": "60"}},
    {"name": "sf-d30", "env": {"RERANK_PROVIDER": "siliconflow",
                               "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
                               "RERANK_BASE_URL": "https://api.siliconflow.cn/v1/rerank",
                               "RERANK_DEPTH": "30", "RERANK_TIMEOUT_S": "60"}},
    {"name": "sf-d20", "env": {"RERANK_PROVIDER": "siliconflow",
                               "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
                               "RERANK_BASE_URL": "https://api.siliconflow.cn/v1/rerank",
                               "RERANK_DEPTH": "20", "RERANK_TIMEOUT_S": "60"}},
    # Does the shipped widening still pay once a strong reranker has already put the
    # right turn first? Volume and rerank may substitute for each other.
    {"name": "sf-d30-nowiden", "env": {"RERANK_PROVIDER": "siliconflow",
                                       "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
                                       "RERANK_BASE_URL": "https://api.siliconflow.cn/v1/rerank",
                                       "RERANK_DEPTH": "30", "RERANK_TIMEOUT_S": "60",
                                       "AML_CONTEXT_RADIUS": "0"}},
    {"name": "sf-d30-w1t100", "env": {"RERANK_PROVIDER": "siliconflow",
                                      "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
                                      "RERANK_BASE_URL": "https://api.siliconflow.cn/v1/rerank",
                                      "RERANK_DEPTH": "30", "RERANK_TIMEOUT_S": "60",
                                      "AML_CONTEXT_RADIUS": "1", "AML_CONTEXT_TOP": "100",
                                      "AML_CONTEXT_FILL": "100"}},
    # volume knob on top of the reranked head: widen the top hits into a
    # same-session window (items are capped, item SIZE is not).
    {"name": "d5k20-rr-api-w1", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                        "RERANK_PROVIDER": "dashscope",
                                        "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                        "AML_CONTEXT_RADIUS": "1", "AML_CONTEXT_TOP": "40"}},
    {"name": "d5k20-rr-api-w2", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                        "RERANK_PROVIDER": "dashscope",
                                        "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                        "AML_CONTEXT_RADIUS": "2", "AML_CONTEXT_TOP": "30"}},
    {"name": "d5k20-rr-api-w1-all", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                            "RERANK_PROVIDER": "dashscope",
                                            "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                            "AML_CONTEXT_RADIUS": "1", "AML_CONTEXT_TOP": "100"}},
    {"name": "d5k20-rr-api-w2t60", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                           "RERANK_PROVIDER": "dashscope",
                                           "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                           "AML_CONTEXT_RADIUS": "2", "AML_CONTEXT_TOP": "60"}},
    {"name": "d5k20-rr-api-w3t20", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                           "RERANK_PROVIDER": "dashscope",
                                           "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                           "AML_CONTEXT_RADIUS": "3", "AML_CONTEXT_TOP": "20"}},
    {"name": "d5k20-rr-api-w1-t80", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                            "RERANK_PROVIDER": "dashscope",
                                            "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                            "AML_CONTEXT_RADIUS": "1", "AML_CONTEXT_TOP": "80",
                                            "AML_TOTAL_CHAR_BUDGET": "300000"}},
    # two-tier: radius 2 on the reranked head (precision) + radius 1 behind it (coverage)
    {"name": "d5k20-rr-api-w2f100", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                            "RERANK_PROVIDER": "dashscope",
                                            "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                            "AML_CONTEXT_RADIUS": "2", "AML_CONTEXT_TOP": "30",
                                            "AML_CONTEXT_FILL": "100"}},
    {"name": "d5k20-rr-api-w2f70", "env": {"AML_CANDIDATE_DEPTH": "500", "AML_RRF_K": "20",
                                           "RERANK_PROVIDER": "dashscope",
                                           "RERANK_MODEL": "gte-rerank-v2", "RERANK_DEPTH": "60",
                                           "AML_CONTEXT_RADIUS": "2", "AML_CONTEXT_TOP": "30",
                                           "AML_CONTEXT_FILL": "70"}},
]


def _stored_vector_model(db: Path) -> str:
    """Which embedding model produced the vectors in this DB ('' if unknown)."""
    path = db / "aml.sqlite3"
    if not path.exists():
        return ""
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT model, COUNT(*) AS n FROM vectors GROUP BY model ORDER BY n DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return ""
    return row[0] if row else ""


# Rerank is billed per query+document token. Prices verified 2026-10-24:
#   gte-rerank-v2 (DashScope)   ¥0.8 / million tokens
#   Qwen3-Reranker-4B (SiliconFlow) — override with SWEEP_RERANK_PRICE_PER_MTOK
_PRICE_PER_MTOK = {
    "dashscope": 0.8, "http": 0.8, "api": 0.8,
    "siliconflow": float(os.environ.get("SWEEP_RERANK_PRICE_PER_MTOK", "0.1")), "sf": 0.1,
}
_DOC_TOKENS = 80   # a 320-character document, our RERANK_MAX_DOC_CHARS cap
_QUERY_TOKENS = 40


def _question_count(data: str, dataset: str, samples: str) -> int:
    """How many questions a sweep of this slice will ask (for the spend estimate)."""
    try:
        from . import datasets as ds

        instances = ds.load(dataset, data)
        limit = int(samples)
        if limit > 0:
            instances = instances[:limit]
        return sum(len(instance.qas) for instance in instances)
    except Exception:  # noqa: BLE001 - the estimate must never break the sweep
        return 0


def _estimate_spend(env: dict, *, dataset_questions: int) -> dict:
    """Estimated yuan for one sweep configuration, so nobody spends blindly again."""
    provider = (env.get("RERANK_PROVIDER") or "none").lower()
    price = _PRICE_PER_MTOK.get(provider, 0.0) if provider not in {"none", "", "off", "disabled"} else 0.0
    depth = int(env.get("RERANK_DEPTH") or 0) if price else 0
    docs = dataset_questions * max(depth, 0)
    tokens = docs * _DOC_TOKENS + dataset_questions * _QUERY_TOKENS
    return {
        "questions": dataset_questions,
        "rerank_docs": docs,
        "embed_calls": dataset_questions,
        "yuan": tokens / 1_000_000 * price,
    }



def main(argv: list[str] | None = None) -> int:
    configs = CONFIGS
    if argv:
        wanted = set(argv)
        configs = [config for config in CONFIGS if config["name"] in wanted]
    samples = os.environ.get("SWEEP_SAMPLES", "1")
    top_k = os.environ.get("SWEEP_TOP_K", "100")
    dataset = os.environ.get("SWEEP_DATASET", "locomo-refined")
    data = os.environ.get("SWEEP_DATA", str(DEFAULT_DATA))
    db = os.environ.get("SWEEP_DB", str(DEFAULT_DB))

    # Infer the embedder from the stored vectors unless the caller overrode it.
    stored = _stored_vector_model(Path(db))
    provider = os.environ.get("SWEEP_EMBED_PROVIDER") or (
        "openai" if stored and "/" not in stored else "local"
    )
    model = os.environ.get("SWEEP_EMBED_MODEL") or (stored or "BAAI/bge-m3")
    print(f"embedder inferred from {Path(db).name}: {provider}/{model}", flush=True)

    rows = []
    for config in configs:
        env = dict(os.environ)
        env.update(
            {
                "AML_DATA_DIR": db,
                "EMBED_PROVIDER": provider,
                "EMBED_MODEL": model,
                "EMBED_DIM": "1024",
                "AML_QUERY_LLM": "0",
                "AML_INGEST_LLM": "off",
                "AML_DISABLE_AUTH": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_ENDPOINT": "https://hf-mirror.com",
            }
        )
        env.update(config["env"])

        # Rerank billed 84 % of the whole experiment voucher (¥156 of ¥185): a sweep
        # re-ranks `RERANK_DEPTH` documents for every question, and eleven unmeasured
        # full-suite configs was the entire bill.  Say the number out loud first, and
        # require an explicit SWEEP_ALLOW_SPEND=1 before a paid sweep starts.
        spend = _estimate_spend(env, dataset_questions=_question_count(data, dataset, samples))
        if spend["yuan"] > 0:
            print(f"{config['name']}: est. ~¥{spend['yuan']:.2f} for {spend['questions']} questions "
                  f"({spend['rerank_docs']:,} reranked docs, {spend['embed_calls']:,} query embeddings)",
                  flush=True)
            if os.environ.get("SWEEP_ALLOW_SPEND") != "1":
                print("  skipped: set SWEEP_ALLOW_SPEND=1 to run a configuration that costs money",
                      flush=True)
                continue

        out = ROOT.parent / "runs" / f"sweep_{dataset.replace('-', '')}_{config['name']}.json"
        command = [
            PY, "-m", "evalharness.runner",
            "--dataset", dataset, "--path", data,
            "--samples", samples, "--top-k", top_k, "--out", str(out),
        ]
        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"{config['name']}: FAILED {proc.stderr[-500:]}", flush=True)
            continue
        summary = json.loads(Path(out).read_text(encoding="utf-8"))["summary"]["overall"]
        rows.append((config["name"], summary))
        print(
            f"{config['name']:<16}{summary.get('R@1', 0):>7.3f}{summary.get('R@10', 0):>7.3f}"
            f"{summary.get('recall_all@k', 0):>8.3f}{summary.get('full@k', 0):>7.3f}"
            f"{summary.get('mrr', 0):>7.3f}{summary.get('ans_str', 0):>7.3f}"
            f"{summary.get('ans_tok', 0):>7.3f}{summary.get('items_avg', 0):>7.1f}"
            f"{summary.get('chars_avg', 0):>9.0f}{summary.get('ms_p50', 0):>7.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    print(f"{'config':<16}{'R@1':>7}{'R@10':>7}{'all':>8}{'full':>7}{'MRR':>7}"
          f"{'ansS':>7}{'ansT':>7}{'items':>7}{'chars':>9}{'ms':>7}")
    raise SystemExit(main(sys.argv[1:]))
