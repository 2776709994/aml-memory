"""Runtime configuration (env-driven, no secrets in code)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("aml.config")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- service -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("AML_DATA_DIR", "./data")))
    api_key: str = field(default_factory=lambda: _env("AML_API_KEY"))
    add_path: str = field(default_factory=lambda: _env("AML_ADD_PATH", "/add"))
    search_path: str = field(default_factory=lambda: _env("AML_SEARCH_PATH", "/search"))
    health_path: str = field(default_factory=lambda: _env("AML_HEALTH_PATH", "/health"))
    disable_auth: bool = field(default_factory=lambda: _env_bool("AML_DISABLE_AUTH", False))
    request_timeout_s: int = field(default_factory=lambda: _env_int("AML_REQUEST_TIMEOUT_S", 1700))
    max_concurrent_ingest: int = field(default_factory=lambda: _env_int("AML_MAX_CONCURRENT_INGEST", 8))
    max_concurrent_search: int = field(default_factory=lambda: _env_int("AML_MAX_CONCURRENT_SEARCH", 16))
    # How many per-user search indexes stay in RAM.  The row budget is the real
    # guard: a 100k-turn user at dim 1024 is ~410 MB of float32 matrix, so a
    # count-only cap can exhaust a 2 GB container during a Full run.  120k rows is
    # ~470 MB of matrix, which leaves room on a 2 GB box; raise it on a big one.
    index_cache_count: int = field(default_factory=lambda: _env_int("AML_INDEX_CACHE_COUNT", 24))
    index_cache_rows: int = field(default_factory=lambda: _env_int("AML_INDEX_CACHE_VECTORS", 120_000))
    # Vectors are stored as int8 (dim bytes/row), so 200k rows is ~200 MB and the
    # cache only ever covers recent text; a miss just costs one embedding call.
    embed_cache_rows: int = field(default_factory=lambda: _env_int("EMBED_CACHE_ROWS", 200_000))
    # Document texts are embedded once and then live in `vectors`, so caching them
    # again doubles the disk cost of every ingested turn for no benefit: a repeated
    # Add is short-circuited by the request_id claim, not by the embed cache.
    embed_cache_documents: bool = field(
        default_factory=lambda: _env_bool("EMBED_CACHE_DOCS", False)
    )
    log_prompts: bool = field(default_factory=lambda: _env_bool("AML_LOG_PROMPTS", False))

    # --- embeddings (Academic track: text-embedding-v4) --------------------
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "hash").lower())
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "text-embedding-v4"))
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL").rstrip("/"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY"))
    embed_dim: int = field(default_factory=lambda: _env_int("EMBED_DIM", 1024))
    embed_batch: int = field(default_factory=lambda: _env_int("EMBED_BATCH", 10))
    embed_concurrency: int = field(default_factory=lambda: _env_int("EMBED_CONCURRENCY", 6))
    embed_max_chars: int = field(default_factory=lambda: _env_int("EMBED_MAX_CHARS", 6000))
    embed_cache_size: int = field(default_factory=lambda: _env_int("EMBED_CACHE_SIZE", 100_000))
    # Byte cap for the in-RAM embedding cache (float16 entries are 2 KB at dim 1024);
    # the SQLite embed cache behind it is unbounded on disk and always authoritative.
    embed_cache_mb: int = field(default_factory=lambda: _env_int("EMBED_CACHE_MB", 256))

    # --- LLM (Academic track: gpt-4o-mini) --------------------------------
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL").rstrip("/"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 60))
    # Off by default: the rules allow only gpt-4o-mini inside Add/Search, and an
    # unconfigured deployment must not spend a failing network round trip per query.
    query_llm: bool = field(default_factory=lambda: _env_bool("AML_QUERY_LLM", False))
    ingest_llm: str = field(default_factory=lambda: _env("AML_INGEST_LLM", "off").lower())
    session_summary_chars: int = field(default_factory=lambda: _env_int("AML_SESSION_SUMMARY_CHARS", 1200))

    # --- retrieval ---------------------------------------------------------
    # These defaults encode the config measured best on LoCoMo-Refined (see
    # README section 3); an env-less deployment must still be a good one.
    top_k_max: int = field(default_factory=lambda: _env_int("AML_TOP_K_MAX", 100))
    candidate_depth: int = field(default_factory=lambda: _env_int("AML_CANDIDATE_DEPTH", 500))
    rerank_provider: str = field(default_factory=lambda: _env("RERANK_PROVIDER", "none").lower())
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    rerank_base_url: str = field(default_factory=lambda: _env("RERANK_BASE_URL").rstrip("/"))
    rerank_api_key: str = field(default_factory=lambda: _env("RERANK_API_KEY"))
    rerank_depth: int = field(default_factory=lambda: _env_int("RERANK_DEPTH", 60))
    rerank_batch: int = field(default_factory=lambda: _env_int("RERANK_BATCH", 16))
    # A hosted reranker occasionally answers in ~20 s under load; past this budget
    # the search keeps fusion order instead of making the caller wait (the fallback
    # is already wired: Reranker.score returns None on timeout and _rerank passes
    # the fused scores through).
    rerank_timeout_s: int = field(default_factory=lambda: _env_int("RERANK_TIMEOUT_S", 12))
    max_content_chars: int = field(default_factory=lambda: _env_int("AML_MAX_CONTENT_CHARS", 700))
    # The reader's answer budget on the leaderboard is ~117,760 input tokens, and it
    # keeps a *prefix* of our list, so unused budget is thrown away while extra tail
    # items cost nothing (they are dropped first if anything is dropped at all).
    # 320k characters is ~80k tokens of English: enough tail to fill all 100 slots on
    # a 10k-turn conversation without ever approaching the reader's ceiling.
    total_char_budget: int = field(default_factory=lambda: _env_int("AML_TOTAL_CHAR_BUDGET", 320000))
    min_items: int = field(default_factory=lambda: _env_int("AML_MIN_ITEMS", 8))
    expand_neighbors: int = field(default_factory=lambda: _env_int("AML_EXPAND_NEIGHBORS", 1))
    merge_max_units: int = field(default_factory=lambda: _env_int("AML_MERGE_MAX_UNITS", 3))
    # Items whose token sets overlap this much are treated as the same memory and
    # dropped from the tail: a repeat is pure distraction for the reader.
    dedupe_jaccard: float = field(default_factory=lambda: _env_float("AML_DEDUPE_JACCARD", 0.85))
    # When two candidates are near-duplicates of each other, which one survives:
    # "score" keeps the better-ranked one, "newer" keeps the more recent statement.
    # Update questions ("where do I live now?") hinge on the newer half of a
    # changed fact, and the higher-ranked half is often the stale one.
    dedupe_prefer: str = field(default_factory=lambda: _env("AML_DEDUPE_PREFER", "score"))
    # Counting and list questions are exhaustive by nature: capping them trims
    # evidence the answer needs.  The cap exists to keep noise down, not volume.
    exhaustive_cap: int = field(default_factory=lambda: _env_int("AML_EXHAUSTIVE_CAP", 40))
    # Multiplier on assistant-role turns at ranking time.  1.0 keeps the fused
    # order; below 1.0 pushes long advice replies behind the user's own statements.
    assistant_prior: float = field(default_factory=lambda: _env_float("AML_ASSISTANT_PRIOR", 1.0))
    # "Still ...? / what is X now" questions are decided by the newest statement, and
    # the answer prompt already prefers the recent fact on conflict.  This tilts the
    # ranking toward newer turns for that question type only; 0.0 keeps plain order.
    update_boost: float = field(default_factory=lambda: _env_float("AML_UPDATE_BOOST", 0.0))
    # A mild, always-on tilt toward newer turns.  The answer prompt tells the reader to
    # prefer the most recent statement when two conflict, so breaking near-ties by recency
    # matches the reader instead of asking it to sort the block itself.
    recency_prior: float = field(default_factory=lambda: _env_float("AML_RECENCY_PRIOR", 0.0))
    # Volume knob: widen the top hits into a same-session context window.  Item
    # size is uncapped by the platform and the answer budget is ~117,760 tokens,
    # so the ranked head can carry far more evidence than one turn per item.
    context_radius: int = field(default_factory=lambda: _env_int("AML_CONTEXT_RADIUS", 2))
    # Counting/list questions are graded on completeness *and* on not adding extras, so
    # widening them can cost more than it pays.  1 keeps the shared behaviour; 0 skips
    # widening for exhaustive queries only.
    exhaustive_widen: int = field(default_factory=lambda: _env_int("AML_EXHAUSTIVE_WIDEN", 1))
    context_top: int = field(default_factory=lambda: _env_int("AML_CONTEXT_TOP", 30))
    # ranks in [context_top, context_fill) get a radius-1 window: the head buys
    # precision, the middle keeps the list at single/degenerate granularity.
    context_fill: int = field(default_factory=lambda: _env_int("AML_CONTEXT_FILL", 100))
    # Widening absorbs turn ids, so the list can end up shorter than `top_k` (we
    # observed ~82 items).  The reader keeps a PREFIX of what we return and its
    # budget is ~117,760 tokens, so an unused slot is wasted evidence: refill it
    # from the pre-widening order.  On a small corpus the reserve is already
    # absorbed and this is a no-op; on a long conversation it is what keeps the
    # list at 100 distinct memories instead of ~60.  0 = off.
    context_fill_to: int = field(default_factory=lambda: _env_int("AML_CONTEXT_FILL_TO", 100))
    rrf_k: int = field(default_factory=lambda: _env_int("AML_RRF_K", 20))
    dense_weight: float = field(default_factory=lambda: _env_float("AML_DENSE_WEIGHT", 1.0))
    bm25_weight: float = field(default_factory=lambda: _env_float("AML_BM25_WEIGHT", 1.0))
    # Relative score floor for assembly.  0 keeps every candidate (maximum evidence
    # coverage, which is what a frozen reader wants); >0 trims the low-precision tail.
    score_floor: float = field(default_factory=lambda: _env_float("AML_SCORE_FLOOR", 0.0))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aml.sqlite3"

    @property
    def embed_cache_bytes(self) -> int:
        return max(1, self.embed_cache_mb) * 1024 * 1024


def load_settings() -> Settings:
    # Real environment variables win: `load_dotenv` never overrides what is already
    # set, so the container (compose env_file / `docker run -e`) is authoritative.
    # Otherwise try, in order: $AML_ENV_FILE, the working directory (a local
    # `uvicorn` run from the repo root), then `.env` beside the package — because
    # starting uvicorn from $HOME used to silently drop every setting in `.env`
    # (no reranker, no key) while the service still answered requests.
    # Tests set AML_SKIP_DOTENV=1 so a developer's real `.env` never leaks into them.
    if _env("AML_SKIP_DOTENV").lower() not in ("1", "true", "yes", "on"):
        try:
            from dotenv import find_dotenv, load_dotenv

            candidates = [_env("AML_ENV_FILE") or None, find_dotenv(usecwd=True) or None,
                          str(Path(__file__).resolve().parent.parent / ".env")]
            loaded = None
            for candidate in candidates:
                if not candidate:
                    continue
                path = Path(candidate)
                if path.is_file():
                    load_dotenv(path, override=False)
                    loaded = path
                    break
            if loaded is None:
                logger.debug("no .env file found; using the process environment only")
        except Exception:  # python-dotenv is optional at runtime
            pass
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
