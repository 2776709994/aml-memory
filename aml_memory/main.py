"""HTTP surface for the AML Textual Memory track.

Contract notes that shape this file:
  * ``/add`` must return only after the data is durable *and* searchable, echoing
    ``request_id`` / ``user_id`` / ``session_id`` byte-for-byte.
  * ``/search`` must return ``{"data": [ ... ]}`` with at most ``top_k`` items,
    best first, no wrapper key other than ``data``.
  * ``/health`` must answer 2xx without credentials.
  * ``user_id`` is the only isolation boundary, so every read/write goes through
    the per-user namespace — no cross-user state ever touches a query.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings, load_settings
from .embeddings import Embedder
from . import governance

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9._\-]{6,}|[A-Za-z0-9]{32,})")


def _sanitize(text: str | None, limit: int = 160) -> str | None:
    """Short, secret-free description of a provider error.

    /health is unauthenticated by design, so nothing that looks like a key may reach
    it: providers like to echo the offending credential inside their error messages.
    """
    if not text:
        return None
    return _SECRET_PATTERN.sub("***", text)[:limit]
from .ingest import Ingestor, parse_messages
from .llm import LLM
from .rerank import Reranker
from .retrieve import Retriever
from .store import Store

VERSION = "1.13.0"
LOG = logging.getLogger("aml")
logging.basicConfig(
    level=os.environ.get("AML_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Every endpoint here is `def`, i.e. FastAPI runs it in the anyio worker pool, and
# the ingest/search gates block a worker while they wait.  With the default limit
# of 40 workers a burst of concurrent searches could hold every worker and leave
# /health queued behind it ?the container healthcheck would then fail and Docker
# would restart the service mid-run.  Two defences: a wider pool (below) and an
# async /health that never touches the pool (see the health endpoint).
def _widen_worker_pool(workers: int = 160) -> None:
    """Raise the anyio thread limit.  Must run inside the event loop: the limiter
    lives in a contextvar that does not exist yet at import time."""
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = workers
        LOG.info("worker pool widened to %d threads", workers)
    except Exception:  # pragma: no cover - depends on anyio internals
        LOG.warning("could not widen the worker pool; heavy load may stall /health")


class AddMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: Any = ""
    timestamp: Any = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = ""
    messages: list[Any] = Field(default_factory=list)
    user_id: str = ""
    session_id: str = ""


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Required, but an empty string is tolerated: a missing field is a client bug
    # (4xx, never retried) while an empty query is a legitimate empty answer.
    query: str
    options: Any = None
    user_id: str = ""
    top_k: Any = 100


def _normalize_path(path: str) -> str:
    path = (path or "").strip()
    if not path:
        return "/"
    return path if path.startswith("/") else "/" + path


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(
        settings.db_path,
        embed_model=settings.embed_model,
        cache_indexes=settings.index_cache_count,
        cache_rows=settings.index_cache_rows,
        embed_cache_rows=settings.embed_cache_rows,
    )
    llm = LLM(settings)
    embedder = Embedder(settings, store)
    reranker = Reranker(settings)
    ingestor = Ingestor(settings, llm)
    retriever = Retriever(settings, store, embedder, reranker, llm)

    ingest_gate = threading.BoundedSemaphore(max(1, settings.max_concurrent_ingest))
    search_gate = threading.BoundedSemaphore(max(1, settings.max_concurrent_search))
    # Counted so /admin/stats shows a provider outage instead of hiding it behind a
    # silently degraded ranking.
    embed_failures = {"add": 0, "search": 0}
    embed_failure_lock = threading.Lock()
    # "unknown" until the first embedding call; then "ok" or "failing". Reported by
    # /health so an external monitor can tell a dead provider from a dead process.
    embed_state = {"value": "unknown"}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _widen_worker_pool()
        LOG.info(
            "aml-memory %s starting: add=%s search=%s health=%s embed=%s/%s dim=%s rerank=%s llm=%s",
            VERSION, settings.add_path, settings.search_path, settings.health_path,
            settings.embed_provider, settings.embed_model, settings.embed_dim,
            settings.rerank_provider, settings.llm_model if llm.enabled else "disabled",
        )
        if settings.embed_provider not in {"hash", "offline", "mock"}:
            threading.Thread(target=embedder.warm, name="embed-warm", daemon=True).start()
        yield
        store.close()

    app = FastAPI(title="AML Memory Service", version=VERSION, lifespan=lifespan,
                  default_response_class=ORJSONResponse)
    for name, component in (
        ("store", store), ("embedder", embedder), ("reranker", reranker), ("retriever", retriever),
    ):
        setattr(app.state, name, component)

    def verify_auth(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
        api_key: str | None = Header(default=None, alias="Api-Key"),
    ) -> None:
        if settings.disable_auth or not settings.api_key:
            return
        supplied = ""
        for candidate in (authorization, x_api_key, api_key):
            if not candidate:
                continue
            value = candidate.strip()
            if value.lower().startswith(("bearer ", "token ")):
                value = value.split(" ", 1)[1].strip()
            if value:
                supplied = value
                break
        if not supplied:
            supplied = request.query_params.get("api_key", "") or ""
        if not supplied or not hmac.compare_digest(supplied, settings.api_key):
            raise HTTPException(status_code=401, detail="unauthorized")

    # One lock per request_id, so a retry storm on the same Add embeds the batch
    # once instead of once per retry: the platform may send up to 32 attempts with
    # an identical request_id while the first is still running.
    inflight: dict[str, threading.Lock] = {}
    inflight_guard = threading.Lock()

    def request_lock(request_id: str) -> threading.Lock:
        with inflight_guard:
            if len(inflight) > 50_000:
                for key in [k for k, lock in inflight.items() if not lock.locked()]:
                    inflight.pop(key, None)
            return inflight.setdefault(request_id, threading.Lock())

    @app.get(_normalize_path(settings.health_path))
    async def health() -> dict[str, Any]:
        # Async on purpose: it must answer even when every worker thread is busy
        # inside a long Add, otherwise the container healthcheck restarts us.
        #
        # `embedding` is the one line that would have caught the 2026-10-24 outage
        # from outside: the model was revoked (403), every Add degraded to text-only,
        # and /health still said "ok" — so the platform retried into a hole for hours
        # while the operator saw green. It is read from the embedder's own counters,
        # i.e. from calls that actually crossed the network: a probe text that hits the
        # embedding cache is not evidence that the provider still answers (that false
        # positive is exactly how a monitor decides the outage is over).
        provider_state, provider_error = embedder.remote_health
        return {
            "status": "ok",
            "service": "aml-memory",
            "version": VERSION,
            "embedding": provider_state if provider_state != "unknown" else embed_state["value"],
            "embedding_error": _sanitize(provider_error),
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "aml-memory",
            "version": VERSION,
            "endpoints": {
                "add": _normalize_path(settings.add_path),
                "search": _normalize_path(settings.search_path),
                "health": _normalize_path(settings.health_path),
            },
        }

    @app.post(_normalize_path(settings.add_path))
    def add(req: AddRequest, _: None = Depends(verify_auth)) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = str(req.request_id or "").strip()
        user_id = str(req.user_id or "").strip()
        session_id = str(req.session_id or "").strip()
        if not request_id:
            raise HTTPException(status_code=422, detail="request_id is required")
        if not user_id:
            raise HTTPException(status_code=422, detail="user_id is required")
        if not session_id:
            session_id = f"{user_id}:default"

        if store.request_seen(request_id):
            LOG.info("add replay request_id=%s user=%s units=0", request_id, user_id)
            return {"success": True, "request_id": req.request_id,
                    "user_id": req.user_id, "session_id": req.session_id}

        lock = request_lock(request_id)
        with lock:
            # Re-check under the lock: a concurrent retry of the same request may
            # have committed while we waited, and embedding twice is wasted money.
            if store.request_seen(request_id):
                LOG.info("add replay (raced) request_id=%s user=%s units=0", request_id, user_id)
                return {"success": True, "request_id": req.request_id,
                        "user_id": req.user_id, "session_id": req.session_id}
            messages = parse_messages(req.messages)
            units = ingestor.build_units(
                request_id=request_id, user_id=user_id, session_id=session_id, messages=messages,
            )
            with ingest_gate:
                # A provider outage must degrade the *ranking*, never the write: the
                # contract says an Add that returns 200 is persisted and searchable, and
                # the platform retries 32 times before failing the whole stage.  Storing
                # the turn without a vector keeps it retrievable through BM25.
                try:
                    vectors: list[Any] = (
                        embedder.embed_documents(
                            [unit.embed_text or unit.text for unit in units]
                        ) if units else []
                    )
                    embed_state["value"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    with embed_failure_lock:
                        embed_failures["add"] += 1
                    embed_state["value"] = "failing"
                    LOG.error(
                        "add %s: embedding failed, storing %d unit(s) for lexical retrieval: %s",
                        request_id, len(units), exc,
                    )
                    vectors = [None] * len(units)
                store.insert_add(
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    units=units,
                    vectors=vectors,
                    model=settings.embed_model,
                )
                # An explicit retraction ("forget that I…", "I no longer…") retires the
                # older turn it refers to, so the reader is never handed both halves of a
                # changed fact. Off by default; a wrong retirement is a silent answer loss.
                if settings.forget_matching and units:
                    try:
                        retired = governance.apply(
                            store, user_id=user_id, new_units=units,
                            threshold=settings.forget_threshold,
                        )
                        if retired:
                            LOG.info("add %s: %d turn(s) superseded for %s",
                                     request_id, retired, user_id)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("add %s: retraction pass skipped: %s", request_id, exc)
        elapsed = (time.perf_counter() - started) * 1000
        LOG.info(
            "add request_id=%s user=%s session=%s messages=%d units=%d %.0fms",
            request_id, user_id, session_id, len(messages), len(units), elapsed,
        )
        return {"success": True, "request_id": req.request_id,
                "user_id": req.user_id, "session_id": req.session_id}

    @app.post(_normalize_path(settings.search_path))
    def search(req: SearchRequest, _: None = Depends(verify_auth)) -> dict[str, Any]:
        started = time.perf_counter()
        user_id = str(req.user_id or "").strip()
        if not user_id:
            raise HTTPException(status_code=422, detail="user_id is required")
        query = str(req.query or "")
        options: list[str] = []
        if isinstance(req.options, str):
            options = [req.options]
        elif isinstance(req.options, (list, tuple)):
            options = [str(option) for option in req.options if str(option).strip()]
        try:
            top_k = int(req.top_k) if req.top_k is not None else settings.top_k_max
        except (TypeError, ValueError):
            top_k = settings.top_k_max
        # Clamp before the slice: a negative top_k would otherwise trim the best
        # items off the END of the list instead of asking for fewer.
        top_k = max(0, min(top_k, settings.top_k_max))

        with search_gate:
            items = retriever.search(query=query, options=options, user_id=user_id, top_k=top_k)
        items = items[:top_k]
        elapsed = (time.perf_counter() - started) * 1000
        LOG.info(
            "search user=%s top_k=%d returned=%d %.0fms query=%.120s",
            user_id, top_k, len(items), elapsed, query,
        )
        if settings.log_prompts and items:
            LOG.info("search payload user=%s items=%s", user_id, [(item["id"], item["score"]) for item in items[:10]])
        return {"data": items}

    @app.post("/admin/backfill")
    def backfill(limit: int = 500, _: None = Depends(verify_auth)) -> dict[str, Any]:
        """Embed units that were stored while the embedding provider was down.

        Only the Add path degrades; this repairs the hole afterwards.  Batched so a
        large backlog cannot exceed the provider's per-request token limit.
        """
        batch = max(1, min(int(limit or 500), 2000))
        rows = store.units_missing_vectors(batch)
        if not rows:
            return {"embedded": 0, "remaining": 0}
        try:
            vectors = embedder.embed_documents([text for _, text in rows])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"embedding still failing: {exc}") from exc
        written = store.set_unit_vectors(settings.embed_model, list(zip([uid for uid, _ in rows], vectors)))
        return {"embedded": written, "remaining": store.count_units_missing_vectors()}

    @app.get("/admin/stats")
    def stats(_: None = Depends(verify_auth)) -> dict[str, Any]:
        out = store.stats()
        out["rerank"] = reranker.stats()
        # Which models the running build actually uses.  Names, not secrets: an operator
        # (or the deployment agent) has to be able to confirm that the live process is
        # on the *compliant* embedder without reading `.env` by hand.
        out["models"] = {
            "embedding": f"{settings.embed_provider}/{settings.embed_model}",
            "embedding_dim": settings.embed_dim,
            "rerank": (
                f"{settings.rerank_provider}/{settings.rerank_model}"
                if settings.rerank_provider not in {"none", ""}
                else "none"
            ),
            "generative": "none",
        }
        with embed_failure_lock:
            out["embed_failures"] = dict(embed_failures)
        return out

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AML_PORT", "8000"))
    uvicorn.run(
        "aml_memory.main:app",
        host=os.environ.get("AML_HOST", "0.0.0.0"),
        port=port,
        log_level=os.environ.get("AML_LOG_LEVEL", "info").lower(),
        timeout_keep_alive=75,
        workers=1,
    )
