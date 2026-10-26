"""Embedding backends.

The academic (open-source methods) board requires ``text-embedding-v4``, so the
production path is an HTTP embedding API.  ``hash`` is a deterministic offline
backend used by tests and by the local harness when no key is configured — it is
lexical feature hashing, weak but dependency-free.  ``local`` uses
sentence-transformers when it happens to be installed.
"""
from __future__ import annotations

import hashlib
import queue
import random
import re
import threading
import time
from collections import OrderedDict
from typing import Iterable, Sequence

import httpx
import numpy as np

from .config import Settings
from .store import Store, content_hash

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return (vec / norm).astype(np.float32)


class Embedder:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.provider = settings.embed_provider or "hash"
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self._local_model = None
        self._local_lock = threading.Lock()
        # The in-RAM cache is a fast path in front of the SQLite embed cache, so
        # it is bounded by BYTES (float16) rather than by a bare entry count: an
        # entry is 2 KB at dim 1024 and a 200k-entry default quietly meant 400 MB
        # of resident set on a 2 GB container.
        self._mem: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mem_bytes = 0
        self._mem_lock = threading.Lock()
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        # Provider-level health, counted only when the network is actually used.
        # A cache hit is not evidence that the provider still answers: on 2026-10-24
        # the account ran dry, a repeated probe text hit the embed cache, and the
        # "pipeline healthy" signal went green while every fresh Add failed.
        self._state_lock = threading.Lock()
        self.provider_calls = 0
        self.provider_failures = 0
        self.consecutive_failures = 0
        self.last_provider_error: str | None = None
        self.last_provider_ok_ns = 0

    # ------------------------------------------------------------- provider health
    @property
    def remote_health(self) -> tuple[str, str | None]:
        """("ok" | "failing" | "unknown", last error) from real provider calls only.

        Cache hits deliberately do not count: only a call that crossed the network can
        say whether the provider still answers.
        """
        with self._state_lock:
            if self.provider_calls == 0:
                return "unknown", None
            if self.consecutive_failures:
                return "failing", self.last_provider_error
            return "ok", None

    def _note_provider(self, ok: bool, error: str | None = None) -> None:
        with self._state_lock:
            self.provider_calls += 1
            if ok:
                self.consecutive_failures = 0
                self.last_provider_ok_ns = time.time_ns()
                self.last_provider_error = None
            else:
                self.provider_failures += 1
                self.consecutive_failures += 1
                self.last_provider_error = error

    # ------------------------------------------------------------------ public
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], text_type="query")[0]

    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Batch query-side embedding (asymmetric models need ``text_type=query``)."""
        return self._embed(list(texts), text_type="query")

    def warm(self) -> None:
        try:
            self._embed(["warmup"], text_type="query")
        except Exception:  # pragma: no cover - warmup is best effort
            pass

    # ----------------------------------------------------------------- internal
    def _embed(self, texts: Sequence[str], text_type: str) -> list[np.ndarray]:
        cleaned = [self._prepare(t) for t in texts]
        hashes = [content_hash(self.model, t) for t in cleaned]

        out: list[np.ndarray | None] = [None] * len(cleaned)
        pending: list[int] = []
        with self._mem_lock:
            for i, h in enumerate(hashes):
                vec = self._mem.get(h)
                if vec is not None:
                    self._mem.move_to_end(h)
                    out[i] = vec.astype(np.float32)
                else:
                    pending.append(i)
        if pending:
            cached = self.store.load_cached_vectors(self.model, [hashes[i] for i in pending])
            still: list[int] = []
            for i in pending:
                vec = cached.get(hashes[i])
                if vec is not None:
                    out[i] = vec
                    self._remember(hashes[i], vec)
                else:
                    still.append(i)
            pending = still

        if pending:
            fresh_texts = [cleaned[i] for i in pending]
            vectors = self._call_backend(fresh_texts, text_type)
            cache_rows: list[tuple[str, np.ndarray]] = []
            for i, vec in zip(pending, vectors):
                if vec is None:
                    vec = np.zeros(self.dim, dtype=np.float32)
                vec = normalize(vec.astype(np.float32))
                out[i] = vec
                self._remember(hashes[i], vec)
                cache_rows.append((hashes[i], vec))
            keep = text_type == "query" or self.settings.embed_cache_documents
            if keep:
                try:
                    self.store.cache_vectors(self.model, cache_rows)
                except Exception:
                    pass

        dim = self.dim
        result = []
        for vec in out:
            if vec is None:
                vec = np.zeros(dim, dtype=np.float32)
            if vec.shape[0] != dim:
                fixed = np.zeros(dim, dtype=np.float32)
                n = min(dim, vec.shape[0])
                fixed[:n] = vec[:n]
                vec = fixed
            result.append(vec)
        return result

    def _remember(self, h: str, vec: np.ndarray) -> None:
        small = vec.astype(np.float16)
        size = int(small.nbytes)
        with self._mem_lock:
            previous = self._mem.pop(h, None)
            if previous is not None:
                self._mem_bytes -= int(previous.nbytes)
            self._mem[h] = small
            self._mem_bytes += size
            while self._mem and (
                self._mem_bytes > self.settings.embed_cache_bytes
                or len(self._mem) > self.settings.embed_cache_size
            ):
                _, evicted = self._mem.popitem(last=False)
                self._mem_bytes -= int(evicted.nbytes)

    def _prepare(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return " "
        limit = max(64, self.settings.embed_max_chars)
        if len(text) > limit:
            text = text[:limit]
        return text

    # ---------------------------------------------------------------- backends
    def _call_backend(self, texts: list[str], text_type: str) -> list[np.ndarray | None]:
        provider = self.provider
        if provider in {"hash", "offline", "mock"}:
            return [self._hash_embed(t) for t in texts]
        if provider == "local":
            return self._local_embed(texts)
        if provider == "dashscope-native":
            return self._http_embed_native(texts, text_type)
        if provider in {"openai", "dashscope", "compatible", "siliconflow", "http", "azure"}:
            return self._http_embed_compatible(texts, text_type)
        return self._http_embed_compatible(texts, text_type)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vec[idx] += sign
            if len(token) > 3:
                for gram in (token[i:i + 3] for i in range(len(token) - 2)):
                    d2 = hashlib.md5(gram.encode("utf-8")).digest()
                    vec[int.from_bytes(d2[:4], "little") % self.dim] += 0.4 * (1.0 if d2[4] % 2 else -1.0)
        return vec

    def _local_embed(self, texts: list[str]) -> list[np.ndarray | None]:
        with self._local_lock:
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._local_model = SentenceTransformer(
                    self.settings.embed_model if "/" in self.settings.embed_model else "BAAI/bge-m3",
                    device="cuda" if _cuda_available() else "cpu",
                )
                self.dim = int(self._local_model.get_sentence_embedding_dimension())
            model = self._local_model
        arr = model.encode(
            texts, batch_size=max(8, self.settings.embed_batch), normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return [np.asarray(row, dtype=np.float32) for row in arr]

    def _get_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                # AML_REQUEST_TIMEOUT_S is the operator's ceiling on one outbound
                # call; 180 s is the working value (a batch of 10 texts never takes
                # that long, a dead endpoint must not hang an Add for 30 minutes).
                limit = float(max(5, min(180, self.settings.request_timeout_s)))
                self._client = httpx.Client(timeout=limit, limits=httpx.Limits(max_connections=32))
            return self._client

    def _http_embed_compatible(self, texts: list[str], text_type: str) -> list[np.ndarray | None]:
        base = self.settings.embed_base_url
        if not base:
            raise RuntimeError("EMBED_BASE_URL is not configured")
        url = base if base.endswith("/embeddings") else f"{base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.settings.embed_api_key:
            headers["Authorization"] = f"Bearer {self.settings.embed_api_key}"
        results: list[np.ndarray | None] = []
        batch_size = max(1, self.settings.embed_batch)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        lock = threading.Lock()
        output: list[list[np.ndarray | None]] = [[] for _ in batches]

        def work(index: int, batch: list[str]) -> None:
            payload: dict[str, object] = {
                "model": self.settings.embed_model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.settings.embed_dim:
                payload["dimensions"] = self.settings.embed_dim
            output[index] = self._post_with_retry(url, headers, payload, len(batch))

        _run_bounded(batches, work, self.settings.embed_concurrency)
        for chunk in output:
            results.extend(chunk)
        while len(results) < len(texts):
            results.append(None)
        return results[: len(texts)]

    def _http_embed_native(self, texts: list[str], text_type: str) -> list[np.ndarray | None]:
        base = self.settings.embed_base_url or "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.embed_api_key}"}
        results: list[np.ndarray | None] = []
        batch_size = max(1, self.settings.embed_batch)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        output: list[list[np.ndarray | None]] = [[] for _ in batches]

        def work(index: int, batch: list[str]) -> None:
            payload = {
                "model": self.settings.embed_model,
                "input": {"texts": batch},
                "parameters": {
                    "dimension": self.settings.embed_dim,
                    "text_type": "query" if text_type == "query" else "document",
                },
            }
            output[index] = self._post_with_retry(base, headers, payload, len(batch), native=True)

        _run_bounded(batches, work, self.settings.embed_concurrency)
        for chunk in output:
            results.extend(chunk)
        while len(results) < len(texts):
            results.append(None)
        return results[: len(texts)]

    def _post_with_retry(
        self, url: str, headers: dict[str, str], payload: dict[str, object], expected: int, native: bool = False,
    ) -> list[np.ndarray | None]:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    body = resp.json()
                    rows = body.get("data") or body.get("output", {}).get("embeddings") or []
                    vectors: list[np.ndarray | None] = []
                    if native:
                        rows = sorted(rows, key=lambda r: r.get("text_index", 0))
                    for row in rows:
                        vec = row.get("embedding")
                        vectors.append(np.asarray(vec, dtype=np.float32) if vec is not None else None)
                    while len(vectors) < expected:
                        vectors.append(None)
                    self._note_provider(True)
                    return vectors[:expected]
                if resp.status_code == 400 and "dimension" in resp.text.lower() and "dimensions" in payload:
                    # Some OpenAI-compatible relays reject `dimensions` outright. The
                    # model's native width is what we store anyway, so retry once without
                    # it instead of failing the whole Add.
                    payload = {k: v for k, v in payload.items() if k != "dimensions"}
                    LOG.warning("embedding endpoint rejected `dimensions`; retrying without it")
                    continue
                if resp.status_code in {400, 413} and len(payload.get("input", []) if isinstance(payload.get("input"), list) else []) > 1:
                    texts = payload["input"]  # type: ignore[index]
                    mid = max(1, len(texts) // 2)
                    left = self._post_with_retry(url, headers, {**payload, "input": texts[:mid]}, mid, native)
                    right = self._post_with_retry(url, headers, {**payload, "input": texts[mid:]}, len(texts) - mid, native)
                    return left + right
                if resp.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(8.0, 0.6 * (2 ** attempt)) + random.random() * 0.3)
                    last_error = RuntimeError(f"embedding HTTP {resp.status_code}: {resp.text[:300]}")
                    continue
                error = f"embedding HTTP {resp.status_code}: {resp.text[:300]}"
                self._note_provider(False, error)
                raise RuntimeError(error)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                time.sleep(min(8.0, 0.6 * (2 ** attempt)))
        self._note_provider(False, str(last_error))
        raise RuntimeError(f"embedding failed after retries: {last_error}")


def _run_bounded(items: Sequence[object], fn, concurrency: int) -> None:
    """Run ``fn(index, item)`` over items with a small worker pool."""
    work: queue.Queue[tuple[int, object]] = queue.Queue()
    for position, item in enumerate(items):
        work.put((position, item))
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                index, item = work.get_nowait()
            except queue.Empty:
                return
            try:
                fn(index, item)
            except BaseException as exc:  # noqa: BLE001
                with error_lock:
                    errors.append(exc)
                return

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(concurrency, len(items))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False
