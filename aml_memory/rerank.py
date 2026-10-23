"""Pluggable cross-encoder reranking (unrestricted by the AML rules)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

import httpx
import numpy as np

from .config import Settings

LOG = logging.getLogger("aml.rerank")


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = (settings.rerank_provider or "none").lower()
        self._model = None
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None
        self._counters = {"calls": 0, "failures": 0, "retries": 0, "fallbacks": 0}
        self._counter_lock = threading.Lock()

    @property
    def available(self) -> bool:
        if self.provider in {"none", "", "off", "disabled"}:
            return False
        if self.provider in {"dashscope", "http", "api"}:
            return bool(self.settings.rerank_api_key)
        return True

    def _bump(self, name: str) -> None:
        with self._counter_lock:
            self._counters[name] = self._counters.get(name, 0) + 1

    def stats(self) -> dict[str, int]:
        """Counters that make a silently degraded deployment visible."""
        with self._counter_lock:
            out = dict(self._counters)
        out["enabled"] = 1 if self.available else 0
        return out

    def score(self, query: str, docs: Sequence[str]) -> list[float] | None:
        if not docs or not self.available:
            return None
        self._bump("calls")
        try:
            if self.provider in {"local", "cross-encoder", "sentence-transformers"}:
                return self._local_scores(query, docs)
            if self.provider in {"dashscope", "http", "api"}:
                return self._http_scores(query, docs)
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the search
            self._bump("failures")
            LOG.warning("rerank failed on %d docs (%s: %s); retrying with a shorter window",
                        len(docs), type(exc).__name__, exc)
        # One retry with fewer documents: a timeout or a 5xx on 60 docs very often
        # succeeds on 20, and half a reranked window still beats pure fusion.
        short = list(docs[: max(8, min(20, len(docs)))])
        if short and len(short) < len(docs):
            try:
                time.sleep(0.2)
                if self.provider in {"local", "cross-encoder", "sentence-transformers"}:
                    vals = self._local_scores(query, short)
                else:
                    vals = self._http_scores(query, short)
                self._bump("retries")
                out = [0.0] * len(docs)
                out[: len(short)] = vals
                return out
            except Exception as exc:  # noqa: BLE001
                LOG.warning("rerank retry failed (%s: %s); falling back to fused order",
                            type(exc).__name__, exc)
        self._bump("fallbacks")
        return None

    # --------------------------------------------------------------- backends
    def _local_scores(self, query: str, docs: Sequence[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self.settings.rerank_model, max_length=512)
            model = self._model
        pairs = [(query, doc) for doc in docs]
        raw = model.predict(pairs, batch_size=max(4, self.settings.rerank_batch), show_progress_bar=False)
        return [float(value) for value in np.asarray(raw).reshape(-1)]

    def _http_scores(self, query: str, docs: Sequence[str]) -> list[float]:
        if self._client is None:
            limit = float(max(3, min(60, self.settings.rerank_timeout_s)))
            self._client = httpx.Client(timeout=limit)
        url = self.settings.rerank_base_url or (
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
        payload = {
            "model": self.settings.rerank_model or "gte-rerank-v2",
            "input": {"query": query, "documents": list(docs)},
            "parameters": {"return_documents": False, "top_n": len(docs)},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.rerank_api_key}",
        }
        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        results = (
            body.get("output", {}).get("results")
            or body.get("results")
            or body.get("data")
            or []
        )
        scores = [0.0] * len(docs)
        for row in results:
            index = row.get("index")
            if index is None:
                continue
            value = row.get("relevance_score", row.get("score", 0.0))
            if 0 <= int(index) < len(scores):
                scores[int(index)] = float(value)
        return scores


def normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [0.5 for _ in scores]
    return [(value - low) / (high - low) for value in scores]
