"""Test-wide guards.

The suite must be hermetic: it uses the offline `hash` embedder, never a paid API,
and never the developer's real `.env`.
"""
from __future__ import annotations

import os

# `load_settings()` honours this and skips `.env` entirely.
os.environ["AML_SKIP_DOTENV"] = "1"
# Belt and braces: even if a `.env` were loaded, no test may reach a network reranker.
os.environ["RERANK_PROVIDER"] = "none"
os.environ["AML_QUERY_LLM"] = "0"
os.environ["AML_INGEST_LLM"] = "off"
