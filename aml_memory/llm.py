"""Thin OpenAI-compatible chat client (academic board fixes this to gpt-4o-mini)."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any, Sequence

import httpx

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        self.enabled = bool(settings.llm_base_url and settings.llm_api_key)

    def _get_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(self.settings.llm_timeout_s, connect=10.0),
                    limits=httpx.Limits(max_connections=64),
                )
            return self._client

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        model: str | None = None,
        retries: int = 3,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("LLM is not configured (LLM_BASE_URL / LLM_API_KEY)")
        url = f"{self.settings.llm_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.llm_api_key}"}
        client = self._get_client()
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    body = resp.json()
                    return (body["choices"][0]["message"]["content"] or "").strip()
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last = RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(min(6.0, 0.5 * (2 ** attempt)) + random.random() * 0.2)
                    continue
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                time.sleep(min(6.0, 0.5 * (2 ** attempt)))
        raise RuntimeError(f"LLM call failed: {last}")

    def chat_json(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 512) -> dict[str, Any]:
        raw = self.chat(messages, json_mode=True, max_tokens=max_tokens)
        return extract_json(raw)


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for position in range(start, len(text)):
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:position + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}
