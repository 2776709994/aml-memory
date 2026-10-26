import os

os.environ.setdefault("AML_SKIP_DOTENV", "1")
os.environ.setdefault("EMBED_PROVIDER", "hash")
os.environ.setdefault("AML_DISABLE_AUTH", "1")


class _FakeResponse:
    def __init__(self, body):
        self._body = body
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self.body)


def _reranker(monkeypatch, provider, model, base, body):
    monkeypatch.setenv("RERANK_PROVIDER", provider)
    monkeypatch.setenv("RERANK_MODEL", model)
    monkeypatch.setenv("RERANK_BASE_URL", base)
    monkeypatch.setenv("RERANK_API_KEY", "test-key")
    monkeypatch.setenv("RERANK_DEPTH", "10")
    from aml_memory.config import load_settings
    from aml_memory.rerank import Reranker

    settings = load_settings()
    reranker = Reranker(settings)
    client = _FakeClient(body)
    reranker._client = client
    return reranker, client


def test_siliconflow_payload_and_score_mapping(monkeypatch):
    body = {"results": [
        {"index": 2, "relevance_score": 0.91},
        {"index": 0, "relevance_score": 0.42},
        {"index": 1, "relevance_score": 0.07},
    ]}
    reranker, client = _reranker(monkeypatch, "siliconflow", "Qwen/Qwen3-Reranker-4B",
                                 "https://api.siliconflow.cn/v1/rerank", body)
    scores = reranker.score("when did she start pottery?", ["a", "b", "c"])
    assert scores == [0.42, 0.07, 0.91]
    payload = client.calls[0]["json"]
    # SiliconFlow takes query/documents at the top level, not inside `input`.
    assert payload["query"] == "when did she start pottery?"
    assert payload["documents"] == ["a", "b", "c"]
    assert payload["top_n"] == 3
    assert client.calls[0]["url"].endswith("/rerank")
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_dashscope_payload_keeps_the_native_shape(monkeypatch):
    body = {"output": {"results": [{"index": 0, "relevance_score": 0.5},
                                   {"index": 1, "relevance_score": 0.2}]}}
    reranker, client = _reranker(monkeypatch, "dashscope", "gte-rerank-v2",
                                 "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                                 body)
    scores = reranker.score("q", ["a", "b"])
    assert scores == [0.5, 0.2]
    payload = client.calls[0]["json"]
    assert payload["input"] == {"query": "q", "documents": ["a", "b"]}
    assert payload["parameters"]["top_n"] == 2


def test_ranked_scores_are_reordered_by_index_not_by_row(monkeypatch):
    # The API returns rows sorted by score; a naive implementation would return them in
    # that order and silently scramble the mapping back onto the candidate list.
    body = {"results": [{"index": 3, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.8},
                        {"index": 0, "relevance_score": 0.1}]}
    reranker, _ = _reranker(monkeypatch, "siliconflow", "Qwen/Qwen3-Reranker-4B",
                            "https://api.siliconflow.cn/v1/rerank", body)
    scores = reranker.score("q", ["a", "b", "c", "d"])
    assert scores == [0.1, 0.8, 0.0, 0.9]
