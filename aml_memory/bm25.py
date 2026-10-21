"""Compact in-process BM25 over a single user namespace.

The retrieval scope of an AML Search call is one ``user_id`` (one conversation),
so a per-user inverted index built on demand is both cheap and exact.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[\u4e00-\u9fff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

STOPWORDS = frozenset(
    """a an the and or but if then than that this these those there here of in on at to for from by with
    without about into over under again further once is are was were be been being do does did doing have
    has had having i you he she it we they me him her them my your his its our their what which who whom
    when where why how all any both each few more most other some such no nor not only own same so too very
    can will just should now also did does""".split()
)


def stem(token: str) -> str:
    if len(token) <= 4 or token in STOPWORDS:
        return token
    for suffix, cut in (("ies", 3), ("ses", 2), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if token.endswith(suffix) and len(token) - cut >= 3:
            base = token[: len(token) - cut]
            if suffix == "ies":
                return base + "y"
            return base
    return token


def tokenize(text: str, *, keep_stop: bool = False) -> list[str]:
    lowered = (text or "").lower()
    tokens: list[str] = []
    for raw in _WORD_RE.findall(lowered):
        if _CJK_RE.match(raw):
            tokens.append(raw)
            continue
        if not keep_stop and raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    # CJK bigrams give a usable lexical signal without a segmenter.
    cjk = _CJK_RE.findall(lowered)
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])
    return tokens


@dataclass
class BM25:
    k1: float = 1.2
    b: float = 0.75
    doc_len: list[int] = field(default_factory=list)
    avg_len: float = 0.0
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    n_docs: int = 0

    @classmethod
    def build(cls, docs: list[str]) -> "BM25":
        index = cls()
        index.n_docs = len(docs)
        df: Counter[str] = Counter()
        for position, doc in enumerate(docs):
            tokens = tokenize(doc)
            index.doc_len.append(len(tokens))
            counts = Counter(tokens)
            for token, tf in counts.items():
                index.postings.setdefault(token, []).append((position, tf))
                df[token] += 1
        total = sum(index.doc_len)
        index.avg_len = (total / len(index.doc_len)) if index.doc_len else 0.0
        index._idf = {
            token: math.log(1.0 + (index.n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        return index

    _idf: dict[str, float] = field(default_factory=dict)

    def search(self, query: str, limit: int = 200) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: dict[int, float] = {}
        q_counts = Counter(q_tokens)
        avg = self.avg_len or 1.0
        for token, qtf in q_counts.items():
            idf = self._idf.get(token)
            if idf is None:
                continue
            for doc_id, tf in self.postings.get(token, ()):
                norm = self.k1 * (1 - self.b + self.b * (self.doc_len[doc_id] / avg))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / (tf + norm) * min(qtf, 3)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
