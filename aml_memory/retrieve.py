"""Retrieval: hybrid recall → optional rerank → prompt-shaped assembly.

Scoring model that drives every choice here: the AML harness concatenates whatever
we return into a fixed answer prompt, so the objective is not "rank one document
first" but "spend 100 slots on the densest, most self-contained, best-ordered
evidence".  Hence: multi-query recall for coverage, cross-encoder for precision,
adjacent-turn merging to cut prompt cost, and dedupe so a repeated fact cannot
crowd out a distinct one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .bm25 import BM25, tokenize
from .config import Settings
from .embeddings import Embedder
from .llm import LLM
from .query import QueryAnalyzer, QueryPlan
from .rerank import Reranker, normalize_scores
from .store import Store, Unit, UserIndex

_MAX_VARIANTS = 6


def rrf(lists: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None) -> dict[int, float]:
    """Reciprocal rank fusion — scale-free combination of dense + lexical lists."""
    fused: dict[int, float] = {}
    for position, ranked in enumerate(lists):
        weight = 1.0
        if weights is not None and position < len(weights):
            weight = float(weights[position])
        if weight <= 0:
            continue
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + rank + 1)
    return fused


def _tokens(text: str) -> set[str]:
    return set(tokenize(text, keep_stop=True))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _body_text(text: str) -> str:
    """Drop the "<speaker> (<date>): " header before comparing two turns.

    Duplicate detection is about the statement, not the envelope: the same sentence
    said on two dates must look identical, and keeping the date in the token set
    makes two repeats of one fact look only 0.67 similar.
    """
    head, sep, rest = text.partition(": ")
    if not sep or len(head) > 80 or "(" not in head:
        return text
    return rest


def _recency_key(unit: Any) -> tuple[int, int]:
    """Order turns by when they were said; undated turns sort first (oldest)."""
    return (int(unit.created_ms or 0), int(unit.seq or 0))


@dataclass
class Block:
    ids: list[str]
    content: str
    score: float
    position: int


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        llm: LLM,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.analyzer = QueryAnalyzer(settings, llm)
        self._bm25: dict[tuple[str, int], BM25] = {}
        self._bm25_lock = threading.Lock()

    # ------------------------------------------------------------------ entry
    def search(
        self,
        *,
        query: str,
        options: Sequence[str] | None,
        user_id: str,
        top_k: int,
    ) -> list[dict]:
        index = self.store.user_index(user_id)
        if not index.ids:
            return []
        top_k = max(1, min(int(top_k or self.settings.top_k_max), self.settings.top_k_max))
        plan = self.analyzer.analyze(query, options)
        variants = plan.variants()[:_MAX_VARIANTS] or [query]

        lists: list[list[int]] = []
        weights: list[float] = []
        dense_weight = max(0.0, float(self.settings.dense_weight))
        bm25_weight = max(0.0, float(self.settings.bm25_weight))
        dense_matrix = index.matrix
        if dense_weight > 0 and dense_matrix is not None and dense_matrix.any():
            try:
                qvecs = np.stack(self.embedder.embed_queries(variants)).astype(np.float32)
                if qvecs.shape[1] == dense_matrix.shape[1]:
                    sims = dense_matrix @ qvecs.T
                    depth = min(self.settings.candidate_depth, len(index.ids))
                    for column in range(sims.shape[1]):
                        order = np.argsort(-sims[:, column])[:depth]
                        lists.append(order.tolist())
                        weights.append(dense_weight)
            except Exception:
                pass

        bm25 = self._bm25_for(index) if bm25_weight > 0 else None
        if bm25 is not None:
            for variant in variants:
                hits = bm25.search(variant, limit=self.settings.candidate_depth)
                if hits:
                    lists.append([doc for doc, _score in hits])
                    weights.append(bm25_weight)

        if not lists:
            return []

        fused = rrf(lists, k=self.settings.rrf_k, weights=weights)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ranked = ranked[: self.settings.candidate_depth]
        if not ranked:
            return []

        scores = {idx: score for idx, score in ranked}
        scores = self._rerank(query, index, scores)
        scores = self._boost(index, plan, scores)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        blocks = self._assemble(index, ordered, plan, top_k)
        return [
            {"id": "+".join(block.ids), "content": block.content, "score": round(block.score, 6)}
            for block in blocks
        ]

    # ------------------------------------------------------------------ stages
    def _bm25_for(self, index: UserIndex) -> BM25 | None:
        key = (index.user_id, index.version)
        with self._bm25_lock:
            cached = self._bm25.get(key)
        if cached is not None:
            return cached
        try:
            built = BM25.build([unit.text for unit in index.units])
        except Exception:
            return None
        with self._bm25_lock:
            if len(self._bm25) > 64:
                self._bm25.clear()
            self._bm25[key] = built
        return built

    def _rerank(self, query: str, index: UserIndex, scores: dict[int, float]) -> dict[int, float]:
        if not self.reranker.available:
            return scores
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.settings.rerank_depth]
        if len(top) <= 4:
            return scores
        docs = [index.units[idx].text[: self.settings.max_content_chars] for idx, _ in top]
        raw = self.reranker.score(query, docs)
        if not raw or len(raw) != len(docs):
            return scores
        rerank_norm = normalize_scores(raw)
        fuse_norm = normalize_scores([score for _, score in top])
        blended: dict[int, float] = dict(scores)
        for position, (idx, _) in enumerate(top):
            blended[idx] = 0.7 * rerank_norm[position] + 0.3 * fuse_norm[position]
        return blended

    def _boost(self, index: UserIndex, plan: QueryPlan, scores: dict[int, float]) -> dict[int, float]:
        top_score = max(scores.values()) if scores else 0.0
        if top_score <= 0:
            return scores
        entities = [entity.lower() for entity in plan.entities if len(entity) > 1][:8]
        dates = plan.dates[:3]
        recency = plan.recency
        update_boost = float(self.settings.update_boost)
        recency_prior = float(self.settings.recency_prior)
        total = max(1, len(index.units))
        role_prior = float(self.settings.assistant_prior)
        out: dict[int, float] = {}
        for idx, score in scores.items():
            unit = index.units[idx]
            value = score
            if role_prior != 1.0 and str(unit.meta.get("role", "")).lower() in ("assistant", "ai", "bot"):
                # Advice-heavy assistant turns crowd out the user's own statements on
                # questions about the user's life; the prior trims them without
                # deleting them (they still answer "what did you recommend?").
                value *= role_prior
            if entities:
                text = unit.text.lower()
                hits = sum(1 for entity in entities if entity in text)
                if hits:
                    value *= 1.0 + min(0.18, 0.045 * hits)
            if dates and unit.date_str:
                if unit.date_str in dates:
                    value *= 1.10
                else:
                    for date in dates:
                        if abs(_day_distance(unit.date_str, date)) <= 3:
                            value *= 1.05
                            break
            if recency:
                value *= 1.0 + 0.06 * (unit.seq / total)
            if recency_prior > 0.0:
                value *= 1.0 + recency_prior * (unit.seq / total)
            if update_boost > 0.0 and "update" in plan.qtypes:
                # "Do I still .../what is X now" questions are answered from the newest
                # statement, and the answer prompt prefers the most recent fact when two
                # conflict. The reader only sees a prefix, so the newer statement has to
                # be ranked above the older one rather than left to the reader to weigh.
                value *= 1.0 + update_boost * (unit.seq / total)
            out[idx] = value
        return out

    def _assemble(
        self,
        index: UserIndex,
        ordered: list[tuple[int, float]],
        plan: QueryPlan,
        top_k: int,
    ) -> list[Block]:
        units = index.units
        cap = top_k if not plan.exhaustive else min(top_k, max(1, self.settings.exhaustive_cap))
        top_score = ordered[0][1] if ordered else 0.0
        base_floor = self.settings.score_floor
        if base_floor <= 0:
            floor = 0.0
        else:
            floor = top_score * (base_floor * 1.8 if plan.exhaustive else base_floor)
        budget = self.settings.total_char_budget
        prefer_newer = self.settings.dedupe_prefer == "newer"
        spent = 0

        selected: list[tuple[int, float]] = []
        pos_of: dict[int, int] = {}
        seen_exact: set[str] = set()
        seen_tokens: list[tuple[int, set[str]]] = []
        window = 120
        for idx, score in ordered:
            if len(selected) >= cap:
                break
            if score < floor and len(selected) >= self.settings.min_items:
                break
            unit = units[idx]
            body = _body_text(unit.text)
            normalized = " ".join(body.lower().split())
            if normalized in seen_exact:
                continue
            tokens = _tokens(body)
            dup_at = -1
            for k in range(max(0, len(seen_tokens) - window), len(seen_tokens)):
                if _jaccard(tokens, seen_tokens[k][1]) > self.settings.dedupe_jaccard:
                    dup_at = k
                    break
            if dup_at >= 0:
                # Near-duplicate.  Which half survives decides update questions:
                # "I moved to Denver" must beat "I live in Boston" for "where do
                # I live now?", even when the stale half ranks higher.
                kept_idx, _ = seen_tokens[dup_at]
                if prefer_newer and _recency_key(unit) > _recency_key(units[kept_idx]):
                    slot = pos_of[kept_idx]
                    selected[slot] = (idx, score)
                    pos_of.pop(kept_idx, None)
                    pos_of[idx] = slot
                    seen_tokens[dup_at] = (idx, tokens)
                    seen_exact.add(normalized)
                    spent += len(unit.text) - len(units[kept_idx].text)
                continue
            selected.append((idx, score))
            pos_of[idx] = len(selected) - 1
            seen_exact.add(normalized)
            seen_tokens.append((idx, tokens))
            spent += len(unit.text)
            if spent >= budget:
                break

        if not selected:
            return []

        blocks = self._merge_adjacent(index, selected)
        blocks = self._expand_neighbours(index, blocks)
        blocks.sort(key=lambda block: block.score, reverse=True)
        reserve = list(blocks)
        blocks = self._widen_top(index, blocks)
        blocks = self._refill(index, blocks, reserve)

        out: list[Block] = []
        spent = 0
        for block in blocks:
            content = block.content
            limit = self.settings.max_content_chars * 3
            if len(content) > limit:
                content = content[:limit].rsplit(" ", 1)[0] + " ..."
            if spent + len(content) > budget and out:
                break
            out.append(Block(ids=block.ids, content=content, score=block.score, position=block.position))
            spent += len(content)
        return out[:top_k]

    def _merge_adjacent(self, index: UserIndex, selected: list[tuple[int, float]]) -> list[Block]:
        """Join consecutive turns into one memory block (cheaper, better context).

        Only same-speaker runs are joined and the run is capped: the answer prompts
        bucket memories per speaker, and one speaker per line keeps every block
        attributable no matter how the platform partitions our flat result.
        """
        limit = max(1, self.settings.merge_max_units)
        position_of = {unit.id: position for position, unit in enumerate(index.units)}
        chosen = sorted(((position_of[unit.id], score) for unit, score in ((index.units[i], s) for i, s in selected)))
        blocks: list[Block] = []
        run: list[tuple[int, float]] = []

        def flush() -> None:
            if not run:
                return
            members = [index.units[position] for position, _ in run]
            content = "\n".join(unit.text for unit in members)
            blocks.append(
                Block(
                    ids=[unit.id for unit in members],
                    content=content,
                    score=max(score for _, score in run),
                    position=run[0][0],
                )
            )
            run.clear()

        for position, score in chosen:
            same_speaker = bool(run) and index.units[position].speaker == index.units[run[-1][0]].speaker
            same_session = bool(run) and index.units[position].session_id == index.units[run[-1][0]].session_id
            contiguous = bool(run) and position == run[-1][0] + 1
            if run and contiguous and same_speaker and same_session and score >= 0.4 * run[-1][1] and len(run) < limit:
                run.append((position, score))
            else:
                flush()
                run.append((position, score))
        flush()
        return blocks

    def _expand_neighbours(self, index: UserIndex, blocks: list[Block]) -> list[Block]:
        depth = self.settings.expand_neighbors
        if depth <= 0 or not blocks:
            return blocks
        position_of = {unit.id: position for position, unit in enumerate(index.units)}
        used = {unit_id for block in blocks for unit_id in block.ids}
        for block in blocks[: max(6, len(blocks) // 2)]:
            if len(block.content) > self.settings.max_content_chars * 2:
                continue
            anchor = position_of.get(block.ids[0])
            if anchor is None:
                continue
            speaker = index.units[anchor].speaker
            session = index.units[anchor].session_id
            added_left: list[str] = []
            added_right: list[str] = []
            for step in range(1, depth + 1):
                left = anchor - step
                right = position_of.get(block.ids[-1], anchor) + step
                if (
                    left >= 0
                    and index.units[left].id not in used
                    and index.units[left].speaker == speaker
                    and index.units[left].session_id == session
                ):
                    added_left.insert(0, index.units[left].text)
                    used.add(index.units[left].id)
                if (
                    right < len(index.units)
                    and index.units[right].id not in used
                    and index.units[right].speaker == speaker
                    and index.units[right].session_id == session
                ):
                    added_right.append(index.units[right].text)
                    used.add(index.units[right].id)
            if added_left:
                block.content = "\n".join([*added_left, block.content])
            if added_right:
                block.content = "\n".join([block.content, *added_right])
        return blocks

    def _widen_top(self, index: UserIndex, blocks: list[Block]) -> list[Block]:
        """Give the highest-ranked hits a wider context window.

        The platform keeps a PREFIX of our list, and the cap is 100 ITEMS, not 100
        characters; item size is uncapped and the answer budget is ~117,760 tokens,
        so a ranked hit is worth much more when it carries the turns around it.
        Multihop questions are exactly the case where two evidence turns sit in the
        same window but only one is retrieved.  Only the top `context_top` blocks
        are widened, so the tail keeps single-turn granularity and recall does not
        pay for the extra volume.
        """
        radius = self.settings.context_radius
        if radius <= 0 or not blocks:
            return blocks
        fill = self.settings.context_fill
        units = index.units
        position_of = {unit.id: position for position, unit in enumerate(units)}
        absorbed: set[str] = set()
        out: list[Block] = []
        for rank, block in enumerate(blocks):
            if rank < self.settings.context_top:
                block_radius = radius
            elif fill > self.settings.context_top and rank < fill:
                block_radius = 1
            else:
                block_radius = 0
            if block_radius <= 0:
                if any(unit_id in absorbed for unit_id in block.ids):
                    continue
                out.append(block)
                continue
            anchor = position_of.get(block.ids[0])
            if anchor is None:
                out.append(block)
                continue
            session = units[anchor].session_id
            last = position_of.get(block.ids[-1], anchor)
            low = anchor
            while low - 1 >= 0 and anchor - (low - 1) <= block_radius and units[low - 1].session_id == session:
                low -= 1
            high = last
            while high + 1 < len(units) and (high + 1) - last <= block_radius and units[high + 1].session_id == session:
                high += 1
            members = [
                units[position] for position in range(low, high + 1)
                if units[position].id not in absorbed
            ]
            if not members:
                continue
            absorbed.update(unit.id for unit in members)
            out.append(
                Block(
                    ids=[unit.id for unit in members],
                    content="\n".join(unit.text for unit in members),
                    score=block.score,
                    position=block.position,
                )
            )
        return out

    def _refill(self, index: UserIndex, blocks: list[Block], reserve: list[Block]) -> list[Block]:
        """Pad the widened list back up to `context_fill_to` items.

        Widening absorbs the turn ids it pulled into a window, so the returned list
        comes out shorter than `top_k` (~82 of 100 measured on LoCoMo-Refined).
        The reader keeps a PREFIX of our list and has ~117,760 tokens of budget,
        so every unused slot is evidence we simply did not deliver.  Refill from
        the pre-widening order, skipping blocks whose turns are already covered.
        """
        target = self.settings.context_fill_to
        if target <= 0 or len(blocks) >= target:
            return blocks
        covered = {unit_id for block in blocks for unit_id in block.ids}
        out = list(blocks)
        for block in reserve:
            if len(out) >= target:
                break
            if any(unit_id in covered for unit_id in block.ids):
                continue
            covered.update(block.ids)
            out.append(block)
        return out


def _day_distance(left: str, right: str) -> int:
    import datetime as _dt

    try:
        a = _dt.date.fromisoformat(left)
        b = _dt.date.fromisoformat(right)
    except ValueError:
        return 9999
    return abs((a - b).days)
