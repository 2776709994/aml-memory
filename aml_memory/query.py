"""Query understanding: what kind of memory does this question need?

A fixed answer model reads whatever we retrieve.  The cheapest large win is to
retrieve *differently* per question type — a "how many times" question needs
exhaustive coverage, a "when" question needs date-anchored evidence, and a
"would they rather" question needs persona/preference facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .bm25 import tokenize
from .config import Settings
from .llm import LLM, extract_json
from .timeutil import extract_dates

_CAP_WORD = re.compile(r"\b([A-Z][a-z][A-Za-z'\-]{1,25})\b")

TEMPORAL_PAT = re.compile(
    r"\b(when|what date|which (day|month|year|week)|how (long|many (days|weeks|months|years) ago)|"
    r"last time|first time|most recent|latest|earliest|order of|before|after|during|since|until)\b",
    re.I,
)
COUNT_PAT = re.compile(r"\b(how many|how much|number of|count of|total number)\b", re.I)
LIST_PAT = re.compile(r"\b(list|all of|which ones|what are (the|all)|name (all|the)|enumerate)\b", re.I)
PREF_PAT = re.compile(r"\b(prefer|favorite|favourite|rather|like best|likes most|value most|care about|enjoy)\b", re.I)
UPDATE_PAT = re.compile(r"\b(no longer|not anymore|anymore|used to|still|changed|update|now\b|instead)\b", re.I)
REASON_PAT = re.compile(r"\b(why|reason|because|motivat|explain)\b", re.I)
RULE_PAT = re.compile(r"\b(format|bullet|json|table|respond|answer in|under \d+ words|steps?|procedure|policy)\b", re.I)
NEGATION_PAT = re.compile(r"\b(never|did not|didn't|has not|hasn't|not|refuse|cannot|can't)\b", re.I)


@dataclass
class QueryPlan:
    raw: str
    options: list[str]
    entities: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    qtypes: set[str] = field(default_factory=set)
    sub_queries: list[str] = field(default_factory=list)
    exhaustive: bool = False
    recency: bool = False
    source: str = "rules"

    def variants(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for text in [self.raw, *self.sub_queries]:
            key = text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(text.strip())
        return out


class QueryAnalyzer:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm

    def analyze(self, query: str, options: Sequence[str] | None = None) -> QueryPlan:
        options = [str(option) for option in (options or []) if str(option).strip()]
        plan = self._rule_plan(query, options)
        if self.settings.query_llm and self.llm.enabled:
            try:
                llm_plan = self._llm_plan(query, options)
                if llm_plan:
                    plan = self._merge(plan, llm_plan)
            except Exception:
                pass
        if not plan.sub_queries:
            plan.sub_queries = self._heuristic_subqueries(plan)
        # Option-aware retrieval: a multiple-choice question is often too vague on
        # its own; each option is a useful hypothesis-specific probe.
        if options and len(options) <= 6:
            plan.sub_queries.extend(options[:6])
        return plan

    # --------------------------------------------------------------- rule layer
    def _rule_plan(self, query: str, options: list[str]) -> QueryPlan:
        plan = QueryPlan(raw=query, options=options)
        lowered = query.lower()
        if TEMPORAL_PAT.search(query):
            plan.qtypes.add("temporal")
        if COUNT_PAT.search(query):
            plan.qtypes.add("count")
            plan.exhaustive = True
        if LIST_PAT.search(query):
            plan.qtypes.add("list")
            plan.exhaustive = True
        if PREF_PAT.search(query):
            plan.qtypes.add("preference")
        if UPDATE_PAT.search(query):
            plan.qtypes.add("update")
        if REASON_PAT.search(query):
            plan.qtypes.add("reason")
        if RULE_PAT.search(query):
            plan.qtypes.add("rule")
        if NEGATION_PAT.search(query):
            plan.qtypes.add("negation")
        if re.search(r"\b(previously|earlier|before|originally)\b", lowered):
            plan.qtypes.add("temporal")
        if re.search(r"\b(now|currently|these days|still)\b", lowered):
            plan.recency = True
        if re.search(r"\b(most recent|latest|last|newest|current)\b", lowered):
            plan.recency = True
        if options:
            plan.qtypes.add("mcq")
        plan.entities = _entities(query)
        plan.dates = [d.strftime("%Y-%m-%d") for d in extract_dates(query)]
        if len(plan.entities) >= 2 or re.search(r"\b(both|respectively|each other|related|connection|relationship)\b", lowered):
            plan.qtypes.add("multihop")
        return plan

    # ---------------------------------------------------------------- llm layer
    def _llm_plan(self, query: str, options: list[str]) -> dict[str, Any]:
        option_block = ""
        if options:
            option_block = "\nOptions:\n" + "\n".join(f"- {option}" for option in options)
        prompt = (
            "You prepare a retrieval plan for a long-term memory system. You never answer the question.\n"
            "Given the question, output JSON with:\n"
            '  "sub_queries": 2-4 short standalone search queries that together would surface the evidence, '
            "including one that focuses on the specific named entities and one that focuses on the relation "
            "or event being asked about;\n"
            '  "entities": the people/places/organisations/objects the evidence must mention;\n'
            '  "qtype": one of fact, temporal, count, list, preference, update, reason, rule, negation, mcq, multihop;\n'
            '  "keywords": up to 8 content words that a supporting memory would contain.\n'
            "Do not include the answer. Do not invent entities that are not implied by the question.\n\n"
            f"Question: {query}{option_block}"
        )
        raw = self.llm.chat(
            [{"role": "system", "content": "You output only compact JSON."},
             {"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=320,
        )
        parsed = extract_json(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _merge(self, plan: QueryPlan, llm_plan: dict[str, Any]) -> QueryPlan:
        plan.source = "llm"
        subs = llm_plan.get("sub_queries") or llm_plan.get("subqueries") or []
        if isinstance(subs, str):
            subs = [subs]
        cleaned: list[str] = []
        for item in subs:
            if isinstance(item, dict):
                item = item.get("query") or item.get("q") or ""
            text = str(item).strip()
            if 2 < len(text) <= 240:
                cleaned.append(text)
        plan.sub_queries = cleaned[:4]
        entities = llm_plan.get("entities") or []
        if isinstance(entities, str):
            entities = [entities]
        plan.entities = list(dict.fromkeys([*plan.entities, *[str(e).strip() for e in entities if str(e).strip()]]))[:12]
        qtype = str(llm_plan.get("qtype") or "").strip().lower()
        if qtype:
            plan.qtypes.add(qtype)
            if qtype in {"count", "list"}:
                plan.exhaustive = True
            if qtype in {"temporal", "update"}:
                plan.qtypes.add("temporal")
        keywords = llm_plan.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        kw = " ".join(str(k).strip() for k in keywords if str(k).strip())
        if kw and len(kw) > 3 and kw.lower() not in plan.raw.lower():
            plan.sub_queries.append(plan.raw + " " + kw)
        return plan

    def _heuristic_subqueries(self, plan: QueryPlan) -> list[str]:
        query = plan.raw
        subs: list[str] = []
        if plan.entities:
            subs.append(" ".join(plan.entities[:4]) + " " + " ".join(
                token for token in tokenize(query) if token not in {e.lower() for e in plan.entities}
            )[:120])
        if plan.qtypes & {"temporal"}:
            subs.append(f"{query} date time year month")
        if plan.qtypes & {"preference"}:
            subs.append(f"{query} likes dislikes prefers favorite")
        if plan.qtypes & {"count", "list"}:
            subs.append(f"{query} list each every")
        return [s.strip() for s in subs if s.strip()][:3]


def _entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in _CAP_WORD.finditer(query or ""):
        word = match.group(1)
        if word.lower() in {"what", "when", "where", "who", "which", "how", "why", "did", "does", "is", "are", "was", "were", "the", "i"}:
            continue
        if match.start() == 0 and not re.match(r"^[A-Z]{2,}", word):
            continue
        entities.append(word)
    quoted = re.findall(r"[\"'“”]([^\"'“”]{2,60})[\"'“”]", query or "")
    for item in quoted:
        entities.append(item.strip())
    return list(dict.fromkeys(entities))[:12]
