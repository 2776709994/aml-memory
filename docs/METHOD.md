# Method disclosure — aml-memory

Submission for the **Agent Memory Leaderboard, Cycle 2**, division *Open-source
Methods*, track *Textual Memory*. This document is the method statement required
by the rules: what the system does, which models it uses, what it does **not** do,
and how every number in the paper/README was produced.

Pinned commit: see the `SOURCE_COMMIT` field in the submission form.
License: MIT (`LICENSE`). Entry point: `aml_memory/main.py`.

---

## 1. One-paragraph summary

`aml-memory` is a **retrieval-only** memory service. `add` splits an incoming
conversation chunk into one record per turn, rewrites relative time expressions
into absolute dates **at their spoken granularity**, and stores the record with its
speaker, absolute date, neighbour links and embedding. `search` embeds the query
with the same model, retrieves candidates with **dense cosine similarity over the
user's own records plus Okapi BM25**, fuses the two ranked lists with **weighted
reciprocal rank fusion on ranks**, optionally **re-ranks the fused head with a
cross-encoder**, and then assembles at most `top_k` items, guaranteeing that (a) no
turn is returned twice, (b) each item is attributable to a single speaker, and
(c) the highest-ranked hits carry the turns around them. No generative model runs
inside `add` or `search`, so the memory itself is a deterministic function of the
input.

---

## 2. Write path (`POST /add`)

1. **Chunking.** The platform already chunks at 20 messages / 2,000 words; the
   service keeps one *unit* per message and never merges across `request_id`
   boundaries, so the caller's chunking is preserved. A message longer than the unit
   limit is split at sentence boundaries, and **every part repeats the
   `Speaker (date):` header** (continuations are marked `[part N]`), so no fragment
   ever reaches the reader anonymous or undated.
2. **Speaker resolution.** The speaker is taken from an inline `Name:` prefix when
   present (the LoCoMo/PersonaMem flattening), otherwise from the message role.
   The speaker is written into the record text, not into a side channel, because
   the platform's answer prompts bucket memories per speaker and may re-partition
   our flat result list.
3. **Time anchoring.** Every relative expression (`yesterday`, `last Monday`,
   `three weeks ago`, `this month`, `last year`, …) is rewritten in place as
   `relative phrase (absolute date)`, using the message timestamp as the anchor and
   **respecting the granularity of the phrase**: `yesterday → (7 May 2023)`,
   `last week → (the week of 1 May 2023)`, `this month → (May 2023)`,
   `last year → (2022)`. Over-precise anchors are worse than none: the judge
   compares dates at the granularity of the gold answer, so injecting a day where
   the conversation only supports a month manufactures contradictions.
4. **Multimodal captions.** A message's image caption is inlined as
   `[shares a photo: …]` so it is retrievable by text.
5. **Persistence.** Units + vectors are written in **one `BEGIN IMMEDIATE`
   transaction**, which also claims the `request_id`. The `200` is returned only
   after the commit, and the per-user search index is invalidated, so the memory is
   searchable the moment `add` returns. A retried `request_id` (the platform retries
   up to 32 times, and two retries can be in flight at once) is a no-op — including
   after a restart, because the claim lives in SQLite. Retries that arrive while the
   first attempt is still running wait on a per-`request_id` lock and re-check the
   claim, so one Add is embedded once even under a retry storm.
6. **Storage.** Units and their embeddings live in SQLite; embeddings are stored as
   **int8 with a per-vector scale** (dim bytes instead of dim × 2) on 8 KiB pages, and
   ingested texts are not duplicated into the embedding cache. Measured: 1.9 KB per
   turn against 11.1 KB for the float16 build, with identical retrieval numbers
   (R@1 .748 / R@10 .957 / MRR .827 on LoCoMo-Refined). `/admin/stats` reports
   `db_bytes` and `embed_cache` rows so growth is visible while a run is in flight.

## 3. Read path (`POST /search`)

1. **Query understanding without a generative model.** A rule-based analyser tags
   the question type (temporal / counting / listing / preference / update /
   reasoning / rule / negation / multiple-choice) and emits a small set of query
   variants (the question, its `options` concatenated for MCQ, and entity-focused
   variants). This is deterministic; `AML_QUERY_LLM=0` in the submitted build.
2. **Dense channel.** Query vectors (one per variant) are dotted against the user's
   normalised matrix — exhaustive search, no ANN. `user_id` is the isolation
   boundary and one user is one conversation, so exact search is both correct and
   cheap; the matrix is cached per user with an LRU that is bounded by **bytes**,
   not by user count.
3. **Lexical channel.** Okapi BM25 (k1=1.5, b=0.75) with a tokenizer that handles
   CJK via character bigrams and a light English stemmer.
4. **Fusion.** Weighted **RRF on ranks** (`k=20`), dense weight 1.0, BM25 weight
   1.0, candidate depth 500. Ranks, not scores: the two channels are not
   calibrated, and rank fusion is what survives a change of embedder.
5. **Rerank.** The fused head (60 candidates) is scored by `gte-rerank-v2` through
   the DashScope rerank endpoint; on failure the reranked scores are ignored and the
   fused order stands (the service degrades, it does not error).
6. **Assembly.** Dedupe (exact + Jaccard > `AML_DEDUPE_JACCARD`, default .85, compared
   on the statement body with the speaker/date header stripped, so the same sentence
   said on two dates is recognised as one memory), join short same-speaker runs
   (≤3 turns), then **two-tier context widening**: each of the 30 highest-ranked
   blocks is expanded into its same-session window of ±2 turns, ranks 30–99 get a
   ±1 window, and every turn absorbed by a window is removed from the tail so no
   turn is returned twice. Merging and neighbour expansion are scoped to one
   `session_id` as well, so an item never splices two sessions into one dated line.
   Last, emit items until `top_k` or a character budget is reached. Item ids are
   `+`-joined unit ids, so a returned item is traceable to the turns it contains.

**Why volume is allowed:** the platform caps the number of items (100) but not the
size of an item, and it truncates the answer prompt by *keeping a prefix of our
list*. The rank order and what sits inside an item therefore matter, and the score
field does not. The assembler is built around that fact: packing neighbours into the
top items lifted R@1 from .591 to **.748** and MRR from .691 to **.827** on
LoCoMo-Refined with everything else held fixed, because a multihop question's second
evidence turn usually sits one or two turns away from the retrieved one.

---

## 4. Models

| component | model | notes |
|---|---|---|
| document + query embeddings | **`text-embedding-v4`**, `dimensions=1024`, L2-normalised | required by the academic board; served over an OpenAI-compatible endpoint |
| reranker | **`gte-rerank-v2`** (hosted) | rerankers are unrestricted; a local MiniLM cross-encoder is also supported and measured |
| generative model inside `add`/`search` | **none** | the rules permit only `gpt-4o-mini`, which the submitted endpoint does not serve; query planning and ingest summarisation are rule-based and off, and the disclosure marks the LLM field N/A |

No component is fine-tuned. No evaluation data is used for training. LoCoMo-Refined
(CC BY-NC 4.0) is used **only** as a local development benchmark here and is deleted
within the 30-day window the rules require.

---

## 5. How the numbers were produced

`evalharness/` replays a public dataset through the **real FastAPI app** in-process
(`TestClient`, no network) and scores recall of the gold evidence turns:

```bash
python -m evalharness.runner --dataset locomo-refined --path ../data/locomo_refined \
    --top-k 100 --out ../runs/v4_baseline.json
python -m evalharness.sweep d5k20-rr-api d5k20-rr-api-w1 d5k20-rr-api-w2
```

Reported metrics: R@1/R@10 (any gold evidence turn inside the first 1/10 items),
`recall_all` (mean fraction of gold turns present anywhere in the result),
`full@k` (all gold turns present), MRR, and two answer-side probes
(`ansS` = the gold answer string appears literally in the returned text,
`ansT` = the fraction of gold answer tokens present) — the last two exist because
the graded pipeline ends in a fixed reader model, so "is the answer reachable from
the returned text" is the metric that matters, not retrieval alone.

Every sweep writes a JSON with the full environment (embedding/retrieval settings,
with API keys redacted) so any row can be reproduced from the artefact.

### End-to-end reader check

Recall is necessary but not sufficient, so a second harness replays the *reader*:
`evalharness/answer.py` renders the official `OPEN_ENDED_ANSWER_TEMPLATE` over the
returned memories, asks a hosted model for the answer, and grades it with the official
`ACCURACY_PROMPT`. The graded pipeline uses `gpt-4o-mini` for both roles; that model is
not reachable from the development environment, so the local runs use `qwen-flash` to
answer and `qwen-max` to judge. **The absolute accuracies are not leaderboard scores** —
only differences between configurations are meaningful.

| configuration | overall (n=1,382) | single-hop | temporal | multi-hop | open-domain | tokens/query |
|---|---|---|---|---|---|---|
| single-turn items (widening off) | .640 | .759 | .528 | .394 | .485 | 5.3 k |
| two-tier widening (shipped) | **.675** | **.808** | .512 | **.465** | .485 | 11.2 k |

Paired over the whole suite: 140 questions won against 91 lost, **+3.5 points overall**
(McNemar χ² = 9.97, p ≈ 0.002); +4.9 single-hop (p < 0.001), +7.0 multi-hop (p ≈ 0.03),
−1.7 temporal (n.s.). The shipped configuration therefore spends 2.1× the prompt tokens
per question, ~11 k of the 117,760-token answer budget. Cropping to 30 items instead of
82 was neutral overall, and a dedupe threshold of .85 vs .70 does not change retrieval
at all, so neither is a lever.

### Second suite

LongMemEval `oracle` (500 questions, 948 sessions, 10,960 turns) was replayed with the
same service to rule out corpus-specific tuning: R@1 .464, R@10 .996, MRR .632,
answer-in-text .444; by type R@1 temporal-reasoning .617, multi-session .602,
knowledge-update .282, single-session-user .114. The oracle split keeps only the
evidence sessions, so at `top_k=100` coverage saturates and only ordering is
informative — it is an overfitting guard, not a tuning target.

---

## 6. Compliance checklist

| rule | implementation |
|---|---|
| self-hosted live endpoint | `Dockerfile` + `deploy/docker-compose.yml` + Caddy auto-HTTPS |
| `add` returns 200 only after persist-and-searchable | single transaction, index invalidated before return |
| `request_id` idempotent across restarts | claimed inside the write transaction (`tests/test_durability.py`) |
| synchronous indexing | no async worker, no queue |
| `top_k` ≤ 100 | clamped server-side; may return fewer, never more |
| `user_id` is the only isolation boundary | every index, cache entry and query keyed by `user_id`; `session_id` is never a filter |
| empty result | `{"data": []}` with 200, never an error |
| health without auth | `GET /health`, no config leak |
| 4xx never retried by us | no outbound calls in the request path except embedding/rerank, which retry only on 408/429/5xx |
| no training on evaluation data | no fine-tuning at all |

Measured latency: **134 ms/query** at `top_k=100` including the embedding and rerank
round-trips (1,382-question LoCoMo-Refined run, 82 items / 38 k characters returned
per query); 8-way concurrent ingest of 1,800 turns completed in 2.2 s wall on a
laptop; the included `scripts/loadtest.py` re-checks that shape against a live
server.

Headline numbers for the submitted configuration (LoCoMo-Refined, 1,382 questions,
`top_k=100`, `text-embedding-v4` + `gte-rerank-v2` + two-tier widening): R@1 **.748**,
R@10 **.957**, recall-all .937, full@k .874, MRR **.827**; by type R@1 single-hop
.794, temporal .796, multi-hop .592, open-domain .470. The same harness scores the
untuned hybrid at R@1 .424, so the two structural changes that matter are the
reranker (+17 points of R@1) and packing neighbouring turns into the top items
(+15.7 points).

---

## 7. Known limitations

* Retrieval-only: questions whose answers must be **computed** (counting, ordering,
  arithmetic over dates) rely on the reader model seeing every relevant turn inside
  the returned item set.
* No memory consolidation or forgetting policy; the store grows monotonically.
* Contradiction handling is passive: an updated fact is stored as a later record and
  the answer prompt instructions ("prefer the most recent when conflicting") do the
  resolution. Explicit fact versioning is future work.
* The Context-Learning category needs the system to *adapt to a system prompt*
  supplied at question time; this service does not consume `{{system_prompt}}`
  beyond what the platform renders into the answer prompt.
