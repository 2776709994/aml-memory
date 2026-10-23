# aml-memory — a self-hosted memory service for the Agent Memory Leaderboard

A single Python service exposing the three endpoints the AML platform calls —
`POST /add`, `POST /search`, `GET /health` — with hybrid lexical + dense retrieval,
write-time temporal anchoring, and an optional cross-encoder reranker.

* **No generative model inside `add`/`search`.** Retrieval is lexical (BM25) plus
  embeddings plus an optional reranker. This keeps the service inside the
  open-source division's model rules by construction (see [Compliance](#compliance)).
* **Contract-first.** Idempotent `add` that survives restarts, synchronous
  persist-then-searchable writes, strict per-`user_id` isolation, and an `top_k`
  cap of 100.
* **Small deployment.** One container, one SQLite file, one volume. No GPU, no
  local model download at query time (the reranker is a hosted API call).

---

## 1. What it does

```
Add   ──▶ parse messages ──▶ split into turns ──▶ anchor relative dates
                                                   │
                                                   ▼
                                          embed (text-embedding-v4)
                                                   │
                                          ┌────────┴────────┐
                                          ▼                 ▼
                                    units (SQLite)     vectors (int8 + scale)

Search ──▶ query analysis ──▶ query variants
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
        dense cosine (all user units)         BM25 over the same units
              └──────────────────┬──────────────────┘
                                 ▼
                       weighted RRF fusion (ranks)
                                 ▼
                        optional cross-encoder rerank
                                 ▼
                    entity / date / recency re-weighting
                                 ▼
             assemble ≤ top_k items (neighbour expansion,
             same-speaker merging, dedupe, char budget)
                                 ▼
                        {"data": [{id, content, score}]}
```

### Write path

Each `add` call's `messages` are parsed into atomic **turns**.  Every turn becomes
one memory unit whose stored text is

```
<speaker> (<absolute date>): <text with relative times anchored in place>
```

Two deliberate choices:

1. **Relative time is resolved at write time.** `"I signed up for a pottery class
   yesterday"` spoken on 2023-07-03 is stored as
   `... signed up for a pottery class yesterday (2 July 2023)`. The original
   wording is preserved, so a reader that ignores the anchor still sees the
   source sentence; a reader that needs a date does not have to resolve it from
   the session clock. Anchors respect the granularity actually spoken — `last
   year` becomes `(2022)`, `this month` becomes `(July 2023)`, never a
   spuriously precise day.
2. **The speaker is part of the record.** The answer prompts bucket memories per
   speaker, so every unit stays attributable no matter how the platform partitions
   a flat result list.

A turn that arrives **without a timestamp** inherits the most recent date already
known in the call, and can itself supply one: if its text carries a session header
(`--- Session 1 started at 1:56 pm on 8 May, 2023 ---`) that date is adopted for the
turn and carried forward. Only an explicit header counts — a date mentioned in
passing ("I met her on 3 May") is an event, not the session's clock. Without this,
a payload with no timestamps loses every temporal question in the session; with it,
a chunk that spans days is still dated per day instead of being pinned to its first
message.

Units are linked to their neighbours (`prev_id`/`next_id`) and carry a monotonic
per-user `seq`, so context can be recovered at read time.

### Read path

* **Query variants.** The question is combined with a light rule-based analysis
  (entity extraction, temporal markers, aggregation/"how many" shape, MCQ options)
  into up to a handful of retrieval views.
* **Two channels, fused by rank.** Dense cosine similarity over every unit of the
  user (exact, no ANN approximation — a user's memory is small enough that brute
  force is both faster and exact) and Okapi BM25 with CJK-aware tokenisation.
  The two lists are combined with weighted Reciprocal Rank Fusion. Fusing *ranks*
  rather than scores avoids having to calibrate two incomparable score scales.
* **Optional rerank.** A cross-encoder rescores the fused head. It is a hosted
  call (`gte-rerank-v2`) so the container stays small and the model is
  multilingual; a local `sentence-transformers` cross-encoder works too.
* **Assembly.** Dedupe, join short same-speaker runs, then **two-tier context
  widening**: the top 30 units are replaced by their ±2 same-session window and the
  rest of the list by ±1, with every absorbed turn removed from the tail (never
  returning a turn twice, never crossing a session). Finally the list is trimmed to
  the content budget and returned in descending score order — the order matters,
  because on overflow the platform's reader keeps a prefix of the returned list.

---

## 2. Endpoints

### `POST /add`

```jsonc
{
  "request_id": "c1f0…",           // idempotency key, echoed back
  "user_id":    "u_123",           // the only isolation boundary
  "session_id": "s_9",             // grouping only, never a filter
  "messages": [
    {"role": "user", "content": "Caroline: I adopted a greyhound in May.",
     "timestamp": 1683093600000}   // optional; absent ⇒ arrival time
  ]
}
```

Returns `200` with `{"success": true, "request_id": …, "user_id": …, "session_id": …}`
**only after** the units and their vectors are committed and searchable.

`request_id` is claimed inside the write transaction, so:
* a retry with the same `request_id` returns `200` without writing a second copy;
* the claim lives in SQLite, so it survives a restart — required because the
  platform retries a timed-out `add` up to 32 times, possibly across restarts;
* two concurrent retries of the same `request_id` cannot both write.

### `POST /search`

```jsonc
{"query": "When did Melanie sign up for a pottery class?",
 "options": ["…"],                 // optional; used as recall signal only
 "user_id": "u_123", "top_k": 100}
```

```jsonc
{"data": [{"id": "m_9f3…+m_a71…", "content": "…", "score": 0.83}]}
```

Never more than `top_k` items, relevance-ordered, `{"data": []}` for an unknown
user. `score` is informational only — the platform treats the list order as the
signal.

### `GET /health`

`200 {"status": "ok", "service": "aml-memory", "version": "…"}` — unauthenticated by
design (the rules require a public health check) and leaks no configuration.

### Auth

When `AML_API_KEY` is set, every `add`/`search` requires it as `Authorization:
Bearer <key>`, `Token <key>`, `X-Api-Key: <key>` or `?api_key=<key>`. Comparison is
constant-time. `/health` stays open.

---

## 3. Measured on LoCoMo-Refined

Public suite, 1,382 questions across all 10 conversations (~4,900 turns), `top_k=100`,
recall measured against the gold `evidence` turn ids. Every row is one run of the
**same** harness on the same data; `ms/query` excludes ingest.

| configuration | R@1 | R@10 | recall-all | full@k | MRR | answer-in-text | ms/query |
|---|---|---|---|---|---|---|---|
| BM25 only | .406 | .701 | .818 | .758 | .509 | .465 | 14 |
| dense only | .379 | .736 | .908 | .858 | .503 | .491 | 19 |
| hybrid (base) | .424 | .764 | .909 | .857 | .543 | .493 | 16 |
| hybrid, deeper pool + sharper RRF | .422 | .788 | .919 | .873 | .547 | .496 | 16 |
| hybrid + local cross-encoder (MiniLM) | .552 | .845 | .911 | .860 | .653 | .493 | 365 |
| hybrid, deep pool + hosted reranker (`gte-rerank-v2`) | .594 | .865 | .921 | .877 | .693 | .496 | 138 |
| **+ two-tier context widening (shipped default)** | **.748** | **.957** | **.937** | .874 | **.827** | **.517** | 134 |

The shipped configuration, broken down by question type:

| slice | n | R@1 | R@10 | recall-all | full@k | MRR | answer-in-text |
|---|---|---|---|---|---|---|---|
| single-hop | 802 | .794 | .979 | .968 | .941 | .867 | .599 |
| temporal | 299 | .796 | .973 | .958 | .923 | .858 | .482 |
| multi-hop | 213 | .592 | .911 | .864 | .662 | .711 | .333 |
| open-domain | 68 | .470 | .773 | .703 | .515 | .576 | .294 |

Read-outs that shaped the default configuration:

* **Hybrid beats either channel alone** on both precision and coverage.
* **A deeper candidate pool (500) with a sharper RRF constant (20) is a free
  coverage win** and costs nothing at query time.
* **The reranker is a large precision lever** (+17.0 R@1, +0.15 MRR over the base
  hybrid). It is legal: the rules restrict embeddings and generative models, not
  rerankers. The hosted model is also 2.6× faster than running MiniLM locally, and it
  keeps torch out of the container.
* **Context widening is the other big lever** (+15.7 R@1, +0.13 MRR): the platform
  caps the *number* of items (100) but not their size, so the top 30 hits carry
  ±2 same-session turns and the rest of the list carries ±1. Multi-hop is exactly
  the case where the second evidence turn sits one turn away from the retrieved one.
  Radius 3 buys nothing further (R@1 .754, recall-all .927).
* **Neighbour expansion and same-speaker merging are neutral at `top_k=100` of a
  ~500-turn corpus** (identical scores with them disabled). They are kept because
  they matter when a conversation is much larger than the result budget — but they
  are the first knobs to audit if a suite regresses.
* `answer-in-text` (the gold answer string appearing literally in the returned text)
  is the ceiling of a retrieval-only system: the rest of the reference answers need
  the reader to aggregate or paraphrase turns. Single-hop is at .60, multi-hop at
  .33 — that gap is a *reasoning* gap, not a retrieval gap.

In `runs/` the same harness re-measures every knob (`evalharness/sweep.py`); treat
gaps below ~0.01 as noise.

### End-to-end check (reader proxy)

Recall metrics reward things the reader does not care about, so the harness can also
replay a **reader**: `evalharness/answer.py` renders the official
`OPEN_ENDED_ANSWER_TEMPLATE` over the returned memories, answers, and scores the answer
with the official `ACCURACY_PROMPT`. The real reader is `gpt-4o-mini`; that model is not
reachable from here, so the local runs use `qwen-flash` to answer and `qwen-max` to
judge. The absolute numbers are therefore **not** leaderboard scores — only the
differences between configurations mean anything.

| configuration | overall (1,382) | single-hop | temporal | multi-hop | open-domain | tokens/query |
|---|---|---|---|---|---|---|
| single-turn items, widening off | .640 | .759 | .528 | .394 | .485 | 5.3 k |
| **two-tier widening (shipped)** | **.675** | **.808** | .512 | **.465** | .485 | 11.2 k |
| + session-scoped expansion, per-part headers, body-level dedupe | .679 | .807 | .525 | .465 | .515 | 11.2 k |

Scored over the whole suite, not a sample, and compared question by question:
+140 questions won against 91 lost, i.e. **+3.5 points overall** (McNemar χ² = 9.97,
p ≈ 0.002) — +4.9 on single-hop (p < 0.001), +7.0 on multi-hop (p ≈ 0.03), and −1.7 on
temporal (not significant: widening a temporal answer's item also brings in the
neighbouring turns' own dates). The cost is 2.1× prompt tokens per question, still only
~11 k of the 117,760-token answer budget. Cropping the list to 30 items instead of 82
was neutral overall, and a dedupe threshold of .85 vs .70 changes retrieval not at all,
so neither is a lever; the context volume is. Widening *every* rank by 2 instead of only
the top 30 is also a wash (.673 vs .675) for 24 % more tokens, so the two-tier split
stays: the cheap radius-1 window is enough to keep the tail's recall.

Failure analysis over 101 wrong verdicts (33 multi-hop, 29 open-domain, 26 temporal,
13 single-hop) says the dominant multi-hop failure is **over-answering a list**, not
missing evidence: gold `Figurines, shoes` → `a necklace, a pair of pink sneakers, and
figurines` is WRONG under the judge's "no extra items" rule. Single-hop (.81) sits near
the reader's ceiling.

### A second suite (LongMemEval oracle)

To check the design is not tuned to one corpus, the same service was replayed on
LongMemEval `oracle` (500 questions, 948 sessions, 10,960 turns): R@1 .464, R@10 .996,
MRR .632, answer-in-text .444, and by question type R@1 temporal-reasoning .617,
multi-session .602, knowledge-update .282, single-session-user .114. Because the oracle
split contains only the evidence sessions, coverage saturates at `top_k=100` and only
ordering is informative — it is an overfitting guard, not a tuning target.

---

## 4. Compliance

| rule | this service |
|---|---|
| embeddings must be `text-embedding-v4` | ✅ `EMBED_PROVIDER=openai`, `EMBED_MODEL=text-embedding-v4`, `dimensions=1024`, L2-normalised locally |
| generative models inside `add`/`search` must be `gpt-4o-mini` | **N/A — no generative call is made.** Query planning and ingest summaries are rule-based/off. (`AML_QUERY_LLM=0`, `AML_INGEST_LLM=off`) |
| rerankers unrestricted | `gte-rerank-v2` (hosted) or a local cross-encoder, or none |
| `add` returns 200 only after persist-and-searchable | ✅ single SQLite transaction, then the per-user index is invalidated before returning |
| `request_id` idempotency across restarts | ✅ claim row in SQLite (`tests/test_durability.py`) |
| `top_k` ≤ 100, never more items | ✅ clamped server-side |
| `user_id` is the only isolation boundary | ✅ every query, index and cache is keyed by `user_id` |
| health unauthenticated, no config leak | ✅ |
| self-hosted live endpoint | ✅ Dockerfile + compose + Caddy in `deploy/` |

---

## 5. Quick start (local)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python

cp .env.example .env        # fill EMBED_API_KEY (DashScope key)
python -m uvicorn aml_memory.main:app --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/add -H 'content-type: application/json' -d '{
  "request_id":"r1","user_id":"u1","session_id":"s1",
  "messages":[{"role":"user","content":"Caroline: I adopted a greyhound named Pepper in May 2023."}]}'
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"What is the name of Caroline\u0027s dog?","user_id":"u1","top_k":10}'
```

Run the tests (offline — the `hash` embedder is used, no API key needed):

```bash
python -m pytest tests -q
```

## 6. Deploy

One command on a fresh Linux box (installs Docker, fetches the pinned tag, writes a
`.env` with a generated token, starts the stack, waits for HTTPS, runs the contract
self-check, prints the values to register):

```bash
curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/2776709994/aml-memory/v1.4/deploy/bootstrap.sh \
  -o /tmp/aml-bootstrap.sh && wc -c /tmp/aml-bootstrap.sh      # expect 11833
sudo AML_DOMAIN=memory.example.com EMBED_API_KEY=sk-... bash /tmp/aml-bootstrap.sh
```

(If GitHub is reachable directly, swap in `https://raw.githubusercontent.com/...v1.4/...`;
the CDN mirror is `https://cdn.jsdelivr.net/gh/2776709994/aml-memory@v1.4/deploy/bootstrap.sh`.
That mirror caches per tag, so always check the byte count before running.)

It is idempotent — re-running never regenerates the token and never touches the data
volume. For a hand-held walkthrough see **[DEPLOY.md](DEPLOY.md)**; to hand the job to
an autonomous operator agent on the server, give it
**[deploy/AGENT_DEPLOY.md](deploy/AGENT_DEPLOY.md)** (inputs, expected output, failure
matrix, forbidden actions, report template).

### No Docker, no root: Cloudflare Tunnel

If the host has neither `sudo` nor Docker (common on a shared lab box), the service still
runs as an ordinary user on `127.0.0.1` and reaches the internet through a Cloudflare
tunnel:

```bash
curl -fsSL https://ghfast.top/https://raw.githubusercontent.com/2776709994/aml-memory/v1.4/deploy/bootstrap-noroot.sh \
  -o /tmp/aml-bootstrap-noroot.sh && wc -c /tmp/aml-bootstrap-noroot.sh   # expect 23084
AML_DOMAIN=memory.example.com CF_TUNNEL_TOKEN=eyJ... EMBED_API_KEY=sk-... \
  bash /tmp/aml-bootstrap-noroot.sh
```

It builds a private Python environment (three fallbacks, because `ensurepip` and
`python3-venv` are often missing), installs the dependencies through the first reachable
PyPI mirror, verifies the embedding key **and** the hosted reranker before spending any
time on installs (a dead key dies in one second instead of surfacing as a bare `500` on
the first `/add`), generates `.env` — reusing the embedding key for the reranker, since
one DashScope key serves both and reranking is worth about +0.17 R@1 — downloads
`cloudflared`, supervises both processes (`deploy/service-noroot.sh
start|stop|restart|status|logs|url`), and finishes with the same 15-check contract probe
against the public HTTPS URL. Re-running it is safe and now *merges* `.env`: an existing
token is kept, while any key that is missing or blank (a hostname or tunnel token left
over from a quick tunnel, for instance) is filled in. Values you pass on the command line
win over the file, which makes key rotation a single command — `EMBED_API_KEY=sk-new
bash /tmp/aml-bootstrap-noroot.sh` swaps both keys, re-probes them, and restarts, leaving
the token, the hostname and the data untouched. On a host with no cron, call
`deploy/watchdog.sh` from whatever supervisor exists — it restarts the API and the tunnel
only when they are actually down. Operator runbook:
**[deploy/AGENT_DEPLOY_NOROOT.md](deploy/AGENT_DEPLOY_NOROOT.md)**.

---

## 7. Configuration

Every knob is an environment variable; see `.env.example` for the annotated list.
The ones that actually change behaviour:

| variable | default | effect |
|---|---|---|
| `AML_DATA_DIR` | `./data` | SQLite lives at `<dir>/aml.sqlite3` |
| `EMBED_MODEL` / `EMBED_DIM` | `text-embedding-v4` / 1024 | must match the rules |
| `EMBED_BATCH` / `EMBED_CONCURRENCY` | 10 / 6 | request shape against the embedding API |
| `AML_CANDIDATE_DEPTH` | 500 | how deep each channel goes before fusion |
| `AML_RRF_K` | 20 | RRF smoothing; smaller ⇒ sharper rank weighting |
| `AML_DENSE_WEIGHT` / `AML_BM25_WEIGHT` | 1.0 / 1.0 | channel weights in the fusion |
| `RERANK_PROVIDER` | `none` | `none` \| `dashscope` \| `local` |
| `RERANK_DEPTH` | 60 | how many fused candidates get reranked |
| `RERANK_TIMEOUT_S` | 12 | abandon a slow rerank and keep fusion order (measured tail: one provider call took 21 s under 40-way load) |
| `AML_QUERY_LLM` / `AML_INGEST_LLM` | off / off | any generative model must be `gpt-4o-mini`; both are off so nothing else is ever called |
| `AML_EXPAND_NEIGHBORS` | 1 | ±1 turn of context around each hit, as separate items |
| `AML_CONTEXT_RADIUS` | 2 | ≥1 packs each top hit *with* its neighbours into one item (see §3) |
| `AML_CONTEXT_TOP` | 30 | how many top hits get the full radius |
| `AML_CONTEXT_FILL` | 100 | ranks below `context_top` get a radius-1 window |
| `AML_CONTEXT_FILL_TO` | 0 | >0 refills the list to that many items after widening absorbs turns |
| `AML_DEDUPE_JACCARD` | 0.85 | token-set overlap above which two items are the same memory (compared on the statement body, so a header date does not mask a repeat) |
| `AML_DEDUPE_PREFER` | `score` | which half of a duplicate survives: the better-ranked one or the more recent statement |
| `AML_EXHAUSTIVE_CAP` | 40 | item cap for counting/list questions (`top_k` still applies) |
| `AML_ASSISTANT_PRIOR` | 1.0 | multiplier on assistant-role turns at ranking time (1.0 = no prior) |
| `AML_MERGE_MAX_UNITS` | 3 | same-speaker run length cap |
| `AML_MAX_CONTENT_CHARS` | 700 | per-item soft cap (merged runs get 3×) |
| `AML_TOTAL_CHAR_BUDGET` | 120000 | total characters handed to the reader |
| `AML_SCORE_FLOOR` | 0.0 | 0 keeps every candidate up to `top_k` |
| `AML_INDEX_CACHE_COUNT` / `AML_INDEX_CACHE_VECTORS` | 24 / 120000 | per-user matrix cache, bounded by users **and** rows (~470 MB of matrix at 1024-d) |
| `EMBED_CACHE_MB` / `EMBED_CACHE_SIZE` | 256 / 100000 | in-process embedding cache, float16, byte-bounded |
| EMBED_CACHE_ROWS | 200000 | cap on the persistent embedding cache table (~200 MB at dim 1024; oldest rows are dropped, 0 = unlimited) |
| EMBED_CACHE_DOCS | 0 | cache ingested turn texts too; off because `vectors` already holds them |
| `AML_MAX_CONCURRENT_INGEST` / `_SEARCH` | 8 / 16 | thread gates |
| `AML_ENV_FILE` | *(empty)* | pin the env file; otherwise `./.env`, then the `.env` beside the package |

### Footprint per turn

A turn costs three things on disk: its text (~160–350 B for conversational data),
its embedding, and (optionally) a row of the embedding cache. The embedding is the
dominant term, so vectors are stored as **int8 plus a per-vector scale** (dim bytes
instead of dim × 2) on 8 KiB pages, and ingested text is **not** copied into the
embedding cache — it already lives in `vectors`, and a repeated Add is caught by the
`request_id` claim rather than by the cache. Measured cost per turn:

| storage | bytes/turn |
|---|---|
| float16 vectors, 4 KiB pages, documents cached | 11.1 KB |
| int8 vectors, 4 KiB pages, documents cached | 4.1 KB |
| **int8, 8 KiB pages, documents not cached (shipped)** | **1.9 KB** |

So ~2 KB per turn: a million turns is ~2 GB. Retrieval is unchanged by the
quantisation — LoCoMo-Refined re-measured at R@1 .748 / R@10 .957 / MRR .827, the
same numbers as the float16 build.

Throughput, measured by replaying 100 000 synthetic turns through `/add` in
20-message chunks against the real `text-embedding-v4` endpoint: **~83 turns/s**
on an 8-core laptop, i.e. one 20-turn Add every ~240 ms. A small VPS with the same
embedding concurrency lands in the tens of turns per second; scale it with
`EMBED_CONCURRENCY` and `EMBED_BATCH` (batch ≤ 10 is an API limit) and watch
`/admin/stats` → `db_bytes` while a run is in flight.

---

## 8. Repository layout

```
aml-memory/
├─ aml_memory/            # the service
│  ├─ main.py             # FastAPI app: /add /search /health, auth, idempotency
│  ├─ config.py           # env-driven Settings
│  ├─ ingest.py           # messages → units (speaker split, long-text split)
│  ├─ timeutil.py         # timestamp parsing, relative→absolute anchoring
│  ├─ store.py            # SQLite: requests, units, vectors, embed cache, index
│  ├─ embeddings.py       # pluggable embedder (openai/dashscope/local/hash) + cache
│  ├─ bm25.py             # CJK-aware BM25
│  ├─ query.py            # rule-based query analysis → variants, boosts
│  ├─ rerank.py           # hosted / local cross-encoder, or none
│  └─ retrieve.py         # fusion, rerank, boost, assembly
├─ evalharness/           # offline evaluation against public suites
│  ├─ datasets.py         # LoCoMo / LoCoMo-Refined / LongMemEval loaders
│  ├─ runner.py           # replay a suite through the real app, score it
│  ├─ answer.py           # replay the official reader prompt + judge
│  ├─ sweep.py            # named configuration grid
│  └─ diagnose.py         # why did this question miss?
├─ scripts/
│  ├─ verify_endpoint.py  # 15-check contract probe for a deployed endpoint
│  ├─ loadtest.py         # concurrent ingest / search / retry-storm check
│  └─ audit_consistency.py# config ↔ .env.example ↔ docs ↔ run artefacts
├─ tests/                 # contract, text-processing, durability, hardening tests
├─ deploy/
│  ├─ docker-compose.yml  # service + Caddy (TLS, 1900 s timeouts)
│  ├─ Caddyfile
│  ├─ bootstrap.sh        # idempotent one-command deployment (Docker + Caddy)
│  ├─ AGENT_DEPLOY.md     # the runbook handed to an operator agent
│  ├─ bootstrap-noroot.sh # same, without Docker or root (Cloudflare Tunnel)
│  ├─ service-noroot.sh   # start/stop/restart/status/logs/url supervisor
│  ├─ watchdog.sh         # health check + restart, for hosts with no cron
│  └─ AGENT_DEPLOY_NOROOT.md
└─ Dockerfile
```

## 9. Known limitations

* Retrieval-only: questions that need aggregation across many turns are bounded by
  what a reader can do with 100 ordered items (~0.49 literal answer coverage here).
* No fact versioning beyond "keep both records, order by time": an explicit
  old-value/new-value channel is the next planned addition for the governance
  category.
* BM25 and the dense index are rebuilt per user in memory on first touch after a
  write; that is deliberate (a user's memory fits comfortably) but would need an
  incremental index for corpora far beyond ~10⁵ turns.
* The reranker is off in `config.py`'s default and enabled per deployment
  (`RERANK_PROVIDER=dashscope`); the hosted one costs ~0.14 s per query. Turning it
  off costs ~0.17 R@1, so a deployment without it is legal but weaker.
