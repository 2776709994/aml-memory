# Submission sheet — paste these into the leaderboard form

Everything below is what the AML submission form asks for, pre-filled. Keep this
file in sync with the pinned commit; after registering, **do not rebase or amend
that commit** (the rules freeze the code version for the duration of a Full run).

| field | value |
|---|---|
| Track | **Textual Memory** |
| Division | **Open-source Methods** (prize-eligible) |
| Public repository | <https://github.com/2776709994/aml-memory> |
| Pinned commit | tag **`v1.2`** — resolve with `git rev-parse v1.2^{commit}` (a tag, so the hash stays valid however many doc commits follow) |
| License | MIT |
| Add URL | `https://<your-domain>/add` |
| Search URL | `https://<your-domain>/search` |
| Health URL (optional) | `https://<your-domain>/health` |
| Auth | `Authorization: Bearer <AML_API_KEY you chose>` |
| Embedding model | `text-embedding-v4` (1024-dim, L2-normalised) |
| Generative model inside add/search | **N/A — none is used** (rules permit only `gpt-4o-mini`; the service calls no LLM, query planning is rule-based) |
| Reranker | `gte-rerank-v2` (hosted; rerankers are unrestricted) |
| Method summary | Retrieval-only hybrid memory: per-turn records with speaker + absolute-date anchoring, dense cosine + BM25 fused by weighted RRF on ranks, hosted cross-encoder rerank, two-tier same-session context widening |
| Full method disclosure | [`docs/METHOD.md`](METHOD.md) |
| Measured on LoCoMo-Refined | retrieval R@1 .748 / R@10 .957 / MRR .827 at `top_k=100`, 134 ms/query; end-to-end with the official reader template on all 1 382 questions **.675** with context widening vs **.640** without (n.s.-free paired test, McNemar p ≈ 0.002) |
| Deployment | one command: `deploy/bootstrap.sh` (idempotent; installs Docker, pins `v1.2`, generates the API token, runs the 15-check contract probe) — see `deploy/AGENT_DEPLOY.md` |
| Contact / team | *fill in* |

## Evidence attached to the repo

* `evalharness/` — the harness that produced every number in `README.md` §3 and
  `docs/METHOD.md` §5 (replays a public suite through the real app), including
  `answer.py`, which renders the official answer prompt and judge so the context
  volume is measured end to end rather than by recall alone.
* `tests/` — 36 offline tests covering the API contract, the text pipeline, crash
  safety and assembly behaviour.
* `scripts/verify_endpoint.py` — 15-check contract probe to run against the deployed
  URL before registering it (health without auth, 401s, write-then-search, replay,
  `top_k` clamping, empty result for an unknown user, 4xx on malformed input).
* `scripts/loadtest.py` — concurrent ingest/search/retry-storm checker.

## Order of operations after the endpoint is live

1. `GET https://<domain>/health` from outside — must answer 2xx **without** a token.
2. Register the endpoint above, wait for the key (issued ~19:00 Beijing time the
   next working day).
3. Run **Smoke** (≤ 30 per track, at least 1 hour apart). Read the report; fix
   nothing structural afterwards — the version is frozen once Full starts.
4. Only when Smoke is clean, run **Full**. It must reach 100 % of instances; a run
   that stops early is wasted, and only two Full runs per key are allowed with the
   second one requiring the first to have completed 30 days earlier.
5. Delete any downloaded evaluation data within 30 days of the run (LoCoMo-Refined
   lives outside the repo, in `../data/`, and is never committed).
