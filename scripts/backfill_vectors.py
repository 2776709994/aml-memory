"""Backfill vectors for units stored while the embedding provider was unreachable.

Run it from the deployment directory (`~/aml-memory`, or wherever `deploy.json`'s
`AML_DIR` points) once the embedding model answers again:

    .venv/bin/python scripts/backfill_vectors.py            # until done
    .venv/bin/python scripts/backfill_vectors.py --limit 2000 --dry-run

The service stores a turn without a vector when the provider fails (an Add must never
return 500), so those turns are only reachable through BM25 until this runs.  Safe to
run while the service is up: it takes the same write lock as an Add, and it invalidates
the affected user indexes so the next search rebuilds them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aml_memory.config import load_settings  # noqa: E402
from aml_memory.embeddings import Embedder  # noqa: E402
from aml_memory.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed units that were stored without vectors")
    parser.add_argument("--limit", type=int, default=200, help="units per batch (default 200)")
    parser.add_argument("--max-batches", type=int, default=0, help="stop after N batches (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="only report the backlog")
    args = parser.parse_args()

    settings = load_settings()
    store = Store(settings.db_path, embed_model=settings.embed_model,
                  cache_rows=settings.index_cache_rows, embed_cache_rows=settings.embed_cache_rows)
    remaining = store.count_units_missing_vectors()
    print(f"db={settings.db_path} model={settings.embed_model} missing={remaining}")
    if args.dry_run or remaining == 0:
        store.close()
        return 0

    embedder = Embedder(settings, store)
    batch_size = max(1, min(args.limit, 1000))
    done = 0
    batches = 0
    started = time.time()
    while True:
        rows = store.units_missing_vectors(batch_size)
        if not rows:
            break
        vectors = embedder.embed_documents([text for _, text in rows])
        written = store.set_unit_vectors(
            settings.embed_model, list(zip([uid for uid, _ in rows], vectors))
        )
        done += written
        batches += 1
        left = store.count_units_missing_vectors()
        print(f"batch {batches}: embedded={written} total={done} remaining={left} "
              f"elapsed={time.time() - started:.0f}s", flush=True)
        if args.max_batches and batches >= args.max_batches:
            break
    print(f"done: embedded={done} remaining={store.count_units_missing_vectors()}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
