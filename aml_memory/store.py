"""SQLite-backed store with lazily built per-user in-memory indexes.

Design note: ``user_id`` is the only retrieval boundary in the AML contract, and a
single ``user_id`` covers one conversation / one task, i.e. thousands of memory
units at most.  Brute-force cosine over that set is faster and far more accurate
than an approximate index, and it keeps the deployment a single process + single
file.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS requests (
    request_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    n_units     INTEGER NOT NULL,
    created_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS units (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    seq         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    speaker     TEXT,
    created_ms  INTEGER,
    date_str    TEXT,
    text        TEXT NOT NULL,
    prev_id     TEXT,
    next_id     TEXT,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_units_user ON units(user_id, seq);
CREATE INDEX IF NOT EXISTS idx_units_session ON units(user_id, session_id, seq);

CREATE TABLE IF NOT EXISTS vectors (
    unit_id     TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS embed_cache (
    hash        TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);
"""


@dataclass(slots=True)
class Unit:
    id: str
    user_id: str
    session_id: str
    seq: int
    kind: str
    speaker: str
    created_ms: int | None
    date_str: str
    text: str
    prev_id: str | None
    next_id: str | None
    meta: dict[str, Any]


@dataclass
class UserIndex:
    user_id: str
    units: list[Unit]
    ids: list[str]
    matrix: np.ndarray | None  # (n, dim) float32, L2-normalised, or None
    dim: int
    version: int
    loaded_ns: int


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def content_hash(model: str, text: str) -> str:
    return _sha1(f"{model}\x00{text}")


class Store:
    def __init__(
        self,
        db_path: str | Path,
        embed_model: str = "",
        cache_indexes: int = 24,
        cache_rows: int = 400_000,
    ):
        self.db_path = str(db_path)
        self.embed_model = embed_model
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._indexes: dict[str, UserIndex] = {}
        self._versions: dict[str, int] = {}
        self._build_locks: dict[str, threading.Lock] = {}
        self._cache_indexes = cache_indexes
        # A user's matrix is len(units) x dim float32: 100k turns at dim 1024 is
        # 410 MB.  A count-only cache therefore cannot bound RSS, and a Full run
        # can touch thousands of users, so cap the cached VECTOR ROWS too
        # (400k rows ~ 1.6 GB worst case, and the eviction loop keeps the most
        # recently used users).
        self._cache_rows = cache_rows
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ---------------------------------------------------------------- plumbing
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------- idempotency
    def request_seen(self, request_id: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------- writes
    def insert_add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        units: Sequence[Unit],
        vectors: Sequence[np.ndarray | None],
        model: str,
    ) -> int:
        """Persist units + vectors atomically and invalidate the user's index."""
        now_ms = int(time.time() * 1000)
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                # Claim the request id inside the write transaction: the platform
                # retries a timed-out Add with the SAME request_id, and two retries
                # can be in flight at once.  Whoever commits first wins; the loser
                # must not write a second copy of the same memories.
                claimed = conn.execute(
                    "SELECT 1 FROM requests WHERE request_id = ?", (request_id,)
                ).fetchone()
                if claimed is not None:
                    conn.rollback()
                    return 0

                # Global insertion order per user: Add calls arrive in chronological
                # order, so a monotonic seq keeps neighbour links and context merging
                # correct across sessions (the platform restarts indexing per call).
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) FROM units WHERE user_id = ?", (user_id,)
                ).fetchone()
                base_seq = int(row[0]) + 1 if row and row[0] is not None else 0
                for offset, unit in enumerate(units):
                    unit.seq = base_seq + offset
                if units:
                    tail = conn.execute(
                        "SELECT id FROM units WHERE user_id = ? ORDER BY seq DESC LIMIT 1", (user_id,)
                    ).fetchone()
                    if tail is not None and tail["id"] != units[0].id:
                        if not units[0].prev_id:
                            units[0].prev_id = tail["id"]
                        conn.execute(
                            "UPDATE units SET next_id = ? WHERE id = ?", (units[0].id, tail["id"])
                        )
                conn.execute(
                    "INSERT OR REPLACE INTO requests(request_id, user_id, session_id, n_units, created_ms)"
                    " VALUES (?,?,?,?,?)",
                    (request_id, user_id, session_id, len(units), now_ms),
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO units(id,user_id,session_id,seq,kind,speaker,created_ms,"
                    "date_str,text,prev_id,next_id,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            u.id, u.user_id, u.session_id, u.seq, u.kind, u.speaker,
                            u.created_ms, u.date_str, u.text, u.prev_id, u.next_id,
                            json.dumps(u.meta, ensure_ascii=False) if u.meta else "",
                        )
                        for u in units
                    ],
                )
                payload = []
                for unit, vec in zip(units, vectors):
                    if vec is None:
                        continue
                    payload.append((unit.id, model, int(vec.shape[0]), _vec_to_blob(vec)))
                if payload:
                    conn.executemany(
                        "INSERT OR REPLACE INTO vectors(unit_id,model,dim,vec) VALUES (?,?,?,?)",
                        payload,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        with self._cache_lock:
            self._versions[user_id] = self._versions.get(user_id, 0) + 1
            self._indexes.pop(user_id, None)
        return len(units)

    def cache_vectors(self, model: str, items: Sequence[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        with self._write_lock:
            conn = self._conn()
            conn.executemany(
                "INSERT OR REPLACE INTO embed_cache(hash,model,dim,vec) VALUES (?,?,?,?)",
                [(h, model, int(v.shape[0]), _vec_to_blob(v)) for h, v in items],
            )
            conn.commit()

    def load_cached_vectors(self, model: str, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        if not hashes:
            return {}
        out: dict[str, np.ndarray] = {}
        conn = self._conn()
        for start in range(0, len(hashes), 500):
            chunk = list(hashes[start:start + 500])
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT hash, dim, vec FROM embed_cache WHERE model = ? AND hash IN ({placeholders})",
                [model, *chunk],
            ).fetchall()
            for row in rows:
                out[row["hash"]] = _blob_to_vec(row["vec"], row["dim"])
        return out

    # -------------------------------------------------------------------- index
    def user_index(self, user_id: str) -> UserIndex:
        with self._cache_lock:
            cached = self._indexes.get(user_id)
            version = self._versions.get(user_id, 0)
            if cached is not None and cached.version == version:
                cached.loaded_ns = time.time_ns()  # LRU is by use, not by build
                return cached
            # One builder per user: a burst of concurrent searches on a cold cache
            # would otherwise each run the same full scan and hold its own copy of
            # the matrix (410 MB at 100k turns), which is how a box runs out of RAM.
            build_lock = self._build_locks.setdefault(user_id, threading.Lock())
        with build_lock:
            with self._cache_lock:
                cached = self._indexes.get(user_id)
                version = self._versions.get(user_id, 0)
                if cached is not None and cached.version == version:
                    cached.loaded_ns = time.time_ns()
                    return cached
            index = self._build_index(user_id, version)
            with self._cache_lock:
                self._indexes[user_id] = index
                self._evict_locked(keep=user_id)
            return index

    def _evict_locked(self, keep: str) -> None:
        """Drop least-recently-used indexes until both budgets are satisfied.

        Called with ``_cache_lock`` held.  ``keep`` (the user we just loaded) is
        never evicted: rebuilding it would cost a full SQLite scan per search,
        which is far worse than being slightly over the memory budget.
        """
        def rows(name: str) -> int:
            data = self._indexes.get(name)
            return len(data.ids) if data is not None else 0

        total = sum(rows(name) for name in self._indexes)
        while len(self._indexes) > 1 and (
            len(self._indexes) > self._cache_indexes or total > self._cache_rows
        ):
            oldest = min(
                (name for name in self._indexes if name != keep),
                key=lambda name: self._indexes[name].loaded_ns,
                default=None,
            )
            if oldest is None:
                return
            total -= rows(oldest)
            self._indexes.pop(oldest, None)

    def _build_index(self, user_id: str, version: int) -> UserIndex:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id,user_id,session_id,seq,kind,speaker,created_ms,date_str,text,prev_id,next_id,meta"
            " FROM units WHERE user_id = ? ORDER BY seq",
            (user_id,),
        ).fetchall()
        units = [
            Unit(
                id=row["id"], user_id=row["user_id"], session_id=row["session_id"] or "",
                seq=row["seq"], kind=row["kind"], speaker=row["speaker"] or "",
                created_ms=row["created_ms"], date_str=row["date_str"] or "",
                text=row["text"], prev_id=row["prev_id"], next_id=row["next_id"],
                meta=json.loads(row["meta"]) if row["meta"] else {},
            )
            for row in rows
        ]
        ids = [u.id for u in units]
        by_id = {u.id: u for u in units}
        matrix = None
        dim = 0
        if ids:
            vectors: list[np.ndarray] = []
            found_ids: list[str] = []
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                placeholders = ",".join("?" * len(chunk))
                vrows = conn.execute(
                    f"SELECT unit_id,dim,vec FROM vectors WHERE unit_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for vrow in vrows:
                    found_ids.append(vrow["unit_id"])
                    vectors.append(_blob_to_vec(vrow["vec"], vrow["dim"]))
            if vectors:
                dim = int(vectors[0].shape[0])
                matrix = np.zeros((len(units), dim), dtype=np.float32)
                id_to_row = {uid: i for i, uid in enumerate(ids)}
                for uid, vec in zip(found_ids, vectors):
                    row_index = id_to_row.get(uid)
                    if row_index is not None and vec.shape[0] == dim:
                        matrix[row_index] = vec
        del by_id
        return UserIndex(
            user_id=user_id, units=units, ids=ids, matrix=matrix, dim=dim,
            version=version, loaded_ns=time.time_ns(),
        )

    # ------------------------------------------------------------------- reads
    def get_units(self, ids: Iterable[str]) -> dict[str, Unit]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {}
        conn = self._conn()
        out: dict[str, Unit] = {}
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            placeholders = ",".join("?" * len(chunk))
            for row in conn.execute(
                "SELECT id,user_id,session_id,seq,kind,speaker,created_ms,date_str,text,prev_id,next_id,meta"
                f" FROM units WHERE id IN ({placeholders})",
                chunk,
            ):
                out[row["id"]] = Unit(
                    id=row["id"], user_id=row["user_id"], session_id=row["session_id"] or "",
                    seq=row["seq"], kind=row["kind"], speaker=row["speaker"] or "",
                    created_ms=row["created_ms"], date_str=row["date_str"] or "",
                    text=row["text"], prev_id=row["prev_id"], next_id=row["next_id"],
                    meta=json.loads(row["meta"]) if row["meta"] else {},
                )
        return out

    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        units = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM units").fetchone()[0]
        reqs = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        vecs = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return {"units": units, "users": users, "add_requests": reqs, "vectors": vecs}


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float16).tobytes()


def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    if arr.shape[0] != dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: min(dim, arr.shape[0])] = arr[: min(dim, arr.shape[0])]
        return out
    return arr
