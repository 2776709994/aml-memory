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
import shutil
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
    meta        TEXT,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_units_user ON units(user_id, seq);
CREATE INDEX IF NOT EXISTS idx_units_session ON units(user_id, session_id, seq);

CREATE TABLE IF NOT EXISTS vectors (
    unit_id     TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    scale       REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS embed_cache (
    hash        TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    scale       REAL NOT NULL DEFAULT 1.0
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
        embed_cache_rows: int = 1_000_000,
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
        # Disk budget for the persistent embedding cache, in rows (0 = unlimited).
        # One row is dim x 2 bytes, so 1M rows at dim 1024 is ~2 GB; a Full run
        # over millions of turns would otherwise grow this table without bound.
        self._embed_cache_rows = embed_cache_rows
        self._cache_writes = 0
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add the int8 `scale` column to a database created by an older version."""
        for table in ("vectors", "embed_cache"):
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if "scale" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN scale REAL NOT NULL DEFAULT 1.0")
                conn.commit()
        # v1.8: a turn can be superseded by a later retraction. NULL = live.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(units)")}
        if "superseded_by" not in cols:
            conn.execute("ALTER TABLE units ADD COLUMN superseded_by TEXT")
            conn.commit()

    # ---------------------------------------------------------------- plumbing
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # 8 KiB pages hold seven int8 vectors instead of three, which cuts the
            # page slack of the vectors table. Only a fresh database takes it.
            conn.execute("PRAGMA page_size=8192")
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
                    blob, scale = _vec_to_blob(vec)
                    payload.append((unit.id, model, int(vec.shape[0]), blob, scale))
                if payload:
                    conn.executemany(
                        "INSERT OR REPLACE INTO vectors(unit_id,model,dim,vec,scale) VALUES (?,?,?,?,?)",
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
                "INSERT OR REPLACE INTO embed_cache(hash,model,dim,vec,scale) VALUES (?,?,?,?,?)",
                [(h, model, int(v.shape[0]), *_vec_to_blob(v)) for h, v in items],
            )
            conn.commit()
            self._cache_writes += len(items)
            if self._embed_cache_rows and self._cache_writes >= 2000:
                # A cache is not worth a full disk: a long Full run writes one row
                # per embedded text (turn + query variant), which is the single
                # largest table. Keep the newest rows and let the oldest go; a
                # miss only costs one embedding call, and the vectors themselves
                # live in `vectors` either way.
                self._cache_writes = 0
                self._prune_embed_cache(conn)

    def _prune_embed_cache(self, conn: sqlite3.Connection) -> None:
        limit = self._embed_cache_rows
        if not limit:
            return
        count = conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
        if count <= limit + limit // 10:
            return
        conn.execute(
            "DELETE FROM embed_cache WHERE rowid IN ("
            " SELECT rowid FROM embed_cache ORDER BY rowid DESC LIMIT -1 OFFSET ?)",
            (limit,),
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
                f"SELECT hash, dim, vec, scale FROM embed_cache WHERE model = ? AND hash IN ({placeholders})",
                [model, *chunk],
            ).fetchall()
            for row in rows:
                out[row["hash"]] = _blob_to_vec(row["vec"], row["dim"], row["scale"])
        return out

    # ----------------------------------------------------------------- backfill
    def units_missing_vectors(self, limit: int = 500) -> list[tuple[str, str]]:
        """Units whose embedding never made it into the database.

        A turn is stored without a vector when the provider is down (see the Add
        handler); those rows stay retrievable through BM25 only.  This is what a
        backfill run has to repair once the provider comes back.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT u.id, u.text FROM units u"
            " LEFT JOIN vectors v ON v.unit_id = u.id"
            " WHERE v.unit_id IS NULL ORDER BY u.user_id, u.seq LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [(row["id"], row["text"]) for row in rows]

    def count_units_missing_vectors(self) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM units u LEFT JOIN vectors v ON v.unit_id = u.id"
            " WHERE v.unit_id IS NULL"
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------- governance
    def earlier_units(self, user_id: str, *, before_seq: int, limit: int = 600) -> list[Unit]:
        """Live turns of one user that predate `before_seq`, newest first."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id,user_id,session_id,seq,kind,speaker,created_ms,date_str,text,prev_id,next_id,meta"
            " FROM units WHERE user_id = ? AND seq < ? AND superseded_by IS NULL"
            " ORDER BY seq DESC LIMIT ?",
            (user_id, int(before_seq), int(limit)),
        ).fetchall()
        return [
            Unit(
                id=row["id"], user_id=row["user_id"], session_id=row["session_id"] or "",
                seq=row["seq"], kind=row["kind"], speaker=row["speaker"] or "",
                created_ms=row["created_ms"], date_str=row["date_str"] or "",
                text=row["text"], prev_id=row["prev_id"], next_id=row["next_id"],
                meta=json.loads(row["meta"]) if row["meta"] else {},
            )
            for row in rows
        ]

    def mark_superseded(self, unit_ids: Sequence[str], *, by_id: str) -> int:
        """Retire turns that a later turn explicitly retracted (never deletes them)."""
        if not unit_ids:
            return 0
        marked = 0
        with self._cache_lock:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for unit_id in unit_ids:
                    cur = conn.execute(
                        "UPDATE units SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL",
                        (by_id, unit_id),
                    )
                    if cur.rowcount:
                        marked += 1
                        row = conn.execute(
                            "SELECT user_id FROM units WHERE id = ?", (unit_id,)
                        ).fetchone()
                        if row is not None:
                            self._versions[row["user_id"]] = self._versions.get(row["user_id"], 0) + 1
                conn.commit()
        return marked

    def count_superseded(self) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM units WHERE superseded_by IS NOT NULL"
        ).fetchone()
        return int(row[0]) if row else 0

    def set_unit_vectors(
        self, model: str, items: Sequence[tuple[str, np.ndarray]]
    ) -> int:
        """Attach vectors to already-stored units and invalidate their user indexes."""
        if not items:
            return 0
        written = 0
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                payload = []
                for unit_id, vec in items:
                    if vec is None:
                        continue
                    blob, scale = _vec_to_blob(vec)
                    payload.append((unit_id, model, int(vec.shape[0]), blob, scale))
                if payload:
                    conn.executemany(
                        "INSERT OR REPLACE INTO vectors(unit_id,model,dim,vec,scale)"
                        " VALUES (?,?,?,?,?)",
                        payload,
                    )
                    written = len(payload)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if written:
            with self._cache_lock:
                self._indexes.clear()
                self._versions = {key: value + 1 for key, value in self._versions.items()}
        return written

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
            " FROM units WHERE user_id = ? AND superseded_by IS NULL ORDER BY seq",
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
                    f"SELECT unit_id,dim,vec,scale FROM vectors WHERE unit_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for vrow in vrows:
                    found_ids.append(vrow["unit_id"])
                    vectors.append(_blob_to_vec(vrow["vec"], vrow["dim"], vrow["scale"]))
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
        cached = conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
        # The operator needs to see the growth of the file itself: the unit count
        # alone does not reveal a runaway cache or a bloated WAL.
        db_bytes = 0
        for suffix in ("", "-wal"):
            try:
                db_bytes += Path(self.db_path + suffix).stat().st_size
            except OSError:
                pass
        # Disk head-room matters as much as the file size: a Full run can add tens of
        # gigabytes over days, and the only cheap way to watch that from outside is to
        # ship the free space in the same response the operator already polls.
        disk_free = disk_total = 0
        try:
            usage = shutil.disk_usage(str(Path(self.db_path).parent))
            disk_free, disk_total = int(usage.free), int(usage.total)
        except OSError:
            pass
        return {
            "units": units,
            "users": users,
            "add_requests": reqs,
            "vectors": vecs,
            "embed_cache": cached,
            "superseded": conn.execute(
                "SELECT COUNT(*) FROM units WHERE superseded_by IS NOT NULL"
            ).fetchone()[0],
            "db_bytes": db_bytes,
            "disk_free_bytes": disk_free,
            "disk_total_bytes": disk_total,
        }


def _vec_to_blob(vec: np.ndarray) -> tuple[bytes, float]:
    """Quantise a vector to int8 plus one scale so a stored turn costs dim bytes.

    The chat embedding of a turn is stored once and read once per search; at
    dim 1024 float16 costs 2 KB per turn and a Full run touches millions of
    turns, so the vector table (and its page slack) is most of the database.
    Per-vector symmetric int8 keeps ~0.4% of the magnitude range and loses about
    1e-3 of cosine, which is far below the reranker's resolution.
    """
    arr = np.asarray(vec, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(arr.shape[0], dtype=np.int8).tobytes(), 1.0
    scale = peak / 127.0
    quant = np.rint(arr / scale).clip(-127, 127).astype(np.int8)
    return quant.tobytes(), scale


def _blob_to_vec(blob: bytes, dim: int, scale: float = 0.0) -> np.ndarray:
    # Backwards compatible: an older database stores float16 at 2 bytes/dim.
    dtype = np.int8 if len(blob) == dim else np.float16
    arr = np.frombuffer(blob, dtype=dtype).astype(np.float32)
    if dtype is np.int8 and scale:
        arr = arr * np.float32(scale)
    if arr.shape[0] != dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: min(dim, arr.shape[0])] = arr[: min(dim, arr.shape[0])]
        return out
    return arr
