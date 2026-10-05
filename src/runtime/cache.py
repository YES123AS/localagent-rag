"""Small SQLite TTL cache used for deterministic local reuse."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class SQLiteTTLCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runtime_cache(
                    namespace TEXT NOT NULL, cache_key TEXT NOT NULL, value_json TEXT NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL,
                    PRIMARY KEY(namespace, cache_key)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def key(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, namespace: str, key: str) -> Any | None:
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value_json, expires_at FROM runtime_cache WHERE namespace=? AND cache_key=?",
                (namespace, key),
            ).fetchone()
            if not row:
                return None
            if float(row[1]) <= now:
                connection.execute(
                    "DELETE FROM runtime_cache WHERE namespace=? AND cache_key=?", (namespace, key)
                )
                return None
            return json.loads(row[0])

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO runtime_cache(namespace, cache_key, value_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    value_json=excluded.value_json, created_at=excluded.created_at,
                    expires_at=excluded.expires_at""",
                (namespace, key, json.dumps(value, ensure_ascii=False, default=str), now, now + ttl_seconds),
            )

    def invalidate(self, namespace: str, key: str | None = None) -> int:
        with self._lock, self._connect() as connection:
            if key is None:
                cursor = connection.execute("DELETE FROM runtime_cache WHERE namespace=?", (namespace,))
            else:
                cursor = connection.execute(
                    "DELETE FROM runtime_cache WHERE namespace=? AND cache_key=?", (namespace, key)
                )
            return int(cursor.rowcount)

    def purge_expired(self) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM runtime_cache WHERE expires_at <= ?", (time.time(),))
            return int(cursor.rowcount)
