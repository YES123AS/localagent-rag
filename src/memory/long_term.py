"""Explainable structured long-term memory with CRUD and deduplication."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: str
    key: str
    value: Any
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongTermMemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories(
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(memory_type, memory_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type_key
                    ON long_term_memories(memory_type, memory_key);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"], memory_type=row["memory_type"], key=row["memory_key"],
            value=json.loads(row["value_json"]), metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def create(
        self, memory_type: str, key: str, value: Any, metadata: dict[str, Any] | None = None
    ) -> MemoryRecord:
        memory_type, key = memory_type.strip(), key.strip()
        if not memory_type or not key:
            raise ValueError("memory type and key are required")
        now = _now()
        memory_id = f"memory-{uuid.uuid4().hex}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO long_term_memories(
                    memory_id, memory_type, memory_key, value_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_type, memory_key) DO UPDATE SET
                    value_json=excluded.value_json, metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at""",
                (memory_id, memory_type, key, json.dumps(value, ensure_ascii=False, default=str),
                 json.dumps(metadata or {}, ensure_ascii=False, default=str), now, now),
            )
        return self.read(memory_type, key)

    def read(self, memory_type: str, key: str) -> MemoryRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM long_term_memories WHERE memory_type=? AND memory_key=?",
                (memory_type, key),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory: {memory_type}/{key}")
        return self._record(row)

    def update(self, memory_id: str, value: Any, metadata: dict[str, Any] | None = None) -> MemoryRecord:
        now = _now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM long_term_memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            merged_metadata = json.loads(row["metadata_json"])
            if metadata:
                merged_metadata.update(metadata)
            connection.execute(
                "UPDATE long_term_memories SET value_json=?, metadata_json=?, updated_at=? WHERE memory_id=?",
                (json.dumps(value, ensure_ascii=False, default=str),
                 json.dumps(merged_metadata, ensure_ascii=False, default=str), now, memory_id),
            )
            updated = connection.execute(
                "SELECT * FROM long_term_memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
        return self._record(updated)

    def delete(self, memory_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM long_term_memories WHERE memory_id=?", (memory_id,))
            return cursor.rowcount > 0

    def search(self, query: str = "", *, memory_type: str | None = None, limit: int = 20) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if memory_type:
            clauses.append("memory_type=?")
            params.append(memory_type)
        terms = [term.lower() for term in query.split() if len(term) > 1]
        sql = "SELECT * FROM long_term_memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit) * 5, 500)))
        with self._lock, self._connect() as connection:
            records = [self._record(row) for row in connection.execute(sql, params).fetchall()]
        if terms:
            records = [
                record for record in records
                if any(term in f"{record.key} {record.value}".lower() for term in terms)
            ]
        return records[:limit]
