"""Persistent shared evidence and structured handoffs for V5."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from src.runtime.models import utc_now

from .models import EvidenceRecord, Handoff


class SharedEvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shared_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    parent_run_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    citation_json TEXT NOT NULL,
                    score REAL,
                    timestamp TEXT NOT NULL,
                    validated INTEGER NOT NULL DEFAULT 0,
                    conflict INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT NOT NULL,
                    UNIQUE(parent_run_id, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_shared_evidence_run
                    ON shared_evidence(parent_run_id, timestamp);
                CREATE TABLE IF NOT EXISTS agent_handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    parent_run_id TEXT NOT NULL,
                    from_agent TEXT NOT NULL,
                    to_agent TEXT NOT NULL,
                    task TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_handoffs_run
                    ON agent_handoffs(parent_run_id, created_at);
                """
            )

    @staticmethod
    def _fingerprint(source_type: str, source: str, content: str) -> str:
        raw = "\n".join((source_type.strip(), source.strip(), content.strip()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add(
        self,
        *,
        parent_run_id: str,
        evidence: dict[str, Any],
        agent_id: str,
        task_id: str,
    ) -> tuple[EvidenceRecord, bool]:
        source_type = str(evidence.get("source_type") or "tool")
        source = str(
            evidence.get("url")
            or evidence.get("document_name")
            or evidence.get("title")
            or source_type
        )
        content = str(evidence.get("content") or "").strip()
        citation = {
            key: evidence.get(key)
            for key in (
                "title", "url", "document_id", "document_name", "page", "chunk_id",
                "claim", "claim_value",
            )
            if evidence.get(key) not in (None, "")
        }
        fingerprint = self._fingerprint(source_type, source, content)
        score = evidence.get("rerank_score", evidence.get("retrieval_score"))
        record = EvidenceRecord(
            parent_run_id=parent_run_id,
            source_type=source_type,
            source=source,
            content=content,
            agent_id=agent_id,
            task_id=task_id,
            citation=citation,
            score=float(score) if score is not None else None,
            timestamp=utc_now(),
            fingerprint=fingerprint,
        )
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM shared_evidence WHERE parent_run_id=? AND fingerprint=?",
                (parent_run_id, fingerprint),
            ).fetchone()
            if existing:
                return self._from_row(existing), True
            connection.execute(
                """INSERT INTO shared_evidence(
                    evidence_id,parent_run_id,source_type,source,content,agent_id,task_id,
                    citation_json,score,timestamp,validated,conflict,fingerprint
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.evidence_id, record.parent_run_id, record.source_type,
                    record.source, record.content, record.agent_id, record.task_id,
                    json.dumps(record.citation, ensure_ascii=False), record.score,
                    record.timestamp, 0, 0, record.fingerprint,
                ),
            )
        return record, False

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=row["evidence_id"], parent_run_id=row["parent_run_id"],
            source_type=row["source_type"], source=row["source"], content=row["content"],
            agent_id=row["agent_id"], task_id=row["task_id"],
            citation=json.loads(row["citation_json"]), score=row["score"],
            timestamp=row["timestamp"], validated=bool(row["validated"]),
            conflict=bool(row["conflict"]), fingerprint=row["fingerprint"],
        )

    def list(self, parent_run_id: str, *, validated_only: bool = False) -> list[EvidenceRecord]:
        query = "SELECT * FROM shared_evidence WHERE parent_run_id=?"
        if validated_only:
            query += " AND validated=1 AND conflict=0"
        query += " ORDER BY timestamp, evidence_id"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, (parent_run_id,)).fetchall()
        return [self._from_row(row) for row in rows]

    def set_verification(
        self,
        parent_run_id: str,
        validated_ids: list[str],
        conflict_ids: list[str] | None = None,
    ) -> None:
        validated = set(validated_ids)
        conflicts = set(conflict_ids or ())
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT evidence_id FROM shared_evidence WHERE parent_run_id=?",
                (parent_run_id,),
            ).fetchall()
            for row in rows:
                evidence_id = row["evidence_id"]
                connection.execute(
                    "UPDATE shared_evidence SET validated=?, conflict=? WHERE evidence_id=?",
                    (int(evidence_id in validated), int(evidence_id in conflicts), evidence_id),
                )

    def save_handoff(self, handoff: Handoff) -> Handoff:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO agent_handoffs(
                    handoff_id,parent_run_id,from_agent,to_agent,task,context_json,
                    evidence_refs_json,reason,status,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    handoff.handoff_id, handoff.parent_run_id,
                    handoff.from_agent.value, handoff.to_agent.value, handoff.task,
                    json.dumps(handoff.context, ensure_ascii=False, default=str),
                    json.dumps(handoff.evidence_refs, ensure_ascii=False),
                    handoff.reason, handoff.status, utc_now(),
                ),
            )
        return handoff

    def list_handoffs(self, parent_run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_handoffs WHERE parent_run_id=? ORDER BY created_at, handoff_id",
                (parent_run_id,),
            ).fetchall()
        return [
            {
                "handoff_id": row["handoff_id"], "parent_run_id": row["parent_run_id"],
                "from_agent": row["from_agent"], "to_agent": row["to_agent"],
                "task": row["task"], "context": json.loads(row["context_json"]),
                "evidence_refs": json.loads(row["evidence_refs_json"]),
                "reason": row["reason"], "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
