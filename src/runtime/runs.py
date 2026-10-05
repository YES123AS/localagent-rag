"""SQLite persistence for Agent Runs, checkpoints, and approval decisions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .errors import ApprovalError, PersistenceError
from .models import AgentRun, RunStatus, TERMINAL_STATUSES, utc_now
from src.observability.tracing import record_runtime_event


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
                        ON agent_runs(conversation_id, updated_at);
                    CREATE TABLE IF NOT EXISTS run_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        state_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(run_id, sequence),
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS run_approvals (
                        approval_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        tool_name TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        decided_at TEXT,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS run_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_run_events_run_id
                        ON run_events(run_id, id);
                    CREATE TABLE IF NOT EXISTS idempotency_records (
                        idempotency_key TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        output_json TEXT,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    );
                    """
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Could not initialize run store: {exc}") from exc

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

    def save_run(self, run: AgentRun) -> AgentRun:
        run.updated_at = utc_now()
        payload = run.to_dict()
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """INSERT INTO agent_runs(run_id, conversation_id, status, goal, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        conversation_id=excluded.conversation_id,
                        status=excluded.status,
                        goal=excluded.goal,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at""",
                    (run.run_id, run.conversation_id, run.status.value, run.goal,
                     self._dump(payload), run.created_at, run.updated_at),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Could not save run {run.run_id}: {exc}") from exc
        return run

    def create_run(self, conversation_id: str, goal: str, **kwargs: Any) -> AgentRun:
        run = self.save_run(AgentRun(conversation_id=conversation_id, goal=goal, **kwargs))
        self.append_event(run.run_id, "run_created", {"status": run.status.value})
        return run

    def append_event(
        self, run_id: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO run_events(run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, event_type, self._dump(payload or {}), created_at),
            )
            sequence = int(cursor.lastrowid)
        event = {
            "sequence": sequence, "run_id": run_id, "event_type": event_type,
            "payload": payload or {}, "created_at": created_at,
        }
        record_runtime_event(
            "agent.run.stream_event", run_id=run_id, stream_event=event_type,
            event_sequence=sequence,
            worker_id=(payload or {}).get("worker_id"),
            queue_time=(payload or {}).get("queue_time_ms"),
            resume_count=(payload or {}).get("resume_count"),
            idempotency_key=(payload or {}).get("idempotency_key"),
            recovery_result=(payload or {}).get("recovery_result"),
        )
        return event

    def list_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id, event_type, payload_json, created_at FROM run_events
                WHERE run_id=? AND id>? ORDER BY id LIMIT ?""",
                (run_id, int(after_sequence), max(1, min(int(limit), 5000))),
            ).fetchall()
        return [
            {
                "sequence": int(row["id"]), "run_id": run_id,
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def begin_idempotent_call(
        self, idempotency_key: str, run_id: str, tool_name: str
    ) -> dict[str, Any]:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required for side-effecting tools")
        now = utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM idempotency_records WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row:
                result = dict(row)
                result["output"] = (
                    json.loads(row["output_json"]) if row["output_json"] is not None else None
                )
                return result
            connection.execute(
                """INSERT INTO idempotency_records(
                    idempotency_key, run_id, tool_name, status, created_at
                ) VALUES (?, ?, ?, 'STARTED', ?)""",
                (idempotency_key, run_id, tool_name, now),
            )
        return {
            "idempotency_key": idempotency_key, "run_id": run_id,
            "tool_name": tool_name, "status": "STARTED", "output": None,
            "created_at": now,
        }

    def complete_idempotent_call(self, idempotency_key: str, output: Any) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE idempotency_records SET status='COMPLETED', output_json=?, completed_at=?
                WHERE idempotency_key=?""",
                (self._dump(output), utc_now(), idempotency_key),
            )
            if cursor.rowcount != 1:
                raise PersistenceError(f"Unknown idempotency key: {idempotency_key}")

    def fail_idempotent_call(self, idempotency_key: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM idempotency_records WHERE idempotency_key=? AND status='STARTED'",
                (idempotency_key,),
            )

    def get_run(self, run_id: str) -> AgentRun:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM agent_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise PersistenceError(f"Could not load run {run_id}: {exc}") from exc
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        return AgentRun.from_dict(json.loads(row["payload_json"]))

    def list_runs(self, conversation_id: str | None = None, limit: int = 50) -> list[AgentRun]:
        query = "SELECT payload_json FROM agent_runs"
        params: list[Any] = []
        if conversation_id:
            query += " WHERE conversation_id = ?"
            params.append(conversation_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [AgentRun.from_dict(json.loads(row["payload_json"])) for row in rows]

    def transition(self, run_id: str, status: RunStatus) -> AgentRun:
        run = self.get_run(run_id)
        run.transition(status)
        return self.save_run(run)

    def pause(self, run_id: str) -> AgentRun:
        return self.transition(run_id, RunStatus.PAUSED)

    def resume(self, run_id: str) -> AgentRun:
        return self.transition(run_id, RunStatus.RUNNING)

    def cancel(self, run_id: str) -> AgentRun:
        run = self.get_run(run_id)
        if run.status in TERMINAL_STATUSES:
            return run
        run.transition(RunStatus.CANCELLED)
        return self.save_run(run)

    def recover_interrupted_runs(self) -> list[AgentRun]:
        """Move stale RUNNING records to PAUSED after a single-instance restart."""
        recovered: list[AgentRun] = []
        for run in self.list_runs(limit=500):
            if run.status != RunStatus.RUNNING:
                continue
            run.transition(RunStatus.PAUSED)
            run.metadata["recovery_reason"] = "Process stopped while the run was active"
            self.save_checkpoint(run)
            recovered.append(run)
        return recovered

    def save_checkpoint(self, run: AgentRun, state: dict[str, Any] | None = None) -> dict[str, Any]:
        run.checkpoint_count += 1
        checkpoint = dict(state or run.to_dict())
        checkpoint["run_id"] = run.run_id
        checkpoint["checkpoint_sequence"] = run.checkpoint_count
        checkpoint["saved_at"] = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO run_checkpoints(run_id, sequence, state_json, created_at) VALUES (?, ?, ?, ?)",
                    (run.run_id, run.checkpoint_count, self._dump(checkpoint), checkpoint["saved_at"]),
                )
                payload = run.to_dict()
                connection.execute(
                    "UPDATE agent_runs SET status=?, payload_json=?, updated_at=? WHERE run_id=?",
                    (run.status.value, self._dump(payload), run.updated_at, run.run_id),
                )
        except sqlite3.Error as exc:
            run.checkpoint_count -= 1
            raise PersistenceError(f"Could not checkpoint run {run.run_id}: {exc}") from exc
        record_runtime_event(
            "agent.run.checkpoint", run_id=run.run_id,
            checkpoint_count=run.checkpoint_count, current_step=run.current_step,
            status=run.status.value,
        )
        return checkpoint

    def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT state_json FROM run_checkpoints WHERE run_id=? ORDER BY sequence DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise PersistenceError(f"Could not load checkpoint for {run_id}: {exc}") from exc
        if row is None:
            raise PersistenceError(f"No checkpoint exists for run {run_id}")
        return json.loads(row["state_json"])

    def request_approval(self, run: AgentRun, step_id: int, tool_name: str, reason: str) -> dict[str, Any]:
        pending = self.pending_approval(run.run_id, step_id)
        if pending:
            return pending
        approval = {
            "approval_id": f"approval-{uuid.uuid4().hex}", "run_id": run.run_id,
            "step_id": int(step_id), "tool_name": tool_name, "reason": reason,
            "status": "PENDING", "created_at": utc_now(), "decided_at": None,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO run_approvals(approval_id, run_id, step_id, tool_name, reason, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                tuple(approval[key] for key in (
                    "approval_id", "run_id", "step_id", "tool_name", "reason", "status", "created_at"
                )),
            )
        run.approval_count += 1
        if run.status == RunStatus.RUNNING:
            run.transition(RunStatus.WAITING_APPROVAL)
        self.save_run(run)
        record_runtime_event(
            "agent.run.approval.requested", run_id=run.run_id,
            approval_id=approval["approval_id"], step_id=step_id, tool=tool_name,
        )
        return approval

    def pending_approval(self, run_id: str, step_id: int | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM run_approvals WHERE run_id=? AND status='PENDING'"
        params: list[Any] = [run_id]
        if step_id is not None:
            query += " AND step_id=?"
            params.append(int(step_id))
        query += " ORDER BY created_at LIMIT 1"
        with self._lock, self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def approval_for_step(self, run_id: str, step_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_approvals WHERE run_id=? AND step_id=? ORDER BY created_at DESC LIMIT 1",
                (run_id, int(step_id)),
            ).fetchone()
        return dict(row) if row else None

    def decide_approval(self, approval_id: str, approve: bool) -> dict[str, Any]:
        status = "APPROVED" if approve else "REJECTED"
        decided_at = utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
            if row is None:
                raise ApprovalError(f"Unknown approval: {approval_id}")
            if row["status"] != "PENDING":
                raise ApprovalError(f"Approval was already decided: {approval_id}")
            connection.execute(
                "UPDATE run_approvals SET status=?, decided_at=? WHERE approval_id=?",
                (status, decided_at, approval_id),
            )
        result = dict(row)
        result.update(status=status, decided_at=decided_at)
        record_runtime_event(
            "agent.run.approval.decided", run_id=result["run_id"],
            approval_id=approval_id, step_id=result["step_id"], decision=status,
        )
        return result
