"""Small SQLite conversation store with explicit, JSON-safe records."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str = "新对话", conversation_id: str | None = None) -> str:
        conversation_id = conversation_id or f"conversation-{uuid.uuid4().hex}"
        now = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported message role")
        now = self._now()
        payload = json.dumps(metadata or {}, ensure_ascii=False, default=str)
        with self._lock, self._connect() as connection:
            self.create_conversation(conversation_id=conversation_id)
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, payload, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def replace_messages(self, conversation_id: str, title: str, messages: list[dict[str, Any]]) -> None:
        now = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
                (conversation_id, title, now, now),
            )
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.executemany(
                "INSERT INTO messages(conversation_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        conversation_id,
                        message.get("role", "assistant"),
                        str(message.get("content", "")),
                        json.dumps(message.get("metadata", {}), ensure_ascii=False, default=str),
                        now,
                    )
                    for message in messages
                ],
            )

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at"
            ).fetchall()
            result = []
            for row in rows:
                messages = connection.execute(
                    "SELECT role, content, metadata_json FROM messages WHERE conversation_id = ? ORDER BY id",
                    (row["id"],),
                ).fetchall()
                result.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "messages": [
                            {
                                "role": message["role"],
                                "content": message["content"],
                                "metadata": json.loads(message["metadata_json"] or "{}"),
                            }
                            for message in messages
                        ],
                    }
                )
            return result

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
