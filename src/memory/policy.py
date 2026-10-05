"""Conservative memory-write policy: durable facts in, transient chatter out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryDecision:
    should_store: bool
    reason: str


class MemoryWritePolicy:
    ALLOWED_TYPES = {"project", "preference", "profile", "reusable_context"}
    TRANSIENT_KEYS = {"calculation", "temporary", "current_time", "one_off", "chat"}

    def evaluate(self, candidate: dict[str, Any]) -> MemoryDecision:
        memory_type = str(candidate.get("type", "")).strip().lower()
        key = str(candidate.get("key", "")).strip().lower()
        value = candidate.get("value")
        if memory_type not in self.ALLOWED_TYPES:
            return MemoryDecision(False, "Only stable structured memory types are allowed")
        if not key or value in (None, "", [], {}):
            return MemoryDecision(False, "Memory key and value are required")
        if key in self.TRANSIENT_KEYS or bool(candidate.get("temporary")):
            return MemoryDecision(False, "Transient information is not long-term memory")
        if len(str(value)) > 4000:
            return MemoryDecision(False, "Candidate is too large for structured memory")
        return MemoryDecision(True, "Stable reusable information")

    def store(self, store: Any, candidate: dict[str, Any]):
        decision = self.evaluate(candidate)
        if not decision.should_store:
            return None
        return store.create(
            str(candidate["type"]), str(candidate["key"]), candidate["value"],
            metadata={"write_reason": decision.reason, **dict(candidate.get("metadata") or {})},
        )
