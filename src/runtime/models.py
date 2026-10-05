"""Agent Run lifecycle contracts kept separate from conversation memory."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .budget import ExecutionBudget


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {RunStatus.FAILED, RunStatus.COMPLETED, RunStatus.CANCELLED}

ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
    RunStatus.RUNNING: {
        RunStatus.WAITING_APPROVAL, RunStatus.PAUSED, RunStatus.FAILED,
        RunStatus.COMPLETED, RunStatus.CANCELLED,
    },
    RunStatus.WAITING_APPROVAL: {
        RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.CANCELLED, RunStatus.FAILED,
    },
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
    RunStatus.FAILED: set(),
    RunStatus.COMPLETED: set(),
    RunStatus.CANCELLED: set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRun:
    conversation_id: str
    goal: str
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex}")
    status: RunStatus = RunStatus.PENDING
    plan: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    completed_steps: list[int] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    replan_count: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str = ""
    error_type: str = ""
    result: str = ""
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    checkpoint_count: int = 0
    approval_count: int = 0
    worker_id: str = ""
    queued_at: str | None = None
    queue_time_ms: float = 0.0
    resume_count: int = 0
    recovery_result: str = ""
    context_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, status: RunStatus) -> None:
        status = RunStatus(status)
        if status == self.status:
            return
        if status not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"Invalid run transition: {self.status.value} -> {status.value}")
        self.status = status
        self.updated_at = utc_now()
        if status == RunStatus.RUNNING and self.started_at is None:
            self.started_at = self.updated_at
        if status in TERMINAL_STATUSES:
            self.completed_at = self.updated_at

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["budget"] = self.budget.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentRun":
        values = dict(payload)
        values["status"] = RunStatus(values.get("status", RunStatus.PENDING))
        values["budget"] = ExecutionBudget.from_dict(values.get("budget"))
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})
