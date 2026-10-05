"""Validated contracts exchanged between V5 agents."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    LOCAL_RESEARCH = "local_research_agent"
    WEB_RESEARCH = "web_research_agent"
    CRITIC = "critic_agent"
    WRITER = "writer_agent"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResearchTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex}")
    title: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=1, max_length=3000)
    assigned_agent: AgentRole
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    run_id: str | None = None
    retry_count: int = Field(default=0, ge=0)
    research_round: int = Field(default=0, ge=0)
    error: str = ""


class MultiAgentPlan(BaseModel):
    goal: str = Field(min_length=1, max_length=3000)
    tasks: list[ResearchTask] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dependencies(self):
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("Research task ids must be unique")
        for task in self.tasks:
            if task.task_id in task.depends_on:
                raise ValueError("A research task cannot depend on itself")
            if not set(task.depends_on).issubset(ids):
                raise ValueError("Research task dependency does not exist")
        return self


class Handoff(BaseModel):
    handoff_id: str = Field(default_factory=lambda: f"handoff-{uuid.uuid4().hex}")
    parent_run_id: str
    from_agent: AgentRole
    to_agent: AgentRole
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str
    status: str = "COMPLETED"


class CriticResult(BaseModel):
    valid: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    need_more_research: bool = False
    validated_evidence_ids: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"evidence-{uuid.uuid4().hex}")
    parent_run_id: str
    source_type: str
    source: str
    content: str
    agent_id: str
    task_id: str
    citation: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    timestamp: str
    validated: bool = False
    conflict: bool = False
    fingerprint: str
