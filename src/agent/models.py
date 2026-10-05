"""Validated plans and the shared evidence contract."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

from .policies import MAX_PLAN_STEPS


ToolName = str


class Evidence(TypedDict, total=False):
    evidence_id: str
    source_type: Literal["knowledge_base", "web", "calculator"]
    content: str
    title: Optional[str]
    url: Optional[str]
    published_at: Optional[str]
    document_id: Optional[str]
    document_name: Optional[str]
    file_type: Optional[str]
    page: Optional[int]
    chunk_id: Optional[str]
    chunk_index: Optional[int]
    retrieval_score: Optional[float]
    rerank_score: Optional[float]
    tool_call_id: str


class PlanStep(BaseModel):
    id: int = Field(ge=1)
    action: Literal["tool", "synthesize"] = "tool"
    tool: Optional[ToolName] = None
    query: str = Field(default="", max_length=3000)
    depends_on: List[int] = Field(default_factory=list)
    reason: str = Field(default="", max_length=500)
    idempotency_key: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "tool" and self.tool is None:
            raise ValueError("A tool step must name a supported tool")
        if self.action == "tool" and not self.query.strip():
            raise ValueError("A tool step must include a non-empty query")
        if self.action == "synthesize" and self.tool is not None:
            raise ValueError("A synthesis step cannot name a tool")
        if self.tool is not None and self.tool not in {
            "knowledge_base", "web_search", "calculator"
        } and not self.tool.startswith("mcp."):
            raise ValueError("Unsupported tool; MCP tool names must start with 'mcp.'")
        return self


class ExecutionPlan(BaseModel):
    goal: str = Field(min_length=1, max_length=1000)
    steps: List[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan step ids must be unique")
        seen: set[int] = set()
        synthesize_count = 0
        for step in self.steps:
            if any(dependency not in seen for dependency in step.depends_on):
                raise ValueError("Plan dependencies must refer to earlier steps")
            seen.add(step.id)
            synthesize_count += int(step.action == "synthesize")
        if synthesize_count > 1:
            raise ValueError("A plan can contain at most one synthesis step")
        if self.steps[-1].action != "synthesize":
            next_id = max(ids) + 1
            if len(self.steps) >= MAX_PLAN_STEPS:
                self.steps[-1] = PlanStep(
                    id=self.steps[-1].id,
                    action="synthesize",
                    depends_on=ids[:-1],
                    reason="Synthesize collected evidence",
                )
            else:
                self.steps.append(
                    PlanStep(
                        id=next_id,
                        action="synthesize",
                        depends_on=ids,
                        reason="Synthesize collected evidence",
                    )
                )
        return self

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ComplexityDecision(BaseModel):
    complexity: Literal["simple", "complex"]
    reason: str = Field(min_length=1, max_length=500)
