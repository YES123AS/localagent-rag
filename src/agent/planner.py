"""Planner helpers kept independent from any specific LLM client."""

import re
from typing import Any, Mapping

from .models import ComplexityDecision, ExecutionPlan, PlanStep


LOCAL_MARKERS = (
    "上传", "知识库", "本地资料", "文档", "论文", "uploaded", "knowledge base", "local document"
)
WEB_MARKERS = (
    "最新", "今天", "近期", "实时", "网络", "联网", "latest", "today", "current", "web"
)
CALC_MARKERS = (
    "计算", "增长率", "同比", "环比", "百分比", "calculate", "growth rate", "percentage"
)
COMBINATION_MARKERS = (
    "结合", "比较", "对比", "综合", "分别", "并根据", "compare", "combine", "and then", "based on"
)


def fallback_complexity(question: str) -> ComplexityDecision:
    """Conservative deterministic classifier used when the model is unavailable."""
    lowered = question.lower()
    families = sum(
        (
            any(marker in lowered for marker in LOCAL_MARKERS),
            any(marker in lowered for marker in WEB_MARKERS),
            any(marker in lowered for marker in CALC_MARKERS),
        )
    )
    explicitly_combined = any(marker in lowered for marker in COMBINATION_MARKERS)
    is_complex = families >= 2 and explicitly_combined
    return ComplexityDecision(
        complexity="complex" if is_complex else "simple",
        reason=(
            "The request explicitly combines multiple evidence or tool families."
            if is_complex
            else "A single bounded capability is sufficient."
        ),
    )


def fallback_plan(question: str) -> ExecutionPlan:
    """Create a valid bounded plan without relying on model availability."""
    lowered = question.lower()
    tools: list[str] = []
    if any(marker in lowered for marker in LOCAL_MARKERS):
        tools.append("knowledge_base")
    if any(marker in lowered for marker in WEB_MARKERS):
        tools.append("web_search")
    if any(marker in lowered for marker in CALC_MARKERS):
        tools.append("calculator")
    if len(tools) < 2:
        tools = ["knowledge_base", "web_search"]

    steps = [
        PlanStep(
            id=index,
            tool=tool,
            query=question,
            reason=f"Collect {tool} evidence for the stated goal",
        )
        for index, tool in enumerate(tools, start=1)
    ]
    steps.append(
        PlanStep(
            id=len(steps) + 1,
            action="synthesize",
            depends_on=[step.id for step in steps],
            reason="Combine and cite the collected evidence",
        )
    )
    return ExecutionPlan(goal=question, steps=steps)


def validate_plan(payload: Mapping[str, Any]) -> ExecutionPlan:
    """Validate untrusted planner output and reject unsupported actions/tools."""
    return ExecutionPlan.model_validate(dict(payload))


def extract_arithmetic_expression(text: str) -> str:
    """Return the most likely standalone arithmetic expression from text."""
    matches = re.findall(r"[-+*/%^().\d\s×÷]+", text)
    candidates = [candidate.strip() for candidate in matches if any(ch.isdigit() for ch in candidate)]
    return max(candidates, key=len, default="")
