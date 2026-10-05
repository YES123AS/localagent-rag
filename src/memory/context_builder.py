"""Budget-aware context assembly with relevance ranking before trimming."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    # Chinese characters are close to one token; latin text averages roughly four chars/token.
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    return max(1, chinese + (len(text) - chinese + 3) // 4)


@dataclass(frozen=True)
class TokenBudget:
    total: int = 8000
    conversation: int = 1800
    memory: int = 1000
    evidence: int = 3600
    prompt: int = 800
    output: int = 800

    def __post_init__(self) -> None:
        if min(self.total, self.conversation, self.memory, self.evidence, self.prompt, self.output) < 0:
            raise ValueError("token budgets cannot be negative")
        if self.conversation + self.memory + self.evidence + self.prompt + self.output > self.total:
            raise ValueError("token budget sections exceed total")


@dataclass
class ContextPackage:
    query: str
    goal: str
    plan: dict[str, Any]
    conversation: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    trimmed: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query, "goal": self.goal, "plan": self.plan,
            "conversation": self.conversation, "memory": self.memory,
            "evidence": self.evidence, "token_usage": self.token_usage,
            "trimmed": self.trimmed,
        }


class ContextBuilder:
    def __init__(self, budget: TokenBudget | None = None):
        self.budget = budget or TokenBudget()

    @staticmethod
    def _terms(query: str) -> set[str]:
        return set(re.findall(r"[\w\u3400-\u9fff]{2,}", query.lower()))

    def _score(self, query: str, item: Any) -> float:
        text = json.dumps(item, ensure_ascii=False, default=str).lower()
        terms = self._terms(query)
        overlap = sum(text.count(term) for term in terms)
        explicit = float(item.get("rerank_score") or item.get("retrieval_score") or 0) if isinstance(item, dict) else 0
        return overlap + explicit

    def _select(self, query: str, items: Iterable[Any], limit: int, *, newest_first: bool = False) -> tuple[list[Any], int]:
        source = list(items)
        ranked = list(reversed(source)) if newest_first else sorted(
            source, key=lambda item: self._score(query, item), reverse=True
        )
        selected: list[Any] = []
        used = 0
        for item in ranked:
            size = estimate_tokens(item)
            if size > limit or used + size > limit:
                continue
            selected.append(item)
            used += size
        if newest_first:
            selected.reverse()
        return selected, len(source) - len(selected)

    @staticmethod
    def _trim_text(text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return ""
        if estimate_tokens(text) <= token_limit:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(text[:middle]) <= token_limit:
                low = middle
            else:
                high = middle - 1
        return text[:low]

    def build(
        self,
        *,
        query: str,
        goal: str = "",
        plan: dict[str, Any] | None = None,
        conversation: Iterable[dict[str, Any]] = (),
        memory: Iterable[dict[str, Any]] = (),
        evidence: Iterable[dict[str, Any]] = (),
    ) -> ContextPackage:
        bounded_query = self._trim_text(query, self.budget.prompt)
        prompt_used = estimate_tokens(bounded_query) if bounded_query else 0
        remaining_prompt = max(0, self.budget.prompt - prompt_used)
        bounded_goal = self._trim_text(goal, remaining_prompt)
        prompt_used += estimate_tokens(bounded_goal) if bounded_goal else 0
        remaining_prompt = max(0, self.budget.prompt - prompt_used)
        bounded_plan = plan or {}
        if estimate_tokens(bounded_plan) > remaining_prompt:
            bounded_plan = {}
        prompt_used += estimate_tokens(bounded_plan) if bounded_plan else 0
        selected_conversation, conversation_trimmed = self._select(
            query, conversation, self.budget.conversation, newest_first=True
        )
        selected_memory, memory_trimmed = self._select(query, memory, self.budget.memory)
        selected_evidence, evidence_trimmed = self._select(query, evidence, self.budget.evidence)
        usage = {
            "prompt": prompt_used,
            "conversation": estimate_tokens(selected_conversation) if selected_conversation else 0,
            "memory": estimate_tokens(selected_memory) if selected_memory else 0,
            "evidence": estimate_tokens(selected_evidence) if selected_evidence else 0,
            "output_reserved": self.budget.output,
        }
        usage["total"] = sum(usage.values())
        return ContextPackage(
            query=bounded_query, goal=bounded_goal, plan=bounded_plan, conversation=selected_conversation,
            memory=selected_memory, evidence=selected_evidence, token_usage=usage,
            trimmed={
                "conversation": conversation_trimmed, "memory": memory_trimmed,
                "evidence": evidence_trimmed,
            },
        )
