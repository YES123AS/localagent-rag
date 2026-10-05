"""Unified execution budget with serializable usage counters."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .errors import BudgetExceededError


@dataclass
class ExecutionBudget:
    max_llm_calls: int = 12
    max_tool_calls: int = 6
    max_web_calls: int = 3
    max_replans: int = 2
    max_tokens: int = 24_000
    max_duration_seconds: int = 900
    llm_calls: int = 0
    tool_calls: int = 0
    web_calls: int = 0
    replans: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_llm_calls", "max_tool_calls", "max_web_calls", "max_replans",
            "max_tokens", "max_duration_seconds",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        self._timer_started = time.monotonic()

    def _current_elapsed(self) -> float:
        return self.elapsed_seconds + max(0.0, time.monotonic() - self._timer_started)

    def check(self, kind: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("budget amount cannot be negative")
        mapping = {
            "llm": (self.llm_calls, self.max_llm_calls),
            "tool": (self.tool_calls, self.max_tool_calls),
            "web": (self.web_calls, self.max_web_calls),
            "replan": (self.replans, self.max_replans),
            "tokens": (self.tokens, self.max_tokens),
        }
        if kind not in mapping:
            raise ValueError(f"Unknown budget kind: {kind}")
        used, maximum = mapping[kind]
        if used + amount > maximum:
            raise BudgetExceededError(f"{kind} budget exhausted ({used}/{maximum})")
        if self._current_elapsed() > self.max_duration_seconds:
            raise BudgetExceededError(
                f"duration budget exhausted ({self._current_elapsed():.1f}/{self.max_duration_seconds}s)"
            )

    def consume(self, kind: str, amount: int = 1) -> None:
        self.check(kind, amount)
        attribute = {
            "llm": "llm_calls", "tool": "tool_calls", "web": "web_calls",
            "replan": "replans", "tokens": "tokens",
        }[kind]
        setattr(self, attribute, int(getattr(self, attribute)) + amount)

    def remaining(self, kind: str) -> int:
        used, maximum = {
            "llm": (self.llm_calls, self.max_llm_calls),
            "tool": (self.tool_calls, self.max_tool_calls),
            "web": (self.web_calls, self.max_web_calls),
            "replan": (self.replans, self.max_replans),
            "tokens": (self.tokens, self.max_tokens),
        }[kind]
        return max(0, maximum - used)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_seconds"] = round(self._current_elapsed(), 6)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ExecutionBudget":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (payload or {}).items() if key in allowed})
