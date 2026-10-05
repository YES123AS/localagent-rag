"""Tool metadata shared by native and MCP-backed tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 8.0


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    timeout: float = 30.0
    max_calls: int = 6
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    permissions: tuple[str, ...] = ()
    external: bool = False
    side_effecting: bool = False


@dataclass
class RegisteredTool:
    metadata: ToolMetadata
    handler: Callable[[dict[str, Any]], Any]

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.handler(dict(arguments))
