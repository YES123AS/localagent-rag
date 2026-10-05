"""Permission, approval, status, call-limit, and budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.runtime.errors import BudgetExceededError
from src.runtime.models import AgentRun, RunStatus
from src.tools.base import ToolMetadata


class PolicyOutcome(str, Enum):
    ALLOW = "ALLOW"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    DENY = "DENY"


@dataclass(frozen=True)
class PolicyDecision:
    outcome: PolicyOutcome
    reason: str


class ToolPolicyEngine:
    def __init__(self, granted_permissions: set[str] | None = None):
        self.granted_permissions = set(granted_permissions or ())

    def evaluate(
        self,
        run: AgentRun,
        metadata: ToolMetadata,
        *,
        approved: bool = False,
    ) -> PolicyDecision:
        if run.status != RunStatus.RUNNING:
            return PolicyDecision(PolicyOutcome.DENY, f"Run status is {run.status.value}")
        missing = set(metadata.permissions) - self.granted_permissions
        if missing:
            return PolicyDecision(
                PolicyOutcome.DENY, f"Missing permissions: {', '.join(sorted(missing))}"
            )
        calls = sum(call.get("tool") == metadata.name for call in run.tool_calls)
        if calls >= metadata.max_calls:
            return PolicyDecision(
                PolicyOutcome.DENY, f"Tool call limit reached ({calls}/{metadata.max_calls})"
            )
        try:
            run.budget.check("tool")
            if metadata.external:
                run.budget.check("web")
        except BudgetExceededError as exc:
            return PolicyDecision(PolicyOutcome.DENY, str(exc))
        if metadata.requires_approval and not approved:
            return PolicyDecision(
                PolicyOutcome.REQUIRES_APPROVAL,
                f"{metadata.risk_level.value}-risk tool requires explicit approval",
            )
        return PolicyDecision(PolicyOutcome.ALLOW, "Policy checks passed")
