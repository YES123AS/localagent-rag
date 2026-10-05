"""Explicit error taxonomy used by the V4 runtime and tool adapters."""


class AgentError(RuntimeError):
    """Base class for errors with a known runtime recovery policy."""


class PlanningError(AgentError):
    pass


class ToolError(AgentError):
    pass


class ToolTimeoutError(ToolError):
    pass


class RateLimitError(ToolError):
    pass


class AuthenticationError(ToolError):
    pass


class InvalidToolResponseError(ToolError):
    pass


class ToolUnavailableError(ToolError):
    pass


class RetrievalError(AgentError):
    pass


class ModelError(AgentError):
    pass


class PersistenceError(AgentError):
    pass


class ApprovalError(AgentError):
    pass


class BudgetExceededError(AgentError):
    pass


class IdempotencyConflictError(AgentError):
    pass


class WorkerCrashError(BaseException):
    """Testable process-crash boundary; intentionally bypasses ``except Exception``."""
