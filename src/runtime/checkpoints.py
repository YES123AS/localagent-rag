"""Small functional API for callers that do not need the full RunStore surface."""

from typing import Any

from .models import AgentRun
from .runs import RunStore


def save_checkpoint(store: RunStore, run: AgentRun, state: dict[str, Any] | None = None) -> dict[str, Any]:
    return store.save_checkpoint(run, state)


def load_checkpoint(store: RunStore, run_id: str) -> dict[str, Any]:
    return store.load_checkpoint(run_id)
