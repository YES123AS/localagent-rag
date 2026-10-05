"""Central execution limits for every V3 agent loop."""

import os


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


MAX_PLAN_STEPS = _bounded_int("MAX_PLAN_STEPS", 5, 1, 10)
MAX_TOOL_CALLS = _bounded_int("MAX_TOOL_CALLS", 6, 1, 20)
MAX_REPLAN_COUNT = _bounded_int("MAX_REPLAN_COUNT", 2, 0, 5)
MAX_WEB_SEARCH_CALLS = _bounded_int("MAX_WEB_SEARCH_CALLS", 3, 1, 10)
