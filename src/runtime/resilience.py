"""Bounded retry/backoff and a small thread-safe rate limiter."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.tools.base import RetryPolicy

from .errors import AuthenticationError, RateLimitError, ToolTimeoutError, ToolUnavailableError


T = TypeVar("T")
RETRYABLE_ERRORS = (ToolTimeoutError, RateLimitError, ToolUnavailableError, TimeoutError, ConnectionError)


def retry_with_backoff(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Any] = time.sleep,
    on_retry: Callable[[BaseException, int, float], Any] | None = None,
) -> T:
    """Run one operation plus at most ``max_retries`` bounded retries."""
    attempt = 0
    while True:
        try:
            return operation()
        except AuthenticationError:
            raise
        except RETRYABLE_ERRORS as exc:
            if attempt >= policy.max_retries:
                raise
            delay = min(
                policy.initial_delay_seconds * (policy.multiplier ** attempt),
                policy.max_delay_seconds,
            )
            attempt += 1
            if on_retry:
                on_retry(exc, attempt, delay)
            sleep(delay)


class RateLimiter:
    """Process-local minimum-interval limiter suitable for a single-instance app."""

    def __init__(self, calls_per_second: float):
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")
        self._interval = 1.0 / calls_per_second
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self, sleep: Callable[[float], Any] = time.sleep) -> float:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                sleep(delay)
            self._next_allowed = max(now, self._next_allowed) + self._interval
            return delay
