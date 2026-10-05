"""Fail-open OpenTelemetry events for reconstructing an Agent Run."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace


_TRACER = trace.get_tracer("localrag.runtime")


def _attribute(value: Any) -> str | int | float | bool:
    return value if isinstance(value, (str, int, float, bool)) else str(value)


@contextmanager
def runtime_span(name: str, **attributes: Any) -> Iterator[Any]:
    try:
        manager = _TRACER.start_as_current_span(name)
    except Exception:
        yield None
        return
    with manager as span:
        try:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, _attribute(value))
        except Exception:
            pass
        yield span


def record_runtime_event(name: str, **attributes: Any) -> None:
    with runtime_span(name, **attributes):
        pass
