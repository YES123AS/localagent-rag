"""Transport-agnostic MCP tool adapter with timeout and response validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from src.runtime.errors import InvalidToolResponseError, ToolTimeoutError, ToolUnavailableError
from src.tools.base import RegisteredTool, ToolMetadata


class MCPToolAdapter:
    def __init__(
        self,
        metadata: ToolMetadata,
        caller: Callable[[str, dict[str, Any]], Any],
    ):
        self.metadata = metadata
        self._caller = caller

    def invoke(self, arguments: dict[str, Any]) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._caller, self.metadata.name, dict(arguments))
            result = future.result(timeout=self.metadata.timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise ToolTimeoutError(f"MCP tool {self.metadata.name} timed out") from exc
        except (ConnectionError, OSError) as exc:
            raise ToolUnavailableError(f"MCP server unavailable: {exc}") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if result is None:
            raise InvalidToolResponseError(f"MCP tool {self.metadata.name} returned no response")
        if isinstance(result, dict) and result.get("isError"):
            raise ToolUnavailableError(str(result.get("error") or result.get("content") or "MCP tool failed"))
        if not isinstance(result, (dict, list, str, int, float, bool)):
            raise InvalidToolResponseError(
                f"MCP tool {self.metadata.name} returned unsupported type {type(result).__name__}"
            )
        return result

    def registered_tool(self) -> RegisteredTool:
        return RegisteredTool(self.metadata, self.invoke)
