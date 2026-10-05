"""One registry for native and MCP tools."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .base import RegisteredTool, ToolMetadata


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        metadata: ToolMetadata,
        handler: Callable[[dict[str, Any]], Any],
        *,
        replace: bool = False,
    ) -> RegisteredTool:
        if not metadata.name.strip():
            raise ValueError("tool name cannot be empty")
        if metadata.name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {metadata.name}")
        tool = RegisteredTool(metadata=metadata, handler=handler)
        self._tools[metadata.name] = tool
        return tool

    def register_tool(self, tool: RegisteredTool, *, replace: bool = False) -> None:
        self.register(tool.metadata, tool.handler, replace=replace)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list_metadata(self) -> list[ToolMetadata]:
        return [tool.metadata for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def extend(self, tools: Iterable[RegisteredTool]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def scoped(self, allowed_names: Iterable[str]) -> "ToolRegistry":
        """Return an allow-listed view copied from this registry."""
        allowed = set(allowed_names)
        scoped = ToolRegistry()
        for name, tool in self._tools.items():
            if name in allowed:
                scoped.register_tool(tool)
        return scoped
