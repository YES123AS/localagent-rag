"""A bounded read-only filesystem capability exposed through the MCP adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.runtime.errors import InvalidToolResponseError


class FilesystemMCPClient:
    """Reference filesystem MCP client used for local integration validation.

    It implements the same ``call_tool`` boundary as a remote transport and can
    be replaced by a real MCP session without changing Tool Registry policy.
    """

    def __init__(self, root: str | Path, max_bytes: int = 1_000_000):
        self.root = Path(root).resolve()
        self.max_bytes = max(1, int(max_bytes))
        if not self.root.is_dir():
            raise ValueError(f"MCP filesystem root does not exist: {self.root}")

    def _resolve(self, requested: str) -> Path:
        target = (self.root / requested).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("Path escapes the configured MCP filesystem root") from exc
        return target

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(str(arguments.get("path") or "."))
        if name == "mcp.filesystem.list_directory":
            if not path.is_dir():
                raise InvalidToolResponseError("Requested path is not a directory")
            return {
                "path": str(path.relative_to(self.root)),
                "entries": [
                    {"name": child.name, "type": "directory" if child.is_dir() else "file"}
                    for child in sorted(path.iterdir(), key=lambda item: item.name.casefold())[:500]
                ],
            }
        if name == "mcp.filesystem.read_text":
            if not path.is_file():
                raise InvalidToolResponseError("Requested path is not a file")
            if path.stat().st_size > self.max_bytes:
                raise InvalidToolResponseError("Requested file exceeds the configured size limit")
            return {"path": str(path.relative_to(self.root)), "content": path.read_text(encoding="utf-8")}
        raise InvalidToolResponseError(f"Unsupported filesystem MCP tool: {name}")
