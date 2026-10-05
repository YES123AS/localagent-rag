"""Minimal MCP JSON-RPC stdio transport with bounded calls and discovery."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Sequence

from src.runtime.errors import (
    InvalidToolResponseError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from src.tools.base import RiskLevel, RetryPolicy, ToolMetadata
from src.tools.mcp.adapter import MCPToolAdapter
from src.tools.registry import ToolRegistry


class StdioMCPTransport:
    """One persistent stdio MCP session using newline-delimited JSON-RPC."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 15.0,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        protocol_version: str = "2024-11-05",
    ):
        if not command:
            raise ValueError("MCP stdio command cannot be empty")
        self.command = [str(part) for part in command]
        self.timeout = max(0.01, float(timeout))
        self.cwd = str(cwd) if cwd else None
        self.env = {**os.environ, **(env or {})}
        self.protocol_version = protocol_version
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.RLock()
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-stdio-reader")

    @property
    def running(self) -> bool:
        return bool(self._process and self._process.poll() is None)

    def start(self) -> None:
        if self.running:
            return
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self.cwd,
                env=self.env,
            )
        except (OSError, ValueError) as exc:
            raise ToolUnavailableError(f"Could not start MCP stdio server: {exc}") from exc
        try:
            self.request(
                "initialize",
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "LocalAgent", "version": "4.5"},
                },
            )
            self.notify("notifications/initialized", {})
        except BaseException:
            self.close()
            raise

    def _write(self, message: dict[str, Any]) -> None:
        process = self._process
        if not process or process.poll() is not None or not process.stdin:
            raise ToolUnavailableError("MCP stdio server is not running")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ToolUnavailableError(f"MCP stdio pipe is unavailable: {exc}") from exc

    def _readline(self) -> str:
        process = self._process
        if not process or not process.stdout:
            raise ToolUnavailableError("MCP stdio stdout is unavailable")
        line = process.stdout.readline()
        if not line:
            stderr = ""
            if process.stderr:
                try:
                    stderr = process.stderr.read(2000)
                except OSError:
                    pass
            raise ToolUnavailableError(
                f"MCP stdio server closed the stream{': ' + stderr if stderr else ''}"
            )
        return line

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            if not self.running and method != "initialize":
                self.start()
            self._request_id += 1
            request_id = self._request_id
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            while True:
                future = self._reader.submit(self._readline)
                try:
                    line = future.result(timeout=self.timeout)
                except FutureTimeout as exc:
                    future.cancel()
                    self.close()
                    raise ToolTimeoutError(
                        f"MCP stdio request {method} timed out after {self.timeout}s"
                    ) from exc
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.close()
                    raise InvalidToolResponseError(
                        "MCP stdio server returned invalid JSON"
                    ) from exc
                if payload.get("id") != request_id:
                    continue
                if "error" in payload:
                    raise ToolUnavailableError(
                        f"MCP {method} failed: {payload.get('error')}"
                    )
                if "result" not in payload:
                    raise InvalidToolResponseError(
                        f"MCP {method} response has no result"
                    )
                return payload["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._write(
                {"jsonrpc": "2.0", "method": method, "params": params or {}}
            )

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        result = self.request("tools/list")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise InvalidToolResponseError("MCP tools/list returned no tools list")
        return [dict(item) for item in tools if isinstance(item, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        clean_arguments = {
            key: value
            for key, value in arguments.items()
            if not key.startswith("_runtime_")
        }
        result = self.request(
            "tools/call", {"name": name, "arguments": clean_arguments}
        )
        if isinstance(result, dict) and result.get("isError"):
            raise ToolUnavailableError(str(result.get("content") or "MCP tool failed"))
        return result

    def close(self) -> None:
        process, self._process = self._process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()


def _safe_server_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    if not normalized:
        raise ValueError("MCP server name cannot be empty")
    return normalized


def register_stdio_mcp_tools(
    registry: ToolRegistry,
    command: Sequence[str],
    *,
    server_name: str = "stdio",
    timeout: float = 15.0,
    cwd: str | Path | None = None,
    require_approval_for_writes: bool = True,
) -> tuple[StdioMCPTransport, list[str]]:
    """Discover one stdio server and register every advertised tool."""
    server_name = _safe_server_name(server_name)
    transport = StdioMCPTransport(command, timeout=timeout, cwd=cwd)
    discovered = transport.list_tools()
    registered: list[str] = []
    for description in discovered:
        remote_name = str(description.get("name") or "").strip()
        if not remote_name:
            continue
        annotations = dict(description.get("annotations") or {})
        read_only = bool(annotations.get("readOnlyHint", False))
        public_name = f"mcp.{server_name}.{remote_name}"
        metadata = ToolMetadata(
            name=public_name,
            description=str(description.get("description") or remote_name),
            risk_level=RiskLevel.LOW if read_only else RiskLevel.MEDIUM,
            requires_approval=bool(require_approval_for_writes and not read_only),
            timeout=timeout,
            max_calls=3,
            retry_policy=RetryPolicy(
                max_retries=2,
                initial_delay_seconds=1,
                multiplier=2,
                max_delay_seconds=4,
            ),
            external=True,
            side_effecting=not read_only,
        )
        adapter = MCPToolAdapter(
            metadata,
            lambda _public_name, arguments, remote=remote_name: transport.call_tool(
                remote, arguments
            ),
        )
        registry.register_tool(adapter.registered_tool())
        registered.append(public_name)
    return transport, registered
