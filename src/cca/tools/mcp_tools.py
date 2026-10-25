"""MCP (Model Context Protocol) tool support.

Reads server configs from .mcp.json (CC-compatible) or config.mcp_servers,
connects at session start, and dynamically registers mcp__<server>__<tool>
wrappers.  Also provides ListMcpResourcesTool, ReadMcpResourceTool, and
McpAuthTool.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from cca.tools.base import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _locate_mcp_config(cwd: str = "") -> Path | None:
    """Path of the nearest .mcp.json, walking up from cwd."""
    search_path = Path(cwd).resolve() if cwd else Path.cwd()
    while True:
        candidate = search_path / ".mcp.json"
        if candidate.exists():
            return candidate
        parent = search_path.parent
        if parent == search_path:
            return None
        search_path = parent


def _read_mcp_config(path: Path) -> dict[str, dict]:
    try:
        with open(path) as f:
            return json.load(f).get("mcpServers", {})
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def _load_mcp_config(cwd: str = "") -> dict[str, dict]:
    """Return {server_name: server_config} from the nearest .mcp.json."""
    path = _locate_mcp_config(cwd)
    return _read_mcp_config(path) if path else {}


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _resolve_headers(server_name: str, cfg: dict) -> dict | None:
    """Config headers plus a stored Bearer token; explicit Authorization wins."""
    headers = dict(cfg.get("headers") or {})
    if not any(k.lower() == "authorization" for k in headers):
        token = _load_tokens().get(server_name)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers or None

async def _connect_stdio(name: str, cfg: dict):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from contextlib import AsyncExitStack

    command = cfg.get("command", "")
    if not command:
        logger.warning("MCP server '%s' has no command", name)
        return None

    params = StdioServerParameters(
        command=command,
        args=cfg.get("args", []),
        env={**os.environ, **cfg.get("env", {})},
        cwd=cfg.get("cwd") or None,
    )
    try:
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        session._exit_stack = stack  # type: ignore[attr-defined]
        return session
    except Exception as e:
        logger.warning("Failed to connect MCP server '%s': %s", name, e)
        return None


async def _connect_sse(name: str, cfg: dict):
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from contextlib import AsyncExitStack

    url = cfg.get("url", "")
    if not url:
        logger.warning("MCP server '%s' has no url", name)
        return None

    try:
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(
            sse_client(url, headers=_resolve_headers(name, cfg))
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        session._exit_stack = stack  # type: ignore[attr-defined]
        return session
    except Exception as e:
        logger.warning("Failed to connect MCP server '%s': %s", name, e)
        return None


async def _connect_http(name: str, cfg: dict):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from contextlib import AsyncExitStack

    url = cfg.get("url", "")
    if not url:
        logger.warning("MCP server '%s' has no url", name)
        return None

    try:
        stack = AsyncExitStack()
        read, write, _get_session_id = await stack.enter_async_context(
            streamablehttp_client(url, headers=_resolve_headers(name, cfg))
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        session._exit_stack = stack  # type: ignore[attr-defined]
        return session
    except Exception as e:
        logger.warning("Failed to connect MCP server '%s': %s", name, e)
        return None


async def _connect(name: str, cfg: dict):
    transport = cfg.get("type", "stdio").lower()
    if transport == "stdio":
        return await _connect_stdio(name, cfg)
    elif transport == "sse":
        return await _connect_sse(name, cfg)
    elif transport in ("http", "streamable_http"):
        return await _connect_http(name, cfg)
    else:
        logger.warning("Unknown MCP transport '%s' for server '%s'", transport, name)
        return None


# ---------------------------------------------------------------------------
# McpServerManager
# ---------------------------------------------------------------------------

class McpServerManager:
    """Manages connections and tool listings for all configured MCP servers."""

    def __init__(self, cwd: str = "", trust_fn=None, trust_all: bool = False) -> None:
        self._cwd = cwd
        self._sessions: dict[str, Any] = {}
        # trust_fn: async (server_name, server_cfg, config_path) -> bool.
        # None means non-interactive: untrusted servers are skipped.
        self.trust_fn = trust_fn
        self.trust_all = trust_all
        self.skipped_servers: list[str] = []

    async def _server_trusted(self, config_path: Path, name: str, cfg: dict) -> bool:
        from cca.tools import mcp_trust
        if self.trust_all or mcp_trust.is_trusted(str(config_path), name, cfg):
            return True
        if self.trust_fn is None:
            return False
        if await self.trust_fn(name, cfg, str(config_path)):
            mcp_trust.record_trust(str(config_path), name, cfg)
            return True
        return False

    async def start(self) -> list["DynamicMcpTool"]:
        """Connect to all configured, trusted servers; return tool wrappers."""
        self.skipped_servers = []
        config_path = _locate_mcp_config(self._cwd)
        if config_path is None:
            return []
        cfg = _read_mcp_config(config_path)
        for name, server_cfg in cfg.items():
            if not await self._server_trusted(config_path, name, server_cfg):
                logger.warning("MCP server '%s' is not trusted — skipped", name)
                self.skipped_servers.append(name)
                continue
            session = await _connect(name, server_cfg)
            if session is not None:
                self._sessions[name] = session
        return await self._build_tools()

    async def _build_tools(self) -> list["DynamicMcpTool"]:
        wrappers: list[DynamicMcpTool] = []
        for server_name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                for mcp_tool in result.tools:
                    wrappers.append(DynamicMcpTool(
                        server_name=server_name,
                        mcp_tool=mcp_tool,
                        manager=self,
                    ))
            except Exception as e:
                logger.warning("Failed to list tools for server '%s': %s", server_name, e)
        return wrappers

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        session = self._sessions.get(server_name)
        if session is None:
            return f"Error: MCP server '{server_name}' is not connected."
        try:
            result = await session.call_tool(tool_name, arguments)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                elif hasattr(content, "data"):
                    parts.append(f"[binary data: {len(content.data)} bytes]")
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(empty result)"
        except Exception as e:
            return f"Error calling tool '{tool_name}' on server '{server_name}': {e}"

    async def list_resources(self, server_name: str | None = None) -> str:
        servers = [server_name] if server_name else list(self._sessions.keys())
        if not servers:
            return "No MCP servers connected."
        lines: list[str] = []
        for sname in servers:
            session = self._sessions.get(sname)
            if session is None:
                lines.append(f"Server '{sname}': not connected")
                continue
            try:
                result = await session.list_resources()
                lines.append(f"Server '{sname}' resources ({len(result.resources)}):")
                for r in result.resources:
                    desc = f" — {r.description}" if r.description else ""
                    lines.append(f"  {r.uri}  [{r.name}]{desc}")
            except Exception as e:
                lines.append(f"Server '{sname}': error listing resources — {e}")
        return "\n".join(lines)

    async def read_resource(self, server_name: str, uri: str) -> str:
        session = self._sessions.get(server_name)
        if session is None:
            return f"Error: MCP server '{server_name}' is not connected."
        try:
            result = await session.read_resource(uri)
            parts = []
            for content in result.contents:
                if hasattr(content, "text"):
                    parts.append(content.text)
                elif hasattr(content, "blob"):
                    parts.append(f"[binary data: {len(content.blob)} bytes]")
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(empty resource)"
        except Exception as e:
            return f"Error reading resource '{uri}' from server '{server_name}': {e}"

    def server_names(self) -> list[str]:
        return list(self._sessions.keys())

    def is_connected(self, server_name: str) -> bool:
        return server_name in self._sessions

    async def stop(self) -> None:
        for name, session in list(self._sessions.items()):
            try:
                stack = getattr(session, "_exit_stack", None)
                if stack is not None:
                    await stack.aclose()
            except Exception as e:
                logger.debug("Error closing MCP server '%s': %s", name, e)
        self._sessions.clear()


# ---------------------------------------------------------------------------
# Dynamic tool wrapper
# ---------------------------------------------------------------------------

class DynamicMcpTool(Tool):
    """Tool wrapper dynamically created from an MCP server's tool listing."""

    def __init__(self, server_name: str, mcp_tool: Any, manager: McpServerManager) -> None:
        self._server_name = server_name
        self._mcp_tool = mcp_tool
        self._manager = manager

    @property
    def name(self) -> str:
        return f"mcp__{self._server_name}__{self._mcp_tool.name}"

    @property
    def description(self) -> str:
        desc = self._mcp_tool.description or ""
        return f"[MCP:{self._server_name}] {desc}"

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = self._mcp_tool.inputSchema or {}
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    async def execute(self, **kwargs: Any) -> str:
        return await self._manager.call_tool(self._server_name, self._mcp_tool.name, kwargs)


# ---------------------------------------------------------------------------
# ListMcpResourcesTool
# ---------------------------------------------------------------------------

class ListMcpResourcesTool(Tool):
    """List available resources on connected MCP servers."""

    @property
    def name(self) -> str:
        return "ListMcpResourcesTool"

    @property
    def description(self) -> str:
        return (
            "List resources available on MCP servers. "
            "Optionally filter to a specific server by name."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server to query. Omit to query all servers.",
                },
            },
            "required": [],
        }

    def __init__(self, manager: McpServerManager) -> None:
        self._manager = manager

    async def execute(self, server_name: str = "", **kwargs: Any) -> str:
        return await self._manager.list_resources(server_name or None)


# ---------------------------------------------------------------------------
# ReadMcpResourceTool
# ---------------------------------------------------------------------------

class ReadMcpResourceTool(Tool):
    """Read a resource from an MCP server by URI."""

    @property
    def name(self) -> str:
        return "ReadMcpResourceTool"

    @property
    def description(self) -> str:
        return "Read a resource from an MCP server by its URI."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server that owns the resource.",
                },
                "uri": {
                    "type": "string",
                    "description": "URI of the resource to read (e.g. file:///path or custom://...).",
                },
            },
            "required": ["server_name", "uri"],
        }

    def __init__(self, manager: McpServerManager) -> None:
        self._manager = manager

    async def execute(self, server_name: str, uri: str, **kwargs: Any) -> str:
        return await self._manager.read_resource(server_name, uri)


# ---------------------------------------------------------------------------
# McpAuthTool
# ---------------------------------------------------------------------------

_TOKEN_FILE = Path.home() / ".cca" / "mcp-tokens.json"


def _load_tokens() -> dict[str, str]:
    if _TOKEN_FILE.exists():
        try:
            with open(_TOKEN_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_tokens(tokens: dict[str, str]) -> None:
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    _TOKEN_FILE.chmod(0o600)


class McpAuthTool(Tool):
    """Store, check, or delete authentication tokens for MCP servers.

    Tokens are saved in ~/.cca/mcp-tokens.json (mode 0600) and injected as
    Bearer Authorization headers when connecting to sse/http servers, unless
    the server config sets an explicit Authorization header.
    """

    @property
    def name(self) -> str:
        return "McpAuthTool"

    @property
    def description(self) -> str:
        return (
            "Manage authentication tokens for MCP servers. Stored tokens are "
            "sent as Bearer Authorization headers when connecting to remote "
            "servers. Use action='set' to store a token, 'get' to check "
            "whether one is stored (the token itself is never echoed), "
            "'delete' to remove it, or 'list' to show all stored server names."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "get", "delete", "list"],
                    "description": "The action to perform.",
                },
                "server_name": {
                    "type": "string",
                    "description": "MCP server name (required for set/get/delete).",
                },
                "token": {
                    "type": "string",
                    "description": "The authentication token (required for action='set').",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, server_name: str = "", token: str = "", **kwargs: Any) -> str:
        tokens = _load_tokens()

        if action == "list":
            if not tokens:
                return "No MCP auth tokens stored."
            return "Stored tokens for: " + ", ".join(sorted(tokens.keys()))

        if action == "set":
            if not server_name:
                return "Error: server_name is required for action='set'."
            if not token:
                return "Error: token is required for action='set'."
            tokens[server_name] = token
            _save_tokens(tokens)
            return f"Token stored for server '{server_name}'."

        if action == "get":
            if not server_name:
                return "Error: server_name is required for action='get'."
            t = tokens.get(server_name)
            if t is None:
                return f"No token stored for server '{server_name}'."
            return (
                f"Token stored for server '{server_name}' ({len(t)} chars). "
                "It is sent as a Bearer Authorization header when connecting; "
                "it is never echoed."
            )

        if action == "delete":
            if not server_name:
                return "Error: server_name is required for action='delete'."
            if server_name in tokens:
                del tokens[server_name]
                _save_tokens(tokens)
                return f"Token deleted for server '{server_name}'."
            return f"No token stored for server '{server_name}'."

        return f"Unknown action '{action}'."
