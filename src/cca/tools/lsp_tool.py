"""LSP tool — code intelligence via local language servers.

Communicates with language servers over stdio JSON-RPC (Language Server Protocol).
Servers are auto-detected from PATH at import time; no user configuration required.
The tool is only registered if at least one server is found.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from cca.tools.base import Tool


# ---------------------------------------------------------------------------
# Known language servers (checked in order; first found wins per extension)
# ---------------------------------------------------------------------------

_KNOWN_SERVERS: list[dict] = [
    {
        "binaries": ["pyright-langserver", "pylsp", "jedi-language-server"],
        "args": {
            "pyright-langserver": ["--stdio"],
            "pylsp": [],
            "jedi-language-server": [],
        },
        "extensions": [".py"],
        "language": "python",
    },
    {
        "binaries": ["typescript-language-server"],
        "args": {"typescript-language-server": ["--stdio"]},
        "extensions": [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"],
        "language": "typescript",
    },
    {
        "binaries": ["rust-analyzer"],
        "args": {"rust-analyzer": []},
        "extensions": [".rs"],
        "language": "rust",
    },
    {
        "binaries": ["gopls"],
        "args": {"gopls": ["serve"]},
        "extensions": [".go"],
        "language": "go",
    },
    {
        "binaries": ["clangd"],
        "args": {"clangd": []},
        "extensions": [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
        "language": "cpp",
    },
    {
        "binaries": ["solargraph"],
        "args": {"solargraph": ["stdio"]},
        "extensions": [".rb"],
        "language": "ruby",
    },
    {
        "binaries": ["zls"],
        "args": {"zls": []},
        "extensions": [".zig"],
        "language": "zig",
    },
]


def _detect_servers() -> list[dict]:
    """Return configs for every language server found in PATH."""
    detected = []
    for spec in _KNOWN_SERVERS:
        for binary in spec["binaries"]:
            if shutil.which(binary):
                detected.append({
                    "command": [binary] + spec["args"][binary],
                    "extensions": spec["extensions"],
                    "language": spec["language"],
                    "binary": binary,
                })
                break  # first found binary wins for this language
    return detected


# Detected at import time — one dict per language server found
DETECTED_SERVERS: list[dict] = _detect_servers()

# Extension → server config index
_EXT_TO_SERVER: dict[str, int] = {}
for _i, _srv in enumerate(DETECTED_SERVERS):
    for _ext in _srv["extensions"]:
        if _ext not in _EXT_TO_SERVER:
            _EXT_TO_SERVER[_ext] = _i


# ---------------------------------------------------------------------------
# Minimal async JSON-RPC stdio client
# ---------------------------------------------------------------------------

class _JsonRpcClient:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None

    def start(self) -> None:
        self._read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        while True:
            try:
                headers: dict[str, str] = {}
                while True:
                    line = await self._proc.stdout.readline()
                    if not line:
                        return
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        break
                    if ":" in decoded:
                        k, v = decoded.split(":", 1)
                        headers[k.strip()] = v.strip()

                length = int(headers.get("Content-Length", 0))
                if not length:
                    continue

                body = await self._proc.stdout.readexactly(length)
                msg = json.loads(body)

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in msg:
                            err = msg["error"]
                            fut.set_exception(
                                RuntimeError(err.get("message", "LSP error"))
                            )
                        else:
                            fut.set_result(msg.get("result"))
            except asyncio.IncompleteReadError:
                break
            except Exception:
                break

        # Wake up any pending futures so callers don't hang
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("LSP server connection closed"))
        self._pending.clear()

    async def request(self, method: str, params: Any, timeout: float = 15.0) -> Any:
        rid = self._next_id
        self._next_id += 1
        msg = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        encoded = msg.encode()
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode()
        assert self._proc.stdin is not None
        self._proc.stdin.write(header + encoded)
        await self._proc.stdin.drain()

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def notify(self, method: str, params: Any) -> None:
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        encoded = msg.encode()
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode()
        assert self._proc.stdin is not None
        self._proc.stdin.write(header + encoded)
        await self._proc.stdin.drain()

    async def close(self) -> None:
        if self._read_task:
            self._read_task.cancel()
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-language-server instance (lazy start)
# ---------------------------------------------------------------------------

class _LspServerInstance:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._client: _JsonRpcClient | None = None
        self._open_uris: set[str] = set()
        self._lock = asyncio.Lock()

    async def _start(self, cwd: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._config["command"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        client = _JsonRpcClient(proc)
        client.start()

        root_uri = Path(cwd).resolve().as_uri()
        await client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "definition": {"linkSupport": False},
                    "references": {},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                },
                "workspace": {"symbol": {}},
            },
            "workspaceFolders": [
                {"uri": root_uri, "name": Path(cwd).name}
            ],
        })
        await client.notify("initialized", {})
        self._client = client

    async def _ensure_started(self, cwd: str) -> _JsonRpcClient:
        async with self._lock:
            if self._client is None:
                await self._start(cwd)
        assert self._client is not None
        return self._client

    async def _open_file(self, client: _JsonRpcClient, abs_path: str) -> None:
        uri = Path(abs_path).resolve().as_uri()
        if uri in self._open_uris:
            return
        text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        await client.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": self._config["language"],
                "version": 1,
                "text": text,
            }
        })
        self._open_uris.add(uri)

    async def file_request(
        self, method: str, abs_path: str, params: dict, cwd: str
    ) -> Any:
        client = await self._ensure_started(cwd)
        await self._open_file(client, abs_path)
        return await client.request(method, params)

    async def workspace_request(self, method: str, params: dict, cwd: str) -> Any:
        client = await self._ensure_started(cwd)
        return await client.request(method, params)

    async def close(self) -> None:
        if self._client:
            await self._client.close()


# Global server instances (one per detected server)
_SERVER_INSTANCES: list[_LspServerInstance] = [
    _LspServerInstance(cfg) for cfg in DETECTED_SERVERS
]


# ---------------------------------------------------------------------------
# Result formatters
# ---------------------------------------------------------------------------

def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[7:]
    return uri


def _fmt_location(loc: dict) -> str:
    path = _uri_to_path(loc.get("uri", ""))
    r = loc.get("range", {}).get("start", {})
    line = r.get("line", 0) + 1
    char = r.get("character", 0) + 1
    return f"{path}:{line}:{char}"


def _fmt_definitions(result: Any) -> str:
    if not result:
        return "No definition found."
    if isinstance(result, dict):
        result = [result]
    lines = [_fmt_location(loc) for loc in result]
    return "\n".join(lines)


def _fmt_references(result: Any) -> str:
    if not result:
        return "No references found."
    by_file: dict[str, list[str]] = {}
    for loc in result:
        path = _uri_to_path(loc.get("uri", ""))
        r = loc.get("range", {}).get("start", {})
        line = r.get("line", 0) + 1
        by_file.setdefault(path, []).append(str(line))
    parts = []
    for path, lines in sorted(by_file.items()):
        parts.append(f"{path}: lines {', '.join(lines)}")
    return f"{len(result)} reference(s) in {len(by_file)} file(s):\n" + "\n".join(parts)


def _fmt_hover(result: Any) -> str:
    if not result:
        return "No hover information."
    contents = result.get("contents", "")
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value", "")
    if isinstance(contents, list):
        parts = []
        for c in contents:
            parts.append(c.get("value", c) if isinstance(c, dict) else str(c))
        return "\n\n".join(parts)
    return str(contents)


def _fmt_document_symbols(result: Any, indent: int = 0) -> str:
    if not result:
        return "No symbols found."
    lines = []
    prefix = "  " * indent
    for sym in result:
        kind = sym.get("kind", 0)
        name = sym.get("name", "?")
        r = sym.get("range", sym.get("location", {}).get("range", {}))
        start = r.get("start", {})
        line = start.get("line", 0) + 1
        lines.append(f"{prefix}{name}  (kind={kind}, line={line})")
        children = sym.get("children", [])
        if children:
            lines.append(_fmt_document_symbols(children, indent + 1))
    return "\n".join(lines)


def _fmt_workspace_symbols(result: Any) -> str:
    if not result:
        return "No symbols found."
    by_file: dict[str, list[str]] = {}
    for sym in result:
        loc = sym.get("location", {})
        path = _uri_to_path(loc.get("uri", ""))
        r = loc.get("range", {}).get("start", {})
        line = r.get("line", 0) + 1
        name = sym.get("name", "?")
        kind = sym.get("kind", 0)
        by_file.setdefault(path, []).append(f"  {name}  (kind={kind}, line={line})")
    parts = []
    for path, syms in sorted(by_file.items()):
        parts.append(path + "\n" + "\n".join(syms))
    return f"{len(result)} symbol(s):\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

class LspTool(Tool):
    name = "LSP"

    @property
    def description(self) -> str:
        langs = sorted({s["language"] for s in DETECTED_SERVERS})
        return (
            "Query a local language server for precise code intelligence. "
            f"Available for: {', '.join(langs)}. "
            "Operations: goToDefinition, findReferences, hover, "
            "documentSymbol, workspaceSymbol. "
            "line and character are 1-based (as shown in editors)."
        )

    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "goToDefinition",
                    "findReferences",
                    "hover",
                    "documentSymbol",
                    "workspaceSymbol",
                ],
                "description": "The LSP operation to perform.",
            },
            "filePath": {
                "type": "string",
                "description": "Absolute path to the file (required for all ops except workspaceSymbol).",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number (required for goToDefinition, findReferences, hover).",
            },
            "character": {
                "type": "integer",
                "description": "1-based character offset (required for goToDefinition, findReferences, hover).",
            },
            "query": {
                "type": "string",
                "description": "Symbol search query (required for workspaceSymbol).",
            },
            "includeDeclaration": {
                "type": "boolean",
                "description": "For findReferences: include the declaration site (default true).",
            },
        },
        "required": ["operation"],
    }

    def __init__(self, cwd: str = "") -> None:
        self._cwd = cwd or os.getcwd()

    def _server_for_file(self, file_path: str) -> _LspServerInstance | None:
        ext = Path(file_path).suffix.lower()
        idx = _EXT_TO_SERVER.get(ext)
        return _SERVER_INSTANCES[idx] if idx is not None else None

    def _any_server(self) -> _LspServerInstance | None:
        return _SERVER_INSTANCES[0] if _SERVER_INSTANCES else None

    async def execute(
        self,
        operation: str,
        filePath: str = "",
        line: int = 1,
        character: int = 1,
        query: str = "",
        includeDeclaration: bool = True,
    ) -> str:
        cwd = self._cwd

        if operation == "workspaceSymbol":
            srv = self._server_for_file(filePath) if filePath else self._any_server()
            if srv is None:
                return "Error: no language server available."
            try:
                result = await srv.workspace_request(
                    "workspace/symbol", {"query": query or ""}, cwd
                )
                return _fmt_workspace_symbols(result)
            except Exception as e:
                return f"Error: {e}"

        if not filePath:
            return "Error: filePath is required for this operation."

        abs_path = str(Path(filePath).resolve())
        if not Path(abs_path).is_file():
            return f"Error: file not found: {filePath}"

        srv = self._server_for_file(abs_path)
        if srv is None:
            ext = Path(abs_path).suffix
            return f"Error: no language server available for {ext} files."

        # LSP uses 0-based positions; tool accepts 1-based
        lsp_line = max(0, line - 1)
        lsp_char = max(0, character - 1)
        uri = Path(abs_path).as_uri()
        pos = {"line": lsp_line, "character": lsp_char}

        try:
            if operation == "goToDefinition":
                result = await srv.file_request(
                    "textDocument/definition",
                    abs_path,
                    {"textDocument": {"uri": uri}, "position": pos},
                    cwd,
                )
                return _fmt_definitions(result)

            if operation == "findReferences":
                result = await srv.file_request(
                    "textDocument/references",
                    abs_path,
                    {
                        "textDocument": {"uri": uri},
                        "position": pos,
                        "context": {"includeDeclaration": includeDeclaration},
                    },
                    cwd,
                )
                return _fmt_references(result)

            if operation == "hover":
                result = await srv.file_request(
                    "textDocument/hover",
                    abs_path,
                    {"textDocument": {"uri": uri}, "position": pos},
                    cwd,
                )
                return _fmt_hover(result)

            if operation == "documentSymbol":
                result = await srv.file_request(
                    "textDocument/documentSymbol",
                    abs_path,
                    {"textDocument": {"uri": uri}},
                    cwd,
                )
                return _fmt_document_symbols(result)

            return f"Error: unknown operation '{operation}'"

        except asyncio.TimeoutError:
            return f"Error: language server timed out on {operation}."
        except Exception as e:
            return f"Error: {e}"
