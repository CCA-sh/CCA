---
title: MCP servers
sidebar:
  order: 8
---

MCP (Model Context Protocol) lets the agent use tools and resources served by external processes — a database inspector, a browser driver, your company's internal API — without any changes to cca itself. Drop a `.mcp.json` in your project, and every tool the server exposes becomes callable by the agent.

The 80% case — a stdio server:

```json
// .mcp.json (project root)
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
    }
  }
}
```

Start cca in (or under) that directory and you'll see:

```
MCP: connected 1 server(s), 26 tool(s) registered.
```

`/mcp` in the REPL shows what's connected.

## Setup

cca looks for `.mcp.json` by walking up from the working directory and uses the first one it finds. The file format is compatible with Claude Code's project-scoped `.mcp.json`.

## Trusting servers

A server is only launched after you trust it once. The first interactive session that sees a new server shows a prompt with its command (or URL) and source file — answer yes and the decision is remembered in `~/.cca/mcp-trust.json`. Trust is tied to the server's exact configuration: editing its `command`, `args`, `env`, or `url` re-prompts, and trust in one project's `.mcp.json` never carries over to another.

Headless runs (`--print`, `--machine`) can't prompt, so untrusted servers are *skipped* with a warning. Either run interactively once to approve them, pre-approve with `/mcp trust`, or pass `--auto-approve` (which already means full trust and connects everything for that run).

> **Security note:** a trusted stdio server is an external process launched with your environment. Treat an unfamiliar `.mcp.json` like an unfamiliar shell script and decline the prompt unless you know what the command does.

## Transports

The `type` field selects the transport (default `stdio`):

| Transport | Runs | Config fields |
|---|---|---|
| `stdio` | A local subprocess | `command` (required), `args`, `env`, `cwd` |
| `sse` | A remote server over Server-Sent Events | `url` (required), `headers` |
| `http` (or `streamable_http`) | A remote server over streamable HTTP | `url` (required), `headers` |

A remote example:

```json
{
  "mcpServers": {
    "internal-api": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": "Bearer YOUR_TOKEN"}
    }
  }
}
```

For remote servers, credentials can live in `headers`, or be stored per server with `McpAuthTool` — a stored token is sent as an `Authorization: Bearer <token>` header on connect whenever the config doesn't set an explicit `Authorization` header (explicit config always wins). For stdio servers, put credentials in `env`.

A server that fails to connect is skipped with a warning — it never blocks the session.

## How server tools appear

Each tool a server advertises is registered as `mcp__<server>__<tool>` — e.g. `mcp__github__create_issue`. These go through the normal [permission flow](permissions-and-tools.md): the agent's first call prompts you, and you can answer once, always-allow, or never-allow per tool.

Three static tools are always registered alongside them:

| Tool | Purpose |
|---|---|
| `ListMcpResourcesTool` | List resources (files, documents, data) exposed by connected servers |
| `ReadMcpResourceTool` | Read one resource by server name + URI |
| `McpAuthTool` | Store per-server auth tokens in `~/.cca/mcp-tokens.json`; injected as Bearer headers on remote connects. `get` reports whether a token exists — it never echoes the secret |

## The `/mcp` command

| Command | Does |
|---|---|
| `/mcp` or `/mcp status` | Show connected servers and registered tool count |
| `/mcp list [server]` | List registered MCP tools, optionally for one server |
| `/mcp resources [server]` | List resources exposed by servers |
| `/mcp trust [server]` | Trust one server (or all) from the current `.mcp.json` without connecting |
| `/mcp untrust <server>` | Revoke trust for a server |
| `/mcp restart` | Disconnect, re-read `.mcp.json`, reconnect, re-register tools |

`/mcp restart` is how you pick up edits to `.mcp.json` without restarting the session.

## Run-mode parity

MCP starts identically in all three [run modes](run-modes.md): the interactive REPL, `--print` one-shot mode, and `--machine` JSON-lines mode. Servers are connected before the first turn and disconnected when the session ends.

## Disabling MCP

Turn the whole subsystem off via [feature toggles](../configuration/feature-toggles.md):

```yaml
# ~/.cca/config.yaml
features:
  mcp: false
```

The `deterministic` preset also disables it.
