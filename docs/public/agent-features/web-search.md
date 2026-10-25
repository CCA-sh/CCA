---
title: Web search
sidebar:
  order: 9
---

The agent can search the web and read pages out of the box — the default DuckDuckGo backend needs no API key and no setup. For higher-quality results, plug in Brave or Serper.

```bash
cca config set search_provider brave
cca config set search_api_key YOUR_BRAVE_KEY
```

## Backends

| Provider | Key needed | Notes |
|---|---|---|
| `duckduckgo` | No | Default — works immediately |
| `brave` | Yes | Brave Search API |
| `serper` | Yes | Google-backed results via serper.dev |

Both key-based providers read the same `search_api_key` config value; set `search_provider` to pick which one is active. If a key-based provider is selected without a key, searches fall back to DuckDuckGo.

## The tools

Two tools are involved, both in the always-allowed set (no permission prompt):

- **WebSearch** — runs a query against the configured backend and returns results. The model can constrain results with allowed/blocked domain lists.
- **WebFetch** — fetches a URL and converts the HTML to markdown for reading. HTTP URLs are upgraded to HTTPS automatically, and responses are cached for 15 minutes, so repeated reads of the same page within a session are free.

## Headless note

These built-in tools work in every [run mode](run-modes.md), including `--print`. (Search tools provided by MCP servers, by contrast, are only available in interactive and machine mode.)
