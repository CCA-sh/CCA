<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cca-logo-dark.svg">
    <img alt="CCA" src="docs/assets/cca-logo-light.svg" width="360">
  </picture>
</p>

<h3 align="center">The Code CLI Agent You Own</h3>

<div align="center">

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
[![PyPI version](https://img.shields.io/pypi/v/cca-cli?label=version&color=orange)](https://pypi.org/project/cca-cli/)
![Tools](https://img.shields.io/badge/built--in%20tools-33-purple)
![Last commit](https://img.shields.io/github/last-commit/CCA-sh/CCA)
![Stars](https://img.shields.io/github/stars/CCA-sh/CCA?style=social)

**Good enough to be your default agent, simple enough to be your research substrate.**

**Python Native / 33 available built-in tools / Every Trace Can Be Your Data**

[**Website**](https://cca-sh.github.io/CCA/) &nbsp;·&nbsp;
[**Docs**](https://cca-sh.github.io/CCA/docs/configuration/)

</div>

## Installation

Download like another Python package

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the `cca` command (Python import package: `cca`; PyPI distribution: `cca-cli`).

## Quick Start

Set a model and API key once:

```bash
cca config set model anthropic:claude-sonnet-4-6
cca config set-key anthropic sk-ant-...
```

persists to `~/.cca/config.yaml`. Env vars like `ANTHROPIC_API_KEY` work too.
See [Configuration](docs/public/configuration/index.md) for every configurable option.

## Run The Agent

See [Run modes](docs/public/agent-features/run-modes.md) for details on each mode.

### (Interactive Mode)

```bash
cca                                        # interactive REPL
cca --version                              # show version and exit
cca --model google:gemini-3.5-flash        # one-session model override
cca --auto-approve                         # skip all tool permission prompts
cca --cwd /path/to/project                 # start in a specific directory
cca --trace                                # show detailed LLM request/response
cca --verbose                              # show more detailed tool output
```

The interactive startup banner shows the running CCA version. Inside the REPL,
`/help` lists all commands and `/doctor` checks your setup.

Browse bundled model metadata without making an API call:

```bash
cca models
cca models anthropic
cca models qwen
cca models anthropic:claude-opus-4-8
```

To use a [LiteLLM](https://docs.litellm.ai/docs/simple_proxy) proxy, point CCA at your proxy and pass a route name it serves:

```bash
export LITE_LLM_KEY=sk-...
export LITE_LLM_URL=https://your-litellm-proxy.example.com/v1
cca --model litellm:openai/gpt-5.6-sol
```

### (Print Mode)

```bash
cca --print "summarize this repo"          # non-interactive, fully automatic
```

The fully automatic headless mode. Good for benchmarking

### (Machine Mode)

```bash
echo '{"type": "user_turn", "text": "summarize this repo"}' | cca --machine
```

for driving cca from another program — orchestrators, interactive benchmarks, RL loops

## **Access Raw Code Session**

Every session is automatically saved as a flat JSON trajectory — every prompt, LLM call, and tool result — ready to resume, replay, or export as your own data.

```bash
cca --continue                 # resume the most recent session for this directory
cca --resume <id>              # resume a specific session by ID
cca sessions list              # list saved sessions (most recent first)
```

- **Session transcripts:** `~/.cca/sessions/<session_id>.json`
- **Raw LLM I/O logs:** `~/.cca/llm-calls/session_<id>.jsonl`
- [Sessions & Traces](docs/public/agent-features/sessions-and-traces.md)

---

## Detailed Configuration

- [Models](docs/public/configuration/models-and-providers.md) — browse bundled metadata or bring your own API/local model
- [Thinking level](docs/public/configuration/thinking-mode.md) — adjust thinking level if supported
- [Configuration](docs/public/configuration/index.md) — configure everything to make your scaffold deterministic at runtime
- [Feature toggles & presets](docs/public/configuration/feature-toggles.md) — turn off features you don't need; `--preset deterministic` for reproducible traces
- [Randomization](docs/public/configuration/randomization.md) — `--randomize` varies the prompt, tool descriptions, and tool order per run so collected traces don't overfit to one scaffold

## Agent Features

- [Subagents](docs/public/agent-features/subagents.md) — native global/project profiles, depth-one delegation, and separated training traces
- [Run modes](docs/public/agent-features/run-modes.md) — interactive REPL, headless print mode, and the machine-mode JSON protocol
- [MCP servers](docs/public/agent-features/mcp.md) — connect external tools and resources via the Model Context Protocol
- [REPL commands](docs/public/agent-features/repl-commands.md) — all supported slash commands
- [Key bindings](docs/public/agent-features/key-bindings.md) — key bindings supported in interactive mode
- [Permissions & tools](docs/public/agent-features/permissions-and-tools.md) — permission controls and the built-in toolset
- [Hooks](docs/public/agent-features/hooks.md) — lifecycle hooks, exit codes, JSON control
- [Compaction](docs/public/agent-features/compaction.md) — automatic context compaction, `/compact`
- [Memory](docs/public/agent-features/memory.md) — the agent remembers preferences and project context across sessions
- [Skills](docs/public/agent-features/skills.md) — install reusable prompt packages from GitHub with `cca skills add owner/repo@name`
- [Web search](docs/public/agent-features/web-search.md) — DuckDuckGo, Brave, and Serper backends
- [Sessions & Traces](docs/public/agent-features/sessions-and-traces.md) — every session and LLM call saved as plain JSON, ready to export
- [Observability](docs/public/agent-features/observability.md) — optional live lifecycle traces to W&B Weave or any OTLP backend

## Tutorials

- [Run your first task](docs/public/tutorials/first-run.md) — install CCA and choose between interactive, print, and machine modes

## Requirements

- Python 3.11+
- `rg` (ripgrep) recommended for Grep — falls back to Python if absent

## Disclaimer

CCA is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Anthropic, OpenAI, Google, Scale AI, or any other model provider. Product and model names are trademarks of their respective owners.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## Citation

If you use CCA in your research, please cite:

```bibtex
@misc{cca2026,
  title         = {},
  author        = {},
  year          = {2026},
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://github.com/CCA-sh/CCA}
}
```
