---
title: Run your first task
description: Install CCA, configure a model, and choose the right run mode for your first task.
sidebar:
  order: 1
---
## Install and configure

Create an isolated environment and install the CLI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install cca-cli
```

Choose a model and save its API key:

```bash
cca config set model anthropic:claude-sonnet-4-6
cca config set-key anthropic sk-ant-...
```

## Work interactively

Start CCA in a repository when you want a conversational workflow with
permission prompts and slash commands:

```bash
cca --cwd ./my-project
```

Describe the outcome you want. CCA can inspect the repository, propose a plan,
edit files, run commands, and verify the result.

## Run one headless task

Use print mode from a script, CI job, or benchmark:

```bash
cca --print "summarize this repo"
```

Tasks that edit files or run commands need explicit approval because print mode
cannot show an interactive permission prompt:

```bash
cca --print "fix the failing test" --auto-approve
```

Use `--auto-approve` only in a repository or sandbox where automated changes
are acceptable.

## Drive CCA from another program

Machine mode accepts and emits one JSON object per line:

```bash
echo '{"type":"user_turn","text":"summarize this repo"}' | cca --machine
```

Use machine mode for orchestrators, applications, interactive benchmarks, and
RL loops that need a persistent multi-turn agent process.

See [Run modes](../agent-features/run-modes.md) for the complete behavior and
options of each interface.