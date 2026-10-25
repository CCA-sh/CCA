---
title: Key bindings
sidebar:
  order: 11
---

The interactive REPL uses familiar terminal editing keys. The ones worth knowing on day one: `Enter` submits, `Ctrl+J` inserts a newline, `Escape` cancels a running turn, and `Ctrl+C` twice quits.

## Input

| Key | Action |
|---|---|
| `Enter` | Submit the message (or accept the highlighted palette command) |
| `Ctrl+J` | Insert a newline |
| `Esc` then `Enter` | Insert a newline (alternative; not in vim mode) |
| `↑` / `↓` | Navigate input history (or move within a multi-line message) |
| `Option+←` / `Option+→` | Jump by word (macOS; sent as `Esc b` / `Esc f`) |
| `Option+Backspace` | Delete the previous word |
| `Option+D` | Delete the next word |

## Control

| Key | Action |
|---|---|
| `Escape` | Cancel the running turn; close the command palette; deny an open permission prompt |
| `Ctrl+C` | Clear the input / cancel the running turn — a second press within 2 seconds quits |
| `Ctrl+D` | Quit (when the input is empty) |

## Command palette

Typing `/` opens an autocomplete palette of [slash commands](repl-commands.md). `↑`/`↓` move the selection, `Enter` accepts, `Escape` closes it.

## Permission prompts

When a tool asks for permission, answer with a single key: `y` allow once, `n` deny, `a` always allow this tool, `!` never allow it. `Escape` denies and cancels the turn. Details in [Permissions & tools](permissions-and-tools.md).

## Vim mode

`cca config set vim true` (or `/vim` to toggle in-session) switches the input line to vi keybindings with normal/insert modes. In vim mode, `Escape` first drops to normal mode; pressing it in normal mode cancels the running turn.
