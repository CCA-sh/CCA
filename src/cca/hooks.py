"""Hooks system: shell commands that fire at lifecycle events.

Hook events:
  SessionStart   — fires once when cca starts (or after compaction)
  PreToolUse     — fires before each tool call; can block it
  PostToolUse    — fires after each tool call; exit 2 injects stderr into model
  PreCompact     — fires before context compaction
  PostCompact    — fires after context compaction

Config format (in ~/.cca/config.yaml or .cca/config.yaml):

  hooks:
    PreToolUse:
      - matcher: "Bash"          # glob pattern matched against tool_name
        command: "my-logger.sh"
      - matcher: "Edit"
        command: "lint-check.sh"
        timeout: 30
    PostToolUse:
      - command: "audit.sh"
        async: true              # fire-and-forget; does not block
    SessionStart:
      - command: "echo 'cca session started'"
    PreCompact:
      - command: "save-state.sh"
    PostCompact:
      - command: "restore-state.sh"

Hook input/output contract
--------------------------
Every hook receives JSON on stdin describing the event. Example for PreToolUse:

  {
    "session_id": "abc123",
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "ls -la"},
    "tool_use_id": "toolu_01xxx"
  }

Exit code semantics:
  PreToolUse
    exit 0   — allow (stdout ignored unless async-rewake)
    exit 2   — block the tool; stderr shown to model as reason
    other    — allow; stderr shown to user only

  PostToolUse
    exit 0   — success; stdout shown in trace mode
    exit 2   — stderr appended to tool result seen by model
    other    — stderr shown to user only

  SessionStart / PreCompact / PostCompact
    any exit — logged only, never blocks

Stdout JSON override (any event):
  {
    "continue": false,             # block tool / stop session
    "stopReason": "...",           # message shown when blocked
    "decision": "block",           # alternative to continue:false
    "reason": "...",               # alternative to stopReason
    "additionalContext": "..."     # prepended to model context
  }
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import subprocess
from dataclasses import dataclass

TOOL_HOOK_TIMEOUT = 10.0   # default seconds per hook


@dataclass
class HookResult:
    allowed: bool = True           # False → block the tool call
    additional_context: str = ""   # injected as <system-reminder> to model
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class HookRunner:
    """Executes configured hooks for lifecycle events."""

    def __init__(self, hooks_config: dict[str, list[dict]], session_id: str = "", cwd: str = "") -> None:
        # hooks_config: event_name → list of hook defs
        self._hooks: dict[str, list[dict]] = hooks_config or {}
        self._session_id = session_id
        self._cwd = cwd

    def has_hooks(self, event: str) -> bool:
        return bool(self._hooks.get(event))

    def _matching(self, event: str, tool_name: str | None = None) -> list[dict]:
        hooks = self._hooks.get(event, [])
        if tool_name is None:
            return hooks
        result = []
        for h in hooks:
            matcher = h.get("matcher", "*")
            if fnmatch.fnmatch(tool_name, matcher):
                result.append(h)
        return result

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def pre_tool_use(
        self, tool_name: str, tool_input: dict, tool_use_id: str
    ) -> HookResult:
        hooks = self._matching("PreToolUse", tool_name)
        if not hooks:
            return HookResult()
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": tool_use_id,
        }
        return await self._run_list(hooks, payload, can_block=True)

    async def post_tool_use(
        self, tool_name: str, tool_input: dict, tool_use_id: str, response: str
    ) -> HookResult:
        hooks = self._matching("PostToolUse", tool_name)
        if not hooks:
            return HookResult()
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": tool_use_id,
            "response": response,
        }
        return await self._run_list(hooks, payload, can_block=False)

    async def session_start(self) -> None:
        hooks = self._matching("SessionStart")
        if not hooks:
            return
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "SessionStart",
            "cwd": self._cwd,
        }
        await self._run_list(hooks, payload, can_block=False)

    async def session_end(self) -> None:
        hooks = self._matching("SessionEnd")
        if not hooks:
            return
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "SessionEnd",
            "cwd": self._cwd,
        }
        await self._run_list(hooks, payload, can_block=False)

    async def pre_compact(self, message_count: int) -> None:
        hooks = self._matching("PreCompact")
        if not hooks:
            return
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "PreCompact",
            "message_count": message_count,
            "cwd": self._cwd,
        }
        await self._run_list(hooks, payload, can_block=False)

    async def post_compact(self, summary_length: int) -> None:
        hooks = self._matching("PostCompact")
        if not hooks:
            return
        payload = {
            "session_id": self._session_id,
            "hook_event_name": "PostCompact",
            "summary_length": summary_length,
            "cwd": self._cwd,
        }
        await self._run_list(hooks, payload, can_block=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_list(
        self, hooks: list[dict], payload: dict, can_block: bool
    ) -> HookResult:
        combined = HookResult()
        for hook in hooks:
            r = await self._run_one(hook, payload)
            if can_block and not r.allowed:
                return r   # first blocking hook wins
            if r.additional_context:
                combined.additional_context = (
                    (combined.additional_context + "\n" + r.additional_context).strip()
                )
            # Accumulate exit-2 stderr from PostToolUse as additional context
            if not can_block and r.exit_code == 2 and r.stderr:
                combined.additional_context = (
                    (combined.additional_context + "\n" + r.stderr.strip()).strip()
                )
        return combined

    async def _run_one(self, hook: dict, payload: dict) -> HookResult:
        command = hook.get("command", "").strip()
        if not command:
            return HookResult()

        timeout = float(hook.get("timeout", TOOL_HOOK_TIMEOUT))
        is_async = hook.get("async", False)
        stdin_bytes = json.dumps(payload).encode()
        env = {**os.environ, "CCA_HOOK_EVENT": payload.get("hook_event_name", "")}

        loop = asyncio.get_event_loop()

        def _exec() -> tuple[int, str, str]:
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    input=stdin_bytes,
                    capture_output=True,
                    timeout=timeout,
                    env=env,
                )
                return (
                    proc.returncode,
                    proc.stdout.decode(errors="replace"),
                    proc.stderr.decode(errors="replace"),
                )
            except subprocess.TimeoutExpired:
                return 1, "", f"[Hook timed out after {timeout}s]"
            except Exception as exc:
                return 1, "", str(exc)

        if is_async:
            # Fire-and-forget — don't await
            loop.run_in_executor(None, _exec)
            return HookResult()

        rc, stdout, stderr = await loop.run_in_executor(None, _exec)

        # Parse optional JSON output from the hook
        allowed = True
        additional_context = ""
        if stdout.strip():
            try:
                parsed = json.loads(stdout.strip())
                if isinstance(parsed, dict):
                    if (
                        parsed.get("continue") is False
                        or parsed.get("decision") == "block"
                    ):
                        allowed = False
                        if not stderr:
                            stderr = (
                                parsed.get("reason")
                                or parsed.get("stopReason")
                                or "Blocked by hook"
                            )
                    additional_context = str(parsed.get("additionalContext", ""))
            except (json.JSONDecodeError, ValueError):
                pass

        # Exit code 2 always blocks (PreToolUse) or injects (PostToolUse)
        if rc == 2 and allowed:
            allowed = False

        return HookResult(
            allowed=allowed,
            additional_context=additional_context,
            stdout=stdout,
            stderr=stderr.strip(),
            exit_code=rc,
        )
