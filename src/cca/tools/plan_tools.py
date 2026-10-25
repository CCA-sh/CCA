"""Plan mode tools: EnterPlanMode and ExitPlanMode.

Plan mode is a session flag. When active, the agent explores the codebase
and writes a plan to a file before asking the user for approval.
"""

from __future__ import annotations

from pathlib import Path

from typing import Callable, Coroutine

from cca.tools.base import Tool
from cca.config import PLANS_DIR

# Session state
_plan_mode_active: bool = False
_current_plan_file: Path | None = None
_non_interactive: bool = False  # set to True in --print / --auto-approve mode


def is_plan_mode_active() -> bool:
    return _plan_mode_active


def get_plan_file() -> Path | None:
    return _current_plan_file


def set_non_interactive(flag: bool) -> None:
    """Call this before starting the agent when stdin won't be available."""
    global _non_interactive
    _non_interactive = flag


class EnterPlanModeTool(Tool):
    name = "EnterPlanMode"
    description = (
        "Enter plan mode before implementing a non-trivial task. "
        "In plan mode you should explore the codebase, design an approach, "
        "write it to the plan file provided, then call ExitPlanMode to "
        "present the plan for user approval. "
        "Use this proactively for any task touching 3+ files or requiring "
        "architectural decisions."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> str:
        global _plan_mode_active, _current_plan_file
        # Idempotent: if already in plan mode, return the existing file path
        if _plan_mode_active and _current_plan_file is not None:
            return (
                f"Already in plan mode. Current plan file:\n{_current_plan_file}\n\n"
                "Continue writing your plan there, then call ExitPlanMode."
            )
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
        import time
        plan_file = PLANS_DIR / f"plan_{int(time.time())}.md"
        _plan_mode_active = True
        _current_plan_file = plan_file
        return (
            f"Entered plan mode. Write your implementation plan to:\n{plan_file}\n\n"
            "Explore the codebase, then write a structured plan including:\n"
            "1. Summary of approach\n"
            "2. Files to modify/create\n"
            "3. Step-by-step implementation\n"
            "4. Risks and trade-offs\n"
            "5. Testing approach\n\n"
            "Then call ExitPlanMode to request user approval."
        )


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    description = (
        "Signal that your plan is complete and request user approval. "
        "You must have written your plan to the file specified by EnterPlanMode "
        "before calling this tool. "
        "Do NOT use this for research/exploration tasks — only for implementation planning."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "allowedPrompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                },
                "description": "Permissions needed to implement the plan.",
            },
        },
    }

    def __init__(self, ask_fn: Callable[..., Coroutine] | None = None) -> None:
        # ask_fn mirrors AskUserQuestionTool: async (question, options, multi, header) -> str
        # Injected from TerminalUI so the approval dialog routes through prompt_toolkit,
        # not a raw input() call that deadlocks against the TUI's stdin ownership.
        self._ask_fn = ask_fn

    async def execute(self, allowedPrompts: list | None = None) -> str:
        global _plan_mode_active

        plan_file = _current_plan_file
        if not plan_file or not plan_file.exists():
            return (
                "Error: no plan file found. Write your plan to the file specified "
                "by EnterPlanMode before calling ExitPlanMode."
            )

        plan_content = plan_file.read_text()
        _plan_mode_active = False

        import sys

        print("\n" + "=" * 60, file=sys.stderr)
        print("IMPLEMENTATION PLAN", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(plan_content, file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # In non-interactive mode (--print / --auto-approve / no tty) skip the
        # prompt and auto-approve so the agent isn't blocked waiting for stdin.
        if _non_interactive or not sys.stdin.isatty():
            print("\n[Plan auto-approved: non-interactive mode]", file=sys.stderr)
            return "Plan approved. Proceed with implementation."

        if self._ask_fn is not None:
            # UI mode: route through prompt_toolkit's question dialog.
            # Using raw input() here deadlocks because prompt_toolkit owns stdin.
            answer_str = await self._ask_fn(
                "Approve this plan and proceed with implementation?",
                [
                    {"label": "Yes, proceed", "description": "Implement the plan as written"},
                    {"label": "No, revise", "description": "Go back and change the approach"},
                ],
                False,
                "Plan approval",
            )
            approved = answer_str.startswith("Yes")
        else:
            loop = __import__("asyncio").get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: input("\nApprove this plan and proceed? [y/n]: ").strip().lower(),
            )
            approved = answer in ("y", "yes")

        if approved:
            return "Plan approved. Proceed with implementation."
        else:
            return "Plan rejected. Revise the approach and try again."
