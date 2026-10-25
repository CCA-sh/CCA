"""AskUserQuestion — present structured multiple-choice questions in the terminal."""

from __future__ import annotations

import asyncio
from typing import Callable, Coroutine

from cca.tools.base import Tool


class AskUserQuestionTool(Tool):
    name = "AskUserQuestion"
    description = (
        "Present 1-4 structured questions to the user with predefined options. "
        "Users can also type custom answers. "
        "Use this to clarify ambiguous requirements or gather preferences. "
        "In plan mode, use before finalizing the plan. "
        "Do NOT use to ask 'Is my plan ready?' — use ExitPlanMode for that."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask.",
                        },
                        "header": {
                            "type": "string",
                            "description": "Short label for the question (max 12 chars).",
                        },
                        "options": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label", "description"],
                            },
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "Allow multiple selections.",
                        },
                    },
                    "required": ["question", "header", "options", "multiSelect"],
                },
                "description": "Questions to ask (1-4).",
            },
        },
        "required": ["questions"],
    }

    def __init__(
        self,
        # async (question, options, multi, header) -> str
        # Provided by TerminalUI; when None, falls back to blocking input() which
        # is safe in non-interactive (print/test) mode where prompt_toolkit is not running.
        ask_fn: Callable[..., Coroutine] | None = None,
    ) -> None:
        self._ask_fn = ask_fn

    async def execute(
        self,
        questions: list[dict],
        answers: dict | None = None,
        annotations: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        results = {}

        for q in questions:
            question_text = q.get("question", "")
            options = q.get("options", [])
            multi = q.get("multiSelect", False)
            header = q.get("header", "")

            if self._ask_fn is not None:
                answer = await self._ask_fn(question_text, options, multi, header)
            else:
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None,
                    lambda qt=question_text, opts=options, m=multi, h=header:
                        self._ask_sync(qt, opts, m, h),
                )
            results[question_text] = answer

        lines = []
        for q_text, ans in results.items():
            lines.append(f"Q: {q_text}\nA: {ans}")
        return "\n\n".join(lines)

    def _ask_sync(
        self,
        question: str,
        options: list[dict],
        multi: bool,
        header: str,
    ) -> str:
        """Fallback used only when no TerminalUI is active (print / test mode)."""
        import sys
        print(f"\n[{header}] {question}", file=sys.stderr)
        for i, opt in enumerate(options, 1):
            label = opt.get("label", "")
            desc = opt.get("description", "")
            print(f"  {i}. {label} — {desc}", file=sys.stderr)
        print(f"  {len(options) + 1}. Other (type your own)", file=sys.stderr)

        while True:
            raw = input("Your choice (number or text): ").strip()
            if not raw:
                continue
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]["label"]
                elif idx == len(options):
                    return input("Enter your answer: ").strip()
            return raw
