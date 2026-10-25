"""Multi-turn session test: two user messages in the same conversation."""

from __future__ import annotations

import pytest

from cca.llm.base import Message, TextContent, ToolResultContent, ToolUseContent, ToolSchema
from cca.llm.openai_backend import OpenAIBackend, _to_openai_messages


def _make_gemini_backend() -> OpenAIBackend:
    from cca.config import Config
    import os
    cfg = Config.load()
    api_key = cfg.api_keys.get("google") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No Google API key configured")
    return OpenAIBackend(
        model="gemini-2.5-pro",
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


TOOLS = [
    ToolSchema(
        name="Bash",
        description="Execute a shell command and return stdout+stderr.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["command"],
        },
    )
]

SYSTEM = (
    "You are a helpful coding assistant. "
    "Use the Bash tool to answer questions about the filesystem."
)


async def _do_turn(backend, messages: list[Message]) -> tuple[list[Message], str]:
    """
    Run one LLM turn: stream response, execute any tool calls with real bash,
    loop until no more tool calls.  Returns updated message list and final text.
    """
    import subprocess

    final_text = ""

    for _ in range(10):  # max tool-call rounds per user turn
        tool_uses: list[ToolUseContent] = []
        text_parts: list[str] = []
        assistant_blocks = []

        async for event in backend.stream(messages, TOOLS, SYSTEM, max_tokens=1024):
            if event.type == "text_delta":
                text_parts.append(event.data)
            elif event.type == "tool_use_complete":
                tool_uses.append(event.data)
                assistant_blocks.append(event.data)

        if text_parts:
            assistant_blocks.insert(0, TextContent(text="".join(text_parts)))
            final_text = "".join(text_parts)

        messages.append(Message(role="assistant", content=assistant_blocks))

        if not tool_uses:
            break

        # Execute tool calls with real bash
        tool_results = []
        for tc in tool_uses:
            cmd = tc.input.get("command", "")
            print(f"  [bash] $ {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            output = (result.stdout + result.stderr).strip() or f"[exit {result.returncode}]"
            print(f"  [bash] → {output[:200]}")
            tool_results.append(ToolResultContent(tool_use_id=tc.id, content=output))

        messages.append(Message(role="user", content=tool_results))

    return messages, final_text


@pytest.mark.asyncio
async def test_two_turn_directory_session():
    """
    Turn 1: list contents of current directory.
    Turn 2 (same session): count the files.
    Verifies multi-turn context is preserved and no 400 errors occur.
    """
    backend = _make_gemini_backend()
    messages: list[Message] = []

    # ── Turn 1 ──────────────────────────────────────────────
    print("\n─── Turn 1: list contents ───")
    messages.append(Message(role="user", content=[
        TextContent(text="List the contents of the current directory.")
    ]))
    messages, reply1 = await _do_turn(backend, messages)
    print(f"Reply: {reply1[:300]}")
    assert reply1, "Expected a text reply for turn 1"

    # ── Turn 2 ──────────────────────────────────────────────
    print("\n─── Turn 2: count the files ───")
    messages.append(Message(role="user", content=[
        TextContent(text="How many files are in the current directory?")
    ]))
    messages, reply2 = await _do_turn(backend, messages)
    print(f"Reply: {reply2[:300]}")
    assert reply2, "Expected a text reply for turn 2"

    # The reply should mention a number
    import re
    has_number = bool(re.search(r'\d+', reply2))
    assert has_number, f"Expected a number in reply2, got: {reply2}"

    print(f"\n✓ Two-turn session completed. Messages in history: {len(messages)}")
