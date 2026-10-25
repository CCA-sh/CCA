"""End-to-end test: thought_signature through the full Repl.process_turn() pipeline.

This test exercises the exact same code path as the CLI:
  User message
  → Repl.process_turn()
  → Repl._stream_response()
  → OpenAIBackend.stream()
  → httpx HTTP call (intercepted)
  → SSE parse → ToolUseContent(thought_signature=...)
  → self.messages.append(assistant message)
  → tool executes (mocked)
  → self.messages.append(tool results)
  → second httpx HTTP call (captured)
  → ASSERT thought_signature is in the second request

The thought_signature is injected at the raw SSE level, matching what real Gemini sends.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cca.config import Config
from cca.llm.base import ToolSchema
from cca.llm.factory import create_backend
from cca.permissions import PermissionManager
from cca.repl import Repl
from cca.tools import build_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_SIG = "FAKE_THOUGHT_SIG_" + "A" * 40


def _sse_lines(*chunks: dict) -> bytes:
    """Build SSE bytes from chunk dicts, terminated with [DONE]."""
    parts = []
    for c in chunks:
        parts.append(f"data: {json.dumps(c)}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _bash_tool_call_sse(cmd: str, thought_signature: str | None = None) -> bytes:
    """SSE for a Bash tool call — mirrors real Gemini format (everything in one chunk)."""
    func: dict = {
        "name": "Bash",
        "arguments": json.dumps({"command": cmd, "description": f"Run {cmd}"}),
    }
    if thought_signature:
        func["thought_signature"] = thought_signature

    chunk = {
        "choices": [{
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_e2e_001",
                    "type": "function",
                    "function": func,
                }],
            },
            "finish_reason": "tool_calls",
            "index": 0,
        }],
        "id": "resp_e2e_001",
        "model": "gemini-2.5-pro",
        "object": "chat.completion.chunk",
    }
    return _sse_lines(chunk)


def _glob_tool_call_sse_real_gemini(pattern: str, thought_signature: str) -> bytes:
    """SSE exactly matching the real Gemini format observed in sse.log:
    thought_signature lives at tc_delta['extra_content']['google']['thought_signature'].
    """
    chunk = {
        "choices": [{
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "extra_content": {"google": {"thought_signature": thought_signature}},
                    "function": {
                        "arguments": json.dumps({"pattern": pattern}),
                        "name": "Glob",
                    },
                    "id": "real_gemini_001",
                    "type": "function",
                }],
            },
            "finish_reason": "stop",
            "index": 0,
        }],
        "model": "gemini-3-flash-preview",
        "object": "chat.completion.chunk",
    }
    return _sse_lines(chunk)


def _text_response_sse(text: str) -> bytes:
    """SSE for a plain text response."""
    chunks = [
        {
            "choices": [{
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
                "index": 0,
            }]
        },
        {
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
            }]
        },
    ]
    return _sse_lines(*chunks)


class _FakeSSEStream:
    """Minimal async stream that yields lines from pre-baked SSE bytes."""

    def __init__(self, body: bytes):
        self._body = body

    def iter_lines(self):
        async def _gen():
            for raw in self._body.split(b"\n"):
                line = raw.decode().rstrip("\r")
                if line:
                    yield line
        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


def _outgoing_sig(tc_entry: dict) -> str | None:
    """Extract thought_signature from the Gemini-format extra_content.google location."""
    return (tc_entry.get("extra_content") or {}).get("google", {}).get("thought_signature")


class _PatchedWSR:
    """
    Replaces client.chat.completions.with_streaming_response so we can
    intercept calls and inject fake SSE.

    Each call pops from the `responses` queue.  If `capture_last` is True,
    the final call is ALSO recorded in `.last_request`.
    """

    def __init__(self, responses: list[bytes]):
        self._responses = list(responses)
        self.last_request: dict | None = None
        self.call_count = 0

    @property
    def create(self):
        def _create(**kwargs):
            self.call_count += 1
            self.last_request = kwargs
            if self._responses:
                body = self._responses.pop(0)
            else:
                body = _text_response_sse("(no more fake responses)")
            return _FakeSSEStream(body)
        return _create


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_repl() -> Repl:
    cfg = Config.load()
    api_key = cfg.api_keys.get("google") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No Google API key configured")

    backend = create_backend("google:gemini-2.5-pro", api_key=api_key)
    registry = build_registry(config=cfg)
    perms = PermissionManager(auto_approve=True)
    return Repl(backend=backend, registry=registry, config=cfg, permissions=perms)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_thought_signature_captured_by_repl():
    """
    Verify that Repl._stream_response() stores thought_signature on the
    ToolUseContent in self.messages after a streaming response that includes it.
    """
    from cca import notifications
    repl = _make_repl()
    repl._notif_queue = asyncio.Queue()

    wsr = _PatchedWSR([_bash_tool_call_sse("ls", thought_signature=FAKE_SIG)])

    with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
        system = repl._build_system()
        content = await repl._stream_response(system)

    # The ToolUseContent in content should have the signature
    tool_uses = [b for b in content[0] if hasattr(b, "thought_signature")]
    assert tool_uses, "No ToolUseContent in content"
    assert tool_uses[0].thought_signature == FAKE_SIG, (
        f"thought_signature not captured. Got: {tool_uses[0].thought_signature!r}"
    )

    # It should also be in self.messages
    asst_msg = repl.messages[-1]
    tool_uses_in_msg = [b for b in asst_msg.content if hasattr(b, "thought_signature")]
    assert tool_uses_in_msg[0].thought_signature == FAKE_SIG, (
        "thought_signature lost when appended to self.messages"
    )
    print(f"\n✓ thought_signature captured: {FAKE_SIG[:30]}...")


@pytest.mark.asyncio
async def test_thought_signature_echoed_in_second_request():
    """
    Full pipeline test: two LLM turns.

    Turn 1: mock SSE returns Bash tool call WITH thought_signature.
    Turn 2: we capture the outgoing HTTP request and verify thought_signature
            is included in the assistant message's tool_calls.

    This is the exact scenario that causes the Gemini 400 error when
    thought_signature is NOT echoed.
    """
    from cca import notifications

    repl = _make_repl()
    repl._notif_queue = asyncio.Queue()

    # Turn 1 response: Bash tool call WITH thought_signature
    turn1_sse = _bash_tool_call_sse("ls -la", thought_signature=FAKE_SIG)

    # Turn 2 response: plain text (we just want to see what was sent)
    turn2_sse = _text_response_sse("There are 4 files in total.")

    wsr = _PatchedWSR([turn1_sse, turn2_sse])

    # Mock the Bash tool execution so we don't need a real shell
    async def _fake_bash(tc):
        return "file1.txt\nfile2.txt\npyproject.toml\nREADME.md"

    with patch.object(repl, "_execute_one", side_effect=_fake_bash):
        with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
            # Turn 1: triggers the tool call
            await repl.process_turn("List the files in the current directory.")

    assert wsr.call_count == 2, f"Expected 2 LLM calls, got {wsr.call_count}"

    # Inspect what was sent in the SECOND call
    sent_messages = wsr.last_request.get("messages", [])
    asst_messages = [m for m in sent_messages if m.get("role") == "assistant"]
    assert asst_messages, "No assistant message in second request"

    # Find the assistant message that has tool_calls
    asst_with_tools = [m for m in asst_messages if m.get("tool_calls")]
    assert asst_with_tools, (
        f"No assistant message with tool_calls in second request.\n"
        f"Assistant messages: {json.dumps(asst_messages, indent=2)}"
    )

    tool_calls_sent = asst_with_tools[0]["tool_calls"]
    tc_entry = tool_calls_sent[0]
    actual_sig = _outgoing_sig(tc_entry)

    print(f"\nTurn 2 outgoing tool_call entry: {json.dumps(tc_entry, indent=2)}")
    print(f"thought_signature in second request: {actual_sig!r}")

    assert actual_sig == FAKE_SIG, (
        f"thought_signature NOT echoed in extra_content.google in second request!\n"
        f"Expected: {FAKE_SIG!r}\n"
        f"Got:      {actual_sig!r}\n"
        f"Full tc_entry: {json.dumps(tc_entry, indent=2)}"
    )
    print(f"\n✓ thought_signature correctly echoed in second request.")


@pytest.mark.asyncio
async def test_thought_signature_after_finish_reason():
    """
    Gemini can send thought_signature in a chunk that arrives AFTER finish_reason.
    Verify that case is handled correctly through the full Repl pipeline.
    """
    from cca import notifications

    repl = _make_repl()
    repl._notif_queue = asyncio.Queue()

    # SSE where thought_signature arrives in a separate chunk after finish_reason
    chunk1 = {
        "choices": [{
            "delta": {
                "role": "assistant",
                "tool_calls": [{"id": "call_split_001", "type": "function",
                                "function": {"name": "Bash", "arguments": ""}}],
            },
            "finish_reason": None,
            "index": 0,
        }]
    }
    chunk2 = {
        "choices": [{
            "delta": {
                "tool_calls": [{"function": {
                    "arguments": json.dumps({"command": "pwd", "description": "show cwd"})
                }}],
            },
            "finish_reason": "tool_calls",
            "index": 0,
        }]
    }
    # thought_signature arrives AFTER finish_reason
    chunk3 = {
        "choices": [{
            "delta": {
                "tool_calls": [{"function": {"thought_signature": FAKE_SIG}}],
            },
            "finish_reason": None,
            "index": 0,
        }]
    }
    split_sse = _sse_lines(chunk1, chunk2, chunk3)
    turn2_sse = _text_response_sse("Done.")

    wsr = _PatchedWSR([split_sse, turn2_sse])

    async def _fake_bash(tc):
        return "/home/user/project"

    with patch.object(repl, "_execute_one", side_effect=_fake_bash):
        with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
            await repl.process_turn("What is the current directory?")

    assert wsr.call_count == 2, f"Expected 2 LLM calls, got {wsr.call_count}"

    sent_messages = wsr.last_request.get("messages", [])
    asst_with_tools = [m for m in sent_messages
                       if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst_with_tools, "No assistant tool_call message in second request"

    tc_entry = asst_with_tools[0]["tool_calls"][0]
    actual_sig = _outgoing_sig(tc_entry)

    print(f"\nthought_signature in second request (split case): {actual_sig!r}")
    assert actual_sig == FAKE_SIG, (
        f"thought_signature NOT echoed in extra_content.google when it arrives after finish_reason!\n"
        f"Got: {actual_sig!r}"
    )
    print("✓ thought_signature correctly echoed (split chunk case).")


@pytest.mark.asyncio
async def test_no_thought_signature_no_error():
    """
    Sanity check: when Gemini does NOT return thought_signature, the second
    request also omits it (no spurious None echoed).
    """
    from cca import notifications

    repl = _make_repl()
    repl._notif_queue = asyncio.Queue()

    turn1_sse = _bash_tool_call_sse("echo hi", thought_signature=None)
    turn2_sse = _text_response_sse("Done.")

    wsr = _PatchedWSR([turn1_sse, turn2_sse])

    async def _fake_bash(tc):
        return "hi"

    with patch.object(repl, "_execute_one", side_effect=_fake_bash):
        with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
            await repl.process_turn("Run echo hi.")

    sent_messages = wsr.last_request.get("messages", [])
    asst_with_tools = [m for m in sent_messages
                       if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst_with_tools

    tc_entry = asst_with_tools[0]["tool_calls"][0]
    assert "extra_content" not in tc_entry, (
        f"extra_content should be absent when model doesn't return thought_signature, got: {tc_entry}"
    )
    assert "thought_signature" not in tc_entry["function"], (
        f"thought_signature should not be in function dict, got: {tc_entry['function']}"
    )
    print("\n✓ No spurious thought_signature when model doesn't return one.")


@pytest.mark.asyncio
async def test_real_gemini_extra_content_format():
    """
    Reproduces the exact SSE format from the user's sse.log:

      tc_delta['extra_content']['google']['thought_signature'] = '...'

    This was the actual root cause of the persistent 400 error — thought_signature
    was at a path we never checked.
    """
    from cca import notifications

    repl = _make_repl()
    repl._notif_queue = asyncio.Queue()

    turn1_sse = _glob_tool_call_sse_real_gemini("*", FAKE_SIG)
    turn2_sse = _text_response_sse("Found 4 files.")

    wsr = _PatchedWSR([turn1_sse, turn2_sse])

    async def _fake_glob(tc):
        return "file1.txt\nfile2.txt"

    with patch.object(repl, "_execute_one", side_effect=_fake_glob):
        with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
            await repl.process_turn("List files matching *.")

    assert wsr.call_count == 2, f"Expected 2 LLM calls, got {wsr.call_count}"

    sent_messages = wsr.last_request.get("messages", [])
    asst_with_tools = [m for m in sent_messages
                       if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst_with_tools, "No tool_calls in second request"

    tc_entry = asst_with_tools[0]["tool_calls"][0]
    actual_sig = _outgoing_sig(tc_entry)
    print(f"\nReal Gemini format — thought_signature echoed: {actual_sig!r}")
    assert actual_sig == FAKE_SIG, (
        f"thought_signature NOT echoed in extra_content.google!\n"
        f"Got: {actual_sig!r}"
    )
    print("✓ Real Gemini extra_content.google.thought_signature format handled correctly.")


@pytest.mark.asyncio
async def test_thought_signature_in_standalone_chunk():
    """
    Gemini may deliver thought_signature in a completely separate SSE chunk
    that has NO tool_calls array at all — just a bare signature field.

    This is the scenario most likely to cause the 400 error if unhandled.
    We test all three possible nesting levels for the standalone chunk:
      - delta.thought_signature
      - choice.thought_signature
      - chunk.thought_signature (top-level)
    """
    from cca import notifications

    for location, standalone_chunk in [
        ("delta level",  {"choices": [{"delta": {"thought_signature": FAKE_SIG}, "index": 0}]}),
        ("choice level", {"choices": [{"delta": {}, "thought_signature": FAKE_SIG, "index": 0}]}),
        ("chunk level",  {"choices": [{"delta": {}, "index": 0}], "thought_signature": FAKE_SIG}),
    ]:
        # Tool call comes first (no thought_signature), then standalone sig chunk
        tool_chunk = {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_standalone_001", "type": "function",
                        "function": {"name": "Bash",
                                     "arguments": json.dumps({"command": "ls", "description": "list"})},
                    }],
                },
                "finish_reason": "tool_calls",
                "index": 0,
            }]
        }
        standalone_sse = _sse_lines(tool_chunk, standalone_chunk)
        turn2_sse = _text_response_sse("Done.")

        repl = _make_repl()
        repl._notif_queue = asyncio.Queue()
        wsr = _PatchedWSR([standalone_sse, turn2_sse])

        async def _fake_bash(tc):
            return "file.txt"

        with patch.object(repl, "_execute_one", side_effect=_fake_bash):
            with patch.object(repl.backend.client.chat.completions, "with_streaming_response", wsr):
                await repl.process_turn("List files.")

        sent_messages = wsr.last_request.get("messages", [])
        asst_with_tools = [m for m in sent_messages
                           if m.get("role") == "assistant" and m.get("tool_calls")]
        assert asst_with_tools, f"[{location}] No tool_calls in second request"
        tc_entry = asst_with_tools[0]["tool_calls"][0]
        actual_sig = _outgoing_sig(tc_entry)
        assert actual_sig == FAKE_SIG, (
            f"[{location}] thought_signature NOT echoed in extra_content.google!\n"
            f"Got: {actual_sig!r}"
        )
        print(f"\n✓ [{location}] standalone thought_signature chunk handled correctly.")
