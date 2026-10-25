"""Tests for cca.llm.retry — classification, backoff, budget, stream safety."""
from __future__ import annotations

import asyncio

import pytest

from cca.llm import retry as R

RetryDefaults = R.RetryPolicy()


class _Resp:
    def __init__(self, status: int, headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}


class _ApiError(Exception):
    """Stand-in for openai/anthropic APIStatusError."""

    def __init__(self, status: int, headers: dict | None = None, msg: str = "boom"):
        super().__init__(msg)
        self.status_code = status
        self.response = _Resp(status, headers)


@pytest.fixture(autouse=True)
def _clean_budget():
    R.budget().reset()
    R.set_wait_observer(None)
    yield
    R.budget().reset()
    R.set_wait_observer(None)


@pytest.fixture
def _instant_sleep(monkeypatch):
    """Run the retry loops without actually waiting."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(R.asyncio, "sleep", fake_sleep)
    return slept


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (429, R.THROTTLE),
    (529, R.THROTTLE),
    (500, R.TRANSIENT),
    (502, R.TRANSIENT),
    (503, R.TRANSIENT),
    (504, R.TRANSIENT),
    (408, R.TRANSIENT),
    (400, R.FATAL),
    (401, R.FATAL),
    (404, R.FATAL),
    (422, R.FATAL),
])
def test_classify_by_status(status, expected):
    assert R.classify(_ApiError(status)) == expected


def test_classify_transport_errors():
    assert R.classify(ConnectionError("reset")) == R.TRANSIENT
    assert R.classify(asyncio.TimeoutError()) == R.TRANSIENT
    assert R.classify(ValueError("nonsense")) == R.FATAL


def test_retry_after_numeric_and_bogus():
    assert R.retry_after_seconds(_ApiError(429, {"retry-after": "12"})) == 12.0
    assert R.retry_after_seconds(_ApiError(429, {"retry-after": "-1"})) is None
    assert R.retry_after_seconds(_ApiError(429, {})) is None
    assert R.retry_after_seconds(ValueError("no response")) is None
    # A hostile value is clamped rather than parking the run for an hour.
    assert R.retry_after_seconds(
        _ApiError(429, {"retry-after": "99999"})
    ) == R._MAX_RETRY_AFTER_S


# ---------------------------------------------------------------------------
# call_with_retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_retries_then_succeeds(_instant_sleep):
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _ApiError(429)
        return "answer"

    assert await R.call_with_retry(fn) == "answer"
    assert calls["n"] == 3
    assert len(_instant_sleep) == 2


@pytest.mark.asyncio
async def test_call_never_retries_fatal(_instant_sleep):
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _ApiError(400, msg="bad request")

    with pytest.raises(_ApiError):
        await R.call_with_retry(fn)
    assert calls["n"] == 1
    assert _instant_sleep == []


@pytest.mark.asyncio
async def test_call_raises_llm_unavailable_when_exhausted(_instant_sleep):
    async def fn():
        raise _ApiError(429)

    policy = R.RetryPolicy(max_attempts=3, base_s=1.0, cap_s=4.0)
    with pytest.raises(R.LLMUnavailable) as excinfo:
        await R.call_with_retry(fn, policy=policy)
    assert excinfo.value.kind == R.THROTTLE
    assert excinfo.value.attempts == 3
    assert len(_instant_sleep) == 2


@pytest.mark.asyncio
async def test_wait_budget_stops_before_the_deadline(_instant_sleep):
    async def fn():
        raise _ApiError(429, {"retry-after": "30"})

    # One 30s wait fits; a second would not.
    policy = R.RetryPolicy(max_attempts=10, max_total_wait_s=45.0)
    with pytest.raises(R.LLMUnavailable) as excinfo:
        await R.call_with_retry(fn, policy=policy)
    assert "budget" in str(excinfo.value)
    assert len(_instant_sleep) == 1
    assert R.budget().waited_s <= 45.0


@pytest.mark.asyncio
async def test_observer_is_notified_before_sleeping(_instant_sleep):
    seen: list[dict] = []
    R.set_wait_observer(seen.append)

    async def fn():
        raise _ApiError(429, {"retry-after": "5"})

    policy = R.RetryPolicy(max_attempts=2)
    with pytest.raises(R.LLMUnavailable):
        await R.call_with_retry(fn, policy=policy, what="stream[m]")
    assert len(seen) == 1
    assert seen[0]["kind"] == R.THROTTLE
    assert seen[0]["what"] == "stream[m]"
    assert 5.0 <= seen[0]["wait_s"] <= 6.0


@pytest.mark.asyncio
async def test_observer_failure_does_not_break_the_call(_instant_sleep):
    def boom(_payload):
        raise RuntimeError("observer is broken")

    R.set_wait_observer(boom)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _ApiError(429)
        return "ok"

    assert await R.call_with_retry(fn) == "ok"


# ---------------------------------------------------------------------------
# stream_with_retry
# ---------------------------------------------------------------------------

def _stream_factory(scripts):
    """Build a make_stream that plays `scripts` one attempt at a time."""
    state = {"i": 0}

    async def make_stream():
        script = scripts[state["i"]]
        state["i"] += 1
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item

    return make_stream


@pytest.mark.asyncio
async def test_stream_retries_before_first_event(_instant_sleep):
    make = _stream_factory([[_ApiError(429)], ["a", "b"]])
    got = [e async for e in R.stream_with_retry(make)]
    assert got == ["a", "b"]
    assert len(_instant_sleep) == 1


@pytest.mark.asyncio
async def test_stream_never_retries_after_first_event(_instant_sleep):
    """A committed stream must not restart — the caller already has partials."""
    make = _stream_factory([["a", _ApiError(429)], ["a", "b"]])
    got = []
    with pytest.raises(_ApiError):
        async for event in R.stream_with_retry(make):
            got.append(event)
    assert got == ["a"]
    assert _instant_sleep == []


@pytest.mark.asyncio
async def test_stream_fatal_is_not_retried(_instant_sleep):
    make = _stream_factory([[_ApiError(400)], ["a"]])
    with pytest.raises(_ApiError):
        async for _ in R.stream_with_retry(make):
            pass
    assert _instant_sleep == []


@pytest.mark.asyncio
async def test_stream_exhaustion_raises_llm_unavailable(_instant_sleep):
    make = _stream_factory([[_ApiError(503)]] * 4)
    policy = R.RetryPolicy(max_attempts=3, base_s=1.0)
    with pytest.raises(R.LLMUnavailable) as excinfo:
        async for _ in R.stream_with_retry(make, policy=policy):
            pass
    assert excinfo.value.kind == R.TRANSIENT


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_policy_reads_environment(monkeypatch):
    monkeypatch.setenv("CCA_LLM_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("CCA_LLM_MAX_THROTTLE_WAIT_S", "120")
    monkeypatch.setenv("CCA_LLM_BACKOFF_CAP_S", "not-a-number")
    policy = R.RetryPolicy.from_env()
    assert policy.max_attempts == 9
    assert policy.max_total_wait_s == 120.0
    assert policy.cap_s == RetryDefaults.cap_s



def test_delay_is_bounded_by_the_cap():
    policy = R.RetryPolicy(base_s=2.0, cap_s=10.0)
    for attempt in range(1, 12):
        delay = R._delay_for(_ApiError(500), attempt, policy)
        assert 0.0 < delay <= policy.cap_s


# ---------------------------------------------------------------------------
# Repl wiring: the harness has to hear about a wait before it happens
# ---------------------------------------------------------------------------

def _machine_repl(**kwargs):
    from unittest.mock import AsyncMock, MagicMock

    from cca.repl import Repl

    backend = MagicMock()
    backend.stream = AsyncMock()
    registry = MagicMock()
    registry.get_schemas = MagicMock(return_value=[])
    registry._mcp_manager = None
    config = MagicMock()
    config.model = "litellm:fireworks_ai/kimi-k3"
    config.trace = False
    config.verbose = False
    config.hooks = {}
    config.max_tokens = 8192
    config.skills_enabled = False
    config.system_prompt_append = None
    config.vim = False
    perms = MagicMock()
    perms.check = AsyncMock(return_value=True)
    return Repl(
        backend=backend, registry=registry, config=config, permissions=perms,
        cwd="/tmp", print_mode=False, **kwargs,
    )


def test_machine_mode_announces_a_wait_as_a_json_event():
    """The harness owns a per-turn deadline it cannot otherwise tell apart
    from the agent being slow, so it is told before the sleep, not after."""
    repl = _machine_repl(run_mode="machine")
    written: list[dict] = []
    repl._machine_writeline = written.append

    repl._on_llm_wait({"kind": R.THROTTLE, "attempt": 1, "wait_s": 12.5,
                       "what": "stream[kimi-k3]", "detail": "429"})

    assert len(written) == 1
    assert written[0]["type"] == "throttled"
    assert written[0]["wait_s"] == 12.5
    assert repl._throttle_wait_s == 12.5
    assert repl._throttle_events == 1


def test_print_mode_does_not_write_events_to_stdout():
    """`--print` stdout carries the answer; a wait belongs on stderr."""
    repl = _machine_repl(run_mode="print")
    written: list[dict] = []
    repl._machine_writeline = written.append

    repl._on_llm_wait({"kind": R.THROTTLE, "attempt": 1, "wait_s": 3.0})

    assert written == []
    assert repl._throttle_wait_s == 3.0


def test_turn_complete_carries_the_throttle_accounting():
    repl = _machine_repl(run_mode="machine")
    repl._machine_writeline = lambda _obj: None
    repl._on_llm_wait({"kind": R.THROTTLE, "attempt": 1, "wait_s": 8.0})
    repl._turn_throttled = True

    payload = repl._throttle_payload()
    assert payload == {"throttled": True, "throttle_wait_s": 8.0, "throttle_events": 1}


def test_the_repl_registers_itself_as_the_wait_observer():
    """Without this the backends wait in silence and the harness kills them."""
    repl = _machine_repl(run_mode="machine")
    written: list[dict] = []
    repl._machine_writeline = written.append

    R._notify({"kind": R.THROTTLE, "attempt": 1, "wait_s": 1.0})
    assert written and written[0]["type"] == "throttled"


# ---------------------------------------------------------------------------
# _complete_structured: litellm's allow-list goes in extra_body
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_litellm_schema_allowlist_is_nested_in_extra_body():
    """The openai SDK rejects any top-level kwarg it does not define, so a
    top-level `allowed_openai_params` raises TypeError before the request is
    sent — and it must not clobber the entry `_build_thinking_kwargs` already
    put in extra_body for reasoning_effort."""
    from unittest.mock import AsyncMock, MagicMock

    from cca.llm.base import Message, TextContent
    from cca.llm.openai_backend import OpenAIBackend

    backend = OpenAIBackend(model="fireworks_ai/kimi-k3", provider="litellm")
    seen: dict = {}

    async def fake_create(**kwargs):
        seen.update(kwargs)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"ok": true}'
        response.choices[0].finish_reason = "stop"
        response.usage = None
        return response

    backend.client = MagicMock()
    backend.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    result = await backend.complete(
        messages=[Message(role="user", content=[TextContent(text="hi")])],
        tools=[], system="s", max_tokens=64, think_level="medium",
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )

    assert result.mode_used == "json_schema"
    assert "allowed_openai_params" not in seen, "top-level would raise TypeError"
    allowed = seen["extra_body"]["allowed_openai_params"]
    assert "response_format" in allowed
    assert "reasoning_effort" in allowed, "the thinking entry must survive"


def test_a_typeerror_is_never_read_as_an_unsupported_schema():
    """Degrading to prompt_only on a locally malformed call would hide the bug
    behind permanently worse output."""
    from cca.llm.openai_backend import _is_unsupported_schema_error

    exc = TypeError(
        "AsyncCompletions.create() got an unexpected keyword argument "
        "'response_format'"
    )
    assert _is_unsupported_schema_error(exc) is False


@pytest.mark.asyncio
async def test_unsupported_native_schema_uses_forced_tool_before_prompt_only():
    from unittest.mock import AsyncMock, MagicMock

    from cca.llm.base import Message, TextContent
    from cca.llm.openai_backend import OpenAIBackend

    class UnsupportedSchema(Exception):
        status_code = 400

    backend = OpenAIBackend(
        model="anthropic/claude-sonnet-5", provider="litellm"
    )
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnsupportedSchema("response_format is not supported")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        response.choices[0].message.tool_calls = [MagicMock()]
        function = response.choices[0].message.tool_calls[0].function
        function.name = "cca_structured_response"
        function.arguments = '{"ok": true}'
        response.choices[0].finish_reason = "tool_calls"
        response.usage = None
        return response

    backend.client = MagicMock()
    backend.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    result = await backend.complete(
        messages=[Message(role="user", content=[TextContent(text="hi")])],
        tools=[],
        system="s",
        max_tokens=64,
        think_level="off",
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )

    assert result.mode_used == "forced_tool"
    assert result.text == '{"ok": true}'
    assert len(calls) == 2
    assert calls[1]["extra_body"]["thinking"] == {"type": "disabled"}
    assert calls[1]["tool_choice"]["function"]["name"] == (
        "cca_structured_response"
    )
    assert calls[1]["tools"][0]["function"]["parameters"][
        "additionalProperties"
    ] is False
    assert set(calls[1]["extra_body"]["allowed_openai_params"]) == {
        "tools",
        "tool_choice",
    }


@pytest.mark.asyncio
async def test_unsupported_forced_tool_is_the_only_path_to_prompt_only():
    from unittest.mock import AsyncMock, MagicMock

    from cca.llm.base import Message, TextContent
    from cca.llm.openai_backend import OpenAIBackend

    class UnsupportedFeature(Exception):
        status_code = 400

    backend = OpenAIBackend(model="some-route", provider="litellm")
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnsupportedFeature("json_schema is not supported")
        if len(calls) == 2:
            raise UnsupportedFeature("tool_choice is not supported")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"ok": true}'
        response.choices[0].finish_reason = "stop"
        response.usage = None
        return response

    backend.client = MagicMock()
    backend.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    result = await backend.complete(
        messages=[Message(role="user", content=[TextContent(text="hi")])],
        tools=[],
        system="s",
        max_tokens=64,
        think_level="off",
        output_schema={"type": "object", "properties": {}},
    )

    assert result.mode_used == "prompt_only"
    assert result.text == '{"ok": true}'
    assert len(calls) == 3
    assert "response_format" in calls[0]
    assert "tools" in calls[1]
    assert "response_format" not in calls[2]
    assert "tools" not in calls[2]


def test_rate_limit_is_not_read_as_unsupported_tool_use():
    from cca.llm.openai_backend import _is_unsupported_tool_error

    class RateLimit(Exception):
        status_code = 429

    assert not _is_unsupported_tool_error(
        RateLimit("tool calling was rate limited")
    )


def test_machine_mode_says_it_once():
    """The event and a console line on the same stderr made every wait appear
    twice in a live view."""
    repl = _machine_repl(run_mode="machine")
    written: list[dict] = []
    printed: list = []
    repl._machine_writeline = written.append
    repl._console = type("C", (), {"print": lambda _self, *a, **k: printed.append(a)})()

    repl._on_llm_wait({"kind": R.THROTTLE, "attempt": 1, "wait_s": 2.0})

    assert len(written) == 1
    assert printed == [], "machine mode must not also print prose"
    assert repl._throttle_wait_s == 2.0
