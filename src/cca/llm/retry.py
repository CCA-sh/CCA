"""Classify LLM transport failures and wait out the ones worth waiting out.

Behind a shared LiteLLM proxy the provider's 429 arrives verbatim
(`x-litellm-attempted-retries: 0`), so patience is entirely the client's job.
Without it a single busy moment upstream ends a whole agent turn: the stream
raises, `repl._stream_response_impl` returns an empty message, and a harness
downstream cannot tell "the API was busy" from "the model failed".

Three rules shape what follows.

1. **Classify before reacting.** 429/529 is throttling and deserves patience;
   5xx and dropped connections deserve backoff; a 400 deserves none at all —
   retrying a malformed request only burns quota.
2. **Never retry a stream that has already produced output.** The caller
   accumulates text as events arrive, so restarting mid-stream would duplicate
   it. A failure before the first event is a clean slate and is retried; a
   failure after one is passed straight through.
3. **Waiting is not free wall-clock.** Callers run under an external deadline
   (a harness per-turn timeout, a `subprocess.run` timeout), so total sleep is
   capped by a budget the caller resets at whatever it considers a turn.
   Exceeding the budget raises rather than sleeping past the deadline and
   being killed with nothing to show.

Waits are announced through `set_wait_observer` so the caller can tell its own
harness that the delay was the API, not the agent.
"""

from __future__ import annotations

import asyncio
import email.utils
import os
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Failure classes.
THROTTLE = "throttle"    # 429/529 — the provider is asking us to slow down
TRANSIENT = "transient"  # 5xx, reset connection, read timeout
FATAL = "fatal"          # 400/401/404/422 — retrying cannot help

# Retryable-but-not-throttling HTTP statuses.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 500, 502, 503, 504})
_THROTTLE_STATUS = frozenset({429, 529})

# Upper bound on a server-supplied Retry-After, so a hostile or buggy value
# cannot park a run for an hour.
_MAX_RETRY_AFTER_S = 300.0


class LLMUnavailable(RuntimeError):
    """Retries were exhausted, or the wait budget was, with no answer.

    `kind` is THROTTLE or TRANSIENT — the distinction a harness needs to
    separate "the API was busy" from "the agent failed". Never raised for a
    FATAL error, which propagates unchanged so existing handlers (e.g. the
    unsupported-schema fallback in `openai_backend`) still see the real
    provider exception.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        attempts: int,
        waited_s: float,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts
        self.waited_s = waited_s
        self.cause = cause


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _status_of(exc: BaseException) -> int | None:
    """HTTP status behind `exc`, if it carries one."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify(exc: BaseException) -> str:
    """Sort `exc` into THROTTLE, TRANSIENT or FATAL."""
    status = _status_of(exc)
    if status is not None:
        if status in _THROTTLE_STATUS:
            return THROTTLE
        if status in _TRANSIENT_STATUS:
            return TRANSIENT
        # Any other status came from a server that understood the request and
        # rejected it. Waiting changes nothing.
        return FATAL

    # No status: a transport-level failure that never reached a response.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return TRANSIENT
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name or "APIConnection" in name:
        return TRANSIENT
    return FATAL


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds requested by a `Retry-After` header, if present and sane."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:
        return None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        seconds = float(raw)
    except ValueError:
        # HTTP-date form.
        parsed = email.utils.parsedate_to_datetime(raw)
        if parsed is None:
            return None
        seconds = parsed.timestamp() - time.time()
    if seconds != seconds or seconds < 0:  # NaN or negative
        return None
    return min(seconds, _MAX_RETRY_AFTER_S)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


@dataclass(frozen=True)
class RetryPolicy:
    """How patient to be. Every field is overridable by environment."""

    max_attempts: int = 6
    base_s: float = 2.0
    cap_s: float = 60.0
    # Total sleep allowed between resets of the shared budget. Sized well under
    # a typical harness per-turn timeout; a caller that extends its own
    # deadline on wait notifications can raise it.
    max_total_wait_s: float = 600.0

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        default = cls()
        return cls(
            max_attempts=max(1, _env_int("CCA_LLM_MAX_ATTEMPTS", default.max_attempts)),
            base_s=_env_float("CCA_LLM_BACKOFF_BASE_S", default.base_s),
            cap_s=_env_float("CCA_LLM_BACKOFF_CAP_S", default.cap_s),
            max_total_wait_s=_env_float(
                "CCA_LLM_MAX_THROTTLE_WAIT_S", default.max_total_wait_s
            ),
        )


def _delay_for(exc: BaseException, attempt: int, policy: RetryPolicy) -> float:
    """Seconds to sleep before attempt `attempt + 1`.

    A server-supplied `Retry-After` wins, plus a second of jitter so that N
    runs throttled by the same shared key do not all wake on the same instant
    and collide again. Otherwise equal-jitter exponential backoff: half the
    window is fixed so a saturated key is not hammered by a run of near-zero
    sleeps, half is random to spread concurrent runs apart.
    """
    hinted = retry_after_seconds(exc)
    if hinted is not None:
        return hinted + random.uniform(0.0, 1.0)
    window = min(policy.cap_s, policy.base_s * (2 ** (attempt - 1)))
    return window / 2.0 + random.uniform(0.0, window / 2.0)


# ---------------------------------------------------------------------------
# Wait budget and observer (process-global: one cca process is one session)
# ---------------------------------------------------------------------------

class ThrottleBudget:
    """Cumulative sleep since the last reset, measured against a cap."""

    def __init__(self) -> None:
        self._waited_s = 0.0

    def reset(self) -> None:
        self._waited_s = 0.0

    @property
    def waited_s(self) -> float:
        return self._waited_s

    def allows(self, seconds: float, limit: float) -> bool:
        return self._waited_s + seconds <= limit

    def consume(self, seconds: float) -> None:
        self._waited_s += seconds


_budget = ThrottleBudget()

_wait_observer: Callable[[dict[str, Any]], None] | None = None


def budget() -> ThrottleBudget:
    """The process-wide wait budget."""
    return _budget


def set_wait_observer(observer: Callable[[dict[str, Any]], None] | None) -> None:
    """Register a callback invoked immediately before each sleep.

    The payload carries `wait_s`, `kind`, `attempt` and `detail`. Called
    before sleeping, not after, so a caller can extend an external deadline
    while the wait is still ahead of it rather than behind.
    """
    global _wait_observer
    _wait_observer = observer


def _notify(payload: dict[str, Any]) -> None:
    if _wait_observer is None:
        return
    try:
        _wait_observer(payload)
    except Exception:
        pass  # observability must never be the reason a call fails


# ---------------------------------------------------------------------------
# The retry loops
# ---------------------------------------------------------------------------

def _prepare_wait(
    exc: BaseException,
    kind: str,
    attempt: int,
    policy: RetryPolicy,
    what: str,
) -> float | None:
    """Charge a sleep to the budget and announce it, or None to give up."""
    delay = _delay_for(exc, attempt, policy)
    if not _budget.allows(delay, policy.max_total_wait_s):
        return None
    _budget.consume(delay)
    _notify({
        "what": what,
        "kind": kind,
        "attempt": attempt,
        "wait_s": round(delay, 2),
        "total_wait_s": round(_budget.waited_s, 2),
        "detail": str(exc)[:200],
    })
    return delay


def _exhausted(
    exc: BaseException, kind: str, attempt: int, what: str, reason: str
) -> LLMUnavailable:
    return LLMUnavailable(
        f"{what}: {reason} after {attempt} attempt(s), "
        f"{_budget.waited_s:.1f}s waited: {str(exc)[:200]}",
        kind=kind,
        attempts=attempt,
        waited_s=_budget.waited_s,
        cause=exc,
    )


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    what: str = "llm.call",
    policy: RetryPolicy | None = None,
) -> T:
    """Await `fn()`, retrying throttled and transient failures.

    `fn` must be replayable — it is called again from scratch on each attempt.
    FATAL failures propagate unchanged and immediately.
    """
    policy = policy or RetryPolicy.from_env()
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as exc:
            kind = classify(exc)
            if kind == FATAL:
                raise
            if attempt >= policy.max_attempts:
                raise _exhausted(exc, kind, attempt, what, "gave up") from exc
            delay = _prepare_wait(exc, kind, attempt, policy, what)
            if delay is None:
                raise _exhausted(exc, kind, attempt, what, "wait budget exhausted") from exc
            await asyncio.sleep(delay)


async def stream_with_retry(
    make_stream: Callable[[], AsyncIterator[T]],
    *,
    what: str = "llm.stream",
    policy: RetryPolicy | None = None,
) -> AsyncIterator[T]:
    """Yield from `make_stream()`, restarting only before the first event.

    Once an event has been handed to the caller the stream is committed: the
    caller has begun accumulating it, so a restart would duplicate output.
    That is the safe half of the trade, and it is also the common case for
    throttling — a 429 is an HTTP status, delivered before any SSE body, so it
    lands squarely in the retryable window.
    """
    policy = policy or RetryPolicy.from_env()
    attempt = 0
    while True:
        attempt += 1
        started = False
        try:
            async for event in make_stream():
                started = True
                yield event
            return
        except Exception as exc:
            if started:
                raise  # partial output already delivered — the caller owns it
            kind = classify(exc)
            if kind == FATAL:
                raise
            if attempt >= policy.max_attempts:
                raise _exhausted(exc, kind, attempt, what, "gave up") from exc
            delay = _prepare_wait(exc, kind, attempt, policy, what)
            if delay is None:
                raise _exhausted(exc, kind, attempt, what, "wait budget exhausted") from exc
            await asyncio.sleep(delay)
