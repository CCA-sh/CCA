"""Turn a finished `--print` answer into a schema-conformant JSON object.

Why a second call rather than constraining the agent's own reply: the agentic
loop needs tools, and a forced schema is incompatible with tool use on most
providers. It also mixes two jobs in one token budget — reasoning about the
repo *and* emitting JSON — which is how a reasoning model ends up spending its
whole allowance on thinking and returning an empty reply.

So the loop runs normally, and this module reformats its final text in one
extra call with no tools and thinking off. That call has nothing to reason
about, so it needs a few hundred tokens and cannot starve.

Nothing here fabricates. When the reply cannot be made to satisfy the schema
after `MAX_ATTEMPTS`, `StructuredOutputError` is raised and the caller decides.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cca.llm.base import Message, TextContent

MAX_ATTEMPTS = 3

# Providers report a request they declined to answer as an empty reply with one of
# these stop reasons (`content_filter` from OpenAI-compatible routes, `refusal` from Anthropic).
_REFUSAL_STOP_REASONS = frozenset({"content_filter", "refusal"})

# Some models decline to restate a field whose name reads like reasoning, even when
# the text was written moments earlier by the same model. Renaming it for the retry
# fixes that without changing what the caller receives.
_REASONING_LIKE_NAME = re.compile(r"thought|reasoning|think", re.I)

# Small and fixed: this call reformats text that already exists.
FORMAT_MAX_TOKENS = 4096

_SYSTEM = (
    "You convert an assistant's answer into a single JSON object.\n"
    "Reply with ONLY that object — no prose before or after it, no markdown "
    "fences, no explanation. Preserve the answer's meaning and wording; do not "
    "summarize it, add to it, or answer the question yourself."
)

_INSTRUCTION = (
    "Convert the following answer into a single JSON object matching this "
    "JSON Schema.\n\nSCHEMA:\n{schema}\n\nANSWER:\n{answer}"
)

_RETRY = (
    "\n\nYour previous attempt was rejected: {problems}. "
    "Return only the corrected JSON object."
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class StructuredOutputError(RuntimeError):
    """No attempt produced an object satisfying the schema."""

    def __init__(self, attempts: int, problems: list[str], raw_text: str = ""):
        self.attempts = attempts
        self.problems = list(problems)
        self.raw_text = raw_text
        super().__init__(
            f"could not produce schema-conformant output after {attempts} "
            f"attempt(s): {'; '.join(self.problems) or 'no JSON object found'}"
        )


def extract_json_object(text: str) -> dict | None:
    """Best-effort parse of a JSON object out of a model reply.

    Handles the three shapes that actually occur: a bare object, an object in a
    markdown fence, and an object with prose around it. Returns None rather
    than raising — the caller retries.
    """
    if not text:
        return None
    candidates = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def validate_against_schema(obj: Any, schema: dict) -> list[str]:
    """Check `obj` against the parts of JSON Schema worth enforcing here.

    Deliberately not a full JSON Schema implementation — no `$ref`, no
    composition keywords. It covers required fields, types, enums and numeric
    bounds, which is what a flat response schema uses and what actually goes
    wrong. A schema feature it does not understand is not treated as a
    violation, so this never rejects something a real validator would accept.
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["response was not a JSON object"]
    if not isinstance(schema, dict):
        return problems

    for name in schema.get("required", []) or []:
        if name not in obj:
            problems.append(f"required field `{name}` is missing")

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return problems

    if schema.get("additionalProperties") is False:
        for key in obj:
            if key not in properties:
                problems.append(f"unexpected field `{key}`")

    for name, spec in properties.items():
        if name not in obj or not isinstance(spec, dict):
            continue
        problems.extend(_check_value(name, obj[name], spec))
    return problems


_TYPES: dict[str, Any] = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
    "null": type(None),
}


def _check_value(name: str, value: Any, spec: dict) -> list[str]:
    problems: list[str] = []
    expected = spec.get("type")
    types = [expected] if isinstance(expected, str) else list(expected or [])
    if types:
        # `bool` is a subclass of `int`, so a boolean would satisfy "number"
        # and "integer" without this guard.
        ok = False
        for type_name in types:
            python_type = _TYPES.get(type_name)
            if python_type is None:
                ok = True  # unknown type keyword: don't invent a violation
                break
            if type_name in ("integer", "number") and isinstance(value, bool):
                continue
            if isinstance(value, python_type):
                ok = True
                break
        if not ok:
            problems.append(f"`{name}` must be of type {' or '.join(types)}")
            return problems

    choices = spec.get("enum")
    if isinstance(choices, list) and value not in choices:
        problems.append(f"`{name}` must be one of {choices}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum, maximum = spec.get("minimum"), spec.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            problems.append(f"`{name}` must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            problems.append(f"`{name}` must be <= {maximum}")

    if isinstance(value, list):
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(value):
                problems.extend(_check_value(f"{name}[{index}]", item, item_spec))
    return problems


def _neutral_aliases(schema: dict) -> dict[str, str]:
    """Neutral names for top-level properties whose names read like reasoning."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return {}
    taken = set(properties)
    aliases: dict[str, str] = {}
    for name in properties:
        if not _REASONING_LIKE_NAME.search(name):
            continue
        alias = _REASONING_LIKE_NAME.sub("notes", name)
        while alias in taken:
            alias += "_"
        aliases[name] = alias
        taken.add(alias)
    return aliases


def _rename_schema_properties(schema: dict, mapping: dict[str, str]) -> dict:
    renamed = dict(schema)
    renamed["properties"] = {mapping.get(k, k): v for k, v in schema["properties"].items()}
    if "required" in schema:
        renamed["required"] = [mapping.get(k, k) for k in schema["required"]]
    return renamed


def _rename_answer_keys(answer: str, mapping: dict[str, str]) -> str:
    """Use the alias names inside the answer text too, so the model sees one vocabulary."""
    try:
        parsed = json.loads(answer)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return json.dumps({mapping.get(k, k): v for k, v in parsed.items()})
    for name, alias in mapping.items():
        answer = answer.replace(f'"{name}"', f'"{alias}"')
    return answer


async def format_as_schema(
    backend,
    answer: str,
    schema: dict,
    *,
    think_level: str,
) -> dict:
    """Reformat `answer` into an object satisfying `schema`.

    Returns the parsed object, normalized text, mode, attempt count, and the
    capability-resolved thinking level used by the formatting call.
    Raises `StructuredOutputError` when every attempt fails.
    """
    problems: list[str] = []
    last_text = ""
    mode_used = None
    aliases: dict[str, str] = {}
    request_schema, request_answer = schema, answer
    retry_note = ""

    for attempt in range(MAX_ATTEMPTS):
        message = _INSTRUCTION.format(schema=json.dumps(request_schema, indent=2), answer=request_answer) + retry_note
        response = await backend.complete(
            messages=[Message(role="user", content=[TextContent(text=message)])],
            tools=[],
            system=_SYSTEM,
            max_tokens=FORMAT_MAX_TOKENS,
            think_level=think_level,
            output_schema=request_schema,
        )
        mode_used = getattr(response, "mode_used", None) or mode_used
        last_text = response.text
        if not aliases and getattr(response, "stop_reason", None) in _REFUSAL_STOP_REASONS:
            aliases = _neutral_aliases(schema)
            if aliases:
                request_schema = _rename_schema_properties(schema, aliases)
                request_answer = _rename_answer_keys(answer, aliases)
                retry_note = ""
                problems = [f"the provider declined the request ({response.stop_reason})"]
                continue
        parsed = extract_json_object(last_text)
        if parsed is None:
            problems = ["the reply was not a JSON object"]
            retry_note = _RETRY.format(problems="; ".join(problems))
            continue
        if aliases:
            parsed = {next((n for n, a in aliases.items() if a == k), k): v for k, v in parsed.items()}
        problems = validate_against_schema(parsed, schema)
        if not problems:
            result = {
                "object": parsed,
                "text": json.dumps(parsed),
                "mode_used": mode_used or "prompt_only",
                "attempts": attempt + 1,
                "think_level": think_level,
            }
            if aliases:
                result["field_aliases"] = aliases
            return result
        retry_note = _RETRY.format(problems="; ".join(problems))

    raise StructuredOutputError(MAX_ATTEMPTS, problems, last_text)
