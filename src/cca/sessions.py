"""Session persistence — save and restore conversation history between runs.

Sessions are stored under ~/.cca/sessions/<id>.json as flat Anthropic schema arrays.
An index at ~/.cca/history.jsonl maps session-id to metadata for discovery.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cca.llm.base import Message

logger = logging.getLogger(__name__)

from cca.paths import SESSIONS_DIR, HISTORY_FILE  # noqa: E402


def _now_ms() -> int:
    return int(time.time() * 1000)


def generate_session_id(
    parent_id: str | None = None,
    is_compaction: bool = False,
    current_id: str | None = None,
) -> str:
    """Generate a descriptive session ID.
    
    Format: <YYYYMMDD>-<HHMMSS>-<hex_id>-comp<N>[-sub-<hex>]
    """
    if is_compaction and current_id:
        m = re.search(r'-comp(\d+)', current_id)
        if m:
            comp_n = int(m.group(1)) + 1
            return re.sub(r'-comp\d+', f'-comp{comp_n}', current_id)
        else:
            return f"{current_id}-comp1"

    if parent_id:
        sub_hex = uuid.uuid4().hex[:6]
        return f"{parent_id}-sub-{sub_hex}"

    now = datetime.now()
    date_str = now.strftime("%Y%m%d-%H%M%S")
    base_hex = uuid.uuid4().hex[:6]
    return f"{date_str}-{base_hex}-comp0"


def _extract_display(messages: "list[Message]") -> str:
    from cca.llm.base import TextContent
    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                text = msg.content.strip()
                if text:
                    return text if len(text) <= 60 else text[:57] + "..."
            else:
                for block in msg.content:
                    if isinstance(block, TextContent) and block.text.strip():
                        text = block.text.strip()
                        # Strip out compaction summary blocks to find real user text if needed
                        if not text.startswith("Conversation compacted") and not text.startswith("<compaction_summary>"):
                            return text if len(text) <= 60 else text[:57] + "..."
    return "New Session"


def save_session(
    session_id: str,
    cwd: str,
    messages: "list[Message]",
    display_override: str | None = None,
    parent_id: str | None = None,
    system_prompt: str | None = None,
    token_counts: dict | None = None,
    compacted_from_index: int | None = None,
    features: dict | None = None,
    randomization: dict | None = None,
    settings: dict | None = None,
    state: dict | None = None,
) -> None:
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        msg_file = SESSIONS_DIR / f"{session_id}.json"
        payload: dict = {"version": 1, "messages": [m.to_dict() for m in messages]}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if token_counts:
            payload["token_counts"] = token_counts
        if compacted_from_index is not None:
            payload["compacted_from_index"] = compacted_from_index
        if features is not None:
            payload["features"] = features
        if settings is not None:
            payload["settings"] = settings
        if state is not None:
            payload["state"] = state
        # Omitted entirely when randomization is off, so an ordinary run's session
        # file stays byte-identical to what earlier versions wrote.
        if randomization is not None:
            payload["randomization"] = randomization
        with open(msg_file, "w") as f:
            json.dump(payload, f, indent=2)

        display = display_override or _extract_display(messages)

        entry = {
            "display": display,
            "timestamp": _now_ms(),
            "project": cwd,
            "sessionId": session_id,
        }
        if parent_id:
            entry["parentSessionId"] = parent_id

        lines = []
        found = False
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("sessionId") == session_id:
                            entry["timestamp"] = data.get("timestamp", entry["timestamp"])
                            lines.append(json.dumps(entry) + "\n")
                            found = True
                        else:
                            lines.append(line)
                    except json.JSONDecodeError:
                        lines.append(line)

        if not found:
            lines.append(json.dumps(entry) + "\n")

        with open(HISTORY_FILE, "w") as f:
            f.writelines(lines)
            
    except Exception as e:
        logger.debug("save_session failed for %s: %s", session_id, e)


def mark_compacted_in_history(session_id: str) -> None:
    """Update history.jsonl to mark the given session_id as compacted."""
    if not HISTORY_FILE.exists():
        return
    lines = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("sessionId") == session_id:
                    if not data.get("display", "").startswith("[Compacted]"):
                        data["display"] = f"[Compacted] {data.get('display', '')}"
                lines.append(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                lines.append(line)
    with open(HISTORY_FILE, "w") as f:
        f.writelines(lines)


def _load_flat_json(path: Path) -> "list[Message]":
    """Load a flat-JSON session file ({version, messages} or bare array)."""
    from cca.llm.base import Message
    with open(path, "r") as f:
        data = json.load(f)
    msg_list = data if isinstance(data, list) else data.get("messages", [])
    messages: list[Message] = []
    for msg_dict in msg_list:
        try:
            messages.append(Message.from_dict(msg_dict))
        except Exception as e:
            logger.warning("Skipping malformed message in %s: %s", path, e)
    return messages


def load_session(session_id: str) -> "list[Message]":
    primary = SESSIONS_DIR / f"{session_id}.json"
    if primary.exists():
        return _load_flat_json(primary)
    raise FileNotFoundError(f"Session '{session_id}' not found at {primary}")


def load_session_state(session_id: str) -> tuple["list[Message]", dict]:
    """Load messages plus additive resumable state when the primary snapshot has it."""
    primary = SESSIONS_DIR / f"{session_id}.json"
    messages = load_session(session_id)
    if not primary.exists():
        return messages, {}
    try:
        payload = json.loads(primary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return messages, {}
    state = payload.get("state") if isinstance(payload, dict) else None
    return messages, state if isinstance(state, dict) else {}


def _read_history(path: Path) -> list[dict]:
    """Read all valid entries from a history.jsonl file."""
    if not path.exists():
        return []
    entries = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def find_latest_for_cwd(cwd: str) -> str | None:
    cwd_resolved = str(Path(cwd).resolve())
    matches = [
        e for e in _read_history(HISTORY_FILE)
        if str(Path(e.get("project", "")).resolve()) == cwd_resolved
    ]
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return matches[0]["sessionId"]


def list_sessions(limit: int = 20) -> list[dict]:
    entries = _read_history(HISTORY_FILE)
    entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return entries[:limit]
