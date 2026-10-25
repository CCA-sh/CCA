"""Persistent work-item tools for planning and progress tracking.

Work items are persisted to ``~/.cca/work_items.json`` and protected by an
exclusive file lock so parallel subagents can safely share the work list.
On platforms without ``fcntl`` (Windows), a threading lock is used instead.
"""

from __future__ import annotations

import json
import time
from typing import Any

from cca.config import CONFIG_DIR
from cca.tools.base import Tool

WORK_ITEMS_FILE = CONFIG_DIR / "work_items.json"


try:
    import fcntl as _fcntl

    def _lock(fh, exclusive: bool) -> None:
        _fcntl.flock(fh, _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH)

    def _unlock(fh) -> None:
        _fcntl.flock(fh, _fcntl.LOCK_UN)

except ImportError:
    import threading as _threading

    _file_lock = _threading.Lock()

    def _lock(fh, exclusive: bool) -> None:
        _file_lock.acquire()

    def _unlock(fh) -> None:
        _file_lock.release()


def _load() -> dict[str, dict]:
    if not WORK_ITEMS_FILE.exists():
        return {}
    try:
        with open(WORK_ITEMS_FILE, "r") as fh:
            _lock(fh, exclusive=False)
            try:
                return json.load(fh) or {}
            except (json.JSONDecodeError, ValueError):
                return {}
            finally:
                _unlock(fh)
    except OSError:
        return {}


def _save(work_items: dict[str, dict]) -> None:
    WORK_ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORK_ITEMS_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as fh:
            _lock(fh, exclusive=True)
            try:
                json.dump(work_items, fh, indent=2)
            finally:
                _unlock(fh)
        tmp.replace(WORK_ITEMS_FILE)
    except OSError:
        pass


def _new_id(work_items: dict) -> str:
    return str(len(work_items) + 1)


_cache: dict[str, dict] | None = None


def _get_work_items() -> dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = _load()
    return _cache


def _put_work_items(work_items: dict[str, dict]) -> None:
    global _cache
    _cache = work_items
    _save(work_items)


class _WorkItemsProxy:
    def __iter__(self):
        return iter(_get_work_items())

    def items(self):
        return _get_work_items().items()

    def __bool__(self):
        return bool(_get_work_items())

    def get(self, key, default=None):
        return _get_work_items().get(key, default)


_work_items = _WorkItemsProxy()


class WorkCreateTool(Tool):
    name = "WorkCreate"
    description = (
        "Create an item in the session work list. Use this to track progress on "
        "multi-step work. Returns the work item ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Brief work-item title in imperative form.",
            },
            "description": {
                "type": "string",
                "description": "What needs to be done.",
            },
            "activeForm": {
                "type": "string",
                "description": "Present continuous form shown while in progress.",
            },
            "metadata": {
                "type": "object",
                "description": "Arbitrary key-value metadata.",
            },
        },
        "required": ["subject", "description"],
    }

    async def execute(
        self,
        subject: str,
        description: str,
        activeForm: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        work_items = _load()
        work_id = _new_id(work_items)
        work_items[work_id] = {
            "id": work_id,
            "subject": subject,
            "description": description,
            "activeForm": activeForm or subject,
            "status": "pending",
            "owner": None,
            "metadata": metadata or {},
            "blocks": [],
            "blockedBy": [],
            "created_at": time.time(),
        }
        _put_work_items(work_items)
        return f"Work item {work_id} created: {subject}"


class WorkGetTool(Tool):
    name = "WorkGet"
    description = "Retrieve full details of a work item by ID."
    input_schema = {
        "type": "object",
        "properties": {
            "workId": {"type": "string", "description": "The work item ID to retrieve."},
        },
        "required": ["workId"],
    }

    async def execute(self, workId: str) -> str:
        work_item = _load().get(workId)
        if not work_item:
            return f"Error: work item '{workId}' not found"
        return json.dumps(work_item, indent=2)


class WorkListTool(Tool):
    name = "WorkList"
    description = "List all work items and their statuses."
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> str:
        work_items = _load()
        if not work_items:
            return "No work items."
        lines = []
        for work_id, work_item in work_items.items():
            blocked = (
                f" [blocked by: {', '.join(work_item['blockedBy'])}]"
                if work_item.get("blockedBy")
                else ""
            )
            lines.append(
                f"[{work_id}] {work_item['status'].upper():12} "
                f"{work_item['subject']}{blocked}"
            )
        return "\n".join(lines)


class WorkUpdateTool(Tool):
    name = "WorkUpdate"
    description = (
        "Update a work item's status, subject, description, dependencies, or owner. "
        "Status values: pending | in_progress | completed | deleted."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workId": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "deleted"],
            },
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "activeForm": {"type": "string"},
            "owner": {"type": "string"},
            "metadata": {"type": "object"},
            "addBlocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Work item IDs that this item blocks.",
            },
            "addBlockedBy": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Work item IDs that must complete before this one.",
            },
        },
        "required": ["workId"],
    }

    async def execute(self, workId: str, **kwargs: Any) -> str:
        work_items = _load()
        work_item = work_items.get(workId)
        if not work_item:
            return f"Error: work item '{workId}' not found"

        if kwargs.get("status") == "deleted":
            del work_items[workId]
            _put_work_items(work_items)
            return f"Work item {workId} deleted."

        for field in ("status", "subject", "description", "activeForm", "owner"):
            if field in kwargs and kwargs[field] is not None:
                work_item[field] = kwargs[field]

        if "metadata" in kwargs and kwargs["metadata"]:
            for key, value in kwargs["metadata"].items():
                if value is None:
                    work_item["metadata"].pop(key, None)
                else:
                    work_item["metadata"][key] = value

        if "addBlocks" in kwargs:
            work_item["blocks"] = list(
                set(work_item["blocks"] + (kwargs["addBlocks"] or []))
            )

        if "addBlockedBy" in kwargs:
            work_item["blockedBy"] = list(
                set(work_item["blockedBy"] + (kwargs["addBlockedBy"] or []))
            )

        _put_work_items(work_items)
        return (
            f"Work item {workId} updated: {work_item['subject']} "
            f"[{work_item['status']}]"
        )
