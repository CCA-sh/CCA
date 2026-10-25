"""Monitor tool — stream stdout from a long-running command as events."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid

from cca.tools.base import Tool

_monitors: dict[str, dict] = {}


class MonitorTool(Tool):
    name = "Monitor"
    description = (
        "Start a background monitor that streams stdout events from a shell command. "
        "Each line of stdout becomes an event. Use for log tailing, file watching, "
        "or polling loops. "
        "persistent=true runs until you call JobCancel. "
        "Returns a monitor ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run. Each stdout line is an event.",
            },
            "description": {
                "type": "string",
                "description": "Short label for this monitor (shown in notifications).",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Kill after this many ms (max 3600000, default 300000).",
                "default": 300000,
            },
            "persistent": {
                "type": "boolean",
                "description": "Run for full session lifetime (default false).",
                "default": False,
            },
        },
        "required": ["description", "timeout_ms", "persistent", "command"],
    }

    async def execute(
        self,
        command: str,
        description: str,
        timeout_ms: int = 300_000,
        persistent: bool = False,
    ) -> str:
        monitor_id = str(uuid.uuid4())[:8]
        timeout_s = min(timeout_ms / 1000, 3600) if not persistent else None
        events: list[str] = []

        def run():
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _monitors[monitor_id]["pid"] = proc.pid
            start = time.time()
            try:
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    events.append(line)
                    from cca.notifications import push_threadsafe
                    push_threadsafe({
                        "type": "monitor_event",
                        "monitor_id": monitor_id,
                        "description": description,
                        "line": line,
                    })
                    if timeout_s and (time.time() - start) > timeout_s:
                        proc.terminate()
                        break
            finally:
                proc.wait()
                _monitors[monitor_id]["done"] = True
                _monitors[monitor_id]["returncode"] = proc.returncode

        _monitors[monitor_id] = {
            "id": monitor_id,
            "command": command,
            "description": description,
            "events": events,
            "done": False,
            "pid": None,
            "returncode": None,
            "started_at": time.time(),
        }
        t = threading.Thread(target=run, daemon=True)
        t.start()
        _monitors[monitor_id]["thread"] = t

        return (
            f"Monitor {monitor_id} started: {description}. "
            f"{'Persistent.' if persistent else f'Timeout: {timeout_ms}ms.'} "
            "Events will print as they arrive. Use JobCancel to cancel early."
        )


def stop_monitor(monitor_id: str) -> bool:
    mon = _monitors.get(monitor_id)
    if not mon:
        return False
    pid = mon.get("pid")
    if pid:
        try:
            import os
            import signal
            os.kill(pid, signal.SIGTERM)
            mon["done"] = True
            return True
        except ProcessLookupError:
            return False
    return False
