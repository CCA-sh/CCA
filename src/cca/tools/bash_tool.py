"""Bash tool — execute shell commands with timeout and background support."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
import uuid

from cca.tools.base import Tool

# Global store for background jobs
_background_jobs: dict[str, dict] = {}


class BashTool(Tool):
    name = "Bash"
    description = (
        "Execute a shell command. Returns stdout and stderr. "
        "Set run_in_background=true to start the process without waiting; "
        "a job_id will be returned that you can pass to JobOutput. "
        "Default timeout is 120 seconds (max 600). "
        "Do not use for file reads/writes/searches — use the dedicated tools instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute."},
            "description": {
                "type": "string",
                "description": "Short description of what the command does.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 120000, max 600000).",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Run without waiting for completion (default false).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, allow_background: bool = True):
        self._allow_background = allow_background
        if not allow_background:
            import copy
            schema = copy.deepcopy(BashTool.input_schema)
            schema["properties"].pop("run_in_background", None)
            self.input_schema = schema
            self.description = (
                "Execute a shell command. Returns stdout and stderr. "
                "Default timeout is 120 seconds (max 600). "
                "Do not use for file reads/writes/searches — use the dedicated tools instead."
            )

    async def execute(
        self,
        command: str,
        description: str | None = None,
        timeout: int | None = None,
        run_in_background: bool = False,
    ) -> str:
        timeout_s = min((timeout or 120_000) / 1000, 600)

        if run_in_background:
            if not self._allow_background:
                return "Error: background execution is disabled by feature config."
            return self._start_background(command)

        return await self._run_async(command, timeout_s)

    async def _run_async(self, command: str, timeout_s: float) -> str:
        """Run command as async subprocess, emitting output lines to the event bus."""
        from cca.ui.events import bus

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return f"Error: {e}"

        lines: list[str] = []
        stderr_lines: list[str] = []

        from cca.ui.events import current_tool_id
        tid = current_tool_id.get()

        async def _read_stdout() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                lines.append(line)
                bus.emit("bash_output_line", line=line, is_stderr=False, tool_id=tid)

        async def _read_stderr() -> None:
            assert proc.stderr is not None
            async for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace")
                stderr_lines.append(line)
                bus.emit("bash_output_line", line=line, is_stderr=True, tool_id=tid)

        try:
            await asyncio.wait_for(
                asyncio.gather(_read_stdout(), _read_stderr(), proc.wait()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return f"Error: command timed out after {timeout_s:.0f} seconds"
        except Exception as e:
            return f"Error: {e}"

        output = "".join(lines)
        if stderr_lines:
            output += "\n" + "".join(stderr_lines)
        if proc.returncode != 0:
            output += f"\n[Exit code: {proc.returncode}]"
        return output.strip() or f"[Command completed with exit code {proc.returncode}]"

    def _start_background(self, command: str) -> str:
        job_id = str(uuid.uuid4())[:8]
        output_lines: list[str] = []
        proc_ref: list[subprocess.Popen] = []

        def run():
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc_ref.append(proc)
            _background_jobs[job_id]["pid"] = proc.pid
            for line in proc.stdout:
                output_lines.append(line)
            proc.wait()
            _background_jobs[job_id]["returncode"] = proc.returncode
            _background_jobs[job_id]["done"] = True
            from cca.notifications import push_threadsafe
            push_threadsafe({
                "type": "bash_done",
                "job_id": job_id,
                "command": command,
                "output": "".join(output_lines),
                "returncode": proc.returncode,
            })

        _background_jobs[job_id] = {
            "command": command,
            "output": output_lines,
            "done": False,
            "returncode": None,
            "pid": None,
            "started_at": time.time(),
        }
        t = threading.Thread(target=run, daemon=True)
        t.start()
        _background_jobs[job_id]["thread"] = t
        return f"Background job started. job_id={job_id}"


def get_background_job(job_id: str) -> dict | None:
    return _background_jobs.get(job_id)


def stop_background_job(job_id: str) -> bool:
    job = _background_jobs.get(job_id)
    if not job:
        return False
    pid = job.get("pid")
    if pid:
        try:
            import os
            import signal
            os.kill(pid, signal.SIGTERM)
            job["done"] = True
            return True
        except ProcessLookupError:
            return False
    return False
