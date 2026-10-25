"""Background-job control tools for shell commands, monitors, and subagents."""

from __future__ import annotations

import asyncio
import json
import time

from cca.tools.base import Tool


class JobCancelTool(Tool):
    name = "JobCancel"
    description = "Cancel a running background job."
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "The background job ID to cancel."},
        },
        "required": ["job_id"],
    }

    def __init__(self, root_id: str | None = None) -> None:
        self.root_id = root_id

    def bind_root(self, root_id: str) -> None:
        self.root_id = root_id

    async def execute(self, job_id: str) -> str:
        from cca.tools.agent_tool import stop_background_agent

        if stop_background_agent(job_id, root_id=self.root_id):
            return f"Agent {job_id} cancellation requested."

        from cca.tools.bash_tool import stop_background_job

        if stop_background_job(job_id):
            return f"Job {job_id} cancellation requested."

        from cca.tools.monitor_tool import stop_monitor

        if stop_monitor(job_id):
            return f"Monitor {job_id} cancellation requested."

        return f"Job {job_id} not found or already done."


class JobOutputTool(Tool):
    name = "JobOutput"
    description = (
        "Retrieve output from a running or completed background job. "
        "Set block=false to check current status without waiting."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Background job ID."},
            "block": {
                "type": "boolean",
                "description": "Wait for completion (default true).",
                "default": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Max wait time in ms (default 30000).",
                "default": 30000,
            },
        },
        "required": ["job_id", "block", "timeout"],
    }

    def __init__(self, root_id: str | None = None) -> None:
        self.root_id = root_id

    def bind_root(self, root_id: str) -> None:
        self.root_id = root_id

    async def execute(
        self,
        job_id: str,
        block: bool = True,
        timeout: int = 30000,
    ) -> str:
        from cca.tools.agent_tool import get_background_agent, get_persisted_agent
        from cca.tools.bash_tool import get_background_job

        agent = get_background_agent(job_id)
        if (
            agent is not None
            and self.root_id is not None
            and agent.get("root_id") != self.root_id
        ):
            agent = None
        if agent is not None:
            execution = agent.get("task")
            if block and not agent.get("done") and execution is not None:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(execution), timeout=timeout / 1000
                    )
                    agent.update(status=result.status, done=True, result=result)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    agent.update(status="cancelled", done=True)
            result = agent.get("result")
            if result is not None:
                return result.to_tool_result()
            return json.dumps({
                "agent_id": job_id,
                "status": agent.get("status", "running"),
                "description": agent.get("description", ""),
            }, sort_keys=True)

        persisted = get_persisted_agent(job_id, root_id=self.root_id)
        if persisted is not None:
            result = persisted.get("result")
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, sort_keys=True)
            return json.dumps({
                "agent_id": job_id,
                "status": persisted.get("status", "unknown"),
                "description": persisted.get("description", ""),
            }, ensure_ascii=False, sort_keys=True)

        job = get_background_job(job_id)
        if not job:
            return f"Error: background job '{job_id}' not found"

        timeout_s = timeout / 1000
        if block and not job["done"]:
            deadline = time.time() + timeout_s
            while not job["done"] and time.time() < deadline:
                await asyncio.sleep(0.2)

        output = "".join(job.get("output", []))
        status = "done" if job["done"] else "running"
        return_code = job.get("returncode")
        header = f"[Job {job_id} — {status}"
        if return_code is not None:
            header += f", exit {return_code}"
        header += "]\n"
        return header + (output.strip() or "(no output)")
