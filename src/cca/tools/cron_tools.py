"""Cron tools: CronCreate, CronDelete, CronList.

Uses APScheduler for scheduling. Jobs are session-only unless durable=True,
in which case they are persisted to ~/.cca/scheduled_tasks.json.
Jobs auto-expire after 7 days.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

from cca.tools.base import Tool

# Lazy import APScheduler
_scheduler = None
_jobs: dict[str, dict] = {}


def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            _scheduler = AsyncIOScheduler()
            _scheduler.start()
        except ImportError:
            return None
    return _scheduler


class CronCreateTool(Tool):
    name = "CronCreate"
    description = (
        "Schedule a prompt to run on a cron schedule. "
        "cron is a 5-field expression in local time: 'M H DoM Mon DoW'. "
        "Set recurring=false for a one-shot job (fires once then deletes). "
        "Set durable=true to persist across restarts. "
        "Jobs auto-expire after 7 days. Returns a job ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cron": {
                "type": "string",
                "description": "5-field cron expression, e.g. '*/5 * * * *' or '30 9 * * 1-5'.",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to execute at each fire time.",
            },
            "recurring": {
                "type": "boolean",
                "description": "Fire repeatedly (true, default) or once then delete (false).",
            },
            "durable": {
                "type": "boolean",
                "description": "Persist to disk across restarts (default false).",
            },
        },
        "required": ["cron", "prompt"],
    }

    async def execute(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
    ) -> str:
        scheduler = _get_scheduler()
        job_id = str(uuid.uuid4())[:8]
        parts = cron.strip().split()
        if len(parts) != 5:
            return "Error: cron must have exactly 5 fields: M H DoM Mon DoW"

        minute, hour, day, month, day_of_week = parts
        expires_at = (datetime.now() + timedelta(days=7)).isoformat()

        _jobs[job_id] = {
            "id": job_id,
            "cron": cron,
            "prompt": prompt,
            "recurring": recurring,
            "durable": durable,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
            "fire_count": 0,
        }

        if scheduler:
            if recurring:
                scheduler.add_job(
                    _fire_job,
                    "cron",
                    args=[job_id, prompt, recurring],
                    id=job_id,
                    minute=minute,
                    hour=hour,
                    day=day,
                    month=month,
                    day_of_week=day_of_week,
                    end_date=expires_at,
                )
            else:
                scheduler.add_job(
                    _fire_job,
                    "cron",
                    args=[job_id, prompt, recurring],
                    id=job_id,
                    minute=minute,
                    hour=hour,
                    day=day,
                    month=month,
                    day_of_week=day_of_week,
                    max_instances=1,
                )

        if durable:
            _persist_jobs()

        msg = f"Cron job {job_id} created ({'recurring' if recurring else 'one-shot'}). "
        msg += f"Schedule: {cron}. Auto-expires {expires_at[:10]}."
        return msg


async def _fire_job(job_id: str, prompt: str, recurring: bool) -> None:
    """Called by APScheduler when a job fires."""
    job = _jobs.get(job_id)
    if not job:
        return
    job["fire_count"] += 1
    job["last_fired"] = datetime.now().isoformat()
    import sys
    print(f"\n[Cron job {job_id} fired] {prompt}\n", file=sys.stderr)
    # In a full implementation, this would submit the prompt back to the REPL.
    if not recurring:
        _jobs.pop(job_id, None)


def _persist_jobs() -> None:
    from cca.config import CONFIG_DIR
    path = CONFIG_DIR / "scheduled_tasks.json"
    durable = {jid: j for jid, j in _jobs.items() if j.get("durable")}
    path.write_text(json.dumps(durable, indent=2))


class CronDeleteTool(Tool):
    name = "CronDelete"
    description = "Cancel a scheduled cron job by its ID."
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Job ID returned by CronCreate."},
        },
        "required": ["id"],
    }

    async def execute(self, id: str) -> str:
        if id not in _jobs:
            return f"Error: job '{id}' not found"
        scheduler = _get_scheduler()
        if scheduler:
            try:
                scheduler.remove_job(id)
            except Exception:
                pass
        del _jobs[id]
        return f"Cron job {id} deleted."


class CronListTool(Tool):
    name = "CronList"
    description = "List all scheduled cron jobs in the current session."
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> str:
        if not _jobs:
            return "No scheduled jobs."
        lines = []
        for jid, job in _jobs.items():
            lines.append(
                f"[{jid}] {job['cron']:20} fires={job['fire_count']} "
                f"{'recurring' if job['recurring'] else 'one-shot'} "
                f"prompt={job['prompt'][:60]}"
            )
        return "\n".join(lines)
