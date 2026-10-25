"""ScheduleWakeup — self-pace iterations in /loop dynamic mode."""

from __future__ import annotations

import asyncio

from cca.tools.base import Tool


class ScheduleWakeupTool(Tool):
    name = "ScheduleWakeup"
    description = (
        "Schedule when to resume work in /loop dynamic mode. "
        "Pass delaySeconds between 60-3600. "
        "Omit this call to end the loop. "
        "Under 300s: cache stays warm. Over 300s: cache miss but longer wait. "
        "Use ~1200-1800s for idle ticks with no specific signal to watch."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "delaySeconds": {
                "type": "number",
                "description": "Seconds to wait before resuming (clamped to 60-3600).",
            },
            "prompt": {
                "type": "string",
                "description": "The /loop input to fire on wake-up.",
            },
            "reason": {
                "type": "string",
                "description": "Short sentence explaining the chosen delay.",
            },
        },
        "required": ["delaySeconds", "reason", "prompt"],
    }

    async def execute(
        self,
        delaySeconds: float,
        prompt: str,
        reason: str,
    ) -> str:
        delay = max(60, min(3600, delaySeconds))

        async def _wakeup() -> None:
            await asyncio.sleep(delay)
            from cca import notifications
            await notifications.push({
                "type": "wakeup",
                "prompt": prompt,
                "reason": reason,
            })

        asyncio.create_task(_wakeup())
        return f"Wakeup scheduled in {delay:.0f}s. Reason: {reason}."
