"""Tests for the work-item and background-job tool surfaces."""

from __future__ import annotations

import asyncio
import json

from click.testing import CliRunner


def test_work_item_lifecycle(tmp_path, monkeypatch):
    from cca.tools import work_tools

    monkeypatch.setattr(work_tools, "WORK_ITEMS_FILE", tmp_path / "work_items.json")
    monkeypatch.setattr(work_tools, "_cache", None)

    created = asyncio.run(work_tools.WorkCreateTool().execute(
        subject="Ship release",
        description="Complete the public release.",
        activeForm="Shipping release",
    ))
    assert created == "Work item 1 created: Ship release"

    detail = json.loads(asyncio.run(work_tools.WorkGetTool().execute(workId="1")))
    assert detail["subject"] == "Ship release"
    assert detail["status"] == "pending"

    listed = asyncio.run(work_tools.WorkListTool().execute())
    assert "Ship release" in listed

    updated = asyncio.run(work_tools.WorkUpdateTool().execute(
        workId="1", status="completed"
    ))
    assert updated == "Work item 1 updated: Ship release [completed]"

    deleted = asyncio.run(work_tools.WorkUpdateTool().execute(
        workId="1", status="deleted"
    ))
    assert deleted == "Work item 1 deleted."
    assert asyncio.run(work_tools.WorkListTool().execute()) == "No work items."


def test_job_cancel_can_stop_monitor(monkeypatch):
    from cca.tools.agent_tool import stop_background_agent
    from cca.tools.bash_tool import stop_background_job
    from cca.tools.job_tools import JobCancelTool
    from cca.tools.monitor_tool import stop_monitor

    monkeypatch.setattr(
        "cca.tools.agent_tool.stop_background_agent", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("cca.tools.bash_tool.stop_background_job", lambda job_id: False)
    monkeypatch.setattr("cca.tools.monitor_tool.stop_monitor", lambda job_id: True)

    result = asyncio.run(JobCancelTool(root_id="root").execute(job_id="monitor-1"))
    assert result == "Monitor monitor-1 cancellation requested."

    # Keep imports above exercised so an accidental public helper rename is caught.
    assert callable(stop_background_agent)
    assert callable(stop_background_job)
    assert callable(stop_monitor)


def test_background_bash_round_trips_through_job_output():
    from cca.tools.bash_tool import BashTool
    from cca.tools.job_tools import JobOutputTool

    async def exercise() -> None:
        started = await BashTool().execute(
            command="printf job-output-ok", run_in_background=True
        )
        assert started.startswith("Background job started. job_id=")
        job_id = started.rsplit("=", 1)[1]

        output = await JobOutputTool().execute(
            job_id=job_id, block=True, timeout=5000
        )
        assert output.startswith(f"[Job {job_id} — done, exit 0]")
        assert "job-output-ok" in output

    asyncio.run(exercise())


def test_cli_exposes_only_new_unsafe_permission_alias():
    from cca.cli import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--unsafe-auto-approve" in result.output
    assert "--dangerously-skip-permissions" not in result.output
