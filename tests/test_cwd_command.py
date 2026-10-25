"""Regression tests for the interactive /cwd command."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock

from rich.console import Console

from cca.permissions import PermissionManager
from cca.repl import Repl
from cca.tools.registry import ToolRegistry


def _make_repl(cwd: str) -> tuple[Repl, io.StringIO]:
    repl = Repl.__new__(Repl)
    repl.cwd = cwd
    repl.permissions = PermissionManager(cwd=cwd)
    repl.registry = ToolRegistry()
    repl._ui = MagicMock()
    output = io.StringIO()
    repl._console = Console(file=output, force_terminal=False, width=160)
    return repl, output


async def test_cwd_accepts_shell_quoted_absolute_path(tmp_path, monkeypatch):
    start = tmp_path / "start"
    target = tmp_path / "target folder"
    start.mkdir()
    target.mkdir()
    monkeypatch.chdir(start)
    repl, output = _make_repl(str(start))

    handled = await repl._handle_slash(f"/cwd '{target}'")

    assert handled is True
    assert os.getcwd() == str(target.resolve())
    assert repl.cwd == str(target.resolve())
    assert repl.permissions._cwd == target.resolve()
    repl._ui.update_cwd.assert_called_once_with(str(target.resolve()))
    assert "Working directory:" in output.getvalue()


async def test_cwd_resolves_relative_path_before_chdir(tmp_path, monkeypatch):
    start = tmp_path / "start"
    target = start / "child"
    target.mkdir(parents=True)
    monkeypatch.chdir(start)
    repl, _ = _make_repl(str(start))

    handled = await repl._handle_slash("/cwd child")

    assert handled is True
    assert repl.cwd == str(target.resolve())
    assert repl.permissions._cwd == target.resolve()


async def test_cwd_reports_invalid_path_without_mutating_state(tmp_path, monkeypatch):
    start = tmp_path / "start"
    start.mkdir()
    monkeypatch.chdir(start)
    repl, output = _make_repl(str(start))
    missing = tmp_path / "missing"

    handled = await repl._handle_slash(f"/cwd '{missing}'")

    assert handled is True
    assert os.getcwd() == str(start.resolve())
    assert repl.cwd == str(start.resolve())
    assert repl.permissions._cwd == start.resolve()
    repl._ui.update_cwd.assert_not_called()
    assert "Cannot change working directory:" in output.getvalue()
