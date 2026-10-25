"""CLI startup and invocation smoke tests.

These tests verify the full import chain (cli → ui → components → input_app)
and basic CLI behaviour without requiring API keys or a real LLM.
"""

from __future__ import annotations

from click.testing import CliRunner

from cca.cli import main


def test_help_exits_cleanly():
    """--help must exit 0 and mention key options."""
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--print" in result.output
    assert "--model" in result.output


def test_config_help_exits_cleanly():
    """config subcommand --help must exit 0."""
    result = CliRunner().invoke(main, ["config", "--help"])
    assert result.exit_code == 0, result.output


def test_sessions_help_exits_cleanly():
    """sessions subcommand --help must exit 0."""
    result = CliRunner().invoke(main, ["sessions", "--help"])
    assert result.exit_code == 0, result.output


def test_unknown_flag_exits_nonzero():
    """Unrecognised flags must produce a non-zero exit code."""
    result = CliRunner().invoke(main, ["--not-a-real-flag"])
    assert result.exit_code != 0


def test_sessions_list_exits_cleanly(tmp_path, monkeypatch):
    """sessions list must exit 0 even when no sessions exist."""
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(main, ["sessions", "list"])
    assert result.exit_code == 0, result.output
