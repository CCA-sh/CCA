"""Tests for project instruction file discovery."""

import pytest

from cca import instructions


@pytest.fixture(autouse=True)
def _no_global_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(instructions, "_GLOBAL_INSTRUCTIONS", tmp_path / "no-global.md")


def test_loads_shared_then_local_file(tmp_path):
    (tmp_path / "AGENTS.md").write_text("shared rule")
    (tmp_path / "AGENTS.local.md").write_text("personal rule")

    block, loaded = instructions.collect(str(tmp_path))

    assert loaded == [str(tmp_path / "AGENTS.md"), str(tmp_path / "AGENTS.local.md")]
    assert block.index("shared rule") < block.index("personal rule")


def test_other_instruction_file_names_are_not_loaded(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("claude rule")
    (tmp_path / "CCA.md").write_text("old cca rule")

    assert instructions.collect(str(tmp_path)) == ("", [])


def test_inner_directory_comes_after_outer(tmp_path):
    inner = tmp_path / "pkg"
    inner.mkdir()
    (tmp_path / "AGENTS.md").write_text("outer rule")
    (inner / "AGENTS.md").write_text("inner rule")

    block, loaded = instructions.collect(str(inner))

    assert loaded == [str(tmp_path / "AGENTS.md"), str(inner / "AGENTS.md")]
    assert block.index("outer rule") < block.index("inner rule")


def test_global_instructions_come_first(tmp_path, monkeypatch):
    global_file = tmp_path / "global.md"
    global_file.write_text("global rule")
    monkeypatch.setattr(instructions, "_GLOBAL_INSTRUCTIONS", global_file)
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("project rule")

    block, loaded = instructions.collect(str(project))

    assert loaded[0] == str(global_file)
    assert block.index("global rule") < block.index("project rule")


def test_at_import_is_expanded(tmp_path):
    (tmp_path / "extra.md").write_text("imported rule")
    (tmp_path / "AGENTS.md").write_text("@extra.md")

    block, _ = instructions.collect(str(tmp_path))

    assert "imported rule" in block
