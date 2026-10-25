"""Tests for file-based memory freshness handling."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from cca import memory


_NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is not None else _NOW.replace(tzinfo=None)


def _set_age(path, *, days: int) -> None:
    modified = _NOW.timestamp() - days * 24 * 60 * 60
    os.utime(path, (modified, modified))


def test_recent_memory_has_no_review_note(tmp_path, monkeypatch):
    path = tmp_path / "recent.md"
    path.write_text("recent")
    _set_age(path, days=6)
    monkeypatch.setattr(memory, "datetime", _FixedDateTime)

    assert memory._staleness_warning(path) == ""


def test_old_memory_note_uses_modification_date(tmp_path, monkeypatch):
    path = tmp_path / "old.md"
    path.write_text("old")
    _set_age(path, days=8)
    monkeypatch.setattr(memory, "datetime", _FixedDateTime)

    note = memory._staleness_warning(path)

    assert "last modified on 2026-01-02" in note
    assert "current files, commands, configuration, or dependencies" in note
    assert note.startswith("<system-reminder>")
    assert note.endswith("</system-reminder>\n")


def test_missing_memory_has_no_review_note(tmp_path):
    assert memory._staleness_warning(tmp_path / "missing.md") == ""


def test_manifest_includes_review_note_for_old_memory(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(memory, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(memory, "datetime", _FixedDateTime)

    path = memory.save_memory(
        name="Build workflow",
        description="Repository build notes",
        memory_type="project",
        body="Run the repository build target.",
        cwd=str(tmp_path),
    )
    _set_age(path, days=8)

    manifest = memory.build_memory_manifest(cwd=str(tmp_path))

    assert "Memory review note" in manifest
    assert "last modified on 2026-01-02" in manifest
    assert "Run the repository build target." in manifest
