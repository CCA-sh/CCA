"""Tests for `cca skills` — installing skills from GitHub or local git repos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from cca.skills_cli import (
    parse_skill_ref,
    discover_skills,
    install_skill,
    skills,
)


# ---------------------------------------------------------------------------
# parse_skill_ref
# ---------------------------------------------------------------------------

def test_parse_owner_repo():
    ref = parse_skill_ref("anthropics/skills")
    assert ref.clone_url == "https://github.com/anthropics/skills.git"
    assert ref.skill_name is None
    assert ref.display == "anthropics/skills"


def test_parse_owner_repo_with_skill():
    ref = parse_skill_ref("anthropics/skills@frontend-design")
    assert ref.clone_url == "https://github.com/anthropics/skills.git"
    assert ref.skill_name == "frontend-design"


def test_parse_full_github_url():
    ref = parse_skill_ref("https://github.com/obra/superpowers")
    assert ref.clone_url == "https://github.com/obra/superpowers.git"
    assert ref.skill_name is None


def test_parse_full_github_url_with_git_suffix_and_skill():
    ref = parse_skill_ref("https://github.com/obra/superpowers.git@brainstorming")
    assert ref.clone_url == "https://github.com/obra/superpowers.git"
    assert ref.skill_name == "brainstorming"


def test_parse_local_path(tmp_path):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    ref = parse_skill_ref(f"{repo}@commit")
    assert ref.clone_url == str(repo)
    assert ref.skill_name == "commit"


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_skill_ref("not-a-valid-ref")


# ---------------------------------------------------------------------------
# discover_skills
# ---------------------------------------------------------------------------

def _make_skill(root: Path, name: str, description: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True)
    fm = f"---\nname: {name}\ndescription: {description}\n---\n" if description else ""
    (d / "SKILL.md").write_text(f"{fm}Do the {name} thing.")
    return d


def test_discover_top_level_skills(tmp_path):
    _make_skill(tmp_path, "commit", "make a commit")
    _make_skill(tmp_path, "review")
    found = discover_skills(tmp_path)
    assert sorted(s.name for s in found) == ["commit", "review"]


def test_discover_nested_skills(tmp_path):
    _make_skill(tmp_path / "skills", "deploy", "deploy things")
    found = discover_skills(tmp_path)
    assert [s.name for s in found] == ["deploy"]
    assert found[0].description == "deploy things"


def test_discover_skips_git_dir(tmp_path):
    _make_skill(tmp_path / ".git", "phantom")
    _make_skill(tmp_path, "real")
    found = discover_skills(tmp_path)
    assert [s.name for s in found] == ["real"]


def test_discover_empty_repo(tmp_path):
    (tmp_path / "README.md").write_text("no skills here")
    assert discover_skills(tmp_path) == []


# ---------------------------------------------------------------------------
# install_skill
# ---------------------------------------------------------------------------

def test_install_copies_and_records_source(tmp_path):
    src = _make_skill(tmp_path / "repo", "commit", "make a commit")
    (src / "references").mkdir()
    (src / "references" / "extra.md").write_text("extra")
    dest_root = tmp_path / "skills"

    installed = install_skill(src, dest_root, source_ref="owner/repo@commit")

    assert (installed / "SKILL.md").read_text() == (src / "SKILL.md").read_text()
    assert (installed / "references" / "extra.md").exists()
    assert (installed / ".source").read_text().strip() == "owner/repo@commit"


def test_install_refuses_collision_without_force(tmp_path):
    src = _make_skill(tmp_path / "repo", "commit")
    dest_root = tmp_path / "skills"
    install_skill(src, dest_root, source_ref="owner/repo@commit")
    with pytest.raises(FileExistsError):
        install_skill(src, dest_root, source_ref="owner/repo@commit")


def test_install_force_overwrites(tmp_path):
    src = _make_skill(tmp_path / "repo", "commit")
    dest_root = tmp_path / "skills"
    install_skill(src, dest_root, source_ref="v1")
    (src / "SKILL.md").write_text("updated body")

    installed = install_skill(src, dest_root, source_ref="v2", force=True)

    assert (installed / "SKILL.md").read_text() == "updated body"
    assert (installed / ".source").read_text().strip() == "v2"


# ---------------------------------------------------------------------------
# CLI end-to-end (local git repo, no network)
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_repo(tmp_path):
    """A local git repo containing two skills, usable as a clone source."""
    repo = tmp_path / "source-repo"
    _make_skill(repo, "commit", "make a commit")
    _make_skill(repo / "skills", "deploy", "deploy things")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    return repo


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Point cca's native skills directory at a temp location."""
    d = tmp_path / "cca-skills"
    from cca import paths
    from cca.tools import skill_tool
    monkeypatch.setattr(paths, "SKILLS_DIR", d)
    monkeypatch.setattr(skill_tool, "SKILLS_DIR", d)
    monkeypatch.setattr(skill_tool, "CLAUDE_SKILLS_DIR", tmp_path / "claude-skills")
    return d


def test_cli_add_single_skill(skill_repo, skills_dir):
    result = CliRunner().invoke(skills, ["add", f"{skill_repo}@commit"])
    assert result.exit_code == 0, result.output
    assert (skills_dir / "commit" / "SKILL.md").exists()
    assert (skills_dir / "commit" / ".source").read_text().strip() == f"{skill_repo}@commit"


def test_cli_add_without_name_lists_available(skill_repo, skills_dir):
    result = CliRunner().invoke(skills, ["add", str(skill_repo)])
    assert result.exit_code != 0
    assert "commit" in result.output
    assert "deploy" in result.output
    assert not (skills_dir / "commit").exists()


def test_cli_add_all(skill_repo, skills_dir):
    result = CliRunner().invoke(skills, ["add", str(skill_repo), "--all"])
    assert result.exit_code == 0, result.output
    assert (skills_dir / "commit" / "SKILL.md").exists()
    assert (skills_dir / "deploy" / "SKILL.md").exists()


def test_cli_add_collision_requires_force(skill_repo, skills_dir):
    runner = CliRunner()
    assert runner.invoke(skills, ["add", f"{skill_repo}@commit"]).exit_code == 0
    result = runner.invoke(skills, ["add", f"{skill_repo}@commit"])
    assert result.exit_code != 0
    assert "--force" in result.output
    result = runner.invoke(skills, ["add", f"{skill_repo}@commit", "--force"])
    assert result.exit_code == 0, result.output


def test_cli_add_unknown_skill_name(skill_repo, skills_dir):
    result = CliRunner().invoke(skills, ["add", f"{skill_repo}@nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_cli_remove(skill_repo, skills_dir):
    runner = CliRunner()
    runner.invoke(skills, ["add", f"{skill_repo}@commit"])
    result = runner.invoke(skills, ["remove", "commit"])
    assert result.exit_code == 0, result.output
    assert not (skills_dir / "commit").exists()


def test_cli_remove_missing_skill_fails(skills_dir):
    result = CliRunner().invoke(skills, ["remove", "ghost"])
    assert result.exit_code != 0


def test_cli_list_shows_installed(skill_repo, skills_dir):
    runner = CliRunner()
    runner.invoke(skills, ["add", f"{skill_repo}@commit"])
    result = runner.invoke(skills, ["list"])
    assert result.exit_code == 0, result.output
    assert "commit" in result.output


def test_skills_registered_on_main_cli():
    from cca.cli import main
    result = CliRunner().invoke(main, ["skills", "--help"])
    assert result.exit_code == 0, result.output
    assert "add" in result.output
