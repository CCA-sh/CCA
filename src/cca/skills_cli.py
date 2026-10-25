"""`cca skills` subcommands — install, list, and remove skills from git repos.

A skill is a directory containing a SKILL.md. Installing means cloning the
source repo and copying the skill directory into ~/.cca/skills, where the
Skill tool discovers it automatically. The original ref is recorded in a
`.source` file inside the installed skill for future updates.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import click
from rich.console import Console

console = Console(stderr=True)

_GITHUB_SHORTHAND = re.compile(r"^[\w.-]+/[\w.-]+$")

MAX_DISCOVERY_DEPTH = 3


class SkillRef(NamedTuple):
    clone_url: str
    skill_name: str | None
    display: str


class FoundSkill(NamedTuple):
    name: str
    path: Path
    description: str


def parse_skill_ref(ref: str) -> SkillRef:
    """Parse 'owner/repo[@skill]', a git URL, or a local path into a SkillRef."""
    base, sep, skill_name = ref.rpartition("@")
    if not sep:
        base, skill_name = ref, None
    skill = skill_name or None

    if base.startswith(("https://", "http://", "git@")):
        clone_url = base if base.endswith(".git") else f"{base}.git"
        return SkillRef(clone_url, skill, base)
    if Path(base).is_dir():
        return SkillRef(base, skill, base)
    if _GITHUB_SHORTHAND.match(base):
        return SkillRef(f"https://github.com/{base}.git", skill, base)
    raise ValueError(
        f"Cannot parse skill ref '{ref}'. Expected owner/repo, owner/repo@skill, "
        f"a git URL, or a local repo path."
    )


def _read_description(skill_md: Path) -> str:
    from cca.tools.skill_tool import _parse_skill
    try:
        fm, _ = _parse_skill(skill_md)
        return str(fm.get("description", "") or "")
    except Exception:
        return ""


def discover_skills(root: Path, max_depth: int = MAX_DISCOVERY_DEPTH) -> list[FoundSkill]:
    """Find directories containing a SKILL.md, searching a few levels deep."""
    found: list[FoundSkill] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            skill_md = child / "SKILL.md"
            if skill_md.exists():
                found.append(FoundSkill(child.name, child, _read_description(skill_md)))
            else:
                walk(child, depth + 1)

    walk(root, 1)
    return found


def install_skill(
    skill_dir: Path,
    dest_root: Path,
    source_ref: str,
    force: bool = False,
) -> Path:
    """Copy a skill directory into dest_root and record where it came from."""
    dest = dest_root / skill_dir.name
    if dest.exists():
        if not force:
            raise FileExistsError(f"Skill '{skill_dir.name}' already installed at {dest}")
        shutil.rmtree(dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_dir, dest)
    (dest / ".source").write_text(source_ref + "\n")
    return dest


def _clone(clone_url: str, dest: Path) -> None:
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", clone_url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"git clone failed for {clone_url}:\n{result.stderr.strip()}")


def _skills_dir() -> Path:
    from cca import paths
    return paths.SKILLS_DIR


@click.group()
def skills():
    """Install and manage skills (~/.cca/skills)."""


@skills.command("add")
@click.argument("ref")
@click.option("--all", "install_all", is_flag=True, default=False,
              help="Install every skill found in the repo.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite an already-installed skill of the same name.")
def skills_add(ref: str, install_all: bool, force: bool):
    """Install a skill from a git repo.

    \b
    Examples:
      cca skills add anthropics/skills@frontend-design
      cca skills add https://github.com/obra/superpowers@brainstorming
      cca skills add owner/repo --all
    """
    try:
        parsed = parse_skill_ref(ref)
    except ValueError as e:
        raise click.ClickException(str(e))

    with tempfile.TemporaryDirectory(prefix="cca-skill-") as tmp:
        checkout = Path(tmp) / "repo"
        console.print(f"[dim]Cloning {parsed.display}...[/dim]")
        _clone(parsed.clone_url, checkout)

        available = discover_skills(checkout)
        if not available:
            raise click.ClickException(f"No skills (directories with SKILL.md) found in {parsed.display}")

        if parsed.skill_name:
            targets = [s for s in available if s.name == parsed.skill_name]
            if not targets:
                names = ", ".join(s.name for s in available)
                raise click.ClickException(
                    f"Skill '{parsed.skill_name}' not found in {parsed.display}. Available: {names}"
                )
        elif install_all:
            targets = available
        elif len(available) == 1:
            targets = available
        else:
            click.echo(f"{parsed.display} contains {len(available)} skills:")
            for s in available:
                desc = f" — {s.description}" if s.description else ""
                click.echo(f"  {s.name}{desc}")
            click.echo(f"\nInstall one with:  cca skills add {parsed.display}@<name>")
            click.echo(f"Or install all:    cca skills add {parsed.display} --all")
            sys.exit(1)

        for target in targets:
            source_ref = f"{parsed.display}@{target.name}" if not parsed.skill_name else ref
            try:
                dest = install_skill(target.path, _skills_dir(), source_ref, force=force)
            except FileExistsError:
                raise click.ClickException(
                    f"Skill '{target.name}' is already installed. Use --force to overwrite."
                )
            console.print(f"[green]Installed '{target.name}'[/green] [dim]→ {dest}[/dim]")


@skills.command("list")
def skills_list():
    """List installed skills (cca-native and Claude Code users shared)."""
    from cca.tools.skill_tool import list_skills
    entries = list_skills()
    if not entries:
        click.echo("No skills installed. Add one with: cca skills add <owner/repo@skill>")
        return
    for e in entries:
        desc = f" — {e.description}" if e.description else ""
        click.echo(f"  [{e.source}] {e.name}{desc}")


@skills.command("remove")
@click.argument("name")
def skills_remove(name: str):
    """Remove an installed cca-native skill by name."""
    target = _skills_dir() / name
    if not (target / "SKILL.md").exists():
        raise click.ClickException(f"Skill '{name}' not found in {_skills_dir()}")
    shutil.rmtree(target)
    console.print(f"[green]Removed '{name}'[/green]")
