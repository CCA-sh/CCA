---
title: Skills
sidebar:
  order: 7
---

Skills are reusable prompt packages — a folder with a `SKILL.md` inside — that teach the agent how to do a specific task well: making commits, reviewing PRs, designing frontends, debugging systematically. The agent sees every installed skill in its system prompt and invokes them on its own when relevant, or you can ask for one by name.

Skills use the same `SKILL.md` format as Claude Code, so any skills you've already installed for Claude Code in `~/.claude/skills/` work in cca without reinstalling.

```bash
cca skills add anthropics/skills@frontend-design   # install a skill from GitHub
cca skills list                                    # see what's installed
cca skills remove frontend-design                  # uninstall
```

Inside the REPL, `/skills` lists them.

## Installing skills

`cca skills add` accepts a GitHub shorthand, a full git URL, or a local repo path:

```bash
cca skills add owner/repo@skill-name               # one skill from a repo
cca skills add https://github.com/owner/repo@name  # same, full URL
cca skills add owner/repo                          # repo with a single skill
cca skills add owner/repo --all                    # install every skill in the repo
cca skills add /path/to/local/repo@name            # from a local git repo
```

If a repo contains several skills and you don't name one, cca lists what's available so you can pick:

```
$ cca skills add anthropics/skills
anthropics/skills contains 18 skills:
  frontend-design — Guidance for distinctive, intentional visual design...
  mcp-builder — Guide for creating high-quality MCP servers...
  ...
Install one with:  cca skills add anthropics/skills@<name>
```

Installing a skill that already exists fails safely; pass `--force` to overwrite (this is also how you update a skill to its latest version). Each installed skill keeps a `.source` file recording where it came from.

Good places to find skills:

- [skills.sh](https://skills.sh/) — searchable directory of the open skills ecosystem
- [anthropics/skills](https://github.com/anthropics/skills) — official Anthropic skills
- Any git repo with `SKILL.md` folders works — there is no required registry.

## Where skills live

| Scope | Location | Notes |
|---|---|---|
| cca-native | `~/.cca/skills/<name>/SKILL.md` | Where `cca skills add` installs. Wins on name collisions. |
| Claude Code compatibility | `~/.claude/skills/<name>/SKILL.md` | Read-only. Skills you installed for Claude Code are picked up automatically; a cca skill with the same name takes precedence. |

Override the native directory with the `CCA_SKILLS_DIR` environment variable.

New skills are picked up automatically — the directories are re-scanned at session start, no registration step needed.

## Using skills

You usually don't have to do anything: every skill's name and description is listed in the agent's system prompt, and the agent invokes the `Skill` tool when a task matches. To use one explicitly, just ask:

```
> use the frontend-design skill to restyle the settings page
```

## Writing your own

A skill is just a markdown file. Create `~/.cca/skills/<name>/SKILL.md` with optional YAML frontmatter and a prompt body:

```markdown
---
name: commit
description: Generate and create a git commit for staged changes
---
Review the staged changes with `git diff --cached` and create a
well-formed commit following the conventional commits format.
```

The `description` matters most — it's what the agent reads when deciding whether the skill applies. Supporting files (references, scripts) can live alongside `SKILL.md` in the same folder.

To share a skill, push the folder to a git repo; anyone can then install it with `cca skills add your/repo@name`.
