---
title: Randomization
sidebar:
  order: 5
---

Every CCA run normally shows the model an identical scaffold: the same system prompt, the same tool descriptions in the same order, the same compaction prompt. That is exactly what you want day to day — and exactly what you do not want when you are **collecting traces to train on**. A model distilled from thousands of identical-scaffold rollouts learns your harness's wording as if it were a fact about the world.

`--randomize` varies the *surface* of the scaffold per run, drawing from a bank of pre-written, verified variants that ships inside CCA.

```bash
cca --randomize system_prompt|tool_descriptions|tool_order|compaction_prompt     # just one axis
cca --randomize                      # all axes, random seed (reported on stderr)
cca --randomize --seed 42            # reproducible
cca --randomize off                  # explicit no-op (also the default)
```

Without the flag nothing changes at all — same prompt, same tools, same files on disk.

## What gets randomized


| Axis                | What varies                                                                           |
| ------------------- | ------------------------------------------------------------------------------------- |
| `system_prompt`     | Which phrasing of the system prompt is used, **and** the order of its middle sections |
| `tool_descriptions` | The prose in each tool's `description` and its parameter descriptions                 |
| `tool_order`        | The order tools are presented to the model                                            |
| `compaction_prompt` | The phrasing of the [compaction](../agent-features/compaction.md) prompt              |


Use `all` (the default for a bare `--randomize`), `off`, or a comma-separated list. Hyphens work too, so `--randomize tool-order` is fine. An unrecognised axis is a fatal error rather than a silent no-op — a typo in a collection run would otherwise quietly cost you a whole dataset.

## What never changes

This is the important half. Randomization is only allowed to change how the scaffold *reads*, never what it *does*:

- **Tool names** — `Read`, `Edit`, `Bash` and the rest are identical in every variant. They are referenced by name throughout the harness and in anything that parses your traces.
- **Parameter names** — only the descriptions change. `file_path` is always `file_path`.
- **Every behavioral rule** — each variant must carry the same directives (read before editing, call independent tools in parallel, never force-push to main, and so on). A variant that drops one is rejected before it can ship.
- **The `--machine` protocol** — stdout stays pure JSON lines.
- **Your configuration** — `--randomize` is independent of `--preset`, `features:`, and `tools:`. It cannot turn a tool on or off.

The randomization is resolved **once at startup and then frozen** for the whole session, so a conversation never shifts underneath the model.

## Seeds and reproducibility

A run with `--seed N` is fully reproducible: the same seed always selects the same variants and the same section order.

If you omit `--seed`, one is generated and printed to **stderr**:

```
randomize: axes=compaction_prompt,system_prompt,tool_descriptions,tool_order seed=2624828706
```

Keep that number and you can reproduce the run exactly. It goes to stderr specifically so that `--machine` mode's stdout stays machine-readable.

To see what a seed selects without running anything:

```bash
$ cca variants resolve --seed 42
seed 42   axes compaction_prompt, system_prompt, tool_descriptions, tool_order
  compaction_prompt    v01_direct
  system_prompt        v01_terse
  tool_descriptions    v00_baseline
  section order        role → tool-narration → git-safety → ... → env
```

Seeds are stable across axis changes: adding `tool_order` to your spec will not change which system prompt seed 42 picks.

## Environment variables

For Docker images and orchestrators that cannot easily pass flags:

```bash
export CCA_RANDOMIZE=all
export CCA_SEED=42
```

Command-line flags win over both.

## Inspecting the bank

```bash
cca variants list                          # every variant, per axis
cca variants show system_prompt v01_terse  # print one in full
cca variants show system_prompt v01_terse --seed 42   # ...with that seed's section order
cca variants verify                        # check the bank against its contracts
```

Each axis includes a `v00_baseline` variant that is byte-for-byte the shipped default, so it is always in the pool and randomization can never produce something *worse* than a normal run.

## Adding your own variants

Drop files in `~/.cca/variants/<axis>/` (or point `CCA_VARIANTS_DIR` elsewhere). They are added to the bundled bank; a file whose `id` matches a bundled one replaces it.

A system prompt variant is markdown with frontmatter and `## @name` section markers. The markers are scaffolding — they control ordering and are stripped before the model sees anything:

```markdown
---
id: v10_mine
axis: system_prompt
derived_from: v00_baseline
summary: My house style.
---
## @role {pinned=first}
You are a coding assistant that runs in a terminal.

## @tool-usage
# Tool usage
- Use the Read tool to read files — never `cat` via Bash.

## @env {pinned=last}
<<CACHE_BOUNDARY>>
# Environment
{env_info}
{extra}
```

`{pinned=first}` and `{pinned=last}` sections never move when the middle is shuffled. The `@env` section is pinned last because `{extra}` carries your project's AGENTS.md, which has to override what came before it. A line ending in `\` joins with the next, so you can wrap long prose.

Tool description variants are YAML and may be partial — anything you leave out keeps its built-in text:

```yaml
id: v10_mine
axis: tool_descriptions
derived_from: v00_baseline
summary: Reword the two tools I care about.
tools:
  Read:
    description: Pull a file's contents back, numbered like cat -n.
    params:
      file_path: The file's absolute path.
```

Run `cca variants verify` after adding one. It checks that your variant keeps every required directive, names only real tools and real parameters, keeps the `{env_info}`/`{extra}` placeholders intact and in order, and has not drifted far from the baseline in length.

## What lands in your traces

When randomization is active, a `randomization` block is recorded in the session file, in every LLM call log record, and in `--machine` mode's `ready` event:

```json
{
  "seed": 42,
  "seed_source": "explicit",
  "bank_version": 1,
  "axes": ["compaction_prompt", "system_prompt", "tool_descriptions", "tool_order"],
  "variants": {"system_prompt": "v01_terse", "tool_descriptions": "v00_baseline",
               "compaction_prompt": "v01_direct"},
  "section_order": ["role", "tool-narration", "git-safety", "...", "env"]
}
```

The call log already stores the full system prompt and the complete tool array for every call, so the randomized *content* was always captured; this block adds the *provenance* needed to group, filter, or reproduce runs by variant. See [Sessions & Traces](../agent-features/sessions-and-traces.md).

When randomization is off, the key is omitted entirely and your files are byte-identical to what earlier versions wrote.

## Collecting traces

A practical setup for a batch of rollouts: derive the seed from whatever ID your orchestrator already assigns each session, so runs vary independently but any one of them can be replayed.

```bash
cca --machine --randomize all --seed "$SESSION_SEED" --cwd "$REPO"
```

Then slice your dataset by `randomization.variants.system_prompt` to check that task success rate holds across variants before you scale the batch up.
