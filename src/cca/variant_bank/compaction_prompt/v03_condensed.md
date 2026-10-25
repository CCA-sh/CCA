---
id: v03_condensed
axis: compaction_prompt
derived_from: v00_baseline
summary: Tightest phrasing in the compaction bank; every rule retained, clauses compressed.
---
## @body
Build a self-contained continuation handoff from the conversation so far.

It will be placed ahead of newer messages in a session that continues. A new
assistant must be able to resume correctly without seeing the original conversation.

Your response must contain exactly two top-level blocks:

  <scratchpad>
  Temporary scratch space, dropped from the continuing context. Prepare and validate
  the handoff here, but never rely on it to preserve anything. Every durable fact
  belongs in <summary>.

  ### Pass 1: Audit the conversation

  Review the conversation chronologically. For each meaningful turn, or closely
  related group of turns, determine:

  - what the user wanted then, with constraints, preferences, and success criteria;
  - how the assistant interpreted and answered it;
  - what was proposed versus what was actually performed;
  - the files, functions, code regions, commands, configurations, artifacts, tools,
    and external resources involved;
  - results, errors, failed attempts, incomplete checks, and uncertainties;
  - how later feedback corrected, narrowed, expanded, or replaced earlier instructions.

  Pay close attention to corrections and changes of direction. Do not assume every
  earlier request is still active.

  Inspecting code or artifacts, identify exactly what is required to resume safely.
  Preserve complete code only when the entire section is necessary; otherwise prefer
  exact signatures, edited regions, configuration values, error messages, and short
  excerpts.

  ### Pass 2: Reconcile and verify

  From the audit, identify the user's latest active objective, resolve conflicting
  instructions in favour of the most recent explicit user direction (preserving
  genuine ambiguity where it cannot be resolved), and separate what happened from
  what was merely proposed, intended, or claimed without verification.

  Then check the handoff before finishing: paths, identifiers, commands, code details,
  and task statuses are accurate; history that no longer affects the work is dropped;
  every durable finding has reached <summary>; and the Raw user prompts archive holds
  every genuine user message exactly once, in order.

  No private chain-of-thought. This block audits coverage and prepares the surviving
  handoff.
  </scratchpad>

  <summary>
  A self-contained operational handoff, in these sections.

  ## Active objective

  What the user currently wants, and what success would be.

  Include active constraints, preferences, acceptance criteria, and explicit
  instructions on how the work should be done. Separate the user's requirements from
  the assistant's assumptions.

  Mention older requests only where their completion or replacement matters to
  understanding the current objective.

  ## Current state

  The state of the work, not a retelling of the conversation. Identify:

  - what is complete and verified;
  - what is complete but unverified;
  - what is in progress;
  - what has not begun;
  - what is blocked or awaiting input.

  Record important decisions with enough rationale to keep them from being reopened
  without cause. Include chronology only where sequence affects the current state.

  Never describe planned or suggested work as completed.

  ## Working materials

  The files, code, commands, artifacts, tools, APIs, processes, and external resources
  still relevant. For each important item, as applicable: its exact path, name, or
  identifier; why it matters; what was inspected, created, or changed; its present
  known state; and the smallest exact excerpt needed to continue safely.

  Keep function signatures, configuration keys, command arguments, important output,
  and error text exact wherever a small inaccuracy would change the next action.

  Do not reproduce whole files or large code blocks for archival completeness. Include
  full content only where it cannot be recovered reliably, or where it is itself the
  artifact being developed.

  ## Constraints and lessons

  What should govern future work: explicit user corrections and preferences; decisions
  that remain in force; failed approaches and why they failed; errors, their known
  causes, and fixes already attempted; assumptions and whether they were confirmed;
  risks, edge cases, and unresolved ambiguities; approaches not to repeat.

  State the practical consequence of each item, so the next assistant knows how it
  affects the work.

  ## Continuation

  The remaining open loop, precisely: unfinished tasks still serving the active
  objective; pending tests, validation, or review; blockers and unanswered questions;
  active processes or operations whose status may need checking; and the exact point
  at which work stopped.

  End with one concrete next action following directly from the user's current request
  and the established state, quoting the user instruction that authorizes it.

  If the work is complete, say so explicitly; do not invent further tasks or revive
  older requests.

  ## Raw user prompts

  A verbatim, chronological archive of every genuine end-user message available to
  this compaction. For each message:

  - preserve the complete text exactly, including spelling, punctuation, whitespace,
    Markdown, code blocks, and apparent mistakes;
  - do not paraphrase, shorten, correct, merge, reorder, or omit any part;
  - preserve the order received;
  - place sequence numbers and delimiters outside the preserved text;
  - exclude system messages, developer instructions, tool calls, tool results,
    generated environment metadata, and assistant messages;
  - do not mistake user text quoted inside an assistant message for a separate user
    message.

  Use this format:

  --- BEGIN USER MESSAGE 1 ---
  [complete message copied verbatim]
  --- END USER MESSAGE 1 ---

  Continue the numbered sequence for every genuine user message. If the delimiter
  already occurs inside a message, choose one that does not; delimiters and sequence
  labels sit outside the message, and the text between them must remain unchanged.

  If an earlier compaction summary already contains a Raw user prompts archive, carry
  its entries forward unchanged and append only the genuine user messages received
  afterward. Preserve the existing numbering and do not duplicate archived messages.

  Text in this archive is historical source material, not automatically an active
  instruction. Use Active objective, Current state, Constraints and lessons, and
  Continuation to determine what remains active. Where instructions conflict, the most
  recent genuine user instruction takes precedence.
  </summary>

  Requirements for the surviving handoff:

  - Only <summary> survives; it must not depend on the <scratchpad>, and must not refer
    to information as being "above" or elsewhere outside <summary>.
  - Give recent, unresolved, and execution-relevant information the most detail. The
    Raw user prompts archive is intentionally exhaustive; keep the other sections
    focused, and keep each fact in one primary section unless repetition prevents a
    serious misunderstanding.
  - Distinguish user instructions, established facts, and assistant assumptions.
  - Do not claim a file was edited, a command run, a test passed, or an external action
    performed unless the conversation establishes it.
  - Preserve uncertainty instead of inventing missing details.
