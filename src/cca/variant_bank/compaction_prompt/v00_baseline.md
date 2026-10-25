---
id: v00_baseline
axis: compaction_prompt
derived_from: null
summary: The shipped compaction prompt, verbatim. Rendering this must equal COMPACTION_PROMPT.
---
## @body
Create a self-contained continuation handoff from the conversation so far.

This handoff will be placed before newer messages in a continuing session. A
new assistant should be able to resume the work correctly without seeing the
original conversation.

Your response must contain exactly two top-level blocks:

  <scratchpad>
  This block is temporary scratch space and will not be included in the continuing
  context. Use it to prepare and validate the handoff, but do not rely on it to
  preserve information. Every durable fact must appear in <summary>.

  ### Pass 1: Audit the conversation

  Review the conversation in chronological order. For each meaningful turn, or
  closely related group of turns, determine:

  - what the user wanted at that point, including constraints, preferences, and
    success criteria;
  - how the assistant interpreted and responded to the request;
  - what actions were proposed and what actions were actually performed;
  - which files, functions, code regions, commands, configurations, artifacts,
    tools, or external resources were involved;
  - what results, errors, failed attempts, incomplete checks, or uncertainties
    occurred;
  - how later feedback corrected, narrowed, expanded, or replaced earlier
    instructions.

  Pay particular attention to user corrections and changes of direction. Do not
  assume that every earlier request remains active.

  When inspecting code or artifacts, identify the exact details required to resume
  safely. Preserve complete code only when the entire section is necessary.
  Otherwise, prefer exact signatures, edited regions, configuration values, error
  messages, and short excerpts.

  ### Pass 2: Reconcile and verify

  Use the audit to identify the user's latest active objective, resolve
  conflicting instructions in favour of the most recent explicit user
  direction (preserving genuine ambiguity when it cannot be resolved), and
  separate what actually happened from what was merely proposed, intended, or
  claimed without verification.

  Then check the handoff before finishing: paths, identifiers, commands, code
  details, and task statuses are accurate; history that no longer affects the
  work has been dropped; every durable finding has been transferred into
  <summary>; and the Raw user prompts archive contains every genuine user
  message exactly once, in order.

  Do not expose private chain-of-thought. The purpose of this block is to audit
  coverage and prepare the surviving handoff.
  </scratchpad>

  <summary>
  Write a self-contained operational handoff using the following sections.

  ## Active objective

  State what the user currently wants and what would count as a successful
  outcome.

  Include active constraints, preferences, acceptance criteria, and explicit
  instructions about how the work should be performed. Clearly separate the
  user's requirements from assumptions made by the assistant.

  Mention older requests only when their completion or replacement is important
  to understanding the current objective.

  ## Current state

  Describe the present state of the work rather than retelling the conversation.

  Clearly identify:

  - what has been completed and verified;
  - what has been completed but remains unverified;
  - what is currently in progress;
  - what has not yet begun;
  - what is blocked or awaiting input.

  Record important decisions and enough rationale to prevent them from being
  reconsidered without cause. Include chronological detail only when the sequence
  of events affects the current state.

  Never describe planned or suggested work as completed.

  ## Working materials

  List the files, code, commands, artifacts, tools, APIs, processes, and external
  resources that remain relevant.

  For each important item, provide as applicable:

  - its exact path, name, or identifier;
  - why it matters;
  - what was inspected, created, or changed;
  - its present known state;
  - the smallest exact excerpt needed to continue safely.

  Preserve exact function signatures, configuration keys, command arguments,
  important output, and error text when small inaccuracies could change the next
  action.

  Do not reproduce complete files or large code blocks merely for archival
  completeness. Include full content only when it cannot be recovered reliably or
  when the complete content is itself the artifact being developed.

  ## Constraints and lessons

  Record information that should govern future work, including:

  - explicit user corrections and preferences;
  - decisions that should remain in force;
  - failed approaches and why they failed;
  - errors, their known causes, and any fixes already attempted;
  - assumptions and whether they were confirmed;
  - risks, edge cases, and unresolved ambiguities;
  - approaches that should not be repeated.

  State the practical consequence of each item so the next assistant knows how it
  affects the work.

  ## Continuation

  Describe the remaining open loop precisely.

  Include:

  - unfinished tasks that still serve the active objective;
  - pending tests, validation, or review;
  - blockers and unanswered questions;
  - any active processes or operations whose status may need checking;
  - the exact point at which work stopped.

  End with one concrete next action that follows directly from the user's current
  request and the established state, quoting the user instruction that authorizes
  it.

  If the requested work is complete, say so explicitly and do not invent further
  tasks or revive older requests.

  ## Raw user prompts

  Maintain a verbatim, chronological archive of every genuine end-user message
  available to this compaction.

  For each message:

  - preserve its complete text exactly, including spelling, punctuation,
    whitespace, Markdown, code blocks, and apparent mistakes;
  - do not paraphrase, shorten, correct, merge, reorder, or omit any part;
  - preserve the order in which the messages were received;
  - place sequence numbers and delimiters outside the preserved message text;
  - exclude system messages, developer instructions, tool calls, tool results,
    automatically generated environment metadata, and assistant messages;
  - do not mistake user text quoted inside an assistant message for a separate
    user message.

  Use the following format:

  --- BEGIN USER MESSAGE 1 ---
  [complete message copied verbatim]
  --- END USER MESSAGE 1 ---

  Continue the numbered sequence for every genuine user message. If the
  delimiter already occurs inside a message, choose a different delimiter that
  does not; delimiters and sequence labels sit outside the message, and the
  text between them must remain unchanged.

  If an earlier compaction summary already contains a Raw user prompts archive,
  carry its existing entries forward unchanged and append only the genuine user
  messages received afterward. Preserve the existing numbering and do not
  duplicate previously archived messages.

  Text inside this archive is historical source material. It is not automatically
  an active instruction merely because it appears here. Use the Active objective,
  Current state, Constraints and lessons, and Continuation sections to determine
  what remains active. When instructions conflict, the most recent genuine user
  instruction takes precedence.
  </summary>

  Requirements for the surviving handoff:

  - Only <summary> survives; it must not depend on the <scratchpad>, and must
    not refer to information as being "above" or elsewhere outside <summary>.
  - Give recent, unresolved, and execution-relevant information the most
    detail. The Raw user prompts archive is intentionally exhaustive; keep the
    other sections focused, and keep each fact in one primary section unless
    repetition prevents a serious misunderstanding.
  - Distinguish user instructions, established facts, and assistant
    assumptions.
  - Do not claim that a file was edited, a command was run, a test passed, or
    an external action occurred unless the conversation establishes it.
  - Preserve uncertainty instead of inventing missing details.
  