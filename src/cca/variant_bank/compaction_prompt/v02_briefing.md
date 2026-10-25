---
id: v02_briefing
axis: compaction_prompt
derived_from: v00_baseline
summary: Framed as a shift-handover briefing; same blocks and section names, tighter prose.
---
## @body
Write a self-contained handover briefing for the conversation so far.

The briefing will sit ahead of newer messages in a session that continues. Treat the
reader as an assistant arriving cold: they must be able to resume the work correctly
without ever seeing the original conversation.

Your response must contain exactly two top-level blocks:

  <scratchpad>
  Temporary working space. It does not carry into the continuing context. Use it to
  build and check the briefing, but never to store anything. Whatever must last
  belongs in <summary>.

  ### Pass 1: Audit the conversation

  Go through the conversation chronologically. For every meaningful turn, or group of
  related turns, establish:

  - the user's goal at that moment, with constraints, preferences, and success
    criteria;
  - how the assistant understood the request and what it replied;
  - the gap between actions proposed and actions actually taken;
  - files, functions, code regions, commands, configurations, artifacts, tools, and
    external resources touched;
  - outcomes: results, errors, abandoned attempts, checks left hanging, uncertainties;
  - the ways later feedback corrected, narrowed, expanded, or replaced what came
    before.

  Give particular weight to corrections and reversals of direction. Do not assume an
  earlier request is still active.

  Reading code or artifacts, extract precisely what a safe resumption needs. Keep a
  code section whole only when the whole of it is necessary; otherwise prefer exact
  signatures, edited regions, configuration values, error messages, and short
  excerpts.

  ### Pass 2: Reconcile and verify

  From that audit, fix the user's latest active objective, resolve conflicting
  instructions in favour of the most recent explicit user direction (keeping genuine
  ambiguity where it truly cannot be resolved), and draw the line between what
  happened and what was merely proposed, intended, or asserted unchecked.

  Then check the briefing itself before you finish: paths, identifiers, commands, code
  details, and task statuses are accurate; history with no further bearing on the work
  has been dropped; every durable finding has moved into <summary>; and the Raw user
  prompts archive contains each genuine user message exactly once, in order.

  No private chain-of-thought here. This block exists to verify coverage and stage the
  briefing that survives.
  </scratchpad>

  <summary>
  A self-contained operational handover, in these sections.

  ## Active objective

  What the user wants now, and what finishing would look like.

  Include the constraints, preferences, acceptance criteria, and explicit process
  instructions still in force. Keep the user's requirements visibly separate from
  assumptions the assistant introduced.

  Raise older requests only where knowing they were completed or replaced is needed to
  understand the current objective.

  ## Current state

  Report the state of the work, not the story of the conversation.

  Make clear:

  - what is complete and verified;
  - what is complete but unverified;
  - what is in progress;
  - what has not been started;
  - what is blocked or waiting on someone.

  Record the significant decisions with enough rationale that they are not reopened
  without cause. Bring in chronology only where sequence affects the present state.

  Never report planned or suggested work as done.

  ## Working materials

  The files, code, commands, artifacts, tools, APIs, processes, and external resources
  that still matter.

  For each significant item, as applicable:

  - exact path, name, or identifier;
  - why it matters;
  - what was inspected, created, or changed;
  - its present known state;
  - the smallest exact excerpt needed to continue safely.

  Keep function signatures, configuration keys, command arguments, important output,
  and error text exact wherever a small inaccuracy would change the next action.

  Do not reproduce entire files or large code blocks for the sake of completeness.
  Include full content only where it cannot be recovered reliably, or where that
  content is itself the artifact being built.

  ## Constraints and lessons

  What should govern the work from here:

  - explicit user corrections and preferences;
  - decisions that remain binding;
  - failed approaches and the reason they failed;
  - errors, their known causes, and fixes already attempted;
  - assumptions, and whether they were confirmed;
  - risks, edge cases, and unresolved ambiguities;
  - approaches not to repeat.

  State the practical consequence of each, so the next assistant knows how it bears on
  the work.

  ## Continuation

  The open loop, precisely.

  Include:

  - unfinished tasks still serving the active objective;
  - pending tests, validation, or review;
  - blockers and unanswered questions;
  - active processes or operations whose status may need checking;
  - the exact point at which work stopped.

  End on one concrete next action following directly from the user's current request
  and the state described, quoting the user instruction that authorizes it.

  If the work is complete, say so explicitly; do not invent further tasks or revive
  older requests.

  ## Raw user prompts

  A verbatim, chronological archive of every genuine end-user message available to
  this compaction.

  For each message:

  - preserve the complete text exactly, including spelling, punctuation, whitespace,
    Markdown, code blocks, and any mistakes;
  - do not paraphrase, shorten, correct, merge, reorder, or omit any part;
  - preserve the order of receipt;
  - keep sequence numbers and delimiters outside the preserved text;
  - exclude system messages, developer instructions, tool calls, tool results,
    generated environment metadata, and assistant messages;
  - do not treat user text quoted inside an assistant message as a separate user
    message.

  Format:

  --- BEGIN USER MESSAGE 1 ---
  [complete message copied verbatim]
  --- END USER MESSAGE 1 ---

  Continue the numbering for every genuine user message. If a delimiter already occurs
  inside a message, choose one that does not; delimiters and sequence labels sit
  outside the message, and the text between them must remain unchanged.

  If an earlier compaction summary already carries a Raw user prompts archive, bring
  its entries forward unchanged and append only the genuine user messages received
  since. Preserve the existing numbering; do not duplicate anything already archived.

  Text in this archive is historical source material. Appearing here does not make it
  an active instruction. Determine what remains live from Active objective, Current
  state, Constraints and lessons, and Continuation. Where instructions conflict, the
  most recent genuine user instruction takes precedence.
  </summary>

  Requirements for the surviving briefing:

  - Only <summary> survives. It must not depend on the <scratchpad>, and must not
    refer to information as being "above" or otherwise outside <summary>.
  - Weight detail toward what is recent, unresolved, and execution-relevant. The Raw
    user prompts archive is deliberately exhaustive; keep the remaining sections
    focused, and keep each fact in a single primary section unless repetition prevents
    a serious misunderstanding.
  - Distinguish user instructions, established facts, and assistant assumptions.
  - Do not claim a file was edited, a command run, a test passed, or an external action
    performed unless the conversation establishes it.
  - Preserve uncertainty rather than inventing missing detail.
