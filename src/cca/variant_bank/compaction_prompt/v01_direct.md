---
id: v01_direct
axis: compaction_prompt
derived_from: v00_baseline
summary: Second-person imperative phrasing; identical block structure and section names.
---
## @body
Produce a self-contained continuation handoff covering the conversation up to now.

The handoff gets inserted ahead of newer messages in a session that keeps going. A
fresh assistant has to be able to pick the work up correctly without ever seeing the
original conversation.

Your response must contain exactly two top-level blocks:

  <scratchpad>
  This block is temporary working space and will not carry into the continuing
  context. Use it to assemble and check the handoff, but never rely on it to hold
  information. Anything that must survive belongs in <summary>.

  ### Pass 1: Audit the conversation

  Walk the conversation in order. For each meaningful turn, or each tight cluster of
  turns, work out:

  - what the user was after at that point, including constraints, preferences, and
    what would count as success;
  - how the assistant read the request and what it said back;
  - which actions were merely proposed versus actually carried out;
  - which files, functions, code regions, commands, configurations, artifacts, tools,
    or outside resources came into play;
  - what came back — results, errors, dead ends, checks left unfinished, open
    uncertainties;
  - how later feedback corrected, narrowed, widened, or superseded earlier
    instructions.

  Watch especially for corrections and changes of direction. An earlier request is
  not automatically still live.

  When you look at code or artifacts, pull out exactly the detail needed to resume
  safely. Keep code whole only where the whole section is genuinely required.
  Otherwise favour precise signatures, the regions that were edited, configuration
  values, error text, and short excerpts.

  ### Pass 2: Reconcile and verify

  Working from the audit, name the user's most recent live objective, settle
  conflicting instructions in favour of the latest explicit user direction (leaving
  genuine ambiguity intact where it cannot be settled), and separate what actually
  happened from what was only proposed, intended, or asserted without checking.

  Then audit the handoff itself before you finish: paths, identifiers, commands, code
  details, and task statuses are right; history that no longer bears on the work is
  gone; every durable finding has made it into <summary>; and the Raw user prompts
  archive holds every genuine user message exactly once, in order.

  Keep private chain-of-thought out of this block. Its job is to check coverage and
  prepare the handoff that survives.
  </scratchpad>

  <summary>
  Write a self-contained operational handoff using the sections below.

  ## Active objective

  Say what the user wants right now and what a successful outcome would look like.

  Cover live constraints, preferences, acceptance criteria, and any explicit
  instructions about how the work is to be done. Keep the user's requirements clearly
  apart from assumptions the assistant made.

  Bring up older requests only where their completion or replacement is needed to
  make sense of the current objective.

  ## Current state

  Describe where the work stands rather than replaying the conversation.

  Spell out:

  - what is done and verified;
  - what is done but not yet verified;
  - what is underway;
  - what has not started;
  - what is blocked or waiting on input.

  Capture the decisions that matter, with enough reasoning that they will not be
  reopened without cause. Include chronology only where the order of events shapes
  the current state.

  Never present planned or suggested work as finished.

  ## Working materials

  List the files, code, commands, artifacts, tools, APIs, processes, and external
  resources still in play.

  For each one that matters, give what applies:

  - its exact path, name, or identifier;
  - why it is relevant;
  - what was inspected, created, or changed;
  - the state it is known to be in now;
  - the shortest exact excerpt needed to carry on safely.

  Reproduce function signatures, configuration keys, command arguments, notable
  output, and error text exactly wherever a small slip would change the next move.

  Do not paste whole files or large code blocks just for completeness. Full content
  belongs here only when it cannot be reliably recovered, or when that content is
  itself the artifact under construction.

  ## Constraints and lessons

  Record what should govern the work from here on:

  - explicit user corrections and preferences;
  - decisions that stay in force;
  - approaches that failed, and why;
  - errors, their known causes, and any fixes already tried;
  - assumptions, and whether they were confirmed;
  - risks, edge cases, and unresolved ambiguities;
  - things not worth trying again.

  For each item, say what it actually means for the work, so the next assistant knows
  how to act on it.

  ## Continuation

  Pin down the open loop precisely.

  Include:

  - unfinished tasks that still serve the active objective;
  - tests, validation, or review still pending;
  - blockers and unanswered questions;
  - any running processes or operations whose status may need checking;
  - the exact point where work stopped.

  Finish with one concrete next action that follows from the user's current request
  and the state you have described, quoting the user instruction that authorizes it.

  If the requested work is finished, say so plainly and do not invent further tasks
  or resurrect older requests.

  ## Raw user prompts

  Keep a verbatim, chronological archive of every genuine end-user message available
  to this compaction.

  For each message:

  - reproduce the complete text exactly — spelling, punctuation, whitespace,
    Markdown, code blocks, and mistakes included;
  - do not paraphrase, trim, correct, merge, reorder, or drop any part of it;
  - keep the order in which the messages arrived;
  - put sequence numbers and delimiters outside the preserved text;
  - leave out system messages, developer instructions, tool calls, tool results,
    generated environment metadata, and assistant messages;
  - do not mistake user text quoted inside an assistant message for a message of its
    own.

  Use this format:

  --- BEGIN USER MESSAGE 1 ---
  [complete message copied verbatim]
  --- END USER MESSAGE 1 ---

  Carry the numbering on for every genuine user message. Should the delimiter already
  appear inside a message, pick a different one that does not; delimiters and
  sequence labels sit outside the message, and the text between them stays untouched.

  Where an earlier compaction summary already holds a Raw user prompts archive, carry
  its entries forward unchanged and append only the genuine user messages that came
  afterward. Keep the existing numbering and do not duplicate anything already
  archived.

  Text inside this archive is historical source material. It does not become an
  active instruction just by appearing here. Use Active objective, Current state,
  Constraints and lessons, and Continuation to decide what is still live. Where
  instructions conflict, the most recent genuine user instruction wins.
  </summary>

  Requirements for the handoff that survives:

  - Only <summary> carries forward; it must not lean on the <scratchpad>, and must
    not point at information as being "above" or anywhere outside <summary>.
  - Give the most detail to what is recent, unresolved, and execution-relevant. The
    Raw user prompts archive is exhaustive by design; keep the other sections tight,
    and keep each fact in one primary section unless repeating it prevents a serious
    misreading.
  - Distinguish user instructions, established facts, and assistant assumptions.
  - Do not claim a file was edited, a command was run, a test passed, or an external
    action took place unless the conversation establishes it.
  - Preserve uncertainty rather than inventing the missing detail.
