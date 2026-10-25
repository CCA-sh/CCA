"""Pure render functions for terminal UI components."""

from __future__ import annotations

import re

_ANSI_ESC = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_LINE_NUM  = re.compile(r'^\d+\t')   # Read-tool cat-n prefix: "14\tcontent"

# Spinner: grows from tiny dot to large star and back; palindromic so it loops
# smoothly without a jump.
SPINNER_FRAMES = "·-÷+✦★✳•✱⏺✱•✳★✦+÷-"
ANIMATION_INTERVAL = 0.12  # seconds

DONE_INDICATOR = "●"
ERR_INDICATOR = "✗"


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def spinner_frame(tick: int) -> str:
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def shimmer_title(title: str, tick: int) -> str:
    """Dim the title and slide a 3-char bright window right-to-left."""
    n = len(title)
    WINDOW = 3
    dim   = "\033[2m"
    reset = "\033[0m"

    # One step every 4 ticks (~0.5 s); full cycle covers title + WINDOW padding
    step     = (tick // 4) % (n + WINDOW)
    hi_start = n - step          # right edge enters first, slides left
    hi_end   = hi_start + WINDOW - 1

    vis_start = max(0, hi_start)
    vis_end   = min(n - 1, hi_end)

    if vis_start > vis_end:
        return f"{dim}{title}{reset}"

    prefix = title[:vis_start]
    window = title[vis_start:vis_end + 1]
    suffix = title[vis_end + 1:]

    out = f"{dim}{prefix}{reset}" if prefix else ""
    out += window
    out += f"{dim}{suffix}{reset}" if suffix else ""
    return out


def render_action_inline_ansi(
    indicator: str,
    title: str,
    lines: list[str],
    total: int,
    has_error: bool,
    elapsed: float,
    max_lines: int = 10,
    tick: int = -1,
) -> str:
    """Render a running/done action as an inline ANSI string.

    Format while running (indicator = spinner frame):
        · Read  file_path: /tmp/foo   0.3s
           1  content line
           2  more content

    Format when done (indicator = DONE_INDICATOR):
        ● Read  file_path: /tmp/foo   18 lines  0.3s
           1  content line
           2  more content
    """
    color = "\033[31m" if has_error else "\033[32m"
    reset = "\033[0m"
    dim   = "\033[2m"

    right_parts = []
    if total > 0:
        right_parts.append(f"{total} lines")
    right_parts.append(fmt_elapsed(elapsed))
    right = "  ".join(right_parts)

    running = indicator != DONE_INDICATOR
    display_title = shimmer_title(title, tick) if (running and tick >= 0) else title
    header = f"{color}{indicator}{reset} {display_title}  {dim}{right}{reset}"

    rows: list[str] = []
    if lines:
        n_take = min(len(lines), max_lines)
        raw_slice = lines[-n_take:]
        # Line numbers reflect the actual position within the full output stream.
        start_num = total - n_take + 1

        for i, raw in enumerate(raw_slice):
            clean = _ANSI_ESC.sub("", raw)
            clean = _LINE_NUM.sub("", clean)
            clean = clean.rstrip()
            if not clean.strip():
                continue                     # skip blank lines
            num = start_num + i
            rows.append(f"  {dim}{num:3d}{reset}  {clean}")

    return "\n".join([header] + rows)
