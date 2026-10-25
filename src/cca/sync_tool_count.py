"""Synchronize public tool-count claims with CCA's built-in tool catalog."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cca.features import BUILTIN_TOOL_COUNT


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    ROOT / "README.md": (
        (r"built--in%20tools-\d+", f"built--in%20tools-{BUILTIN_TOOL_COUNT}"),
        (
            r"\d+ available built-in tools",
            f"{BUILTIN_TOOL_COUNT} available built-in tools",
        ),
    ),
    ROOT / "website/src/content/docs/blog/introducing-cca.md": (
        (
            r"\d+ available built-in tools",
            f"{BUILTIN_TOOL_COUNT} available built-in tools",
        ),
    ),
}


def synchronized_text(path: Path) -> str:
    text = path.read_text()
    for pattern, replacement in TARGETS[path]:
        text, replacements = re.subn(pattern, replacement, text)
        if replacements != 1:
            raise RuntimeError(
                f"Expected one tool-count marker matching {pattern!r} in {path}"
            )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting files when public counts are stale.",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for path in TARGETS:
        rendered = synchronized_text(path)
        if rendered == path.read_text():
            continue
        stale.append(path)
        if not args.check:
            path.write_text(rendered)

    if stale and args.check:
        for path in stale:
            print(f"stale tool count: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
