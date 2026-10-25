"""File tools: Read, Write, Edit, Glob, Grep.

Grep uses ripgrep (rg) as the backend
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from cca.tools.base import Tool

# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class ReadTool(Tool):
    name = "Read"
    description = (
        "Reads a file from the filesystem. Returns content with line numbers in "
        "cat -n format. Supports text files, images (returns description), PDFs, "
        "and Jupyter notebooks. Use offset and limit to read large files in chunks. "
        "Avoid re-reading the same file range within the same turn unless you need "
        "to verify a change you just made; re-reading across turns is appropriate "
        "to catch modifications made by Edit, Write, or Bash."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "description": (
                    "0-based line offset: offset=0 starts at line 1 (the first line), "
                    "offset=49 starts at line 50. Default 0."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
            "pages": {
                "type": "string",
                "description": "For PDFs: page range, e.g. '1-5' or '3'.",
            },
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        offset: int = 0,
        limit: int | None = None,
        pages: str | None = None,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"

        suffix = path.suffix.lower()

        # Images
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
            return self._read_image(path)

        # PDFs
        if suffix == ".pdf":
            return self._read_pdf(path, pages)

        # Jupyter notebooks
        if suffix == ".ipynb":
            return self._read_notebook(path)

        # Text
        return self._read_text(path, offset, limit)

    def _read_text(self, path: Path, offset: int, limit: int | None) -> str:
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        lines = text.splitlines(keepends=True)
        if offset:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        result = []
        for i, line in enumerate(lines, start=offset + 1):
            result.append(f"{i}\t{line}")
        return "".join(result) if result else "(empty file)"

    def _read_image(self, path: Path) -> str:
        size = path.stat().st_size
        return (
            f"[Image file: {path.name}, size: {size} bytes]\n"
            "Note: use a vision-capable model to analyze this image."
        )

    def _read_pdf(self, path: Path, pages: str | None) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            return "Error: pypdf not installed. Run: pip install pypdf"

        try:
            reader = PdfReader(str(path))
            total = len(reader.pages)

            # Parse page range
            if pages:
                page_nums = _parse_page_range(pages, total)
            else:
                page_nums = list(range(total))

            parts = []
            for i in page_nums:
                if 0 <= i < total:
                    text = reader.pages[i].extract_text() or ""
                    parts.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(parts) if parts else "(no text extracted from PDF)"
        except Exception as e:
            return f"Error reading PDF: {e}"

    def _read_notebook(self, path: Path) -> str:
        try:
            import nbformat
            nb = nbformat.read(str(path), as_version=4)
        except Exception as e:
            return f"Error reading notebook: {e}"

        parts = []
        for i, cell in enumerate(nb.cells):
            cell_type = cell.cell_type
            src = cell.source
            parts.append(f"[Cell {i} — {cell_type}]\n{src}")
            if cell.get("outputs"):
                for out in cell.outputs:
                    if out.get("output_type") == "stream":
                        text = "".join(out.get("text", []))
                        parts.append(f"[Output]\n{text.strip()}")
                    elif out.get("output_type") in ("execute_result", "display_data"):
                        data = out.get("data", {})
                        if "text/plain" in data:
                            txt = "".join(data["text/plain"])
                            parts.append(f"[Output]\n{txt.strip()}")
        return "\n\n".join(parts)


def _parse_page_range(spec: str, total: int) -> list[int]:
    result = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a) - 1, int(b)))
        else:
            result.append(int(part) - 1)
    return result


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class WriteTool(Tool):
    name = "Write"
    description = (
        "Writes content to a file, creating parent directories as needed. "
        "Overwrites the entire file. Prefer Edit for modifications to existing files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Written {lines} lines to {file_path}"


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class EditTool(Tool):
    name = "Edit"
    description = (
        "Performs an exact string replacement in a file. "
        "old_string must appear in the file exactly as given (including whitespace). "
        "Set replace_all=true to replace every occurrence. "
        "The edit will fail if old_string is not found or is ambiguous."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "old_string": {"type": "string", "description": "Exact text to find and replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false).",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"

        content = path.read_text()
        count = content.count(old_string)

        if count == 0:
            return f"Error: old_string not found in {file_path}"

        if count > 1 and not replace_all:
            # Show where each occurrence starts so the model can add context
            lines = content.splitlines(keepends=True)
            cumlen = 0
            hit_lines = []
            for lineno, ln in enumerate(lines, 1):
                if old_string.splitlines()[0] in ln:
                    hit_lines.append(lineno)
                cumlen += len(ln)
            locations = ", ".join(f"line {n}" for n in hit_lines[:5])
            return (
                f"Error: old_string appears {count} times in {file_path} "
                f"(at {locations}). "
                "Provide more surrounding context to make it unique, or set replace_all=true."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced = 1

        path.write_text(new_content)
        return _edit_diff_summary(file_path, old_string, new_string, replaced, max_lines=50)


def _edit_diff_summary(
    file_path: str,
    old_string: str,
    new_string: str,
    replaced: int,
    max_lines: int = 50,
) -> str:
    """Return a human-readable +/- diff summary for an Edit result."""
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()
    removed, added = len(old_lines), len(new_lines)

    header = (
        f"{file_path}\n"
        f"Added {added} line(s), removed {removed} line(s)"
        + (f" ({replaced}x)" if replaced > 1 else "")
    )

    diff_lines = (
        [f"- {old_line}" for old_line in old_lines]
        + [f"+ {new_line}" for new_line in new_lines]
    )
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines]
        diff_lines.append("…")

    return header + "\n" + "\n".join(diff_lines)


# ---------------------------------------------------------------------------
# Glob
# ---------------------------------------------------------------------------

class GlobTool(Tool):
    name = "Glob"
    description = (
        "Fast file pattern matching. Returns matching file paths sorted by "
        "modification time (most recent first). "
        "Examples: '**/*.py', 'src/**/*.ts', '*.json'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files against.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to current working directory.",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, pattern: str, path: str | None = None) -> str:
        base = Path(path) if path else Path.cwd()
        if not base.exists():
            return f"Error: directory not found: {base}"

        try:
            matches = list(base.glob(pattern))
        except Exception as e:
            return f"Error: {e}"

        # Sort by modification time, newest first
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        if not matches:
            return f"No files matched pattern '{pattern}' in {base}"

        lines = [str(m) for m in matches]
        result = "\n".join(lines)
        if len(lines) > 500:
            result = "\n".join(lines[:500]) + f"\n... ({len(lines)} total, showing first 500)"
        return result


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------

class GrepTool(Tool):
    name = "Grep"
    description = (
        "Search file contents using regular expressions. Powered by ripgrep (rg). "
        "output_mode controls what is returned: 'auto' (default) returns matching "
        "lines if the result is small (≤50 lines and ≤5000 chars), otherwise falls "
        "back to file names only; 'files_with_matches' always returns only file paths; "
        "'content' always returns matching lines; 'count' shows match counts per file. "
        "Use glob to filter by filename pattern, type to filter by language (e.g. 'py', 'js', 'rust')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search. Defaults to CWD."},
            "glob": {"type": "string", "description": "Glob filter for filenames, e.g. '*.py'."},
            "type": {"type": "string", "description": "ripgrep file type filter, e.g. 'py', 'js'."},
            "output_mode": {
                "type": "string",
                "enum": ["auto", "content", "files_with_matches", "count"],
                "description": "What to return. Default: auto.",
            },
            "-i": {"type": "boolean", "description": "Case-insensitive search."},
            "-n": {"type": "boolean", "description": "Show line numbers (default true for content mode)."},
            "-A": {"type": "integer", "description": "Lines after each match (content mode only)."},
            "-B": {"type": "integer", "description": "Lines before each match (content mode only)."},
            "-C": {"type": "integer", "description": "Lines of context around matches (content mode)."},
            "context": {"type": "integer", "description": "Alias for -C."},
            "multiline": {"type": "boolean", "description": "Enable multiline matching."},
            "head_limit": {"type": "integer", "description": "Limit output to first N lines/entries. Default 250."},
            "offset": {"type": "integer", "description": "Skip first N entries. Default 0."},
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        type: str | None = None,
        output_mode: str = "auto",
        **flags: Any,
    ) -> str:
        search_path = path or str(Path.cwd())
        head_limit: int = flags.get("head_limit", 250) or 250
        offset: int = flags.get("offset", 0) or 0

        if output_mode == "auto":
            # Try content mode; fall back to file mode if result is too large.
            content_result = await self.execute(
                pattern, path=path, glob=glob, type=type,
                output_mode="content", **flags,
            )
            lines = content_result.splitlines()
            if len(lines) <= 50 and len(content_result) <= 5000:
                return content_result
            # Too large — switch to file names only
            files_result = await self.execute(
                pattern, path=path, glob=glob, type=type,
                output_mode="files_with_matches", **flags,
            )
            file_lines = files_result.splitlines()
            if len(file_lines) <= 25:
                return (
                    files_result
                    + f"\n\n[auto mode: result too large for content view "
                    f"({len(lines)} lines); showing file names only]"
                )
            truncated = "\n".join(file_lines[:25])
            extra = len(file_lines) - 25
            return (
                truncated
                + f"\n\n[auto mode: output truncated — {extra} more file(s) not shown. "
                "Use a more specific pattern or set output_mode='files_with_matches' to see all.]"
            )

        # Build rg command
        cmd = ["rg"]

        # Output mode flags
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        # content: default rg behavior

        # Line numbers
        if output_mode == "content" and flags.get("-n", True):
            cmd.append("-n")

        # Case insensitive
        if flags.get("-i"):
            cmd.append("-i")

        # Context
        context = flags.get("context") or flags.get("-C")
        if context and output_mode == "content":
            cmd += ["-C", str(context)]
        else:
            after = flags.get("-A")
            before = flags.get("-B")
            if after and output_mode == "content":
                cmd += ["-A", str(after)]
            if before and output_mode == "content":
                cmd += ["-B", str(before)]

        # Multiline
        if flags.get("multiline"):
            cmd += ["-U", "--multiline-dotall"]

        # File type
        if type:
            cmd += ["--type", type]

        # Glob filter
        if glob:
            cmd += ["--glob", glob]

        cmd += ["--", pattern, search_path]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            # rg not installed — fall back to Python grep
            return self._python_grep(pattern, search_path, output_mode, flags)
        except subprocess.TimeoutExpired:
            return "Error: grep timed out after 30 seconds"

        output = result.stdout
        if not output.strip():
            if result.returncode == 1:
                return f"No matches for '{pattern}'"
            if result.returncode > 1:
                return f"Error: {result.stderr.strip()}"
            return f"No matches for '{pattern}'"

        lines = output.splitlines()
        if offset:
            lines = lines[offset:]
        if head_limit and len(lines) > head_limit:
            shown = lines[:head_limit]
            return "\n".join(shown) + f"\n... ({len(output.splitlines())} total lines, showing first {head_limit})"
        return "\n".join(lines)

    def _python_grep(
        self,
        pattern: str,
        search_path: str,
        output_mode: str,
        flags: dict,
    ) -> str:
        """Fallback when rg is not available."""
        re_flags = re.IGNORECASE if flags.get("-i") else 0
        try:
            rx = re.compile(pattern, re_flags)
        except re.error as e:
            return f"Invalid regex: {e}"

        p = Path(search_path)
        files = [p] if p.is_file() else list(p.rglob("*"))
        results = []

        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue

            lines = text.splitlines()
            file_matches = []
            for lineno, line in enumerate(lines, 1):
                if rx.search(line):
                    file_matches.append((lineno, line))

            if file_matches:
                if output_mode == "files_with_matches":
                    results.append(str(f))
                elif output_mode == "count":
                    results.append(f"{f}:{len(file_matches)}")
                else:
                    for lineno, line in file_matches:
                        results.append(f"{f}:{lineno}:{line}")

        return "\n".join(results) if results else f"No matches for '{pattern}'"


# ---------------------------------------------------------------------------
# SearchInFile
# ---------------------------------------------------------------------------

class SearchInFileTool(Tool):
    name = "SearchInFile"
    description = (
        "Search for a regex pattern within a single file and return each match "
        "with surrounding context lines. Use this instead of reading a large file "
        "in chunks when you need to find a specific function, class, or string — "
        "it jumps directly to the right location without paginating."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "context_lines": {
                "type": "integer",
                "description": "Lines of context to show before and after each match. Default 10.",
            },
        },
        "required": ["file_path", "pattern"],
    }

    async def execute(
        self,
        file_path: str,
        pattern: str,
        context_lines: int = 10,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        if not path.is_file():
            return f"Error: not a file: {file_path}"

        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as e:
            return f"Error reading file: {e}"

        blocks: list[str] = []
        seen_ranges: list[tuple[int, int]] = []

        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            # Merge with previous block if they overlap
            if seen_ranges and start <= seen_ranges[-1][1]:
                prev_start, _ = seen_ranges[-1]
                seen_ranges[-1] = (prev_start, end)
                blocks[-1] = "\n".join(
                    f"{j + 1:5}: {lines[j]}" for j in range(prev_start, end)
                )
            else:
                seen_ranges.append((start, end))
                blocks.append(
                    f"--- line {i + 1} ---\n"
                    + "\n".join(f"{j + 1:5}: {lines[j]}" for j in range(start, end))
                )

        if not blocks:
            return f"No matches for '{pattern}' in {file_path}"
        return "\n\n".join(blocks)
