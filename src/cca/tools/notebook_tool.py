"""NotebookEdit — replace/insert/delete cells in Jupyter notebooks."""

from __future__ import annotations

from pathlib import Path

from cca.tools.base import Tool


class NotebookEditTool(Tool):
    name = "NotebookEdit"
    description = (
        "Replaces, inserts, or deletes a cell in a Jupyter notebook (.ipynb). "
        "edit_mode='replace' (default): replace cell source at cell_number. "
        "edit_mode='insert': insert a new cell after cell_id, or at beginning. "
        "edit_mode='delete': delete the cell at cell_number. "
        "Cell numbers are 0-indexed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "notebook_path": {
                "type": "string",
                "description": "Absolute path to the .ipynb file.",
            },
            "new_source": {
                "type": "string",
                "description": "New source content for the cell.",
            },
            "cell_id": {
                "type": "string",
                "description": "Cell ID or 0-indexed cell number. For insert, new cell goes after this.",
            },
            "cell_type": {
                "type": "string",
                "enum": ["code", "markdown"],
                "description": "Cell type (required for insert).",
            },
            "edit_mode": {
                "type": "string",
                "enum": ["replace", "insert", "delete"],
                "description": "Type of edit. Default: replace.",
            },
        },
        "required": ["notebook_path", "new_source"],
    }

    async def execute(
        self,
        notebook_path: str,
        new_source: str,
        cell_id: str | None = None,
        cell_type: str | None = None,
        edit_mode: str = "replace",
    ) -> str:
        try:
            import nbformat
        except ImportError:
            return "Error: nbformat not installed. Run: pip install nbformat"

        path = Path(notebook_path)
        if not path.exists():
            return f"Error: notebook not found: {notebook_path}"

        nb = nbformat.read(str(path), as_version=4)
        cells = nb.cells

        if edit_mode == "delete":
            idx = self._resolve_idx(cells, cell_id)
            if idx is None:
                return f"Error: cell '{cell_id}' not found"
            del cells[idx]
            nbformat.write(nb, str(path))
            return f"Deleted cell {cell_id} from {notebook_path}"

        elif edit_mode == "insert":
            ct = cell_type or "code"
            new_cell = nbformat.v4.new_code_cell(source=new_source) if ct == "code" \
                else nbformat.v4.new_markdown_cell(source=new_source)
            if cell_id is None:
                cells.insert(0, new_cell)
            else:
                idx = self._resolve_idx(cells, cell_id)
                if idx is None:
                    cells.append(new_cell)
                else:
                    cells.insert(idx + 1, new_cell)
            nbformat.write(nb, str(path))
            return f"Inserted {ct} cell into {notebook_path}"

        else:  # replace
            idx = self._resolve_idx(cells, cell_id)
            if idx is None:
                return f"Error: cell '{cell_id}' not found"
            cells[idx]["source"] = new_source
            nbformat.write(nb, str(path))
            return f"Replaced source of cell {cell_id} in {notebook_path}"

    def _resolve_idx(self, cells: list, cell_id: str | None) -> int | None:
        if cell_id is None:
            return 0 if cells else None
        # Try as integer index
        try:
            idx = int(cell_id)
            if 0 <= idx < len(cells):
                return idx
        except ValueError:
            pass
        # Try as cell id string
        for i, cell in enumerate(cells):
            if cell.get("id") == cell_id:
                return i
        return None
