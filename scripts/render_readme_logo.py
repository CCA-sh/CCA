"""Render the README's ASCII CCA logo as font-independent light and dark SVGs."""

import sys
from pathlib import Path

LOGO = """\
 ██████╗ ██████╗ █████╗
██╔════╝██╔════╝██╔══██╗
██║     ██║     ███████║
██║     ██║     ██╔══██║
╚██████╗╚██████╗██║  ██║
 ╚═════╝ ╚═════╝╚═╝  ╚═╝"""

CELL_W, CELL_H = 12, 24
PAD = 24
NEAR, FAR = 0.3, 0.7  # positions of the two strokes in a double-line glyph
LETTER_COLS = 8
SHADOW_COLORS = ("#02c39a", "#ea638c", "#067bc2")  # one per letter: C, C, A


def shadow_paths(ch, x, y):
    w, h = CELL_W, CELL_H
    x1, x2 = x + w * NEAR, x + w * FAR
    y1, y2 = y + h * 0.4, y + h * 0.6
    r, b = x + w, y + h
    shapes = {
        "═": [f"M{x},{y1}H{r}", f"M{x},{y2}H{r}"],
        "║": [f"M{x1},{y}V{b}", f"M{x2},{y}V{b}"],
        "╔": [f"M{r},{y1}H{x1}V{b}", f"M{r},{y2}H{x2}V{b}"],
        "╗": [f"M{x},{y1}H{x2}V{b}", f"M{x},{y2}H{x1}V{b}"],
        "╚": [f"M{x1},{y}V{y2}H{r}", f"M{x2},{y}V{y1}H{r}"],
        "╝": [f"M{x2},{y}V{y2}H{x}", f"M{x1},{y}V{y1}H{x}"],
    }
    return shapes.get(ch, [])


def render(block_color):
    lines = LOGO.splitlines()
    cols = max(len(line) for line in lines)
    width, height = cols * CELL_W + 2 * PAD, len(lines) * CELL_H + 2 * PAD
    rects = []
    paths = [[] for _ in SHADOW_COLORS]
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            x, y = PAD + col * CELL_W, PAD + row * CELL_H
            if ch == "█":
                rects.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}"/>')
            paths[col // LETTER_COLS].extend(shadow_paths(ch, x, y))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="CCA">\n'
        f'<g fill="none" stroke-width="2" stroke-linecap="square">'
        + "".join(
            f'<path stroke="{color}" d="{" ".join(letter_paths)}"/>'
            for color, letter_paths in zip(SHADOW_COLORS, paths)
        )
        + "</g>\n"
        f'<g fill="{block_color}" shape-rendering="crispEdges">{"".join(rects)}</g>\n'
        "</svg>\n"
    )


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "cca-logo-dark.svg").write_text(render("#f0f6fc"))
(out / "cca-logo-light.svg").write_text(render("#1f2328"))
