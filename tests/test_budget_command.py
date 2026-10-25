"""Regression tests for the /budget slash command.

Guards against a regression introduced while tightening the inline
`/budget <amount>` token-budget parser: `_handle_slash`'s `/budget` branch
used to call `_parse_token_budget(arg)` on the bare amount (e.g. "500k"),
which broke once that parser started requiring the `/budget` sigil itself
to be present in the text. The slash command now calls the dedicated
`_parse_budget_amount` helper instead.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from rich.console import Console

from cca.repl import Repl


def _make_repl() -> tuple[Repl, io.StringIO]:
    repl = Repl.__new__(Repl)
    repl._ui = MagicMock()
    repl._token_budget = None
    repl.config = MagicMock(max_tokens=8192)
    output = io.StringIO()
    repl._console = Console(file=output, force_terminal=False, width=160)
    return repl, output


async def test_budget_k_suffix_sets_token_budget():
    repl, output = _make_repl()

    handled = await repl._handle_slash("/budget 500k")

    assert handled is True
    assert repl._token_budget == 500_000
    assert "500,000" in output.getvalue()


async def test_budget_plain_integer_sets_token_budget():
    repl, output = _make_repl()

    handled = await repl._handle_slash("/budget 200000")

    assert handled is True
    assert repl._token_budget == 200_000


async def test_budget_m_suffix_sets_token_budget():
    repl, output = _make_repl()

    handled = await repl._handle_slash("/budget 2.5m")

    assert handled is True
    assert repl._token_budget == 2_500_000


async def test_budget_invalid_arg_shows_usage():
    repl, output = _make_repl()

    handled = await repl._handle_slash("/budget banana")

    assert handled is True
    assert repl._token_budget is None
    assert "Usage" in output.getvalue()


async def test_budget_no_arg_shows_current_or_default():
    repl, output = _make_repl()

    handled = await repl._handle_slash("/budget")

    assert handled is True
    assert "No token budget set" in output.getvalue()
