# tests/cli/test_cli_help.py
from __future__ import annotations

import re
from typer.testing import CliRunner

from hqp.cli import app

runner = CliRunner()


def test_cli_help_lists_commands():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    text = res.stdout
    for cmd in [
        "ingest",
        "validate",
        "split",
        "ratings",
        "features",
        "train",
        "evaluate-model",
        "market",
        "evaluate-market",
        "edge",
        "backtest",
        "report",
        "run-all",
    ]:
        assert re.search(rf"\b{re.escape(cmd)}\b", text), f"missing {cmd} in --help"
