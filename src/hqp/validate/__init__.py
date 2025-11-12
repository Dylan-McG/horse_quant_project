# src/hqp/validate/__init__.py
# -----------------------------------------------------------------------------
# Public surface of the validation package.
#
# Notes on CLI integration:
# - The CLI tries `from hqp.validate import run` (see cli.validate). We do not
#   export a `run` function here; that import will fail and the CLI will execute
#   a minimal fallback check (keys exist) — by design.
# - We do expose `validate_canonical`, which can be called directly from tests
#   or ad-hoc scripts to validate the canonical runner-level table. It is not
#   currently wired to the CLI's `validate` command.
# -----------------------------------------------------------------------------
from __future__ import annotations

from .expectations import validate_canonical

__all__ = ["validate_canonical"]
