# src/hqp/common/logging.py
# -----------------------------------------------------------------------------
# Simple, dependency-light logging & run-ledger utilities used by ingestion.
# - log_stdout: timestamped console output
# - sha256_file: content hash for provenance
# - RunRecord / RunLedger: append-only CSV of ingest runs
# - Timed: minimal context manager for durations
# -----------------------------------------------------------------------------
from __future__ import annotations

import csv
import hashlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType


@dataclass
class RunRecord:
    ts: datetime
    input_path: str
    output_path: str
    rows_in: int
    rows_out: int
    n_dupes_removed: int
    sha256_out: str
    duration_s: float


class RunLedger:
    """Append-only CSV ledger for ingest runs (best-effort provenance)."""

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: RunRecord) -> None:
        write_header = not self.ledger_path.exists()
        with self.ledger_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "ts",
                        "input_path",
                        "output_path",
                        "rows_in",
                        "rows_out",
                        "n_dupes_removed",
                        "sha256_out",
                        "duration_s",
                    ]
                )
            w.writerow(
                [
                    rec.ts.isoformat(timespec="seconds"),
                    rec.input_path,
                    rec.output_path,
                    rec.rows_in,
                    rec.rows_out,
                    rec.n_dupes_removed,
                    rec.sha256_out,
                    f"{rec.duration_s:.3f}",
                ]
            )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (chunked) for reproducibility/provenance."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log_stdout(msg: str) -> None:
    """Timestamped line to stdout (no external logging dependency)."""
    ts = datetime.now().isoformat(timespec="seconds")
    sys.stdout.write(f"[{ts}] {msg}\n")
    sys.stdout.flush()


class Timed:
    """Context manager to measure durations of small blocks."""

    def __init__(self) -> None:
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Timed:
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        end = time.perf_counter()
        self.elapsed = end - (self.start or end)
