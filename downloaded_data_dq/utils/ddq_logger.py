"""
Downloaded Data DQ Engine — Dual-Output Logger
downloaded_data_dq/utils/ddq_logger.py

Writes to:
  1. Terminal  — ANSI colours
  2. Log file  — plain text, ANSI stripped, filename PREFIXED with RUN_ID
                 e.g.  logs/{RUN_ID}_ddq.log
"""
from __future__ import annotations
import logging, re, sys
from datetime import datetime
from pathlib import Path

_current_log_path: Path | None = None


def get_log_path() -> Path | None:
    return _current_log_path


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    def format(self, record):
        return _ANSI_RE.sub("", super().format(record))


class _ColourTerminalFormatter(logging.Formatter):
    pass


def setup_ddq_logger(
    log_dir: str | Path = "logs",
    run_id: str | None = None,
    logger_name: str = "ddq",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create the DDQ dual-output logger.

    Log filename:  {log_dir}/{run_id}_ddq.log   (RUN_ID prefixed)
    If run_id is None, falls back to timestamp prefix.
    """
    global _current_log_path

    ddq_log = logging.getLogger(logger_name)
    ddq_log.setLevel(level)
    ddq_log.propagate = False
    ddq_log.handlers.clear()

    # Terminal handler (ANSI colours intact)
    term = logging.StreamHandler(sys.stdout)
    term.setLevel(level)
    term.setFormatter(_ColourTerminalFormatter(fmt="%(message)s"))
    ddq_log.addHandler(term)

    # File handler (ANSI stripped, RUN_ID prefixed)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    prefix = run_id if run_id else datetime.now().strftime("%Y%m%d%H%M")
    log_file = log_dir_path / f"{prefix}_ddq.log"
    _current_log_path = log_file

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_StripAnsiFormatter(fmt="%(message)s"))
    ddq_log.addHandler(fh)

    return ddq_log
