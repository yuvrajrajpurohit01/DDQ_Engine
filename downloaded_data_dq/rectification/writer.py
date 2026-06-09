"""
DDQ Engine — Rectification Data Writer
downloaded_data_dq/rectification/writer.py

Writes rectified DataFrames to data/rectified/{RUN_ID}/ preserving
the original directory structure.  Original data is NEVER modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_rectified(
    project_root: Path,
    run_id: str,
    rectified_frames: dict,   # {(symbol, source, exchange, timeframe): DataFrame}
) -> Path:
    """
    Write all rectified DataFrames to disk.

    Returns the output directory path.
    """
    out_dir = project_root / "data" / "rectified" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for (symbol, source, exchange, timeframe), df in rectified_frames.items():
        if df is None or df.empty:
            continue
        tf_dir = out_dir / timeframe.upper() / source.capitalize()/exchange.upper()/"EQUITY"
        tf_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{symbol}.csv"
        fpath = tf_dir / fname
        df["date"] = df["date"].dt.strftime("%d-%m-%Y")
        df.to_csv(fpath, index=False)
        written += 1
        logger.info("  Wrote rectified: %s (%d rows)", fpath.relative_to(project_root), len(df))

    logger.info("  Total rectified files written: %d to %s", written, out_dir.relative_to(project_root))
    return out_dir
