"""
DDQ Engine v12 — Simplified Data Normaliser
downloaded_data_dq/ingestion/normaliser.py

All CSV files (synthetic or raw) follow canonical column schema:
  EOD:      date, open, high, low, close, adj_close, volume, open_interest
  INTRADAY: datetimestamp, open, high, low, close, adj_close, volume, open_interest

This normaliser validates and coerces types. No source-specific transforms.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical columns
EOD_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume", "open_interest"]
INTRADAY_COLUMNS = ["datetimestamp", "open", "high", "low", "close", "adj_close", "volume", "open_interest"]
NUMERIC_COLS = ["open", "high", "low", "close", "adj_close", "volume", "open_interest"]


def normalise_eod(df: pd.DataFrame, ref: str = "") -> pd.DataFrame:
    """
    Validate and normalise an EOD DataFrame to canonical schema.

    Args:
        df:  Raw DataFrame from CSV
        ref: Reference string for logging (e.g. "TCS_DHAN_NSE_EOD_EQUITY")

    Returns:
        Normalised DataFrame, or empty DataFrame on critical failure.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=EOD_COLUMNS)

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Check required date column
    if "date" not in df.columns:
        logger.error("[%s] Missing 'date' column. Found: %s", ref, list(df.columns))
        return pd.DataFrame(columns=EOD_COLUMNS)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    null_dates = df["date"].isna().sum()
    if null_dates > 0:
        logger.warning("[%s] %d unparseable dates dropped", ref, null_dates)
        df = df.dropna(subset=["date"])

    # Fill missing canonical columns
    for col in EOD_COLUMNS:
        if col not in df.columns:
            if col == "adj_close":
                df[col] = df.get("close", 0.0)
            elif col in ("volume", "open_interest"):
                df[col] = 0.0
            else:
                df[col] = np.nan

    # Coerce numerics
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and order canonical columns (plus any extra we want to keep)
    cols = [c for c in EOD_COLUMNS if c in df.columns]
    df = df[cols].copy()

    # Sort by date ascending
    df = df.sort_values("date").reset_index(drop=True)

    return df


def normalise_intraday(df: pd.DataFrame, ref: str = "") -> pd.DataFrame:
    """
    Validate and normalise an Intraday DataFrame to canonical schema.
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Check required timestamp column
    if "datetimestamp" not in df.columns:
        logger.error("[%s] Missing 'datetimestamp' column. Found: %s", ref, list(df.columns))
        return pd.DataFrame(columns=INTRADAY_COLUMNS)

    # Parse timestamp
    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce").dt.tz_localize("Asia/Kolkata").dt.tz_localize(None)
    #df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)            # remove tz info
    #print(df["datetimestamp"].head())

    null_ts = df["datetimestamp"].isna().sum()
    if null_ts > 0:
        logger.warning("[%s] %d unparseable timestamps dropped", ref, null_ts)
        df = df.dropna(subset=["datetimestamp"])

    # Fill missing canonical columns
    for col in INTRADAY_COLUMNS:
        if col not in df.columns:
            if col == "adj_close":
                df[col] = df.get("close", 0.0)
            elif col in ("volume", "open_interest"):
                df[col] = 0.0
            else:
                df[col] = np.nan

    # Coerce numerics
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select canonical columns
    cols = [c for c in INTRADAY_COLUMNS if c in df.columns]
    df = df[cols].copy()

    # Sort by timestamp ascending
    df = df.sort_values("datetimestamp").reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: PRE-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def pre_validate(df: pd.DataFrame, timeframe: str, ref: str = "") -> dict:
    """
    Step 0 pre-validation — run basic timeseries integrity checks
    BEFORE the 248 DQ tests.

    Checks:
      1. Date/timestamp column parseable and consistent format
      2. All dates/timestamps unique (flag duplicates)
      3. All dates/timestamps in ascending order
      4. OHLCV columns castable to float (not text)
      5. adj_close and open_interest columns present and numeric

    Args:
        df:        Normalised DataFrame
        timeframe: "eod" or "intraday"
        ref:       Reference string for logging

    Returns:
        dict with check results:
          {
            "passed": bool,
            "checks": [{"name": str, "status": "OK"|"WARN"|"FAIL", "details": str}, ...],
            "fixes_applied": [str, ...],
            "rows_before": int,
            "rows_after": int,
          }
    """
    result = {
        "passed": True,
        "checks": [],
        "fixes_applied": [],
        "rows_before": len(df),
        "rows_after": len(df),
    }

    if df.empty:
        result["checks"].append({"name": "non_empty", "status": "FAIL", "details": "DataFrame is empty"})
        result["passed"] = False
        return result

    ts_col = "date" if timeframe == "eod" else "datetimestamp"

    # Check 1: Timestamp column present and parseable
    if ts_col not in df.columns:
        result["checks"].append({"name": f"{ts_col}_present", "status": "FAIL",
                                  "details": f"Missing {ts_col} column"})
        result["passed"] = False
        return result

    n_unparsed = df[ts_col].isna().sum()
    if n_unparsed > 0:
        result["checks"].append({"name": f"{ts_col}_parseable", "status": "WARN",
                                  "details": f"{n_unparsed} unparseable values"})
    else:
        result["checks"].append({"name": f"{ts_col}_parseable", "status": "OK",
                                  "details": f"All {len(df)} values parsed"})

    # Check 2: Unique timestamps
    n_dups = df[ts_col].duplicated().sum()
    if n_dups > 0:
        result["checks"].append({"name": f"{ts_col}_unique", "status": "WARN",
                                  "details": f"{n_dups} duplicates found — will be removed"})
        result["fixes_applied"].append(f"Removed {n_dups} duplicate {ts_col} values")
    else:
        result["checks"].append({"name": f"{ts_col}_unique", "status": "OK",
                                  "details": "All timestamps unique"})

    # Check 3: Ascending order
    if not df[ts_col].is_monotonic_increasing:
        result["checks"].append({"name": f"{ts_col}_sorted", "status": "WARN",
                                  "details": "Not in ascending order — will be sorted"})
        result["fixes_applied"].append("Sorted by timestamp ascending")
    else:
        result["checks"].append({"name": f"{ts_col}_sorted", "status": "OK",
                                  "details": "Ascending order confirmed"})

    # Check 4: Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
            if non_numeric > 0:
                result["checks"].append({"name": f"{col}_numeric", "status": "WARN",
                                          "details": f"{non_numeric} non-numeric values"})
            else:
                result["checks"].append({"name": f"{col}_numeric", "status": "OK",
                                          "details": "All numeric"})
        else:
            result["checks"].append({"name": f"{col}_present", "status": "WARN",
                                      "details": f"Column {col} missing — filled with default"})

    # Overall pass/fail
    fails = [c for c in result["checks"] if c["status"] == "FAIL"]
    result["passed"] = len(fails) == 0
    result["rows_after"] = len(df) - n_dups

    return result


def apply_pre_validation_fixes(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Apply the fixes identified by pre_validate: deduplicate, sort, coerce types."""
    if df.empty:
        return df

    ts_col = "date" if timeframe == "eod" else "datetimestamp"

    # Remove duplicate timestamps (keep first)
    if ts_col in df.columns:
        df = df.drop_duplicates(subset=[ts_col], keep="first")
        df = df.sort_values(ts_col).reset_index(drop=True)

    # Coerce all numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
