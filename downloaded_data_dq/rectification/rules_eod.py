"""
DDQ Engine — EOD Rectification Rules
downloaded_data_dq/rectification/rules_eod.py

24 rules for automated rectification of EOD time-series data issues.
Each rule is decorated with @rect_rule() and registered automatically.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any

from downloaded_data_dq.rectification.registry import RuleSpec, rect_rule
from downloaded_data_dq.rectification.audit import AuditEntry, RectificationResult

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _ae(rule_id, sym, src, exch, tf, row, col, old, new, reason, conf):
    return AuditEntry(
        rule_id=rule_id, symbol=sym, source=src, exchange=exch,
        timeframe=tf, row_index=int(row), column=col,
        old_value=str(old), new_value=str(new),
        reason=reason, confidence=conf,
    )


def _make_result(rule_id, test_id, sym, src, tf, exch):
    return RectificationResult(
        rule_id=rule_id, test_id=test_id, symbol=sym,
        source=src, timeframe=tf, exchange=exch,
    )


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-001  Null Value Fill
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-001", test_ids=("EOD-001",), name="Null Value Fill",
    timeframe="EOD", priority=10, default_conf=0.95,
    description="Forward-fill then backward-fill for price columns; volume nulls -> 0; date nulls -> drop row."))
def rect_eod_001(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-001", "EOD-001", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.95)
    entries = []
    df = df.copy()

    # Drop rows with null dates
    null_dates = df["date"].isna()
    if null_dates.any():
        for idx in df.index[null_dates]:
            entries.append(_ae("RECT-EOD-001", symbol, source, exchange, "EOD",
                              idx, "date", "NaT", "DROPPED", "Null date row dropped", conf))
        df = df[~null_dates].reset_index(drop=True)

    # Fill price columns: ffill then bfill
    for col in PRICE_COLS:
        if col in df.columns:
            mask = df[col].isna()
            if mask.any():
                old_vals = df.loc[mask, col].copy()
                df[col] = df[col].ffill().bfill()
                for idx in old_vals.index:
                    entries.append(_ae("RECT-EOD-001", symbol, source, exchange, "EOD",
                                      idx, col, "NaN", str(df.at[idx, col]),
                                      f"Null {col} filled via ffill/bfill", conf))

    # Fill volume nulls with 0
    if "volume" in df.columns:
        mask = df["volume"].isna()
        if mask.any():
            for idx in df.index[mask]:
                entries.append(_ae("RECT-EOD-001", symbol, source, exchange, "EOD",
                                  idx, "volume", "NaN", "0", "Null volume set to 0", conf))
            df.loc[mask, "volume"] = 0

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    if entries:
        r.action = "FIXED"
        r.details = f"Filled {len(entries)} null values across OHLCV columns"
    else:
        r.action = "SKIPPED"
        r.details = "No null values found"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-002  Missing Date Insertion
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-002", test_ids=("EOD-002",), name="Missing Date Insertion",
    timeframe="EOD", priority=20, default_conf=0.85,
    description="Insert rows for missing trading dates with flat prices from previous close."))
def rect_eod_002(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-002", "EOD-002", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.85)
    max_gap = config.get("max_gap_days", 5)
    #print(f"Max gap days for {symbol}: {max_gap}")
    entries = []
    df = df.copy()

    if "date" not in df.columns or df.empty:
        r.action = "SKIPPED"; r.details = "No date column or empty"; return df, r

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    full_range = pd.bdate_range(df["date"].min(), df["date"].max())
    existing = set(df["date"].dt.normalize())
    missing = [d for d in full_range if d not in existing]

    #print(f"Missing dates {missing} and {len(missing)} dates for {symbol}")

    # Generate complete range and find gaps
    all_dates = pd.date_range(start=df["date"].min(), end=df["date"].max())
    missing1 = all_dates.difference(df["date"])
    # Group consecutive gaps
    gap_groups = (missing1.to_series().diff() > pd.Timedelta(days=1)).cumsum()
    #max_gap = missing1.to_series().groupby(gap_groups).count().max()

    #print(f"Max gap days for {symbol}: {max_gap}")

    #print(f"Max continuous missing: {max_gap}")

    if len(missing) > max_gap * 50:
        r.action = "FLAGGED"
        r.details = f"Too many missing dates ({len(missing)}) - manual review needed"
        r.confidence = conf * 0.5
        return df, r

    new_rows = []
    for md in missing:
        prev_rows = df[df["date"] < md]
        if prev_rows.empty:
            continue
        prev = prev_rows.iloc[-1]
        row = {"date": md, "open": prev["close"], "high": prev["close"],
               "low": prev["close"], "close": prev["close"], "volume": 0}
        if "adj_close" in df.columns:
            row["adj_close"] = prev.get("adj_close", prev["close"])
        if "_rectified_by" in df.columns:
            row["_rectified_by"] = "RECT-EOD-002"
        new_rows.append(row)
        entries.append(_ae("RECT-EOD-002", symbol, source, exchange, "EOD",
                          -1, "date", "MISSING", str(md.date()),
                          f"Inserted missing date with flat price {prev['close']}", conf))

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Inserted {len(entries)} missing trading dates" if entries else "No missing dates"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-003  Duplicate Row Removal
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-003", test_ids=("EOD-003",), name="Duplicate Row Removal",
    timeframe="EOD", priority=5, default_conf=0.98,
    description="Remove exact duplicate rows, keeping first occurrence."))
def rect_eod_003(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-003", "EOD-003", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.98)
    entries = []
    before_len = len(df)
    df = df.copy()

    ## Added by Dk on 04052026
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(by=["date", "close"], ascending=[True, True])
    dup_mask = df.duplicated(subset=["date"], keep="first")
    ## END

    #dup_mask = df.duplicated(keep="first")
    if dup_mask.any():
        for idx in df.index[dup_mask]:
            entries.append(_ae("RECT-EOD-003", symbol, source, exchange, "EOD",
                              idx, "ROW", "duplicate", "REMOVED",
                              "Exact duplicate row removed (kept first)", conf))
        df = df[~dup_mask].reset_index(drop=True)

    # Also handle same-date duplicates (keep first by date)
    if "date" in df.columns:
        date_dup = df.duplicated(subset=["date"], keep="first")
        if date_dup.any():
            for idx in df.index[date_dup]:
                entries.append(_ae("RECT-EOD-003", symbol, source, exchange, "EOD",
                                  idx, "ROW", f"date_dup:{df.at[idx, 'date']}", "REMOVED",
                                  "Same-date duplicate removed (kept first)", conf * 0.95))
            df = df[~date_dup].reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Removed {len(entries)} duplicate rows ({before_len} -> {len(df)})" if entries else "No duplicates"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-004  Negative/Zero Price Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-004", test_ids=("EOD-004",), name="Negative/Zero Price Fix",
    timeframe="EOD", priority=15, default_conf=0.90,
    description="abs() for negative prices; ffill for zero prices."))
def rect_eod_004(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-004", "EOD-004", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.90)
    entries = []
    df = df.copy()

    for col in PRICE_COLS:
        if col not in df.columns:
            continue
        # Fix negatives
        neg_mask = df[col] < 0
        if neg_mask.any():
            for idx in df.index[neg_mask]:
                old_val = df.at[idx, col]
                new_val = abs(old_val)
                df.at[idx, col] = new_val
                entries.append(_ae("RECT-EOD-004", symbol, source, exchange, "EOD",
                                  idx, col, str(old_val), str(new_val),
                                  f"Negative {col} -> abs()", conf))
        # Fix zeros with ffill
        zero_mask = df[col] == 0
        if zero_mask.any():
            for idx in df.index[zero_mask]:
                old_val = 0.0
                df.at[idx, col] = np.nan
            df[col] = df[col].ffill().bfill()
            for idx in df.index[zero_mask]:
                entries.append(_ae("RECT-EOD-004", symbol, source, exchange, "EOD",
                                  idx, col, "0.0", str(df.at[idx, col]),
                                  f"Zero {col} filled from adjacent row", conf * 0.9))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} negative/zero prices" if entries else "No negative/zero prices"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-005  High-Low Inversion Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-005", test_ids=("EOD-005",), name="High-Low Inversion Fix",
    timeframe="EOD", priority=12, default_conf=0.99,
    description="Swap High and Low when High < Low."))
def rect_eod_005(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-005", "EOD-005", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.99)
    entries = []
    df = df.copy()

    if "high" in df.columns and "low" in df.columns:
        mask = df["high"] < df["low"]
        if mask.any():
            for idx in df.index[mask]:
                old_h, old_l = df.at[idx, "high"], df.at[idx, "low"]
                entries.append(_ae("RECT-EOD-005", symbol, source, exchange, "EOD",
                                  idx, "high", str(old_h), str(old_l),
                                  f"High-Low inversion: swapped high={old_h} with low={old_l}", conf))
                entries.append(_ae("RECT-EOD-005", symbol, source, exchange, "EOD",
                                  idx, "low", str(old_l), str(old_h),
                                  f"High-Low inversion: swapped low={old_l} with high={old_h}", conf))
            # Vectorised swap
            inv = df.loc[mask, ["high", "low"]].copy()
            df.loc[mask, "high"] = inv["low"]
            df.loc[mask, "low"] = inv["high"]

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries) // 2
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Swapped {len(entries)//2} High-Low inversions" if entries else "No inversions"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-006  Close Outside Range Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-006", test_ids=("EOD-006",), name="Close Outside Range Fix",
    timeframe="EOD", priority=14, default_conf=0.92,
    description="Clamp Close to [Low, High] range."))
def rect_eod_006(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-006", "EOD-006", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.92)
    entries = []
    df = df.copy()

    for c in ["close", "open"]:
        if c not in df.columns or "low" not in df.columns or "high" not in df.columns:
            continue
        above = df[c] > df["high"]
        below = df[c] < df["low"]
        for idx in df.index[above]:
            old = df.at[idx, c]
            df.at[idx, c] = df.at[idx, "high"]
            entries.append(_ae("RECT-EOD-006", symbol, source, exchange, "EOD",
                              idx, c, str(old), str(df.at[idx, c]),
                              f"{c} above high: clamped {old} -> {df.at[idx, c]}", conf))
        for idx in df.index[below]:
            old = df.at[idx, c]
            df.at[idx, c] = df.at[idx, "low"]
            entries.append(_ae("RECT-EOD-006", symbol, source, exchange, "EOD",
                              idx, c, str(old), str(df.at[idx, c]),
                              f"{c} below low: clamped {old} -> {df.at[idx, c]}", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Clamped {len(entries)} out-of-range prices" if entries else "All prices in range"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-007  Zero/Negative Volume Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-007", test_ids=("EOD-007",), name="Zero/Negative Volume Fix",
    timeframe="EOD", priority=25, default_conf=0.80,
    description="Replace zero/negative volume with rolling median of surrounding window."))
def rect_eod_007(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-007", "EOD-007", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.80)
    entries = []
    df = df.copy()

    if "volume" not in df.columns:
        r.action = "SKIPPED"; r.details = "No volume column"; return df, r

    bad_mask = (df["volume"] <= 0) | df["volume"].isna()
    if bad_mask.any():
        rolling_med = df["volume"].rolling(window=5, min_periods=1, center=True).median()
        for idx in df.index[bad_mask]:
            old_val = df.at[idx, "volume"]
            new_val = rolling_med.at[idx] if pd.notna(rolling_med.at[idx]) and rolling_med.at[idx] > 0 else 0
            new_val = int(round(new_val))  # ← median is float, volume is int64
            df.at[idx, "volume"] = new_val
            entries.append(_ae("RECT-EOD-007", symbol, source, exchange, "EOD",
                              idx, "volume", str(old_val), str(new_val),
                              "Zero/negative volume replaced with rolling median", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} volume values" if entries else "No volume issues"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-008  Stale Price Break
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-008", test_ids=("EOD-008",), name="Stale Price Flag",
    timeframe="EOD", priority=30, default_conf=0.70,
    description="Flag runs of identical OHLC for manual review."))
def rect_eod_008(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-008", "EOD-008", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.70)
    max_stale = config.get("max_stale_days", 3)
    entries = []
    df = df.copy()

    if not all(c in df.columns for c in PRICE_COLS):
        r.action = "SKIPPED"; r.details = "Missing OHLC columns"; return df, r

    # Detect consecutive identical OHLC
    same = (df[PRICE_COLS].diff() == 0).all(axis=1)
    groups = (same != same.shift()).cumsum()
    for _, grp in df[same].groupby(groups[same]):
        if len(grp) >= max_stale:
            for idx in grp.index:
                entries.append(_ae("RECT-EOD-008", symbol, source, exchange, "EOD",
                                  idx, "OHLC", "stale", "FLAGGED",
                                  f"Stale OHLC repeated {len(grp)} consecutive days", conf))

    r.audit_entries = entries
    r.changes_count = 0   # flagging, not fixing
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    r.action = "FLAGGED" if entries else "SKIPPED"
    r.details = f"Flagged {len(entries)} stale-price rows for review" if entries else "No stale prices"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-009  Abnormal Return Cap
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-009", test_ids=("EOD-009",), name="Abnormal Return Cap",
    timeframe="EOD", priority=35, default_conf=0.75,
    description="Cap single-day returns at configurable threshold with circuit-breaker awareness."))
def rect_eod_009(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-009", "EOD-009", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.75)
    max_ret = config.get("max_return_pct", 20.0) / 100.0
    entries = []
    df = df.copy()
    df["close"] = df["close"].astype("float64")

    if "close" not in df.columns or len(df) < 2:
        r.action = "SKIPPED"; r.details = "Insufficient data"; return df, r

    rets = df["close"].pct_change()
    extreme = rets.abs() > max_ret
    for idx in df.index[extreme]:
        if idx == df.index[0]:
            continue
        old_close = df.at[idx, "close"]
        prev_close = df.at[idx - 1, "close"] if (idx - 1) in df.index else df["close"].iloc[df.index.get_loc(idx) - 1]
        direction = 1 if rets.at[idx] > 0 else -1
        capped_close = prev_close * (1 + direction * max_ret)
        df.at[idx, "close"] = round(capped_close, 2)
        entries.append(_ae("RECT-EOD-009", symbol, source, exchange, "EOD",
                          idx, "close", str(old_close), str(df.at[idx, "close"]),
                          f"Return {rets.at[idx]*100:.1f}% capped to {direction*max_ret*100:.0f}%", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Capped {len(entries)} extreme returns" if entries else "No extreme returns"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-010  OHLC Consistency Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-010", test_ids=("EOD-010",), name="OHLC Consistency Fix",
    timeframe="EOD", priority=13, default_conf=0.90,
    description="Ensure Low <= min(O,C) and High >= max(O,C)."))
def rect_eod_010(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-010", "EOD-010", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.90)
    entries = []
    df = df.copy()

    if not all(c in df.columns for c in PRICE_COLS):
        r.action = "SKIPPED"; r.details = "Missing OHLC"; return df, r

    # Low should be <= min(open, close)
    min_oc = df[["open", "close"]].min(axis=1)
    bad_low = df["low"] > min_oc
    for idx in df.index[bad_low]:
        old = df.at[idx, "low"]
        df.at[idx, "low"] = min_oc.at[idx]
        entries.append(_ae("RECT-EOD-010", symbol, source, exchange, "EOD",
                          idx, "low", str(old), str(df.at[idx, "low"]),
                          f"Low > min(O,C): adjusted {old} -> {df.at[idx, 'low']}", conf))

    # High should be >= max(open, close)
    max_oc = df[["open", "close"]].max(axis=1)
    bad_high = df["high"] < max_oc
    for idx in df.index[bad_high]:
        old = df.at[idx, "high"]
        df.at[idx, "high"] = max_oc.at[idx]
        entries.append(_ae("RECT-EOD-010", symbol, source, exchange, "EOD",
                          idx, "high", str(old), str(df.at[idx, "high"]),
                          f"High < max(O,C): adjusted {old} -> {df.at[idx, 'high']}", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} OHLC consistency issues" if entries else "OHLC consistent"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-011  Adj Close Fill
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-011", test_ids=("EOD-011",), name="Adj Close Fill",
    timeframe="EOD", priority=40, default_conf=0.85,
    description="Fill missing adj_close from close price."))
def rect_eod_011(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-011", "EOD-011", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.85)
    entries = []
    df = df.copy()

    if "adj_close" not in df.columns:
        if "close" in df.columns:
            df["adj_close"] = df["close"]
            entries.append(_ae("RECT-EOD-011", symbol, source, exchange, "EOD",
                              0, "adj_close", "MISSING_COL", "=close",
                              "Created adj_close column from close", conf))
    else:
        mask = df["adj_close"].isna()
        if mask.any() and "close" in df.columns:
            for idx in df.index[mask]:
                df.at[idx, "adj_close"] = df.at[idx, "close"]
                entries.append(_ae("RECT-EOD-011", symbol, source, exchange, "EOD",
                                  idx, "adj_close", "NaN", str(df.at[idx, "close"]),
                                  "Null adj_close filled from close", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Filled {len(entries)} adj_close values" if entries else "adj_close complete"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-012  OI Non-Negative
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-012", test_ids=("EOD-012",), name="OI Non-Negative Fix",
    timeframe="EOD", priority=42, default_conf=0.95,
    description="abs() for negative OI; 0-fill for NaN."))
def rect_eod_012(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-012", "EOD-012", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.95)
    entries = []
    df = df.copy()

    for col in ["open_interest", "oi"]:
        if col not in df.columns:
            continue
        neg = df[col] < 0
        for idx in df.index[neg]:
            old = df.at[idx, col]; df.at[idx, col] = abs(old)
            entries.append(_ae("RECT-EOD-012", symbol, source, exchange, "EOD",
                              idx, col, str(old), str(df.at[idx, col]), "Negative OI -> abs()", conf))
        na = df[col].isna()
        if na.any():
            df.loc[na, col] = 0
            for idx in df.index[na]:
                entries.append(_ae("RECT-EOD-012", symbol, source, exchange, "EOD",
                                  idx, col, "NaN", "0", "Null OI -> 0", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} OI values" if entries else "OI OK"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-014  Volume Outlier Winsorize
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-014", test_ids=("EOD-014",), name="Volume Outlier Winsorize",
    timeframe="EOD", priority=45, default_conf=0.70,
    description="Winsorize volume outliers beyond z-score threshold."))
def rect_eod_014(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-014", "EOD-014", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.70)
    z_thresh = config.get("z_score_threshold", 4.0)
    entries = []
    df = df.copy()

    if "volume" not in df.columns or df["volume"].std() == 0:
        r.action = "SKIPPED"; r.details = "No volume variance"; return df, r

    vol = df["volume"].astype(float)
    z = (vol - vol.mean()) / vol.std()
    outlier = z.abs() > z_thresh
    if outlier.any():
        cap_high = vol.mean() + z_thresh * vol.std()
        cap_low = max(0, vol.mean() - z_thresh * vol.std())
        for idx in df.index[outlier]:
            old = df.at[idx, "volume"]
            new_val = int(cap_high if z.at[idx] > 0 else cap_low)
            df.at[idx, "volume"] = new_val
            entries.append(_ae("RECT-EOD-014", symbol, source, exchange, "EOD",
                              idx, "volume", str(old), str(new_val),
                              f"Volume outlier (z={z.at[idx]:.1f}) winsorized", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Winsorized {len(entries)} volume outliers" if entries else "No volume outliers"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-015  Price Outlier Winsorize
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-015", test_ids=("EOD-015",), name="Price Outlier Winsorize",
    timeframe="EOD", priority=46, default_conf=0.65,
    description="Winsorize close return outliers beyond z-score threshold."))
def rect_eod_015(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-015", "EOD-015", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.65)
    z_thresh = config.get("z_score_threshold", 4.0)
    entries = []
    df = df.copy()
    df["close"] = df["close"].astype("float64")

    if "close" not in df.columns or len(df) < 10:
        r.action = "SKIPPED"; return df, r

    rets = df["close"].pct_change()
    if rets.std() == 0 or pd.isna(rets.std()):
        r.action = "SKIPPED"; return df, r

    z = (rets - rets.mean()) / rets.std()
    outlier = (z.abs() > z_thresh) & (rets.index > rets.index[0])
    if outlier.any():
        for idx in df.index[outlier]:
            pos = df.index.get_loc(idx)
            if pos == 0: continue
            prev_close = df.iloc[pos - 1]["close"]
            direction = 1 if rets.at[idx] > 0 else -1
            max_ret_val = rets.mean() + direction * z_thresh * rets.std()
            capped = prev_close * (1 + max_ret_val)
            old = df.at[idx, "close"]
            df.at[idx, "close"] = round(capped, 2)
            entries.append(_ae("RECT-EOD-015", symbol, source, exchange, "EOD",
                              idx, "close", str(old), str(df.at[idx, "close"]),
                              f"Return z={z.at[idx]:.1f} winsorized", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Winsorized {len(entries)} price outliers" if entries else "No price outliers"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-020  Sort Order Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-020", test_ids=("EOD-020",), name="Sort Order Fix",
    timeframe="EOD", priority=3, default_conf=0.99,
    description="Sort by date ascending; remove future dates."))
def rect_eod_020(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-020", "EOD-020", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.99)
    entries = []
    df = df.copy()

    if "date" not in df.columns:
        r.action = "SKIPPED"; return df, r

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove future dates
    today = pd.Timestamp.now().normalize()
    future = df["date"] > today
    if future.any():
        for idx in df.index[future]:
            entries.append(_ae("RECT-EOD-020", symbol, source, exchange, "EOD",
                              idx, "date", str(df.at[idx, "date"].date()), "REMOVED",
                              "Future date removed", conf))
        df = df[~future]

    # Sort
    if not df["date"].is_monotonic_increasing:
        entries.append(_ae("RECT-EOD-020", symbol, source, exchange, "EOD",
                          0, "sort_order", "unsorted", "date_asc",
                          "Data re-sorted by date ascending", conf))
        df = df.sort_values("date").reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"{len(entries)} sort/future-date fixes" if entries else "Already sorted"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-EOD-024  Trailing Whitespace Trim
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-EOD-024", test_ids=("EOD-024",), name="Whitespace Trim",
    timeframe="EOD", priority=1, default_conf=0.99,
    description="Strip whitespace from string columns; uppercase symbols."))
def rect_eod_024(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _make_result("RECT-EOD-024", "EOD-024", symbol, source, "EOD", exchange)
    conf = config.get("confidence", 0.99)
    entries = []
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        before = df[col].copy()
        df[col] = df[col].str.strip()
        changed = before != df[col]
        if changed.any():
            for idx in df.index[changed.fillna(False)]:
                entries.append(_ae("RECT-EOD-024", symbol, source, exchange, "EOD",
                                  idx, col, str(before.at[idx]), str(df.at[idx, col]),
                                  "Whitespace trimmed", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Trimmed {len(entries)} string values" if entries else "No whitespace issues"
    return df, r
