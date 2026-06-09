"""
DDQ Engine — Intraday Rectification Rules
downloaded_data_dq/rectification/rules_intraday.py

14 rules for automated rectification of Intraday time-series issues.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from downloaded_data_dq.rectification.registry import RuleSpec, rect_rule
from downloaded_data_dq.rectification.audit import AuditEntry, RectificationResult

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
SESSION_START_MIN = 9 * 60 + 15   # 09:15
SESSION_END_MIN   = 15 * 60 + 30  # 15:30


def _ae(rule_id, sym, src, exch, row, col, old, new, reason, conf):
    return AuditEntry(rule_id=rule_id, symbol=sym, source=src, exchange=exch,
                      timeframe="INTRADAY", row_index=int(row), column=col,
                      old_value=str(old), new_value=str(new), reason=reason, confidence=conf)


def _mr(rule_id, test_id, sym, src, exch):
    return RectificationResult(rule_id=rule_id, test_id=test_id, symbol=sym,
                                source=src, timeframe="INTRADAY", exchange=exch)


def _tod_minutes(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-001  Missing Bar Fill
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-001", test_ids=("INT-001",), name="Missing Bar Fill",
    timeframe="INTRADAY", priority=20, default_conf=0.85,
    description="Insert missing minute bars within session with flat price from previous bar."))
def rect_int_001(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-001", "INT-001", symbol, source, exchange)
    conf = config.get("confidence", 0.85)
    entries = []
    df = df.copy()

    if "datetimestamp" not in df.columns or df.empty:
        r.action = "SKIPPED"; r.details = "No timestamp column"; return df, r

    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce")
    df = df.sort_values("datetimestamp").reset_index(drop=True)

    # Per-day: fill missing minutes within session
    new_rows = []
    for date, day_df in df.groupby(df["datetimestamp"].dt.date):
        tod = _tod_minutes(day_df["datetimestamp"])
        in_session = day_df[(tod >= SESSION_START_MIN) & (tod <= SESSION_END_MIN)]
        if in_session.empty:
            continue
        session_start = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=15)
        session_end   = pd.Timestamp(date) + pd.Timedelta(hours=15, minutes=29)
        full_range = pd.date_range(session_start, session_end, freq="1min")
        existing_ts = set(in_session["datetimestamp"].dt.floor("min"))
        missing = [t for t in full_range if t not in existing_ts]

        for ts in missing[:50]:  # cap per day to avoid runaway
            prev_rows = in_session[in_session["datetimestamp"] < ts]
            if prev_rows.empty:
                continue
            prev = prev_rows.iloc[-1]
            row = {"datetimestamp": ts, "open": prev["close"], "high": prev["close"],
                   "low": prev["close"], "close": prev["close"], "volume": 0}
            new_rows.append(row)
            entries.append(_ae("RECT-INT-001", symbol, source, exchange,
                              -1, "datetimestamp", "MISSING", str(ts),
                              f"Inserted missing bar with flat price {prev['close']}", conf))

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("datetimestamp").reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Inserted {len(entries)} missing bars" if entries else "No missing bars"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-002  Null Timestamp Drop
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-002", test_ids=("INT-002",), name="Null Timestamp Drop",
    timeframe="INTRADAY", priority=5, default_conf=0.99,
    description="Drop rows with null datetimestamp."))
def rect_int_002(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-002", "INT-002", symbol, source, exchange)
    conf = config.get("confidence", 0.99)
    entries = []
    df = df.copy()

    if "datetimestamp" not in df.columns:
        r.action = "SKIPPED"; return df, r

    null_ts = df["datetimestamp"].isna()
    if null_ts.any():
        for idx in df.index[null_ts]:
            entries.append(_ae("RECT-INT-002", symbol, source, exchange,
                              idx, "datetimestamp", "NaT", "DROPPED",
                              "Null timestamp row dropped", conf))
        df = df[~null_ts].reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Dropped {len(entries)} null-timestamp rows" if entries else "No null timestamps"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-003  Duplicate Bar Remove
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-003", test_ids=("INT-003",), name="Duplicate Bar Remove",
    timeframe="INTRADAY", priority=6, default_conf=0.98,
    description="Remove duplicate bars for same timestamp."))
def rect_int_003(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-003", "INT-003", symbol, source, exchange)
    conf = config.get("confidence", 0.98)
    entries = []
    df = df.copy()

    before = len(df)
    dup = df.duplicated(keep="first")
    if dup.any():
        for idx in df.index[dup]:
            entries.append(_ae("RECT-INT-003", symbol, source, exchange,
                              idx, "ROW", "duplicate", "REMOVED",
                              "Duplicate bar removed", conf))
        df = df[~dup].reset_index(drop=True)

    if "datetimestamp" in df.columns:
        ts_dup = df.duplicated(subset=["datetimestamp"], keep="first")
        if ts_dup.any():
            for idx in df.index[ts_dup]:
                entries.append(_ae("RECT-INT-003", symbol, source, exchange,
                                  idx, "ROW", "ts_dup", "REMOVED",
                                  "Same-timestamp duplicate removed", conf * 0.95))
            df = df[~ts_dup].reset_index(drop=True)

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Removed {len(entries)} duplicates ({before} -> {len(df)})" if entries else "No duplicates"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-004  Null OHLCV Fill (within same day only)
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-004", test_ids=("INT-004", "INT-005"), name="Null OHLCV Fill",
    timeframe="INTRADAY", priority=15, default_conf=0.90,
    description="Forward-fill nulls within same trading day only."))
def rect_int_004(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-004", "INT-004", symbol, source, exchange)
    conf = config.get("confidence", 0.90)
    entries = []
    df = df.copy()

    cols = [c for c in PRICE_COLS + ["volume"] if c in df.columns]
    if not cols or "datetimestamp" not in df.columns:
        r.action = "SKIPPED"; return df, r

    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce")
    df["_day"] = df["datetimestamp"].dt.date

    for col in cols:
        null_mask = df[col].isna()
        if not null_mask.any():
            continue
        # ffill within day groups
        filled = df.groupby("_day")[col].ffill()
        still_null = filled.isna()
        filled = filled.fillna(df.groupby("_day")[col].bfill())
        if col == "volume":
            filled = filled.fillna(0)
        changed = null_mask & ~filled.isna() & (filled != df[col]).fillna(True)
        for idx in df.index[changed]:
            old_val = df.at[idx, col]
            new_val = filled.at[idx]
            entries.append(_ae("RECT-INT-004", symbol, source, exchange,
                              idx, col, str(old_val), str(new_val),
                              f"Null {col} ffill within day", conf))
        df[col] = filled

    df.drop(columns=["_day"], inplace=True, errors="ignore")

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.rows_modified = len(set(e.row_index for e in entries))
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Filled {len(entries)} null values (day-scoped)" if entries else "No nulls"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-005  OHLC Inversion Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-005", test_ids=("INT-006", "INT-007"), name="OHLC Inversion Fix",
    timeframe="INTRADAY", priority=12, default_conf=0.95,
    description="Swap High/Low inversions; clamp Close to [Low, High]."))
def rect_int_005(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-005", "INT-006", symbol, source, exchange)
    conf = config.get("confidence", 0.95)
    entries = []
    df = df.copy()

    if "high" in df.columns and "low" in df.columns:
        mask = df["high"] < df["low"]
        if mask.any():
            inv = df.loc[mask, ["high", "low"]].copy()
            df.loc[mask, "high"] = inv["low"]
            df.loc[mask, "low"] = inv["high"]
            for idx in df.index[mask]:
                entries.append(_ae("RECT-INT-005", symbol, source, exchange,
                                  idx, "high-low", "inverted", "swapped",
                                  "High-Low inversion fixed", conf))

    if all(c in df.columns for c in ["close", "low", "high"]):
        above = df["close"] > df["high"]
        for idx in df.index[above]:
            old = df.at[idx, "close"]; df.at[idx, "close"] = df.at[idx, "high"]
            entries.append(_ae("RECT-INT-005", symbol, source, exchange,
                              idx, "close", str(old), str(df.at[idx, "close"]),
                              "Close clamped to high", conf))
        below = df["close"] < df["low"]
        for idx in df.index[below]:
            old = df.at[idx, "close"]; df.at[idx, "close"] = df.at[idx, "low"]
            entries.append(_ae("RECT-INT-005", symbol, source, exchange,
                              idx, "close", str(old), str(df.at[idx, "close"]),
                              "Close clamped to low", conf))

    r.audit_entries = entries
    r.changes_count = len(entries)
    r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} OHLC issues" if entries else "OHLC OK"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-006  Negative Price Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-006", test_ids=("INT-008",), name="Negative Price Fix",
    timeframe="INTRADAY", priority=14, default_conf=0.90,
    description="abs() for negative; ffill for zero."))
def rect_int_006(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-006", "INT-008", symbol, source, exchange)
    conf = config.get("confidence", 0.90)
    entries = []
    df = df.copy()

    for col in PRICE_COLS:
        if col not in df.columns: continue
        neg = df[col] < 0
        for idx in df.index[neg]:
            old = df.at[idx, col]; df.at[idx, col] = abs(old)
            entries.append(_ae("RECT-INT-006", symbol, source, exchange,
                              idx, col, str(old), str(df.at[idx, col]),
                              f"Negative {col} -> abs()", conf))

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Fixed {len(entries)} negatives" if entries else "No negatives"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-007  Out-of-Session Trim
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-007", test_ids=("INT-009", "INT-010"), name="Out-of-Session Trim",
    timeframe="INTRADAY", priority=8, default_conf=0.95,
    description="Remove bars outside 09:15-15:30 IST."))
def rect_int_007(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-007", "INT-009", symbol, source, exchange)
    conf = config.get("confidence", 0.95)
    entries = []
    df = df.copy()

    if "datetimestamp" not in df.columns:
        r.action = "SKIPPED"; return df, r

    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce")
    tod = _tod_minutes(df["datetimestamp"])
    out_of_session = (tod < SESSION_START_MIN) | (tod > SESSION_END_MIN)

    if out_of_session.any():
        for idx in df.index[out_of_session]:
            entries.append(_ae("RECT-INT-007", symbol, source, exchange,
                              idx, "datetimestamp", str(df.at[idx, "datetimestamp"]), "REMOVED",
                              "Out-of-session bar removed", conf))
        df = df[~out_of_session].reset_index(drop=True)

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Removed {len(entries)} out-of-session bars" if entries else "All bars in session"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-010  Monotonic Time Fix
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-010", test_ids=("INT-011", "INT-012"), name="Monotonic Time Fix",
    timeframe="INTRADAY", priority=7, default_conf=0.92,
    description="Sort by timestamp; remove non-monotonic rows."))
def rect_int_010(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-010", "INT-011", symbol, source, exchange)
    conf = config.get("confidence", 0.92)
    entries = []
    df = df.copy()

    if "datetimestamp" not in df.columns:
        r.action = "SKIPPED"; return df, r

    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce")
    if not df["datetimestamp"].is_monotonic_increasing:
        entries.append(_ae("RECT-INT-010", symbol, source, exchange,
                          0, "sort_order", "non-monotonic", "sorted",
                          "Timestamps re-sorted to monotonic", conf))
        df = df.sort_values("datetimestamp").reset_index(drop=True)

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = "Re-sorted to monotonic" if entries else "Already monotonic"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-011  Volume Spike Cap
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-011", test_ids=("INT-013", "INT-014"), name="Volume Spike Cap",
    timeframe="INTRADAY", priority=45, default_conf=0.70,
    description="Winsorize intraday volume spikes."))
def rect_int_011(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-011", "INT-013", symbol, source, exchange)
    conf = config.get("confidence", 0.70)
    z_thresh = config.get("z_score_threshold", 4.0)
    entries = []
    df = df.copy()

    if "volume" not in df.columns or df["volume"].std() == 0:
        r.action = "SKIPPED"; return df, r

    z = (df["volume"] - df["volume"].mean()) / df["volume"].std()
    outlier = z.abs() > z_thresh
    if outlier.any():
        cap = int(df["volume"].mean() + z_thresh * df["volume"].std())
        for idx in df.index[outlier]:
            old = df.at[idx, "volume"]
            df.at[idx, "volume"] = min(old, cap) if z.at[idx] > 0 else max(old, 0)
            entries.append(_ae("RECT-INT-011", symbol, source, exchange,
                              idx, "volume", str(old), str(df.at[idx, "volume"]),
                              f"Volume spike winsorized (z={z.at[idx]:.1f})", conf))

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Winsorized {len(entries)} spikes" if entries else "No spikes"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-019  Weekend/Holiday Trim
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-019", test_ids=("INT-038", "INT-039", "INT-040"),
    name="Weekend/Holiday Trim",
    timeframe="INTRADAY", priority=9, default_conf=0.95,
    description="Remove bars on weekends/holidays."))
def rect_int_019(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-019", "INT-038", symbol, source, exchange)
    conf = config.get("confidence", 0.95)
    entries = []
    df = df.copy()

    if "datetimestamp" not in df.columns:
        r.action = "SKIPPED"; return df, r

    df["datetimestamp"] = pd.to_datetime(df["datetimestamp"], errors="coerce")
    weekend = df["datetimestamp"].dt.dayofweek >= 5
    if weekend.any():
        for idx in df.index[weekend]:
            entries.append(_ae("RECT-INT-019", symbol, source, exchange,
                              idx, "datetimestamp", str(df.at[idx, "datetimestamp"]), "REMOVED",
                              "Weekend bar removed", conf))
        df = df[~weekend].reset_index(drop=True)

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Removed {len(entries)} weekend bars" if entries else "No weekend bars"
    return df, r


# ─────────────────────────────────────────────────────────────────────────
# RECT-INT-020  Whitespace & Encoding
# ─────────────────────────────────────────────────────────────────────────
@rect_rule(RuleSpec(
    rule_id="RECT-INT-020", test_ids=("INT-041", "INT-042"),
    name="Whitespace & Encoding Fix",
    timeframe="INTRADAY", priority=1, default_conf=0.99,
    description="Strip whitespace from string columns."))
def rect_int_020(df: pd.DataFrame, symbol: str, source: str, exchange: str,
                  config: dict, **kw) -> tuple[pd.DataFrame, RectificationResult]:
    r = _mr("RECT-INT-020", "INT-041", symbol, source, exchange)
    conf = config.get("confidence", 0.99)
    entries = []
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        before = df[col].copy()
        df[col] = df[col].str.strip()
        changed = (before != df[col]).fillna(False)
        for idx in df.index[changed]:
            entries.append(_ae("RECT-INT-020", symbol, source, exchange,
                              idx, col, str(before.at[idx]), str(df.at[idx, col]),
                              "Whitespace trimmed", conf))

    r.audit_entries = entries; r.changes_count = len(entries); r.confidence = conf
    r.action = "FIXED" if entries else "SKIPPED"
    r.details = f"Trimmed {len(entries)} values" if entries else "No whitespace issues"
    return df, r
