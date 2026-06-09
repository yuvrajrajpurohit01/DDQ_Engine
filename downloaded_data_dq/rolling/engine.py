"""
DDQ Engine v12 — Rolling Window DQ Engine
downloaded_data_dq/rolling/engine.py

Slices data by configurable time windows and computes DQ scores
per Source+Exchange for each period. Uses fast vectorized checks
(not the full 248-test suite) for per-window scoring.

Window frequencies: monthly, quarterly, half_yearly, yearly,
    2_yearly, 3_yearly, 5_yearly, 10_yearly, 15_yearly, all

Output: rolling_results dict used by Phase 4 (Best Source Selection)
and Phase 5 (Dashboard Rolling Analysis view).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from downloaded_data_dq.utils.ddq_logger import setup_ddq_logger, get_log_path

logger = logging.getLogger(__name__)

# ── Window frequency definitions (months) ────────────────────────────────────
WINDOW_MONTHS = {
    "monthly":     1,
    "quarterly":   3,
    "half_yearly": 6,
    "yearly":      12,
    "2_yearly":    24,
    "3_yearly":    36,
    "5_yearly":    60,
    "10_yearly":   120,
    "15_yearly":   180,
    "all":         0,   # 0 = full data range
}


@dataclass
class WindowScore:
    """DQ score for one Source+Exchange in one time window."""
    symbol:       str = ""
    source:       str = ""
    exchange:     str = ""
    timeframe:    str = ""
    product_class: str = ""
    window_freq:  str = ""
    window_label: str = ""
    window_start: str = ""
    window_end:   str = ""
    n_rows:       int = 0
    n_expected:   int = 0       # expected trading days/bars in this window
    # Individual check scores (0.0 to 1.0 each)
    completeness:     float = 0.0   # 1 - (null_pct)
    no_duplicates:    float = 0.0   # 1 - (dup_pct)
    ohlc_consistency: float = 0.0   # pct rows where H>=L, C in [L,H]
    volume_quality:   float = 0.0   # 1 - (zero_vol_pct)
    price_continuity: float = 0.0   # 1 - (extreme_return_pct)
    coverage:         float = 0.0   # n_rows / n_expected
    adj_close_valid:  float = 0.0   # 1 - (null_adj_close_pct)
    oi_valid:         float = 0.0   # 1 - (negative_oi_pct)
    src_e002:         float = 0.0
    src_e003:         float = 0.0
    # Composite
    composite_score:  float = 0.0   # weighted average of above
    n_clean_points:   int   = 0     # rows passing ALL checks (for tiebreak)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WindowPeriod:
    """A time window period."""
    label:  str
    start:  pd.Timestamp
    end:    pd.Timestamp
    freq:   str


# ══════════════════════════════════════════════════════════════════════════════
# WINDOW GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    frequencies: list[str] | None = None,
) -> dict[str, list[WindowPeriod]]:
    """
    Generate time window periods for requested frequencies.

    Returns: {freq_name: [WindowPeriod, ...]}
    """
    if frequencies is None:
        frequencies = list(WINDOW_MONTHS.keys())

    result: dict[str, list[WindowPeriod]] = {}

    for freq in frequencies:
        months = WINDOW_MONTHS.get(freq, 0)
        windows = []

        if months == 0 or freq == "all":
            # Full range as single window
            windows.append(WindowPeriod(
                label="ALL",
                start=data_start,
                end=data_end,
                freq=freq,
            ))
        else:
            # Generate rolling windows
            cursor = pd.Timestamp(data_start.year, data_start.month, 1)
            while cursor <= data_end:
                w_end = cursor + pd.DateOffset(months=months) - pd.Timedelta(days=1)
                if w_end > data_end:
                    w_end = data_end

                # Generate human-readable label
                label = _window_label(cursor, w_end, freq)
                windows.append(WindowPeriod(
                    label=label,
                    start=cursor,
                    end=w_end,
                    freq=freq,
                ))
                cursor = cursor + pd.DateOffset(months=months)

        result[freq] = windows

    return result


def _window_label(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> str:
    """Generate human-readable window label."""
    if freq == "monthly":
        return start.strftime("%b-%Y")
    elif freq == "quarterly":
        q = (start.month - 1) // 3 + 1
        return f"Q{q}-{start.year}"
    elif freq == "half_yearly":
        h = "H1" if start.month <= 6 else "H2"
        return f"{h}-{start.year}"
    elif freq == "yearly":
        return str(start.year)
    elif freq == "2_yearly":
        return f"{start.year}-{end.year}"
    elif freq == "3_yearly":
        return f"{start.year}-{end.year}"
    elif freq == "5_yearly":
        return f"{start.year}-{end.year}"
    elif freq == "10_yearly":
        return f"{start.year}-{end.year}"
    elif freq == "15_yearly":
        return f"{start.year}-{end.year}"
    else:
        return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


# ══════════════════════════════════════════════════════════════════════════════
# FAST DQ SCORING PER WINDOW
# ══════════════════════════════════════════════════════════════════════════════

# Weights for composite score
_WEIGHTS = {
    "completeness":     0.10,
    "no_duplicates":    0.05,
    "ohlc_consistency": 0.10,
    "volume_quality":   0.10,
    "price_continuity": 0.10,
    "coverage":         0.10,
    "adj_close_valid":  0.10,
    "oi_valid":         0.05,
    "src_e002":         0.15,
    "src_e003":         0.15,

}


def score_window_eod(df: pd.DataFrame, window: WindowPeriod,
                      symbol: str, source: str, exchange: str,
                      product_class: str = "", src_002_score: float | None = None, src_003_score: float | None = None) -> WindowScore:
    """Score an EOD DataFrame slice for one window period."""
    ws = WindowScore(
        symbol=symbol, source=source, exchange=exchange,
        product_class=product_class, timeframe="EOD", window_freq=window.freq,
        window_label=window.label,
        window_start=str(window.start.date()),
        window_end=str(window.end.date()),
    )


    ws.src_e002 = src_002_score if src_002_score is not None else 0
    ws.src_e003=src_003_score if src_003_score is not None else 0
    print(
        f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: SRC-E002: (EOD: Close match across sources) Score {ws.src_e002}")

    print(
        f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: SRC-E003: (EOD: OHLC match across sources) Score {src_003_score}")

    if df is None or df.empty:
        return ws

    # Slice to window
    if "date" not in df.columns:
        return ws
    dates = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    mask = (dates >= window.start) & (dates <= window.end)
    slc = df[mask].copy()

    if slc.empty:
        return ws

    n = len(slc)
    ws.n_rows = n

    # Expected trading days in window (~21 per month)
    try:
        bdays = len(pd.bdate_range(window.start, window.end))
        ws.n_expected = max(bdays, 1)
    except Exception:
        ws.n_expected = max(n, 1)

    # 1. Completeness (null check)
    null_count = 0
    for col in ["open", "high", "low", "close"]:
        if col in slc.columns:
            null_count += slc[col].isna().sum()
    total_cells = n * 4
    ws.completeness = 1.0 - (null_count / max(total_cells, 1))
    print(f"{str(window.start.date())} to {str(window.end.date())}:: data: {source}/{exchange}/{symbol} :: Completness Score: {ws.completeness}")

    # 2. No duplicates
    if "date" in slc.columns:
        dup_count = slc["date"].duplicated().sum()
        ws.no_duplicates = 1.0 - (dup_count / max(n, 1))
    else:
        ws.no_duplicates = 1.0
    print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: no Duplicate score: {ws.no_duplicates}")

    # 3. OHLC consistency (H>=L, C in [L,H], O in [L,H])
    ohlc_ok = 0
    if all(c in slc.columns for c in ["open", "high", "low", "close"]):
        h = slc["high"].astype(float)
        l = slc["low"].astype(float)
        c = slc["close"].astype(float)
        o = slc["open"].astype(float)
        valid = (h >= l) & (c >= l) & (c <= h) & (o >= l) & (o <= h)
        ohlc_ok = valid.sum()
    ws.ohlc_consistency = ohlc_ok / max(n, 1)

    print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: OHLC consistency Score: {ws.ohlc_consistency}")

    # 4. Volume quality
    if "volume" in slc.columns:
        vol = pd.to_numeric(slc["volume"], errors="coerce")
        zero_vol = ((vol == 0) | vol.isna()).sum()
        ws.volume_quality = 1.0 - (zero_vol / max(n, 1))
    else:
        ws.volume_quality = 0.5  # no volume column = half score

    print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: Volume Quality Score: {ws.volume_quality}")

    # 5. Price continuity (no extreme daily returns)
    if "close" in slc.columns and n > 1:
        closes = pd.to_numeric(slc["close"], errors="coerce").dropna()
        if len(closes) > 1:
            rets = closes.pct_change().dropna()
            extreme = (rets.abs() > 0.20).sum()  # >20% daily move
            ws.price_continuity = 1.0 - (extreme / max(len(rets), 1))
        else:
            ws.price_continuity = 1.0
    else:
        ws.price_continuity = 0.0

    print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: Price Continuity Score {ws.price_continuity}")

    # 6. Coverage
    ws.coverage = min(n / max(ws.n_expected, 1), 1.0)
    print(
        f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: Coverage Score: {ws.coverage}")

    # 7. Adj close valid
    if "adj_close" in slc.columns:
        null_adj = slc["adj_close"].isna().sum()
        ws.adj_close_valid = 1.0 - (null_adj / max(n, 1))
    else:
        ws.adj_close_valid = 0.0

        print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: Adj. Close Score: {ws.adj_close_valid}")

    # 8. OI valid
    if "open_interest" in slc.columns:
        oi = pd.to_numeric(slc["open_interest"], errors="coerce")
        neg_oi = (oi < 0).sum()
        ws.oi_valid = 1.0 - (neg_oi / max(n, 1))
    else:
        ws.oi_valid = 1.0  # no OI column is fine for equity spot

        print(f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: OI Validation Score: {ws.oi_valid}")

#30 lines

#    univariate_score = (PAIR_TEST_SCORE [30x2] * 0.4 *_PAIR_TESTS_WEIGHTS [2x1])+(0.6*_UNIVARIATE_TESTS_WEIGHTS *_UNIVARIATE_TEST_SCORE)  # Composite score
    ws.composite_score = sum(
        getattr(ws, k) * v for k, v in _WEIGHTS.items()
    )

    msg=f"{str(window.start.date())} to {str(window.end.date())}: :: data: {source}/{exchange}/{symbol} :: Composite Score: {ws.composite_score}"
    print(f"\033[92m{msg}\033[0m")
    logging.info(f"\033[92m{msg}\033[0m")

    # Clean data points (rows passing ALL basic checks)
    if all(c in slc.columns for c in ["open", "high", "low", "close"]):
        h = slc["high"].astype(float)
        l = slc["low"].astype(float)
        c = slc["close"].astype(float)
        o = slc["open"].astype(float)
        clean = (
            h.notna() & l.notna() & c.notna() & o.notna() &
            (h >= l) & (c >= l) & (c <= h) & (o >= l) & (o <= h)
        )
        ws.n_clean_points = int(clean.sum())
    else:
        ws.n_clean_points = 0


    return ws


def score_window_intraday(df: pd.DataFrame, window: WindowPeriod,
                           symbol: str, source: str, exchange: str,
                           product_class: str = "") -> WindowScore:
    """Score an Intraday DataFrame slice for one window period."""
    ws = WindowScore(
        symbol=symbol, source=source, exchange=exchange,
        product_class=product_class, timeframe="INTRADAY", window_freq=window.freq,
        window_label=window.label,
        window_start=str(window.start.date()),
        window_end=str(window.end.date()),
    )

    if df is None or df.empty:
        return ws

    if "datetimestamp" not in df.columns:
        return ws

    ts = pd.to_datetime(df["datetimestamp"], errors="coerce")
    mask = (ts >= window.start) & (ts <= window.end + pd.Timedelta(days=1))
    slc = df[mask].copy()

    if slc.empty:
        return ws

    n = len(slc)
    ws.n_rows = n

    # Expected bars: ~375 bars/day × trading days
    try:
        bdays = len(pd.bdate_range(window.start, window.end))
        ws.n_expected = max(bdays * 375, 1)
    except Exception:
        ws.n_expected = max(n, 1)

    # Same checks as EOD but adapted for intraday
    # 1. Completeness
    null_count = sum(slc[c].isna().sum() for c in ["open", "high", "low", "close"] if c in slc.columns)
    ws.completeness = 1.0 - (null_count / max(n * 4, 1))

    # 2. No duplicates
    dup_count = slc["datetimestamp"].duplicated().sum() if "datetimestamp" in slc.columns else 0
    ws.no_duplicates = 1.0 - (dup_count / max(n, 1))

    # 3. OHLC consistency
    if all(c in slc.columns for c in ["open", "high", "low", "close"]):
        h, l, c, o = (slc[x].astype(float) for x in ["high", "low", "close", "open"])
        valid = (h >= l) & (c >= l) & (c <= h) & (o >= l) & (o <= h)
        ws.ohlc_consistency = valid.sum() / max(n, 1)

    # 4. Volume quality
    if "volume" in slc.columns:
        vol = pd.to_numeric(slc["volume"], errors="coerce")
        ws.volume_quality = 1.0 - (((vol == 0) | vol.isna()).sum() / max(n, 1))
    else:
        ws.volume_quality = 0.5

    # 5. Price continuity (no extreme inter-bar jumps)
    if "close" in slc.columns and n > 1:
        closes = pd.to_numeric(slc["close"], errors="coerce").dropna()
        if len(closes) > 1:
            rets = closes.pct_change().dropna()
            extreme = (rets.abs() > 0.05).sum()  # >5% per bar is extreme for intraday
            ws.price_continuity = 1.0 - (extreme / max(len(rets), 1))

    # 6. Coverage
    ws.coverage = min(n / max(ws.n_expected, 1), 1.0)

    # 7-8. adj_close and OI
    if "adj_close" in slc.columns:
        ws.adj_close_valid = 1.0 - (slc["adj_close"].isna().sum() / max(n, 1))
    if "open_interest" in slc.columns:
        oi = pd.to_numeric(slc["open_interest"], errors="coerce")
        ws.oi_valid = 1.0 - ((oi < 0).sum() / max(n, 1))
    else:
        ws.oi_valid = 1.0

    # Composite
    ws.composite_score = sum(getattr(ws, k) * v for k, v in _WEIGHTS.items())

    # Clean points
    if all(c in slc.columns for c in ["open", "high", "low", "close"]):
        h, l, c, o = (slc[x].astype(float) for x in ["high", "low", "close", "open"])
        clean = h.notna() & l.notna() & c.notna() & o.notna() & (h >= l) & (c >= l) & (c <= h)
        ws.n_clean_points = int(clean.sum())

    return ws

SOURCES = ["dhan", "kite", "upstox"]
def _eod_pairs(sym_data: dict, exchange: str = "NSE",
               timeframes: list[str] | None = None) -> list[tuple]:
    """Return list of (src, df) tuples for ONE exchange."""
    if timeframes is None:
        timeframes = ["eod"]
    result = []
    for tf in timeframes:
        tf_data = sym_data.get(tf, {})
        if not isinstance(tf_data, dict):
            continue
        src_dict = tf_data.get(exchange, {})        # ← use the requested exchange
        if not isinstance(src_dict, dict):
            continue
        for src, df in src_dict.items():
            if df is not None and not df.empty:
                result.append((src, df))
    return result

_SOURCE_INFO = {
    "dhan": {
        "display_name":       "Dhan",
        "folder_name":        "Dhan",
        "is_adjusted_prices": True,   # UNADJUSTED spot prices
        "is_adjusted_volume": True,
    },
    "kite": {
        "display_name":       "Kite",
        "folder_name":        "Kite",
        "is_adjusted_prices": True,    # Zerodha Kite serves adjusted prices
        "is_adjusted_volume": True,
    },
    "upstox": {
        "display_name":       "Upstox",
        "folder_name":        "Upstox",
        "is_adjusted_prices": True,    # Upstox serves adjusted prices
        "is_adjusted_volume": True,
    },
}

_SKIP_ON_ADJUSTMENT_MISMATCH = True

def _adjusted_skip(src_a, src_b) -> bool:
    """Return True if cross-price comparison should be skipped due to adjustment mismatch."""
    if not _SKIP_ON_ADJUSTMENT_MISMATCH:
        return False
    adj_a = _SOURCE_INFO.get(src_a.lower(), {}).get("is_adjusted_prices", True)
    adj_b = _SOURCE_INFO.get(src_b.lower(), {}).get("is_adjusted_prices", True)
    return adj_a != adj_b

def _align_eod(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner join two EOD frames on date.
    Deduplicates by date first (keep first) so sources with duplicate dates
    (e.g. Dhan CP_CAP) do not produce extra rows after .loc[idx].
    """
    a = df_a.set_index("date")
    b = df_b.set_index("date")
    a = a[~a.index.duplicated(keep="first")]
    b = b[~b.index.duplicated(keep="first")]
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]
# ─────────────────────────────────────────────────────────────────────────────
# SRC-E002  EOD: Close match across sources — PER-SOURCE scoring
# ─────────────────────────────────────────────────────────────────────────────
def test_src_e002(window: WindowPeriod, timeframes: list[str], sym_data: dict,
                  symbol: str, product_class: str = "") -> dict:
    """
    Cross-source close-match test across 12 (source,exchange) pair combinations.
    Returns:
      {
        "metrics": {...},
        "issues":  [...],
        "per_source_score": {
            ("dhan",   "NSE"): 0.94,
            ...
            ("kite",   "BSE"): -1.0,    # ← could not be evaluated
        }
      }
    """
    tol               = 0.1 / 100
    WORST_BREACH_PCT  = 5.0
    ISSUE_THRESHOLD   = 0.5
    UNEVALUATED_SCORE = -1.0          # ← per your spec
    INCLUDE_CROSS_EXCHANGE = True      # ← flip to False to test only same-exchange (gives 6 pairs)

    issues, metrics, breach_pct = [], {}, {}
    per_source_breaches: dict[tuple, list[float]] = {}

    w_start = pd.Timestamp(window.start)
    w_end   = pd.Timestamp(window.end)
    w_label = getattr(window, "label", f"{w_start.date()}_{w_end.date()}")

    # ── STEP 1: Collect ALL (source, exchange) tuples that have windowed data ──
    all_tuples = []   # [(src, exch, sliced_df), ...]
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(sym_data, exch, timeframes)
        for src, df in pairs:
            if df is None or df.empty or "date" not in df.columns:
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            mask = (d["date"] >= w_start) & (d["date"] <= w_end)
            d = d.loc[mask]
            if not d.empty:
                all_tuples.append((src, exch, d))
                # Register every tuple — empty breach list means "couldn't evaluate"
                per_source_breaches.setdefault((src, exch), [])

    # ── STEP 2: Build the 12 valid cross-source pairs ──
    for i, (src_a, exch_a, df_a) in enumerate(all_tuples):
        for src_b, exch_b, df_b in all_tuples[i+1:]:
            # Skip same-source: Dhan-NSE vs Dhan-BSE is arbitrage, not DQ
            if src_a == src_b:
                continue

            # Optionally skip cross-exchange (legit price differences will skew results)
            if not INCLUDE_CROSS_EXCHANGE and exch_a != exch_b:
                continue

            key = f"{src_a}_{exch_a}_vs_{src_b}_{exch_b}"
            print(f" Pair: {key}")

            if _adjusted_skip(src_a, src_b):
                metrics[key] = {"status": "SKIPPED — adjustment mismatch"}
                continue

            a, b = _align_eod(df_a, df_b)
            if len(a) == 0:
                metrics[key] = {"status": "SKIPPED — no aligned dates in window"}
                continue

            diff   = (a["close"] / b["close"] - 1).abs()
            breach = (diff > tol).sum()
            pct    = float(breach) / len(a) * 100

            metrics[key] = {
                "window":         w_label,
                "window_start":   str(w_start.date()),
                "window_end":     str(w_end.date()),
                "src_a":          src_a,
                "exch_a":         exch_a,
                "src_b":          src_b,
                "exch_b":         exch_b,
                "is_cross_exch":  exch_a != exch_b,
                "aligned_dates":  len(a),
                "breaches":       int(breach),
                "breach_pct":     round(pct, 3),
                "max_diff_pct":   round(float(diff.max() * 100), 4),
            }

            print(f"Metrics: {metrics[key]}")

            breach_pct[key]= {
                int(breach)
            }

            print(f"Breach PCT: {breach_pct[key]}")


            # Attribute the breach to BOTH (source, exchange) tuples in this pair
            per_source_breaches[(src_a, exch_a)].append(pct)
            per_source_breaches[(src_b, exch_b)].append(pct)

            if pct > ISSUE_THRESHOLD:
                cross_marker = " [CROSS-EXCH]" if exch_a != exch_b else ""
                issues.append(
                    f"[{w_label}] {src_a}/{exch_a} vs {src_b}/{exch_b}{cross_marker}: "
                    f"{breach} close divergences > {tol*100:.2f}% ({pct:.2f}%)"
                )

    print(f"per_source_breaches: {per_source_breaches}")



    # ── STEP 3: Convert per-tuple breach lists to scores ──
    per_source_score: dict[tuple, float] = {}
    for (src, exch), breach_list in per_source_breaches.items():
        if not breach_list:
            # No comparison ran for this tuple — return -1 (unevaluated)
            per_source_score[(src, exch)] = UNEVALUATED_SCORE
        else:
            avg_breach = sum(breach_list) / len(breach_list)
            score = max(0, 1.0 - (avg_breach / 100))
            per_source_score[(src, exch)] = round(score, 4)
            print(f"SRC-E002 Score: {score:.2f}% Avg Breach: {avg_breach:.2f}% per source: {breach_list}, per source Score: {per_source_score}: {round(score, 4)} {WORST_BREACH_PCT}")

    return {
        "metrics":          metrics,
        "issues":           issues,
        "per_source_score": per_source_score,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SRC-E003  EOD: OHL match across sources (12-pair scheme) — PER-SOURCE scoring
# ─────────────────────────────────────────────────────────────────────────────
def test_src_e003(window: WindowPeriod, timeframes: list[str], sym_data: dict,
                  symbol: str, product_class: str = "") -> dict:
    """
    Cross-source consistency check for Open / High / Low fields across all 12
    (source, exchange) cross-source pair combinations.

    With 3 sources × 2 exchanges = 6 (source, exchange) tuples, the unordered
    pair count is C(6,2) = 15 → minus 3 same-source pairs (e.g. Dhan-NSE vs
    Dhan-BSE, which is arbitrage, not a DQ check) = 12 evaluated pairs.

    A (source, exchange) tuple that has no evaluated comparisons (single source,
    all adjustment-skipped, no aligned dates) returns score = -1.0 (UNEVALUATED).

    Returns:
      {
        "metrics": {...},                     # per-pair, per-field breach metrics
        "issues":  [...],                     # human-readable warnings
        "per_source_score": {
            ("dhan",   "NSE"): 0.92,
            ("kite",   "NSE"): 0.96,
            ("upstox", "BSE"): -1.0,          # ← could not be evaluated
            ...
        }
      }
    """
    tol               = 0.1 / 100
    WORST_BREACH_PCT  = 5.0
    ISSUE_THRESHOLD   = 1.0
    UNEVALUATED_SCORE = -1.0
    INCLUDE_CROSS_EXCHANGE = True     # ← flip to False for same-exchange-only (6 pairs)
    FIELDS = ["open", "high", "low"]

    issues, metrics = [], {}
    per_source_breaches: dict[tuple, list[float]] = {}

    w_start = pd.Timestamp(window.start)
    w_end   = pd.Timestamp(window.end)
    w_label = getattr(window, "label", f"{w_start.date()}_{w_end.date()}")

    # ── STEP 1: Collect ALL (source, exchange) tuples that have windowed data ──
    all_tuples = []   # [(src, exch, sliced_df), ...]
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(sym_data, exch, timeframes)
        for src, df in pairs:
            if df is None or df.empty or "date" not in df.columns:
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            mask = (d["date"] >= w_start) & (d["date"] <= w_end)
            d = d.loc[mask]
            if not d.empty:
                all_tuples.append((src, exch, d))
                # Register every tuple — empty breach list later means "unevaluated"
                per_source_breaches.setdefault((src, exch), [])

    # ── STEP 2: Build cross-source pairs (skip same-source) ──
    for i, (src_a, exch_a, df_a) in enumerate(all_tuples):
        for src_b, exch_b, df_b in all_tuples[i+1:]:
            # Skip same-source: Dhan-NSE vs Dhan-BSE is arbitrage, not DQ
            if src_a == src_b:
                continue

            # Optionally skip cross-exchange (legit price differences will skew results)
            if not INCLUDE_CROSS_EXCHANGE and exch_a != exch_b:
                continue

            pair_key = f"{src_a}_{exch_a}_vs_{src_b}_{exch_b}_{w_label}"
            print(f" Pair: {pair_key}")

            if _adjusted_skip(src_a, src_b):
                metrics[pair_key] = {"status": "SKIPPED — adjustment mismatch"}
                continue

            a, b = _align_eod(df_a, df_b)
            if len(a) == 0:
                metrics[pair_key] = {"status": "SKIPPED — no aligned dates in window"}
                continue

            # ── Compute breach % for each of O, H, L ──
            field_breaches = []   # list of breach_pct across O/H/L for THIS pair
            for field in FIELDS:
                if field not in a.columns or field not in b.columns:
                    continue
                diff     = (a[field] / b[field] - 1).abs()
                breach   = (diff > tol).sum()
                pct      = float(breach) / len(a) * 100
                max_diff = float(diff.max() * 100)

                metrics[f"{pair_key}_{field}"] = {
                    "window":         w_label,
                    "window_start":   str(w_start.date()),
                    "window_end":     str(w_end.date()),
                    "src_a":          src_a,
                    "exch_a":         exch_a,
                    "src_b":          src_b,
                    "exch_b":         exch_b,
                    "is_cross_exch":  exch_a != exch_b,
                    "field":          field,
                    "aligned_dates":  len(a),
                    "breaches":       int(breach),
                    "breach_pct":     round(pct, 3),
                    "max_diff_pct":   round(max_diff, 4),
                }
                field_breaches.append(pct)

                if pct > ISSUE_THRESHOLD:
                    cross_marker = " [CROSS-EXCH]" if exch_a != exch_b else ""
                    issues.append(
                        f"[{w_label}] {src_a}/{exch_a} vs {src_b}/{exch_b}{cross_marker} "
                        f"{field}: {breach} divergences > {tol*100:.2f}% ({pct:.2f}%)"
                    )

            if not field_breaches:
                continue

            # Aggregate this pair's breach across O/H/L (average across fields)
            pair_avg_breach = sum(field_breaches) / len(field_breaches)

            # ── Attribute pair's breach to BOTH (source, exchange) tuples ──
            per_source_breaches[(src_a, exch_a)].append(pair_avg_breach)
            per_source_breaches[(src_b, exch_b)].append(pair_avg_breach)


    print(f"SRC-E003: Per Source Breaches: {per_source_breaches}")

    # ── STEP 3: Convert per-tuple breach lists to scores ──
    per_source_score: dict[tuple, float] = {}
    for (src, exch), breach_list in per_source_breaches.items():
        if not breach_list:
            # No evaluated comparison for this tuple → -1 (unevaluated)
            per_source_score[(src, exch)] = UNEVALUATED_SCORE
        else:
            avg_breach = sum(breach_list) / len(breach_list)
            score = max(0, 1.0 - (avg_breach / 100))
            per_source_score[(src, exch)] = round(score, 4)
            print(f"SRC-E003: Per Source Score: {per_source_score[(src, exch)]}")


    result={
        "metrics":          metrics,
        "issues":           issues,
        "per_source_score": per_source_score,
        }
    field_keys = [k for k, v in result["metrics"].items() if "field" in v]
    unique_pairs = {k.rsplit("_", 1)[0] for k in field_keys}  # strip the trailing field
    skipped_pairs = [k for k, v in result["metrics"].items() if v.get("status")]

    print(f"Evaluated pairs: {len(unique_pairs)}")
    print(f"Skipped pairs:   {len(skipped_pairs)}")
    print(f"Total:           {len(unique_pairs) + len(skipped_pairs)}")  # → 12

    print("\nPer-tuple scores:")
    for (src, exch), score in sorted(result["per_source_score"].items()):
        marker = " (UNEVALUATED)" if score == -1.0 else ""
        print(f"  {src:8s} / {exch} : {score:>6.4f}{marker}")

    return {
        "metrics":          metrics,
        "issues":           issues,
        "per_source_score": per_source_score,
    }




# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_rolling_analysis(
    data_store: dict,
    config: dict,
    frequencies: list[str] | None = None,
    mode: str = "Both",
    verbose: bool = True,
    log_dir: str | Path = "logs",
    run_id: str = None,
) -> dict:
    """
    Run rolling window DQ analysis across all symbols and source+exchange combos.

    Args:
        data_store: {symbol: data_dict} from loader
        config:     Full config dict
        frequencies: List of window frequency names, or None for all
        mode:       "EOD", "Intraday", or "Both"
        verbose:    Print progress

    Returns:
        {
          "windows": {freq: [WindowPeriod_dict, ...]},
          "scores": {symbol: {freq: [WindowScore_dict, ...]}},
          "summary": {
            "total_windows": int,
            "total_scores": int,
            "frequencies": [str, ...],
            "symbols": [str, ...],
          }
        }
    """

    ddq_log = setup_ddq_logger(log_dir=log_dir, run_id=run_id)
    t0 = time.time()

    # Resolve frequencies
    instr_cfg = config.get("instruments", config)
    wf_cfg = instr_cfg.get("window_frequencies", {})
    if frequencies is None:
        frequencies = list(wf_cfg.keys()) if wf_cfg else list(WINDOW_MONTHS.keys())

    if verbose:
        ddq_log.info("\n" + "═" * 70)
        ddq_log.info("  ROLLING WINDOW DQ ANALYSIS")
        ddq_log.info("  Frequencies: %s", ", ".join(frequencies))
        ddq_log.info("═" * 70)

    all_scores: dict[str, dict[str, list]] = {}
    all_windows: dict[str, list] = {}
    total_scores = 0

    for symbol, sym_data in data_store.items():
        if not isinstance(sym_data, dict):
            continue

        sym_scores: dict[str, list] = {}

        # Find data date range across all sources
        data_start, data_end = _find_date_range(sym_data, mode)
        if data_start is None:
            if verbose:
                ddq_log.info("  ⚠  %s: No data found — skipping", symbol)
            continue

        # Generate windows
        windows = generate_windows(data_start, data_end, frequencies)
        if symbol == list(data_store.keys())[0]:
            # Store windows from first symbol (they're the same for all)
            for freq, wlist in windows.items():
                all_windows[freq] = [
                    {"label": w.label, "start": str(w.start.date()),
                     "end": str(w.end.date()), "freq": w.freq}
                    for w in wlist
                ]

        if verbose:
            ddq_log.info("\n  ── %s  (%s to %s) ──", symbol, data_start.date(), data_end.date())

        for freq, wlist in windows.items():
            freq_scores = []

            for window in wlist:
                # Score each source+exchange combo for this window
                timeframes = []
                if mode in ("Both", "EOD"):
                    timeframes.append("eod")
                if mode in ("Both", "Intraday"):
                    timeframes.append("intraday")

                #----------SRC-E002 & SRC-E003 Test result
                e002_result = test_src_e002(window, timeframes, sym_data, symbol)
                print(f"E002 result: {e002_result}")
                e002_per_source = e002_result["per_source_score"]  # {(src, exch): score}

                e003_result = test_src_e003(window, timeframes, sym_data, symbol)
                print(f"E003 result: {e003_result}")
                e003_per_source = e003_result["per_source_score"]
                ##----------END
                for tf in timeframes:
                    # Score PRIMARY product data
                    tf_data = sym_data.get(tf, {})
                    primary_pc = sym_data.get("_primary_product", "EQUITY")
                    for exch, src_dict in tf_data.items():

                        if not isinstance(src_dict, dict):
                            continue
                        for src, df in src_dict.items():
                            if df is None or df.empty:
                                continue
                            if tf == "eod":
                                print(f"\033[91m{src}/{exch}/{symbol}/{window.start.date()} to {window.end.date()}\033[0m")
                                ddq_log.info(f"\033[91m{src}/{exch}/{symbol}/{window.start.date()} to {window.end.date()}\033[0m")

                                this_e002   =e002_per_source.get((src, exch), 1)
                                print(f"E002 Score: Source: {src}, Exchange: {exch}: Score: {this_e002}")



                                this_e003 = e003_per_source.get((src, exch), 1)
                                print(f"E002 Score: Source: {src}, Exchange: {exch}: Score: {this_e003}")


                                ws = score_window_eod(df, window, symbol, src, exch, primary_pc, src_002_score=this_e002, src_003_score=this_e003)
                            else:
                                ws = score_window_intraday(df, window, symbol, src, exch, primary_pc)

                            print()
                            #print(ws.to_dict())
                            freq_scores.append(ws.to_dict())
                            total_scores += 1

                    # Score ALL OTHER products from products dict
                    prod_data = sym_data.get("products", {}).get(tf, {})
                    for exch, src_dict in prod_data.items():
                        if not isinstance(src_dict, dict):
                            continue
                        for src, pc_dict in src_dict.items():
                            if not isinstance(pc_dict, dict):
                                continue
                            for pc, df in pc_dict.items():
                                if df is None or df.empty or pc == primary_pc:
                                    continue  # skip primary (already scored above)
                                if tf == "eod":
                                    ws = score_window_eod(df, window, symbol, src, exch, pc)
                                else:
                                    ws = score_window_intraday(df, window, symbol, src, exch, pc)
                                freq_scores.append(ws.to_dict())
                                total_scores += 1

            sym_scores[freq] = freq_scores

            if verbose and freq_scores:
                # Summary for this frequency
                avg = np.mean([s["composite_score"] for s in freq_scores]) if freq_scores else 0
                n_w = len(wlist)
                logger.info("    %s: %d windows × %d combos = %d scores  (avg=%.1f%%)",
                           freq, n_w, len(freq_scores) // max(n_w, 1),
                           len(freq_scores), avg * 100)

        all_scores[symbol] = sym_scores

    elapsed = time.time() - t0
    summary = {
        "total_windows": sum(len(wl) for wl in all_windows.values()),
        "total_scores": total_scores,
        "frequencies": frequencies,
        "symbols": list(all_scores.keys()),
        "elapsed_s": round(elapsed, 2),
    }

    if verbose:
        logger.info("\n  Rolling analysis complete: %d scores in %.1fs", total_scores, elapsed)

    return {
        "windows": all_windows,
        "scores": all_scores,
        "summary": summary,
    }


def _find_date_range(sym_data: dict, mode: str) -> tuple:
    """Find min/max dates across all source+exchange data for a symbol."""
    all_dates = []

    # Also check products dict for date range
    for tf_key in ("eod", "intraday"):
        prod_data = sym_data.get("products", {}).get(tf_key, {})
        for exch, src_dict in prod_data.items():
            if not isinstance(src_dict, dict):
                continue
            for src, pc_dict in src_dict.items():
                if not isinstance(pc_dict, dict):
                    continue
                for pc, df in pc_dict.items():
                    if df is not None and not df.empty:
                        ts_col = "date" if tf_key == "eod" else "datetimestamp"
                        if ts_col in df.columns:
                            all_dates.extend([df[ts_col].min(), df[ts_col].max()])

    timeframes = []
    if mode in ("Both", "EOD"):
        timeframes.append("eod")
    if mode in ("Both", "Intraday"):
        timeframes.append("intraday")

    for tf in timeframes:
        tf_data = sym_data.get(tf, {})
        for exch, src_dict in tf_data.items():
            if not isinstance(src_dict, dict):
                continue
            for src, df in src_dict.items():
                if df is None or df.empty:
                    continue
                ts_col = "date" if tf == "eod" else "datetimestamp"
                if ts_col in df.columns:
                    dates = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                    if not dates.empty:
                        all_dates.extend([dates.min(), dates.max()])

    if not all_dates:
        return None, None

    return min(all_dates), max(all_dates)


# ══════════════════════════════════════════════════════════════════════════════
# BEST SOURCE SELECTION (used by Phase 4)
# ══════════════════════════════════════════════════════════════════════════════

def select_best_sources(rolling_results: dict) -> dict:
    """
    For each symbol+timeframe+window, pick the best Source+Exchange.

    Tiebreak logic (cascading):
      1. Highest composite_score wins
      2. If tied: most clean data points wins
      3. If still tied: check the next coarser frequency window that
         contains this period — whichever source scores higher there wins
      4. If all frequencies tie: first source in priority order wins

    Returns:
        {symbol: {freq: [selection_dict, ...]}}
    """
    _FREQ_ORDER = ["monthly","quarterly","half_yearly","yearly",
                    "2_yearly","3_yearly","5_yearly","10_yearly","15_yearly","all"]

    selections: dict = {}
    scores = rolling_results.get("scores", {})

    for symbol, sym_scores in scores.items():
        sym_sel: dict = {}

        for freq, freq_scores in sym_scores.items():
            by_window: dict[str, list] = {}
            for s in freq_scores:
                pc = s.get('product_class', '')
                key = f"{s['window_label']}|{s['timeframe']}|{pc}"
                by_window.setdefault(key, []).append(s)

            freq_sel = []
            for wkey, candidates in by_window.items():

                parts = wkey.split("|")
                w_label = parts[0]
                w_tf = parts[1] if len(parts) > 1 else ""
                w_pc = parts[2] if len(parts) > 2 else ""

                # Primary sort: score desc, clean_points desc
                candidates.sort(key=lambda c: (-c["composite_score"], -c["n_clean_points"]))


                best = candidates[0]

                # Check for tie (top 2 same score AND clean points)
                if (len(candidates) > 1 and
                    candidates[0]["composite_score"] == candidates[1]["composite_score"] and
                    candidates[0]["n_clean_points"] == candidates[1]["n_clean_points"]):
                    # Cascading tiebreak: check coarser frequencies
                    best = _cascading_tiebreak(
                        candidates, freq, w_label, w_tf, symbol,
                        sym_scores, _FREQ_ORDER
                    )

                freq_sel.append({
                    "window_label": w_label,
                    "timeframe": w_tf,
                    "product_class": w_pc,
                    "window_start": best.get("window_start", ""),
                    "window_end": best.get("window_end", ""),
                    "best_source": best["source"],
                    "best_exchange": best["exchange"],
                    "best_score": round(best["composite_score"], 4),
                    "best_clean_points": best["n_clean_points"],
                    "n_candidates": len(candidates),
                    "all_candidates": [
                        {"source": c["source"], "exchange": c["exchange"],
                         "score": round(c["composite_score"], 4),
                         "clean_points": c["n_clean_points"]}
                        for c in candidates
                    ],
                })

            sym_sel[freq] = freq_sel

        selections[symbol] = sym_sel

    return selections


def _cascading_tiebreak(
    candidates: list, current_freq: str, w_label: str, w_tf: str,
    symbol: str, sym_scores: dict, freq_order: list,
) -> dict:
    """
    Break tie by checking coarser frequency windows.
    Returns the winning candidate.
    """
    current_idx = freq_order.index(current_freq) if current_freq in freq_order else 0
    tied_sources = [(c["source"], c["exchange"]) for c in candidates
                    if c["composite_score"] == candidates[0]["composite_score"]
                    and c["n_clean_points"] == candidates[0]["n_clean_points"]]

    # Check each coarser frequency
    for fi in range(current_idx + 1, len(freq_order)):
        coarser_freq = freq_order[fi]
        coarser_scores = sym_scores.get(coarser_freq, [])
        if not coarser_scores:
            continue

        # Find scores for the tied sources in the coarser frequency
        # that covers the same time period
        for cs in coarser_scores:
            if cs["timeframe"] != w_tf:
                continue
            src_key = (cs["source"], cs["exchange"])
            if src_key in tied_sources:
                # Build a lookup: source+exchange -> coarser score
                coarser_lookup = {}
                for cs2 in coarser_scores:
                    if cs2["timeframe"] == w_tf and cs2["window_label"] == cs["window_label"]:
                        k = (cs2["source"], cs2["exchange"])
                        if k in tied_sources:
                            coarser_lookup[k] = (cs2["composite_score"], cs2["n_clean_points"])

                if len(coarser_lookup) >= 2:
                    # Sort by coarser score
                    sorted_coarser = sorted(coarser_lookup.items(),
                                           key=lambda x: (-x[1][0], -x[1][1]))
                    if sorted_coarser[0][1] != sorted_coarser[1][1]:
                        # Tie broken!
                        winner_src, winner_exch = sorted_coarser[0][0]
                        for c in candidates:
                            if c["source"] == winner_src and c["exchange"] == winner_exch:
                                return c
                break  # Only check the first matching coarser frequency

    # No tiebreak found — return first candidate (arbitrary but deterministic)
    return candidates[0]
