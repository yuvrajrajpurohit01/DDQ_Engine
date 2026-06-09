"""
Downloaded Data DQ Engine — EOD Tests: EOD-001 to EOD-016
downloaded_data_dq/tests/eod/eod_tests.py

Sprint S3 deliverable: Core EOD completeness, uniqueness, validity tests.

Tests implemented:
  EOD-001  Null / missing values in required columns
  EOD-002  Missing trading dates vs calendar
  EOD-003  Duplicate rows (same date appears more than once)
  EOD-004  Negative or zero prices
  EOD-005  High-Low inversion (High < Low)
  EOD-006  Close outside [Low, High] range
  EOD-007  Zero or negative volume
  EOD-008  Stale / flat prices (same OHLC repeated consecutively)
  EOD-009  Abnormal single-day return (|return| > threshold)
  EOD-010  Open-High-Low-Close internal consistency
  EOD-011  adj_close completeness check
  EOD-012  Open interest non-negative check
  EOD-013  Date continuity (no gap > N calendar days)
  EOD-014  Volume outlier detection (z-score)
  EOD-015  Price outlier detection (z-score on close returns)
  EOD-016  Data coverage / history length check
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)

SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _eod_frames(ctx: DQContext, exchange: str = "BSE"):
    """Yield (source, df) for each available EOD frame."""
    for src in SOURCES:
        df = ctx.data.get("eod", {}).get(exchange, {}).get(src)
        if df is not None and not df.empty:
            yield src, df


def _all_eod_frames(ctx: DQContext):
    """Yield (source, exchange, df) for every available EOD frame."""
    for exch, src_dict in ctx.data.get("eod", {}).items():
        for src, df in src_dict.items():
            if df is not None and not df.empty:
                yield src, exch, df


def _make_result(ctx: DQContext, spec: TestSpec) -> TestResult:
    return TestResult(
        test_id=spec.test_id, symbol=ctx.symbol,
        layer=spec.layer, category=spec.category,
        severity=spec.severity, gate_type=spec.gate_type,
        weight=spec.weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EOD-001  Null / missing values in required columns
# ─────────────────────────────────────────────────────────────────────────────
_S001 = TestSpec(
    test_id="EOD-001", name="Null / missing values",
    layer="EOD", category="Completeness",
    description="Detect NULL / blank values in required EOD columns per source.",
    method_formula="count of NaN per required column; pass if all = 0",
    success_threshold="0 nulls in required columns",
    gate_type="Hard", severity="Critical", weight=5.0,
)

@dq_test(_S001)
def test_eod_001(ctx: DQContext) -> TestResult:
    required = ["date", "open", "high", "low", "close", "volume"]
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        null_counts = {c: int(df[c].isna().sum()) for c in required if c in df.columns}
        total_nulls = sum(null_counts.values())
        metrics[f"{src}_{exch}"] = null_counts
        if total_nulls > 0:
            issues.append(
                f"{src}/{exch}: {total_nulls} nulls found — "
                + ", ".join(f"{c}={n}" for c, n in null_counts.items() if n > 0)
            )

    r = TestResult(test_id="EOD-001", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Critical",
                   gate_type="Hard", weight=5.0)
    if issues:
        r.set_fail(
            f"Null values in required EOD columns: " + " | ".join(issues),
            metrics,
        )
    else:
        r.set_pass("No nulls in required EOD columns across all sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-002  Missing trading dates vs calendar
# ─────────────────────────────────────────────────────────────────────────────
_S002 = TestSpec(
    test_id="EOD-002", name="Missing trading dates",
    layer="EOD", category="Completeness",
    description="Check for missing exchange trading dates per source vs BSE calendar.",
    method_formula="Expected trading days in date range minus actual dates present",
    success_threshold="0 missing trading dates",
    gate_type="Soft", severity="High", weight=5.0,
)

@dq_test(_S002)
def test_eod_002(ctx: DQContext) -> TestResult:
    from downloaded_data_dq.utils.trading_calendar_util import missing_trading_days

    issues: list[str] = []
    metrics: dict = {}
    r = TestResult(test_id="EOD-002", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="High",
                   gate_type="Soft", weight=5.0)

    found_any = False
    for src, exch, df in _all_eod_frames(ctx):
        found_any = True
        try:
            missing = missing_trading_days(df["date"], config=ctx.config)
            metrics[f"{src}_{exch}"] = {
                "expected_range": f"{df['date'].min().date()} → {df['date'].max().date()}",
                "actual_rows": len(df),
                "missing_count": len(missing),
                "missing_sample": [str(d.date()) for d in missing[:5]],
            }
            if missing:
                issues.append(
                    f"{src}/{exch}: {len(missing)} missing trading days "
                    f"(sample: {[str(d.date()) for d in missing[:3]]})"
                )
        except Exception as exc:
            r.set_redo(f"Calendar check failed for {src}/{exch}: {exc}", metrics)
            return r

    if not found_any:
        r.set_data_not_present(ctx.symbol, timeframe="EOD")
        return r

    if issues:
        r.set_fail("Missing trading dates: " + " | ".join(issues), metrics)
    else:
        r.set_pass("No missing trading dates in any source.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-003  Duplicate rows (same date, same symbol)
# ─────────────────────────────────────────────────────────────────────────────
_S003 = TestSpec(
    test_id="EOD-003", name="Duplicate rows",
    layer="EOD", category="Uniqueness",
    description="Ensure one row per symbol per date per source.",
    method_formula="GROUPBY(symbol, date) count > 1",
    success_threshold="0 duplicates",
    gate_type="Soft", severity="Critical", weight=5.0,
)

@dq_test(_S003)
def test_eod_003(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        dups = df[df.duplicated(subset=["date"], keep=False)]
        n = len(dups)
        dup_dates = sorted(dups["date"].dt.strftime("%d-%b-%Y").unique().tolist())
        metrics[f"{src}_{exch}"] = {
            "total_rows": len(df),
            "duplicate_rows": n,
            "duplicate_dates": dup_dates[:10],
        }
        if n > 0:
            issues.append(
                f"{src}/{exch}: {n} duplicate rows on {len(dup_dates)} dates "
                f"(sample: {dup_dates[:3]})"
            )

    r = TestResult(test_id="EOD-003", symbol=ctx.symbol, layer="EOD",
                   category="Uniqueness", severity="Critical",
                   gate_type="Hard", weight=5.0)
    if issues:
        r.set_fail("Duplicate date rows found: " + " | ".join(issues), metrics)
    else:
        r.set_pass("No duplicate date rows in any source.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-004  Negative or zero prices
# ─────────────────────────────────────────────────────────────────────────────
_S004 = TestSpec(
    test_id="EOD-004", name="Negative or zero price",
    layer="EOD", category="Validity",
    description="OHLC prices must be > 0 for equities.",
    method_formula="MIN(open, high, low, close) > 0",
    success_threshold="No invalid price values",
    gate_type="Hard", severity="Critical", weight=5.0,
)

@dq_test(_S004)
def test_eod_004(ctx: DQContext) -> TestResult:
    price_cols = ["open", "high", "low", "close"]
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        src_issues = {}
        for col in price_cols:
            if col in df.columns:
                bad = df[df[col] <= 0]
                if len(bad) > 0:
                    src_issues[col] = {
                        "count": len(bad),
                        "min_value": float(bad[col].min()),
                        "sample_dates": bad["date"].dt.strftime("%d-%b-%Y").tolist()[:3],
                    }
        metrics[f"{src}_{exch}"] = src_issues
        if src_issues:
            detail = ", ".join(f"{c}: {v['count']} rows" for c, v in src_issues.items())
            issues.append(f"{src}/{exch}: {detail}")

    r = TestResult(test_id="EOD-004", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Critical",
                   gate_type="Hard", weight=5.0)
    if issues:
        r.set_fail("Zero/negative prices: " + " | ".join(issues), metrics)
    else:
        r.set_pass("All OHLC prices > 0.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-005  High-Low inversion (High < Low)
# ─────────────────────────────────────────────────────────────────────────────
_S005 = TestSpec(
    test_id="EOD-005", name="High-Low inversion",
    layer="EOD", category="Validity",
    description="High must be >= Low on every row.",
    method_formula="high >= low for all rows",
    success_threshold="No inversions",
    gate_type="Hard", severity="Critical", weight=5.0,
)

@dq_test(_S005)
def test_eod_005(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        inverted = df[df["high"] < df["low"]]
        n = len(inverted)
        metrics[f"{src}_{exch}"] = {
            "inverted_rows": n,
            "sample": inverted[["date","high","low"]].head(3).assign(
                date=lambda x: x["date"].dt.strftime("%d-%b-%Y")
            ).to_dict("records") if n > 0 else [],
        }
        if n > 0:
            issues.append(f"{src}/{exch}: {n} rows where high < low")

    r = TestResult(test_id="EOD-005", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Critical",
                   gate_type="Hard", weight=5.0)
    if issues:
        r.set_fail("High < Low inversions: " + " | ".join(issues), metrics)
    else:
        r.set_pass("High >= Low on all rows.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-006  Close outside [Low, High] range
# ─────────────────────────────────────────────────────────────────────────────
_S006 = TestSpec(
    test_id="EOD-006", name="Close outside High-Low range",
    layer="EOD", category="Validity",
    description="Close price must be within [Low, High] on each day.",
    method_formula="low <= close <= high",
    success_threshold="No out-of-range closes",
    gate_type="Hard", severity="Critical", weight=5.0,
)

@dq_test(_S006)
def test_eod_006(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        bad = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        n = len(bad)
        metrics[f"{src}_{exch}"] = {
            "violations": n,
            "sample": bad[["date","low","close","high"]].head(3).assign(
                date=lambda x: x["date"].dt.strftime("%d-%b-%Y")
            ).to_dict("records") if n > 0 else [],
        }
        if n > 0:
            issues.append(f"{src}/{exch}: {n} rows where close ∉ [low, high]")

    r = TestResult(test_id="EOD-006", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Critical",
                   gate_type="Hard", weight=5.0)
    if issues:
        r.set_fail("Close outside [Low, High]: " + " | ".join(issues), metrics)
    else:
        r.set_pass("Close within [Low, High] on all rows.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-007  Zero or negative volume
# ─────────────────────────────────────────────────────────────────────────────
_S007 = TestSpec(
    test_id="EOD-007", name="Zero or negative volume",
    layer="EOD", category="Validity",
    description="Volume must be >= 0. Flag rows where volume = 0 or < 0.",
    method_formula="volume >= 0; warn if zero_pct > 1%",
    success_threshold="No negative volume; < 1% zero-volume rows",
    gate_type="Soft", severity="Medium", weight=3.0,
)

@dq_test(_S007)
def test_eod_007(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        neg = int((df["volume"] < 0).sum())
        zero = int((df["volume"] == 0).sum())
        zero_pct = zero / len(df) * 100 if len(df) > 0 else 0
        metrics[f"{src}_{exch}"] = {
            "negative_volume_rows": neg,
            "zero_volume_rows": zero,
            "zero_volume_pct": round(zero_pct, 2),
        }
        if neg > 0:
            issues.append(f"{src}/{exch}: {neg} rows with negative volume")
        if zero_pct > 1.0:
            issues.append(
                f"{src}/{exch}: {zero} zero-volume rows ({zero_pct:.1f}%)"
            )

    r = TestResult(test_id="EOD-007", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Medium",
                   gate_type="Soft", weight=3.0)
    if issues:
        r.set_fail("Volume issues: " + " | ".join(issues), metrics)
    else:
        r.set_pass("Volume valid in all sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-008  Stale / flat prices (same close N days in a row)
# ─────────────────────────────────────────────────────────────────────────────
_S008 = TestSpec(
    test_id="EOD-008", name="Stale / flat close price",
    layer="EOD", category="Consistency",
    description="Flag repeated identical close price for N+ consecutive days.",
    method_formula="Run-length encoding on close; flag runs >= threshold",
    success_threshold="No close price repeated >= 5 consecutive days",
    gate_type="Soft", severity="Medium", weight=3.0,
)

@dq_test(_S008)
def test_eod_008(ctx: DQContext) -> TestResult:
    threshold: int = ctx.threshold("eod.EOD_008_max_consecutive_stale_days", 5)
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        df_s = df.sort_values("date")
        # compute run-length of consecutive identical close
        rle = (df_s["close"] != df_s["close"].shift()).cumsum()
        run_lengths = df_s.groupby(rle)["close"].transform("count")
        stale = df_s[run_lengths >= threshold]
        n = len(stale)
        metrics[f"{src}_{exch}"] = {
            "stale_rows": n,
            "threshold": threshold,
            "max_run": int(run_lengths.max()),
        }
        if n > 0:
            issues.append(
                f"{src}/{exch}: {n} rows in stale-price runs "
                f"(max consecutive flat days: {int(run_lengths.max())})"
            )

    r = TestResult(test_id="EOD-008", symbol=ctx.symbol, layer="EOD",
                   category="Consistency", severity="Medium",
                   gate_type="Soft", weight=3.0)
    if issues:
        r.set_fail("Stale/flat prices: " + " | ".join(issues), metrics)
    else:
        r.set_pass(f"No stale-price runs >= {threshold} days.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-009  Abnormal single-day return
# ─────────────────────────────────────────────────────────────────────────────
_S009 = TestSpec(
    test_id="EOD-009", name="Abnormal single-day return",
    layer="EOD", category="Statistical",
    description="Flag single-day close-to-close returns beyond ±25%.",
    method_formula="abs(close/close.shift(1) - 1) > threshold",
    success_threshold="No single-day return > 25%",
    gate_type="Soft", severity="High", weight=4.0,
)

@dq_test(_S009)
def test_eod_009(ctx: DQContext) -> TestResult:
    threshold: float = ctx.threshold("eod.EOD_009_max_daily_return_pct", 25.0) / 100.0
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        df_s = df.sort_values("date")
        returns = df_s["close"].pct_change()
        extreme = df_s[returns.abs() > threshold]
        n = len(extreme)
        metrics[f"{src}_{exch}"] = {
            "extreme_return_rows": n,
            "threshold_pct": threshold * 100,
            "max_return_pct": round(float(returns.abs().max() * 100), 2),
            "sample": extreme[["date", "close"]].assign(
                date=lambda x: x["date"].dt.strftime("%d-%b-%Y"),
                return_pct=lambda _: (returns[extreme.index] * 100).round(2),
            ).head(5).to_dict("records") if n > 0 else [],
        }
        if n > 0:
            issues.append(
                f"{src}/{exch}: {n} days with |return| > {threshold*100:.0f}% "
                f"(max={metrics[f'{src}_{exch}']['max_return_pct']:.1f}%)"
            )

    r = TestResult(test_id="EOD-009", symbol=ctx.symbol, layer="EOD",
                   category="Statistical", severity="High",
                   gate_type="Soft", weight=4.0)
    if issues:
        r.set_fail("Abnormal returns: " + " | ".join(issues), metrics)
    else:
        r.set_pass(f"No returns > ±{threshold*100:.0f}% in any source.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-010  OHLC internal consistency (Open within High-Low)
# ─────────────────────────────────────────────────────────────────────────────
_S010 = TestSpec(
    test_id="EOD-010", name="Open outside High-Low range",
    layer="EOD", category="Validity",
    description="Open price must also lie within [Low, High].",
    method_formula="low <= open <= high",
    success_threshold="No violations",
    gate_type="Soft", severity="High", weight=4.0,
)

@dq_test(_S010)
def test_eod_010(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        bad = df[(df["open"] < df["low"]) | (df["open"] > df["high"])]
        n = len(bad)
        metrics[f"{src}_{exch}"] = {
            "violations": n,
            "sample": bad[["date","low","open","high"]].head(3).assign(
                date=lambda x: x["date"].dt.strftime("%d-%b-%Y")
            ).to_dict("records") if n > 0 else [],
        }
        if n > 0:
            issues.append(f"{src}/{exch}: {n} rows where open ∉ [low, high]")

    r = TestResult(test_id="EOD-010", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="High",
                   gate_type="Soft", weight=4.0)
    if issues:
        r.set_fail("Open outside [Low, High]: " + " | ".join(issues), metrics)
    else:
        r.set_pass("Open within [Low, High] on all rows.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-011  adj_close completeness
# ─────────────────────────────────────────────────────────────────────────────
_S011 = TestSpec(
    test_id="EOD-011", name="adj_close completeness",
    layer="EOD", category="Completeness",
    description="Flag sources where adj_close is entirely NaN.",
    method_formula="count of non-null adj_close / total rows",
    success_threshold="adj_close present for >= 95% of rows",
    gate_type="Soft", severity="Medium", weight=3.0,
)

@dq_test(_S011)
def test_eod_011(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        nan_pct = df["adj_close"].isna().mean() * 100 if "adj_close" in df.columns else 100.0
        metrics[f"{src}_{exch}"] = {
            "adj_close_null_pct": round(nan_pct, 2),
            "total_rows": len(df),
        }
        if nan_pct > 5.0:
            issues.append(
                f"{src}/{exch}: adj_close is {nan_pct:.1f}% NaN "
                f"({'entirely missing' if nan_pct == 100.0 else 'mostly missing'})"
            )

    r = TestResult(test_id="EOD-011", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Medium",
                   gate_type="Soft", weight=3.0)
    if issues:
        r.set_fail("adj_close gaps: " + " | ".join(issues), metrics)
    else:
        r.set_pass("adj_close sufficiently populated in all sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-012  Open interest non-negative
# ─────────────────────────────────────────────────────────────────────────────
_S012 = TestSpec(
    test_id="EOD-012", name="Negative open interest",
    layer="EOD", category="Validity",
    description="open_interest must be >= 0.",
    method_formula="MIN(open_interest) >= 0",
    success_threshold="No negative OI values",
    gate_type="Soft", severity="Low", weight=2.0,
)

@dq_test(_S012)
def test_eod_012(ctx: DQContext) -> TestResult:
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        if "open_interest" not in df.columns:
            metrics[f"{src}_{exch}"] = {"status": "column absent"}
            continue
        neg = int((df["open_interest"] < 0).sum())
        metrics[f"{src}_{exch}"] = {
            "negative_oi_rows": neg,
            "min_oi": float(df["open_interest"].min()),
            "max_oi": float(df["open_interest"].max()),
        }
        if neg > 0:
            issues.append(f"{src}/{exch}: {neg} rows with negative OI")

    r = TestResult(test_id="EOD-012", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Low",
                   gate_type="Soft", weight=2.0)
    if issues:
        r.set_fail("Negative open interest: " + " | ".join(issues), metrics)
    else:
        r.set_pass("open_interest >= 0 in all sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-013  Date continuity — no gap > N calendar days
# ─────────────────────────────────────────────────────────────────────────────
_S013 = TestSpec(
    test_id="EOD-013", name="Date continuity gaps",
    layer="EOD", category="Completeness",
    description="Flag any gap > 10 calendar days between consecutive trading dates.",
    method_formula="max(date.diff()) <= 10 calendar days",
    success_threshold="No gap > 10 calendar days",
    gate_type="Soft", severity="Medium", weight=3.0,
)

@dq_test(_S013)
def test_eod_013(ctx: DQContext) -> TestResult:
    threshold_days: int = ctx.threshold("eod.EOD_021_max_gap_days", 10)
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        df_s = df.sort_values("date")
        gaps = df_s["date"].diff().dt.days.dropna()
        large_gaps = gaps[gaps > threshold_days]
        metrics[f"{src}_{exch}"] = {
            "max_gap_days": int(gaps.max()),
            "gaps_over_threshold": len(large_gaps),
            "threshold": threshold_days,
            "sample_gaps": [
                {
                    "from_date": df_s.loc[idx - 1, "date"].strftime("%d-%b-%Y")
                    if idx > 0 else "N/A",
                    "to_date": df_s.loc[idx, "date"].strftime("%d-%b-%Y"),
                    "gap_days": int(g),
                }
                for idx, g in large_gaps.items()
            ][:5],
        }
        if len(large_gaps) > 0:
            issues.append(
                f"{src}/{exch}: {len(large_gaps)} gap(s) > {threshold_days} days "
                f"(max={int(gaps.max())} days)"
            )

    r = TestResult(test_id="EOD-013", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Medium",
                   gate_type="Soft", weight=3.0)
    if issues:
        r.set_fail("Date continuity gaps: " + " | ".join(issues), metrics)
    else:
        r.set_pass(f"No gaps > {threshold_days} days in any source.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-014  Volume outlier detection (z-score)
# ─────────────────────────────────────────────────────────────────────────────
_S014 = TestSpec(
    test_id="EOD-014", name="Volume outlier (z-score)",
    layer="EOD", category="Statistical",
    description="Flag volume observations with z-score > 5 (extreme outliers).",
    method_formula="z = (volume - mean) / std; flag if |z| > 5",
    success_threshold="< 0.5% of rows flagged",
    gate_type="Soft", severity="Low", weight=2.0,
)

@dq_test(_S014)
def test_eod_014(ctx: DQContext) -> TestResult:
    z_threshold: float = ctx.threshold("eod.EOD_013_volume_zscore_threshold", 5.0)
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        vol = df["volume"].replace(0, np.nan).dropna().astype(float)
        if len(vol) < 30:
            metrics[f"{src}_{exch}"] = {"status": "insufficient data"}
            continue
        mean_v, std_v = vol.mean(), vol.std()
        if std_v == 0:
            metrics[f"{src}_{exch}"] = {"status": "zero std — uniform volume"}
            continue
        z = ((vol - mean_v) / std_v).abs()
        outliers = df.loc[z[z > z_threshold].index]
        outlier_pct = len(outliers) / len(df) * 100
        metrics[f"{src}_{exch}"] = {
            "outlier_rows": len(outliers),
            "outlier_pct": round(outlier_pct, 3),
            "z_threshold": z_threshold,
            "max_z": round(float(z.max()), 2),
        }
        if outlier_pct > 0.5:
            issues.append(
                f"{src}/{exch}: {len(outliers)} volume outliers "
                f"({outlier_pct:.2f}%, z>{z_threshold})"
            )

    r = TestResult(test_id="EOD-014", symbol=ctx.symbol, layer="EOD",
                   category="Statistical", severity="Low",
                   gate_type="Soft", weight=2.0)
    if issues:
        r.set_fail("Volume outliers: " + " | ".join(issues), metrics)
    else:
        r.set_pass("Volume outliers within acceptable limits.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-015  Price outlier detection (z-score on log returns)
# ─────────────────────────────────────────────────────────────────────────────
_S015 = TestSpec(
    test_id="EOD-015", name="Price return outlier (z-score)",
    layer="EOD", category="Statistical",
    description="Flag extreme log-return outliers on close price using z-score.",
    method_formula="z = (log_return - mean) / std; flag |z| > 5",
    success_threshold="< 0.5% of return observations flagged",
    gate_type="Soft", severity="Low", weight=2.0,
)

@dq_test(_S015)
def test_eod_015(ctx: DQContext) -> TestResult:
    z_threshold: float = ctx.threshold("eod.EOD_014_price_zscore_threshold", 5.0)
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        df_s = df.sort_values("date")
        log_ret = np.log(df_s["close"] / df_s["close"].shift(1)).dropna()
        if len(log_ret) < 30:
            metrics[f"{src}_{exch}"] = {"status": "insufficient data"}
            continue
        mean_r, std_r = log_ret.mean(), log_ret.std()
        if std_r == 0:
            metrics[f"{src}_{exch}"] = {"status": "zero std — uniform returns"}
            continue
        z = ((log_ret - mean_r) / std_r).abs()
        outliers = df_s.loc[z[z > z_threshold].index]
        outlier_pct = len(outliers) / len(df) * 100
        metrics[f"{src}_{exch}"] = {
            "outlier_rows": len(outliers),
            "outlier_pct": round(outlier_pct, 3),
            "z_threshold": z_threshold,
            "max_z": round(float(z.max()), 2),
        }
        if outlier_pct > 0.5:
            issues.append(
                f"{src}/{exch}: {len(outliers)} return outliers "
                f"({outlier_pct:.2f}%, z>{z_threshold})"
            )

    r = TestResult(test_id="EOD-015", symbol=ctx.symbol, layer="EOD",
                   category="Statistical", severity="Low",
                   gate_type="Soft", weight=2.0)
    if issues:
        r.set_fail("Return outliers: " + " | ".join(issues), metrics)
    else:
        r.set_pass("Price return outliers within limits.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-016  Data coverage / history length
# ─────────────────────────────────────────────────────────────────────────────
_S016 = TestSpec(
    test_id="EOD-016", name="History length coverage",
    layer="EOD", category="Completeness",
    description="Each source must have at least 2 years of EOD history.",
    method_formula="(max_date - min_date).years >= 2",
    success_threshold="At least 2 years of data",
    gate_type="Soft", severity="Medium", weight=3.0,
)

@dq_test(_S016)
def test_eod_016(ctx: DQContext) -> TestResult:
    min_years: int = ctx.threshold("eod.EOD_020_min_history_years", 2)
    issues: list[str] = []
    metrics: dict = {}

    for src, exch, df in _all_eod_frames(ctx):
        min_d, max_d = df["date"].min(), df["date"].max()
        years = (max_d - min_d).days / 365.25
        metrics[f"{src}_{exch}"] = {
            "start_date": min_d.strftime("%d-%b-%Y"),
            "end_date": max_d.strftime("%d-%b-%Y"),
            "years": round(years, 1),
            "rows": len(df),
            "min_required_years": min_years,
        }
        if years < min_years:
            issues.append(
                f"{src}/{exch}: only {years:.1f} years of data "
                f"({min_d.strftime('%d-%b-%Y')} → {max_d.strftime('%d-%b-%Y')})"
            )

    r = TestResult(test_id="EOD-016", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Medium",
                   gate_type="Soft", weight=3.0)
    if issues:
        r.set_fail(f"Insufficient history (< {min_years} years): " + " | ".join(issues), metrics)
    else:
        r.set_pass(f"All sources have >= {min_years} years of data.", metrics)
    return r
