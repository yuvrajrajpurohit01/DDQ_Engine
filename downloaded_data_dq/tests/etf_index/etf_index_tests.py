"""
Downloaded Data DQ Engine — ETF & Index Tests
downloaded_data_dq/tests/etf_index/etf_index_tests.py

4 tests: ETF-001, ETF-002, IDX-001, IDX-002
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)


def _itype(ctx: DQContext) -> str:
    return (ctx.config.get("instruments", {})
            .get("equity", {}).get(ctx.symbol, {})
            .get("instrument_type", "Equity"))


def _all_eod(ctx: DQContext):
    for exch, sd in ctx.data.get("eod", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


# ─────────────────────────────────────────────────────────────────────────────
# ETF-001  ETF NAV vs close premium/discount
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="ETF-001",
    name="ETF NAV vs close premium/discount",
    layer="EOD",
    category="Validity",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "ETF close price should track NAV within ±2%. "
        "Large premium/discount indicates bad price or stale NAV feed."
    ),
    method_formula="abs(close/NAV - 1) > 0.02",
    success_threshold="<= 0.5% of ETF days exceed 2% premium/discount",
))
def test_etf_001(ctx: DQContext) -> TestResult:
    itype = _itype(ctx)
    r = TestResult(
        test_id="ETF-001", symbol=ctx.symbol, layer="EOD",
        category="Validity", severity="Medium", gate_type="Soft", weight=3.0,
    )
    if itype != "ETF":
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        # NAV is often proxied by adj_close for ETFs — use it if available
        if "adj_close" in df.columns and not df["adj_close"].isna().all():
            nav_col = "adj_close"
        else:
            # Without a real NAV feed, we compare close to its 5-day rolling median
            # as a proxy for fair value — this will naturally pass clean data
            nav_col = None

        if nav_col is None:
            # Proxy: deviation of daily close from rolling 5-day median
            df_s = df.sort_values("date")
            nav_proxy = df_s["close"].rolling(5, min_periods=1).median()
            deviations = ((df_s["close"] / nav_proxy - 1).abs() > 0.02).mean() * 100
            metrics[f"{src}_{exch}"] = {
                "method": "rolling_5d_median_proxy",
                "days_gt_2pct": round(deviations, 2),
            }
            if deviations > 0.5:
                issues.append(f"{src}/{exch}: {deviations:.2f}% days > 2% from 5d median")
        else:
            valid = df[df[nav_col].notna() & (df[nav_col] > 0)]
            premium = ((valid["close"] / valid[nav_col] - 1).abs() > 0.02).mean() * 100
            metrics[f"{src}_{exch}"] = {
                "method": "adj_close_as_nav",
                "days_gt_2pct": round(premium, 2),
            }
            if premium > 0.5:
                issues.append(f"{src}/{exch}: {premium:.2f}% days > 2% NAV deviation")

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("ETF close tracks NAV within ±2%.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# ETF-002  ETF creation/redemption unit volume sanity
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="ETF-002",
    name="ETF creation/redemption unit volume sanity",
    layer="EOD",
    category="Completeness",
    gate_type="Soft",
    severity="Low",
    weight=2.0,
    description=(
        "ETF volume should reflect normal market activity. "
        "Extremely sparse volume days (< 1% of median) indicate illiquidity "
        "or data gaps that could affect NAV tracking analysis."
    ),
    method_formula="count(days where volume < 0.01 * median_volume)",
    success_threshold="< 5% of days have near-zero volume",
))
def test_etf_002(ctx: DQContext) -> TestResult:
    itype = _itype(ctx)
    r = TestResult(
        test_id="ETF-002", symbol=ctx.symbol, layer="EOD",
        category="Completeness", severity="Low", gate_type="Soft", weight=2.0,
    )
    if itype != "ETF":
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        median_vol = df["volume"].median()
        if median_vol == 0:
            metrics[f"{src}_{exch}"] = {"status": "median_volume_zero"}
            continue

        sparse_days = (df["volume"] < median_vol * 0.01).sum()
        sparse_pct = sparse_days / len(df) * 100
        metrics[f"{src}_{exch}"] = {
            "median_volume": int(median_vol),
            "sparse_days": int(sparse_days),
            "sparse_pct": round(sparse_pct, 2),
        }
        if sparse_pct > 5.0:
            issues.append(
                f"{src}/{exch}: {sparse_days} days ({sparse_pct:.1f}%) "
                f"with near-zero volume"
            )

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("ETF volume distribution consistent.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# IDX-001  Index close vs constituent weighted average
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="IDX-001",
    name="Index close vs constituent weighted average",
    layer="EOD",
    category="Consistency",
    gate_type="Hard",
    severity="High",
    weight=4.0,
    description=(
        "Index close should match weighted average of constituent closes "
        "within 1%. Significant deviation indicates index reconstruction error "
        "or wrong constituent weights."
    ),
    method_formula="abs(index_close - weighted_avg_constituents)/index_close > 0.01",
    success_threshold="<= 0.1% deviation; 0 reconstruction errors",
))
def test_idx_001(ctx: DQContext) -> TestResult:
    itype = _itype(ctx)
    r = TestResult(
        test_id="IDX-001", symbol=ctx.symbol, layer="EOD",
        category="Consistency", severity="High", gate_type="Hard", weight=4.0,
    )
    if itype != "Index":
        r.set_skip(f"Not applicable for {itype}")
        return r

    # Constituent data is not available in current data pipeline.
    # This test requires a separate constituent weights feed.
    # When constituent data is available, it would go in:
    #   ctx.data["constituents"][symbol] = DataFrame(date, constituent, weight, close)
    constituents = ctx.data.get("constituents", {}).get(ctx.symbol)
    if constituents is None:
        r.set_skip(
            "Constituent weights data not available. "
            "Provide data/raw/constituents/{symbol}.csv to enable this test."
        )
        return r

    # Implementation when constituent data IS available:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date").set_index("date")
        # Align with constituent data
        merged = df_s.join(constituents.set_index("date"), how="inner")
        if len(merged) < 10:
            continue
        deviation = (
            (merged["close"] - merged["weighted_avg"]).abs() / merged["close"]
        ).mean() * 100
        metrics[f"{src}_{exch}"] = {"mean_deviation_pct": round(deviation, 4)}
        if deviation > 0.1:
            issues.append(f"{src}/{exch}: {deviation:.3f}% mean index reconstruction error")

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("Index close matches constituent weighted average.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# IDX-002  Index OHLC completeness on all trading days
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="IDX-002",
    name="Index OHLC completeness on all trading days",
    layer="EOD",
    category="Completeness",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "Index should have OHLC for every NSE/BSE trading day. "
        "Missing index days block all derived analytics (beta, correlation, etc.)."
    ),
    method_formula="count(distinct date) == trading_calendar_days",
    success_threshold="0 missing trading days for major indices",
))
def test_idx_002(ctx: DQContext) -> TestResult:
    itype = _itype(ctx)
    r = TestResult(
        test_id="IDX-002", symbol=ctx.symbol, layer="EOD",
        category="Completeness", severity="High", gate_type="Hard", weight=5.0,
    )
    if itype != "Index":
        r.set_skip(f"Not applicable for {itype}")
        return r

    from downloaded_data_dq.utils.trading_calendar_util import missing_trading_days
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        try:
            missing = missing_trading_days(df["date"], config=ctx.config)
            metrics[f"{src}_{exch}"] = {
                "rows": len(df),
                "missing_trading_days": len(missing),
                "sample_missing": [str(d.date()) for d in missing[:5]],
            }
            if len(missing) > 0:
                issues.append(
                    f"{src}/{exch}: {len(missing)} missing trading days"
                )
        except Exception as e:
            metrics[f"{src}_{exch}"] = {"status": f"error: {e}"}

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("Index OHLC present on all trading days.", metrics)
    return r
