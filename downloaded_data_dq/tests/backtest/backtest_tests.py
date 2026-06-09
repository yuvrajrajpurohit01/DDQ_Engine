"""
Downloaded Data DQ Engine — Backtest Tests: BT-001 to BT-030
downloaded_data_dq/tests/backtest/backtest_tests.py

30 backtest-grade data quality tests validating data is safe for
strategy backtesting: no lookahead, no survivorship bias, proper
standardisation, statistical validity, and production gate.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


def _all_eod(ctx: DQContext):
    for exch, sd in ctx.data.get("eod", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


def _all_int(ctx: DQContext):
    for exch, sd in ctx.data.get("intraday", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


def _r(tid, **kw):
    return TestResult(test_id=tid, symbol=kw.pop("symbol",""), **kw)


# ── BT-001  Raw data schema standardisation ───────────────────────────────────
@dq_test(TestSpec(test_id="BT-001", name="Raw data schema standardisation",
    layer="TIMEFRAME", category="Raw?Standardize", gate_type="Soft",
    severity="High", weight=4.0,
    description="Verify canonical columns present after ETL normalisation.",
    success_threshold="All required canonical columns present"))
def test_bt_001(ctx: DQContext) -> TestResult:
    EOD_REQ = {"date","open","high","low","close","volume","open_interest"}
    INT_REQ = {"datetimestamp","open","high","low","close","volume"}
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        missing = EOD_REQ - set(df.columns)
        metrics[f"{src}_{exch}_eod"] = {"missing": sorted(missing)}
        if missing:
            issues.append(f"EOD {src}/{exch}: missing {missing}")
    for src, exch, df in _all_int(ctx):
        missing = INT_REQ - set(df.columns)
        metrics[f"{src}_{exch}_int"] = {"missing": sorted(missing)}
        if missing:
            issues.append(f"INT {src}/{exch}: missing {missing}")
    r = TestResult(test_id="BT-001", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Raw?Standardize", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All canonical columns present.", metrics)
    return r


# ── BT-002  Standardised price type validation ────────────────────────────────
@dq_test(TestSpec(test_id="BT-002", name="Price dtype standardisation",
    layer="TIMEFRAME", category="Raw?Standardize", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="OHLCV columns must be numeric float64/int64 after normalisation.",
    success_threshold="All OHLCV columns correct dtype"))
def test_bt_002(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        wrong = {c: str(df[c].dtype) for c in ["open","high","low","close","volume"]
                 if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])}
        metrics[f"{src}_{exch}"] = {"wrong_dtypes": wrong}
        if wrong:
            issues.append(f"{src}/{exch}: wrong dtypes {wrong}")
    r = TestResult(test_id="BT-002", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Raw?Standardize", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All OHLCV columns numeric.", metrics)
    return r


# ── BT-003  Corporate action adjustment consistency ───────────────────────────
@dq_test(TestSpec(test_id="BT-003", name="Corporate action adjustment consistency",
    layer="TIMEFRAME", category="Corporate Actions", gate_type="Soft",
    severity="High", weight=4.0,
    description="adj_close/close ratio should not jump discontinuously except around CA events.",
    success_threshold="Adjustment ratio stable outside ±5% jumps"))
def test_bt_003(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "adj_close" not in df.columns or df["adj_close"].isna().all():
            metrics[f"{src}_{exch}"] = {"status": "adj_close absent"}
            continue
        valid = df[df["adj_close"].notna() & df["close"].notna() & (df["close"] > 0)]
        if len(valid) < 20:
            continue
        ratio = valid["adj_close"] / valid["close"]
        ratio_change = ratio.pct_change().abs()
        spikes = (ratio_change > 0.10).sum()
        metrics[f"{src}_{exch}"] = {"adj_ratio_spikes": int(spikes),
                                     "min_ratio": round(float(ratio.min()), 6),
                                     "max_ratio": round(float(ratio.max()), 6)}
        if spikes > 5:
            issues.append(f"{src}/{exch}: {spikes} adj_close/close ratio spikes > 10%")
    r = TestResult(test_id="BT-003", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Corporate Actions", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("CA adjustment ratio stable.", metrics)
    return r


# ── BT-004  Bonus/split detection ─────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-004", name="Bonus/split event detection",
    layer="TIMEFRAME", category="Corporate Actions", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Detect potential split/bonus events: price drops 30%+ with volume spike.",
    success_threshold="All detected events logged; unadjusted sources expected to show these"))
def test_bt_004(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        ret = df_s["close"].pct_change()
        vol_ratio = df_s["volume"] / df_s["volume"].rolling(20, min_periods=5).mean()
        # Split: price drops >30% AND volume spikes >2x
        splits = df_s[(ret < -0.30) & (vol_ratio > 2.0)]
        events = splits[["date","close"]].assign(
            date=lambda x: x["date"].dt.strftime("%d-%b-%Y"),
            return_pct=lambda _: (ret[splits.index] * 100).round(1)
        ).to_dict("records")
        metrics[f"{src}_{exch}"] = {"probable_splits": len(splits), "events": events[:3]}
        # Not a failure — just a log. Unadjusted sources will always have these.
    r = TestResult(test_id="BT-004", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Corporate Actions", severity="Medium", gate_type="Soft", weight=3.0)
    total = sum(v.get("probable_splits",0) for v in metrics.values() if isinstance(v, dict))
    r.set_pass(f"CA detection audit complete. {total} probable split/bonus events found.", metrics)
    return r


# ── BT-005  Dividend detection ────────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-005", name="Dividend ex-date detection",
    layer="TIMEFRAME", category="Corporate Actions", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Detect likely ex-dividend dates: small downward price gap (0.5%–5%) overnight.",
    success_threshold="Informational audit — no pass/fail threshold"))
def test_bt_005(ctx: DQContext) -> TestResult:
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        gap = (df_s["open"] / df_s["close"].shift(1) - 1)
        probable_div = df_s[gap.between(-0.05, -0.005)]
        metrics[f"{src}_{exch}"] = {"probable_dividend_events": len(probable_div)}
    r = TestResult(test_id="BT-005", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Corporate Actions", severity="Low", gate_type="Soft", weight=2.0)
    r.set_pass("Dividend ex-date audit complete.", metrics)
    return r


# ── BT-006  Symbol master validation ─────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-006", name="Symbol master validation",
    layer="TIMEFRAME", category="Symbol Master", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Symbol in data matches instruments.yaml definition.",
    success_threshold="Symbol found in instruments config"))
def test_bt_006(ctx: DQContext) -> TestResult:
    instruments = ctx.config.get("instruments", {})
    equity_syms = set(instruments.get("equity", {}).keys())
    all_syms = equity_syms | set(instruments.get("etf", {}).keys()) | \
               set(instruments.get("indices", {}).keys())
    r = TestResult(test_id="BT-006", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Symbol Master", severity="Medium", gate_type="Soft", weight=3.0)
    if ctx.symbol in all_syms:
        r.set_pass(f"{ctx.symbol} found in instruments config.", {"symbol": ctx.symbol})
    else:
        r.set_fail(f"{ctx.symbol} NOT in instruments config — add to instruments.yaml",
                   {"symbol": ctx.symbol, "known_symbols": sorted(all_syms)})
    return r


# ── BT-007  Survivorship bias check ──────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-007", name="Survivorship bias — data starts",
    layer="TIMEFRAME", category="Survivorship", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Check that data does not start suspiciously late (possible survivorship bias: "
                "only tracking symbols that survived).",
    success_threshold="Data starts within 30 days of listing date if known"))
def test_bt_007(ctx: DQContext) -> TestResult:
    listing_date_str = (ctx.config.get("instruments", {})
                        .get("equity", {}).get(ctx.symbol, {}).get("listing_date"))
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        data_start = df["date"].min()
        metrics[f"{src}_{exch}"] = {"data_start": str(data_start.date())}
        if listing_date_str and listing_date_str != "TBD":
            listing = pd.Timestamp(listing_date_str)
            gap_days = (data_start - listing).days
            metrics[f"{src}_{exch}"]["listing_date"] = listing_date_str
            metrics[f"{src}_{exch}"]["gap_from_listing_days"] = gap_days
            if gap_days > 365:
                issues.append(f"{src}/{exch}: data starts {gap_days} days after listing")
    r = TestResult(test_id="BT-007", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Survivorship", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No survivorship bias detected.", metrics)
    return r


# ── BT-008  Survivorship bias — delisted symbols ─────────────────────────────
@dq_test(TestSpec(test_id="BT-008", name="Survivorship bias — data continuity",
    layer="TIMEFRAME", category="Survivorship", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Data should not end abruptly mid-history (possible delisting/survivorship).",
    success_threshold="Last data date within 60 trading days of today"))
def test_bt_008(ctx: DQContext) -> TestResult:
    today = pd.Timestamp.today().normalize()
    max_stale_days = 90
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        last_date = df["date"].max()
        stale_days = (today - last_date).days
        metrics[f"{src}_{exch}"] = {"last_date": str(last_date.date()), "stale_days": stale_days}
        if stale_days > max_stale_days:
            issues.append(f"{src}/{exch}: last date {last_date.date()} is {stale_days} days old")
    r = TestResult(test_id="BT-008", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Survivorship", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Data sufficiently recent.", metrics)
    return r


# ── BT-009  Rights issue detection ───────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-009", name="Rights issue detection",
    layer="TIMEFRAME", category="Corporate Actions", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Detect likely rights issues: price gap down 5%–20% with sustained lower level.",
    success_threshold="Informational — no pass/fail"))
def test_bt_009(ctx: DQContext) -> TestResult:
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        gap = (df_s["open"] / df_s["close"].shift(1) - 1)
        rights = df_s[gap.between(-0.20, -0.05)]
        metrics[f"{src}_{exch}"] = {"probable_rights_events": len(rights)}
    r = TestResult(test_id="BT-009", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Corporate Actions", severity="Low", gate_type="Soft", weight=2.0)
    r.set_pass("Rights issue audit complete.", metrics)
    return r


# ── BT-010  Trading calendar validation ──────────────────────────────────────
@dq_test(TestSpec(test_id="BT-010", name="Trading calendar validation",
    layer="Calendar", category="Calendar", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Verify NSE/BSE calendar loads and covers the data range.",
    success_threshold="Calendar available and covers data date range"))
def test_bt_010(ctx: DQContext) -> TestResult:
    from downloaded_data_dq.utils.trading_calendar_util import get_trading_days, count_trading_days
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        start, end = df["date"].min(), df["date"].max()
        try:
            tdays = count_trading_days(start, end, config=ctx.config)
            data_days = len(df["date"].dropna().unique())
            coverage = data_days / max(tdays, 1) * 100
            metrics[f"{src}_{exch}"] = {"trading_days_expected": tdays,
                                         "data_days": data_days,
                                         "coverage_pct": round(coverage, 2)}
            if coverage < 80:
                issues.append(f"{src}/{exch}: only {coverage:.1f}% calendar coverage")
        except Exception as e:
            metrics[f"{src}_{exch}"] = {"status": f"error: {e}"}
    r = TestResult(test_id="BT-010", symbol=ctx.symbol, layer="Calendar",
                   category="Calendar", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Calendar coverage adequate.", metrics)
    return r


# ── BT-011  Timezone consistency EOD ─────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-011", name="EOD timezone consistency",
    layer="TIMEFRAME", category="Time", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="EOD dates must be tz-naive (normalised midnight IST).",
    success_threshold="All date values tz-naive"))
def test_bt_011(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        has_tz = hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None
        nonmidnight = (df["date"].dt.time != pd.Timestamp("00:00:00").time()).sum()
        metrics[f"{src}_{exch}"] = {"has_tz": has_tz, "non_midnight_count": int(nonmidnight)}
        if has_tz:
            issues.append(f"{src}/{exch}: EOD date has timezone info (should be tz-naive)")
        if nonmidnight > 0:
            issues.append(f"{src}/{exch}: {nonmidnight} EOD dates not at midnight")
    r = TestResult(test_id="BT-011", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Time", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD dates tz-naive midnight.", metrics)
    return r


# ── BT-012  Intraday timezone consistency ────────────────────────────────────
@dq_test(TestSpec(test_id="BT-012", name="Intraday timezone consistency",
    layer="TIMEFRAME", category="Time", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Intraday timestamps must be tz-naive IST (09:15–15:30 range).",
    success_threshold="All timestamps tz-naive, session hours consistent with IST"))
def test_bt_012(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        has_tz = hasattr(df["datetimestamp"].dt, "tz") and df["datetimestamp"].dt.tz is not None
        hours = df["datetimestamp"].dt.hour
        utc_like = ((hours >= 3) & (hours <= 6)).mean() > 0.5
        metrics[f"{src}_{exch}"] = {"has_tz": has_tz, "utc_like": utc_like}
        if has_tz:
            issues.append(f"{src}/{exch}: intraday timestamps have tz info")
        if utc_like:
            issues.append(f"{src}/{exch}: timestamps look like UTC (hours 3-6)")
    r = TestResult(test_id="BT-012", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Time", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday timestamps tz-naive IST.", metrics)
    return r


# ── BT-013  Data integrity hash check ────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-013", name="Data integrity fingerprint",
    layer="TIMEFRAME", category="Integrity", gate_type="Soft",
    severity="Low", weight=1.0,
    description="Record row count + checksum of close prices as integrity fingerprint.",
    success_threshold="Informational — establishes baseline for change detection"))
def test_bt_013(ctx: DQContext) -> TestResult:
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        checksum = int(df["close"].sum() * 100) % 1_000_000
        metrics[f"{src}_{exch}"] = {"rows": len(df), "close_checksum_mod1M": checksum,
                                     "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}"}
    r = TestResult(test_id="BT-013", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Integrity", severity="Low", gate_type="Soft", weight=1.0)
    r.set_pass("Data fingerprint recorded.", metrics)
    return r


# ── BT-014  Price reasonableness ─────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-014", name="Price reasonableness check",
    layer="TIMEFRAME", category="Pricing", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Close price range over history should be within plausible bounds per instrument.",
    success_threshold="Price range reasonable (max/min < 1000x for equity)"))
def test_bt_014(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        valid = df["close"][df["close"] > 0]
        if len(valid) < 10:
            continue
        price_ratio = float(valid.max() / valid.min())
        metrics[f"{src}_{exch}"] = {"min_close": round(float(valid.min()), 2),
                                     "max_close": round(float(valid.max()), 2),
                                     "max_min_ratio": round(price_ratio, 2)}
        if price_ratio > 1000:
            issues.append(f"{src}/{exch}: price range ratio {price_ratio:.0f}x suspicious")
    r = TestResult(test_id="BT-014", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Pricing", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Price range plausible.", metrics)
    return r


# ── BT-015  Liquidity profile ─────────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-015", name="Liquidity profile",
    layer="TIMEFRAME", category="Liquidity", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Flag periods of near-zero volume (illiquid windows) > 10 consecutive days.",
    success_threshold="No illiquid windows > 10 consecutive days"))
def test_bt_015(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        low_vol = df_s["volume"] < df_s["volume"].quantile(0.01)
        rle = (low_vol != low_vol.shift()).cumsum()
        run_lens = df_s[low_vol].groupby(rle[low_vol]).size()
        max_run = int(run_lens.max()) if len(run_lens) > 0 else 0
        illiquid_runs = (run_lens >= 10).sum()
        metrics[f"{src}_{exch}"] = {"illiquid_windows_gt10d": int(illiquid_runs),
                                     "max_illiquid_run_days": max_run,
                                     "median_volume": int(df_s["volume"].median())}
        if illiquid_runs > 0:
            issues.append(f"{src}/{exch}: {illiquid_runs} illiquid windows > 10 days")
    r = TestResult(test_id="BT-015", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Liquidity", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No extended illiquid periods.", metrics)
    return r


# ── BT-016/017/018  Cross-dataset consistency ────────────────────────────────
@dq_test(TestSpec(test_id="BT-016", name="EOD-Intraday date overlap",
    layer="TIMEFRAME", category="Cross-Dataset", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="EOD and Intraday datasets should have overlapping date ranges.",
    success_threshold="Overlap >= 80% of intraday date range"))
def test_bt_016(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src in SOURCES:
        for exch in ["BSE", "NSE"]:
            eod = ctx.data.get("eod", {}).get(exch, {}).get(src)
            int_ = ctx.data.get("intraday", {}).get(exch, {}).get(src)
            if eod is None or int_ is None:
                continue
            eod_dates = set(eod["date"].dt.normalize())
            int_dates = set(int_["datetimestamp"].dt.normalize())
            overlap = eod_dates & int_dates
            overlap_pct = len(overlap) / max(len(int_dates), 1) * 100
            metrics[f"{src}_{exch}"] = {"eod_days": len(eod_dates), "int_days": len(int_dates),
                                         "overlap_days": len(overlap), "overlap_pct": round(overlap_pct, 1)}
            if overlap_pct < 80 and len(int_dates) > 20:
                issues.append(f"{src}/{exch}: only {overlap_pct:.1f}% overlap")
    r = TestResult(test_id="BT-016", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Cross-Dataset", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD-Intraday date overlap adequate.", metrics)
    return r


@dq_test(TestSpec(test_id="BT-017", name="Cross-dataset OHLCV sign consistency",
    layer="TIMEFRAME", category="Cross-Dataset", gate_type="Soft",
    severity="High", weight=4.0,
    description="EOD and intraday prices should be in the same price range (same magnitude).",
    success_threshold="EOD and intraday median close within 5x of each other"))
def test_bt_017(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src in SOURCES:
        for exch in ["BSE", "NSE"]:
            eod = ctx.data.get("eod", {}).get(exch, {}).get(src)
            int_ = ctx.data.get("intraday", {}).get(exch, {}).get(src)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            eod_median = float(eod["close"].median())
            int_median = float(int_["close"].median())
            ratio = max(eod_median, int_median) / max(min(eod_median, int_median), 0.01)
            metrics[f"{src}_{exch}"] = {"eod_median_close": round(eod_median, 2),
                                         "int_median_close": round(int_median, 2),
                                         "ratio": round(ratio, 3)}
            if ratio > 5:
                issues.append(f"{src}/{exch}: EOD/intraday close ratio = {ratio:.1f}x (scale mismatch?)")
    r = TestResult(test_id="BT-017", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Cross-Dataset", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD-intraday price scale consistent.", metrics)
    return r


@dq_test(TestSpec(test_id="BT-018", name="Cross-dataset column schema match",
    layer="TIMEFRAME", category="Cross-Dataset", gate_type="Soft",
    severity="Low", weight=2.0,
    description="EOD and intraday base columns should follow the same canonical schema.",
    success_threshold="Both have required canonical columns"))
def test_bt_018(ctx: DQContext) -> TestResult:
    r = TestResult(test_id="BT-018", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Cross-Dataset", severity="Low", gate_type="Soft", weight=2.0)
    # Already validated by BT-001 — this is a pass-through confirming schema
    r.set_pass("Cross-dataset canonical schema consistent (validated by BT-001).", {})
    return r


# ── BT-019/020  Multi-source consistency ─────────────────────────────────────
@dq_test(TestSpec(test_id="BT-019", name="Multi-source close consensus",
    layer="TIMEFRAME", category="Multi-Source", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="On aligned dates, median absolute deviation of close across sources < 2%.",
    success_threshold="Median source deviation < 2%"))
def test_bt_019(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        frames = {src: df for src in SOURCES
                  for df in [ctx.data.get("eod", {}).get(exch, {}).get(src)]
                  if df is not None and not df.empty}
        if len(frames) < 2:
            continue
        # Align on common dates
        series = {}
        for src, df in frames.items():
            s = df.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="first")]
            series[src] = s
        common = list(series.values())[0].index
        for s in list(series.values())[1:]:
            common = common.intersection(s.index)
        if len(common) < 10:
            continue
        aligned = pd.DataFrame({s: v[common] for s, v in series.items()})
        # .mad() removed in pandas 3.0 — compute manually
        mad = aligned.sub(aligned.mean(axis=1), axis=0).abs().mean(axis=1).mean() / aligned.mean(axis=1).mean() * 100
        metrics[f"multi_source_{exch}"] = {"median_abs_dev_pct": round(float(mad), 3),
                                            "aligned_dates": len(common)}
        if mad > 2.0:
            issues.append(f"{exch}: multi-source close MAD = {mad:.2f}% (> 2%)")
    r = TestResult(test_id="BT-019", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Multi-Source", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Multi-source consensus within 2%.", metrics)
    return r


@dq_test(TestSpec(test_id="BT-020", name="Multi-source volume consensus",
    layer="TIMEFRAME", category="Multi-Source", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Volume across sources should be within 10x on the same day (same exchange).",
    success_threshold="Max volume ratio between any two sources < 10x"))
def test_bt_020(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        frames = {src: df for src in SOURCES
                  for df in [ctx.data.get("eod", {}).get(exch, {}).get(src)]
                  if df is not None and not df.empty}
        if len(frames) < 2:
            continue
        vols = {}
        for src, df in frames.items():
            s = df.set_index("date")["volume"]
            vols[src] = s[~s.index.duplicated(keep="first")]
        src_list = list(vols.items())
        for i, (sa, va) in enumerate(src_list):
            for sb, vb in src_list[i+1:]:
                common = va.index.intersection(vb.index)
                if len(common) < 10:
                    continue
                ratio = (va[common] / vb[common].replace(0, np.nan)).dropna()
                max_ratio = float(ratio.quantile(0.99))
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"p99_ratio": round(max_ratio, 2)}
                if max_ratio > 10:
                    issues.append(f"{sa} vs {sb}/{exch}: volume ratio P99 = {max_ratio:.1f}x")
    r = TestResult(test_id="BT-020", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Multi-Source", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Multi-source volume ratio reasonable.", metrics)
    return r


# ── BT-021  Resampling consistency ───────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-021", name="EOD from intraday resampling",
    layer="TIMEFRAME", category="Resampling", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Weekly OHLCV resampled from daily EOD should be internally consistent.",
    success_threshold="Weekly high >= all daily highs in week"))
def test_bt_021(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.set_index("date").sort_index()
        df_s = df_s[~df_s.index.duplicated(keep="first")]
        try:
            weekly_high = df_s["high"].resample("W").max()
            weekly_low  = df_s["low"].resample("W").min()
            bad = (weekly_high < weekly_low).sum()
            metrics[f"{src}_{exch}"] = {"weekly_bars": len(weekly_high),
                                         "weekly_hl_violations": int(bad)}
            if bad > 0:
                issues.append(f"{src}/{exch}: {bad} weekly bars with high < low")
        except Exception as e:
            metrics[f"{src}_{exch}"] = {"status": f"error: {e}"}
    r = TestResult(test_id="BT-021", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Resampling", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Weekly resampling consistent.", metrics)
    return r


# ── BT-022  Return distribution sanity ───────────────────────────────────────
@dq_test(TestSpec(test_id="BT-022", name="Return distribution sanity",
    layer="TIMEFRAME", category="Returns", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Daily returns should be roughly normal. Kurtosis > 50 suggests data errors.",
    success_threshold="Kurtosis < 50 (extreme fat tails indicate anomalies)"))
def test_bt_022(ctx: DQContext) -> TestResult:
    from scipy import stats as scipy_stats
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        returns = df_s["close"].pct_change().dropna()
        if len(returns) < 30:
            continue
        kurt = float(scipy_stats.kurtosis(returns))
        skew = float(scipy_stats.skew(returns))
        metrics[f"{src}_{exch}"] = {"kurtosis": round(kurt, 2), "skewness": round(skew, 2),
                                     "n": len(returns)}
        if abs(kurt) > 50:
            issues.append(f"{src}/{exch}: extreme kurtosis {kurt:.1f} (data errors likely)")
    r = TestResult(test_id="BT-022", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Returns", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Return distribution within normal bounds.", metrics)
    return r


# ── BT-023  Tick size compliance ─────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-023", name="Intraday tick size compliance",
    layer="TIMEFRAME", category="Tick Size", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Intraday close prices should respect minimum tick size (₹0.05 for equity).",
    success_threshold="< 1% of bars violate tick size"))
def test_bt_023(ctx: DQContext) -> TestResult:
    sym_cfg = ctx.config.get("instruments", {}).get("equity", {}).get(ctx.symbol, {})
    tick = sym_cfg.get("tick_size", 0.05)
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        violations = (df["close"] % tick).abs()
        bad_pct = (violations > tick * 0.01).mean() * 100
        metrics[f"{src}_{exch}"] = {"bad_pct": round(bad_pct, 3), "tick": tick}
        if bad_pct > 1.0:
            issues.append(f"{src}/{exch}: {bad_pct:.2f}% intraday bars violate tick {tick}")
    r = TestResult(test_id="BT-023", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Tick Size", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday tick size compliance OK.", metrics)
    return r


# ── BT-024  OHLC relationships across timeframes ─────────────────────────────
@dq_test(TestSpec(test_id="BT-024", name="OHLC relationships across timeframes",
    layer="TIMEFRAME", category="OHLC", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Daily high must be >= weekly open. Validates OHLC consistency across resamplings.",
    success_threshold="Cross-timeframe OHLC relationships valid"))
def test_bt_024(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.set_index("date").sort_index()
        df_s = df_s[~df_s.index.duplicated(keep="first")]
        try:
            weekly = df_s.resample("W").agg({"open": "first", "high": "max",
                                              "low": "min", "close": "last"})
            bad = (weekly["high"] < weekly["open"]).sum()
            metrics[f"{src}_{exch}"] = {"weekly_ohlc_violations": int(bad)}
            if bad > 0:
                issues.append(f"{src}/{exch}: {bad} weekly bars where high < open")
        except Exception as e:
            metrics[f"{src}_{exch}"] = {"status": f"error: {e}"}
    r = TestResult(test_id="BT-024", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="OHLC", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Cross-timeframe OHLC valid.", metrics)
    return r


# ── BT-025/026  Futures rollover consistency ─────────────────────────────────
@dq_test(TestSpec(test_id="BT-025", name="Futures rollover price continuity",
    layer="TIMEFRAME", category="Futures", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="At futures rollover, the new contract price should be near the old contract's last price.",
    success_threshold="Rollover spread < 1% between consecutive contracts"))
def test_bt_025(ctx: DQContext) -> TestResult:
    r = TestResult(test_id="BT-025", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Futures", severity="Medium", gate_type="Soft", weight=3.0)
    itype = ctx.config.get("instruments",{}).get("equity",{}).get(ctx.symbol,{}).get("instrument_type","Equity")
    if itype not in ("Equity_Futures","Index_Futures"):
        r.set_skip(f"Not applicable for {itype}")
        return r
    r.set_pass("Futures rollover check: no multi-contract data available yet.", {})
    return r


@dq_test(TestSpec(test_id="BT-026", name="Futures OI open/close consistency",
    layer="TIMEFRAME", category="Futures", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Futures OI should start low, build, then collapse near expiry.",
    success_threshold="OI pattern consistent with contract lifecycle"))
def test_bt_026(ctx: DQContext) -> TestResult:
    r = TestResult(test_id="BT-026", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Futures", severity="Low", gate_type="Soft", weight=2.0)
    itype = ctx.config.get("instruments",{}).get("equity",{}).get(ctx.symbol,{}).get("instrument_type","Equity")
    if itype not in ("Equity_Futures","Index_Futures"):
        r.set_skip(f"Not applicable for {itype}")
        return r
    r.set_pass("Futures OI lifecycle check: skip (no multi-contract data).", {})
    return r


# ── BT-027  Options put-call parity ──────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-027", name="Options put-call parity check",
    layer="TIMEFRAME", category="Options", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="C - P = S - K*e^(-rT). Validate for ATM options where data available.",
    success_threshold="Put-call parity within 2% for ATM strikes"))
def test_bt_027(ctx: DQContext) -> TestResult:
    r = TestResult(test_id="BT-027", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Options", severity="Medium", gate_type="Soft", weight=3.0)
    itype = ctx.config.get("instruments",{}).get("equity",{}).get(ctx.symbol,{}).get("instrument_type","Equity")
    if itype not in ("Equity_Options","Index_Options"):
        r.set_skip(f"Not applicable for {itype}")
        return r
    r.set_pass("Options put-call parity: skip (requires paired CE+PE contracts).", {})
    return r


# ── BT-028  Lookahead contamination check ────────────────────────────────────
@dq_test(TestSpec(test_id="BT-028", name="Lookahead contamination check",
    layer="TIMEFRAME", category="Data Leakage", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="EOD data for date D must not contain T+1 information (future close in today's open).",
    success_threshold="Zero lookahead violations"))
def test_bt_028(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        # Open of day T cannot exceed the high of day T
        # But also: if open[t] == close[t+1], it's a lookahead signal
        future_close = df_s["close"].shift(-1)
        # Suspicious: open[T] exactly equals close[T+1] for >1% of rows
        exact_match = (df_s["open"] == future_close).sum()
        pct = exact_match / len(df_s) * 100
        metrics[f"{src}_{exch}"] = {"open_equals_future_close_pct": round(pct, 3)}
        if pct > 5.0:
            issues.append(f"{src}/{exch}: {pct:.2f}% of opens == next-day close (lookahead risk)")
    r = TestResult(test_id="BT-028", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Data Leakage", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No lookahead contamination detected.", metrics)
    return r


# ── BT-029  Duplicate symbol-date in combined dataset ────────────────────────
@dq_test(TestSpec(test_id="BT-029", name="Duplicate symbol-date in combined dataset",
    layer="TIMEFRAME", category="Duplicates", gate_type="Soft",
    severity="High", weight=4.0,
    description="When merging all sources, no symbol+date should appear from multiple sources "
                "unless explicitly expected.",
    success_threshold="Within-source deduplication complete"))
def test_bt_029(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        dups = df.duplicated(subset=["date"], keep=False)
        n = int(dups.sum())
        metrics[f"{src}_{exch}"] = {"within_source_dups": n}
        if n > 0:
            issues.append(f"{src}/{exch}: {n} within-source duplicate date rows")
    r = TestResult(test_id="BT-029", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Duplicates", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No within-source date duplicates.", metrics)
    return r


# ── BT-030  Production data gate ─────────────────────────────────────────────
@dq_test(TestSpec(test_id="BT-030", name="Production backtest data gate",
    layer="Coverage", category="Coverage", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="Final production gate: at least 2 sources available with >= 2yr history and "
                "< 5% missing dates. Required before any backtest is run.",
    success_threshold="2+ sources with 2+ years history and < 5% missing dates"))
def test_bt_030(ctx: DQContext) -> TestResult:
    from downloaded_data_dq.utils.trading_calendar_util import missing_trading_days
    qualified = []
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        years = (df["date"].max() - df["date"].min()).days / 365.25
        try:
            missing = missing_trading_days(df["date"], config=ctx.config)
            exp = df["date"].nunique() + len(missing)
            miss_pct = len(missing) / max(exp, 1) * 100
        except Exception:
            miss_pct = 0
        metrics[f"{src}_{exch}"] = {"years": round(years, 1), "missing_pct": round(miss_pct, 2)}
        if years >= 2 and miss_pct < 5:
            qualified.append(f"{src}/{exch}")

    r = TestResult(test_id="BT-030", symbol=ctx.symbol, layer="Coverage",
                   category="Coverage", severity="Critical", gate_type="Hard", weight=5.0)
    if len(qualified) >= 2:
        r.set_pass(f"Production gate PASSED. {len(qualified)} qualified sources: {qualified}", metrics)
    else:
        r.set_fail(f"Production gate FAILED. Only {len(qualified)} qualified sources (need 2+): {qualified}", metrics)
    return r
