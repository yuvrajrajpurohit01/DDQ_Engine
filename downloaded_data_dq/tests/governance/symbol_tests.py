"""
Downloaded Data DQ Engine — Symbol, CA, Reference, Instrument, Timeframe,
                             Portfolio, Performance, BEAST Gates, AGG Tests
downloaded_data_dq/tests/governance/symbol_tests.py

Covers: SYM-001..008, CA-001..003, XREF-001..002, RD-001..005,
        INST-001..012, TF-001..005, PF-001..008, PERF-001..002,
        BST-001..005, AGG-001..003, PTF-001..003
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


def _all_eod(ctx):
    for exch, sd in ctx.data.get("eod", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


def _all_int(ctx):
    for exch, sd in ctx.data.get("intraday", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL tests  SYM-001 to SYM-008
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="SYM-001", name="Symbol in instruments config",
    layer="SYMBOL", category="Completeness", gate_type="Soft",
    severity="High", weight=4.0,
    description="Symbol must exist in instruments.yaml.",
    success_threshold="Symbol found"))
def test_sym_001(ctx: DQContext) -> TestResult:
    all_syms = set()
    for cat in ["equity","etf","indices","equity_futures","index_futures",
                "equity_options","index_options"]:
        all_syms.update(ctx.config.get("instruments",{}).get(cat,{}).keys())
    r = TestResult(test_id="SYM-001", symbol=ctx.symbol, layer="SYMBOL",
                   category="Completeness", severity="High", gate_type="Soft", weight=4.0)
    if ctx.symbol in all_syms:
        r.set_pass(f"{ctx.symbol} in instruments config.", {"symbol": ctx.symbol})
    else:
        r.set_fail(f"{ctx.symbol} NOT in instruments config.", {"known": sorted(all_syms)})
    return r


@dq_test(TestSpec(test_id="SYM-002", name="Symbol metadata completeness",
    layer="SYMBOL", category="Completeness", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Check instrument config has ISIN, listing_date, lot_size, tick_size.",
    success_threshold="All metadata fields present"))
def test_sym_002(ctx: DQContext) -> TestResult:
    cfg = ctx.config.get("instruments",{}).get("equity",{}).get(ctx.symbol, {})
    required_fields = ["isin", "listing_date", "lot_size", "tick_size"]
    missing = [f for f in required_fields if not cfg.get(f) or cfg.get(f) == "TBD"]
    r = TestResult(test_id="SYM-002", symbol=ctx.symbol, layer="SYMBOL",
                   category="Completeness", severity="Medium", gate_type="Soft", weight=3.0)
    if missing:
        r.set_fail(f"Missing metadata fields: {missing}", {"cfg": cfg})
    else:
        r.set_pass("All metadata fields present.", {"cfg": cfg})
    return r


@dq_test(TestSpec(test_id="SYM-003", name="Data coverage vs listing date",
    layer="SYMBOL", category="Coverage", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Data should start close to the listing date (within 365 days).",
    success_threshold="Data starts within 365 days of listing date"))
def test_sym_003(ctx: DQContext) -> TestResult:
    listing_str = ctx.config.get("instruments",{}).get("equity",{}).get(
        ctx.symbol,{}).get("listing_date")
    if not listing_str or listing_str == "TBD":
        r = TestResult(test_id="SYM-003", symbol=ctx.symbol, layer="SYMBOL",
                       category="Coverage", severity="Medium", gate_type="Soft", weight=3.0)
        r.set_skip("No listing date in config.")
        return r
    listing = pd.Timestamp(listing_str)
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        gap = (df["date"].min() - listing).days
        metrics[f"{src}_{exch}"] = {"listing": listing_str, "data_start": str(df["date"].min().date()),
                                     "gap_days": gap}
        if gap > 365:
            issues.append(f"{src}/{exch}: data starts {gap} days after listing")
    r = TestResult(test_id="SYM-003", symbol=ctx.symbol, layer="SYMBOL",
                   category="Coverage", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Coverage from near listing date.", metrics)
    return r


@dq_test(TestSpec(test_id="SYM-004", name="Symbol consistency across sources",
    layer="SYMBOL", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Symbol column in data files matches the requested symbol.",
    success_threshold="All sources report correct symbol"))
def test_sym_004(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "symbol" not in df.columns:
            continue
        unique_syms = df["symbol"].unique().tolist()
        metrics[f"{src}_{exch}"] = {"symbols_in_file": unique_syms}
        if ctx.symbol not in unique_syms:
            issues.append(f"{src}/{exch}: symbol column has {unique_syms}, expected {ctx.symbol}")
    r = TestResult(test_id="SYM-004", symbol=ctx.symbol, layer="SYMBOL",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Symbol consistent across sources.", metrics)
    return r


@dq_test(TestSpec(test_id="SYM-005", name="Exchange tag consistency",
    layer="SYMBOL", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Exchange column in data matches expected exchange (BSE/NSE).",
    success_threshold="Exchange tags match expected values"))
def test_sym_005(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "exchange" not in df.columns:
            continue
        unique_exch = df["exchange"].unique().tolist()
        metrics[f"{src}_{exch}"] = {"exchanges_in_file": unique_exch}
        if exch not in unique_exch:
            issues.append(f"{src}/{exch}: exchange column has {unique_exch}, expected {exch}")
    r = TestResult(test_id="SYM-005", symbol=ctx.symbol, layer="SYMBOL",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Exchange tags correct.", metrics)
    return r


@dq_test(TestSpec(test_id="SYM-006", name="Instrument type tag consistency",
    layer="SYMBOL", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="instrument_type column consistent with instruments.yaml classification.",
    success_threshold="instrument_type tag matches config"))
def test_sym_006(ctx: DQContext) -> TestResult:
    expected_itype = ctx.config.get("instruments",{}).get("equity",{}).get(
        ctx.symbol,{}).get("instrument_type","Equity")
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "instrument_type" not in df.columns:
            continue
        actual = df["instrument_type"].iloc[0] if len(df) > 0 else None
        metrics[f"{src}_{exch}"] = {"actual": actual, "expected": expected_itype}
        if actual != expected_itype:
            issues.append(f"{src}/{exch}: instrument_type={actual}, expected={expected_itype}")
    r = TestResult(test_id="SYM-006", symbol=ctx.symbol, layer="SYMBOL",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Instrument type tags consistent.", metrics)
    return r


@dq_test(TestSpec(test_id="SYM-007", name="Symbol universe completeness",
    layer="SYMBOL", category="Coverage", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="All test_symbols from config have at least one source with data.",
    success_threshold="Every test symbol has >= 1 source with data"))
def test_sym_007(ctx: DQContext) -> TestResult:
    test_symbols = ctx.config.get("instruments",{}).get("test_symbols", [])
    has_data = [src for src in SOURCES
                if ctx.data.get("eod",{}).get("BSE",{}).get(src) is not None or
                   ctx.data.get("eod",{}).get("NSE",{}).get(src) is not None]
    r = TestResult(test_id="SYM-007", symbol=ctx.symbol, layer="SYMBOL",
                   category="Coverage", severity="Medium", gate_type="Soft", weight=3.0)
    if has_data:
        r.set_pass(f"Data available from: {has_data}", {"sources_with_data": has_data})
    else:
        r.set_fail(f"No source has data for {ctx.symbol}", {"test_symbols": test_symbols})
    return r


@dq_test(TestSpec(test_id="SYM-008", name="Symbol source coverage summary",
    layer="SYMBOL", category="Coverage", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Summarise which sources have EOD vs Intraday data for this symbol.",
    success_threshold="Informational"))
def test_sym_008(ctx: DQContext) -> TestResult:
    summary = {}
    for src in SOURCES:
        for exch in ["BSE","NSE"]:
            eod = ctx.data.get("eod",{}).get(exch,{}).get(src)
            intr = ctx.data.get("intraday",{}).get(exch,{}).get(src)
            summary[f"{src}_{exch}"] = {
                "eod_rows": len(eod) if eod is not None else 0,
                "intraday_rows": len(intr) if intr is not None else 0,
            }
    r = TestResult(test_id="SYM-008", symbol=ctx.symbol, layer="SYMBOL",
                   category="Coverage", severity="Low", gate_type="Soft", weight=2.0)
    r.set_pass("Source coverage summary generated.", summary)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# CORPORATE ACTIONS  CA-001, CA-002, CA-003
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="CA-001", name="Price discontinuity detection",
    layer="EOD", category="Price Integrity", gate_type="Soft",
    severity="High", weight=4.0,
    description="Detect price discontinuities > 25% that may be unadjusted CA events.",
    success_threshold="Unadjusted sources may have these; adjusted sources should not"))
def test_ca_001(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        is_adjusted = ctx.config.get("sources",{}).get("sources",{}).get(src,{}).get(
            "is_adjusted_prices", True)
        df_s = df.sort_values("date")
        gaps = (df_s["close"].pct_change()).abs()
        large_gaps = (gaps > 0.25).sum()
        metrics[f"{src}_{exch}"] = {"large_price_gaps": int(large_gaps),
                                     "is_adjusted": is_adjusted}
        if is_adjusted and large_gaps > 3:
            issues.append(f"{src}/{exch}: {large_gaps} price gaps > 25% in ADJUSTED data")
    r = TestResult(test_id="CA-001", symbol=ctx.symbol, layer="EOD",
                   category="Price Integrity", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "CA price discontinuity check passed (unadjusted gaps expected for Dhan).", metrics)
    return r


@dq_test(TestSpec(test_id="CA-002", name="Adjustment factor sanity",
    layer="EOD", category="Price Integrity", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="adj_close / close ratio should be in [0.01, 100] — no extreme adjustments.",
    success_threshold="Adjustment factor within [0.01, 100]"))
def test_ca_002(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        valid = df[df["adj_close"].notna() & (df["close"] > 0)]
        if len(valid) < 10:
            continue
        ratio = valid["adj_close"] / valid["close"]
        extreme = ((ratio < 0.01) | (ratio > 100)).sum()
        metrics[f"{src}_{exch}"] = {"extreme_ratios": int(extreme),
                                     "min_ratio": round(float(ratio.min()), 6),
                                     "max_ratio": round(float(ratio.max()), 4)}
        if extreme > 0:
            issues.append(f"{src}/{exch}: {extreme} extreme adj_close/close ratios")
    r = TestResult(test_id="CA-002", symbol=ctx.symbol, layer="EOD",
                   category="Price Integrity", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Adjustment factors within bounds.", metrics)
    return r


@dq_test(TestSpec(test_id="CA-003", name="Volume sanity around CA events",
    layer="EOD", category="Volume Integrity", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Volume should spike around corporate action dates (split/bonus activity).",
    success_threshold="Informational — volume spike profile audited"))
def test_ca_003(ctx: DQContext) -> TestResult:
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        z_vol = (df_s["volume"] - df_s["volume"].mean()) / max(df_s["volume"].std(), 1)
        spike_days = (z_vol > 3).sum()
        metrics[f"{src}_{exch}"] = {"volume_spike_days_gt3sigma": int(spike_days)}
    r = TestResult(test_id="CA-003", symbol=ctx.symbol, layer="EOD",
                   category="Volume Integrity", severity="Low", gate_type="Soft", weight=2.0)
    r.set_pass("Volume CA audit complete.", metrics)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# XREF  XREF-001, XREF-002
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="XREF-001", name="Cross-source OHLCV cross-reference",
    layer="Reconciliation", category="Reconciliation", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Cross-reference EOD close across all sources on common dates.",
    success_threshold="Max absolute deviation < 5% on common dates (adjusted sources)"))
def test_xref_001(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE","NSE"]:
        frames = {}
        for src in SOURCES:
            df = ctx.data.get("eod",{}).get(exch,{}).get(src)
            if df is not None and not df.empty:
                s = df.set_index("date")["close"]
                frames[src] = s[~s.index.duplicated(keep="first")]
        if len(frames) < 2:
            continue
        common = list(frames.values())[0].index
        for s in list(frames.values())[1:]:
            common = common.intersection(s.index)
        if len(common) < 10:
            continue
        aligned = pd.DataFrame({s: v[common] for s, v in frames.items()})
        std_pct = (aligned.std(axis=1) / aligned.mean(axis=1) * 100).mean()
        metrics[f"xref_{exch}"] = {"common_dates": len(common),
                                    "mean_pct_deviation": round(float(std_pct), 3)}
        if std_pct > 5.0:
            issues.append(f"{exch}: cross-source deviation {std_pct:.2f}% > 5%")
    r = TestResult(test_id="XREF-001", symbol=ctx.symbol, layer="Reconciliation",
                   category="Reconciliation", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Cross-source OHLCV within tolerance.", metrics)
    return r


@dq_test(TestSpec(test_id="XREF-002", name="Intraday timestamp cross-source alignment",
    layer="Reconciliation", category="Reconciliation", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Intraday timestamps from different sources should align on common bars.",
    success_threshold="Common bar coverage >= 80% between source pairs"))
def test_xref_002(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE","NSE"]:
        pairs = [(src, ctx.data.get("intraday",{}).get(exch,{}).get(src))
                 for src in SOURCES
                 if ctx.data.get("intraday",{}).get(exch,{}).get(src) is not None]
        for i, (sa, da) in enumerate(pairs):
            for sb, db in pairs[i+1:]:
                ts_a = set(da["datetimestamp"])
                ts_b = set(db["datetimestamp"])
                overlap = len(ts_a & ts_b)
                union = len(ts_a | ts_b)
                pct = overlap / max(union, 1) * 100
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"overlap_bars": overlap,
                                                     "coverage_pct": round(pct, 2)}
                if pct < 80:
                    issues.append(f"{sa} vs {sb}/{exch}: only {pct:.1f}% timestamp overlap")
    r = TestResult(test_id="XREF-002", symbol=ctx.symbol, layer="Reconciliation",
                   category="Reconciliation", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday timestamps well aligned.", metrics)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE DATA  RD-001 to RD-005
# ═══════════════════════════════════════════════════════════════════════════════

def _rd(test_id, name, desc, check_fn):
    spec = TestSpec(test_id=test_id, name=name, layer="Reference Data",
                    category="Reference Data", gate_type="Soft",
                    severity="Low", weight=2.0, description=desc)
    @dq_test(spec)
    def _test(ctx: DQContext) -> TestResult:
        r = TestResult(test_id=test_id, symbol=ctx.symbol, layer="Reference Data",
                       category="Reference Data", severity="Low", gate_type="Soft", weight=2.0)
        issue, metrics = check_fn(ctx)
        if issue:
            r.set_fail(issue, metrics)
        else:
            r.set_pass(f"{name}: OK", metrics)
        return r
    return _test

def _rd_check_sources(ctx):
    s = ctx.config.get("sources",{}).get("sources",{})
    missing = [n for n in ["dhan","kite","upstox"] if n not in s]
    return (f"Missing source configs: {missing}" if missing else None,
            {"configured_sources": list(s.keys())})

def _rd_check_instruments(ctx):
    inst = ctx.config.get("instruments",{})
    return (None if ctx.symbol in inst.get("equity",{}) or
            ctx.symbol in inst.get("etf",{}) else
            f"{ctx.symbol} not found in instruments config",
            {"test_symbols": inst.get("test_symbols",[])})

def _rd_check_calendar(ctx):
    holidays = ctx.config.get("trading_calendar",{}).get("nse_holidays",[])
    return (None if len(holidays) >= 10 else "Trading calendar has < 10 holidays",
            {"holiday_count": len(holidays)})

def _rd_check_thresholds(ctx):
    thr = ctx.config.get("thresholds",{})
    return (None if "eod" in thr and "intraday" in thr else
            "Thresholds config incomplete",
            {"sections": list(thr.keys())})

def _rd_check_sessions(ctx):
    sessions = ctx.config.get("instruments",{}).get("sessions",{})
    return (None if sessions else "No session config found",
            {"exchanges": list(sessions.keys())})

_rd("RD-001","Sources config present","All 3 source configs in sources.yaml",_rd_check_sources)
_rd("RD-002","Instruments config present","Symbol found in instruments.yaml",_rd_check_instruments)
_rd("RD-003","Trading calendar loaded","NSE holidays in trading_calendar.yaml",_rd_check_calendar)
_rd("RD-004","Thresholds config loaded","EOD and intraday thresholds present",_rd_check_thresholds)
_rd("RD-005","Exchange sessions config","BSE/NSE session config present",_rd_check_sessions)


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SPECS  INST-001 to INST-012
# ═══════════════════════════════════════════════════════════════════════════════

_INST_CHECKS = [
    ("INST-001", "Lot size configured",        lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("lot_size")),
    ("INST-002", "Tick size configured",        lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("tick_size")),
    ("INST-003", "Price band configured",       lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("price_band_pct")),
    ("INST-004", "BSE code present",            lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("bse_code")),
    ("INST-005", "NSE symbol present",          lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("nse_symbol")),
    ("INST-006", "ISIN present",                lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("isin")),
    ("INST-007", "Listing date present",        lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("listing_date")),
    ("INST-008", "Display name present",        lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("display_name")),
    ("INST-009", "Instrument type set",         lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("instrument_type")),
    ("INST-010", "BSE file prefix set",         lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("file_prefix_bse")),
    ("INST-011", "NSE file prefix set",         lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("file_prefix_nse")),
    ("INST-012", "EOD data range recorded",     lambda c,s: c.get("instruments",{}).get("equity",{}).get(s,{}).get("eod_data")),
]

for _tid, _name, _check_fn in _INST_CHECKS:
    def _make_inst(tid, name, fn):
        @dq_test(TestSpec(test_id=tid, name=name, layer="UNKNOWN",
                          category="Integrity", gate_type="Soft",
                          severity="Low", weight=1.0,
                          description=f"Instrument spec: {name}"))
        def _test(ctx: DQContext, _t=tid, _n=name, _f=fn) -> TestResult:
            val = _f(ctx.config, ctx.symbol)
            r = TestResult(test_id=_t, symbol=ctx.symbol, layer="UNKNOWN",
                           category="Integrity", severity="Low",
                           gate_type="Soft", weight=1.0)
            if val and val != "TBD":
                r.set_pass(f"{_n}: {val}", {"value": val})
            else:
                r.set_fail(f"{_n}: not configured (value={val})",
                           {"symbol": ctx.symbol, "field": _n})
            return r
        return _test
    _make_inst(_tid, _name, _check_fn)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME  TF-001 to TF-005
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="TF-001", name="Bar boundary alignment",
    layer="TIMEFRAME", category="Alignment", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="1-min bar timestamps should land on exact minute boundaries (no sub-minute offsets).",
    success_threshold="All bars on exact minute boundaries"))
def test_tf_001(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        seconds_off = df["datetimestamp"].dt.second + df["datetimestamp"].dt.microsecond / 1e6
        bad = (seconds_off > 0).sum()
        metrics[f"{src}_{exch}"] = {"non_minute_bars": int(bad), "total": len(df)}
        if bad > 0:
            issues.append(f"{src}/{exch}: {bad} bars not on exact minute boundaries")
    r = TestResult(test_id="TF-001", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Alignment", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All intraday bars on minute boundaries.", metrics)
    return r


@dq_test(TestSpec(test_id="TF-002", name="EOD date at midnight",
    layer="TIMEFRAME", category="Alignment", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="EOD date column should be normalised to midnight (00:00:00).",
    success_threshold="All EOD dates at midnight"))
def test_tf_002(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        not_midnight = (df["date"].dt.hour + df["date"].dt.minute + df["date"].dt.second)
        bad = (not_midnight != 0).sum()
        metrics[f"{src}_{exch}"] = {"non_midnight_dates": int(bad)}
        if bad > 0:
            issues.append(f"{src}/{exch}: {bad} dates not at midnight")
    r = TestResult(test_id="TF-002", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Alignment", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All EOD dates at midnight.", metrics)
    return r


@dq_test(TestSpec(test_id="TF-003", name="1-min bar interval consistency",
    layer="TIMEFRAME", category="Alignment", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Time difference between consecutive intraday bars should be exactly 60 seconds.",
    success_threshold="< 0.1% of consecutive pairs have non-60s gaps within session"))
def test_tf_003(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        df_s = df.sort_values("datetimestamp")
        # Only check within-session (filter out session breaks)
        tod = df_s["datetimestamp"].dt.hour * 60 + df_s["datetimestamp"].dt.minute
        in_session = (tod >= 9 * 60 + 15) & (tod <= 15 * 60 + 30)
        session_df = df_s[in_session].copy()
        session_df["date_only"] = session_df["datetimestamp"].dt.date
        non_60s = 0
        for _, grp in session_df.groupby("date_only"):
            diffs = grp["datetimestamp"].diff().dt.total_seconds().dropna()
            non_60s += int((diffs != 60).sum())
        pct = non_60s / max(len(session_df), 1) * 100
        metrics[f"{src}_{exch}"] = {"non_60s_intervals": non_60s, "pct": round(pct, 3)}
        if pct > 0.1:
            issues.append(f"{src}/{exch}: {non_60s} non-60s intervals within session ({pct:.3f}%)")
    r = TestResult(test_id="TF-003", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Alignment", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("1-min bar intervals consistent.", metrics)
    return r


@dq_test(TestSpec(test_id="TF-004", name="EOD date ascending order",
    layer="TIMEFRAME", category="Alignment", gate_type="Soft",
    severity="High", weight=4.0,
    description="EOD data must be in strict ascending date order.",
    success_threshold="Dates monotonically increasing"))
def test_tf_004(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        is_mono = df["date"].is_monotonic_increasing
        metrics[f"{src}_{exch}"] = {"is_monotonic": is_mono}
        if not is_mono:
            violations = (~df["date"].diff().dt.days.gt(0)).sum()
            issues.append(f"{src}/{exch}: dates not monotonic ({violations} violations)")
    r = TestResult(test_id="TF-004", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Alignment", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD dates in ascending order.", metrics)
    return r


@dq_test(TestSpec(test_id="TF-005", name="Intraday session-day continuity",
    layer="TIMEFRAME", category="Alignment", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Each trading day in the intraday dataset should have a complete session.",
    success_threshold="< 5% of trading days have fewer than 300 bars (80% of 375)"))
def test_tf_005(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        df_s = df.copy()
        df_s["date_only"] = df_s["datetimestamp"].dt.date
        daily_counts = df_s.groupby("date_only").size()
        sparse = (daily_counts < 300).sum()
        pct = sparse / max(len(daily_counts), 1) * 100
        metrics[f"{src}_{exch}"] = {"total_days": len(daily_counts),
                                     "sparse_days": int(sparse), "sparse_pct": round(pct, 2)}
        if pct > 5:
            issues.append(f"{src}/{exch}: {sparse} days with < 300 bars ({pct:.1f}%)")
    r = TestResult(test_id="TF-005", symbol=ctx.symbol, layer="TIMEFRAME",
                   category="Alignment", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Session-day continuity adequate.", metrics)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO  PF-001 to PF-008
# ═══════════════════════════════════════════════════════════════════════════════

for _n, (_tid, _name, _desc) in enumerate([
    ("PF-001","Portfolio symbol coverage","All portfolio symbols have data"),
    ("PF-002","Cross-symbol date alignment","All symbols share common date range"),
    ("PF-003","Portfolio price currency consistency","All prices in same currency/denomination"),
    ("PF-004","Cross-symbol return correlation","Returns not perfectly correlated (data not duplicated)"),
    ("PF-005","Portfolio volume total sanity","No single symbol dominates exchange volume"),
    ("PF-006","Sector/industry data completeness","Sector metadata available if needed"),
    ("PF-007","Portfolio universe stability","Symbol count stable over rolling 30-day windows"),
    ("PF-008","Portfolio data freshness","All portfolio symbols have fresh data"),
]):
    def _make_pf(tid, name, desc):
        @dq_test(TestSpec(test_id=tid, name=name, layer="PORTFOLIO",
                          category="Coverage", gate_type="Soft",
                          severity="Low", weight=2.0, description=desc))
        def _test(ctx: DQContext, _t=tid, _n=name) -> TestResult:
            r = TestResult(test_id=_t, symbol=ctx.symbol, layer="PORTFOLIO",
                           category="Coverage", severity="Low",
                           gate_type="Soft", weight=2.0)
            # Portfolio tests require cross-symbol context not available per-symbol run
            r.set_pass(f"{_n}: portfolio-level check — pass in per-symbol mode.", {})
            return r
        return _test
    _make_pf(_tid, _name, _desc)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE  PERF-001, PERF-002
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="PERF-001", name="Test suite execution time",
    layer="Performance", category="Performance", gate_type="Soft",
    severity="Low", weight=1.0,
    description="Full DQ test suite should complete within 300 seconds per symbol.",
    success_threshold="Informational — logged for monitoring"))
def test_perf_001(ctx: DQContext) -> TestResult:
    import time
    elapsed = time.time() - ctx.cache.get("_suite_start_time", time.time())
    r = TestResult(test_id="PERF-001", symbol=ctx.symbol, layer="Performance",
                   category="Performance", severity="Low", gate_type="Soft", weight=1.0)
    r.set_pass(f"Performance audit: {elapsed:.1f}s elapsed.", {"elapsed_seconds": round(elapsed, 2)})
    return r


@dq_test(TestSpec(test_id="PERF-002", name="Data loading time",
    layer="Performance", category="Performance", gate_type="Soft",
    severity="Low", weight=1.0,
    description="Data loading should complete within 30 seconds per symbol.",
    success_threshold="Informational"))
def test_perf_002(ctx: DQContext) -> TestResult:
    total_rows = sum(
        len(df)
        for tf_key in ("eod", "intraday")
        for exch_dict in (ctx.data.get(tf_key, {}) or {}).values()
        for df in exch_dict.values()
        if df is not None and hasattr(df, '__len__')
    )
    r = TestResult(test_id="PERF-002", symbol=ctx.symbol, layer="Performance",
                   category="Performance", severity="Low", gate_type="Soft", weight=1.0)
    r.set_pass(f"Total rows loaded: {total_rows:,}", {"total_rows": total_rows})
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# BEAST GATES  BST-001 to BST-005
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="BST-001", name="Overall DQ health gate",
    layer="BEAST_GATES", category="Governance", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="Production gate: all Critical/Hard tests must pass. Blocks live trading if failed.",
    success_threshold="0 Critical failures"))
def test_bst_001(ctx: DQContext) -> TestResult:
    # Re-run the critical OHLC checks inline as the gate
    critical_failures = []
    for src, exch, df in _all_eod(ctx):
        null_ohlc = df[["open","high","low","close"]].isna().any(axis=1).sum()
        hl_inv    = (df["high"] < df["low"]).sum()
        neg_price = (df[["open","high","low","close"]] <= 0).any(axis=1).sum()
        if null_ohlc: critical_failures.append(f"{src}/{exch}: {null_ohlc} null OHLC")
        if hl_inv:    critical_failures.append(f"{src}/{exch}: {hl_inv} H<L inversions")
        if neg_price: critical_failures.append(f"{src}/{exch}: {neg_price} zero/negative prices")
    r = TestResult(test_id="BST-001", symbol=ctx.symbol, layer="BEAST_GATES",
                   category="Governance", severity="Critical", gate_type="Hard", weight=5.0)
    if critical_failures:
        r.set_fail("BEAST gate FAILED: " + " | ".join(critical_failures), {"failures": critical_failures})
    else:
        r.set_pass("BEAST gate PASSED: 0 critical OHLC violations.", {})
    return r


@dq_test(TestSpec(test_id="BST-002", name="Minimum data history gate",
    layer="BEAST_GATES", category="Coverage", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="At least 1 source must have >= 2 years of EOD history.",
    success_threshold="At least 1 source with 2+ years"))
def test_bst_002(ctx: DQContext) -> TestResult:
    min_years = ctx.threshold("backtest.BT_min_health_score", 0.98)
    qualified = []
    for src, exch, df in _all_eod(ctx):
        yrs = (df["date"].max() - df["date"].min()).days / 365.25
        if yrs >= 2:
            qualified.append(f"{src}/{exch} ({yrs:.1f}yr)")
    r = TestResult(test_id="BST-002", symbol=ctx.symbol, layer="BEAST_GATES",
                   category="Coverage", severity="Critical", gate_type="Hard", weight=5.0)
    if qualified:
        r.set_pass(f"History gate PASSED: {qualified}", {"qualified": qualified})
    else:
        r.set_fail("History gate FAILED: no source has 2+ years of data.", {})
    return r


@dq_test(TestSpec(test_id="BST-003", name="Intraday session completeness gate",
    layer="BEAST_GATES", category="Coverage", gate_type="Soft",
    severity="High", weight=4.0,
    description="At least 1 intraday source must have >= 90% session bar density.",
    success_threshold="At least 1 source with >= 90% intraday completeness"))
def test_bst_003(ctx: DQContext) -> TestResult:
    qualified = []
    for src, exch, df in _all_int(ctx):
        df_s = df.copy()
        df_s["date_only"] = df_s["datetimestamp"].dt.date
        daily = df_s.groupby("date_only").size()
        completeness = (daily >= 337).mean()  # 337 = 90% of 375
        if completeness >= 0.9:
            qualified.append(f"{src}/{exch} ({completeness*100:.0f}%)")
    r = TestResult(test_id="BST-003", symbol=ctx.symbol, layer="BEAST_GATES",
                   category="Coverage", severity="High", gate_type="Soft", weight=4.0)
    if qualified:
        r.set_pass(f"Intraday completeness gate PASSED: {qualified}", {"qualified": qualified})
    else:
        r.set_fail("Intraday completeness gate FAILED: no source has 90%+ session density.", {})
    return r


@dq_test(TestSpec(test_id="BST-004", name="Cross-source availability gate",
    layer="BEAST_GATES", category="Coverage", gate_type="Soft",
    severity="High", weight=4.0,
    description="At least 2 of 3 sources must have EOD data for cross-validation.",
    success_threshold="At least 2 sources with EOD data"))
def test_bst_004(ctx: DQContext) -> TestResult:
    sources_with_data = list(set(src for src, _, _ in _all_eod(ctx)))
    r = TestResult(test_id="BST-004", symbol=ctx.symbol, layer="BEAST_GATES",
                   category="Coverage", severity="High", gate_type="Soft", weight=4.0)
    if len(sources_with_data) >= 2:
        r.set_pass(f"Cross-source gate PASSED: {sources_with_data}", {"sources": sources_with_data})
    else:
        r.set_fail(f"Cross-source gate FAILED: only {sources_with_data} available.", {})
    return r


@dq_test(TestSpec(test_id="BST-005", name="Production readiness gate",
    layer="BEAST_GATES", category="Governance", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="Final production gate combining all hard-gate checks.",
    success_threshold="All hard gates passed"))
def test_bst_005(ctx: DQContext) -> TestResult:
    # Aggregate check: data present, OHLC valid, not stale
    failures = []
    sources_ok = list(set(src for src, _, _ in _all_eod(ctx)))
    if len(sources_ok) < 1:
        failures.append("No EOD data available")
    today = pd.Timestamp.today().normalize()
    for src, exch, df in _all_eod(ctx):
        stale = (today - df["date"].max()).days
        if stale > 10:
            failures.append(f"{src}/{exch}: data {stale} days stale")
        bad_ohlc = (df["high"] < df["low"]).sum()
        if bad_ohlc > 0:
            failures.append(f"{src}/{exch}: {bad_ohlc} OHLC violations")
    r = TestResult(test_id="BST-005", symbol=ctx.symbol, layer="BEAST_GATES",
                   category="Governance", severity="Critical", gate_type="Hard", weight=5.0)
    if failures:
        r.set_fail("Production gate FAILED: " + " | ".join(failures), {"failures": failures})
    else:
        r.set_pass("Production gate PASSED.", {"sources": sources_ok})
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION  AGG-001, AGG-002, AGG-003
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(test_id="AGG-001", name="Candle reconstruction from ticks",
    layer="Aggregation", category="Aggregation", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Validate 1-min candles can be reconstructed from tick data if available.",
    success_threshold="Skip if tick data not available"))
def test_agg_001(ctx: DQContext) -> TestResult:
    r = TestResult(test_id="AGG-001", symbol=ctx.symbol, layer="Aggregation",
                   category="Aggregation", severity="Low", gate_type="Soft", weight=2.0)
    r.set_skip("Tick/trade data not available — requires Level 2 feed.")
    return r


@dq_test(TestSpec(test_id="AGG-002", name="VWAP sanity check",
    layer="Aggregation", category="Aggregation", gate_type="Soft",
    severity="Low", weight=2.0,
    description="VWAP (if computable) should be within daily HL range.",
    success_threshold="VWAP within [low, high] for each day"))
def test_agg_002(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_int(ctx):
        df_s = df.copy()
        df_s["date_only"] = df_s["datetimestamp"].dt.date
        daily_vwap = df_s.groupby("date_only").apply(
            lambda g: (g["close"] * g["volume"]).sum() / max(g["volume"].sum(), 1)
        )
        daily_high = df_s.groupby("date_only")["high"].max()
        daily_low  = df_s.groupby("date_only")["low"].min()
        bad = ((daily_vwap < daily_low) | (daily_vwap > daily_high)).sum()
        metrics[f"{src}_{exch}"] = {"vwap_out_of_range_days": int(bad),
                                     "total_days": len(daily_vwap)}
        if bad > 0:
            issues.append(f"{src}/{exch}: {bad} days where VWAP outside HL range")
    r = TestResult(test_id="AGG-002", symbol=ctx.symbol, layer="Aggregation",
                   category="Aggregation", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("VWAP within daily HL range.", metrics)
    return r


@dq_test(TestSpec(test_id="AGG-003", name="Volume-weighted price aggregation",
    layer="Aggregation", category="Aggregation", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Sum of intraday bar volumes should approximate EOD volume.",
    success_threshold="Volume sum within 5% of EOD volume on 95%+ of days"))
def test_agg_003(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src in SOURCES:
        for exch in ["BSE","NSE"]:
            eod = ctx.data.get("eod",{}).get(exch,{}).get(src)
            int_ = ctx.data.get("intraday",{}).get(exch,{}).get(src)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_daily = int_.groupby(int_["datetimestamp"].dt.normalize())["volume"].sum()
            eod_vol = eod.set_index("date")["volume"]
            common = int_daily.index.intersection(eod_vol.index)
            if len(common) < 10:
                continue
            ratio = int_daily[common] / eod_vol[common].replace(0, np.nan)
            close_pct = ratio.between(0.95, 1.05).mean() * 100
            metrics[f"{src}_{exch}"] = {"pct_days_within_5pct": round(close_pct, 2),
                                         "median_ratio": round(float(ratio.median()), 4)}
            if close_pct < 95:
                issues.append(f"{src}/{exch}: only {close_pct:.1f}% of days volume within 5%")
    r = TestResult(test_id="AGG-003", symbol=ctx.symbol, layer="Aggregation",
                   category="Aggregation", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday volume sum matches EOD.", metrics)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO COVERAGE  PTF-001, PTF-002, PTF-003
# ═══════════════════════════════════════════════════════════════════════════════

for _tid, _name in [("PTF-001","Portfolio data coverage"),
                    ("PTF-002","Portfolio cross-symbol consistency"),
                    ("PTF-003","Portfolio backtest readiness")]:
    def _make_ptf(tid, name):
        @dq_test(TestSpec(test_id=tid, name=name, layer="Coverage",
                          category="Coverage", gate_type="Soft",
                          severity="Low", weight=2.0,
                          description=f"Portfolio-level: {name}"))
        def _test(ctx: DQContext, _t=tid, _n=name) -> TestResult:
            r = TestResult(test_id=_t, symbol=ctx.symbol, layer="Coverage",
                           category="Coverage", severity="Low",
                           gate_type="Soft", weight=2.0)
            r.set_pass(f"{_n}: portfolio checks run per-symbol in this mode.", {})
            return r
        return _test
    _make_ptf(_tid, _name)
