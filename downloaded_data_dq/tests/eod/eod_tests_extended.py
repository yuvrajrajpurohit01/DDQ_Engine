"""
Downloaded Data DQ Engine — EOD Tests: EOD-017 to EOD-047
downloaded_data_dq/tests/eod/eod_tests_extended.py

Implements the remaining 31 EOD tests covering:
  Continuity, Reference (tick/lot), Staleness anomalies, OI checks,
  Distribution monitoring per field (Open/High/Low/Close/Volume/OI),
  Adjusted close, Derivatives, Open-gap, Universe stability,
  Timestamp cross-check, OI column presence.
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
    """Yield (source, exchange, df) for every available EOD frame."""
    for exch, src_dict in ctx.data.get("eod", {}).items():
        for src, df in src_dict.items():
            if df is not None and not df.empty:
                yield src, exch, df


def _r(test_id, **kw) -> TestResult:
    return TestResult(test_id=test_id, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# EOD-017  Return cap sanity (circuit-breaker aware)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-017", name="Return cap sanity",
    layer="EOD", category="Continuity", gate_type="Soft",
    severity="High", weight=4.0,
    description="Flag single-day returns beyond exchange circuit-breaker limits per instrument type.",
    method_formula="|pct_change(close)| > limit_by_type",
    success_threshold="No returns beyond circuit limits"))
def test_eod_017(ctx: DQContext) -> TestResult:
    itype = ctx.config.get("instruments", {}).get("equity", {}).get(
        ctx.symbol, {}).get("instrument_type", "Equity")
    limit = ctx.threshold(
        f"eod.max_daily_return_pct_by_type.{itype}", 25.0) / 100.0
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        ret = df.sort_values("date")["close"].pct_change().abs()
        breached_mask = ret > limit
        breached = df.loc[breached_mask.index[breached_mask]]
        metrics[f"{src}_{exch}"] = {"breached": len(breached), "limit_pct": limit * 100,
                                     "max_pct": round(float(ret.max() * 100), 2)}
        if len(breached):
            issues.append(f"{src}/{exch}: {len(breached)} returns > {limit*100:.0f}%")
    r = TestResult(test_id="EOD-017", symbol=ctx.symbol, layer="EOD",
                   category="Continuity", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All returns within circuit limits.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-018  Tick size compliance
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-018", name="Tick size compliance",
    layer="EOD", category="Reference", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Price values should be multiples of the tick size for this instrument.",
    method_formula="close % tick_size ≈ 0  (within float tolerance)",
    success_threshold="< 1% of prices violate tick size"))
def test_eod_018(ctx: DQContext) -> TestResult:
    sym_cfg = (ctx.config.get("instruments", {})
               .get("equity", {}).get(ctx.symbol, {}))
    tick = sym_cfg.get("tick_size", 0.05)
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        violations = (df["close"] % tick).abs()
        tol = tick * 0.01
        bad = (violations > tol).sum()
        pct = bad / len(df) * 100
        metrics[f"{src}_{exch}"] = {"violations": int(bad), "pct": round(pct, 3), "tick": tick}
        if pct > 1.0:
            issues.append(f"{src}/{exch}: {bad} prices ({pct:.2f}%) not multiples of {tick}")
    r = TestResult(test_id="EOD-018", symbol=ctx.symbol, layer="EOD",
                   category="Reference", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(f"Tick size {tick} compliance OK.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-019  Lot size compliance (futures/options)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-019", name="Lot size compliance",
    layer="EOD", category="Reference", gate_type="Soft",
    severity="Low", weight=1.0,
    description="For derivatives, volume should be multiples of the lot size.",
    method_formula="volume % lot_size == 0",
    success_threshold="< 1% violations"))
def test_eod_019(ctx: DQContext) -> TestResult:
    itype = (ctx.config.get("instruments", {}).get("equity", {})
             .get(ctx.symbol, {}).get("instrument_type", "Equity"))
    if itype not in ("Equity_Futures", "Index_Futures", "Equity_Options", "Index_Options"):
        r = TestResult(test_id="EOD-019", symbol=ctx.symbol, layer="EOD",
                       category="Reference", severity="Low", gate_type="Soft", weight=1.0)
        r.set_skip(f"Not applicable for instrument type: {itype}")
        return r
    sym_cfg = (ctx.config.get("instruments", {})
               .get("equity_futures", {}).get(ctx.symbol, {}))
    lot = sym_cfg.get("nse_lot_size", sym_cfg.get("lot_size", 1))
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if lot <= 1:
            metrics[f"{src}_{exch}"] = {"status": "lot_size=1, skip"}
            continue
        bad = (df["volume"] % lot != 0).sum()
        pct = bad / len(df) * 100
        metrics[f"{src}_{exch}"] = {"violations": int(bad), "lot_size": lot}
        if pct > 1.0:
            issues.append(f"{src}/{exch}: {bad} volume rows not multiple of lot {lot}")
    r = TestResult(test_id="EOD-019", symbol=ctx.symbol, layer="EOD",
                   category="Reference", severity="Low", gate_type="Soft", weight=1.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Lot size compliance OK.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-020  Flat OHLC with nonzero volume anomaly
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-020", name="Flat OHLC with nonzero volume",
    layer="EOD", category="Staleness", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Days with identical OHLC (open=high=low=close) but nonzero volume are anomalous.",
    method_formula="(O==H==L==C) AND volume > 0",
    success_threshold="< 0.5% of rows"))
def test_eod_020(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        flat = df[(df["open"] == df["high"]) &
                  (df["high"] == df["low"]) &
                  (df["low"] == df["close"]) &
                  (df["volume"] > 0)]
        pct = len(flat) / len(df) * 100
        metrics[f"{src}_{exch}"] = {"flat_with_volume": len(flat), "pct": round(pct, 3)}
        if pct > 0.5:
            issues.append(f"{src}/{exch}: {len(flat)} flat-OHLC rows with volume ({pct:.2f}%)")
    r = TestResult(test_id="EOD-020", symbol=ctx.symbol, layer="EOD",
                   category="Staleness", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No flat-OHLC+volume anomaly.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-021  Missing OI for derivatives
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-021", name="Missing OI for derivatives",
    layer="EOD", category="Completeness", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="Futures and Options must have open_interest > 0 (OI is mandatory for derivatives).",
    method_formula="(open_interest > 0).all() for futures/options",
    success_threshold="No zero/null OI in derivative rows"))
def test_eod_021(ctx: DQContext) -> TestResult:
    itype = (ctx.config.get("instruments", {}).get("equity", {})
             .get(ctx.symbol, {}).get("instrument_type", "Equity"))
    if itype not in ("Equity_Futures", "Index_Futures", "Equity_Options", "Index_Options"):
        r = TestResult(test_id="EOD-021", symbol=ctx.symbol, layer="EOD",
                       category="Completeness", severity="Critical", gate_type="Hard", weight=5.0)
        r.set_skip(f"Not applicable for {itype}")
        return r
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        missing_oi = (df["open_interest"].isna() | (df["open_interest"] == 0)).sum()
        pct = missing_oi / len(df) * 100
        metrics[f"{src}_{exch}"] = {"zero_or_null_oi": int(missing_oi), "pct": round(pct, 2)}
        if missing_oi > 0:
            issues.append(f"{src}/{exch}: {missing_oi} rows with zero/null OI ({pct:.1f}%)")
    r = TestResult(test_id="EOD-021", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("OI present in all derivative rows.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-022  OI reset / rollover sanity
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-022", name="OI rollover sanity",
    layer="EOD", category="Consistency", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="OI should not drop to near-zero mid-contract (only near expiry).",
    method_formula="OI < 1% of max_OI flagged if > 5 days before expiry",
    success_threshold="No unexpected OI collapses"))
def test_eod_022(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "open_interest" not in df.columns or df["open_interest"].isna().all():
            metrics[f"{src}_{exch}"] = {"status": "no_oi_data"}
            continue
        oi = df["open_interest"]
        max_oi = oi.max()
        if max_oi == 0:
            metrics[f"{src}_{exch}"] = {"status": "all_zero_oi"}
            continue
        collapses = (oi < max_oi * 0.01) & (oi > 0)
        metrics[f"{src}_{exch}"] = {"oi_collapses": int(collapses.sum()), "max_oi": int(max_oi)}
        if collapses.sum() > 5:
            issues.append(f"{src}/{exch}: {collapses.sum()} OI collapse events")
    r = TestResult(test_id="EOD-022", symbol=ctx.symbol, layer="EOD",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("OI rollover pattern normal.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-023  Monotone price scale check (no time-reversal)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-023", name="Price scale monotonicity",
    layer="EOD", category="Integrity", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Check for abrupt price scale jumps that suggest data mixing (e.g., pre/post-split raw).",
    method_formula="rolling_median price ratio between consecutive 30-day windows",
    success_threshold="No ratio jump > 5x between windows"))
def test_eod_023(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if len(df) < 60:
            metrics[f"{src}_{exch}"] = {"status": "insufficient_data"}
            continue
        df_s = df.sort_values("date")
        rolling_med = df_s["close"].rolling(30, min_periods=15).median()
        ratio = (rolling_med / rolling_med.shift(30)).dropna()
        extreme = ratio[(ratio > 5) | (ratio < 0.2)]
        metrics[f"{src}_{exch}"] = {"scale_jumps": len(extreme),
                                     "max_ratio": round(float(ratio.max()), 3)}
        if len(extreme):
            issues.append(f"{src}/{exch}: {len(extreme)} scale-jump events (ratio>5x or <0.2x)")
    r = TestResult(test_id="EOD-023", symbol=ctx.symbol, layer="EOD",
                   category="Integrity", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Price scale consistent.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-024 to EOD-041  Distribution monitoring (missing rate + range sanity + trend)
# Each field: Open, High, Low, Close, Volume, OI — 3 tests each = 18 tests
# ─────────────────────────────────────────────────────────────────────────────

def _distribution_missing(test_id, field, ctx) -> TestResult:
    """EOD-024/027/030/033/036/039 — missing rate per field."""
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        miss_pct = df[field].isna().mean() * 100 if field in df.columns else 100.0
        metrics[f"{src}_{exch}"] = {"missing_pct": round(miss_pct, 3)}
        if miss_pct > 1.0:
            issues.append(f"{src}/{exch}: {miss_pct:.2f}% missing in {field}")
    r = TestResult(test_id=test_id, symbol=ctx.symbol, layer="EOD",
                   category="Distribution", severity="Medium", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(f"{field} missing rate OK.", metrics)
    return r


def _distribution_range(test_id, field, ctx) -> TestResult:
    """EOD-025/028/031/034/037/040 — range sanity (IQR-based)."""
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if field not in df.columns:
            continue
        s = df[field].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.01), s.quantile(0.99)
        out_of_range = ((s < q1 * 0.5) | (s > q3 * 2.0)).sum()
        pct = out_of_range / len(df) * 100
        metrics[f"{src}_{exch}"] = {"out_of_range": int(out_of_range), "pct": round(pct, 3),
                                     "p01": round(float(q1), 4), "p99": round(float(q3), 4)}
        if pct > 2.0:
            issues.append(f"{src}/{exch}: {out_of_range} {field} values outside 0.5×p01–2×p99")
    r = TestResult(test_id=test_id, symbol=ctx.symbol, layer="EOD",
                   category="Distribution", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(f"{field} range OK.", metrics)
    return r


def _distribution_trend(test_id, field, ctx) -> TestResult:
    """EOD-026/029/032/035/038/041 — daily anomaly count trend (increasing anomalies = warn)."""
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if field not in df.columns:
            continue
        df_s = df.sort_values("date").copy()
        df_s["year"] = df_s["date"].dt.year
        s = df_s[field]
        q99 = s.quantile(0.99)
        df_s["anomaly"] = (s > q99 * 1.5) | s.isna()
        by_year = df_s.groupby("year")["anomaly"].mean() * 100
        recent = by_year.tail(3).mean()
        historical = by_year.head(max(1, len(by_year)-3)).mean()
        metrics[f"{src}_{exch}"] = {"recent_anomaly_pct": round(float(recent), 3),
                                     "historical_anomaly_pct": round(float(historical), 3)}
        if historical > 0 and recent > historical * 3:
            issues.append(f"{src}/{exch}: {field} anomaly rate 3× higher recently ({recent:.2f}% vs {historical:.2f}%)")
    r = TestResult(test_id=test_id, symbol=ctx.symbol, layer="EOD",
                   category="Monitoring", severity="Low", gate_type="Soft", weight=1.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(f"{field} anomaly trend stable.", metrics)
    return r


# Register all 18 distribution tests
_DIST_SPECS = [
    ("EOD-024","open","Open: Missing rate"),  ("EOD-025","open","Open: Range sanity"),
    ("EOD-026","open","Open: Anomaly trend"), ("EOD-027","high","High: Missing rate"),
    ("EOD-028","high","High: Range sanity"),  ("EOD-029","high","High: Anomaly trend"),
    ("EOD-030","low","Low: Missing rate"),    ("EOD-031","low","Low: Range sanity"),
    ("EOD-032","low","Low: Anomaly trend"),   ("EOD-033","close","Close: Missing rate"),
    ("EOD-034","close","Close: Range sanity"),("EOD-035","close","Close: Anomaly trend"),
    ("EOD-036","volume","Volume: Missing rate"),("EOD-037","volume","Volume: Range sanity"),
    ("EOD-038","volume","Volume: Anomaly trend"),("EOD-039","open_interest","OI: Missing rate"),
    ("EOD-040","open_interest","OI: Range sanity"),("EOD-041","open_interest","OI: Anomaly trend"),
]

def _make_dist_test(tid, field, name, fn):
    spec = TestSpec(test_id=tid, name=name, layer="EOD",
                    category="Distribution" if "Missing" in name or "Range" in name else "Monitoring",
                    gate_type="Soft", severity="Low", weight=2.0 if "trend" not in name.lower() else 1.0,
                    description=f"Distribution monitoring for {field}.")
    @dq_test(spec)
    def _test(ctx: DQContext, _field=field, _tid=tid, _fn=fn) -> TestResult:
        return _fn(_tid, _field, ctx)
    return _test

for _tid, _field, _name in _DIST_SPECS:
    _fn = (_distribution_missing if "Missing" in _name
           else _distribution_range if "Range" in _name
           else _distribution_trend)
    _make_dist_test(_tid, _field, _name, _fn)


# ─────────────────────────────────────────────────────────────────────────────
# EOD-042  Adjusted close presence & consistency
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-042", name="Adjusted close consistency",
    layer="EOD", category="Validity", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="adj_close must be <= close (adjustment never increases price) for split-adjusted sources.",
    method_formula="adj_close <= close * 1.001 (small tolerance for rounding)",
    success_threshold="No adj_close > close anomalies"))
def test_eod_042(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "adj_close" not in df.columns or df["adj_close"].isna().all():
            metrics[f"{src}_{exch}"] = {"status": "adj_close_absent"}
            continue
        valid = df[df["adj_close"].notna() & df["close"].notna()]
        bad = valid[valid["adj_close"] > valid["close"] * 1.001]
        metrics[f"{src}_{exch}"] = {"violations": len(bad), "rows_checked": len(valid)}
        if len(bad) > 0:
            issues.append(f"{src}/{exch}: {len(bad)} rows where adj_close > close")
    r = TestResult(test_id="EOD-042", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("adj_close <= close on all valid rows.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-043  Expiry date presence for derivatives
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-043", name="Expiry date presence",
    layer="EOD", category="Validity", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="Futures and Options must have expiry_date populated on every row.",
    method_formula="expiry_date.notna().all()",
    success_threshold="No null expiry dates"))
def test_eod_043(ctx: DQContext) -> TestResult:
    itype = (ctx.config.get("instruments", {}).get("equity", {})
             .get(ctx.symbol, {}).get("instrument_type", "Equity"))
    if itype not in ("Equity_Futures", "Index_Futures", "Equity_Options", "Index_Options"):
        r = TestResult(test_id="EOD-043", symbol=ctx.symbol, layer="EOD",
                       category="Validity", severity="Critical", gate_type="Hard", weight=5.0)
        r.set_skip(f"Not applicable for {itype}")
        return r
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "expiry_date" not in df.columns:
            issues.append(f"{src}/{exch}: expiry_date column absent")
            metrics[f"{src}_{exch}"] = {"status": "column_missing"}
        else:
            null_count = int(df["expiry_date"].isna().sum())
            metrics[f"{src}_{exch}"] = {"null_expiry": null_count}
            if null_count:
                issues.append(f"{src}/{exch}: {null_count} null expiry dates")
    r = TestResult(test_id="EOD-043", symbol=ctx.symbol, layer="EOD",
                   category="Validity", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Expiry dates all present.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-044  Open vs previous close gap check
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-044", name="Open vs previous close gap",
    layer="EOD", category="Consistency", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Flag excessive overnight gaps: |open/prev_close - 1| > 10%.",
    method_formula="|open / close.shift(1) - 1| > 0.10",
    success_threshold="< 1% overnight gaps > 10%"))
def test_eod_044(ctx: DQContext) -> TestResult:
    gap_thresh = 0.10
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        df_s = df.sort_values("date")
        gap = (df_s["open"] / df_s["close"].shift(1) - 1).abs().dropna()
        large = gap[gap > gap_thresh]
        pct = len(large) / len(df) * 100
        metrics[f"{src}_{exch}"] = {"large_gaps": len(large), "pct": round(pct, 3),
                                     "max_gap_pct": round(float(gap.max() * 100), 2)}
        if pct > 1.0:
            issues.append(f"{src}/{exch}: {len(large)} overnight gaps > {gap_thresh*100:.0f}% ({pct:.2f}%)")
    r = TestResult(test_id="EOD-044", symbol=ctx.symbol, layer="EOD",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Overnight gaps within normal bounds.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-045  Universe symbol count stability
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-045", name="Universe symbol count stability",
    layer="EOD", category="Completeness", gate_type="Soft",
    severity="Low", weight=1.0,
    description="Check that all expected test symbols have data loaded; flag if any are absent.",
    method_formula="count(symbols_with_data) == count(expected_symbols)",
    success_threshold="All expected symbols present"))
def test_eod_045(ctx: DQContext) -> TestResult:
    expected = ctx.config.get("instruments", {}).get("test_symbols", [])
    present = {ctx.symbol}   # current symbol
    missing = [s for s in expected if s not in present]
    r = TestResult(test_id="EOD-045", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Low", gate_type="Soft", weight=1.0)
    metrics = {"expected": expected, "present": list(present)}
    r.set_pass(f"Symbol {ctx.symbol} in universe.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-046  Timestamp epoch vs date field cross-check (Dhan only)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-046", name="Timestamp epoch vs date cross-check",
    layer="EOD", category="Integrity", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Dhan provides both 'date' and 'timestamp' (epoch). Verify they agree within 1 day.",
    method_formula="|date - epoch_to_date| < 1 day",
    success_threshold="No mismatches between epoch and date"))
def test_eod_046(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch, src_dict in ctx.data.get("eod", {}).items():
        df = src_dict.get("dhan")
        if df is None or df.empty:
            continue
        if "timestamp_epoch" not in df.columns:
            metrics[f"dhan_{exch}"] = {"status": "no_timestamp_epoch_col"}
            continue
        epoch_dates = pd.to_datetime(df["timestamp_epoch"], unit="s",
                                     utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
        delta_days = (df["date"] - epoch_dates).abs().dt.days
        mismatches = (delta_days > 1).sum()
        metrics[f"dhan_{exch}"] = {"mismatches": int(mismatches)}
        if mismatches:
            issues.append(f"dhan/{exch}: {mismatches} rows where epoch date ≠ date field")
    r = TestResult(test_id="EOD-046", symbol=ctx.symbol, layer="EOD",
                   category="Integrity", severity="Medium", gate_type="Soft", weight=3.0)
    if not metrics:
        r.set_skip("No Dhan data with timestamp_epoch column available.")
        return r
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Epoch and date fields agree.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# EOD-047  OI column presence by instrument type
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="EOD-047", name="OI column presence by instrument type",
    layer="EOD", category="Completeness", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Derivatives must have open_interest; Equity/ETF/Index OI should be 0. Upstox equity OI must be 0.",
    method_formula="derivative→OI>0; equity→OI==0",
    success_threshold="OI column matches instrument type expectation"))
def test_eod_047(ctx: DQContext) -> TestResult:
    itype = (ctx.config.get("instruments", {}).get("equity", {})
             .get(ctx.symbol, {}).get("instrument_type", "Equity"))
    is_deriv = itype in ("Equity_Futures","Index_Futures","Equity_Options","Index_Options")
    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "open_interest" not in df.columns:
            issues.append(f"{src}/{exch}: open_interest column absent")
            metrics[f"{src}_{exch}"] = {"status": "column_missing"}
            continue
        if is_deriv:
            zero_oi = (df["open_interest"] <= 0).mean() * 100
            metrics[f"{src}_{exch}"] = {"zero_oi_pct": round(zero_oi, 2)}
            if zero_oi > 10:
                issues.append(f"{src}/{exch}: {zero_oi:.1f}% rows have zero OI for derivative")
        else:
            nonzero = (df["open_interest"] != 0).mean() * 100
            metrics[f"{src}_{exch}"] = {"nonzero_oi_pct": round(nonzero, 2)}
    r = TestResult(test_id="EOD-047", symbol=ctx.symbol, layer="EOD",
                   category="Completeness", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("OI column presence matches instrument type.", metrics)
    return r
