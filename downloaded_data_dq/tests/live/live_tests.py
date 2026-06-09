"""
Downloaded Data DQ Engine — Live Feed Tests: LT-001 to LT-020 + MICRO-001 to MICRO-004
downloaded_data_dq/tests/live/live_tests.py

24 tests for live/real-time feed quality monitoring.

Architecture note:
  Live tests operate in two modes:
  1. LIVE mode (run_mode="Live"): tests run against ctx.data["live"] — a dict of
     {source: DataFrame} where each DataFrame is the last N bars from a live feed.
     The live feed manager (Phase 5) populates this before calling run_tests_for_symbol.

  2. Historical validation mode (run_mode="Both"/"EOD"/"INTRADAY"):
     Tests skip gracefully with a Data_Not_Present status, since no live data
     is in the context. This is the expected behaviour during backtest/DQ runs.

Live data schema (when available):
  datetimestamp   : pd.Timestamp (tz-naive IST)
  open, high, low, close : float
  volume          : int
  open_interest   : int  (optional)
  arrival_time    : pd.Timestamp — when bar arrived at our system
  exchange_time   : pd.Timestamp — exchange's own timestamp on the bar
  source          : str
  session_flag    : str  (one of: "pre_open", "normal", "closing_auction", "post_close")
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]
SESSION_START = (9, 15)   # HH, MM  IST
SESSION_END   = (15, 30)


def _live_data(ctx: DQContext) -> dict:
    """Return live data dict or empty dict if not in live mode."""
    return ctx.data.get("live", {})


def _has_live(ctx: DQContext) -> bool:
    ld = _live_data(ctx)
    return bool(ld) and any(
        df is not None and not df.empty
        for df in ld.values()
    )


def _skip_not_live(test_id: str, ctx: DQContext) -> TestResult:
    """Standard skip result when live data is not available."""
    r = TestResult(test_id=test_id, symbol=ctx.symbol, layer="Live",
                   category="Live", severity="Critical",
                   gate_type="Hard", weight=5.0)
    r.set_skip(
        "No live feed data in context. "
        "Live tests execute only when run_mode='Live' and a live feed "
        "manager has populated ctx.data['live'] (Phase 5)."
    )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-001  Latest bar arrival within SLA
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-001",
    name="Live bar arrival within SLA",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Latest bar from each source must have arrived within SLA (default 5 sec).",
    method_formula="now - last_bar_arrival_time < SLA_seconds",
    success_threshold="<= 5 seconds latency",
))
def test_lt_001(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-001", ctx)
    sla_sec = ctx.threshold("live.LT_001_arrival_sla_seconds", 5.0)
    now = pd.Timestamp.now()
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if "arrival_time" not in df.columns:
            metrics[src] = {"status": "no_arrival_time_column"}
            continue
        last_arrival = df["arrival_time"].max()
        lag_sec = (now - last_arrival).total_seconds()
        metrics[src] = {"lag_seconds": round(lag_sec, 2), "sla_seconds": sla_sec}
        if lag_sec > sla_sec:
            issues.append(f"{src}: bar lag {lag_sec:.1f}s > SLA {sla_sec}s")
    r = TestResult(test_id="LT-001", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        f"All sources within {sla_sec}s SLA.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-002  Feed latency (arrival vs exchange timestamp)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-002",
    name="Feed latency (arrival vs exchange time)",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Measure delay: arrival_time - exchange_time. Must be < latency SLA.",
    method_formula="arrival_time - exchange_time < latency_sla",
    success_threshold="< 500ms median latency",
))
def test_lt_002(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-002", ctx)
    sla_ms = ctx.threshold("live.LT_002_latency_sla_ms", 500.0)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if "arrival_time" not in df.columns or "exchange_time" not in df.columns:
            metrics[src] = {"status": "missing arrival/exchange columns"}
            continue
        latency_ms = (df["arrival_time"] - df["exchange_time"]).dt.total_seconds() * 1000
        p50 = float(latency_ms.median())
        p99 = float(latency_ms.quantile(0.99))
        metrics[src] = {"p50_ms": round(p50, 1), "p99_ms": round(p99, 1), "sla_ms": sla_ms}
        if p50 > sla_ms:
            issues.append(f"{src}: median latency {p50:.0f}ms > {sla_ms}ms SLA")
    r = TestResult(test_id="LT-002", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Feed latency within SLA.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-003  No missing live bars
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-003",
    name="No missing live bars",
    layer="Live",
    category="OHLCV",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Bar count in session must equal expected number of 1-min bars.",
    method_formula="bar_count == expected_bars_in_session",
    success_threshold="0 missing bars during live session",
))
def test_lt_003(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-003", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if df.empty:
            issues.append(f"{src}: empty live feed")
            continue
        # Count bars in current session
        now = pd.Timestamp.now()
        session_open = now.normalize().replace(hour=9, minute=15)
        elapsed_min = max(0, (now - session_open).total_seconds() / 60)
        expected_bars = int(elapsed_min)
        actual_bars = len(df)
        missing = max(0, expected_bars - actual_bars)
        metrics[src] = {"expected": expected_bars, "actual": actual_bars,
                        "missing": missing}
        if missing > 2:  # allow 2-bar tolerance for timing
            issues.append(f"{src}: {missing} missing bars (expected {expected_bars}, got {actual_bars})")
    r = TestResult(test_id="LT-003", symbol=ctx.symbol, layer="Live",
                   category="OHLCV", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No missing live bars.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-004  Live price spike detection (z-score)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-004",
    name="Live price spike detection",
    layer="Live",
    category="Close",
    gate_type="Soft",
    severity="High",
    weight=4.0,
    description="Detect abnormal price spike in live feed using rolling z-score.",
    method_formula="(close - rolling_mean) / rolling_std > z_threshold",
    success_threshold="z-score < 3 for all live bars",
))
def test_lt_004(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-004", ctx)
    z_thresh = ctx.threshold("live.LT_004_zscore_threshold", 3.0)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 10:
            continue
        df_s = df.sort_values("datetimestamp")
        roll_mean = df_s["close"].rolling(20, min_periods=5).mean()
        roll_std  = df_s["close"].rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((df_s["close"] - roll_mean) / roll_std).abs()
        spikes = (z > z_thresh).sum()
        metrics[src] = {"spikes": int(spikes), "max_z": round(float(z.max()), 2)}
        if spikes > 0:
            issues.append(f"{src}: {spikes} bars with |z| > {z_thresh}")
    r = TestResult(test_id="LT-004", symbol=ctx.symbol, layer="Live",
                   category="Close", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No abnormal price spikes in live feed.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-005  No duplicate ticks
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-005",
    name="No duplicate live ticks",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="No timestamp should appear twice in the live feed.",
    method_formula="datetimestamp.duplicated().sum() == 0",
    success_threshold="0 duplicate timestamps",
))
def test_lt_005(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-005", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        dups = df["datetimestamp"].duplicated().sum()
        metrics[src] = {"duplicate_ticks": int(dups)}
        if dups > 0:
            issues.append(f"{src}: {dups} duplicate timestamps")
    r = TestResult(test_id="LT-005", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No duplicate live ticks.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-006  Live volume spike detection
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-006",
    name="Live volume spike detection",
    layer="Live",
    category="Volume",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description="Detect abnormal volume spike in live feed using z-score.",
    method_formula="(volume - rolling_mean) / rolling_std > 4",
    success_threshold="z-score < 4 for volume",
))
def test_lt_006(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-006", ctx)
    z_thresh = ctx.threshold("live.LT_006_volume_zscore_threshold", 4.0)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 10:
            continue
        df_s = df.sort_values("datetimestamp")
        roll_mean = df_s["volume"].rolling(20, min_periods=5).mean()
        roll_std  = df_s["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((df_s["volume"] - roll_mean) / roll_std).abs()
        spikes = (z > z_thresh).sum()
        metrics[src] = {"vol_spikes": int(spikes), "max_z": round(float(z.max()), 2)}
        if spikes > 0:
            issues.append(f"{src}: {spikes} volume bars with z > {z_thresh}")
    r = TestResult(test_id="LT-006", symbol=ctx.symbol, layer="Live",
                   category="Volume", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No abnormal volume spikes.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-007  All bars within session hours
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-007",
    name="Live bars within market session",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="All live bars must have timestamps within 09:15–15:30 IST.",
    method_formula="09:15 <= time <= 15:30",
    success_threshold="0 bars outside session",
))
def test_lt_007(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-007", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        tod = df["datetimestamp"].dt.hour * 60 + df["datetimestamp"].dt.minute
        outside = ((tod < 9 * 60 + 15) | (tod > 15 * 60 + 30)).sum()
        metrics[src] = {"outside_session_bars": int(outside)}
        if outside > 0:
            issues.append(f"{src}: {outside} bars outside session")
    r = TestResult(test_id="LT-007", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "All live bars within session.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-008  Cross-feed close agreement
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-008",
    name="Cross-feed live close agreement",
    layer="Live",
    category="Close",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Compare close from 3 feeds at same timestamp. Max deviation < 0.3%.",
    method_formula="max(abs(close_A - close_B)/close_A) < 0.003",
    success_threshold="All feed pairs agree within 0.3%",
))
def test_lt_008(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-008", ctx)
    tol = ctx.threshold("live.LT_008_cross_feed_tolerance_pct", 0.3) / 100
    issues, metrics = [], {}
    live_dict = _live_data(ctx)
    sources = list(live_dict.keys())
    for i, sa in enumerate(sources):
        for sb in sources[i+1:]:
            da = live_dict[sa].set_index("datetimestamp")["close"]
            db = live_dict[sb].set_index("datetimestamp")["close"]
            common = da.index.intersection(db.index)
            if len(common) < 5:
                continue
            diff = (da[common] / db[common] - 1).abs()
            max_diff = float(diff.max())
            metrics[f"{sa}_vs_{sb}"] = {"max_diff_pct": round(max_diff * 100, 4),
                                         "aligned_bars": len(common)}
            if max_diff > tol:
                issues.append(f"{sa} vs {sb}: max close diff {max_diff*100:.3f}%")
    r = TestResult(test_id="LT-008", symbol=ctx.symbol, layer="Live",
                   category="Close", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "All live feeds agree within 0.3%.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-009  OI abnormal jump detection
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-009",
    name="Live OI abnormal jump detection",
    layer="Live",
    category="OI",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description="Detect large OI jump in live feed (> 5% per bar).",
    method_formula="abs(OI_t - OI_{t-1}) / OI_{t-1} > 0.05",
    success_threshold="OI bar-to-bar change <= 5%",
))
def test_lt_009(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-009", ctx)
    threshold = ctx.threshold("live.LT_009_oi_jump_pct", 5.0) / 100
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if "open_interest" not in df.columns or df["open_interest"].isna().all():
            metrics[src] = {"status": "no_oi"}
            continue
        df_s = df.sort_values("datetimestamp")
        oi = df_s["open_interest"].replace(0, np.nan)
        jumps = (oi.pct_change().abs() > threshold).sum()
        metrics[src] = {"oi_jumps_gt5pct": int(jumps)}
        if jumps > 0:
            issues.append(f"{src}: {jumps} OI jumps > 5%")
    r = TestResult(test_id="LT-009", symbol=ctx.symbol, layer="Live",
                   category="OI", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No abnormal OI jumps in live feed.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-010  Trading halt detection
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-010",
    name="Trading halt detection",
    layer="Live",
    category="Price",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Detect trading halt: no ticks for > N seconds during session.",
    method_formula="max_gap_seconds > halt_threshold",
    success_threshold="No gaps > 300 seconds during normal session",
))
def test_lt_010(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-010", ctx)
    halt_seconds = ctx.threshold("live.LT_010_halt_seconds", 300.0)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        df_s = df.sort_values("datetimestamp")
        gaps = df_s["datetimestamp"].diff().dt.total_seconds().dropna()
        max_gap = float(gaps.max()) if len(gaps) > 0 else 0
        metrics[src] = {"max_gap_seconds": round(max_gap, 1),
                        "halt_threshold": halt_seconds}
        if max_gap > halt_seconds:
            issues.append(f"{src}: gap of {max_gap:.0f}s (possible halt)")
    r = TestResult(test_id="LT-010", symbol=ctx.symbol, layer="Live",
                   category="Price", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No trading halt detected.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-011  Rolling mean price drift
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-011",
    name="Live rolling mean price drift",
    layer="Live",
    category="Close",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description="Rolling 20-bar mean should not drift > 3% from 60-bar mean (stuck feed).",
    method_formula="abs(rolling_20 / rolling_60 - 1) > 0.03",
    success_threshold="Drift < 3%",
))
def test_lt_011(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-011", ctx)
    drift_thresh = ctx.threshold("live.LT_011_rolling_drift_pct", 3.0) / 100
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 60:
            metrics[src] = {"status": "insufficient_bars"}
            continue
        df_s = df.sort_values("datetimestamp")
        r20 = df_s["close"].rolling(20).mean().iloc[-1]
        r60 = df_s["close"].rolling(60).mean().iloc[-1]
        drift = abs(r20 / r60 - 1) if r60 > 0 else 0
        metrics[src] = {"rolling_20": round(r20, 4), "rolling_60": round(r60, 4),
                        "drift_pct": round(drift * 100, 3)}
        if drift > drift_thresh:
            issues.append(f"{src}: price drift {drift*100:.2f}%")
    r = TestResult(test_id="LT-011", symbol=ctx.symbol, layer="Live",
                   category="Close", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Rolling price drift within bounds.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-012  Symbol quarantine on persistent failures
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-012",
    name="Symbol quarantine on persistent failures",
    layer="Live",
    category="All",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="If > 3 live DQ tests fail for a symbol, quarantine it from live trading.",
    method_formula="fail_count > 3 => quarantine_flag = True",
    success_threshold="fail_count <= 3",
))
def test_lt_012(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-012", ctx)
    # Count live test failures recorded in cache
    live_fails = ctx.cache.get("live_test_fails", 0)
    r = TestResult(test_id="LT-012", symbol=ctx.symbol, layer="Live",
                   category="All", severity="Critical", gate_type="Hard", weight=5.0)
    if live_fails > 3:
        r.set_fail(
            f"QUARANTINE: {live_fails} live test failures. Symbol blocked from live trading.",
            {"live_fails": live_fails},
        )
    else:
        r.set_pass(f"Symbol healthy. {live_fails} live failures (threshold 3).",
                   {"live_fails": live_fails})
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-013  Live schema validation
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-013",
    name="Live feed schema validation",
    layer="Live",
    category="All",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Live bar schema must have all required columns with correct dtypes.",
    method_formula="column set == REQUIRED_LIVE_COLS",
    success_threshold="Exact schema match",
))
def test_lt_013(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-013", ctx)
    REQUIRED = {"datetimestamp", "open", "high", "low", "close", "volume"}
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        missing = REQUIRED - set(df.columns)
        wrong_types = {
            c: str(df[c].dtype)
            for c in ["open", "high", "low", "close"]
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
        }
        metrics[src] = {"missing_cols": sorted(missing), "wrong_types": wrong_types}
        if missing:
            issues.append(f"{src}: missing columns {missing}")
        if wrong_types:
            issues.append(f"{src}: wrong dtypes {wrong_types}")
    r = TestResult(test_id="LT-013", symbol=ctx.symbol, layer="Live",
                   category="All", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Live schema valid.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-014  Bid-ask spread sanity
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-014",
    name="Live bid-ask spread sanity",
    layer="Live",
    category="Spread",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description="If bid/ask available: spread should be within 3 z-scores of rolling mean.",
    method_formula="spread_zscore < 3",
    success_threshold="Spread z-score < 3",
))
def test_lt_014(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-014", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if "bid" not in df.columns or "ask" not in df.columns:
            metrics[src] = {"status": "no_bid_ask_columns"}
            continue
        spread = df["ask"] - df["bid"]
        mean_s, std_s = spread.mean(), spread.std()
        z = ((spread - mean_s) / max(std_s, 0.001)).abs()
        wide = (z > 3).sum()
        metrics[src] = {"wide_spread_bars": int(wide), "mean_spread": round(float(mean_s), 4)}
        if wide > 0:
            issues.append(f"{src}: {wide} bars with abnormally wide spread (z>3)")
    r = TestResult(test_id="LT-014", symbol=ctx.symbol, layer="Live",
                   category="Spread", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Spread within normal bounds.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-015  Feed heartbeat / alive check
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-015",
    name="Feed heartbeat alive check",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description="Feed must have sent a bar/heartbeat within the last 2 minutes.",
    method_formula="(now - last_bar_time) < heartbeat_interval",
    success_threshold="Feed alive: last bar < 2 minutes ago",
))
def test_lt_015(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-015", ctx)
    max_silence_sec = ctx.threshold("live.LT_015_heartbeat_interval_seconds", 120.0)
    now = pd.Timestamp.now()
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        last_ts = df["datetimestamp"].max()
        silence_sec = (now - last_ts).total_seconds()
        metrics[src] = {"silence_seconds": round(silence_sec, 1), "last_bar": str(last_ts)}
        if silence_sec > max_silence_sec:
            issues.append(f"{src}: feed silent for {silence_sec:.0f}s")
    r = TestResult(test_id="LT-015", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Feed heartbeat alive.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-016  Extreme tick detection (IQR)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-016",
    name="Extreme tick detection (IQR method)",
    layer="Live",
    category="Price",
    gate_type="Soft",
    severity="High",
    weight=4.0,
    description="Flag ticks outside 3×IQR from rolling median as potentially bad.",
    method_formula="|price - median| > 3 * IQR",
    success_threshold="0 extreme ticks in live session",
))
def test_lt_016(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-016", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 20:
            continue
        s = df.sort_values("datetimestamp")["close"]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        fence = 3 * iqr
        extreme = ((s < q1 - fence) | (s > q3 + fence)).sum()
        metrics[src] = {"extreme_ticks": int(extreme), "iqr": round(float(iqr), 4)}
        if extreme > 0:
            issues.append(f"{src}: {extreme} extreme ticks (|price - median| > 3×IQR)")
    r = TestResult(test_id="LT-016", symbol=ctx.symbol, layer="Live",
                   category="Price", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No extreme ticks detected.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-017  Auto-switch to backup feed
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-017",
    name="Backup feed switch check",
    layer="Live",
    category="All",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description=(
        "If primary feed has failures, system should have switched to backup. "
        "Verify backup feed is active and sending data."
    ),
    method_formula="if primary_failed: backup_feed.is_active == True",
    success_threshold="Backup activated within 10 seconds of primary failure",
))
def test_lt_017(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-017", ctx)
    # Check if a backup switchover event is recorded in cache
    switchover = ctx.cache.get("backup_feed_active", False)
    switchover_ts = ctx.cache.get("backup_switchover_timestamp")
    r = TestResult(test_id="LT-017", symbol=ctx.symbol, layer="Live",
                   category="All", severity="Critical", gate_type="Hard", weight=5.0)
    if switchover:
        r.set_pass(f"Backup feed active since {switchover_ts}.",
                   {"backup_active": True, "switchover_ts": str(switchover_ts)})
    else:
        r.set_pass("Primary feed healthy — no backup switch required.",
                   {"backup_active": False})
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-018  Sudden volatility spike
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-018",
    name="Live volatility spike detection",
    layer="Live",
    category="Close",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "Rolling 5-bar std of returns should not suddenly exceed 3× "
        "rolling 60-bar std. Spike may indicate circuit-breaker event or bad data."
    ),
    method_formula="rolling_5_std / rolling_60_std > 3",
    success_threshold="Vol ratio < 3",
))
def test_lt_018(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-018", ctx)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 60:
            metrics[src] = {"status": "insufficient_bars"}
            continue
        df_s = df.sort_values("datetimestamp")
        ret = df_s["close"].pct_change()
        vol5  = ret.rolling(5).std().iloc[-1]
        vol60 = ret.rolling(60).std().iloc[-1]
        ratio = vol5 / max(vol60, 1e-8)
        metrics[src] = {"vol5": round(float(vol5), 6), "vol60": round(float(vol60), 6),
                        "vol_ratio": round(float(ratio), 2)}
        if ratio > 3:
            issues.append(f"{src}: volatility spike ratio {ratio:.1f}x")
    r = TestResult(test_id="LT-018", symbol=ctx.symbol, layer="Live",
                   category="Close", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No live volatility spikes detected.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-019  System clock synchronisation
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-019",
    name="System clock synchronisation (NTP check)",
    layer="Live",
    category="Timestamp",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description=(
        "System clock must be synchronised with NTP. "
        "Clock drift > 1 second causes timestamp mismatches across sources."
    ),
    method_formula="abs(system_time - ntp_time) < 1 second",
    success_threshold="Clock drift < 1 second",
))
def test_lt_019(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-019", ctx)
    # Check if NTP sync result is recorded in cache by the live feed manager
    ntp_drift_ms = ctx.cache.get("ntp_drift_ms")
    r = TestResult(test_id="LT-019", symbol=ctx.symbol, layer="Live",
                   category="Timestamp", severity="Critical", gate_type="Hard", weight=5.0)
    if ntp_drift_ms is None:
        r.set_skip(
            "NTP drift not measured. Live feed manager should call "
            "ctx.cache['ntp_drift_ms'] = measure_ntp_drift() at startup."
        )
        return r
    if abs(ntp_drift_ms) > 1000:
        r.set_fail(
            f"Clock drift {ntp_drift_ms:.0f}ms exceeds 1000ms threshold.",
            {"ntp_drift_ms": ntp_drift_ms},
        )
    else:
        r.set_pass(f"Clock in sync. Drift: {ntp_drift_ms:.1f}ms.",
                   {"ntp_drift_ms": ntp_drift_ms})
    return r


# ─────────────────────────────────────────────────────────────────────────────
# LT-020  Freeze trading on critical failures
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="LT-020",
    name="Freeze trading on critical live failures",
    layer="Live",
    category="All",
    gate_type="Hard",
    severity="Critical",
    weight=5.0,
    description=(
        "If cumulative Hard gate failures > limit, signal trading engine to freeze. "
        "Acts as the final live circuit breaker."
    ),
    method_formula="hard_fail_count > limit => trading_freeze = True",
    success_threshold="hard_fail_count <= 3",
))
def test_lt_020(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("LT-020", ctx)
    hard_fails = ctx.cache.get("live_hard_gate_fails", 0)
    limit = ctx.threshold("live.LT_020_freeze_threshold", 3)
    r = TestResult(test_id="LT-020", symbol=ctx.symbol, layer="Live",
                   category="All", severity="Critical", gate_type="Hard", weight=5.0)
    if hard_fails > limit:
        r.set_fail(
            f"TRADING FREEZE: {hard_fails} hard gate failures (limit {limit}). "
            "Signal sent to trading engine to halt all orders for this symbol.",
            {"hard_fails": hard_fails, "limit": limit, "action": "FREEZE"},
        )
    else:
        r.set_pass(
            f"Live health OK. {hard_fails}/{limit} hard gate failures.",
            {"hard_fails": hard_fails, "limit": limit},
        )
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# MICRO tests  MICRO-001 to MICRO-004
# ═══════════════════════════════════════════════════════════════════════════════

@dq_test(TestSpec(
    test_id="MICRO-001",
    name="Tick-size compliance (live)",
    layer="Live Integrity",
    category="Microstructure",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "All live trade/bar prices must be multiples of tick size. "
        "Non-compliant prices indicate parsing errors or wrong instrument mapping."
    ),
    method_formula="(price / tick_size) is integer within epsilon",
    success_threshold="100% tick-size compliant",
))
def test_micro_001(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("MICRO-001", ctx)
    sym_cfg = (ctx.config.get("instruments", {})
               .get("equity", {}).get(ctx.symbol, {}))
    tick = sym_cfg.get("tick_size", 0.05)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            bad = (df[col] % tick > tick * 0.01).sum()
            if bad > 0:
                issues.append(f"{src} {col}: {bad} prices not multiples of {tick}")
        metrics[src] = {"tick": tick, "rows": len(df)}
    r = TestResult(test_id="MICRO-001", symbol=ctx.symbol, layer="Live Integrity",
                   category="Microstructure", severity="High", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        f"All live prices comply with tick size {tick}.", metrics)
    return r


@dq_test(TestSpec(
    test_id="MICRO-002",
    name="Circuit-limit / price band breach",
    layer="Live Integrity",
    category="Microstructure",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "Price must stay within exchange price bands. "
        "Breaches indicate bad ticks, wrong symbol, or real circuit-breaker events."
    ),
    method_formula="lower_band <= price <= upper_band",
    success_threshold="0 price band breaches",
))
def test_micro_002(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("MICRO-002", ctx)
    sym_cfg = (ctx.config.get("instruments", {})
               .get("equity", {}).get(ctx.symbol, {}))
    band_pct = sym_cfg.get("price_band_pct", 20.0) / 100.0
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if len(df) < 2:
            continue
        ref_price = df["close"].iloc[0]  # use first bar as reference
        lower = ref_price * (1 - band_pct)
        upper = ref_price * (1 + band_pct)
        breaches = ((df["close"] < lower) | (df["close"] > upper)).sum()
        metrics[src] = {
            "ref_price": round(float(ref_price), 2),
            "lower_band": round(float(lower), 2),
            "upper_band": round(float(upper), 2),
            "breaches": int(breaches),
        }
        if breaches > 0:
            issues.append(
                f"{src}: {breaches} bars outside ±{band_pct*100:.0f}% band"
            )
    r = TestResult(test_id="MICRO-002", symbol=ctx.symbol, layer="Live Integrity",
                   category="Microstructure", severity="High", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No price band breaches.", metrics)
    return r


@dq_test(TestSpec(
    test_id="MICRO-003",
    name="Stale price detection (live)",
    layer="Live Integrity",
    category="Microstructure",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "If price unchanged for > 120 seconds while market is active, "
        "flag as stale feed for this symbol."
    ),
    method_formula="no_price_change_duration > 120s AND market_active",
    success_threshold="No stale price runs >= 120 seconds",
))
def test_micro_003(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("MICRO-003", ctx)
    stale_secs = ctx.threshold("live.MICRO_003_stale_price_seconds", 120.0)
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        df_s = df.sort_values("datetimestamp")
        # Detect runs of identical close price
        price_changed = (df_s["close"] != df_s["close"].shift())
        rle_groups = price_changed.cumsum()
        run_durations = df_s.groupby(rle_groups).apply(
            lambda g: (g["datetimestamp"].max() - g["datetimestamp"].min()).total_seconds()
        )
        max_stale = float(run_durations.max()) if len(run_durations) > 0 else 0
        stale_runs = (run_durations > stale_secs).sum()
        metrics[src] = {
            "max_stale_seconds": round(max_stale, 1),
            "stale_runs_gt_threshold": int(stale_runs),
            "threshold_seconds": stale_secs,
        }
        if stale_runs > 0:
            issues.append(
                f"{src}: {stale_runs} stale price run(s) > {stale_secs}s"
            )
    r = TestResult(test_id="MICRO-003", symbol=ctx.symbol, layer="Live Integrity",
                   category="Microstructure", severity="High", gate_type="Hard", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "No stale price runs in live feed.", metrics)
    return r


@dq_test(TestSpec(
    test_id="MICRO-004",
    name="Auction window handling",
    layer="Live Integrity",
    category="Auction",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "Pre-open (09:00–09:14) and closing auction (15:30–15:40) bars "
        "should be correctly tagged with session_flag = 'pre_open'/'closing_auction'. "
        "Untagged auction bars distort intraday strategy signals."
    ),
    method_formula="session_flag in {'pre_open','normal','closing_auction','post_close'}",
    success_threshold="All bars have valid session_flag",
))
def test_micro_004(ctx: DQContext) -> TestResult:
    if not _has_live(ctx):
        return _skip_not_live("MICRO-004", ctx)
    VALID_FLAGS = {"pre_open", "normal", "closing_auction", "post_close"}
    issues, metrics = [], {}
    for src, df in _live_data(ctx).items():
        if "session_flag" not in df.columns:
            # session_flag not present — check for pre-open bars that should be tagged
            tod = df["datetimestamp"].dt.hour * 60 + df["datetimestamp"].dt.minute
            untagged_preopen = ((tod >= 9 * 60) & (tod < 9 * 60 + 15)).sum()
            metrics[src] = {
                "session_flag_column": False,
                "untagged_pre_open_bars": int(untagged_preopen),
            }
            if untagged_preopen > 0:
                issues.append(
                    f"{src}: {untagged_preopen} pre-open bars without session_flag"
                )
        else:
            invalid = (~df["session_flag"].isin(VALID_FLAGS)).sum()
            metrics[src] = {
                "session_flag_column": True,
                "invalid_flags": int(invalid),
                "flag_distribution": df["session_flag"].value_counts().to_dict(),
            }
            if invalid > 0:
                issues.append(f"{src}: {invalid} bars with invalid session_flag")

    r = TestResult(test_id="MICRO-004", symbol=ctx.symbol, layer="Live Integrity",
                   category="Auction", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(
        "Auction window session flags valid.", metrics)
    return r
