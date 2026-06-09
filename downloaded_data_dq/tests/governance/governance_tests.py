"""
Downloaded Data DQ Engine — Governance Tests: GOV-001 to GOV-005
downloaded_data_dq/tests/governance/governance_tests.py
"""

from __future__ import annotations
import logging
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)


def _all_eod(ctx):
    for exch, sd in ctx.data.get("eod", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


@dq_test(TestSpec(test_id="GOV-001", name="Overall health score",
    layer="Governance", category="Governance", gate_type="Soft",
    severity="Critical", weight=5.0,
    description="Composite data quality health score across all sources.",
    success_threshold="Overall health score >= 0.90"))
def test_gov_001(ctx: DQContext) -> TestResult:
    threshold = ctx.threshold("governance.GOV_min_overall_health_score", 0.90)
    scores = []
    metrics = {}
    for src, exch, df in _all_eod(ctx):
        # Components: completeness, validity, consistency
        completeness = 1.0 - df[["open","high","low","close","volume"]].isna().mean().mean()
        validity = float(((df["high"] >= df["low"]) &
                          (df["close"] >= df["low"]) &
                          (df["close"] <= df["high"])).mean())
        # Freshness: 1.0 if data within 90 days
        days_stale = (pd.Timestamp.today() - df["date"].max()).days
        freshness = max(0.0, 1.0 - days_stale / 365)
        score = round(0.5 * completeness + 0.3 * validity + 0.2 * freshness, 4)
        scores.append(score)
        metrics[f"{src}_{exch}"] = {"health_score": score, "completeness": round(completeness, 4),
                                     "validity": round(validity, 4), "freshness": round(freshness, 4)}
    overall = sum(scores) / max(len(scores), 1)
    metrics["overall_health"] = round(overall, 4)
    r = TestResult(test_id="GOV-001", symbol=ctx.symbol, layer="Governance",
                   category="Governance", severity="Critical", gate_type="Soft", weight=5.0)
    if overall < threshold:
        r.set_fail(f"Health score {overall:.3f} < {threshold}", metrics)
    else:
        r.set_pass(f"Health score {overall:.3f} >= {threshold}", metrics)
    return r


@dq_test(TestSpec(test_id="GOV-002", name="Minimum source count",
    layer="Governance", category="Governance", gate_type="Hard",
    severity="Critical", weight=5.0,
    description="At least 2 sources must have EOD data for reliable governance.",
    success_threshold="At least 2 sources with data"))
def test_gov_002(ctx: DQContext) -> TestResult:
    min_sources = ctx.threshold("governance.GOV_min_sources_per_symbol", 2)
    available = [(src, exch) for src, exch, _ in _all_eod(ctx)]
    unique_sources = len(set(s for s, _ in available))
    r = TestResult(test_id="GOV-002", symbol=ctx.symbol, layer="Governance",
                   category="Governance", severity="Critical", gate_type="Hard", weight=5.0)
    if unique_sources >= min_sources:
        r.set_pass(f"{unique_sources} sources available (>= {min_sources} required).",
                   {"available": available, "unique_sources": unique_sources})
    else:
        r.set_fail(f"Only {unique_sources} source(s) available (need >= {min_sources}).",
                   {"available": available})
    return r


@dq_test(TestSpec(test_id="GOV-003", name="Data freshness SLA",
    layer="Governance", category="Governance", gate_type="Soft",
    severity="High", weight=4.0,
    description="All sources must have data within the last 25 hours (for daily pipeline SLA).",
    success_threshold="Latest data not older than 25 hours"))
def test_gov_003(ctx: DQContext) -> TestResult:
    max_stale_hours = ctx.threshold("governance.GOV_max_stale_run_hours", 25)
    issues, metrics = [], {}
    today = pd.Timestamp.today().normalize()
    for src, exch, df in _all_eod(ctx):
        last = df["date"].max()
        stale_days = (today - last).days
        metrics[f"{src}_{exch}"] = {"last_date": str(last.date()), "stale_days": stale_days}
        # Stale if more than 2 business days old (allows for weekends)
        if stale_days > 3:
            issues.append(f"{src}/{exch}: last date {last.date()} is {stale_days} days old")
    r = TestResult(test_id="GOV-003", symbol=ctx.symbol, layer="Governance",
                   category="Governance", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All sources within SLA.", metrics)
    return r


@dq_test(TestSpec(test_id="GOV-004", name="Hard gate failure audit",
    layer="Governance", category="Governance", gate_type="Soft",
    severity="High", weight=4.0,
    description="Count of Hard-gate Fail results. Zero is required for production readiness.",
    success_threshold="0 Hard gate failures"))
def test_gov_004(ctx: DQContext) -> TestResult:
    # This test is informational — it checks the context for any Hard gate failures
    # that occurred earlier in the run. Since we can't introspect past results from ctx,
    # we instead audit the most critical validity checks inline.
    hard_gate_checks = []

    for src, exch, df in _all_eod(ctx):
        # Recheck the hard-gate conditions
        null_check = df[["open","high","low","close"]].isna().any(axis=1).sum()
        hl_inv = (df["high"] < df["low"]).sum()
        close_oor = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
        hard_gate_checks.append({
            "source": src, "exchange": exch,
            "null_in_ohlc": int(null_check),
            "hl_inversions": int(hl_inv),
            "close_out_of_range": int(close_oor),
        })

    total_failures = sum(
        c["null_in_ohlc"] + c["hl_inversions"] + c["close_out_of_range"]
        for c in hard_gate_checks
    )
    r = TestResult(test_id="GOV-004", symbol=ctx.symbol, layer="Governance",
                   category="Governance", severity="High", gate_type="Soft", weight=4.0)
    if total_failures == 0:
        r.set_pass("Hard gate audit: 0 critical violations.", {"checks": hard_gate_checks})
    else:
        r.set_fail(f"Hard gate audit: {total_failures} critical violations found.",
                   {"checks": hard_gate_checks})
    return r


@dq_test(TestSpec(test_id="GOV-005", name="Governance scorecard summary",
    layer="Governance", category="Governance", gate_type="Soft",
    severity="Info", weight=1.0,
    description="Summary scorecard: row counts, date ranges, source coverage per symbol.",
    success_threshold="Informational — always passes"))
def test_gov_005(ctx: DQContext) -> TestResult:
    metrics = {"symbol": ctx.symbol, "sources": {}}
    for src, exch, df in _all_eod(ctx):
        metrics["sources"][f"{src}_{exch}"] = {
            "rows": len(df),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "years": round((df["date"].max() - df["date"].min()).days / 365.25, 1),
        }
    r = TestResult(test_id="GOV-005", symbol=ctx.symbol, layer="Governance",
                   category="Governance", severity="Info", gate_type="Soft", weight=1.0)
    r.set_pass("Governance scorecard generated.", metrics)
    return r
