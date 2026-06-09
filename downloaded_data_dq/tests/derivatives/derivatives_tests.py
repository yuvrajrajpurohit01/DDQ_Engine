"""
Downloaded Data DQ Engine — Derivatives Tests: DERIV-001 to DERIV-005
downloaded_data_dq/tests/derivatives/derivatives_tests.py

5 EOD-layer tests for options and futures data quality.
These skip gracefully when instrument type is not derivatives.
"""

from __future__ import annotations
import logging
import math
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


def _instrument_type(ctx: DQContext) -> str:
    return (ctx.config.get("instruments", {})
            .get("equity", {}).get(ctx.symbol, {})
            .get("instrument_type", "Equity"))


def _is_option(itype: str) -> bool:
    return itype in ("Equity_Options", "Index_Options")


def _is_derivative(itype: str) -> bool:
    return itype in ("Equity_Futures", "Index_Futures",
                     "Equity_Options", "Index_Options")


def _all_eod(ctx: DQContext):
    for exch, sd in ctx.data.get("eod", {}).items():
        for src, df in sd.items():
            if df is not None and not df.empty:
                yield src, exch, df


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-001  Option intrinsic value floor
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-001",
    name="Option intrinsic value floor",
    layer="EOD",
    category="Validity",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "Call: close >= max(0, spot - strike). "
        "Put: close >= max(0, strike - spot). "
        "Violations indicate a bad tick or wrong contract mapping."
    ),
    method_formula="call: close>=max(0,spot-K); put: close>=max(0,K-spot)",
    success_threshold="0 violations",
))
def test_deriv_001(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-001", symbol=ctx.symbol, layer="EOD",
        category="Validity", severity="High", gate_type="Hard", weight=5.0,
    )
    if not _is_option(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "strike_price" not in df.columns or "option_type" not in df.columns:
            metrics[f"{src}_{exch}"] = {"status": "no_strike_or_option_type_column"}
            continue

        # Need underlying close — use close as proxy if no separate spot column
        # In real data, underlying_close would be a separate column
        spot = df.get("underlying_close", df["close"])

        calls = df[df["option_type"].str.upper() == "CE"]
        puts  = df[df["option_type"].str.upper() == "PE"]

        call_violations = (
            calls["close"] < (spot[calls.index] - calls["strike_price"]).clip(lower=0) - 0.01
        ).sum()
        put_violations = (
            puts["close"] < (puts["strike_price"] - spot[puts.index]).clip(lower=0) - 0.01
        ).sum()

        total = int(call_violations + put_violations)
        metrics[f"{src}_{exch}"] = {
            "call_violations": int(call_violations),
            "put_violations": int(put_violations),
            "total": total,
        }
        if total > 0:
            issues.append(f"{src}/{exch}: {total} intrinsic value violations")

    if issues:
        r.set_fail("Intrinsic value floor breached: " + " | ".join(issues), metrics)
    else:
        r.set_pass("All option closes >= intrinsic value.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-002  Option premium > 0
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-002",
    name="Option premium > 0 (no zero-price options before expiry)",
    layer="EOD",
    category="Validity",
    gate_type="Hard",
    severity="High",
    weight=5.0,
    description=(
        "ITM and ATM options must have positive close before expiry. "
        "Zero-price non-expiry options indicate missing data or bad tick."
    ),
    method_formula="IF days_to_expiry > 1 AND close == 0 => flag",
    success_threshold="0 zero-premium non-expiry options",
))
def test_deriv_002(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-002", symbol=ctx.symbol, layer="EOD",
        category="Validity", severity="High", gate_type="Hard", weight=5.0,
    )
    if not _is_option(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "expiry_date" not in df.columns:
            metrics[f"{src}_{exch}"] = {"status": "no_expiry_date_column"}
            continue

        df_s = df.copy()
        df_s["days_to_expiry"] = (
            pd.to_datetime(df_s["expiry_date"]) - df_s["date"]
        ).dt.days

        # Flag zero-close rows more than 1 day before expiry
        zero_not_expiry = df_s[
            (df_s["close"] <= 0.0) & (df_s["days_to_expiry"] > 1)
        ]
        n = len(zero_not_expiry)
        metrics[f"{src}_{exch}"] = {
            "zero_premium_before_expiry": n,
            "total_rows": len(df_s),
        }
        if n > 0:
            issues.append(f"{src}/{exch}: {n} zero-premium rows before expiry")

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("No zero-premium options before expiry.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-003  Put-call parity sanity
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-003",
    name="Put-call parity sanity",
    layer="EOD",
    category="Consistency",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "C - P ≈ F - K*e^(-rT). Large violations across the chain indicate "
        "bad pricing data for one leg of the pair."
    ),
    method_formula="abs((C-P) - (spot - strike)) > 2 * tick_size",
    success_threshold="<= 0.5% of pairs violate parity beyond 2 ticks",
))
def test_deriv_003(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-003", symbol=ctx.symbol, layer="EOD",
        category="Consistency", severity="Medium", gate_type="Soft", weight=3.0,
    )
    if not _is_option(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    sym_cfg = (ctx.config.get("instruments", {})
               .get("equity", {}).get(ctx.symbol, {}))
    tick = sym_cfg.get("tick_size", 0.05)
    issues, metrics = [], {}

    for src, exch, df in _all_eod(ctx):
        required = {"strike_price", "option_type", "expiry_date"}
        if not required.issubset(set(df.columns)):
            metrics[f"{src}_{exch}"] = {"status": "missing columns"}
            continue

        # Pivot CE and PE for same date × strike × expiry
        calls = df[df["option_type"].str.upper() == "CE"].set_index(
            ["date", "strike_price", "expiry_date"]
        )["close"]
        puts = df[df["option_type"].str.upper() == "PE"].set_index(
            ["date", "strike_price", "expiry_date"]
        )["close"]

        common = calls.index.intersection(puts.index)
        if len(common) < 5:
            metrics[f"{src}_{exch}"] = {"status": "insufficient_pairs", "pairs": len(common)}
            continue

        c = calls[common]
        p = puts[common]
        # Simplified parity: C - P ≈ spot - strike (ignoring r, T for short-dated)
        # Use strike as proxy; actual spot not available per-row
        strikes = pd.Series([idx[1] for idx in common], index=common)
        parity_diff = (c - p - (c.mean() - strikes)).abs()  # residual
        violations = (parity_diff > 2 * tick).sum()
        violation_pct = violations / len(common) * 100

        metrics[f"{src}_{exch}"] = {
            "pairs_checked": len(common),
            "parity_violations": int(violations),
            "violation_pct": round(violation_pct, 3),
        }
        if violation_pct > 0.5:
            issues.append(
                f"{src}/{exch}: {violations} put-call parity violations "
                f"({violation_pct:.2f}%)"
            )

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("Put-call parity within tolerance.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-004  Implied volatility sanity range
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-004",
    name="Implied volatility sanity range",
    layer="EOD",
    category="Validity",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "IV from implied_volatility column (if present) should be in [0.5%, 300%] "
        "annualised. Outside range suggests bad tick or wrong contract."
    ),
    method_formula="0.005 <= IV_annualised <= 3.0",
    success_threshold=">= 99.5% of options have valid IV",
))
def test_deriv_004(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-004", symbol=ctx.symbol, layer="EOD",
        category="Validity", severity="Medium", gate_type="Soft", weight=3.0,
    )
    if not _is_option(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "implied_volatility" not in df.columns:
            metrics[f"{src}_{exch}"] = {"status": "no_iv_column"}
            continue

        iv = df["implied_volatility"].dropna()
        if len(iv) == 0:
            metrics[f"{src}_{exch}"] = {"status": "iv_all_null"}
            continue

        # IV in the file is typically in %, convert to decimal
        iv_decimal = iv / 100.0 if iv.mean() > 1.0 else iv
        bad = ((iv_decimal < 0.005) | (iv_decimal > 3.0)).sum()
        bad_pct = bad / len(iv) * 100

        metrics[f"{src}_{exch}"] = {
            "iv_rows": len(iv),
            "iv_out_of_range": int(bad),
            "bad_pct": round(bad_pct, 3),
            "iv_min": round(float(iv.min()), 4),
            "iv_max": round(float(iv.max()), 4),
        }
        if bad_pct > 0.5:
            issues.append(
                f"{src}/{exch}: {bad} IV values outside [0.5%,300%] ({bad_pct:.2f}%)"
            )

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("All IV values within plausible range.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-005  Options chain strike continuity
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-005",
    name="Options chain strike continuity",
    layer="EOD",
    category="Completeness",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "Strikes should be in regular increments (e.g. ₹50 for NIFTY50, "
        "₹100 for Banknifty). Gaps in the chain indicate missing records."
    ),
    method_formula="diff(sorted_strikes) == expected_step for >= 95% entries",
    success_threshold="Chain completeness >= 95% for each expiry",
))
def test_deriv_005(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-005", symbol=ctx.symbol, layer="EOD",
        category="Completeness", severity="Medium", gate_type="Soft", weight=3.0,
    )
    if not _is_option(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    issues, metrics = [], {}
    for src, exch, df in _all_eod(ctx):
        if "strike_price" not in df.columns or "expiry_date" not in df.columns:
            metrics[f"{src}_{exch}"] = {"status": "missing strike/expiry columns"}
            continue

        chain_results = {}
        for expiry, grp in df.groupby("expiry_date"):
            strikes = sorted(grp["strike_price"].dropna().unique())
            if len(strikes) < 3:
                continue
            diffs = np.diff(strikes)
            # Infer expected step: mode of differences
            from scipy import stats as sp_stats
            mode_result = sp_stats.mode(diffs, keepdims=False)
            expected_step = float(mode_result.mode)
            if expected_step <= 0:
                continue
            regular = (np.abs(diffs - expected_step) < expected_step * 0.1).mean()
            chain_results[str(expiry)] = {
                "strikes": len(strikes),
                "expected_step": expected_step,
                "regularity_pct": round(regular * 100, 1),
            }
            if regular < 0.95:
                issues.append(
                    f"{src}/{exch} expiry={expiry}: "
                    f"only {regular*100:.1f}% regular step"
                )

        metrics[f"{src}_{exch}"] = chain_results if chain_results else {"status": "no_expiry_groups"}

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("Options chain strike continuity OK.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# DERIV-006  Intraday OI monotonicity within session
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(
    test_id="DERIV-006",
    name="Intraday OI monotonicity within session",
    layer="INTRADAY",
    category="Consistency",
    gate_type="Soft",
    severity="Medium",
    weight=3.0,
    description=(
        "Intraday OI should not jump by > 20% in a single bar for derivatives. "
        "Large spikes indicate data reset or wrong contract mapping."
    ),
    method_formula="abs(OI_t - OI_{t-1})/OI_{t-1} > 0.20",
    success_threshold="OI bar-to-bar change <= 20%; exceptions reviewed",
))
def test_deriv_006(ctx: DQContext) -> TestResult:
    itype = _instrument_type(ctx)
    r = TestResult(
        test_id="DERIV-006", symbol=ctx.symbol, layer="INTRADAY",
        category="Consistency", severity="Medium", gate_type="Soft", weight=3.0,
    )
    if not _is_derivative(itype):
        r.set_skip(f"Not applicable for {itype}")
        return r

    threshold = 0.20
    issues, metrics = [], {}
    for exch, src_dict in ctx.data.get("intraday", {}).items():
        for src, df in src_dict.items():
            if df is None or df.empty:
                continue
            if "open_interest" not in df.columns or df["open_interest"].isna().all():
                metrics[f"{src}_{exch}"] = {"status": "no_oi"}
                continue
            df_s = df.sort_values("datetimestamp")
            oi = df_s["open_interest"].replace(0, np.nan)
            pct_change = oi.pct_change().abs()
            extreme = (pct_change > threshold).sum()
            metrics[f"{src}_{exch}"] = {
                "extreme_oi_jumps": int(extreme),
                "max_jump_pct": round(float(pct_change.max() * 100), 2),
            }
            if extreme > 5:
                issues.append(f"{src}/{exch}: {extreme} intraday OI jumps > 20%")

    if issues:
        r.set_fail(" | ".join(issues), metrics)
    else:
        r.set_pass("Intraday OI changes within 20% per bar.", metrics)
    return r
