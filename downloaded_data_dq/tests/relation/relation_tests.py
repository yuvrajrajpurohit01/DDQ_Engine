"""
Downloaded Data DQ Engine — Relation Tests: REL-001 to REL-012
downloaded_data_dq/tests/relation/relation_tests.py

12 tests reconciling EOD data against Intraday aggregations per source:
  REL-001  EOD Close vs last intraday close
  REL-002  EOD High vs intraday max high
  REL-003  EOD Low vs intraday min low
  REL-004  EOD Open vs first intraday open
  REL-005  EOD Volume vs sum intraday volume
  REL-006  Paired t-test on close differences
  REL-007  Correlation of daily returns (EOD vs intraday-derived)
  REL-008  Date alignment across EOD and intraday datasets
  REL-009  EOD from intraday aggregation reconstruction
  REL-010  Bland-Altman agreement on Close
  REL-011  Close rounding / tick-size tolerance
  REL-012  Session completeness vs EOD
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


def _aggregate_intraday_to_eod(df_int: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min intraday bars to daily OHLCV."""
    df = df_int.copy()
    df["date"] = df["datetimestamp"].dt.normalize()
    agg = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        open_interest=("open_interest", "last"),
    ).reset_index()
    return agg


def _align_eod_int(df_eod: pd.DataFrame, df_int_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner join EOD and intraday-aggregated frames on date.
    Returns reset-indexed DataFrames guaranteed to have identical length and row order.
    """
    e = df_eod.copy()
    i = df_int_agg.copy()

    # Normalize dates to remove any time component
    e["date"] = pd.to_datetime(e["date"]).dt.normalize()
    i["date"] = pd.to_datetime(i["date"]).dt.normalize()

    # Merge on date — guarantees same length and row-for-row alignment
    merged = e.merge(i, on="date", suffixes=("_eod", "_int"))

    # Reconstruct two aligned DataFrames from the merged result
    eod_cols = {c.replace("_eod", ""): c for c in merged.columns if c.endswith("_eod")}
    int_cols = {c.replace("_int", ""): c for c in merged.columns if c.endswith("_int")}
    shared_price = ["open", "high", "low", "close", "volume", "open_interest"]

    e_out = pd.DataFrame({"date": merged["date"]})
    i_out = pd.DataFrame({"date": merged["date"]})

    for field in shared_price:
        ecol = f"{field}_eod" if f"{field}_eod" in merged.columns else field
        icol = f"{field}_int" if f"{field}_int" in merged.columns else field
        if ecol in merged.columns:
            e_out[field] = merged[ecol].values
        if icol in merged.columns:
            i_out[field] = merged[icol].values

    return e_out.reset_index(drop=True), i_out.reset_index(drop=True)


def _get_pair(ctx: DQContext, src: str, exch: str):
    """Return (eod_df, intraday_df) for a source+exchange, or (None, None)."""
    eod = ctx.data.get("eod", {}).get(exch, {}).get(src)
    int_ = ctx.data.get("intraday", {}).get(exch, {}).get(src)
    return eod, int_


# ─────────────────────────────────────────────────────────────────────────────
# REL-001  EOD Close vs last intraday close
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-001", name="EOD Close vs last intraday close",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="High", weight=5.0,
    description="EOD close should match the last intraday bar's close within 0.5% per source.",
    method_formula="|eod_close / last_int_close - 1| <= 0.005",
    success_threshold="< 1% of days have close divergence > 0.5%"))
def test_rel_001(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("relation.REL_close_tolerance_pct", 0.5) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            diff = (e["close"] / i["close"] - 1).abs()
            breach_pct = (diff > tol).mean() * 100
            metrics[f"{src}_{exch}"] = {"aligned_days": len(e), "breach_pct": round(breach_pct, 3),
                                         "max_diff_pct": round(float(diff.max() * 100), 4)}
            if breach_pct > 1.0:
                issues.append(f"{src}/{exch}: {breach_pct:.2f}% EOD-intraday close divergences > {tol*100:.1f}%")
    r = TestResult(test_id="REL-001", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="High", gate_type="Soft", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD close matches last intraday close.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-002  EOD High vs intraday max high
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-002", name="EOD High vs intraday max high",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="High", weight=5.0,
    description="EOD high should be >= intraday max high. EOD high significantly below intraday max is an error.",
    method_formula="eod_high >= int_max_high - tolerance",
    success_threshold="No days where EOD high < intraday max high - 0.1%"))
def test_rel_002(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("relation.REL_high_tolerance_pct", 0.1) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            bad = (e["high"] < i["high"] * (1 - tol)).sum()
            metrics[f"{src}_{exch}"] = {"aligned_days": len(e), "violations": int(bad)}
            if bad > 0:
                issues.append(f"{src}/{exch}: {bad} days where EOD high < intraday max high")
    r = TestResult(test_id="REL-002", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="High", gate_type="Soft", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD high >= intraday max high.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-003  EOD Low vs intraday min low
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-003", name="EOD Low vs intraday min low",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="High", weight=5.0,
    description="EOD low should be <= intraday min low. EOD low significantly above intraday min is an error.",
    method_formula="eod_low <= int_min_low + tolerance",
    success_threshold="No days where EOD low > intraday min low + 0.1%"))
def test_rel_003(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("relation.REL_low_tolerance_pct", 0.1) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            bad = (e["low"] > i["low"] * (1 + tol)).sum()
            metrics[f"{src}_{exch}"] = {"aligned_days": len(e), "violations": int(bad)}
            if bad > 0:
                issues.append(f"{src}/{exch}: {bad} days where EOD low > intraday min low")
    r = TestResult(test_id="REL-003", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="High", gate_type="Soft", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD low <= intraday min low.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-004  EOD Open vs first intraday open
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-004", name="EOD Open vs first intraday open",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="Medium", weight=4.0,
    description="EOD open should match the first intraday bar open within 0.5%.",
    method_formula="|eod_open / first_int_open - 1| <= 0.005",
    success_threshold="< 1% of days deviate > 0.5%"))
def test_rel_004(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("relation.REL_open_tolerance_pct", 0.5) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            diff = (e["open"] / i["open"] - 1).abs()
            breach_pct = (diff > tol).mean() * 100
            metrics[f"{src}_{exch}"] = {"aligned_days": len(e), "breach_pct": round(breach_pct, 3)}
            if breach_pct > 1.0:
                issues.append(f"{src}/{exch}: {breach_pct:.2f}% open divergence > {tol*100:.1f}%")
    r = TestResult(test_id="REL-004", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD open matches intraday first open.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-005  EOD Volume vs sum of intraday volume
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-005", name="EOD Volume vs intraday sum",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="Medium", weight=4.0,
    description="EOD volume should roughly equal the sum of all 1-min intraday volumes.",
    method_formula="|eod_vol / sum(int_vol) - 1| <= 0.02",
    success_threshold="< 5% of days diverge > 2%"))
def test_rel_005(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("relation.REL_volume_tolerance_pct", 2.0) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            valid = (e["volume"] > 0) & (i["volume"] > 0)
            diff = (e["volume"][valid] / i["volume"][valid] - 1).abs()
            breach_pct = (diff > tol).mean() * 100
            metrics[f"{src}_{exch}"] = {"aligned_days": len(e), "breach_pct": round(breach_pct, 3),
                                         "median_ratio": round(float((e["volume"][valid] / i["volume"][valid]).median()), 4)}
            if breach_pct > 5.0:
                issues.append(f"{src}/{exch}: {breach_pct:.2f}% volume divergence > {tol*100:.0f}%")
    r = TestResult(test_id="REL-005", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD volume matches intraday sum.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-006  Paired t-test on EOD vs intraday-derived close differences
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-006", name="Paired t-test EOD vs intraday close",
    layer="RELATION", category="Statistical", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Paired t-test on (eod_close - int_last_close). Significant difference signals systematic bias.",
    method_formula="scipy.stats.ttest_rel; p < 0.001 = flag",
    success_threshold="p-value >= 0.001 (no systematic bias)"))
def test_rel_006(ctx: DQContext) -> TestResult:
    from scipy import stats
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) < 30:
                continue
            diffs = e["close"].values - i["close"].values
            t_stat, p_val = stats.ttest_1samp(diffs, 0)
            metrics[f"{src}_{exch}"] = {"t_stat": round(t_stat, 4), "p_value": round(p_val, 6),
                                         "mean_diff": round(float(np.mean(diffs)), 4), "n": len(e)}
            if p_val < 0.001:
                issues.append(f"{src}/{exch}: Significant close bias (t={t_stat:.2f}, p={p_val:.4f})")
    r = TestResult(test_id="REL-006", symbol=ctx.symbol, layer="RELATION",
                   category="Statistical", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("No systematic EOD-intraday close bias.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-007  Correlation of daily returns (EOD vs intraday-derived)
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-007", name="Return correlation EOD vs intraday",
    layer="RELATION", category="Correlation", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Daily returns computed from EOD and from intraday close should correlate >= 0.98.",
    method_formula="corr(eod_returns, int_returns) >= 0.98",
    success_threshold="Correlation >= 0.98 per source"))
def test_rel_007(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) < 30:
                continue
            eod_ret = e["close"].pct_change().dropna()
            int_ret = i["close"].pct_change().dropna()
            common_idx = eod_ret.index.intersection(int_ret.index)
            if len(common_idx) < 30:
                continue
            corr = float(eod_ret[common_idx].corr(int_ret[common_idx]))
            metrics[f"{src}_{exch}"] = {"correlation": round(corr, 6), "n": len(common_idx)}
            if corr < 0.98:
                issues.append(f"{src}/{exch}: EOD-intraday return correlation = {corr:.4f} (< 0.98)")
    r = TestResult(test_id="REL-007", symbol=ctx.symbol, layer="RELATION",
                   category="Correlation", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD-intraday return correlation adequate.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-008  Date alignment across EOD and intraday datasets
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-008", name="Date alignment EOD vs intraday",
    layer="RELATION", category="Coverage", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Dates present in EOD but absent from intraday (and vice versa) should be minimal.",
    method_formula="symmetric_diff(eod_dates, intraday_dates) / union < 5%",
    success_threshold="< 5% date misalignment"))
def test_rel_008(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            eod_dates = set(eod["date"].dt.normalize())
            int_dates = set(int_["datetimestamp"].dt.normalize())
            # Restrict to overlapping date range
            start = max(min(eod_dates), min(int_dates))
            end = min(max(eod_dates), max(int_dates))
            eod_in_range = {d for d in eod_dates if start <= d <= end}
            int_in_range = {d for d in int_dates if start <= d <= end}
            in_eod_not_int = eod_in_range - int_in_range
            in_int_not_eod = int_in_range - eod_in_range
            union = eod_in_range | int_in_range
            misalign_pct = len(in_eod_not_int | in_int_not_eod) / max(len(union), 1) * 100
            metrics[f"{src}_{exch}"] = {"in_eod_not_intraday": len(in_eod_not_int),
                                         "in_intraday_not_eod": len(in_int_not_eod),
                                         "misalignment_pct": round(misalign_pct, 2)}
            if misalign_pct > 5.0:
                issues.append(f"{src}/{exch}: {misalign_pct:.1f}% date misalignment between EOD and intraday")
    r = TestResult(test_id="REL-008", symbol=ctx.symbol, layer="RELATION",
                   category="Coverage", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD and intraday dates well aligned.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-009  EOD reconstruction from intraday aggregation
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-009", name="EOD from intraday aggregation",
    layer="RELATION", category="Aggregation", gate_type="Soft",
    severity="High", weight=4.0,
    description="Reconstruct EOD OHLCV from 1-min bars and compare with stated EOD values.",
    method_formula="agg(1min) → compare OHLCV vs EOD file within tolerance",
    success_threshold="OHLCV all within tolerance on >= 95% of days"))
def test_rel_009(ctx: DQContext) -> TestResult:
    tols = {"open": 0.005, "high": 0.001, "low": 0.001, "close": 0.005, "volume": 0.02}
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) < 10:
                continue
            field_results = {}
            for field, tol in tols.items():
                if field == "volume":
                    valid = (e[field] > 0) & (i[field] > 0)
                    diff = (e[field][valid] / i[field][valid] - 1).abs()
                else:
                    diff = (e[field] / i[field] - 1).abs()
                match_pct = (diff <= tol).mean() * 100
                field_results[field] = round(match_pct, 2)
                if match_pct < 95.0:
                    issues.append(f"{src}/{exch} {field}: only {match_pct:.1f}% match (tol={tol*100:.1f}%)")
            metrics[f"{src}_{exch}"] = field_results
    r = TestResult(test_id="REL-009", symbol=ctx.symbol, layer="RELATION",
                   category="Aggregation", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD successfully reconstructed from intraday.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-010  Bland-Altman agreement on Close
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-010", name="Bland-Altman EOD vs intraday close",
    layer="RELATION", category="Statistical", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Bland-Altman method of agreement: (eod - intraday_last) vs mean. Flag systematic bias.",
    method_formula="mean_diff ± 1.96*sd; flag if mean_diff > 0.5% of average close",
    success_threshold="Mean difference < 0.5% of average close"))
def test_rel_010(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) < 30:
                continue
            diff = e["close"].values - i["close"].values
            mean_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff))
            avg_close = float(np.mean([e["close"].mean(), i["close"].mean()]))
            bias_pct = abs(mean_diff) / avg_close * 100 if avg_close > 0 else 0
            metrics[f"{src}_{exch}"] = {"mean_diff": round(mean_diff, 4), "sd_diff": round(sd_diff, 4),
                                         "loa_lower": round(mean_diff - 1.96 * sd_diff, 4),
                                         "loa_upper": round(mean_diff + 1.96 * sd_diff, 4),
                                         "bias_pct": round(bias_pct, 4)}
            if bias_pct > 0.5:
                issues.append(f"{src}/{exch}: Bland-Altman mean bias = {bias_pct:.3f}% of avg close")
    r = TestResult(test_id="REL-010", symbol=ctx.symbol, layer="RELATION",
                   category="Statistical", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Bland-Altman bias within acceptable limits.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-011  Close rounding / tick-size tolerance
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-011", name="Close rounding tick-size tolerance",
    layer="RELATION", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Small close differences between EOD and intraday-derived may be due to rounding to tick size.",
    method_formula="diff = abs(eod_close - int_close); flag if diff > 2 * tick_size",
    success_threshold="< 1% of days have close diff > 2 ticks"))
def test_rel_011(ctx: DQContext) -> TestResult:
    sym_cfg = (ctx.config.get("instruments", {}).get("equity", {}).get(ctx.symbol, {}))
    tick = sym_cfg.get("tick_size", 0.05)
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            int_agg = _aggregate_intraday_to_eod(int_)
            e, i = _align_eod_int(eod, int_agg)
            if len(e) == 0:
                continue
            diff = (e["close"] - i["close"]).abs()
            beyond_2ticks = (diff > 2 * tick).sum()
            pct = beyond_2ticks / len(e) * 100
            metrics[f"{src}_{exch}"] = {"beyond_2ticks": int(beyond_2ticks), "pct": round(pct, 3),
                                         "tick_size": tick}
            if pct > 1.0:
                issues.append(f"{src}/{exch}: {beyond_2ticks} days close diff > 2 ticks ({pct:.2f}%)")
    r = TestResult(test_id="REL-011", symbol=ctx.symbol, layer="RELATION",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Close differences within 2 ticks.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# REL-012  Session completeness vs EOD
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="REL-012", name="Session completeness vs EOD",
    layer="RELATION", category="Coverage", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Every EOD date should have corresponding intraday bars. Orphan EOD dates (no intraday) flagged.",
    method_formula="count(eod_dates with no intraday bars) / total_eod_dates < 5%",
    success_threshold="< 5% of EOD dates without intraday data"))
def test_rel_012(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        for src in SOURCES:
            eod, int_ = _get_pair(ctx, src, exch)
            if eod is None or int_ is None or eod.empty or int_.empty:
                continue
            eod_dates = set(eod["date"].dt.normalize())
            int_dates = set(int_["datetimestamp"].dt.normalize())
            # Focus on overlapping period
            start = max(min(eod_dates), min(int_dates))
            end = min(max(eod_dates), max(int_dates))
            eod_in_range = {d for d in eod_dates if start <= d <= end}
            orphan = eod_in_range - int_dates
            orphan_pct = len(orphan) / max(len(eod_in_range), 1) * 100
            metrics[f"{src}_{exch}"] = {"eod_in_range": len(eod_in_range),
                                         "orphan_eod_dates": len(orphan),
                                         "orphan_pct": round(orphan_pct, 2)}
            if orphan_pct > 5.0:
                issues.append(f"{src}/{exch}: {len(orphan)} EOD dates without intraday ({orphan_pct:.1f}%)")
    r = TestResult(test_id="REL-012", symbol=ctx.symbol, layer="RELATION",
                   category="Coverage", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All EOD dates have intraday session data.", metrics)
    return r
