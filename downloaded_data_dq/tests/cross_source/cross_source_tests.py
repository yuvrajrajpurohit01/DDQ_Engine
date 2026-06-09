"""
Downloaded Data DQ Engine — Cross-Source Tests
downloaded_data_dq/tests/cross_source/cross_source_tests.py

22 tests comparing data across Dhan, Kite, Upstox:
  SRC-E001–E006  EOD cross-source coverage, price/volume consistency, ANOVA, correlation
  SRC-I007–I012  Intraday equivalents
  SRC-001–007    Consensus deviation, reliability score, timestamp alignment
  SRC-E013/E014  Schema consistency
  SRC-I015       Intraday close divergence
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from downloaded_data_dq.framework import DQContext, TestResult, TestSpec, dq_test

logger = logging.getLogger(__name__)
SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]


def _eod_pairs(ctx: DQContext, exchange: str = "BSE"):
    """Return list of (src, df) tuples for available EOD frames on this exchange."""
    result = []
    for src in SOURCES:
        df = ctx.data.get("eod", {}).get(exchange, {}).get(src)
        if df is not None and not df.empty:
            result.append((src, df))
    return result


def _int_pairs(ctx: DQContext, exchange: str = "BSE"):
    result = []
    for src in SOURCES:
        df = ctx.data.get("intraday", {}).get(exchange, {}).get(src)
        if df is not None and not df.empty:
            result.append((src, df))
    return result


def _adjusted_skip(ctx, src_a, src_b) -> bool:
    """Return True if cross-price comparison should be skipped due to adjustment mismatch."""
    if not ctx.config.get("sources", {}).get("cross_source_rules", {}).get(
            "skip_price_comparison_if_adjustment_differs", True):
        return False
    adj_a = ctx.config.get("sources", {}).get("sources", {}).get(src_a, {}).get("is_adjusted_prices", True)
    adj_b = ctx.config.get("sources", {}).get("sources", {}).get(src_b, {}).get("is_adjusted_prices", True)
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
# SRC-E001  EOD: Missing dates per source
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E001", name="EOD: Missing dates per source",
    layer="SOURCE", category="Coverage", gate_type="Soft",
    severity="High", weight=4.0,
    description="Compare which trading dates each source is missing vs the union of all sources.",
    method_formula="union(all_dates) - source_dates = missing_dates",
    success_threshold="No source missing > 5% of union dates"))
def test_src_e001(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        if len(pairs) < 2:
            continue
        all_dates = pd.DatetimeIndex(
            sorted(set().union(*[set(df["date"]) for _, df in pairs]))
        )
        for src, df in pairs:
            src_dates = set(df["date"])
            missing = [d for d in all_dates if d not in src_dates]
            miss_pct = len(missing) / len(all_dates) * 100
            metrics[f"{src}_{exch}"] = {"union_dates": len(all_dates),
                                         "source_dates": len(src_dates),
                                         "missing": len(missing),
                                         "missing_pct": round(miss_pct, 2)}
            if miss_pct > 5:
                issues.append(f"{src}/{exch}: {len(missing)} dates missing vs union ({miss_pct:.1f}%)")
    r = TestResult(test_id="SRC-E001", symbol=ctx.symbol, layer="SOURCE",
                   category="Coverage", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD date coverage consistent across sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E002  EOD: Close match across sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E002", name="EOD: Close match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="High", weight=5.0,
    description="EOD close prices for same-adjusted sources should match within 0.1%.",
    method_formula="|close_A / close_B - 1| <= tolerance",
    success_threshold="< 0.5% of aligned dates diverge beyond tolerance"))
def test_src_e002(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_E_price_tolerance_pct", 0.1) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        for i, (src_a, df_a) in enumerate(pairs):
            for src_b, df_b in pairs[i+1:]:
                if _adjusted_skip(ctx, src_a, src_b):
                    metrics[f"{src_a}_{src_b}_{exch}"] = {"status": f"SKIPPED — adjustment mismatch"}
                    continue
                a, b = _align_eod(df_a, df_b)
                if len(a) == 0:
                    continue
                diff = (a["close"] / b["close"] - 1).abs()
                breach = (diff > tol).sum()
                pct = breach / len(a) * 100
                metrics[f"{src_a}_vs_{src_b}_{exch}"] = {"aligned_dates": len(a), "breaches": int(breach),
                                                           "breach_pct": round(pct, 3),
                                                           "max_diff_pct": round(float(diff.max() * 100), 4)}
                if pct > 0.5:
                    issues.append(f"{src_a} vs {src_b}/{exch}: {breach} close divergences > {tol*100:.2f}% ({pct:.2f}%)")
    r = TestResult(test_id="SRC-E002", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="High", gate_type="Soft", weight=5.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD close prices consistent across sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E003  EOD: OHLC match across sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E003", name="EOD: OHLC match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="Medium", weight=4.0,
    description="Open, High, Low should match within tolerance for same-adjusted sources.",
    method_formula="|field_A / field_B - 1| <= tolerance for O, H, L",
    success_threshold="< 1% of aligned dates diverge per field"))
def test_src_e003(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_E_price_tolerance_pct", 0.1) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        for i, (src_a, df_a) in enumerate(pairs):
            for src_b, df_b in pairs[i+1:]:
                if _adjusted_skip(ctx, src_a, src_b):
                    continue
                a, b = _align_eod(df_a, df_b)
                if len(a) == 0:
                    continue
                for field in ["open", "high", "low"]:
                    diff = (a[field] / b[field] - 1).abs()
                    breach_pct = (diff > tol).mean() * 100
                    key = f"{src_a}_vs_{src_b}_{exch}_{field}"
                    metrics[key] = {"breach_pct": round(breach_pct, 3)}
                    if breach_pct > 1.0:
                        issues.append(f"{src_a} vs {src_b}/{exch} {field}: {breach_pct:.2f}% beyond tolerance")
    r = TestResult(test_id="SRC-E003", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD OHLC consistent across sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E004  EOD: Volume match across sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E004", name="EOD: Volume match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="EOD volume should match across sources within 5% on the same exchange.",
    method_formula="|vol_A / vol_B - 1| <= 0.05",
    success_threshold="< 5% of dates diverge beyond 5%"))
def test_src_e004(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_E_volume_tolerance_pct", 5.0) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        for i, (src_a, df_a) in enumerate(pairs):
            for src_b, df_b in pairs[i+1:]:
                a, b = _align_eod(df_a, df_b)
                if len(a) == 0:
                    continue
                valid = (a["volume"] > 0) & (b["volume"] > 0)
                diff = (a["volume"][valid] / b["volume"][valid] - 1).abs()
                breach_pct = (diff > tol).mean() * 100
                metrics[f"{src_a}_vs_{src_b}_{exch}"] = {"breach_pct": round(breach_pct, 3)}
                if breach_pct > 5.0:
                    issues.append(f"{src_a} vs {src_b}/{exch} volume: {breach_pct:.2f}% beyond 5%")
    r = TestResult(test_id="SRC-E004", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD volume consistent across sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E005  EOD: ANOVA across 3 sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E005", name="EOD: ANOVA across sources",
    layer="SOURCE", category="Statistical", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="One-way ANOVA on log-returns: a significant F-stat suggests sources are systematically different.",
    method_formula="scipy.stats.f_oneway(returns_A, returns_B, returns_C); p < 0.001 = flag",
    success_threshold="ANOVA p-value >= 0.001 (sources not systematically different)"))
def test_src_e005(ctx: DQContext) -> TestResult:
    from scipy import stats
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        same_adj = [p for p in pairs if not _adjusted_skip(ctx, p[0], SOURCES[0])]
        returns_groups = []
        src_names = []
        for src, df in pairs:
            ret = np.log(df.sort_values("date")["close"] / df.sort_values("date")["close"].shift(1)).dropna()
            if len(ret) > 30:
                returns_groups.append(ret.values)
                src_names.append(src)
        if len(returns_groups) >= 2:
            f_stat, p_val = stats.f_oneway(*returns_groups)
            metrics[f"ANOVA_{exch}"] = {"f_stat": round(f_stat, 4), "p_value": round(p_val, 6),
                                         "sources": src_names}
            if p_val < 0.001:
                issues.append(f"{exch}: ANOVA significant (F={f_stat:.2f}, p={p_val:.4f}) — sources systematically differ")
    r = TestResult(test_id="SRC-E005", symbol=ctx.symbol, layer="SOURCE",
                   category="Statistical", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("ANOVA: sources not systematically different.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E006  EOD: Correlation matrix of returns
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E006", name="EOD: Return correlation across sources",
    layer="SOURCE", category="Correlation", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Same-instrument returns across same-adjusted sources should correlate >= 0.99.",
    method_formula="corr(log_returns_A, log_returns_B) >= 0.99",
    success_threshold="All source-pair correlations >= 0.99"))
def test_src_e006(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        ret_map = {}
        for src, df in pairs:
            df_s = df.sort_values("date").set_index("date")
            df_s = df_s[~df_s.index.duplicated(keep="first")]
            ret_map[src] = np.log(df_s["close"] / df_s["close"].shift(1)).dropna()
        for i, (sa, ra) in enumerate(ret_map.items()):
            for sb, rb in list(ret_map.items())[i+1:]:
                if _adjusted_skip(ctx, sa, sb):
                    continue
                common = ra.index.intersection(rb.index)
                if len(common) < 30:
                    continue
                corr = float(ra[common].corr(rb[common]))
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"correlation": round(corr, 6)}
                if corr < 0.99:
                    issues.append(f"{sa} vs {sb}/{exch}: return correlation = {corr:.4f} (< 0.99)")
    r = TestResult(test_id="SRC-E006", symbol=ctx.symbol, layer="SOURCE",
                   category="Correlation", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Return correlations >= 0.99 across sources.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-I007  INTRADAY: Missing dates per source
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-I007", name="INTRADAY: Missing dates per source",
    layer="SOURCE", category="Coverage", gate_type="Soft",
    severity="High", weight=4.0,
    description="Compare intraday trading dates across sources; flag if any source > 5% behind.",
    method_formula="union(dates) - source_dates",
    success_threshold="< 5% missing vs union"))
def test_src_i007(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        if len(pairs) < 2:
            continue
        all_dates = set()
        for _, df in pairs:
            all_dates.update(df["datetimestamp"].dt.date.unique())
        for src, df in pairs:
            src_dates = set(df["datetimestamp"].dt.date.unique())
            missing = all_dates - src_dates
            miss_pct = len(missing) / max(len(all_dates), 1) * 100
            metrics[f"{src}_{exch}"] = {"union": len(all_dates), "present": len(src_dates),
                                         "missing": len(missing), "miss_pct": round(miss_pct, 2)}
            if miss_pct > 5:
                issues.append(f"{src}/{exch}: {len(missing)} intraday dates missing ({miss_pct:.1f}%)")
    r = TestResult(test_id="SRC-I007", symbol=ctx.symbol, layer="SOURCE",
                   category="Coverage", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday date coverage consistent.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-I008  INTRADAY: Close match across sources at same timestamp
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-I008", name="INTRADAY: Close match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="High", weight=4.0,
    description="At aligned timestamps, closes should match within 0.5% for same-adjusted sources.",
    method_formula="|close_A / close_B - 1| <= 0.005",
    success_threshold="< 1% of aligned bars diverge"))
def test_src_i008(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_I_price_tolerance_pct", 0.5) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        for i, (sa, da) in enumerate(pairs):
            for sb, db in pairs[i+1:]:
                if _adjusted_skip(ctx, sa, sb):
                    metrics[f"{sa}_{sb}_{exch}"] = {"status": "SKIPPED adj mismatch"}
                    continue
                a = da.set_index("datetimestamp")["close"]
                b = db.set_index("datetimestamp")["close"]
                common = a.index.intersection(b.index)
                if len(common) < 100:
                    continue
                diff = (a[common] / b[common] - 1).abs()
                breach_pct = (diff > tol).mean() * 100
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"aligned_bars": len(common),
                                                     "breach_pct": round(breach_pct, 3)}
                if breach_pct > 1.0:
                    issues.append(f"{sa} vs {sb}/{exch}: {breach_pct:.2f}% intraday close divergences")
    r = TestResult(test_id="SRC-I008", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday close prices consistent.", metrics)
    return r


# SRC-I009, SRC-I010, SRC-I011, SRC-I012 follow same pattern with minor field changes
@dq_test(TestSpec(test_id="SRC-I009", name="INTRADAY: OHLC match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft", severity="Medium", weight=3.0,
    description="Open/High/Low match across sources at aligned timestamps.", success_threshold="< 1% breach"))
def test_src_i009(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_I_price_tolerance_pct", 0.5) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        for i, (sa, da) in enumerate(pairs):
            for sb, db in pairs[i+1:]:
                if _adjusted_skip(ctx, sa, sb):
                    continue
                common_ts = set(da["datetimestamp"]).intersection(set(db["datetimestamp"]))
                if len(common_ts) < 100:
                    continue
                a = da[da["datetimestamp"].isin(common_ts)].set_index("datetimestamp")
                b = db[db["datetimestamp"].isin(common_ts)].set_index("datetimestamp")
                for field in ["open", "high", "low"]:
                    diff = (a[field] / b[field] - 1).abs()
                    bp = (diff > tol).mean() * 100
                    metrics[f"{sa}_vs_{sb}_{exch}_{field}"] = {"breach_pct": round(bp, 3)}
                    if bp > 1.0:
                        issues.append(f"{sa} vs {sb}/{exch} {field}: {bp:.2f}% divergence")
    r = TestResult(test_id="SRC-I009", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday OHLC consistent.", metrics)
    return r


@dq_test(TestSpec(test_id="SRC-I010", name="INTRADAY: Volume match across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft", severity="Low", weight=2.0,
    description="Volume at same timestamp across sources should match within 5%.", success_threshold="< 5% breach"))
def test_src_i010(ctx: DQContext) -> TestResult:
    tol = ctx.threshold("cross_source.SRC_I_volume_tolerance_pct", 5.0) / 100
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        for i, (sa, da) in enumerate(pairs):
            for sb, db in pairs[i+1:]:
                common_ts = set(da["datetimestamp"]).intersection(set(db["datetimestamp"]))
                if len(common_ts) < 100:
                    continue
                a = da[da["datetimestamp"].isin(common_ts)].set_index("datetimestamp")["volume"]
                b = db[db["datetimestamp"].isin(common_ts)].set_index("datetimestamp")["volume"]
                valid = (a > 0) & (b > 0)
                diff = (a[valid] / b[valid] - 1).abs()
                bp = (diff > tol).mean() * 100
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"breach_pct": round(bp, 3)}
                if bp > 5.0:
                    issues.append(f"{sa} vs {sb}/{exch} volume: {bp:.2f}% breach")
    r = TestResult(test_id="SRC-I010", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday volume consistent.", metrics)
    return r


@dq_test(TestSpec(test_id="SRC-I011", name="INTRADAY: ANOVA across sources",
    layer="SOURCE", category="Statistical", gate_type="Soft", severity="Medium", weight=3.0,
    description="One-way ANOVA on intraday bar-to-bar returns across sources.", success_threshold="p >= 0.001"))
def test_src_i011(ctx: DQContext) -> TestResult:
    from scipy import stats
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        groups = []
        src_names = []
        for src, df in pairs:
            ret = df.sort_values("datetimestamp")["close"].pct_change().dropna()
            if len(ret) > 100:
                groups.append(ret.values)
                src_names.append(src)
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            metrics[f"ANOVA_{exch}"] = {"f_stat": round(f_stat, 4), "p_value": round(p_val, 6)}
            if p_val < 0.001:
                issues.append(f"{exch}: Intraday ANOVA significant (F={f_stat:.2f}, p={p_val:.4f})")
    r = TestResult(test_id="SRC-I011", symbol=ctx.symbol, layer="SOURCE",
                   category="Statistical", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday ANOVA not significant.", metrics)
    return r


@dq_test(TestSpec(test_id="SRC-I012", name="INTRADAY: Return correlation across sources",
    layer="SOURCE", category="Correlation", gate_type="Soft", severity="Low", weight=2.0,
    description="Intraday bar returns should correlate >= 0.95 across sources.", success_threshold="corr >= 0.95"))
def test_src_i012(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        ret_map = {src: df.set_index("datetimestamp")["close"].pct_change().dropna()
                   for src, df in pairs}
        src_list = list(ret_map.items())
        for i, (sa, ra) in enumerate(src_list):
            for sb, rb in src_list[i+1:]:
                if _adjusted_skip(ctx, sa, sb):
                    continue
                common = ra.index.intersection(rb.index)
                if len(common) < 100:
                    continue
                corr = float(ra[common].corr(rb[common]))
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"correlation": round(corr, 6)}
                if corr < 0.95:
                    issues.append(f"{sa} vs {sb}/{exch}: intraday corr = {corr:.4f}")
    r = TestResult(test_id="SRC-I012", symbol=ctx.symbol, layer="SOURCE",
                   category="Correlation", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday correlations adequate.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-001  Consensus median deviation
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-001", name="Consensus median deviation",
    layer="SOURCE", category="Consensus", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="Each source's close vs median of all sources. Sources > 1% from median flagged.",
    method_formula="|close_src / median_all - 1| > 0.01",
    success_threshold="< 0.5% of dates per source deviate > 1% from consensus"))
def test_src_001(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        if len(pairs) < 2:
            continue
        # Align all sources on common dates (deduplicate first)
        dfs = {}
        for src, df in pairs:
            s = df.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="first")]
            dfs[src] = s
        common_dates = list(dfs.values())[0].index
        for s in list(dfs.values())[1:]:
            common_dates = common_dates.intersection(s.index)
        if len(common_dates) < 10:
            continue
        aligned = pd.DataFrame({src: s[common_dates] for src, s in dfs.items()})
        consensus = aligned.median(axis=1)
        for src in aligned.columns:
            dev = ((aligned[src] / consensus - 1).abs() > 0.01).mean() * 100
            metrics[f"{src}_{exch}"] = {"deviation_pct": round(dev, 3)}
            if dev > 0.5:
                issues.append(f"{src}/{exch}: {dev:.2f}% dates > 1% from consensus median")
    r = TestResult(test_id="SRC-001", symbol=ctx.symbol, layer="SOURCE",
                   category="Consensus", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All sources within 1% of consensus.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-002  Source reliability score
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-002", name="Source reliability score",
    layer="SOURCE", category="Reliability", gate_type="Soft",
    severity="High", weight=4.0,
    description="Composite score per source: completeness + consistency + freshness. Alert if score < threshold.",
    method_formula="score = 0.5*completeness + 0.3*consistency + 0.2*freshness",
    success_threshold="score >= 0.75 for EOD-only mode"))
def test_src_002(ctx: DQContext) -> TestResult:
    threshold = ctx.threshold("cross_source.reliability_eod_only_threshold", 0.75)
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        for src, df in pairs:
            if df.empty:
                continue
            # Completeness: non-null OHLCV rate
            completeness = 1.0 - df[["open","high","low","close","volume"]].isna().mean().mean()
            # Consistency: % rows with valid OHLC (H>=L, C in [L,H])
            valid = ((df["high"] >= df["low"]) &
                     (df["close"] >= df["low"]) &
                     (df["close"] <= df["high"])).mean()
            # Freshness: data recency (< 30 days old = 1.0)
            last_date = df["date"].max()
            now = pd.Timestamp.today().normalize()
            days_stale = (now - last_date).days
            freshness = max(0.0, 1.0 - days_stale / 365)
            score = 0.5 * completeness + 0.3 * float(valid) + 0.2 * freshness
            metrics[f"{src}_{exch}"] = {"score": round(score, 4), "completeness": round(completeness, 4),
                                         "consistency": round(float(valid), 4), "freshness": round(freshness, 4),
                                         "days_stale": days_stale}
            if score < threshold:
                issues.append(f"{src}/{exch}: reliability score {score:.3f} < {threshold}")
    r = TestResult(test_id="SRC-002", symbol=ctx.symbol, layer="SOURCE",
                   category="Reliability", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All sources meet reliability threshold.", metrics)
    return r


# SRC-003 to SRC-007  Intraday per-field timestamp alignment
_SRC_INT_FIELDS = [
    ("SRC-003", "open",   "Intraday: Open alignment at same timestamp"),
    ("SRC-004", "high",   "Intraday: High alignment at same timestamp"),
    ("SRC-005", "low",    "Intraday: Low alignment at same timestamp"),
    ("SRC-006", "close",  "Intraday: Close alignment at same timestamp"),
    ("SRC-007", "volume", "Intraday: Volume alignment at same timestamp"),
]

for _tid, _field, _name in _SRC_INT_FIELDS:
    _tol = 0.005 if _field != "volume" else 0.05
    _spec = TestSpec(test_id=_tid, name=_name, layer="SOURCE",
                     category="Temporal", gate_type="Soft", severity="Medium", weight=3.0,
                     description=f"Intraday {_field} at aligned timestamps within tolerance.",
                     success_threshold=f"< 1% breaches at {_tol*100}% tolerance")
    def _make_src_int(tid, field, tol, spec):
        @dq_test(spec)
        def _test(ctx: DQContext, _t=tid, _f=field, _tol=tol) -> TestResult:
            issues, metrics = [], {}
            for exch in ["BSE", "NSE"]:
                pairs = _int_pairs(ctx, exch)
                for i, (sa, da) in enumerate(pairs):
                    for sb, db in pairs[i+1:]:
                        if _f != "volume" and _adjusted_skip(ctx, sa, sb):
                            continue
                        common = set(da["datetimestamp"]).intersection(set(db["datetimestamp"]))
                        if len(common) < 100:
                            continue
                        a = da[da["datetimestamp"].isin(common)].set_index("datetimestamp")[_f]
                        b = db[db["datetimestamp"].isin(common)].set_index("datetimestamp")[_f]
                        if _f == "volume":
                            valid = (a > 0) & (b > 0)
                            diff = (a[valid] / b[valid] - 1).abs()
                        else:
                            diff = (a / b - 1).abs()
                        bp = (diff > _tol).mean() * 100
                        metrics[f"{sa}_vs_{sb}_{exch}"] = {"breach_pct": round(bp, 3)}
                        if bp > 1.0:
                            issues.append(f"{sa} vs {sb}/{exch} {_f}: {bp:.2f}% breach")
            r = TestResult(test_id=_t, symbol=ctx.symbol, layer="SOURCE",
                           category="Temporal", severity="Medium", gate_type="Soft", weight=3.0)
            r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass(f"Intraday {_f} aligned.", metrics)
            return r
        return _test
    _make_src_int(_tid, _field, _tol, _spec)


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E013  EOD start date consistency across sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E013", name="EOD start date consistency",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="Low", weight=2.0,
    description="Flag sources whose EOD history starts significantly later than the earliest source.",
    method_formula="max(start_date) - min(start_date) > 365 days",
    success_threshold="Start dates within 2 years of each other"))
def test_src_e013(ctx: DQContext) -> TestResult:
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        starts = {src: df["date"].min() for src, df in pairs}
        if len(starts) >= 2:
            span_days = (max(starts.values()) - min(starts.values())).days
            metrics[f"{exch}_start_span_days"] = span_days
            for src, s in starts.items():
                metrics[f"{src}_{exch}_start"] = str(s.date())
            if span_days > 730:
                issues.append(f"{exch}: Source start dates span {span_days} days")
    r = TestResult(test_id="SRC-E013", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Low", gate_type="Soft", weight=2.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("EOD start dates consistent.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-E014  Column schema alignment across sources
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-E014", name="Column schema alignment across sources",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="Medium", weight=3.0,
    description="All sources should have the same canonical columns after normalisation.",
    method_formula="set(cols_A) == set(cols_B) == set(cols_C)",
    success_threshold="All sources have identical canonical column sets"))
def test_src_e014(ctx: DQContext) -> TestResult:
    BASE_COLS = {"date","open","high","low","close","adj_close","volume","open_interest",
                 "source","symbol","exchange","instrument_type"}
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _eod_pairs(ctx, exch)
        for src, df in pairs:
            cols = set(df.columns)
            missing = BASE_COLS - cols
            extra = cols - BASE_COLS - {"timestamp_epoch","datetime_ist","nav","expiry_date",
                                         "strike_price","option_type","implied_volatility","lot_size"}
            metrics[f"{src}_{exch}"] = {"columns": sorted(cols), "missing": sorted(missing), "extra": sorted(extra)}
            if missing:
                issues.append(f"{src}/{exch}: missing canonical cols: {missing}")
    r = TestResult(test_id="SRC-E014", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="Medium", gate_type="Soft", weight=3.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("All sources have canonical schema.", metrics)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SRC-I015  Intraday close divergence at same timestamp
# ─────────────────────────────────────────────────────────────────────────────
@dq_test(TestSpec(test_id="SRC-I015", name="Intraday close divergence",
    layer="SOURCE", category="Consistency", gate_type="Soft",
    severity="High", weight=4.0,
    description="At the same timestamp, close prices across same-adjusted sources should be near-identical.",
    method_formula="|close_A / close_B - 1| at each aligned bar",
    success_threshold="< 0.5% of aligned bars diverge > 0.5%"))
def test_src_i015(ctx: DQContext) -> TestResult:
    tol = 0.005  # 0.5%
    issues, metrics = [], {}
    for exch in ["BSE", "NSE"]:
        pairs = _int_pairs(ctx, exch)
        for i, (sa, da) in enumerate(pairs):
            for sb, db in pairs[i+1:]:
                if _adjusted_skip(ctx, sa, sb):
                    metrics[f"{sa}_{sb}_{exch}"] = {"status": "SKIPPED adj mismatch"}
                    continue
                a = da.set_index("datetimestamp")["close"]
                b = db.set_index("datetimestamp")["close"]
                common = a.index.intersection(b.index)
                if len(common) < 100:
                    continue
                diff = (a[common] / b[common] - 1).abs()
                bp = (diff > tol).mean() * 100
                metrics[f"{sa}_vs_{sb}_{exch}"] = {"aligned": len(common), "breach_pct": round(bp, 3),
                                                     "max_diff_pct": round(float(diff.max() * 100), 4)}
                if bp > 0.5:
                    issues.append(f"{sa} vs {sb}/{exch}: {bp:.2f}% intraday close divergence > 0.5%")
    r = TestResult(test_id="SRC-I015", symbol=ctx.symbol, layer="SOURCE",
                   category="Consistency", severity="High", gate_type="Soft", weight=4.0)
    r.set_fail(" | ".join(issues), metrics) if issues else r.set_pass("Intraday close divergence within tolerance.", metrics)
    return r
