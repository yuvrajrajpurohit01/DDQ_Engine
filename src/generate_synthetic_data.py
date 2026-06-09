#!/usr/bin/env python3
"""
DDQ Engine v12 — Synthetic Data Generator
generate_synthetic_data.py

Generates realistic synthetic market data with canonical column schema.

Folder structure:
  data/synthetic/{eod|intraday}/{source}/{exchange}/{product_class}/{Symbol}.csv

Canonical columns:
  EOD:      date, open, high, low, close, adj_close, volume, open_interest
  INTRADAY: datetimestamp, open, high, low, close, adj_close, volume, open_interest

12 Symbols: TCS, RELIANCE, HDFCBANK, INFY, ICICIBANK,
            NIFTYBEES, BANKBEES, GOLDBEES, SILVERBEES,
            NIFTY50, SENSEX, BANKNIFTY

9 Product Classes: EQUITY, ETF, EQUITY_OPT_CE, EQUITY_OPT_PE, EQUITY_FUT,
                   INDEX_EQUITY, INDEX_OPT_CE, INDEX_OPT_PE, INDEX_FUT

All timestamps in IST (Asia/Kolkata). All numerics as float.
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

DEFAULT_SEED = 2025
SOURCES = ["dhan", "kite", "upstox"]

# ── Trading holidays (major Indian holidays 2015-2025) ───────────────────────
_HOLIDAY_STRS = [
    # 2024
    "2024-01-22","2024-01-26","2024-03-25","2024-03-29","2024-04-11",
    "2024-04-14","2024-04-17","2024-04-21","2024-05-01","2024-05-23",
    "2024-06-17","2024-07-17","2024-08-15","2024-09-07","2024-10-02",
    "2024-10-12","2024-10-31","2024-11-01","2024-11-15","2024-12-25",
    # 2025
    "2025-01-26","2025-02-26","2025-03-14","2025-03-31","2025-04-10",
    "2025-04-14","2025-04-18","2025-05-01","2025-06-07","2025-07-10",
    "2025-08-15","2025-08-16","2025-08-27","2025-10-02","2025-10-20",
    "2025-10-21","2025-10-22","2025-11-05","2025-11-26","2025-12-25",
]
HOLIDAYS = set(pd.to_datetime(_HOLIDAY_STRS))


def _trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end).difference(HOLIDAYS)


def _monthly_expiries(start: str, end: str) -> list[pd.Timestamp]:
    """Last Thursday of each month."""
    expiries = []
    for m in pd.date_range(start, end, freq="MS"):
        last = m + pd.offsets.MonthEnd(0)
        off = (last.weekday() - 3) % 7
        exp = last - pd.Timedelta(days=off)
        if exp < m:
            exp += pd.Timedelta(days=7)
        expiries.append(exp)
    return expiries


# ══════════════════════════════════════════════════════════════════════════════
# PRICE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _gen_eod_base(rng, n: int, base_price: float, vol: float) -> np.ndarray:
    """Generate n close prices via GBM."""
    rets = rng.normal(0.0001, vol / np.sqrt(252), n)
    return base_price * np.cumprod(1 + rets)


def _make_eod_df(rng, dates: pd.DatetimeIndex, closes: np.ndarray,
                  has_volume: bool, has_adj_close: bool,
                  has_oi: bool) -> pd.DataFrame:
    """Build canonical EOD DataFrame."""
    n = len(dates)
    spread = np.std(np.diff(closes[:min(n, 100)])) * 0.5 if n > 1 else closes[0] * 0.01
    opens = closes * (1 + rng.uniform(-0.005, 0.005, n))
    highs = np.maximum(opens, closes) + rng.uniform(0, spread, n)
    lows  = np.minimum(opens, closes) - rng.uniform(0, spread, n)

    df = pd.DataFrame({
        "date":          dates[:n],
        "open":          np.round(opens, 2),
        "high":          np.round(highs, 2),
        "low":           np.round(lows, 2),
        "close":         np.round(closes, 2),
        "adj_close":     np.round(closes, 2) if has_adj_close else np.round(closes, 2),
        "volume":        rng.integers(100_000, 10_000_000, n).astype(float) if has_volume else np.zeros(n),
        "open_interest": np.cumsum(rng.integers(0, 3000, n)).clip(0).astype(float) if has_oi else np.zeros(n),
    })
    return df


def _make_intraday_df(rng, eod_df: pd.DataFrame, n_bars: int = 375,
                       has_volume: bool = True, has_oi: bool = False) -> pd.DataFrame:
    """Build canonical INTRADAY DataFrame from EOD data subset."""
    all_bars = []
    for _, row in eod_df.iterrows():
        day = pd.Timestamp(row["date"])
        o, c = row["open"], row["close"]
        day_ret = (c - o) / o if o != 0 else 0
        bar_ret = day_ret / n_bars
        noise = rng.normal(0, abs(day_ret) / np.sqrt(n_bars) + 0.0002, n_bars)
        prices = o * np.cumprod(1 + bar_ret + noise)
        prices[-1] = c
        sp = np.abs(prices) * 0.001
        highs = prices + rng.uniform(0, 1, n_bars) * sp
        lows  = prices - rng.uniform(0, 1, n_bars) * sp
        opens = np.roll(prices, 1); opens[0] = o

        ts_start = day + pd.Timedelta(hours=9, minutes=15)
        timestamps = pd.date_range(ts_start, periods=n_bars, freq="1min")

        bars = pd.DataFrame({
            "datetimestamp":  timestamps,
            "open":          np.round(opens, 2),
            "high":          np.round(highs, 2),
            "low":           np.round(lows, 2),
            "close":         np.round(prices, 2),
            "adj_close":     np.round(prices, 2),
            "volume":        rng.integers(500, 50_000, n_bars).astype(float) if has_volume else np.zeros(n_bars),
            "open_interest": rng.integers(0, 10_000, n_bars).astype(float) if has_oi else np.zeros(n_bars),
        })
        all_bars.append(bars)
    if not all_bars:
        return pd.DataFrame()
    return pd.concat(all_bars, ignore_index=True)


def _make_option_eod(rng, base_eod: pd.DataFrame, opt_type: str,
                      expiry: pd.Timestamp) -> pd.DataFrame:
    """Option EOD from underlying. adj_close=close, has OI."""
    df = base_eod[pd.to_datetime(base_eod["date"]) <= expiry].copy()
    if df.empty:
        return df
    strike = round(df["close"].iloc[0], -1)
    if opt_type == "CE":
        intrinsic = np.maximum(df["close"] - strike, 0)
    else:
        intrinsic = np.maximum(strike - df["close"], 0)
    moneyness = np.abs((df["close"] - strike) / strike)
    time_val = np.maximum(0.5 - moneyness, 0.01) * strike * 0.05
    premium = intrinsic + time_val * rng.uniform(0.8, 1.2, len(df))
    df["open"]          = np.round(premium * rng.uniform(0.95, 1.05, len(df)), 2)
    df["high"]          = np.round(premium * rng.uniform(1.0, 1.15, len(df)), 2)
    df["low"]           = np.round(premium * rng.uniform(0.85, 1.0, len(df)), 2)
    df["close"]         = np.round(premium, 2)
    df["adj_close"]     = df["close"]
    df["volume"]        = rng.integers(1_000, 500_000, len(df)).astype(float)
    df["open_interest"] = np.cumsum(rng.integers(0, 2000, len(df))).clip(0).astype(float)
    return df


def _make_futures_eod(rng, base_eod: pd.DataFrame,
                       expiry: pd.Timestamp) -> pd.DataFrame:
    """Futures EOD from underlying. adj_close=close, has OI."""
    df = base_eod[pd.to_datetime(base_eod["date"]) <= expiry].copy()
    if df.empty:
        return df
    carry = rng.uniform(1.003, 1.015)
    for c in ["open", "high", "low", "close"]:
        df[c] = np.round(df[c] * carry, 2)
    df["adj_close"]     = df["close"]
    df["volume"]        = rng.integers(5_000, 2_000_000, len(df)).astype(float)
    df["open_interest"] = np.cumsum(rng.integers(0, 5000, len(df))).clip(0).astype(float)
    return df


# ── Variation / Issues ───────────────────────────────────────────────────────

def _source_variation(rng, df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    mag = {"dhan": 0.0002, "kite": 0.0001, "upstox": 0.0003}.get(source, 0.0002)
    noise = rng.uniform(-mag, mag, len(df))
    for c in ["open", "high", "low", "close", "adj_close"]:
        if c in df.columns:
            df[c] = np.round(df[c].astype(float) * (1 + noise), 2)
    return df


def _exchange_spread(rng, df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    if exchange == "NSE":
        return df
    df = df.copy()
    sp = rng.uniform(-0.0005, 0.0005, len(df))
    for c in ["open", "high", "low", "close", "adj_close"]:
        if c in df.columns:
            df[c] = np.round(df[c].astype(float) * (1 + sp), 2)
    return df


def _inject_issues(rng, df: pd.DataFrame, rate: float = 0.015) -> pd.DataFrame:
    n = len(df)
    if n < 20:
        return df
    n_issues = max(1, int(n * rate))
    idxs = rng.choice(df.index[1:-1], size=min(n_issues, n - 2), replace=False)
    for i, idx in enumerate(idxs):
        t = i % 7
        if t == 0 and "close" in df.columns:
            df.at[idx, "close"] = np.nan
        elif t == 1 and "high" in df.columns:
            df.at[idx, "high"], df.at[idx, "low"] = df.at[idx, "low"], df.at[idx, "high"]
        elif t == 2 and "volume" in df.columns:
            df.at[idx, "volume"] = float(df["volume"].mean() * 50)
        elif t == 3:
            df = pd.concat([df, df.iloc[[idx]].copy()], ignore_index=True)
        elif t == 4 and "open" in df.columns:
            df.at[idx, "open"] = -abs(float(df.at[idx, "open"]))
        elif t == 5 and "volume" in df.columns:
            df.at[idx, "volume"] = 0.0
        elif t == 6 and "close" in df.columns:
            prev = max(0, idx - 1)
            for c in ["open", "high", "low", "close"]:
                if c in df.columns:
                    df.at[idx, c] = df.at[prev, c]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(data_dir: Path, config_path: Path | None = None,
                  seed: int = DEFAULT_SEED, symbols_filter: list | None = None,
                  mode: str = "Both", verbose: bool = True,
                  max_intraday_days: int = 50) -> dict:
    """Generate all synthetic data. Returns summary dict."""
    rng = np.random.default_rng(seed)
    cfg_path = config_path or (_SCRIPT_DIR / "config" / "instruments.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    sym_cfg = cfg.get("symbols", {})
    gen_cfg = cfg.get("generation", {})
    pc_cfg  = cfg.get("product_classes", {})

    syms = list(sym_cfg.keys())
    if symbols_filter:
        syms = [s for s in syms if s in symbols_filter]

    eod_days = _trading_days(gen_cfg.get("eod_start", "2015-01-01"),
                              gen_cfg.get("eod_end", "2025-12-31"))
    int_days = _trading_days(gen_cfg.get("intraday_start", "2024-01-01"),
                              gen_cfg.get("intraday_end", "2025-12-31"))
    expiries = _monthly_expiries(gen_cfg.get("eod_start", "2015-01-01"),
                                  gen_cfg.get("eod_end", "2025-12-31"))
    issue_eod = gen_cfg.get("dq_issue_rate_eod", 0.015)
    issue_int = gen_cfg.get("dq_issue_rate_intraday", 0.01)

    summary = {"files": 0, "total_rows": 0, "symbols": [], "errors": []}

    for sym in syms:
        sc = sym_cfg[sym]
        exchanges  = sc.get("exchanges", ["NSE", "BSE"])
        prod_cls   = sc.get("product_classes", ["EQUITY"])
        base_price = sc.get("base_price", 1000)
        vol        = sc.get("volatility", 0.20)

        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  {sym}  ({sc.get('type','?')})  exch={exchanges}  products={prod_cls}")

        # Base EOD close prices
        n_eod = len(eod_days)
        closes = _gen_eod_base(rng, n_eod, base_price, vol)

        for pc in prod_cls:
            pc_info = pc_cfg.get(pc, {})
            has_vol = pc_info.get("has_volume", True)
            has_adj = pc_info.get("has_adj_close", True)
            has_oi  = pc_info.get("has_open_interest", False)

            # Build base EOD for this product class
            if pc in ("EQUITY", "ETF", "INDEX_EQUITY"):
                base_eod = _make_eod_df(rng, eod_days, closes, has_vol, has_adj, has_oi)
            elif pc in ("EQUITY_OPT_CE", "INDEX_OPT_CE"):
                recent_exp = [e for e in expiries if e >= eod_days[-100]]
                exp = recent_exp[0] if recent_exp else expiries[-1]
                tmp = _make_eod_df(rng, eod_days, closes, True, True, False)
                base_eod = _make_option_eod(rng, tmp, "CE", exp)
            elif pc in ("EQUITY_OPT_PE", "INDEX_OPT_PE"):
                recent_exp = [e for e in expiries if e >= eod_days[-100]]
                exp = recent_exp[0] if recent_exp else expiries[-1]
                tmp = _make_eod_df(rng, eod_days, closes, True, True, False)
                base_eod = _make_option_eod(rng, tmp, "PE", exp)
            elif pc in ("EQUITY_FUT", "INDEX_FUT"):
                recent_exp = [e for e in expiries if e >= eod_days[-100]]
                exp = recent_exp[0] if recent_exp else expiries[-1]
                tmp = _make_eod_df(rng, eod_days, closes, True, True, False)
                base_eod = _make_futures_eod(rng, tmp, exp)
            else:
                base_eod = _make_eod_df(rng, eod_days, closes, has_vol, has_adj, has_oi)

            if base_eod.empty:
                continue

            for exch in exchanges:
                for src in SOURCES:
                    timeframes = []
                    if mode in ("Both", "EOD"):
                        timeframes.append("eod")
                    if mode in ("Both", "Intraday"):
                        timeframes.append("intraday")

                    for tf in timeframes:
                        try:
                            if tf == "eod":
                                df = base_eod.copy()
                            else:
                                # Use last N days of EOD for intraday
                                int_eod = base_eod[base_eod["date"].isin(int_days)].tail(max_intraday_days)
                                df = _make_intraday_df(rng, int_eod, 375, has_vol, has_oi)
                                if df.empty:
                                    continue

                            df = _exchange_spread(rng, df, exch)
                            df = _source_variation(rng, df, src)
                            df = _inject_issues(rng, df, issue_eod if tf == "eod" else issue_int)

                            # Sort
                            sort_col = "date" if tf == "eod" else "datetimestamp"
                            df = df.sort_values(sort_col).reset_index(drop=True)

                            # Write: data_dir/{eod|intraday}/{source}/{exchange}/{product_class}/{Symbol}.csv
                            out_dir = data_dir / tf / src / exch / pc
                            out_dir.mkdir(parents=True, exist_ok=True)
                            fpath = out_dir / f"{sym}.csv"
                            df.to_csv(fpath, index=False)
                            summary["files"] += 1
                            summary["total_rows"] += len(df)
                            if verbose:
                                ref = f"{sym}_{src.upper()}_{exch}_{tf.upper()}_{pc}"
                                print(f"    ✅ {ref}  ({len(df):,} rows)")

                        except Exception as exc:
                            err = f"{sym}/{src}/{exch}/{tf}/{pc}: {exc}"
                            summary["errors"].append(err)
                            if verbose:
                                print(f"    ❌ {err}")

        summary["symbols"].append(sym)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DDQ v12 — Synthetic Data Generator")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--mode", choices=["EOD", "Intraday", "Both"], default="Both")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-intraday-days", type=int, default=50)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _SCRIPT_DIR / "data" / "synthetic"

    print(f"\n{'='*60}")
    print(f"  DDQ Synthetic Data Generator v12")
    print(f"  Output: {data_dir}")
    print(f"  Mode:   {args.mode}  |  Seed: {args.seed}")
    print(f"{'='*60}")

    summary = generate_all(data_dir, Path(args.config) if args.config else None,
                            args.seed, args.symbols, args.mode, not args.quiet,
                            args.max_intraday_days)

    print(f"\n{'='*60}")
    print(f"  ✅ Generation complete")
    print(f"  Symbols: {len(summary['symbols'])}")
    print(f"  Files:   {summary['files']}")
    print(f"  Rows:    {summary['total_rows']:,}")
    if summary["errors"]:
        print(f"  Errors:  {len(summary['errors'])}")
        for e in summary["errors"][:5]:
            print(f"    ⚠ {e}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
