"""
DDQ Engine v12 — Final Clean Merged Series Engine
downloaded_data_dq/merge/engine.py

For each Symbol + Timeframe:
  1. Use best-source selections from rolling DQ analysis (Phase 3)
  2. For each time window, extract data from the winning source+exchange
  3. Stitch windows into one continuous series with provenance columns
  4. Write to data/final/{RUN_ID}/{eod|intraday}/{product_class}/{Symbol}.csv

Provenance columns added to final series:
  _source     : broker the row came from (dhan/kite/upstox)
  _exchange   : exchange (NSE/BSE)
  _window     : time window label (e.g. Q1-2024)
  _dq_score   : composite DQ score that earned selection
  _merged_at  : timestamp of merge operation
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Primary product class per symbol type
_PRIMARY_PRODUCT = {
    "equity": "EQUITY",
    "etf":    "ETF",
    "index":  "INDEX_EQUITY",
}


def run_merge(
    data_store: dict,
    rolling_results: dict,
    best_sources: dict,
    config: dict,
    project_root: Path,
    run_id: str,
    mode: str = "Both",
    merge_freq: str = "quarterly",
    verbose: bool = True,
) -> dict:
    """
    Create final merged series from best-source selections.

    Returns:
        {
          "files_written": int,
          "total_rows": int,
          "symbols": [...],
          "details": [{symbol, timeframe, product_class, rows, path, windows_used}, ...],
          "output_dir": str,
          "merge_freq": str,
        }
    """
    t0 = time.time()
    out_dir = project_root / "data" / "final" / run_id
    merged_at = datetime.now().isoformat()

    result = {
        "files_written": 0,
        "total_rows": 0,
        "symbols": [],
        "details": [],
        "output_dir": str(out_dir),
        "merge_freq": merge_freq,
    }

    if not best_sources:
        logger.warning("  No best-source selections — skipping merge")
        return result

    instr_cfg = config.get("instruments", config)
    sym_cfg = instr_cfg.get("symbols", {})
    rolling_scores = rolling_results.get("scores", {})

    if verbose:
        logger.info("\n" + "═" * 70)
        logger.info("  FINAL CLEAN MERGED SERIES — merge_freq=%s", merge_freq)
        logger.info("═" * 70)

    for symbol, sym_sel in best_sources.items():
        s_cfg = sym_cfg.get(symbol, {})
        sym_type = s_cfg.get("type", "equity")
        primary_pc = _PRIMARY_PRODUCT.get(sym_type, "EQUITY")
        all_pcs = s_cfg.get("product_classes", [primary_pc])
        sym_data = data_store.get(symbol, {})

        if not sym_data:
            continue

        # Get selections for the merge frequency
        # Fall back: merge_freq → yearly → all
        freq_sel = sym_sel.get(merge_freq, [])
        if not freq_sel:
            freq_sel = sym_sel.get("yearly", [])
        if not freq_sel:
            freq_sel = sym_sel.get("all", [])
        if not freq_sel:
            if verbose:
                logger.info("  ⚠  %s: No selections for any frequency — skipping", symbol)
            continue

        timeframes = []
        if mode in ("Both", "EOD"):
            timeframes.append("EOD")
        if mode in ("Both", "Intraday"):
            timeframes.append("INTRADAY")

        # Iterate ALL products for this symbol
        for pc in all_pcs:
            # Group selections by timeframe, filtered to this product
            by_tf: dict[str, list] = {}
            for sel in freq_sel:
                sel_pc = sel.get("product_class", primary_pc)
                tf = sel.get("timeframe", "EOD")
                if sel_pc == pc:
                    by_tf.setdefault(tf, []).append(sel)

            # If no product-specific selections, fall back to primary selections
            if not by_tf and pc == primary_pc:
                for sel in freq_sel:
                    if not sel.get("product_class"):
                        tf = sel.get("timeframe", "EOD")
                        by_tf.setdefault(tf, []).append(sel)

            for tf in timeframes:
                tf_sels = by_tf.get(tf, [])
                if not tf_sels:
                    continue

                # Sort windows chronologically by label
                tf_sels.sort(key=lambda s: s.get("window_label", ""))


                # Build the merged series — use product-specific data
                merged_df = _build_merged_series(
                    sym_data, tf, tf_sels, symbol, merged_at,
                    rolling_scores.get(symbol, {}), merge_freq,
                    product_class=pc,
                )

                if merged_df is None or merged_df.empty:
                    if verbose:
                        logger.info("  ⚠  %s/%s/%s: Empty merged series — skipping", symbol, tf, pc)
                    continue

                # Write output
                tf_dir = out_dir / tf.lower() / pc
                tf_dir.mkdir(parents=True, exist_ok=True)
                fpath = tf_dir / f"{symbol}.csv"
                merged_df.to_csv(fpath, index=False)

                n_windows = len(set(merged_df["_window"])) if "_window" in merged_df.columns else 0
                n_sources = len(set(merged_df["_source"])) if "_source" in merged_df.columns else 0

                detail = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "product_class": pc,
                    "rows": len(merged_df),
                    "path": str(fpath.relative_to(project_root)),
                    "windows_used": n_windows,
                    "sources_used": n_sources,
                    "ref": f"{symbol}_{tf}_{pc}",
                }
                result["details"].append(detail)
                result["files_written"] += 1
                result["total_rows"] += len(merged_df)

                if verbose:
                    logger.info("  ✅ %s/%s/%s: %d rows from %d windows (%d sources) → %s",
                               symbol, tf, pc, len(merged_df), n_windows, n_sources,
                               fpath.relative_to(project_root))

        result["symbols"].append(symbol)

    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed

    if verbose:
        logger.info("\n  Merge complete: %d files, %d total rows in %.1fs",
                    result["files_written"], result["total_rows"], elapsed)
        logger.info("  Output: %s", out_dir)

    return result


def _build_merged_series(
    sym_data: dict,
    timeframe: str,
    selections: list[dict],
    symbol: str,
    merged_at: str,
    sym_rolling_scores: dict,
    merge_freq: str,
    product_class: str = "",
) -> pd.DataFrame | None:
    """
    Build one merged DataFrame by stitching best-source data per window.

    For each window selection:
      1. Find the source+exchange data
      2. Slice to window dates
      3. Tag with provenance columns
    Then concatenate, deduplicate, sort.
    """
    tf_key = timeframe.lower()  # "eod" or "intraday"
    ts_col = "date" if tf_key == "eod" else "datetimestamp"
    tf_data = sym_data.get(tf_key, {})

    all_slices = []

    for sel in selections:
        src = sel.get("best_source", "")
        exch = sel.get("best_exchange", "")
        w_label = sel.get("window_label", "")
        dq_score = sel.get("best_score", 0)

        # Find the DataFrame for this source+exchange
        df = tf_data.get(exch, {}).get(src)
        if df is None or df.empty:
            continue

        # Determine window date range from rolling scores
        w_start, w_end = _find_window_range(
            sym_rolling_scores, merge_freq, w_label, timeframe, src, exch
        )

        # Slice data to window period
        if w_start and w_end:
            slc = _slice_to_window(df, ts_col, w_start, w_end)
        else:
            # Fallback: use all data (for "ALL" window)
            slc = df.copy()

        if slc.empty:
            continue

        # Add provenance columns
        slc = slc.copy()
        slc["_source"] = src
        slc["_exchange"] = exch
        slc["_window"] = w_label
        slc["_window_freq"] = merge_freq
        slc["_window_start"] = w_start or ""
        slc["_window_end"] = w_end or ""
        slc["_dq_score"] = round(dq_score, 4)
        slc["_merged_at"] = merged_at

        all_slices.append(slc)

    if not all_slices:
        return None

    # Concatenate all window slices
    merged = pd.concat(all_slices, ignore_index=True)

    # Parse timestamps for sorting and dedup
    if ts_col in merged.columns:
        merged[ts_col] = pd.to_datetime(merged[ts_col], errors="coerce")
        # Deduplicate: keep first occurrence (from earlier/higher-priority window)
        merged = merged.drop_duplicates(subset=[ts_col], keep="first")
        # Sort chronologically
        merged = merged.sort_values(ts_col).reset_index(drop=True)

    return merged


def _find_window_range(
    sym_rolling_scores: dict,
    freq: str,
    window_label: str,
    timeframe: str,
    source: str,
    exchange: str,
) -> tuple:
    """Look up window start/end dates from rolling scores."""
    freq_scores = sym_rolling_scores.get(freq, [])
    for s in freq_scores:
        if (s.get("window_label") == window_label and
            s.get("timeframe") == timeframe and
            s.get("source") == source and
            s.get("exchange") == exchange):
            return s.get("window_start"), s.get("window_end")
    return None, None


def _slice_to_window(
    df: pd.DataFrame,
    ts_col: str,
    w_start: str,
    w_end: str,
) -> pd.DataFrame:
    """Slice DataFrame to a time window."""
    if ts_col not in df.columns:
        return pd.DataFrame()

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    start = pd.Timestamp(w_start)
    end = pd.Timestamp(w_end)

    if ts_col == "datetimestamp":
        # For intraday, include the full end day
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = (ts >= start) & (ts <= end)
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PROVENANCE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarize_provenance(merge_result: dict) -> dict:
    """
    Build a provenance summary for the dashboard.

    Returns per-symbol provenance map showing which source+exchange
    was used for each time window.
    """
    provenance: dict = {}

    for detail in merge_result.get("details", []):
        sym = detail["symbol"]
        tf = detail["timeframe"]
        fpath = detail.get("path", "")


        # Read back the merged file to extract provenance
        try:
            from pathlib import Path as _P
            full_path = Path.cwd() / fpath
            if full_path.exists():
                df = pd.read_csv(full_path, usecols=[
                    "_source", "_exchange", "_window", "_window_freq",
                    "_window_start", "_window_end", "_dq_score"],
                                  low_memory=False)

                # Aggregate by window
                windows = []
                for w_label, grp in df.groupby("_window", sort=False):
                    src = grp["_source"].iloc[0]
                    exch = grp["_exchange"].iloc[0]
                    score = grp["_dq_score"].iloc[0]
                    windows.append({
                        "window": w_label,
                        "source": src,
                        "exchange": exch,
                        "dq_score": round(float(score), 4),
                        "rows": len(grp),
                    })


                key = f"{sym}/{detail.get('product_class', '')}/{tf}"

                provenance[key] = {
                    "symbol": sym,
                    "timeframe": tf,
                    "product_class": detail.get("product_class", ""),
                    "total_rows": detail["rows"],
                    "windows": windows,
                    "n_windows": len(windows),
                    "n_sources": len(set(w["source"] for w in windows)),
                    "path": fpath,
                }

        except Exception as exc:
            logger.warning("  Provenance read error for %s/%s: %s", sym, tf, exc)

    return provenance
