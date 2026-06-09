"""
Downloaded Data DQ Engine — Config Loader
downloaded_data_dq/utils/config_loader.py

Loads and merges all YAML config files into a single flat dict
that is passed to DQContext at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import csv
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default config directory (relative to project root)
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    """Load a single YAML file; return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s", path)
        return {}
    except yaml.YAMLError as exc:
        logger.error("YAML parse error in %s: %s", path, exc)
        return {}

def _csv_to_symbols(csv_path: Path, instruments_yaml: dict) -> dict:
    """
    Read instruments.csv and return a dict in the same shape as
    instruments.yaml → symbols: { SYMBOL: {...}, ... }

    CSV columns (13):
        Symbol, display_name, instrument_type, listing_date,
        PAID UP VALUE, lot_size, ISIN, price_band_pct, tick_size,
        file_prefix_bse, file_prefix_nse, nse_symbol, bse_symbol

    Fields NOT in CSV — derived from instruments.yaml generation block
    or hard-coded defaults:
        exchanges, fno, product_classes, base_price, volatility,
        sector, bse_code, eod_data_range
    """
    if not csv_path.exists():
        logger.warning("instruments.csv not found at %s — falling back to YAML symbols", csv_path)
        return instruments_yaml.get("symbols", {})

    gen = instruments_yaml.get("generation", {})
    eod_start = gen.get("eod_start", "2005-01-01")
    eod_end   = gen.get("eod_end",   "2025-12-31")
    eod_range = f"{eod_start} to {eod_end}"

    df = pd.read_csv(csv_path, dtype=str)

    # Strip accidental whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    # Rename the oddly-named column
    if "PAID UP VALUE" in df.columns:
        df = df.rename(columns={"PAID UP VALUE": "paid_up_value"})

    symbols: dict[str, Any] = {}

    for _, row in df.iterrows():
        sym = row["Symbol"]

        # ── Derive instrument type ──────────────────────────────────────
        raw_type      = row.get("instrument_type", "Equity")
        type_lower    = raw_type.lower()            # "equity" / "etf" / "index"
        instrument_lbl = raw_type                   # "Equity" / "Etf" / "Index"

        # ── Derive exchanges ────────────────────────────────────────────
        # Both nse_symbol and bse_symbol columns are present for all rows;
        # treat non-empty values as "listed on that exchange"
        nse_sym = row.get("nse_symbol", "").strip()
        bse_sym = row.get("bse_symbol", "").strip()
        exchanges = []
        if nse_sym:
            exchanges.append("NSE")
        if bse_sym:
            exchanges.append("BSE")
        if not exchanges:
            exchanges = ["NSE", "BSE"]   # safe default

        # ── Product classes by type ─────────────────────────────────────
        pc=row.get("product_classes").strip()
        print(pc)
        product_classes = pc.split(",")

        """
        if type_lower == "equity":
            product_classes = ["EQUITY"]
        elif type_lower == "etf":
            product_classes = ["ETF"]
        elif type_lower == "index":
            product_classes = ["INDEX_EQUITY"]
        else:
            product_classes = ["EQUITY"]
        """
        # ── Numeric fields — coerce safely ──────────────────────────────
        try:
            lot_size = int(float(row.get("lot_size", 1)))
        except (ValueError, TypeError):
            lot_size = 1

        try:
            tick_size = float(row.get("tick_size", 0.05))
        except (ValueError, TypeError):
            tick_size = 0.05

        try:
            price_band_pct = float(row.get("price_band_pct", 20.0))
        except (ValueError, TypeError):
            price_band_pct = 20.0

        # ── listing_date — normalise DD-Mon-YY → YYYY-MM-DD ────────────
        raw_date = row.get("listing_date", "")
        try:
            listing_date = pd.to_datetime(raw_date, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            listing_date = "2000-01-01"

        symbols[sym] = {
            "name":             row.get("display_name", sym),
            "type":             type_lower,
            "exchanges":        exchanges,
            "fno":              row.get("fno", ""),           # not in CSV; default False
            "product_classes":  product_classes,
            "lot_size":         lot_size,
            "base_price":       row.get("base_price", ""),            # not in CSV
            "volatility":       row.get("volatility", ""),            # not in CSV
            "sector":           row.get("sector", ""),            # not in CSV
            "isin":             row.get("ISIN", ""),
            "listing_date":     listing_date,
            "tick_size":        tick_size,
            "price_band_pct":   price_band_pct,
            "bse_code":         bse_sym,         # CSV has symbol name, not numeric code
            "nse_symbol":       nse_sym,
            "bse_symbol":       bse_sym,
            "display_name":     row.get("display_name", sym),
            "instrument_type":  instrument_lbl,
            "bse_file_prefix":  bse_sym,
            "nse_file_prefix":  nse_sym,
            "eod_data_range":   eod_range,
        }

    logger.debug("Loaded %d symbols from %s", len(symbols), csv_path)
    return symbols

def load_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """
    Load all YAML config files and return a merged config dict.

    Structure of returned dict:
    {
        "sources":          { ... },   # from sources.yaml
        "thresholds":       { ... },   # from thresholds.yaml
        "instruments":      { ... },   # from instruments.yaml (+ compat bridge)
        "trading_calendar": { ... },   # from trading_calendar.yaml
        "rectification":    { ... },   # from rectification.yaml
    }
    """
    cfg_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

    config: dict[str, Any] = {
        "sources":          _load_yaml(cfg_dir / "sources.yaml"),
        "thresholds":       _load_yaml(cfg_dir / "thresholds.yaml"),
        "instruments":      _load_yaml(cfg_dir / "instruments.yaml"),
        "trading_calendar": _load_yaml(cfg_dir / "trading_calendar.yaml"),
        "rectification":    _load_yaml(cfg_dir / "rectification.yaml"),
    }

    instr = config["instruments"]

    # ── Resolve CSV path ─────────────────────────────────────────────────
    csv_path = cfg_dir / "Instruments_final.csv"

    # ── Replace symbols block from CSV (if CSV exists) ───────────────────
    if csv_path.exists():
        instr["symbols"] = _csv_to_symbols(csv_path, instr)
        logger.info(
            "instruments.csv loaded: %d symbols from %s",
            len(instr["symbols"]),
            csv_path,
        )
    else:
        logger.info(
            "No instruments.csv found at %s — using instruments.yaml symbols (%d)",
            csv_path,
            len(instr.get("symbols", {})),
        )

    # ── Backward-compatibility bridge for instruments config ──────────────
    # Existing 248 tests expect: config["instruments"]["equity"]["TCS"] = {...}
    # New instruments.yaml has:  config["instruments"]["symbols"]["TCS"] = {...}
    # Build the old-format lookups from the new-format data.
    #instr = config.get("instruments", {})
    sym_cfg = instr.get("symbols", {})
    if sym_cfg:
        # Create category-based lookups (equity, etf, indices, etc.)
        cat_map: dict[str, dict] = {
            "equity": {}, "etf": {}, "indices": {},
            "equity_futures": {}, "index_futures": {},
            "equity_options": {}, "index_options": {},
        }
        test_symbols = []
        for sym, info in sym_cfg.items():
            stype = info.get("type", "equity")
            cat_key = stype  # "equity", "etf", "index"
            if cat_key == "index":
                cat_key = "indices"
            if cat_key not in cat_map:
                cat_map[cat_key] = {}
            cat_map[cat_key][sym] = info
            test_symbols.append(sym)

            # Also put in "equity" for F&O-capable equities to satisfy
            # tests that always look under "equity"
            if stype in ("equity", "etf", "index"):
                cat_map["equity"][sym] = info

        for cat, syms in cat_map.items():
            if cat not in instr:
                instr[cat] = syms
            else:
                instr[cat].update(syms)

        instr["test_symbols"] = instr.get("demo_symbols", test_symbols)

        # Also add sessions config if missing (tests check for it)
        if "sessions" not in instr:
            gen = instr.get("generation", {})
            instr["sessions"] = {
                "NSE": {
                    "start": gen.get("session_start", "09:15"),
                    "end": gen.get("session_end", "15:30"),
                    "timezone": instr.get("timezone", "Asia/Kolkata"),
                },
                "BSE": {
                    "start": gen.get("session_start", "09:15"),
                    "end": gen.get("session_end", "15:30"),
                    "timezone": instr.get("timezone", "Asia/Kolkata"),
                },
            }

    # Also merge symbols into top-level for easy access
    config["symbols"] = sym_cfg

    logger.debug(
        "Config loaded from %s — sources: %s, instruments: %d symbols",
        cfg_dir,
        list(config["sources"].get("sources", {}).keys()),
        len(sym_cfg),
    )
    return config


def get_source_config(config: dict, source: str) -> dict:
    """Return config block for a specific source (dhan / kite / upstox)."""
    return config.get("sources", {}).get("sources", {}).get(source, {})


def get_instrument_config(config: dict, symbol: str) -> dict:
    """Return config block for a specific instrument symbol."""
    return config.get("instruments", {}).get("instruments", {}).get(symbol, {})


def get_threshold(config: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Fetch a threshold value using a dotted path, e.g.:
        get_threshold(config, "eod.EOD_001_max_null_pct")
        get_threshold(config, "cross_source.SRC_E_price_tolerance_pct")
    """
    parts = dotted_key.split(".")
    node: Any = config.get("thresholds", {})
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
            if node is None:
                return default
        else:
            return default
    return node
