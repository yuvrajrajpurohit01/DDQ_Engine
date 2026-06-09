"""
DDQ Engine v12 — Data Loader
downloaded_data_dq/ingestion/loader.py

Loads CSV data from the canonical folder structure:
  data/{raw|synthetic}/{eod|intraday}/{source}/{exchange}/{product_class}/{Symbol}.csv

Returns data dict compatible with DQContext:
  data[tf][exchange][source] = DataFrame  (primary spot product)
  data["products"][tf][exchange][source][product_class] = DataFrame  (all products)

Pre-validates each file with Step 0 checks before loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from downloaded_data_dq.ingestion.normaliser import (
    normalise_eod, normalise_intraday,
    pre_validate, apply_pre_validation_fixes,
    EOD_COLUMNS, INTRADAY_COLUMNS,
)

logger = logging.getLogger(__name__)

SOURCES = ["dhan", "kite", "upstox"]
#SOURCES = ["upstox", "kite", "dhan"]
EXCHANGES = ["NSE", "BSE"]

# Primary spot product class per symbol type
PRIMARY_PRODUCT = {
    "equity": "EQUITY",
    "etf":    "ETF",
    "index":  "INDEX_EQUITY",
}

NA_VALUES = ["", "NA", "N/A", "null", "NULL", "None", "none", "-", "--", "#N/A", "nan", "NaN"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA AVAILABILITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataAvailability:
    symbol: str = ""
    found: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    pre_validation: list = field(default_factory=list)

    def record_found(self, source, exchange, tf, product_class, n_rows, path):
        self.found.append({
            "source": source, "exchange": exchange, "timeframe": tf,
            "product_class": product_class, "rows": n_rows, "path": str(path),
            "ref": f"{self.symbol}_{source.upper()}_{exchange}_{tf.upper()}_{product_class}",
        })

    def record_missing(self, source, exchange, tf, product_class, reason):
        self.missing.append({
            "source": source, "exchange": exchange, "timeframe": tf,
            "product_class": product_class, "reason": reason,
            "ref": f"{self.symbol}_{source.upper()}_{exchange}_{tf.upper()}_{product_class}",
        })

    def record_prevalidation(self, ref, result):
        self.pre_validation.append({"ref": ref, **result})

    def log_summary(self):
        logger.info("  Data availability for %s:", self.symbol)
        logger.info("    Found: %d files  |  Missing: %d", len(self.found), len(self.missing))
        for f in self.found:
            logger.info("      ✅  %s  (%d rows)", f["ref"], f["rows"])
        for m in self.missing[:5]:
            logger.info("      ⬜  %s  (%s)", m["ref"], m["reason"])
        if len(self.missing) > 5:
            logger.info("      ... and %d more missing", len(self.missing) - 5)


# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT CONFIG RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_instruments_config(config: dict) -> dict:
    """Extract instruments config from main config or instruments.yaml."""
    # Try from loaded config
    if "symbols" in config:
        return config
    # Try loading instruments.yaml directly
    from pathlib import Path as _P
    cfg_path = _P.home() / "downloaded_data_dq" / "config" / "instruments.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def _get_symbol_info(symbol: str, config: dict) -> dict:
    """Get symbol config from instruments.yaml."""
    instr_cfg = _load_instruments_config(config)
    return instr_cfg.get("symbols", {}).get(symbol, {})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_symbol(
    symbol: str,
    data_dir: Path | str,
    config: dict,
    exchanges: list[str] | None = None,
    run_mode: str = "Both",
) -> tuple[dict, DataAvailability]:
    """
    Load and normalise all data for one symbol.

    Folder structure expected:
      data_dir/{eod|intraday}/{source}/{exchange}/{product_class}/{Symbol}.csv

    Returns:
        (data_dict, DataAvailability)

    data_dict structure:
      data["eod"]["BSE"]["dhan"] = DataFrame (primary spot product)
      data["intraday"]["NSE"]["kite"] = DataFrame
      data["products"]["eod"]["BSE"]["dhan"]["EQUITY_OPT_CE"] = DataFrame
    """

    data_dir = Path(data_dir)
    sym_info = _get_symbol_info(symbol, config)

    sym_type = sym_info.get("type", "equity")
    sym_exchanges = exchanges or sym_info.get("exchanges", EXCHANGES)
    product_classes = sym_info.get("product_classes", [PRIMARY_PRODUCT.get(sym_type, "EQUITY")])
    primary_pc = PRIMARY_PRODUCT.get(sym_type, product_classes[0])

    avail = DataAvailability(symbol=symbol)


    # Build data dict
    data: dict = {
        "eod":      {exch: {src: None for src in SOURCES} for exch in sym_exchanges},
        "intraday": {exch: {src: None for src in SOURCES} for exch in sym_exchanges},
        "products": {
            "eod":      {exch: {src: {} for src in SOURCES} for exch in sym_exchanges},
            "intraday": {exch: {src: {} for src in SOURCES} for exch in sym_exchanges},
        },
    }

    timeframes = []
    if run_mode in ("EOD", "Both"):
        timeframes.append("eod")
    if run_mode in ("Intraday", "Both"):
        timeframes.append("intraday")

    data["_primary_product"] = primary_pc
    logger.info("  Loading %s  type=%s  exchanges=%s  products=%s",
                symbol, sym_type, sym_exchanges, product_classes)

    for tf in timeframes:
        for exch in sym_exchanges:
            for src in SOURCES:
                for pc in product_classes:
                    filepath = data_dir / tf / src / exch / pc / f"{symbol}.csv"
                    ref = f"{symbol}_{src.upper()}_{exch}_{tf.upper()}_{pc}"



                    if not filepath.exists():
                        avail.record_missing(src, exch, tf, pc,
                                              f"file not found: {filepath}")
                        continue

                    try:
                        raw_df = pd.read_csv(filepath, low_memory=False,
                                              na_values=NA_VALUES, keep_default_na=True)


                    except Exception as exc:
                        avail.record_missing(src, exch, tf, pc, f"CSV read error: {exc}")
                        continue

                    if raw_df.empty:
                        avail.record_missing(src, exch, tf, pc, "CSV is empty")
                        continue

                    # Normalise
                    if tf == "eod":
                        df = normalise_eod(raw_df, ref)
                    else:
                        df = normalise_intraday(raw_df, ref)

                    if df.empty:
                        avail.record_missing(src, exch, tf, pc, "normalisation returned empty")
                        continue

                    # Pre-validate (Step 0)
                    pv_result = pre_validate(df, tf, ref)
                    avail.record_prevalidation(ref, pv_result)

                    # Apply pre-validation fixes
                    df = apply_pre_validation_fixes(df, tf)

                    # Add metadata columns
                    df["source"]        = src
                    df["exchange"]      = exch
                    df["product_class"] = pc
                    df["symbol"]        = symbol

                    # Store in products dict
                    data["products"][tf][exch][src][pc] = df

                    # If this is the primary spot product, also store in top-level
                    if pc == primary_pc:
                        data[tf][exch][src] = df

                    avail.record_found(src, exch, tf, pc, len(df), filepath)

    avail.log_summary()
    return data, avail


def load_config(config_dir: str | Path | None = None) -> dict:
    """Load all config YAML files into a single dict."""
    from pathlib import Path as _P
    cfg_dir = _P(config_dir) if config_dir else _P.home() / "downloaded_data_dq" / "config"
    config = {}
    for yaml_file in sorted(cfg_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data:
                    # Use filename stem as key, but also merge top-level keys
                    stem = yaml_file.stem
                    config[stem] = data
                    # Also merge top-level keys for backward compat
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if k not in config:
                                config[k] = v
        except Exception as exc:
            logger.warning("Error loading %s: %s", yaml_file, exc)
    return config
