"""
Downloaded Data DQ Engine — Trading Calendar Utility
downloaded_data_dq/utils/trading_calendar_util.py

Two backends — YAML is PRIMARY, pmc is supplementary:
  1. Bundled YAML (primary)  — trading_calendar.yaml in config/
                               Includes Indian special holidays (Ayodhya, elections, etc.)
                               that pmc's XNSE calendar may not carry.
  2. pandas_market_calendars — used only to EXTEND the YAML with dates outside
                               the YAML's explicit coverage range (pre-2020).

Rationale: Indian NSE/BSE have frequent special/one-off market closures that
exchange notifications announce late and pmc may not include. The bundled YAML
is the authoritative source for the dates we have curated.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime, pd.Timestamp]

# ── Internal: try importing pandas_market_calendars ───────────────────────────
try:
    import pandas_market_calendars as mcal  # type: ignore
    _PMC_AVAILABLE = True
    logger.debug("pandas_market_calendars available — used as supplement to YAML.")
except ImportError:
    _PMC_AVAILABLE = False
    logger.info(
        "pandas_market_calendars not installed. "
        "Using bundled YAML calendar only."
    )


# ── YAML fallback calendar ─────────────────────────────────────────────────────
def _load_yaml_holidays(config: dict | None = None) -> set[pd.Timestamp]:
    """
    Load NSE holidays from the config dict (pre-loaded trading_calendar.yaml).
    Returns a set of pd.Timestamp for O(1) lookup.
    """
    if config is None:
        return set()
    holidays_raw = config.get("trading_calendar", {}).get("nse_holidays", [])
    return {pd.Timestamp(h) for h in holidays_raw}


@lru_cache(maxsize=8)
def _pmc_trading_days(start_str: str, end_str: str, exchange: str) -> pd.DatetimeIndex:
    """Cached pandas_market_calendars schedule query."""
    cal = mcal.get_calendar(f"X{exchange}")   # NSE → XNSE
    schedule = cal.schedule(start_date=start_str, end_date=end_str)
    return mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)


def _yaml_trading_days(
    start: pd.Timestamp,
    end: pd.Timestamp,
    holidays: set[pd.Timestamp],
) -> pd.DatetimeIndex:
    """
    Generate trading days from YAML holiday list.
    Mon–Fri minus known holidays.
    """
    all_bdays = pd.bdate_range(start=start, end=end)
    trading = [d for d in all_bdays if d not in holidays]
    return pd.DatetimeIndex(trading)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_trading_days(
    start: DateLike,
    end: DateLike,
    exchange: str = "NSE",
    config: dict | None = None,
) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex of trading days between start and end (inclusive).

    YAML is always PRIMARY — it contains curated Indian market holidays including
    special one-off closures (Ayodhya inauguration, elections, etc.) that
    pandas_market_calendars may not carry.

    For historical dates before 2020 when YAML coverage is sparse, pmc is used
    with YAML holidays applied on top to catch any special closures.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    yaml_holidays = _load_yaml_holidays(config)

    # YAML is authoritative for 2020 onwards (curated special holidays)
    yaml_cutoff = pd.Timestamp("2020-01-01")
    if start_ts >= yaml_cutoff or not _PMC_AVAILABLE:
        return _yaml_trading_days(start_ts, end_ts, yaml_holidays)

    # Pre-2020: use pmc for base calendar then overlay YAML holidays
    try:
        pmc_days = _pmc_trading_days(
            start_ts.strftime("%Y-%m-%d"),
            end_ts.strftime("%Y-%m-%d"),
            exchange,
        )
        return pmc_days[~pmc_days.isin(yaml_holidays)]
    except Exception as exc:
        logger.warning("pmc failed (%s); using YAML only.", exc)
        return _yaml_trading_days(start_ts, end_ts, yaml_holidays)


def is_trading_day(
    dt: DateLike,
    exchange: str = "NSE",
    config: dict | None = None,
) -> bool:
    """Return True if the given date is a trading day."""
    ts = pd.Timestamp(dt).normalize()
    start = end = ts
    trading = get_trading_days(start, end, exchange=exchange, config=config)
    return len(trading) > 0


def count_trading_days(
    start: DateLike,
    end: DateLike,
    exchange: str = "NSE",
    config: dict | None = None,
) -> int:
    """Return the number of trading days between start and end (inclusive)."""
    return len(get_trading_days(start, end, exchange=exchange, config=config))


def missing_trading_days(
    dates: pd.DatetimeIndex | pd.Series,
    start: DateLike | None = None,
    end: DateLike | None = None,
    exchange: str = "NSE",
    config: dict | None = None,
) -> list[pd.Timestamp]:
    """
    Given a set of dates (e.g. from a data file's date column),
    return a list of expected trading days that are absent.

    Args:
        dates    : actual dates present in the data
        start    : start of expected range (defaults to min of dates)
        end      : end of expected range (defaults to max of dates)
        exchange : "NSE" or "BSE"
        config   : loaded config dict

    Returns:
        List of pd.Timestamp representing missing trading days.
    """
    dates_ts = pd.DatetimeIndex(dates).normalize()
    if len(dates_ts) == 0:
        return []

    range_start = pd.Timestamp(start).normalize() if start else dates_ts.min()
    range_end = pd.Timestamp(end).normalize() if end else dates_ts.max()

    expected = get_trading_days(range_start, range_end, exchange=exchange, config=config)
    dates_set = set(dates_ts)
    return [d for d in expected if d not in dates_set]


def extra_trading_days(
    dates: pd.DatetimeIndex | pd.Series,
    start: DateLike | None = None,
    end: DateLike | None = None,
    exchange: str = "NSE",
    config: dict | None = None,
) -> list[pd.Timestamp]:
    """
    Return dates present in the data that are NOT expected trading days.
    (Useful for detecting holiday trading or weekend bars.)
    """
    dates_ts = pd.DatetimeIndex(dates).normalize()
    if len(dates_ts) == 0:
        return []

    range_start = pd.Timestamp(start).normalize() if start else dates_ts.min()
    range_end = pd.Timestamp(end).normalize() if end else dates_ts.max()

    expected_set = set(
        get_trading_days(range_start, range_end, exchange=exchange, config=config)
    )
    return [d for d in sorted(dates_ts.unique()) if d not in expected_set]
