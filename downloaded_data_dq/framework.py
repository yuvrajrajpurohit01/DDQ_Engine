"""
Downloaded Data DQ Engine — Core Framework
downloaded_data_dq/framework.py

Python 3.12.12 | pandas 2.2.x

Provides:
  - DQContext  : per-symbol runtime container passed to every test function
  - TestSpec   : frozen metadata dataclass for a DQ test
  - TestResult : mutable result dataclass populated by each test function
  - dq_test()  : decorator that registers a function as a DQ test
  - run_tests_for_symbol() : orchestrates test execution for one symbol
"""

from __future__ import annotations

import functools
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ── Global test registry ───────────────────────────────────────────────────────
# Maps test_id (str) -> (TestSpec, Callable)
_TESTS: dict[str, tuple["TestSpec", Callable]] = {}


# ══════════════════════════════════════════════════════════════════════════════
# ANSI COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _C:
    RST    = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"
    BLUE   = "\033[34m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"
    ORANGE = "\033[38;5;208m"
    PURPLE = "\033[35m"


# ══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def _classify_cause(test_id: str, details: str) -> str:
    d   = details.lower()
    tid = test_id.upper()
    if "data_not_present" in d or "file not found" in d or "no data" in d:
        return "DATA_MISSING"
    if "stale" in d or tid in ("BST-005", "GOV-003", "BT-008"):
        return "DATA_STALE"
    if tid in ("EOD-018", "BT-023", "INT-011", "EOD-014"):
        return "SYNTHETIC_ARTIFACT"
    if "tbd" in d or "not configured" in d or "instruments.yaml" in d:
        return "CONFIG_MISMATCH"
    if "runtime_error" in d.upper() or "RUNTIME_ERROR" in details:
        return "RUNTIME_ERROR"
    return "DATA_QUALITY"


_CAUSE_ICON = {
    "DATA_MISSING":       "🔴",
    "DATA_STALE":         "🟡",
    "DATA_QUALITY":       "🟠",
    "CONFIG_MISMATCH":    "🔵",
    "SYNTHETIC_ARTIFACT": "⚫",
    "RUNTIME_ERROR":      "💥",
    "OK":                 "✅",
}
_CAUSE_LABEL = {
    "DATA_MISSING":       "Data Missing",
    "DATA_STALE":         "Data Stale",
    "DATA_QUALITY":       "Data Quality Issue",
    "CONFIG_MISMATCH":    "Config Mismatch",
    "SYNTHETIC_ARTIFACT": "Synthetic Artifact",
    "RUNTIME_ERROR":      "Runtime Error",
    "OK":                 "OK",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TestSpec:
    """Immutable metadata for a DQ test. Set once at decoration time."""
    test_id:             str
    name:                str
    layer:               str
    category:            str
    gate_type:           Literal["Hard", "Soft"] = "Soft"
    severity:            Literal["Critical", "High", "Medium", "Low", "Info"] = "Medium"
    weight:              float = 3.0
    description:         str = ""
    method_formula:      str = ""
    success_threshold:   str = ""
    source:              str = "All"
    timeframe:           str = "Both"
    applies_to:          str = "All"
    mode:                str = "Both"


@dataclass
class TestResult:
    """Mutable result produced by one test execution."""
    test_id:        str   = ""
    symbol:         str   = ""
    source:         str   = ""
    exchange:       str   = ""
    product_class:  str   = ""
    timeframe:      str   = ""
    layer:          str   = ""
    category:       str   = ""
    severity:       str   = "Medium"
    gate_type:      str   = "Soft"
    weight:         float = 3.0
    dataset:        str   = ""
    status:         str   = "Redo_As_Results_Unclear"
    success_score:  float = 0.0
    weighted_score: float = 0.0
    details:        str   = ""
    metrics:        dict  = field(default_factory=dict)
    last_run:       str   = field(default_factory=lambda: datetime.now().isoformat())

    def set_pass(self, details: str = "", metrics: dict | None = None) -> None:
        self.status = "Pass"
        self.success_score  = 1.0
        self.weighted_score = self.weight
        self.details  = details
        self.metrics  = metrics or {}
        self.last_run = datetime.now().isoformat()

    def set_fail(self, details: str = "", metrics: dict | None = None) -> None:
        self.status = "Fail"
        self.success_score  = 0.0
        self.weighted_score = 0.0
        self.details  = details
        self.metrics  = metrics or {}
        self.last_run = datetime.now().isoformat()

    def set_redo(self, details: str = "", metrics: dict | None = None) -> None:
        self.status = "Redo_As_Results_Unclear"
        self.success_score  = 0.5
        self.weighted_score = self.weight * 0.5
        self.details  = details
        self.metrics  = metrics or {}
        self.last_run = datetime.now().isoformat()

    def set_skip(self, reason: str = "") -> None:
        self.status = "Skip"
        self.success_score  = 0.0
        self.weighted_score = 0.0
        self.details  = reason
        self.last_run = datetime.now().isoformat()

    def set_data_not_present(
        self, source: str = "", exchange: str = "", timeframe: str = ""
    ) -> None:
        self.status = "Data_Not_Present"
        self.success_score  = 0.0
        self.weighted_score = 0.0
        self.details = (
            f"Data not available: source={source or 'unknown'} "
            f"exchange={exchange or 'unknown'} timeframe={timeframe or 'unknown'}"
        )
        self.last_run = datetime.now().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# DQContext
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DQContext:
    """Runtime context passed to every test function."""
    symbol:   str
    data:     dict[str, Any]
    config:   dict[str, Any]
    run_mode: str = "Both"
    cache:    dict[str, Any] = field(default_factory=dict)

    def eod(self, source: str) -> pd.DataFrame | None:
        """Get EOD DataFrame for a source, searching all exchanges."""
        tf_data = self.data.get("eod", {})
        for exch, src_dict in tf_data.items():
            if isinstance(src_dict, dict):
                df = src_dict.get(source)
                if df is not None and not df.empty:
                    return df
        return None

    def intraday(self, source: str) -> pd.DataFrame | None:
        """Get Intraday DataFrame for a source, searching all exchanges."""
        tf_data = self.data.get("intraday", {})
        for exch, src_dict in tf_data.items():
            if isinstance(src_dict, dict):
                df = src_dict.get(source)
                if df is not None and not df.empty:
                    return df
        return None

    def has_eod(self, source: str) -> bool:
        df = self.eod(source)
        return df is not None and not df.empty

    def has_intraday(self, source: str) -> bool:
        df = self.intraday(source)
        return df is not None and not df.empty

    def threshold(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        cfg = self.config.get("thresholds", {})
        for part in parts:
            if isinstance(cfg, dict):
                cfg = cfg.get(part, default)
            else:
                return default
        return cfg if cfg is not None else default

    def source_flag(self, source: str, flag: str, default: Any = None) -> Any:
        return (
            self.config.get("sources", {})
            .get("sources", {})
            .get(source, {})
            .get(flag, default)
        )


# ══════════════════════════════════════════════════════════════════════════════
# @dq_test DECORATOR
# ══════════════════════════════════════════════════════════════════════════════
def dq_test(spec: TestSpec) -> Callable:
    """
    Decorator that registers a function as a DQ test.

    Usage:
        @dq_test(TestSpec(test_id="EOD-001", name="Null check", layer="EOD", ...))
        def test_eod_001(ctx: DQContext) -> TestResult:
            ...
    """
    if spec.test_id in _TESTS:
        raise ValueError(
            f"Duplicate test_id: '{spec.test_id}'. Each test ID must be unique."
        )

    def decorator(fn: Callable[[DQContext], TestResult]) -> Callable:
        @functools.wraps(fn)
        def wrapper(ctx: DQContext) -> TestResult:
            result = TestResult(
                test_id=spec.test_id, symbol=ctx.symbol,
                source=spec.source, timeframe=spec.timeframe,
                layer=spec.layer, category=spec.category,
                severity=spec.severity, gate_type=spec.gate_type,
                weight=spec.weight, dataset=spec.applies_to,
            )
            try:
                populated = fn(ctx)
                if isinstance(populated, TestResult):
                    if not populated.test_id: populated.test_id = result.test_id
                    if not populated.symbol:  populated.symbol  = result.symbol
                    return populated
                logger.warning("Test %s returned None; recording as Redo.", spec.test_id)
                result.set_redo("Test returned None — check implementation.")
                return result
            except Exception as exc:
                logger.exception("Test %s raised unhandled exception: %s", spec.test_id, exc)
                result.status = "Fail"
                result.success_score = result.weighted_score = 0.0
                result.details  = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
                result.last_run = datetime.now().isoformat()
                return result

        # Store original function name for logging
        wrapper._fn_name = fn.__name__
        _TESTS[spec.test_id] = (spec, wrapper)
        return wrapper

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL LOG HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _sev_col(sev: str) -> str:
    return {
        "Critical": _C.RED + _C.BOLD,
        "High":     _C.ORANGE,
        "Medium":   _C.YELLOW,
        "Low":      _C.BLUE,
        "Info":     _C.GREY,
    }.get(sev, _C.GREY)


def _gate_col(gate: str) -> str:
    return _C.RED if gate == "Hard" else _C.CYAN


def _log(L: logging.Logger | None, msg: str) -> None:
    (L or logger).info(msg)


# ══════════════════════════════════════════════════════════════════════════════
# run_tests_for_symbol
# ══════════════════════════════════════════════════════════════════════════════
def run_tests_for_symbol(
    ctx: DQContext,
    *,
    layer_filter: str | None = None,
    test_ids: list[str] | None = None,
    mode_filter: str | None = None,
    gate_filter: str | None = None,
    ddq_logger: logging.Logger | None = None,
    halt_on_hard: bool = True,
) -> list[TestResult]:
    """
    Run all registered DQ tests for one symbol with detailed per-test logging.

    Each test emits:
      STARTING line  — before execution (test id, severity, gate, function name)
      RESULT  line   — after execution  (status, elapsed, brief details)
      DETAIL lines   — on Fail/Error    (root cause, gate action, warning text)

    Args:
        ctx          : DQContext for the symbol
        layer_filter : only run tests from this layer
        test_ids     : only run these specific test IDs
        mode_filter  : "EOD", "Intraday", or "Both"
        gate_filter  : "Hard" or "Soft"
        ddq_logger   : Logger writing to both terminal and log file

    Returns:
        List of TestResult in registration order.
    """
    L       = ddq_logger
    results: list[TestResult] = []
    IND     = "              │  "    # indentation for continuation lines

    # Build filtered list with sequential run positions
    filtered: list[tuple[int, str, TestSpec, Callable]] = []
    for reg_seq, (tid, (spec, fn)) in enumerate(_TESTS.items(), 1):
        if test_ids     and tid not in test_ids:                          continue
        if layer_filter and spec.layer.upper() != layer_filter.upper():  continue
        if mode_filter  and spec.mode not in ("Both", mode_filter):      continue
        if gate_filter  and spec.gate_type != gate_filter:               continue
        filtered.append((reg_seq, tid, spec, fn))

    total_run = len(filtered)
    total_reg = len(_TESTS)

    for run_pos, (reg_seq, tid, spec, fn) in enumerate(filtered, 1):
        fn_name  = getattr(fn, "_fn_name", fn.__name__)
        sym      = ctx.symbol

        # ── STARTING line ─────────────────────────────────────────────────────
        ts_start = datetime.now().strftime("%H:%M:%S.%f")[:12]
        sev_str  = f"{_sev_col(spec.severity)}{spec.severity:<8}{_C.RST}"
        gate_str = f"{_gate_col(spec.gate_type)}{spec.gate_type:<4}{_C.RST}"
        _log(L,
            f"{_C.DIM}{ts_start}{_C.RST}  "
            f"{_C.GREY}STARTING {_C.RST} "
            f"Test {run_pos:>4}/{total_run}  "
            f"{_C.BOLD}{_C.WHITE}{tid:<14}{_C.RST}  "
            f"[{sev_str}|{gate_str}]  "
            f"[{_C.CYAN}{spec.layer}{_C.RST}]  "
            f"fn={_C.GREY}{fn_name}{_C.RST}  "
            f"symbol={_C.PURPLE}{sym}{_C.RST}"
        )

        # ── Execute ───────────────────────────────────────────────────────────
        t0     = time.perf_counter()
        result = fn(ctx)
        elapsed = time.perf_counter() - t0
        results.append(result)

        status  = result.status
        details = result.details or ""
        brief   = details[:120].replace("\n", " ")
        ts_end  = datetime.now().strftime("%H:%M:%S.%f")[:12]
        elapsed_str = f"{elapsed:.3f}s"

        # ── RESULT line ───────────────────────────────────────────────────────
        if status == "Pass":
            status_str = f"{_C.GREEN}{_C.BOLD}✅ PASS   {_C.RST}"
        elif status == "Fail":
            status_str = f"{_C.RED}{_C.BOLD}❌ FAIL   {_C.RST}"
        elif status == "Skip":
            status_str = f"{_C.GREY}⏭  SKIP   {_C.RST}"
        elif status == "Data_Not_Present":
            status_str = f"{_C.YELLOW}📭 NO DATA{_C.RST}"
        else:  # Redo / Error
            status_str = f"{_C.ORANGE}🔄 REDO   {_C.RST}"

        _log(L,
            f"{_C.DIM}{ts_end}{_C.RST}  "
            f"{status_str}"
            f"Test {run_pos:>4}/{total_run}  "
            f"{_C.BOLD}{_C.WHITE}{tid:<14}{_C.RST}  "
            f"[{sev_str}|{gate_str}]  "
            f"({_C.DIM}{elapsed_str}{_C.RST})  "
            f"{brief}"
        )

        # ── DETAIL lines for Fail / Error ─────────────────────────────────────
        if status == "Fail" or "RUNTIME_ERROR" in details:
            cause = _classify_cause(tid, details)
            cause_icon  = _CAUSE_ICON.get(cause, "❓")
            cause_label = _CAUSE_LABEL.get(cause, cause)
            cause_col   = {
                "DATA_MISSING": _C.RED, "DATA_STALE": _C.YELLOW,
                "DATA_QUALITY": _C.ORANGE, "CONFIG_MISMATCH": _C.BLUE,
                "SYNTHETIC_ARTIFACT": _C.GREY, "RUNTIME_ERROR": _C.RED + _C.BOLD,
            }.get(cause, _C.GREY)

            _log(L, f"{IND}"
                    f"{_C.BOLD}Root Cause  :{_C.RST}  "
                    f"{cause_col}{cause_icon} {cause_label}{_C.RST}")

            _log(L, f"{IND}"
                    f"{_C.BOLD}Severity    :{_C.RST}  "
                    f"{_sev_col(spec.severity)}{spec.severity}{_C.RST}"
                    f"  |  Gate: "
                    f"{_gate_col(spec.gate_type)}{spec.gate_type}{_C.RST}")

            _log(L, f"{IND}"
                    f"{_C.BOLD}Category    :{_C.RST}  {spec.category}  "
                    f"[Layer: {spec.layer}]")

            # Full details (wrapped at 110 chars)
            if details:
                _log(L, f"{IND}{_C.BOLD}Details     :{_C.RST}  {details[:200]}")
                if len(details) > 200:
                    _log(L, f"{IND}              {details[200:400]}")

            # RUNTIME ERROR — show exception prominently
            if "RUNTIME_ERROR" in details:
                _log(L, f"{IND}{_C.RED}{_C.BOLD}⚠  EXCEPTION :{_C.RST}  "
                         f"{_C.RED}{details}{_C.RST}")

            # Hard gate decision
            if spec.gate_type == "Hard":
                remaining = total_run - run_pos
                if remaining > 0:
                    # Check if remaining tests are all Live/Skip anyway
                    remaining_tests = filtered[run_pos:]  # tests after current
                    live_only = all(
                        s.layer in ("Live", "Live Integrity")
                        for _, _, s, _ in remaining_tests
                    )
                    if live_only:
                        _log(L,
                            f"{IND}{_C.YELLOW}{_C.BOLD}⚠  Gate Action :{_C.RST}  "
                            f"{_C.RED}Hard gate FAILED  →  "
                            f"⛔ HALTING after this test{_C.RST}")
                        _log(L,
                            f"{IND}{_C.GREY}   Note        :  "
                            f"{remaining} remaining tests (LT/MICRO) require live "
                            f"feed — would Skip anyway{_C.RST}")
                    else:
                        _log(L,
                            f"{IND}{_C.YELLOW}{_C.BOLD}⚠  Gate Action :{_C.RST}  "
                            f"{_C.RED}Hard gate FAILED  →  "
                            f"⛔ HALTING  ({remaining} tests will not run){_C.RST}")
                else:
                    _log(L,
                        f"{IND}{_C.YELLOW}{_C.BOLD}⚠  Gate Action :{_C.RST}  "
                        f"{_C.RED}Hard gate FAILED  →  "
                        f"Last test — run complete{_C.RST}")
            else:
                _log(L,
                    f"{IND}{_C.BOLD}   Gate Action :{_C.RST}  "
                    f"{_C.GREEN}Soft gate — CONTINUING to next test{_C.RST}")

            _log(L, "")   # blank line after each failure block

        # ── Hard-gate halt ────────────────────────────────────────────────────
        if spec.gate_type == "Hard" and status == "Fail" and halt_on_hard:
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# GRANULAR: run tests per Source + Exchange combination
# ══════════════════════════════════════════════════════════════════════════════

def _enumerate_combos(data: dict) -> list[tuple[str, str, str, str]]:
    """
    Discover all available (source, exchange, timeframe, product_class) combinations.

    Returns list of tuples: [(source, exchange, timeframe, product_class), ...]
    """
    combos = []
    for tf in ("eod", "intraday"):
        tf_data = data.get(tf, {})
        for exch, src_dict in tf_data.items():
            for src, df in src_dict.items():
                if df is not None and not df.empty:
                    # Primary product
                    pc = data.get("_primary_product", "EQUITY")
                    combos.append((src, exch, tf.upper(), pc))
        # Also check products dict
        prod_data = data.get("products", {}).get(tf, {})
        for exch, src_dict in prod_data.items():
            for src, pc_dict in src_dict.items():
                for pc, df in pc_dict.items():
                    if df is not None and not df.empty:
                        combos.append((src, exch, tf.upper(), pc))
    return sorted(set(combos))


def _filter_ctx_for_combo(
    ctx: DQContext,
    source: str,
    exchange: str,
    timeframe: str,
    product_class: str = "",
) -> DQContext:
    """
    Create a new DQContext with data filtered to ONE source+exchange.

    The filtered data dict preserves the same structure but only has
    the single source populated for the specified exchange+timeframe.
    Other entries are set to None so tests see "data not present"
    for them and naturally skip.
    """
    import copy
    new_data: dict = {}
    tf_key = timeframe.lower()  # "eod" or "intraday"
    other_tf = "intraday" if tf_key == "eod" else "eod"

    # Copy the target timeframe, but only the target source+exchange
    for tf in (tf_key, other_tf):
        new_data[tf] = {}
        tf_data = ctx.data.get(tf, {})
        for exch, src_dict in tf_data.items():
            new_data[tf][exch] = {}
            for src, df in src_dict.items():
                if tf == tf_key and exch == exchange and src == source:
                    # Use product-specific data if not primary
                    primary_pc = ctx.data.get("_primary_product", "EQUITY")
                    if product_class and product_class != primary_pc:
                        prod_df = (ctx.data.get("products", {})
                                   .get(tf, {}).get(exch, {}).get(src, {}).get(product_class))
                        new_data[tf][exch][src] = prod_df if prod_df is not None else df
                    else:
                        new_data[tf][exch][src] = df  # keep primary
                else:
                    new_data[tf][exch][src] = None  # mask others

    return DQContext(
        symbol=ctx.symbol,
        data=new_data,
        config=ctx.config,
        run_mode=ctx.run_mode,
    )


def run_tests_granular(
    ctx: DQContext,
    *,
    layer_filter: str | None = None,
    test_ids: list[str] | None = None,
    mode_filter: str | None = None,
    gate_filter: str | None = None,
    ddq_logger: logging.Logger | None = None,
) -> list[TestResult]:
    """
    Run tests per Source+Exchange combination for granular scoring.

    For each available (source, exchange, timeframe) combo:
      1. Filter context to just that combo
      2. Run applicable tests
      3. Tag every result with source, exchange, timeframe

    Cross-source tests (layer=CROSS_SOURCE, RELATION) run once with
    full context and get tagged as source="cross", exchange="all".
    """
    L = ddq_logger
    combos = _enumerate_combos(ctx.data)

    if not combos:
        _log(L, f"  ⚠  No data available for {ctx.symbol} — skipping all tests")
        return []

    all_results: list[TestResult] = []

    # ── Filter combos by product_filter if provided ─────────────────────────
    product_filter = getattr(ctx, '_product_filter', None)
    if product_filter:
        pf_set = set(product_filter)
        combos = [(s,e,t,p) for s,e,t,p in combos if p in pf_set]

    # ── Layers that need ALL sources/exchanges (run once, not per-combo) ─────
    cross_layers = {
        "CROSS_SOURCE", "RELATION", "SOURCE",    # explicitly cross-source
        "GOVERNANCE", "BEAST_GATES", "COVERAGE",  # check multi-source availability
        "SYMBOL", "UNKNOWN", "REFERENCE DATA",    # instrument config checks
        "LIVE", "LIVE INTEGRITY",                 # live feed tests
        "AGGREGATION", "RECONCILIATION",          # aggregate across sources
        "PORTFOLIO", "PERFORMANCE", "CALENDAR",   # portfolio/calendar checks
    }
    cross_test_ids = [
        tid for tid, (spec, _) in _TESTS.items()
        if spec.layer.upper() in cross_layers
    ]
    # Per-combo: exclude cross-source tests
    combo_test_ids = test_ids  # user filter if any
    if combo_test_ids:
        combo_test_ids = [t for t in combo_test_ids if t not in cross_test_ids]
    else:
        # Build explicit list of non-cross test IDs
        combo_test_ids = [
            tid for tid, (spec, _) in _TESTS.items()
            if spec.layer.upper() not in cross_layers
        ]

    # ── Per-combo tests ─────────────────────────────────────────────────────
    for source, exchange, timeframe, product_class in combos:
        # Skip intraday combos if mode is EOD only, and vice versa
        if mode_filter:
            if mode_filter.upper() == "EOD" and timeframe == "INTRADAY":
                continue
            if mode_filter.upper() == "INTRADAY" and timeframe == "EOD":
                continue

        _log(L, f"\n  {'─'*60}")
        _log(L, f"  📋  {ctx.symbol} / {source.upper()} / {exchange} / {timeframe} / {product_class}")
        _log(L, f"  {'─'*60}")

        filtered_ctx = _filter_ctx_for_combo(ctx, source, exchange, timeframe, product_class)

        results = run_tests_for_symbol(
            filtered_ctx,
            layer_filter=layer_filter,
            test_ids=combo_test_ids,
            mode_filter=mode_filter,
            gate_filter=gate_filter,
            ddq_logger=ddq_logger,
            halt_on_hard=False,
        )

        # Tag every result with source+exchange+timeframe
        for r in results:
            r.source = source
            r.exchange = exchange
            r.product_class = product_class
            r.timeframe = timeframe
            all_results.append(r)

    # ── Cross-source tests (need all sources) ──────────────────────────────
    if cross_test_ids and (not test_ids or any(t in cross_test_ids for t in (test_ids or []))):
        _log(L, f"\n  {'─'*60}")
        _log(L, f"  📋  {ctx.symbol} / CROSS-SOURCE / ALL EXCHANGES")
        _log(L, f"  {'─'*60}")

        cross_results = run_tests_for_symbol(
            ctx,  # full context with all sources
            layer_filter=layer_filter,
            test_ids=cross_test_ids if not test_ids else [t for t in test_ids if t in cross_test_ids],
            mode_filter=mode_filter,
            gate_filter=gate_filter,
            ddq_logger=ddq_logger,
            halt_on_hard=False,
        )
        for r in cross_results:
            r.source = "cross"
            r.exchange = "all"
            r.timeframe = mode_filter or "Both"
            all_results.append(r)

    return all_results


# ── Accessors ─────────────────────────────────────────────────────────────────
def get_registered_tests() -> dict[str, TestSpec]:
    """Return a dict of all registered test specs keyed by test_id."""
    return {tid: spec for tid, (spec, _) in _TESTS.items()}


def clear_registry() -> None:
    """Clear the test registry (used in unit tests)."""
    _TESTS.clear()
