"""
Downloaded Data DQ Engine — Test Runner / Orchestrator
downloaded_data_dq/engine/runner.py
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from downloaded_data_dq.framework import (
    DQContext, TestResult, run_tests_for_symbol, run_tests_granular, _TESTS,
)
from downloaded_data_dq.ingestion.loader import load_symbol
from downloaded_data_dq.utils.config_loader import load_config
from downloaded_data_dq.utils.ddq_logger import setup_ddq_logger, get_log_path
from downloaded_data_dq.utils.run_registry import make_run_id

logger = logging.getLogger(__name__)

# ── Test module registry ───────────────────────────────────────────────────────
TEST_MODULES = [
    "downloaded_data_dq.tests.eod.eod_tests",
    "downloaded_data_dq.tests.eod.eod_tests_extended",
    "downloaded_data_dq.tests.intraday.intraday_tests",
    "downloaded_data_dq.tests.cross_source.cross_source_tests",
    "downloaded_data_dq.tests.relation.relation_tests",
    "downloaded_data_dq.tests.backtest.backtest_tests",
    "downloaded_data_dq.tests.governance.governance_tests",
    "downloaded_data_dq.tests.governance.symbol_tests",
    "downloaded_data_dq.tests.derivatives.derivatives_tests",
    "downloaded_data_dq.tests.etf_index.etf_index_tests",
    "downloaded_data_dq.tests.live.live_tests",
]

W = 80   # width for banner lines


def _import_test_modules() -> None:
    for mod_path in TEST_MODULES:
        try:
            importlib.import_module(mod_path)
        except ModuleNotFoundError:
            logger.debug("Test module not yet implemented: %s", mod_path)
        except Exception as exc:
            logger.error("Failed to import %s: %s", mod_path, exc)


def _L(ddq_log: logging.Logger) -> logging.Logger:
    return ddq_log


def _banner(ddq_log: logging.Logger, char: str, width: int = W) -> None:
    from downloaded_data_dq.framework import _C
    ddq_log.info(f"{_C.GREY}{char * width}{_C.RST}")


def _run_header(
    ddq_log: logging.Logger,
    symbols: list[str],
    mode: str,
    data_dir: str | Path,
    n_tests: int,
    log_path: Path | None,
    run_id: str | None = None,
) -> None:
    from downloaded_data_dq.framework import _C
    W2 = 80
    eq = "=" * W2
    ddq_log.info(f"\n{_C.CYAN}{_C.BOLD}{eq}{_C.RST}")
    ddq_log.info(
        f"  {_C.BOLD}{_C.WHITE}DDQ ENGINE RUN{_C.RST}"
        f"  —  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
        f"  |  Log: {_C.GREY}{log_path or '(none)'}{_C.RST}"
    )
    ddq_log.info(
        f"  {_C.BOLD}RUN_ID   :{_C.RST}  "
        f"{_C.YELLOW}{_C.BOLD}{run_id or '—'}{_C.RST}"
    )
    ddq_log.info(
        f"  {_C.BOLD}Symbols  :{_C.RST}  "
        f"{_C.PURPLE}{'  '.join(symbols)}{_C.RST}"
    )
    ddq_log.info(
        f"  {_C.BOLD}Mode     :{_C.RST}  {mode}"
        f"  |  Tests registered: {_C.YELLOW}{n_tests}{_C.RST}"
        f"  |  Data dir: {_C.GREY}{data_dir}{_C.RST}"
    )
    ddq_log.info(f"{_C.CYAN}{_C.BOLD}{eq}{_C.RST}\n")


def _sym_banner_start(
    ddq_log: logging.Logger,
    symbol: str,
    sym_idx: int,
    sym_total: int,
) -> None:
    from downloaded_data_dq.framework import _C
    line = "━" * W
    ddq_log.info(f"\n{_C.CYAN}{line}{_C.RST}")
    ddq_log.info(
        f"  {_C.BOLD}{_C.WHITE}SYMBOL {sym_idx}/{sym_total}{_C.RST}"
        f"  :  {_C.PURPLE}{_C.BOLD}{symbol}{_C.RST}"
        f"  |  Started: {datetime.now().strftime('%H:%M:%S')}"
    )
    ddq_log.info(f"{_C.CYAN}{line}{_C.RST}\n")


def _sym_scorecard(
    ddq_log: logging.Logger,
    symbol: str,
    results: list[TestResult],
    elapsed: float,
) -> None:
    from downloaded_data_dq.framework import _C, _classify_cause
    from collections import Counter

    c       = Counter(r.status for r in results)
    passed  = c["Pass"];  failed  = c["Fail"];  skipped = c.get("Skip", 0)
    total   = len(results)
    rate    = f"{passed/(passed+failed)*100:.1f}%" if (passed+failed) else "N/A"
    hard_f  = sum(1 for r in results if r.status=="Fail" and r.gate_type=="Hard")
    crit_f  = sum(1 for r in results if r.status=="Fail" and r.severity=="Critical")
    wt_pass = sum(r.weighted_score for r in results)
    wt_tot  = sum(r.weight for r in results)
    wt_pct  = f"{wt_pass/wt_tot*100:.1f}%" if wt_tot else "N/A"
    elapsed_str = (f"{int(elapsed//60)}m {int(elapsed%60)}s"
                   if elapsed >= 60 else f"{elapsed:.1f}s")

    rate_col = (_C.GREEN if passed/(passed+failed)>=0.95
                else _C.YELLOW if passed/(passed+failed)>=0.80
                else _C.RED) if (passed+failed) else _C.GREY

    line = "─" * W
    ddq_log.info(f"\n{_C.GREY}{line}{_C.RST}")
    ddq_log.info(
        f"  {_C.BOLD}SCORECARD{_C.RST}  "
        f"{_C.PURPLE}{_C.BOLD}{symbol}{_C.RST}"
        f"  |  Pass={_C.GREEN}{passed}{_C.RST}"
        f"  Fail={_C.RED}{failed}{_C.RST}"
        f"  Skip={_C.GREY}{skipped}{_C.RST}"
        f"  Total={total}"
        f"  Rate={rate_col}{_C.BOLD}{rate}{_C.RST}"
    )
    ddq_log.info(
        f"  {_C.BOLD}Quality  {_C.RST}  "
        f"Weighted={_C.CYAN}{wt_pct}{_C.RST}"
        f"  |  Hard gate fails={_C.RED}{hard_f}{_C.RST}"
        f"  |  Critical fails={_C.RED}{crit_f}{_C.RST}"
        f"  |  Elapsed={elapsed_str}"
    )

    # List failures concisely
    failures = [r for r in results if r.status == "Fail"]
    if failures:
        sev_order = {"Critical":0,"High":1,"Medium":2,"Low":3,"Info":4}
        failures.sort(key=lambda r: sev_order.get(r.severity, 9))
        shown    = failures[:8]
        more     = len(failures) - len(shown)
        fail_str = "  ".join(
            f"{_C.RED}{r.test_id}{_C.RST}({_sev_short(r.severity)})"
            for r in shown
        )
        if more:
            fail_str += f"  {_C.GREY}[+{more} more]{_C.RST}"
        ddq_log.info(f"  {_C.BOLD}Failures {_C.RST}  {fail_str}")

    ddq_log.info(f"{_C.GREY}{line}{_C.RST}\n")


def _sev_short(sev: str) -> str:
    from downloaded_data_dq.framework import _C, _sev_col
    col = _sev_col(sev)
    s = {"Critical":"Crit","High":"High","Medium":"Med","Low":"Low","Info":"Info"}.get(sev, sev[:4])
    return f"{col}{s}{_C.RST}"


def _run_summary(
    ddq_log: logging.Logger,
    all_results: dict[str, list[TestResult]],
    run_start: float,
    log_path: Path | None,
) -> None:
    from downloaded_data_dq.framework import _C

    total_elapsed = time.perf_counter() - run_start
    elapsed_str   = (f"{int(total_elapsed//60)}m {int(total_elapsed%60)}s"
                     if total_elapsed >= 60 else f"{total_elapsed:.1f}s")

    eq = "=" * W
    ddq_log.info(f"\n{_C.CYAN}{_C.BOLD}{eq}{_C.RST}")
    ddq_log.info(
        f"  {_C.BOLD}{_C.WHITE}FINAL SUMMARY{_C.RST}"
        f"  |  Completed: {datetime.now().strftime('%H:%M:%S')}"
        f"  |  Total elapsed: {elapsed_str}"
    )
    ddq_log.info("")

    # Table header
    col_w = 28
    ddq_log.info(
        f"  {_C.GREY}{'Symbol':<{col_w}} {'Pass':>6}  {'Fail':>6}  {'Skip':>6}  "
        f"{'Total':>7}  {'Rate':>7}  {'Wt.Score':>9}{_C.RST}"
    )
    ddq_log.info(f"  {_C.GREY}{'─'*(col_w+60)}{_C.RST}")

    grand = {"pass":0,"fail":0,"skip":0,"total":0,"wt_pass":0.0,"wt_tot":0.0}
    for sym, res in all_results.items():
        from collections import Counter, defaultdict
        # Group by product_class within symbol
        pc_groups = defaultdict(list)
        for r in res:
            pc = getattr(r, "product_class", "") 
            if pc:
                pc_groups[pc].append(r)
            else:
                # Cross-source tests — count toward grand total but don't show separate row
                c2 = Counter([r.status])
                grand["pass"] += c2.get("Pass",0)
                grand["fail"] += c2.get("Fail",0) 
                grand["skip"] += sum(1 for s in c2 if s not in ("Pass","Fail"))
                grand["total"] += 1
                grand["wt_pass"] += r.weighted_score
                grand["wt_tot"] += r.weight
        for pc in sorted(pc_groups.keys()):
            pc_res = pc_groups[pc]
            c  = Counter(r.status for r in pc_res)
            p  = c["Pass"]; f  = c["Fail"]; s  = c.get("Skip",0)
            t  = len(pc_res)
            rt = f"{p/(p+f)*100:.1f}%" if (p+f) else "N/A"
            wp = sum(r.weighted_score for r in pc_res)
            wt = sum(r.weight for r in pc_res)
            ws = f"{wp/wt*100:.1f}%" if wt else "N/A"
            rc = (_C.GREEN if p/(p+f)>=0.95 else _C.YELLOW if p/(p+f)>=0.80
                  else _C.RED) if (p+f) else _C.GREY
            label = f"{sym}/{pc}"
            ddq_log.info(
                f"  {_C.WHITE}{label:<{col_w}}{_C.RST}"
                f" {_C.GREEN}{p:>6}{_C.RST} "
                f" {_C.RED}{f:>6}{_C.RST} "
                f" {_C.GREY}{s:>6}{_C.RST} "
                f" {t:>7} "
                f" {rc}{rt:>7}{_C.RST} "
                f" {_C.CYAN}{ws:>9}{_C.RST}"
            )
            grand["pass"] += p; grand["fail"] += f; grand["skip"] += s
            grand["total"] += t; grand["wt_pass"] += wp; grand["wt_tot"] += wt

    ddq_log.info(f"  {_C.GREY}{'─'*(col_w+60)}{_C.RST}")
    gp = grand["pass"]; gf = grand["fail"]
    grt = f"{gp/(gp+gf)*100:.1f}%" if (gp+gf) else "N/A"
    gws = f"{grand['wt_pass']/grand['wt_tot']*100:.1f}%" if grand["wt_tot"] else "N/A"
    gc  = (_C.GREEN if gp/(gp+gf)>=0.95 else _C.YELLOW if gp/(gp+gf)>=0.80
           else _C.RED) if (gp+gf) else _C.GREY
    ddq_log.info(
        f"  {_C.BOLD}{_C.WHITE}{'TOTAL':<{col_w}}{_C.RST}"
        f" {_C.GREEN}{gp:>6}{_C.RST} "
        f" {_C.RED}{gf:>6}{_C.RST} "
        f" {_C.GREY}{grand['skip']:>6}{_C.RST} "
        f" {grand['total']:>7} "
        f" {gc}{grt:>7}{_C.RST} "
        f" {_C.CYAN}{gws:>9}{_C.RST}"
    )

    ddq_log.info("")
    if log_path:
        log_size = log_path.stat().st_size if log_path.exists() else 0
        ddq_log.info(
            f"  {_C.BOLD}Log saved  :{_C.RST}  "
            f"{_C.GREY}{log_path}  ({log_size//1024} KB){_C.RST}"
        )
    ddq_log.info(f"{_C.CYAN}{_C.BOLD}{eq}{_C.RST}\n")


# ══════════════════════════════════════════════════════════════════════════════
# run()
# ══════════════════════════════════════════════════════════════════════════════
def run(
    symbols: list[str],
    data_dir: str | Path = "data/raw",
    config_dir: str | Path | None = None,
    mode: str = "Both",
    layer_filter: str | None = None,
    test_ids: list[str] | None = None,
    log_dir: str | Path = "logs",
    run_id: str | None = None,
    products: list[str] | None = None,
) -> dict[str, list[TestResult]]:
    """
    Run the full DQ suite for a list of symbols with detailed logging.

    Returns:
        Tuple of (results_dict, run_id)
    """
    _import_test_modules()

    # ── Generate RUN_ID ──────────────────────────────────────────────────────
    if not run_id:
        run_id = make_run_id()

    # ── Set up dual-output logger ─────────────────────────────────────────────
    ddq_log  = setup_ddq_logger(log_dir=log_dir, run_id=run_id)
    log_path = get_log_path()

    config   = load_config(config_dir)
    data_dir = Path(data_dir)
    n_tests  = len(_TESTS)

    _run_header(ddq_log, symbols, mode, data_dir, n_tests, log_path, run_id)

    run_start    = time.perf_counter()
    all_results: dict[str, list[TestResult]] = {}
    all_prevalidation: dict[str, list] = {}

    for sym_idx, symbol in enumerate(symbols, 1):
        sym_start = time.perf_counter()

        _sym_banner_start(ddq_log, symbol, sym_idx, len(symbols))

        # Log data loading (reuse existing ingestion logging)
        ddq_log.info(
            f"  Loading data for {symbol} ..."
        )
        data, avail = load_symbol(symbol, data_dir, config, run_mode=mode)

        # Capture pre-validation results
        all_prevalidation[symbol] = avail.pre_validation

        # Data availability summary with reference names
        avail_lines = []
        for src in ("dhan", "kite", "upstox"):
            for exch in ("BSE", "NSE"):
                for tf in ("eod", "intraday"):
                    df = data.get(tf, {}).get(exch, {}).get(src)
                    if df is not None:
                        from downloaded_data_dq.framework import _C
                        ref = f"{symbol}_{src.upper()}_{exch}_{tf.upper()}"
                        avail_lines.append(
                            f"    {_C.GREEN}✅{_C.RST}  "
                            f"{ref:<40}"
                            f"|  {len(df):>7} rows"
                        )
        total_combos = len(avail_lines)
        ddq_log.info(
            f"  {total_combos} data combinations loaded"
        )
        for line in avail_lines:
            ddq_log.info(line)
        ddq_log.info("")

        ctx = DQContext(
            symbol=symbol,
            data=data,
            config=config,
            run_mode=mode,
        )
        if products:
            ctx._product_filter = products

        results = run_tests_granular(
            ctx,
            layer_filter=layer_filter,
            test_ids=test_ids,
            mode_filter=mode,
            ddq_logger=ddq_log,
        )

        all_results[symbol] = results
        sym_elapsed = time.perf_counter() - sym_start
        _sym_scorecard(ddq_log, symbol, results, sym_elapsed)

    _run_summary(ddq_log, all_results, run_start, log_path)

    return all_results, run_id, all_prevalidation


# ── results_to_dataframe ──────────────────────────────────────────────────────
def results_to_dataframe(results: dict[str, list[TestResult]]) -> pd.DataFrame:
    rows = []
    for symbol, res_list in results.items():
        for r in res_list:
            rows.append({
                "symbol":         symbol,
                "test_id":        r.test_id,
                "source":         getattr(r, "source", ""),
                "exchange":       getattr(r, "exchange", ""),
                "product_class":  getattr(r, "product_class", ""),
                "timeframe":      getattr(r, "timeframe", ""),
                "layer":          r.layer,
                "category":       r.category,
                "severity":       r.severity,
                "gate_type":      r.gate_type,
                "status":         r.status,
                "success_score":  r.success_score,
                "weight":         r.weight,
                "weighted_score": r.weighted_score,
                "details":        r.details,
                "last_run":       r.last_run,
            })
    return pd.DataFrame(rows)


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    # Suppress the noisy root logger — DDQ logger handles terminal output
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="downloaded-data-dq",
        description="Downloaded Data DQ Engine",
    )
    parser.add_argument("--symbols",    nargs="+", required=True)
    parser.add_argument("--data-dir",   default="data/raw")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--mode",       choices=["EOD","Intraday","Both"], default="Both")
    parser.add_argument("--layer",      default=None)
    parser.add_argument("--test-ids",   nargs="*", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--log-dir",    default="logs",
                        help="Directory for log files (default: logs/)")

    args = parser.parse_args()

    results, run_id = run(
        symbols=args.symbols,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        mode=args.mode,
        layer_filter=args.layer,
        test_ids=args.test_ids,
        log_dir=args.log_dir,
    )

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df = results_to_dataframe(results)
        df.to_csv(args.output_csv, index=False)
        print(f"Results written to: {args.output_csv}")
    print(f"RUN_ID: {run_id}")


if __name__ == "__main__":
    main()
