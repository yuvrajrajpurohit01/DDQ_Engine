#!/usr/bin/env python3
"""
run_ddq.py — DDQ Engine orchestrator (v12 — Source+Exchange Edition)

Auto-generates a 10-digit RUN_ID (YYMMDDHHMM) that PREFIXES all output files:
  logs/{RUN_ID}_ddq.log
  reports/scorecard/{RUN_ID}_results.csv
  reports/html/{RUN_ID}_dq_command_centre.html
  reports/runs/run_registry.html  (updated after every run)
  data/rectified/{RUN_ID}/       (rectified CSVs + audit trail)
  data/final/{RUN_ID}/           (final merged series)

Usage:
  python run_ddq.py --symbols TCS RELIANCE NIFTY --mode Both
  python run_ddq.py --symbols TCS --mode Both --rectify
  python run_ddq.py --symbols TCS --rectify --window quarterly
  python run_ddq.py --synthetic --symbols TCS RELIANCE --mode EOD
"""
from __future__ import annotations
import argparse, os, sys, warnings, logging
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# PROJECT_ROOT is always the installed location — never relative to __file__
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from downloaded_data_dq.utils.run_registry import make_run_id, register_run
from downloaded_data_dq.utils.ddq_logger import setup_ddq_logger, get_log_path


def _json_safe(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    import json
    return json.loads(json.dumps(obj, default=lambda x: int(x) if hasattr(x, 'item') else str(x)))


KNOWN_EXCHANGES    = {"NSE", "BSE"}
KNOWN_SOURCES      = {"DHAN", "UPSTOX", "KITE"}
KNOWN_PRODUCT_CLASSES = {
    "EQUITY", "ETF", "INDEX",
    "INDEX_EQUITY", "INDEX_OPT_CE", "INDEX_OPT_PE", "INDEX_FUT",
    "EQUITY_OPT_CE", "EQUITY_OPT_PE", "EQUITY_FUT",
}

def discover_symbols(
    data_dir: str | Path,
    mode: str = "Both",
) -> list[str]:
    """
    Scan the data directory tree and return a sorted list of unique
    symbol names found.

    Expected tree shape:
        data_dir/
        └── {EOD|INTRADAY}/
            └── {Dhan|Upstox|Kite}/
                └── {NSE|BSE}/
                    └── {EQUITY|ETF|INDEX|INDEX_EQUITY|…}/
                        └── {SYMBOL}.csv

    Args:
        data_dir : Root data directory.
        mode     : "EOD", "Intraday", or "Both".

    Returns:
        Sorted list of unique symbol strings (upper-cased).
    """
    data_path = Path(data_dir)

    # ── Which timeframe folders to scan ──────────────────────────────────
    freqs: list[str] = []
    if mode in ("EOD",      "Both"): freqs.append("EOD")
    if mode in ("Intraday", "Both"): freqs.append("INTRADAY")

    symbols: set[str] = set()

    for freq in freqs:
        freq_path = data_path / freq
        if not freq_path.exists():
            continue

        # level 1 — source  (Dhan / Upstox / Kite)
        for source_dir in freq_path.iterdir():
            if not source_dir.is_dir():
                continue
            if source_dir.name.upper() not in KNOWN_SOURCES:
                continue                        # skip unknown folders

            # level 2 — exchange  (NSE / BSE)
            for exchange_dir in source_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue
                if exchange_dir.name.upper() not in KNOWN_EXCHANGES:
                    continue                    # skip unknown folders

                # level 3 — product class  (EQUITY / ETF / INDEX …)
                for pc_dir in exchange_dir.iterdir():
                    if not pc_dir.is_dir():
                        continue
                    if pc_dir.name.upper() not in KNOWN_PRODUCT_CLASSES:
                        continue               # skip unknown folders

                    # level 4 — symbol files  ({SYMBOL}.csv)
                    for csv_file in pc_dir.glob("*.csv"):
                        if csv_file.stat().st_size == 0:
                            continue           # skip empty files

                        symbol = csv_file.stem.strip()
                        if symbol:
                            symbols.add(symbol.upper())

    return sorted(symbols)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_ddq",
        description="DDQ Engine v12 — detect, rectify, window-score, merge",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="One or more symbol names e.g. RELIANCE HDFCBANK. "
             "Omit to auto-discover all symbols from --data-dir.",
    )
    parser.add_argument(
        "--all-symbols", action="store_true",
        help="Auto-discover and run every symbol found in --data-dir. "
             "Equivalent to omitting --symbols entirely.",
    )
    parser.add_argument("--data-dir",    default="data/raw")
    parser.add_argument("--mode",        choices=["EOD", "Intraday", "Both"], default="Both")
    parser.add_argument("--layer",       default=None)
    parser.add_argument("--test-ids",    nargs="*", default=None)
    parser.add_argument("--config-dir",  default=None)
    parser.add_argument("--run-id",      default=None,
                        help="Override auto-generated RUN_ID (10 digits)")
    parser.add_argument("--log-dir",     default="logs")
    parser.add_argument("--output-csv",  default=None,
                        help="Also save results to this path (backward compat)")
    # ── Rectification flags ──────────────────────────────────────────────────
    parser.add_argument("--rectify",     action="store_true",
                        help="Enable data rectification after detection")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show what would be rectified without writing files")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Override minimum confidence threshold for auto-fix")
    parser.add_argument("--rules",       nargs="*", default=None,
                        help="Run only specific rectification rules")
    # ── v12 Window, Merge & Synthetic flags ────────────────────────────────
    parser.add_argument("--window",      nargs="*", default=None,
                        help="Rolling window frequencies (multiple allowed): "
                             "monthly quarterly half_yearly yearly 2_yearly "
                             "3_yearly 5_yearly 10_yearly 15_yearly all "
                             "(default: all frequencies)")
    parser.add_argument("--merge_final", action="store_true",
                        help="Create final clean merged series from best sources")
    parser.add_argument("--products",    nargs="*", default=None,
                         help="Product classes to test (default: all for each symbol)")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Generate synthetic data before running tests")
    args = parser.parse_args()

    # ── Resolve symbol list ───────────────────────────────────────────────────
    if args.all_symbols or args.symbols is None:
        symbols = discover_symbols(args.data_dir, mode=args.mode)
        if not symbols:
            print(
                f"[ERROR] No symbols discovered in '{args.data_dir}'. "
                "Check that the folder contains EXCHANGE_SYMBOL.csv files.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[INFO] Auto-discovered {len(symbols)} symbol(s): {', '.join(symbols[:10])}"
              + (f"  … +{len(symbols) - 10} more" if len(symbols) > 10 else ""))
    else:
        symbols = [s.upper() for s in args.symbols]

    # ── Synthetic data generation if requested ────────────────────────────────
    if args.synthetic:
        from generate_synthetic_data import generate_all
        print("  Generating synthetic data ...")
        syn_dir = PROJECT_ROOT / "data" / "synthetic"
        generate_all(data_dir=syn_dir, seed=2025, symbols_filter=args.symbols,
                     mode=args.mode, verbose=False)
        if args.data_dir == "data/raw":
            args.data_dir = "data/synthetic"
        print("  Synthetic data ready.")

    # ── RUN_ID ────────────────────────────────────────────────────────────────
    run_id = args.run_id or make_run_id()

    print(run_id)

    # ── Work relative to PROJECT_ROOT ─────────────────────────────────────────
    old_dir = os.getcwd()
    os.chdir(str(PROJECT_ROOT))

    start_dt = datetime.now()
    # ── Phase 2: Rectification (v11) ───────────────────────────────────────
    print("Started Phase2: Rectification")
    rect_summary = None
    data_store = {}
    if args.rectify:
        try:
            from downloaded_data_dq.rectification.engine import run_rectification
            from downloaded_data_dq.ingestion.loader import load_symbol
            from downloaded_data_dq.utils.config_loader import load_config as _lc
            _cfg = _lc(args.config_dir)
            _dd = Path(args.data_dir)
            data_store = {}
            for sym in symbols:
                d, _ = load_symbol(sym, _dd, _cfg, run_mode=args.mode)
                data_store[sym] = d
            print("Symbol Loaded for Rectification")
            audit_log, _ = run_rectification(
                data_store=data_store, config=_cfg,
                project_root=PROJECT_ROOT, run_id=run_id, mode=args.mode, log_dir=args.log_dir,
                dry_run=args.dry_run, rule_filter=args.rules,
                min_confidence=args.min_confidence,
            )
            print("Rectification Done")
            rect_summary = audit_log.summary()
            rect_summary["results"] = [r.to_dict() for r in audit_log.results]
        except Exception as exc:
            logging.getLogger(__name__).warning("Rectification: %s", exc)
            import traceback;
            traceback.print_exc()
            rect_summary = {"error": str(exc)}

        stop_dt = datetime.now()
        elapsed = (stop_dt - start_dt).total_seconds()
        print(f"Rectification completed on {stop_dt}")


    try:
        data_dir = args.data_dir
        print(args.products)
        # ── Run engine ────────────────────────────────────────────────────────
        from downloaded_data_dq.engine.runner import run, results_to_dataframe
        results, _rid, prevalidation = run(
            symbols=symbols,
            data_dir=data_dir,
            config_dir=args.config_dir,
            mode=args.mode,
            layer_filter=args.layer,
            test_ids=args.test_ids,
            log_dir=args.log_dir,
            run_id=run_id,
            products=args.products,
        )

        stop_dt  = datetime.now()
        elapsed  = (stop_dt - start_dt).total_seconds()
        log_path = get_log_path()



        # ── Phase 3: Rolling Window DQ Analysis ──────────────────────────────────
        print("Started Phase3: Rolling windows")
        rolling_results = None
        best_sources = None
        data_dir    =f"data/rectified/{run_id}"
        print(data_dir)
        try:
            from downloaded_data_dq.rolling.engine import run_rolling_analysis, select_best_sources
            from downloaded_data_dq.ingestion.loader import load_symbol as _ls2
            from downloaded_data_dq.utils.config_loader import load_config as _lc2

            _cfg2 = _lc2(args.config_dir)
            #_dd2 = Path(args.data_dir)
            _dd2 = Path(data_dir)  ## Added by DK on 04-05-2026

            # Build data_store if not already built during rectification
            if 'data_store' not in locals():
                data_store = {}
            if not data_store:
                for sym in symbols:
                    d, _ = _ls2(sym, _dd2, _cfg2, run_mode=args.mode)
                    data_store[sym] = d

            # Determine frequencies
            window_freqs = args.window if args.window else None  # None = all
            print("All data loaded for Rolling Windows")
            rolling_results = run_rolling_analysis(
                data_store=data_store,
                config=_cfg2,
                frequencies=window_freqs,
                mode=args.mode,
                verbose=True,
                log_dir=args.log_dir,
                run_id=run_id,
            )
            print("Rolling phase finished")
            best_sources = select_best_sources(rolling_results)

            stop_dt  = datetime.now()
            elapsed  = (stop_dt - start_dt).total_seconds()
        except Exception as exc:
            logging.getLogger(__name__).warning("Rolling analysis: %s", exc)
            import traceback; traceback.print_exc()
            rolling_results = {"error": str(exc)}

        # ── Phase 4: Final Clean Merged Series (opt-in via --merge_final) ────────
        print("Phase 4 started: Final Clean Merged Series")
        merge_result = None
        provenance = None
        if args.merge_final and best_sources and rolling_results and "error" not in (rolling_results or {}):
            try:
                from downloaded_data_dq.merge.engine import run_merge, summarize_provenance
                # Use finest requested frequency for merge
                _freq_priority = ["monthly","quarterly","half_yearly","yearly",
                                  "2_yearly","3_yearly","5_yearly","10_yearly","15_yearly","all"]
                merge_freq = "quarterly"  # default
                if args.window:
                    # Pick the finest (smallest) frequency the user requested
                    for fp in _freq_priority:
                        if fp in args.window:
                            merge_freq = fp
                            break

                merge_result = run_merge(
                    data_store=data_store,
                    rolling_results=rolling_results,
                    best_sources=best_sources,
                    config=_cfg2,
                    project_root=PROJECT_ROOT,
                    run_id=run_id,
                    mode=args.mode,
                    merge_freq=merge_freq,
                    verbose=True,
                )
                print("Merge Done")
                provenance = summarize_provenance(merge_result)

                stop_dt = datetime.now()
                elapsed = (stop_dt - start_dt).total_seconds()
            except Exception as exc:
                logging.getLogger(__name__).warning("Merge: %s", exc)
                import traceback; traceback.print_exc()
                merge_result = {"error": str(exc)}

        # ── Save results CSV — prefixed with RUN_ID ───────────────────────────
        print("Result CSV started to save with RUN_ID: %s", run_id)
        df = results_to_dataframe(results)
        defaults = {
            "status": "Skip",
            "weighted_score": 0,
            "weight": 1.0,
            "gate_type": "Soft",
            "severity": "Low",
            "layer": "Unknown",
            "category": "Unknown",
            "symbol": "Unknown",
            "test_id": "Unknown",
            "cause": "",
            "details": "",
            "source": "",
            "product_class": "",
            "exchange": "",
            "timeframe": "",
            "last_run": datetime.now(),
        }

        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
                
        scorecard_dir = PROJECT_ROOT / "reports" / "scorecard"
        scorecard_dir.mkdir(parents=True, exist_ok=True)
        results_csv = scorecard_dir / f"{run_id}_results.csv"
        df.to_csv(results_csv, index=False)
        if args.output_csv:                          # backward compat
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)

        # ── Generate Command Centre dashboard — prefixed with RUN_ID ──────────
        html_dir = PROJECT_ROOT / "reports" / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        dashboard_h = html_dir / f"{run_id}_dq_command_centre.html"

        run_meta = {
            "run_id":        run_id,
            "start":         start_dt.strftime("%Y-%m-%d  %H:%M:%S"),
            "elapsed_s":     int(elapsed),
            "symbols":       symbols,
            "mode":          args.mode,
            "log_file":      f"{run_id}_ddq.log",
            "results_file":  f"{run_id}_results.csv",
            "dash_file":     f"{run_id}_dq_command_centre.html",
            "registry_file": "run_registry.html",
            # Relative paths used by dashboard template (from reports/html/)
            "log_rel":      f"logs/{run_id}_ddq.log",
            "results_rel":  f"reports/scorecard/{run_id}_results.csv",
            "registry_rel": "reports/runs/run_registry.html",
            "dash_rel":     f"reports/html/{run_id}_dq_command_centre.html",
            # v11 rectification summary
            "rectification": rect_summary,
            # v12 pre-validation (Step 0) — convert numpy types to native Python
            "prevalidation": _json_safe(prevalidation),
            # v12 rolling window analysis (Phase 3)
            "rolling": _json_safe(rolling_results) if rolling_results and "error" not in (rolling_results or {}) else rolling_results,
            "best_sources": _json_safe(best_sources) if best_sources else None,
            # v12 Phase 4 merge
            "merge": _json_safe(merge_result) if merge_result and "error" not in (merge_result or {}) else merge_result,
            "provenance": _json_safe(provenance) if provenance else None,
        }

        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "generate_report", PROJECT_ROOT / "generate_report.py")
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        html_content = gen_mod.generate_html(df, run_id=run_id, run_meta=run_meta)
        dashboard_h.write_text(html_content, encoding="utf-8")

        # ── Update Run Registry ───────────────────────────────────────────────
        n_pass = int((df["status"] == "Pass").sum())
        n_fail = int((df["status"] == "Fail").sum())
        n_skip = int((df["status"] == "Skip").sum())
        pr     = round(n_pass/(n_pass+n_fail)*100, 1) if (n_pass+n_fail) else 100.0
        wt_pass= float(df[df["status"]=="Pass"]["weighted_score"].sum())
        wt_tot = float(df["weight"].sum())
        wts    = round(wt_pass/wt_tot*100, 1) if wt_tot else 100.0

        registry_html = register_run(PROJECT_ROOT, {
            "run_id":      run_id,
            "start":       start_dt.isoformat(),
            "stop":        stop_dt.isoformat(),
            "elapsed_s":   elapsed,
            "symbols":     symbols,
            "mode":        args.mode,
            "n_pass":      n_pass,
            "n_fail":      n_fail,
            "n_skip":      n_skip,
            "pass_rate":   pr,
            "wt_score":    wts,
            "hard_fails":  int(((df["status"]=="Fail")&(df["gate_type"]=="Hard")).sum()),
            "crit_fails":  int(((df["status"]=="Fail")&(df["severity"]=="Critical")).sum()),
            "log_file":    f"{run_id}_ddq.log",
            "results_file":f"{run_id}_results.csv",
            "dash_file":   f"{run_id}_dq_command_centre.html",
            "rectify":     args.rectify,
            "rect_fixed":  rect_summary.get("fixed", 0) if (rect_summary and "error" not in rect_summary) else 0,
            "rect_changes":rect_summary.get("total_changes", 0) if (rect_summary and "error" not in rect_summary) else 0,
        })

        # ── Print final summary ───────────────────────────────────────────────
        W = 70
        print(f"\n{'═'*W}")
        print(f"  RUN COMPLETE  ·  RUN_ID: {run_id}")
        print(f"{'─'*W}")
        print(f"  Results CSV  :  {results_csv}")
        print(f"  Dashboard    :  {dashboard_h}")
        print(f"  Open         :  xdg-open {dashboard_h}")
        print(f"  Run Registry :  {registry_html}")
        if log_path:
            print(f"  Debug Log    :  {log_path}")
        if args.rectify and rect_summary and "error" not in rect_summary:
            rect_dir = PROJECT_ROOT / "data" / "rectified" / run_id
            print(f"{'─'*W}")
            print(f"  ✅ Rectification:  {rect_summary.get('fixed',0)} fixed  |  "
                  f"{rect_summary.get('flagged',0)} flagged  |  "
                  f"{rect_summary.get('total_changes',0)} total changes")
            if not args.dry_run:
                print(f"  Rectified Data :  {rect_dir}")
                print(f"  Audit Trail    :  {rect_dir / f'audit_{run_id}.json'}")
        if rolling_results and "error" not in (rolling_results or {}):
            rs = rolling_results.get("summary", {})
            print(f"{'─'*W}")
            print(f"  📊 Rolling DQ:  {rs.get('total_scores',0)} scores  |  "
                  f"{rs.get('total_windows',0)} windows  |  "
                  f"{len(rs.get('frequencies',[]))} frequencies  |  "
                  f"{rs.get('elapsed_s',0):.1f}s")
            if best_sources:
                for sym, sym_sel in (best_sources or {}).items():
                    for freq, sel_list in sym_sel.items():
                        if freq == "all" and sel_list:
                            for s in sel_list:
                                print(f"  🏆 {sym}/{s.get('timeframe','?')}: "
                                      f"Best={s.get('best_source','?')}/{s.get('best_exchange','?')} "
                                      f"(score={s.get('best_score',0)*100:.1f}%)")
        if merge_result and "error" not in (merge_result or {}):
            print(f"{'─'*W}")
            print(f"  📁 Final Merge: {merge_result.get('files_written',0)} files  |  "
                  f"{merge_result.get('total_rows',0):,} rows  |  "
                  f"freq={merge_result.get('merge_freq','?')}")
            final_dir = PROJECT_ROOT / "data" / "final" / run_id
            print(f"  Final Series :  {final_dir}")
            for d in merge_result.get("details", [])[:5]:
                print(f"    📄 {d['ref']}: {d['rows']:,} rows from {d['windows_used']} windows")
        print(f"{'═'*W}\n")

    finally:
        os.chdir(old_dir)


if __name__ == "__main__":
    main()
