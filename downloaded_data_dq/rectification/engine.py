"""
DDQ Engine — Rectification Engine
downloaded_data_dq/rectification/engine.py

Orchestrates the detect-then-rectify pipeline:
  1. Analyse detection failures
  2. Match rectification rules
  3. Execute rules on deep-copied data
  4. Record audit trail
  5. Optionally re-validate
  6. Write rectified data
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from downloaded_data_dq.rectification.audit import AuditLog, RectificationResult
from downloaded_data_dq.rectification.registry import (
    RuleSpec, get_all_rules, get_rules_for_test, import_rule_modules,
)
from downloaded_data_dq.utils.ddq_logger import setup_ddq_logger, get_log_path
from downloaded_data_dq.rectification.writer import write_rectified

logger = logging.getLogger(__name__)



def _load_rect_config(config: dict) -> dict:
    """Extract rectification config from main config dict."""
    rc = config.get("rectification", {})
    if not rc:
        rc = {}
    return {
        "enabled": rc.get("global", {}).get("enabled", True),
        "min_confidence": rc.get("global", {}).get("min_confidence", 0.70),
        "max_changes_per_source": rc.get("global", {}).get("max_changes_per_source", 500),
        "write_rectified": rc.get("global", {}).get("write_rectified", True),
        "re_validate": rc.get("global", {}).get("re_validate", True),
        "source_priority": rc.get("global", {}).get("source_priority", ["kite", "dhan", "upstox"]),
        "eod_rules": rc.get("eod_rules", {}),
        "intraday_rules": rc.get("intraday_rules", {}),
    }


def _get_rule_config(rect_config: dict, rule_id: str, spec: RuleSpec) -> dict:
    """Get per-rule config, falling back to spec defaults."""
    if rule_id.startswith("RECT-EOD"):
        rules_cfg = rect_config.get("eod_rules", {})
    else:
        rules_cfg = rect_config.get("intraday_rules", {})
    rule_cfg = rules_cfg.get(rule_id, {})
    if isinstance(rule_cfg, dict):
        if "confidence" not in rule_cfg:
            rule_cfg["confidence"] = spec.default_conf
        if "enabled" not in rule_cfg:
            rule_cfg["enabled"] = True
    else:
        rule_cfg = {"enabled": True, "confidence": spec.default_conf}
    return rule_cfg


def _extract_failures(results: dict) -> list[dict]:
    """Extract failed test results from the detection results dict."""
    failures = []
    for symbol, sym_results in results.items():
        if not isinstance(sym_results, dict):
            continue
        for test_id, result in sym_results.items():
            if not hasattr(result, "status"):
                continue
            if result.status == "Fail":
                failures.append({
                    "test_id": result.test_id,
                    "symbol": symbol,
                    "source": getattr(result, "source", ""),
                    "timeframe": getattr(result, "timeframe", ""),
                    "layer": getattr(result, "layer", ""),
                    "details": getattr(result, "details", ""),
                    "metrics": getattr(result, "metrics", {}),
                })
    return failures


def run_rectification(
    #results: dict,
    data_store: dict,        # {symbol: DQContext.data} — original loaded data
    config: dict,
    project_root: Path,
    run_id: str,
    mode: str = "Both",
    log_dir: str | Path = "logs",
    dry_run: bool = False,
    rule_filter: list[str] | None = None,
    min_confidence: float | None = None,
) -> tuple[AuditLog, dict]:
    """
    Run the full rectification pipeline.

    Args:
        results:      Detection results dict from runner.run()
        data_store:   Dict of {symbol: ctx.data} with loaded DataFrames
        config:       Full config dict (includes rectification section)
        project_root: Path to ~/downloaded_data_dq
        run_id:       Current RUN_ID
        mode:         "Both" | "EOD" | "Intraday"
        dry_run:      If True, don't write files
        rule_filter:  Optional list of rule_ids to run
        min_confidence: Override minimum confidence threshold

    Returns:
        (AuditLog, rectified_frames_dict)
    """

    # ── Set up dual-output logger ─────────────────────────────────────────────
    ddq_log = setup_ddq_logger(log_dir=log_dir, run_id=run_id)
    log_path = get_log_path()


    t0 = time.time()
    import_rule_modules()
    rect_config = _load_rect_config(config)

    if not rect_config["enabled"]:
        logger.info("Rectification disabled in config")
        return AuditLog(), {}

    if min_confidence is not None:
        rect_config["min_confidence"] = min_confidence

    audit_log = AuditLog()
    rectified_frames: dict = {}   # (symbol, source, exchange, tf) -> DataFrame

    # Step 1: Analyse failures (for context logging only)
    """
    failures = _extract_failures(results)
    ddq_log.info("\n" + "═" * 70)
    ddq_log.info("  RECTIFICATION ENGINE — %d detection failures found", len(failures))
    ddq_log.info("  Running %s rectification rules on all loaded data ...",
                "filtered" if rule_filter else "all enabled")
    ddq_log.info("═" * 70)
    """

    # Step 2 & 3: Get rules and build execution plan
    all_rules = get_all_rules()
    if rule_filter:
        all_rules = [(s, f) for s, f in all_rules if s.rule_id in rule_filter]

    # Filter by mode
    if mode.upper() == "EOD":
        all_rules = [(s, f) for s, f in all_rules if s.timeframe.upper() == "EOD"]
    elif mode.upper() == "INTRADAY":
        all_rules = [(s, f) for s, f in all_rules if s.timeframe.upper() == "INTRADAY"]

    ddq_log.info("  Matched %d rectification rules to execute", len(all_rules))

    # Step 4 & 5: Execute rules per symbol per source
    for symbol, sym_data in data_store.items():
        if not isinstance(sym_data, dict):
            continue

        for timeframe in ["eod", "intraday"]:
            if mode.upper() == "EOD" and timeframe != "eod":
                continue
            if mode.upper() == "INTRADAY" and timeframe != "intraday":
                continue

            tf_data = sym_data.get(timeframe, {})
            for exchange, src_dict in tf_data.items():
                for source, df in src_dict.items():
                    if df is None or df.empty:
                        continue

                    # Deep copy — never modify original
                    working_df = df.copy(deep=True)
                    frame_key = (symbol, source, exchange, timeframe.upper())

                    for spec, rule_fn in all_rules:
                        # Check timeframe match
                        if spec.timeframe.upper() != timeframe.upper():
                            continue

                        rule_cfg = _get_rule_config(rect_config, spec.rule_id, spec)
                        if not rule_cfg.get("enabled", True):
                            continue

                        try:
                            rt0 = time.time()
                            working_df, rect_result = rule_fn(
                                working_df, symbol=symbol, source=source,
                                exchange=exchange, config=rule_cfg,
                            )
                            rect_result.elapsed_s = round(time.time() - rt0, 4)

                            # Confidence gate
                            if (rect_result.action == "FIXED" and
                                rect_result.confidence < rect_config["min_confidence"]):
                                rect_result.action = "FLAGGED"
                                rect_result.details += f" [Confidence {rect_result.confidence:.2f} < threshold {rect_config['min_confidence']:.2f}]"

                            # Safety cap
                            if (rect_result.changes_count > rect_config["max_changes_per_source"]
                                and rect_result.action == "FIXED"):
                                rect_result.action = "FLAGGED"
                                rect_result.details += f" [Changes {rect_result.changes_count} > safety cap {rect_config['max_changes_per_source']}]"

                            audit_log.add_result(rect_result)

                            if rect_result.action == "FIXED":
                                _log_rule(spec, rect_result, symbol, source, exchange, timeframe)

                        except Exception as exc:
                            err_result = RectificationResult(
                                rule_id=spec.rule_id, test_id=spec.test_ids[0],
                                symbol=symbol, source=source, timeframe=timeframe.upper(),
                                exchange=exchange, action="FAILED",
                                details=f"Rule execution error: {exc}",
                            )
                            audit_log.add_result(err_result)
                            ddq_log.warning("  ⚠  %s failed on %s/%s: %s",
                                          spec.rule_id, symbol, source, exc)

                    rectified_frames[frame_key] = working_df

    # Step 6: Write output
    elapsed = round(time.time() - t0, 2)
    summary = audit_log.summary()
    ddq_log.info("\n  ─── Rectification Summary ───")
    ddq_log.info("  Rules executed : %d", summary["total_rules_executed"])
    ddq_log.info("  Fixed          : %d", summary["fixed"])
    ddq_log.info("  Flagged        : %d", summary["flagged"])
    ddq_log.info("  Skipped        : %d", summary["skipped"])
    ddq_log.info("  Failed         : %d", summary["failed"])
    ddq_log.info("  Total changes  : %d", summary["total_changes"])
    ddq_log.info("  Avg confidence : %.2f", summary["avg_confidence"])
    ddq_log.info("  Elapsed        : %.1fs", elapsed)

    if not dry_run and rect_config.get("write_rectified", True):
        out_dir = write_rectified(project_root, run_id, rectified_frames)
        audit_log.write_json(out_dir / f"audit_{run_id}.json")
        audit_log.write_csv(out_dir / f"audit_{run_id}.csv")
        audit_log.write_summary_json(out_dir / f"summary_{run_id}.json")
        ddq_log.info("  Output dir     : %s", out_dir)

    return audit_log, rectified_frames


def _log_rule(spec, r, sym, src, exch, tf):
    logger.info("  ✅ RECT-FIX  %-16s  %s/%s/%s/%s  %d changes  (%.3fs)  %s",
                spec.rule_id, sym, src, exch, tf.upper(),
                r.changes_count, r.elapsed_s, r.details[:80])
