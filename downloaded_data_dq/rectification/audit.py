"""
DDQ Engine — Rectification Audit Trail
downloaded_data_dq/rectification/audit.py

Records every cell-level change made by rectification rules.
Produces JSON and CSV audit files for full traceability.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """One cell-level change made by a rectification rule."""
    rule_id:    str
    symbol:     str
    source:     str
    exchange:   str
    timeframe:  str   # EOD / INTRADAY
    row_index:  int
    column:     str
    old_value:  str
    new_value:  str
    reason:     str
    confidence: float
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RectificationResult:
    """Result of one rectification rule execution on one source/symbol."""
    rule_id:        str
    test_id:        str   # detection test that triggered this rule
    symbol:         str
    source:         str
    timeframe:      str
    exchange:       str
    action:         str   = "SKIPPED"   # FIXED / FLAGGED / SKIPPED / FAILED
    changes_count:  int   = 0
    rows_modified:  int   = 0
    confidence:     float = 0.0
    details:        str   = ""
    elapsed_s:      float = 0.0
    audit_entries:  list  = field(default_factory=list)
    before_status:  str   = ""  # original test status
    after_status:   str   = ""  # re-validation status

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "audit_entries"}
        d["audit_entries_count"] = len(self.audit_entries)
        return d


class AuditLog:
    """Collects all audit entries for a run and writes them out."""

    def __init__(self):
        self.entries: list[AuditEntry] = []
        self.results: list[RectificationResult] = []

    def add_entry(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def add_result(self, result: RectificationResult) -> None:
        self.results.append(result)
        self.entries.extend(result.audit_entries)

    # ── Summary stats ─────────────────────────────────────────────────
    def summary(self) -> dict:
        total_changes = sum(r.changes_count for r in self.results)
        fixed   = sum(1 for r in self.results if r.action == "FIXED")
        flagged = sum(1 for r in self.results if r.action == "FLAGGED")
        skipped = sum(1 for r in self.results if r.action == "SKIPPED")
        failed  = sum(1 for r in self.results if r.action == "FAILED")
        avg_conf = 0.0
        confs = [r.confidence for r in self.results if r.action == "FIXED"]
        if confs:
            avg_conf = sum(confs) / len(confs)
        return {
            "total_rules_executed": len(self.results),
            "total_changes": total_changes,
            "fixed": fixed,
            "flagged": flagged,
            "skipped": skipped,
            "failed": failed,
            "avg_confidence": round(avg_conf, 4),
            "total_audit_entries": len(self.entries),
        }

    # ── Write outputs ─────────────────────────────────────────────────
    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
            "audit_entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("  Audit JSON: %s (%d entries)", path, len(self.entries))

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries:
            path.write_text("rule_id,symbol,source,exchange,timeframe,row_index,column,old_value,new_value,reason,confidence,timestamp\n")
            return
        fields = list(asdict(self.entries[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in self.entries:
                w.writerow(asdict(e))
        logger.info("  Audit CSV : %s (%d rows)", path, len(self.entries))

    def write_summary_json(self, path: Path, run_meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        if run_meta:
            data["run_meta"] = run_meta
        data["results"] = [r.to_dict() for r in self.results]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
