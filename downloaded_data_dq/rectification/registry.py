"""
DDQ Engine — Rectification Rule Registry
downloaded_data_dq/rectification/registry.py

Decorator-based registry mapping detection test_ids to rectification
functions.  Similar pattern to framework.py's dq_test() decorator.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# Global registry:  rule_id -> (RuleSpec, callable)
_RECT_RULES: dict[str, tuple["RuleSpec", Callable]] = {}


@dataclass(frozen=True)
class RuleSpec:
    rule_id:      str           # e.g. RECT-EOD-005
    test_ids:     tuple[str, ...]  # detection tests this rule addresses
    name:         str
    timeframe:    str           # EOD / INTRADAY
    priority:     int           # lower = runs first
    default_conf: float         # default confidence if not in config
    description:  str = ""


def rect_rule(spec: RuleSpec) -> Callable:
    """Decorator that registers a rectification function."""
    def decorator(fn: Callable) -> Callable:
        if spec.rule_id in _RECT_RULES:
            logger.warning("Duplicate rect rule: %s — overwriting", spec.rule_id)
        _RECT_RULES[spec.rule_id] = (spec, fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper._spec = spec
        return wrapper
    return decorator


def get_rules_for_test(test_id: str) -> list[tuple[RuleSpec, Callable]]:
    """Return all rectification rules that address a given detection test_id."""
    matches = []
    for rule_id, (spec, fn) in _RECT_RULES.items():
        if test_id in spec.test_ids:
            matches.append((spec, fn))
    matches.sort(key=lambda x: x[0].priority)
    return matches


def get_all_rules(timeframe: str | None = None) -> list[tuple[RuleSpec, Callable]]:
    """Return all registered rules, optionally filtered by timeframe."""
    rules = list(_RECT_RULES.values())
    if timeframe:
        rules = [(s, f) for s, f in rules if s.timeframe.upper() == timeframe.upper()]
    rules.sort(key=lambda x: x[0].priority)
    return rules


def import_rule_modules():
    """Import all rule modules to trigger registration."""
    import downloaded_data_dq.rectification.rules_eod       # noqa: F401
    import downloaded_data_dq.rectification.rules_intraday   # noqa: F401
