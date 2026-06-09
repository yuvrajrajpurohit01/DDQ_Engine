#!/home/trader/trading_system/venv/bin/python3
"""
Downloaded Data DQ Engine — Package Checker
check_packages.py

Verifies all required packages are installed in trading_system venv.

Usage:
    source ~/trading_system/venv/bin/activate
    cd ~/downloaded_data_dq
    python check_packages.py
"""

from __future__ import annotations
import sys
import importlib.util
from pathlib import Path


VENV_PATH = Path.home() / "trading_system" / "venv"

REQUIRED = [
    # (import_name, pip_name, min_version, purpose)
    ("pandas",    "pandas",    "2.2.0",  "Core data processing"),
    ("numpy",     "numpy",     "1.26.0", "Numerical operations"),
    ("scipy",     "scipy",     "1.12.0", "Statistical tests"),
    ("yaml",      "pyyaml",    "6.0.1",  "YAML config loading"),
    ("pydantic",  "pydantic",  "2.0.0",  "Config validation"),
    ("openpyxl",  "openpyxl",  "3.1.0",  "Excel reports"),
    ("jinja2",    "jinja2",    "3.1.0",  "HTML reports"),
    ("requests",  "requests",  "2.31.0", "Webhook alerting"),
    ("structlog", "structlog", "24.0.0", "Structured logging"),
    ("pyarrow",   "pyarrow",   "14.0.0", "Parquet file support"),
]

OPTIONAL = [
    ("pandas_market_calendars", "pandas-market-calendars", "4.3.0",
     "NSE/BSE trading calendar (falls back to bundled YAML if missing)"),
    ("pytest", "pytest", "8.0.0",
     "Running unit tests"),
    ("rich",   "rich",   "13.0.0",
     "Pretty console output during development"),
]

PHASE_LATER = [
    ("apscheduler",      "apscheduler",      "3.10.0", "Phase 5 — live feed scheduling"),
    ("vollib",           "py-vollib",         "1.0.0",  "Phase 3 — options IV calculation"),
    ("psycopg2",         "psycopg2-binary",   "2.9.9",  "Phase 6 — PostgreSQL storage"),
    ("prometheus_client","prometheus-client", "0.20.0", "Phase 6 — metrics endpoint"),
]


def check(import_name: str) -> tuple[bool, str]:
    try:
        m = __import__(import_name)
        return True, getattr(m, "__version__", "installed")
    except ImportError:
        return False, "NOT INSTALLED"


def version_ok(installed: str, minimum: str) -> bool:
    if installed == "NOT INSTALLED":
        return False
    if installed == "installed":
        return True
    try:
        from packaging.version import Version
        return Version(installed) >= Version(minimum)
    except Exception:
        try:
            i = [int(x) for x in installed.split(".")[:3]]
            m = [int(x) for x in minimum.split(".")[:3]]
            return i >= m
        except Exception:
            return True  # can't compare, assume ok


def section(title: str, pkgs: list) -> list[str]:
    print(f"\n  {title}")
    print(f"  {'─'*63}")
    missing = []
    for imp, pip, minv, purpose in pkgs:
        ok_inst, ver = check(imp)
        ok_ver = version_ok(ver, minv)
        if ok_inst and ok_ver:
            icon = "✅"
        elif ok_inst:
            icon = "⚠️ "  # wrong version
        else:
            icon = "❌"
            missing.append(pip)
        print(f"  {icon}  {pip:<32} {ver:<15}  {purpose}")
    return missing


def main() -> None:
    print()
    print("=" * 67)
    print("  Downloaded Data DQ — Package Check")
    print("=" * 67)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Exec    : {sys.executable}")

    if VENV_PATH.exists() and str(VENV_PATH) not in sys.executable:
        print(f"\n  ⚠️   WARNING: Not in trading_system venv!")
        print(f"       Run: source ~/trading_system/venv/bin/activate")
    elif str(VENV_PATH) in sys.executable:
        print(f"  Venv    : ✅ trading_system/venv")

    missing_req = section("REQUIRED (must be installed)", REQUIRED)
    missing_opt = section("OPTIONAL (recommended)", OPTIONAL)
    section("PHASE 3–6 (not needed yet)", PHASE_LATER)

    print()
    print("=" * 67)

    if not missing_req and not missing_opt:
        print("  ✅  All packages ready. Run: python run_ddq.py")
    else:
        if missing_req:
            print(f"  ❌  Missing required packages:")
            print(f"      pip install {' '.join(missing_req)}")
        if missing_opt:
            print(f"\n  ⚠️   Missing optional packages:")
            print(f"      pip install {' '.join(missing_opt)}")
        print(f"\n  Install everything at once:")
        all_missing = missing_req + missing_opt
        print(f"      pip install {' '.join(all_missing)}")
    print()


if __name__ == "__main__":
    main()
