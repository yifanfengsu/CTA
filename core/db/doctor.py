#!/usr/bin/env python3
"""Headless environment doctor for the OKX + vn.py CTA bootstrap project."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = PROJECT_ROOT / "logs"
VNTRADER_DIR: Path = PROJECT_ROOT / ".vntrader"
VNTRADER_LOG_DIR: Path = VNTRADER_DIR / "log"
ENV_FILE: Path = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE: Path = PROJECT_ROOT / ".env.example"
LOG_FILE: Path = LOG_DIR / "doctor.log"


@dataclass(frozen=True, slots=True)
class PackageCheck:
    """Describe a package that must be importable in the runtime environment."""

    module_name: str
    distribution_name: str
    pip_name: str


@dataclass(frozen=True, slots=True)
class PackageResult:
    """Store import and version check result for one Python package."""

    check: PackageCheck
    ok: bool
    version: str | None
    error: str | None = None


PACKAGE_CHECKS: tuple[PackageCheck, ...] = (
    PackageCheck("vnpy", "vnpy", "vnpy"),
    PackageCheck("vnpy_ctastrategy", "vnpy_ctastrategy", "vnpy_ctastrategy"),
    PackageCheck("vnpy_okx", "vnpy_okx", "vnpy_okx"),
    PackageCheck("vnpy_sqlite", "vnpy_sqlite", "vnpy_sqlite"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run headless environment checks for the OKX vn.py CTA project."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> logging.Logger:
    """Configure console and file logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger: logging.Logger = logging.getLogger("doctor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def prepare_runtime_paths() -> None:
    """Keep vn.py runtime files inside the project root before importing vn.py."""

    VNTRADER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # vn.py prefers an existing .vntrader under the current working directory.
    os.chdir(PROJECT_ROOT)


def get_package_result(check: PackageCheck) -> PackageResult:
    """Import one package and fetch its installed distribution version."""

    version: str | None = None
    error: str | None = None

    try:
        version = importlib.metadata.version(check.distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception as exc:  # pragma: no cover - defensive only
        error = f"metadata error: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module(check.module_name)
        return PackageResult(check=check, ok=True, version=version, error=error)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return PackageResult(check=check, ok=False, version=version, error=error)


def check_okx_gateway() -> tuple[bool, str]:
    """Validate that OkxGateway can be imported from vnpy_okx."""

    try:
        from vnpy_okx import OkxGateway

        exchanges = ",".join(exchange.value for exchange in OkxGateway.exchanges)
        detail = (
            f"import ok | class={OkxGateway.__module__}.{OkxGateway.__name__} | "
            f"default_name={OkxGateway.default_name} | exchanges={exchanges}"
        )
        return True, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def read_database_settings() -> tuple[bool, str, str, str | None]:
    """Read vn.py database settings and check whether sqlite is active."""

    try:
        from vnpy.trader.setting import SETTINGS

        database_name = str(SETTINGS.get("database.name"))
        database_file = str(SETTINGS.get("database.database"))
        if database_name != "sqlite":
            return (
                False,
                database_name,
                database_file,
                "Expected database.name=sqlite for the vnpy_sqlite bootstrap.",
            )
        return True, database_name, database_file, None
    except Exception as exc:
        return False, "unknown", "unknown", f"{type(exc).__name__}: {exc}"


def build_fix_suggestions(
    package_results: list[PackageResult],
    okx_gateway_ok: bool,
    database_ok: bool,
) -> list[str]:
    """Build actionable repair suggestions for failed checks."""

    suggestions: list[str] = []

    failed_packages = [result for result in package_results if not result.ok]
    if failed_packages:
        missing_names = " ".join(result.check.pip_name for result in failed_packages)
        suggestions.append("Activate the project virtualenv: source .venv/bin/activate")
        suggestions.append(f"Install missing packages: python -m pip install {missing_names}")

    if not okx_gateway_ok:
        suggestions.append("Reinstall OKX gateway package: python -m pip install --force-reinstall vnpy_okx")

    if not database_ok:
        suggestions.append("Ensure sqlite backend is active: python -m pip install --force-reinstall vnpy_sqlite")

    if not ENV_FILE.exists():
        suggestions.append("Create runtime env file before gateway scripts: cp .env.example .env")

    return suggestions


def log_system_info(logger: logging.Logger, invocation_cwd: Path) -> None:
    """Print basic runtime information."""

    logger.info("System")
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Python executable: %s", sys.executable)
    logger.info("Operating system: %s", platform.platform())
    logger.info("Invocation working directory: %s", invocation_cwd)
    logger.info("Project root: %s", PROJECT_ROOT)


def log_package_results(logger: logging.Logger, package_results: list[PackageResult]) -> None:
    """Print package import results."""

    logger.info("")
    logger.info("Dependencies")
    for result in package_results:
        status = "OK" if result.ok else "FAIL"
        version = result.version or "unknown"
        if result.ok:
            logger.info("[%s] %s | version=%s", status, result.check.module_name, version)
        else:
            logger.info(
                "[%s] %s | version=%s | error=%s",
                status,
                result.check.module_name,
                version,
                result.error,
            )


def main() -> int:
    """Run the headless project doctor."""

    args = parse_args()
    invocation_cwd: Path = Path.cwd()
    prepare_runtime_paths()
    logger = configure_logging(args.log_level)

    try:
        log_system_info(logger, invocation_cwd)

        package_results = [get_package_result(check) for check in PACKAGE_CHECKS]
        log_package_results(logger, package_results)

        logger.info("")
        logger.info("Gateway")
        okx_gateway_ok, okx_gateway_detail = check_okx_gateway()
        logger.info(
            "[%s] OkxGateway | %s",
            "OK" if okx_gateway_ok else "FAIL",
            okx_gateway_detail,
        )

        logger.info("")
        logger.info("vn.py Settings")
        database_ok, database_name, database_file, database_error = read_database_settings()
        logger.info("database.name: %s", database_name)
        logger.info("database.database: %s", database_file)
        if database_error:
            logger.info("database.check: %s", database_error)

        logger.info("")
        logger.info("Runtime Files")
        logger.info(".env.example exists: %s", ENV_EXAMPLE_FILE.exists())
        logger.info(".env exists: %s", ENV_FILE.exists())
        logger.info(".vntrader path: %s", VNTRADER_DIR)
        logger.info("doctor log: %s", LOG_FILE)

        checks_ok = all(result.ok for result in package_results) and okx_gateway_ok and database_ok
        suggestions = build_fix_suggestions(package_results, okx_gateway_ok, database_ok)

        if suggestions:
            logger.info("")
            logger.info("Suggested Fixes")
            for suggestion in suggestions:
                logger.info("- %s", suggestion)

        logger.info("")
        logger.info("Doctor result: %s", "PASS" if checks_ok else "FAIL")
        return 0 if checks_ok else 1
    except Exception:
        logger.exception("Doctor crashed unexpectedly.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
