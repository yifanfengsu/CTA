#!/usr/bin/env python3
"""Audit feasibility for external regime classifier research only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, print_json_block, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, parse_history_range


DEFAULT_SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "external_regime_feasibility"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".vntrader" / "database.db"
DEFAULT_FUNDING_DIR = PROJECT_ROOT / "data" / "funding" / "okx"
REQUIRED_COMPUTABLE_FEATURES = 8
FUNDING_SOURCE = "OKX Historical Market Data"

OKX_DOCS_URL = "https://www.okx.com/docs-v5/en/"

CSV_COLUMNS = [
    "feature_group",
    "feature_name",
    "availability",
    "computable_now",
    "data_source",
    "coverage_start",
    "coverage_end",
    "requires_download",
    "requires_private_api_key",
    "okx_public_endpoint",
    "uses_failed_policy_output",
    "notes",
]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One candidate regime feature and its source requirements."""

    feature_group: str
    feature_name: str
    data_source: str
    okx_public_endpoint: str = ""
    notes: str = ""


INTERNAL_MARKET_FEATURES = [
    FeatureSpec("internal_market_features", "trend breadth", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "cross-symbol correlation", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "cross-symbol dispersion", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "realized volatility regime", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "market-wide ATR percentile", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "number of symbols above EMA50/EMA200", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "number of symbols in strong trend", "local 1m OHLCV sqlite"),
    FeatureSpec("internal_market_features", "market-wide drawdown / rebound state", "local 1m OHLCV sqlite"),
]

FUNDING_FEATURES = [
    FeatureSpec("funding_features", "average funding rate across symbols", "local OKX funding CSV"),
    FeatureSpec("funding_features", "funding dispersion", "local OKX funding CSV"),
    FeatureSpec("funding_features", "extreme funding count", "local OKX funding CSV"),
    FeatureSpec("funding_features", "funding trend", "local OKX funding CSV"),
    FeatureSpec("funding_features", "funding sign breadth", "local OKX funding CSV"),
]

MISSING_EXTERNAL_FEATURES = [
    FeatureSpec(
        "missing_external_features",
        "open interest",
        "OKX public/trading statistics endpoint; local dbbardata.open_interest is zero-only",
        "/api/v5/public/open-interest; /api/v5/rubik/stat/contracts/open-interest-history",
        "No local non-zero open-interest series was found for the requested symbols.",
    ),
    FeatureSpec(
        "missing_external_features",
        "long/short ratio",
        "OKX Trading Statistics public endpoint",
        "/api/v5/rubik/stat/contracts/long-short-account-ratio",
        "Not present in local data directories; historical backfill limits still need a separate downloader audit.",
    ),
    FeatureSpec(
        "missing_external_features",
        "taker buy/sell volume",
        "OKX Trading Statistics public endpoint",
        "/api/v5/rubik/stat/taker-volume; /api/v5/rubik/stat/contracts/taker-volume",
        "Not present in local data directories; useful as an external flow regime feature after download.",
    ),
    FeatureSpec(
        "missing_external_features",
        "premium index / basis",
        "OKX Public Data / Trading Statistics public endpoint",
        "/api/v5/public/premium-history; OKX basis statistics endpoint",
        "No local basis or premium-index time series was found.",
    ),
    FeatureSpec(
        "missing_external_features",
        "mark/index price divergence",
        "OKX Public Data public endpoint",
        "/api/v5/public/mark-price; /api/v5/public/index-tickers; mark/index candlestick endpoints",
        "Current local 1m market bars are last-trade bars, not mark/index bars.",
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Audit external regime classifier feasibility.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database-path", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--funding-dir", default=str(DEFAULT_FUNDING_DIR))
    parser.add_argument("--json", action="store_true", help="Print full JSON payload after writing outputs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_symbol_list(value: str) -> list[str]:
    """Parse comma/space separated vt_symbols while preserving order."""

    symbols: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", value):
        symbol = token.strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if not symbols:
        raise ValueError("--symbols must contain at least one vt_symbol")
    return symbols


def split_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """Split a vn.py vt_symbol into database symbol/exchange strings."""

    symbol, separator, exchange = vt_symbol.partition(".")
    if not separator or not symbol or not exchange:
        raise ValueError(f"Invalid vt_symbol: {vt_symbol!r}")
    return symbol, exchange


def vt_symbol_to_okx_inst_id(vt_symbol: str) -> str:
    """Best-effort conversion from local vt_symbol to OKX swap instrument id."""

    symbol, _exchange = split_vt_symbol(vt_symbol)
    if symbol.endswith("_SWAP_OKX"):
        base_quote = symbol[: -len("_SWAP_OKX")]
        if base_quote.endswith("USDT"):
            base = base_quote[:-4]
            return f"{base}-USDT-SWAP"
    return symbol.replace("_", "-")


def sqlite_table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a sqlite table exists."""

    row = connection.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def sqlite_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return known columns for a sqlite table."""

    try:
        return {str(row[1]) for row in connection.execute(f"pragma table_info({table_name})").fetchall()}
    except sqlite3.DatabaseError:
        return set()


def format_query_datetime(value: datetime) -> str:
    """Format a local timestamp for the sqlite database query."""

    return value.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def inspect_market_data(
    symbols: list[str],
    database_path: Path,
    history_range: HistoryRange,
) -> dict[str, Any]:
    """Inspect local OHLCV coverage and open-interest presence."""

    expected_count = int((history_range.end_exclusive - history_range.start) / history_range.interval_delta)
    query_start = format_query_datetime(history_range.start)
    query_end = format_query_datetime(history_range.end_exclusive)
    output: dict[str, Any] = {
        "database_path": str(database_path),
        "database_exists": database_path.exists(),
        "dbbardata_table_exists": False,
        "expected_bar_count_per_symbol": expected_count,
        "symbols": [],
        "market_data_available": False,
        "market_data_complete": False,
        "open_interest_nonzero_available": False,
    }
    if not database_path.exists():
        return output

    connection = sqlite3.connect(database_path)
    try:
        if not sqlite_table_exists(connection, "dbbardata"):
            return output
        output["dbbardata_table_exists"] = True
        columns = sqlite_columns(connection, "dbbardata")
        has_open_interest = "open_interest" in columns
        for vt_symbol in symbols:
            db_symbol, db_exchange = split_vt_symbol(vt_symbol)
            row = connection.execute(
                (
                    "select count(distinct datetime), min(datetime), max(datetime) "
                    "from dbbardata "
                    "where symbol = ? and exchange = ? and interval = ? "
                    "and datetime >= ? and datetime < ?"
                ),
                (db_symbol, db_exchange, "1m", query_start, query_end),
            ).fetchone()
            row_count = int(row[0] or 0)
            open_interest_nonzero_count = 0
            if has_open_interest:
                oi_row = connection.execute(
                    (
                        "select count(*) from dbbardata "
                        "where symbol = ? and exchange = ? and interval = ? "
                        "and datetime >= ? and datetime < ? "
                        "and open_interest is not null and open_interest != 0"
                    ),
                    (db_symbol, db_exchange, "1m", query_start, query_end),
                ).fetchone()
                open_interest_nonzero_count = int(oi_row[0] or 0)
            output["symbols"].append(
                {
                    "vt_symbol": vt_symbol,
                    "db_symbol": db_symbol,
                    "db_exchange": db_exchange,
                    "row_count": row_count,
                    "expected_count": expected_count,
                    "first_datetime": row[1],
                    "last_datetime": row[2],
                    "coverage_complete": row_count == expected_count,
                    "open_interest_nonzero_count": open_interest_nonzero_count,
                    "open_interest_local_available": open_interest_nonzero_count > 0,
                }
            )
    finally:
        connection.close()

    symbol_rows = output["symbols"]
    output["market_data_available"] = bool(symbol_rows and all(item["row_count"] > 0 for item in symbol_rows))
    output["market_data_complete"] = bool(symbol_rows and all(item["coverage_complete"] for item in symbol_rows))
    output["open_interest_nonzero_available"] = bool(
        symbol_rows and all(item["open_interest_nonzero_count"] > 0 for item in symbol_rows)
    )
    return output


def parse_csv_timestamp(row: dict[str, str]) -> datetime | None:
    """Parse funding timestamp from a funding CSV row."""

    raw_ms = str(row.get("funding_time") or "").strip()
    if raw_ms:
        try:
            return datetime.fromtimestamp(int(raw_ms) / 1000.0)
        except (TypeError, ValueError, OSError):
            return None
    for column in ("funding_time_local", "funding_time_utc"):
        raw_iso = str(row.get(column) or "").strip()
        if not raw_iso:
            continue
        if raw_iso.endswith("Z"):
            raw_iso = raw_iso[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw_iso).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def inspect_funding_csv(path: Path, history_range: HistoryRange) -> dict[str, Any]:
    """Inspect one local OKX funding CSV for coarse coverage."""

    output: dict[str, Any] = {
        "csv_path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "first_time": None,
        "last_time": None,
        "large_gap_count": 0,
        "coverage_complete": False,
        "warnings": [],
    }
    if not path.exists():
        output["warnings"].append("missing_funding_csv")
        return output

    timestamps: list[datetime] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = parse_csv_timestamp(row)
            if parsed is not None:
                timestamps.append(parsed)

    unique_sorted = sorted(set(timestamps))
    output["row_count"] = len(timestamps)
    if not unique_sorted:
        output["warnings"].append("empty_or_unparseable_funding_csv")
        return output

    first_time = unique_sorted[0]
    last_time = unique_sorted[-1]
    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(unique_sorted, unique_sorted[1:])
        if (right - left).total_seconds() / 3600.0 > 24.0
    ]
    start_date = history_range.start.date()
    end_date = history_range.end_display.date()
    covers_requested_dates = first_time.date() <= start_date and last_time.date() >= end_date
    output.update(
        {
            "first_time": first_time.isoformat(sep=" ", timespec="seconds"),
            "last_time": last_time.isoformat(sep=" ", timespec="seconds"),
            "large_gap_count": len(gaps),
            "coverage_complete": bool(covers_requested_dates and not gaps),
        }
    )
    if not covers_requested_dates:
        output["warnings"].append("requested_date_window_not_covered")
    if gaps:
        output["warnings"].append(f"large_gap_count={len(gaps)}")
    return output


def inspect_funding_data(
    symbols: list[str],
    funding_dir: Path,
    history_range: HistoryRange,
    start_arg: str,
    end_arg: str,
) -> dict[str, Any]:
    """Inspect local OKX funding CSV coverage."""

    rows = []
    for vt_symbol in symbols:
        inst_id = vt_symbol_to_okx_inst_id(vt_symbol)
        path = funding_dir / f"{inst_id}_funding_{start_arg}_{end_arg}.csv"
        row = inspect_funding_csv(path, history_range)
        row["vt_symbol"] = vt_symbol
        row["inst_id"] = inst_id
        rows.append(row)
    return {
        "funding_dir": str(funding_dir),
        "funding_source": FUNDING_SOURCE,
        "symbols": rows,
        "funding_data_available": bool(rows and all(item["exists"] and item["row_count"] > 0 for item in rows)),
        "funding_data_complete": bool(rows and all(item["coverage_complete"] for item in rows)),
    }


def build_feature_rows(
    *,
    market_data_complete: bool,
    funding_data_complete: bool,
    coverage_start: str,
    coverage_end: str,
) -> list[dict[str, Any]]:
    """Build all feature rows with availability classification."""

    rows: list[dict[str, Any]] = []

    for feature in INTERNAL_MARKET_FEATURES:
        rows.append(
            {
                "feature_group": feature.feature_group,
                "feature_name": feature.feature_name,
                "availability": "available" if market_data_complete else "blocked_by_market_data_coverage",
                "computable_now": market_data_complete,
                "data_source": feature.data_source,
                "coverage_start": coverage_start if market_data_complete else "",
                "coverage_end": coverage_end if market_data_complete else "",
                "requires_download": False,
                "requires_private_api_key": False,
                "okx_public_endpoint": "",
                "uses_failed_policy_output": False,
                "notes": "Derived from existing OHLCV bars; classifier feature only, not entry/exit reuse.",
            }
        )

    for feature in FUNDING_FEATURES:
        rows.append(
            {
                "feature_group": feature.feature_group,
                "feature_name": feature.feature_name,
                "availability": "available" if funding_data_complete else "blocked_by_funding_data_coverage",
                "computable_now": funding_data_complete,
                "data_source": feature.data_source,
                "coverage_start": coverage_start if funding_data_complete else "",
                "coverage_end": coverage_end if funding_data_complete else "",
                "requires_download": False,
                "requires_private_api_key": False,
                "okx_public_endpoint": "/api/v5/public/funding-rate-history",
                "uses_failed_policy_output": False,
                "notes": "Derived from existing local funding CSVs; classifier feature only.",
            }
        )

    for feature in MISSING_EXTERNAL_FEATURES:
        rows.append(
            {
                "feature_group": feature.feature_group,
                "feature_name": feature.feature_name,
                "availability": "missing_locally_public_no_key_download_required",
                "computable_now": False,
                "data_source": feature.data_source,
                "coverage_start": "",
                "coverage_end": "",
                "requires_download": True,
                "requires_private_api_key": False,
                "okx_public_endpoint": feature.okx_public_endpoint,
                "uses_failed_policy_output": False,
                "notes": feature.notes,
            }
        )

    return rows


def evaluate_decision(
    feature_rows: list[dict[str, Any]],
    *,
    market_data_complete: bool,
    funding_data_complete: bool,
    requested_start: str,
    requested_end: str,
) -> dict[str, Any]:
    """Evaluate research-only feasibility against the stated gate."""

    computable_rows = [row for row in feature_rows if bool(row.get("computable_now"))]
    no_failed_policy_reuse = all(not bool(row.get("uses_failed_policy_output")) for row in computable_rows)
    no_private_key_needed = all(not bool(row.get("requires_private_api_key")) for row in computable_rows)
    data_coverage_ok = bool(market_data_complete and requested_start <= DEFAULT_START and requested_end >= DEFAULT_END)
    feature_count_ok = len(computable_rows) >= REQUIRED_COMPUTABLE_FEATURES
    classifier_only = True
    allowed = bool(feature_count_ok and no_failed_policy_reuse and data_coverage_ok and no_private_key_needed and classifier_only)

    blockers: list[str] = []
    if not feature_count_ok:
        blockers.append(
            f"computable_regime_features_below_threshold: {len(computable_rows)} < {REQUIRED_COMPUTABLE_FEATURES}"
        )
    if not no_failed_policy_reuse:
        blockers.append("computable_feature_reuses_failed_policy_output")
    if not data_coverage_ok:
        blockers.append("market_data_coverage_not_complete_for_2023_2026")
    if not no_private_key_needed:
        blockers.append("private_api_key_required")
    if not classifier_only:
        blockers.append("research_goal_not_classifier_only")

    return {
        "external_regime_classifier_research_allowed": allowed,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "current_v3_family_failed": True,
        "current_v3_family_failed_after_actual_funding": True,
        "no_policy_can_be_traded": True,
        "classifier_research_only": classifier_only,
        "computable_regime_feature_count": len(computable_rows),
        "required_computable_regime_feature_count": REQUIRED_COMPUTABLE_FEATURES,
        "no_failed_policy_entry_exit_reuse": no_failed_policy_reuse,
        "no_private_api_key_required_for_computable_features": no_private_key_needed,
        "market_data_complete": market_data_complete,
        "funding_data_complete": funding_data_complete,
        "data_coverage_ok": data_coverage_ok,
        "blocking_reasons": blockers,
        "recommended_next_step": (
            "Proceed to research-only external regime classifier feature design and label audit; do not build a strategy."
            if allowed
            else "Pause strategy development until enough non-policy regime data is available."
        ),
    }


def bool_text(value: Any) -> str:
    """Render bool-ish values as lower-case text."""

    return str(bool(value)).lower()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] = CSV_COLUMNS) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_proposed_feature_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build proposed research-only classifier feature CSV rows."""

    proposed = []
    for row in feature_rows:
        if not bool(row.get("computable_now")):
            continue
        proposed.append(
            {
                "feature_group": row["feature_group"],
                "feature_name": row["feature_name"],
                "input_data": row["data_source"],
                "classifier_role": "candidate_regime_feature",
                "status": "research_only_candidate",
                "guardrail": "Do not convert directly into entry/exit logic; no demo/live.",
            }
        )
    return proposed


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render the audit Markdown report."""

    decision = payload["decision"]
    market = payload["data_status"]["market_data"]
    funding = payload["data_status"]["funding_data"]
    feature_rows = payload["features"]
    available_internal = [
        row["feature_name"] for row in feature_rows if row["feature_group"] == "internal_market_features" and row["computable_now"]
    ]
    available_funding = [
        row["feature_name"] for row in feature_rows if row["feature_group"] == "funding_features" and row["computable_now"]
    ]
    missing_external = [row for row in feature_rows if row["feature_group"] == "missing_external_features"]
    downloadable_no_key = [row for row in missing_external if row["requires_download"] and not row["requires_private_api_key"]]
    private_key_required = [row for row in missing_external if row["requires_private_api_key"]]
    fully_missing = [
        row for row in missing_external if row["requires_download"] and not row["computable_now"]
    ]

    def bullet_names(names: list[str]) -> str:
        return "\n".join(f"- {name}" for name in names) if names else "- none"

    missing_table = [
        "| feature | local status | key required | public endpoint |",
        "|---|---|---|---|",
    ]
    for row in missing_external:
        missing_table.append(
            f"| {row['feature_name']} | {row['availability']} | "
            f"{bool_text(row['requires_private_api_key'])} | {row['okx_public_endpoint']} |"
        )

    next_scope = (
        "- Build a research-only dataset of classifier features from existing OHLCV and funding data.\n"
        "- Define labels independently from failed policy entry/exit outcomes.\n"
        "- Audit train/validation/oos stability before any strategy proposal.\n"
        "- Keep strategy_development_allowed=false and demo_live_allowed=false."
    )
    if not decision["external_regime_classifier_research_allowed"]:
        next_scope = "- Pause strategy development; first resolve the blockers listed below."

    blockers = bullet_names(decision["blocking_reasons"])

    return (
        "# External Regime Classifier Feasibility Audit\n\n"
        "## Scope\n"
        f"- symbols={', '.join(payload['scope']['symbols'])}\n"
        f"- start={payload['scope']['start']}\n"
        f"- end={payload['scope']['end']}\n"
        f"- timezone={payload['scope']['timezone']}\n"
        "- audit_type=research_only_feasibility\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n\n"
        "## Data Status\n"
        f"- market_data_complete={bool_text(market['market_data_complete'])}\n"
        f"- funding_data_complete={bool_text(funding['funding_data_complete'])}\n"
        f"- coverage_window={payload['scope']['start']} to {payload['scope']['end']}\n"
        f"- funding_source={funding['funding_source']}\n"
        f"- local_open_interest_nonzero_available={bool_text(market['open_interest_nonzero_available'])}\n\n"
        "## Required Questions\n"
        "1. 当前不扩大币种的前提下，是否还能做新的趋势研究？\n"
        f"   - {('可以，但仅限 external regime classifier research，不是策略开发。' if decision['external_regime_classifier_research_allowed'] else '不建议；应暂停策略开发。')}\n"
        "2. 当前已有数据能构造哪些 regime 特征？\n"
        "   - internal market features:\n"
        f"{bullet_names(available_internal)}\n"
        "   - funding features:\n"
        f"{bullet_names(available_funding)}\n"
        "3. 哪些特征需要额外下载？\n"
        f"{bullet_names([row['feature_name'] for row in missing_external])}\n"
        "4. 哪些特征需要 API key？\n"
        f"{bullet_names([row['feature_name'] for row in private_key_required])}\n"
        "5. 哪些特征可以无密钥从 OKX public endpoint 获取？\n"
        f"{bullet_names([row['feature_name'] for row in downloadable_no_key])}\n"
        "6. 哪些特征完全缺失？\n"
        f"{bullet_names([row['feature_name'] for row in fully_missing])}\n"
        "7. 是否建议进入 External Regime Classifier Research？\n"
        f"   - external_regime_classifier_research_allowed={bool_text(decision['external_regime_classifier_research_allowed'])}\n"
        "8. 如果建议，下一步研究范围是什么？\n"
        f"{next_scope}\n"
        "9. 如果不建议，是否应暂停策略开发？\n"
        f"   - {('否；但仍禁止策略开发和 demo/live，只允许 classifier research。' if decision['external_regime_classifier_research_allowed'] else '是，应暂停策略开发。')}\n\n"
        "## Missing External Feature Detail\n"
        f"{chr(10).join(missing_table)}\n\n"
        "## Decision Gate\n"
        f"- computable_regime_feature_count={decision['computable_regime_feature_count']}\n"
        f"- required_computable_regime_feature_count={decision['required_computable_regime_feature_count']}\n"
        f"- no_failed_policy_entry_exit_reuse={bool_text(decision['no_failed_policy_entry_exit_reuse'])}\n"
        f"- no_private_api_key_required_for_computable_features={bool_text(decision['no_private_api_key_required_for_computable_features'])}\n"
        f"- data_coverage_ok={bool_text(decision['data_coverage_ok'])}\n"
        f"- external_regime_classifier_research_allowed={bool_text(decision['external_regime_classifier_research_allowed'])}\n"
        "- strategy_development_allowed=false\n"
        "- demo_live_allowed=false\n"
        f"- recommended_next_step={decision['recommended_next_step']}\n\n"
        "## Blockers\n"
        f"{blockers}\n\n"
        "## Source Notes\n"
        f"- OKX public API documentation: {OKX_DOCS_URL}\n"
        "- Public endpoint availability here is a feasibility classification only; historical backfill depth still needs a separate downloader/probe before model training.\n"
    )


def build_payload(
    *,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    database_path: Path,
    funding_dir: Path,
) -> dict[str, Any]:
    """Build the full audit payload."""

    history_range = parse_history_range(
        start_arg=start,
        end_arg=end,
        interval_delta=timedelta(minutes=1),
        timezone_name=timezone_name,
    )
    market_data = inspect_market_data(symbols, database_path, history_range)
    funding_data = inspect_funding_data(symbols, funding_dir, history_range, start, end)
    feature_rows = build_feature_rows(
        market_data_complete=bool(market_data["market_data_complete"]),
        funding_data_complete=bool(funding_data["funding_data_complete"]),
        coverage_start=start,
        coverage_end=end,
    )
    decision = evaluate_decision(
        feature_rows,
        market_data_complete=bool(market_data["market_data_complete"]),
        funding_data_complete=bool(funding_data["funding_data_complete"]),
        requested_start=start,
        requested_end=end,
    )
    return {
        "scope": {
            "symbols": symbols,
            "okx_inst_ids": [vt_symbol_to_okx_inst_id(symbol) for symbol in symbols],
            "start": start,
            "end": end,
            "timezone": timezone_name,
            "coverage_window": f"{start} to {end}",
        },
        "data_status": {
            "market_data": market_data,
            "funding_data": funding_data,
        },
        "features": feature_rows,
        "decision": decision,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write all required report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = payload["features"]
    missing_rows = [row for row in feature_rows if row["feature_group"] == "missing_external_features"]
    proposed_rows = build_proposed_feature_rows(feature_rows)
    proposed_columns = ["feature_group", "feature_name", "input_data", "classifier_role", "status", "guardrail"]

    paths = {
        "markdown_report": str(output_dir / "external_regime_feasibility_report.md"),
        "json_report": str(output_dir / "external_regime_feasibility.json"),
        "available_features_csv": str(output_dir / "available_features.csv"),
        "missing_features_csv": str(output_dir / "missing_features.csv"),
        "proposed_regime_features_csv": str(output_dir / "proposed_regime_features.csv"),
    }

    Path(paths["json_report"]).write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["markdown_report"]).write_text(render_markdown_report(payload), encoding="utf-8")
    write_csv(Path(paths["available_features_csv"]), feature_rows)
    write_csv(Path(paths["missing_features_csv"]), missing_rows)
    write_csv(Path(paths["proposed_regime_features_csv"]), proposed_rows, columns=proposed_columns)
    return paths


def run_audit(
    *,
    symbols: list[str],
    start: str,
    end: str,
    timezone_name: str,
    output_dir: Path,
    database_path: Path,
    funding_dir: Path,
) -> dict[str, Any]:
    """Run audit and write outputs."""

    payload = build_payload(
        symbols=symbols,
        start=start,
        end=end,
        timezone_name=timezone_name,
        database_path=database_path,
        funding_dir=funding_dir,
    )
    output_paths = write_outputs(output_dir, payload)
    payload["output_paths"] = output_paths
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    symbols = parse_symbol_list(args.symbols)
    payload = run_audit(
        symbols=symbols,
        start=args.start,
        end=args.end,
        timezone_name=args.timezone,
        output_dir=resolve_path(args.output_dir),
        database_path=resolve_path(args.database_path),
        funding_dir=resolve_path(args.funding_dir),
    )
    summary = {
        "external_regime_classifier_research_allowed": payload["decision"][
            "external_regime_classifier_research_allowed"
        ],
        "strategy_development_allowed": payload["decision"]["strategy_development_allowed"],
        "demo_live_allowed": payload["decision"]["demo_live_allowed"],
        "computable_regime_feature_count": payload["decision"]["computable_regime_feature_count"],
        "market_data_complete": payload["decision"]["market_data_complete"],
        "funding_data_complete": payload["decision"]["funding_data_complete"],
        "output_dir": str(resolve_path(args.output_dir)),
    }
    print_json_block("External regime classifier feasibility summary:", payload if args.json else summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
