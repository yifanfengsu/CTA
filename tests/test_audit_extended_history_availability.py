from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_extended_history_availability as audit_mod
from history_time_utils import parse_history_range
from history_utils import parse_interval, verify_database_coverage
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData


VT_SYMBOL = "BTCUSDT_SWAP_OKX.GLOBAL"
INST_ID = "BTC-USDT-SWAP"
REQUIRED_SYMBOLS = [
    ("BTCUSDT_SWAP_OKX.GLOBAL", "BTC-USDT-SWAP"),
    ("ETHUSDT_SWAP_OKX.GLOBAL", "ETH-USDT-SWAP"),
    ("SOLUSDT_SWAP_OKX.GLOBAL", "SOL-USDT-SWAP"),
    ("LINKUSDT_SWAP_OKX.GLOBAL", "LINK-USDT-SWAP"),
    ("DOGEUSDT_SWAP_OKX.GLOBAL", "DOGE-USDT-SWAP"),
]
TZ = ZoneInfo("Asia/Shanghai")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def instrument_payload(
    vt_symbol: str = VT_SYMBOL,
    inst_id: str = INST_ID,
    *,
    size: float | None = 0.01,
    pricetick: float | None = 0.1,
    min_volume: float | None = 0.01,
) -> dict[str, Any]:
    symbol = vt_symbol.split(".", maxsplit=1)[0]
    return {
        "vt_symbol": vt_symbol,
        "symbol": symbol,
        "exchange": "GLOBAL",
        "name": inst_id,
        "okx_inst_id": inst_id,
        "product": "SWAP",
        "size": size,
        "pricetick": pricetick,
        "min_volume": min_volume,
        "gateway_name": "OKX",
        "history_data": True,
        "needs_okx_contract_metadata_refresh": False,
    }


def write_history_db(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("create table dbbardata(symbol text, exchange text, interval text, datetime text)")
        connection.executemany(
            "insert into dbbardata(symbol, exchange, interval, datetime) values (?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def minute_rows(
    vt_symbol: str = VT_SYMBOL,
    start: str = "2025-01-01 00:00:00",
    minutes: int = 1440,
    skip: set[int] | None = None,
) -> list[tuple[str, str, str, str]]:
    symbol = vt_symbol.split(".", maxsplit=1)[0]
    current = datetime.fromisoformat(start)
    rows: list[tuple[str, str, str, str]] = []
    skip = skip or set()
    for index in range(minutes):
        if index not in skip:
            rows.append((symbol, "GLOBAL", "1m", current.isoformat(sep=" ", timespec="seconds")))
        current += timedelta(minutes=1)
    return rows


def make_windows(value: str = "2025-01-01:2025-01-01") -> list[audit_mod.AuditWindow]:
    return audit_mod.parse_windows(
        value,
        audit_mod.interval_to_delta("1m"),
        "Asia/Shanghai",
    )


def make_db_bar(vt_symbol: str, dt: datetime) -> BarData:
    symbol, _exchange = vt_symbol.split(".", maxsplit=1)
    return BarData(
        gateway_name="DB",
        symbol=symbol,
        exchange=Exchange.GLOBAL,
        datetime=dt,
        interval=Interval.MINUTE,
        volume=1.0,
        turnover=100.0,
        open_interest=0.0,
        open_price=100.0,
        high_price=101.0,
        low_price=99.0,
        close_price=100.0,
    )


def load_sqlite_bars(path: Path, vt_symbol: str) -> list[BarData]:
    symbol, exchange = vt_symbol.split(".", maxsplit=1)
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            (
                "select datetime from dbbardata "
                "where symbol = ? and exchange = ? and interval = ? "
                "order by datetime"
            ),
            (symbol, exchange, "1m"),
        ).fetchall()
    finally:
        connection.close()
    return [
        make_db_bar(vt_symbol, datetime.fromisoformat(row[0]).replace(tzinfo=TZ))
        for row in rows
    ]


class AuditExtendedHistoryAvailabilityTest(unittest.TestCase):
    def run_temp_audit(
        self,
        root: Path,
        *,
        symbols: list[str] | None = None,
        database_path: Path | None = None,
        windows: list[audit_mod.AuditWindow] | None = None,
        fetcher: Any = audit_mod.fetch_okx_instrument,
        check_okx_listing_metadata: bool = False,
    ) -> dict[str, Any]:
        return audit_mod.run_audit(
            symbols=symbols or [VT_SYMBOL],
            windows=windows or make_windows(),
            interval="1m",
            timezone_name="Asia/Shanghai",
            config_dir=root / "config" / "instruments",
            database_path=database_path or root / "database.db",
            output_dir=root / "reports" / "research" / "extended_history_availability",
            check_okx_listing_metadata=check_okx_listing_metadata,
            okx_timeout=0.1,
            fetcher=fetcher,
        )

    def test_expected_count_for_date_window(self) -> None:
        window = make_windows()[0]

        self.assertEqual(audit_mod.expected_bar_count(window.history_range), 1440)

    def test_expected_count_for_full_2023_extended_window(self) -> None:
        window = make_windows("2023-01-01:2026-03-31")[0]

        self.assertEqual(audit_mod.expected_bar_count(window.history_range), 1707840)

    def test_vt_symbol_mapping_matches_verify_style(self) -> None:
        symbol, exchange = audit_mod.split_vt_symbol("ETHUSDT_SWAP_OKX.GLOBAL")

        self.assertEqual(symbol, "ETHUSDT_SWAP_OKX")
        self.assertEqual(exchange, "GLOBAL")

    def test_audit_coverage_matches_verify_coverage_on_same_sqlite_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            rows = minute_rows(minutes=1440, skip={30, 31})
            write_history_db(db_path, rows)
            history_range = parse_history_range(
                start_arg="2025-01-01",
                end_arg="2025-01-01",
                interval_delta=timedelta(minutes=1),
                timezone_name="Asia/Shanghai",
            )
            interval, _delta = parse_interval("1m")
            fake_db = SimpleNamespace(load_bar_data=lambda *_args: load_sqlite_bars(db_path, VT_SYMBOL))

            verify_coverage = verify_database_coverage(
                symbol="BTCUSDT_SWAP_OKX",
                exchange=Exchange.GLOBAL,
                interval=interval,
                history_range=history_range,
                database=fake_db,
            )
            payload = self.run_temp_audit(root, database_path=db_path)
            audit_record = payload["symbol_windows"][0]

        self.assertEqual(verify_coverage.total_count, 1438)
        self.assertEqual(audit_record["total_count"], verify_coverage.total_count)
        self.assertEqual(audit_record["missing_count"], verify_coverage.missing_count)
        self.assertEqual(audit_record["gap_count"], verify_coverage.gap_count)

    def test_identifies_complete_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows())

            payload = self.run_temp_audit(root, database_path=db_path)
            record = payload["symbol_windows"][0]

        self.assertTrue(record["history_ready"])
        self.assertEqual(record["expected_count"], 1440)
        self.assertEqual(record["total_count"], 1440)
        self.assertEqual(record["missing_count"], 0)
        self.assertEqual(record["gap_count"], 0)
        self.assertEqual(record["query_debug"]["db_symbol_used"], "BTCUSDT_SWAP_OKX")
        self.assertEqual(record["query_debug"]["db_exchange_used"], "GLOBAL")
        self.assertEqual(record["query_debug"]["rows_found_in_window"], 1440)
        self.assertTrue(payload["database"]["database_exists"])
        self.assertTrue(payload["database"]["dbbardata_table_exists"])

    def test_identifies_partial_missing_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(skip={30, 31, 32, 33, 34}))

            payload = self.run_temp_audit(root, database_path=db_path)
            record = payload["symbol_windows"][0]

        self.assertFalse(record["history_ready"])
        self.assertEqual(record["total_count"], 1435)
        self.assertEqual(record["missing_count"], 5)
        self.assertEqual(record["gap_count"], 1)

    def test_generates_missing_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(skip={30, 31, 32, 33, 34}))

            payload = self.run_temp_audit(root, database_path=db_path)
            missing = payload["symbol_windows"][0]["missing_ranges"][0]

        self.assertEqual(missing["start"], "2025-01-01 00:30:00")
        self.assertEqual(missing["end"], "2025-01-01 00:34:00")
        self.assertEqual(missing["missing_count"], 5)

    def test_generates_suggested_download_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(skip={30}))

            payload = self.run_temp_audit(root, database_path=db_path)
            command = payload["missing_ranges"][0]["suggested_download_command"]

        self.assertIn("python scripts/download_okx_history.py", command)
        self.assertIn("--vt-symbol BTCUSDT_SWAP_OKX.GLOBAL", command)
        self.assertIn("--start 2025-01-01", command)
        self.assertIn("--end 2025-01-01", command)
        self.assertIn("--repair-missing", command)

    def test_metadata_incomplete_adds_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(
                root / "config" / "instruments" / "btcusdt_swap_okx.json",
                instrument_payload(size=None),
            )
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows())

            payload = self.run_temp_audit(root, database_path=db_path)
            instrument = payload["instruments"][VT_SYMBOL]
            record = payload["symbol_windows"][0]

        self.assertFalse(instrument["metadata_complete"])
        self.assertTrue(any("metadata_incomplete" in item for item in instrument["warnings"]))
        self.assertTrue(record["history_ready"])
        self.assertTrue(record["ready"])

    def test_metadata_complete_does_not_warn_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows())

            payload = self.run_temp_audit(root, database_path=db_path)
            instrument = payload["instruments"][VT_SYMBOL]
            record = payload["symbol_windows"][0]

        self.assertTrue(instrument["metadata_complete"])
        self.assertFalse(any("metadata_incomplete" in item for item in record["warnings"]))

    def test_listing_metadata_unknown_does_not_fail(self) -> None:
        def fetcher(_inst_id: str, _timeout: float) -> dict[str, Any]:
            return {"instId": INST_ID, "state": "live"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows())

            payload = self.run_temp_audit(
                root,
                database_path=db_path,
                check_okx_listing_metadata=True,
                fetcher=fetcher,
            )
            record = payload["symbol_windows"][0]

        self.assertEqual(record["listing_before_window_start"], "unknown")
        self.assertTrue(record["ready"])
        self.assertTrue(any("listing_before_window_start_unknown" in item for item in record["warnings"]))

    def test_complete_2023_window_five_symbols_ready_with_listing_unknown_warning(self) -> None:
        windows = make_windows("2023-01-01:2026-03-31,2021-01-01:2026-03-31")
        records: list[dict[str, Any]] = []
        for vt_symbol, _inst_id in REQUIRED_SYMBOLS:
            for window in windows:
                is_2023 = window.start_arg == "2023-01-01"
                records.append(
                    {
                        "vt_symbol": vt_symbol,
                        "window": window.label,
                        "ready": is_2023,
                        "missing_count": 0 if is_2023 else 1051200,
                        "expected_count": audit_mod.expected_bar_count(window.history_range),
                        "total_count": audit_mod.expected_bar_count(window.history_range) if is_2023 else 1707840,
                        "metadata_complete": False,
                        "listing_before_window_start": "unknown",
                    }
                )

        readiness = audit_mod.build_readiness(records, windows)
        window_2023 = next(item for item in readiness["windows"] if item["window"] == "2023-01-01:2026-03-31")

        self.assertTrue(readiness["can_enter_extended_trend_research"])
        self.assertEqual(len(window_2023["ready_symbols"]), 5)
        self.assertEqual(window_2023["total_missing_count"], 0)
        self.assertEqual(readiness["blocking_reasons"], [])
        self.assertTrue(any(item.startswith("listing_time_unknown:") for item in readiness["optional_warnings"]))

    def test_only_2025_history_suggests_download_2023_to_2024(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(start="2025-01-01 00:00:00", minutes=1440))
            windows = make_windows("2025-01-01:2025-01-01,2023-01-01:2025-01-01")

            payload = self.run_temp_audit(root, database_path=db_path, windows=windows)
            by_window = {item["window"]: item for item in payload["symbol_windows"]}
            missing = by_window["2023-01-01:2025-01-01"]["missing_ranges"][0]

        self.assertTrue(by_window["2025-01-01:2025-01-01"]["history_ready"])
        self.assertFalse(by_window["2023-01-01:2025-01-01"]["history_ready"])
        self.assertEqual(missing["start"], "2023-01-01 00:00:00")
        self.assertEqual(missing["end"], "2024-12-31 23:59:00")
        self.assertIn("--start 2023-01-01", missing["suggested_download_command"])
        self.assertIn("--end 2024-12-31", missing["suggested_download_command"])

    def test_writes_json_markdown_and_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(root / "config" / "instruments" / "btcusdt_swap_okx.json", instrument_payload())
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(skip={30}))

            payload = self.run_temp_audit(root, database_path=db_path)
            outputs = payload["outputs"]
            for key in ("json", "report", "missing_ranges_csv", "download_plan_csv"):
                self.assertTrue(Path(outputs[key]).exists(), key)
            report_text = Path(outputs["report"]).read_text(encoding="utf-8")

        self.assertIn("Extended History Availability Audit", report_text)
        self.assertIn("Can enter Extended Trend Research", report_text)

    def test_repository_makefile_contains_target(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("audit-extended-history:", text)
        self.assertIn("scripts/audit_extended_history_availability.py", text)


if __name__ == "__main__":
    unittest.main()
