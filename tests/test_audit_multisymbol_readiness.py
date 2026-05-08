from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_multisymbol_readiness as audit_mod


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def instrument_payload(
    vt_symbol: str,
    inst_id: str,
    *,
    size: float | None = 1.0,
    pricetick: float | None = 0.01,
    min_volume: float | None = 0.01,
    needs_refresh: bool = False,
) -> dict[str, Any]:
    symbol = vt_symbol.split(".", maxsplit=1)[0]
    payload: dict[str, Any] = {
        "vt_symbol": vt_symbol,
        "symbol": symbol,
        "exchange": "GLOBAL",
        "okx_inst_id": inst_id,
        "name": inst_id,
        "product": "SWAP",
        "size": size,
        "pricetick": pricetick,
        "min_volume": min_volume,
        "gateway_name": "OKX",
        "history_data": True,
        "needs_okx_contract_metadata_refresh": needs_refresh,
    }
    return payload


def write_makefile(path: Path, *, batch_targets: bool = True) -> None:
    lines = [
        ".PHONY: audit-multisymbol\n",
        "audit-multisymbol:\n",
        "\t.venv/bin/python scripts/audit_multisymbol_readiness.py\n",
    ]
    if batch_targets:
        lines.extend(
            [
                ".PHONY: download-history-batch verify-history-batch\n",
                "download-history-batch:\n",
                "\t.venv/bin/python scripts/download_okx_history.py --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL\n",
                "verify-history-batch:\n",
                "\t.venv/bin/python scripts/verify_okx_history.py --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_history_db(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "create table dbbardata(symbol text, exchange text, interval text, datetime text)"
        )
        connection.executemany(
            "insert into dbbardata(symbol, exchange, interval, datetime) values (?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def minute_rows(
    vt_symbol: str,
    start: str = "2025-01-01 00:00:00",
    minutes: int = 1440,
) -> list[tuple[str, str, str, str]]:
    symbol = vt_symbol.split(".", maxsplit=1)[0]
    current = datetime.fromisoformat(start)
    rows: list[tuple[str, str, str, str]] = []
    for _ in range(minutes):
        rows.append((symbol, "GLOBAL", "1m", current.isoformat(sep=" ", timespec="seconds")))
        current += timedelta(minutes=1)
    return rows


REQUIRED_SYMBOLS = [
    ("BTCUSDT_SWAP_OKX.GLOBAL", "BTC-USDT-SWAP"),
    ("ETHUSDT_SWAP_OKX.GLOBAL", "ETH-USDT-SWAP"),
    ("SOLUSDT_SWAP_OKX.GLOBAL", "SOL-USDT-SWAP"),
    ("LINKUSDT_SWAP_OKX.GLOBAL", "LINK-USDT-SWAP"),
    ("DOGEUSDT_SWAP_OKX.GLOBAL", "DOGE-USDT-SWAP"),
]

OPTIONAL_SYMBOLS = [
    ("BNBUSDT_SWAP_OKX.GLOBAL", "BNB-USDT-SWAP"),
    ("XRPUSDT_SWAP_OKX.GLOBAL", "XRP-USDT-SWAP"),
]


class AuditMultiSymbolReadinessTest(unittest.TestCase):
    def run_temp_audit(
        self,
        root: Path,
        *,
        database_path: Path | None = None,
        batch_targets: bool = True,
        start: str = audit_mod.DEFAULT_START,
        end: str = audit_mod.DEFAULT_END,
        required_symbols: list[str] | None = None,
        optional_symbols: list[str] | None = None,
        min_ready_symbols: int = audit_mod.MIN_TREND_V3_READY_SYMBOLS,
    ) -> dict[str, Any]:
        makefile = root / "Makefile"
        write_makefile(makefile, batch_targets=batch_targets)
        readme = root / "README.md"
        readme.write_text("# Trend V3 Data Preparation\n", encoding="utf-8")
        return audit_mod.run_audit(
            config_dir=root / "config" / "instruments",
            database_path=database_path or root / "missing_database.db",
            makefile_path=makefile,
            readme_path=readme,
            output_dir=root / "reports" / "research" / "multisymbol_readiness",
            start=start,
            end=end,
            interval="1m",
            timezone_name="Asia/Shanghai",
            required_symbols=required_symbols,
            optional_symbols=optional_symbols,
            min_ready_symbols=min_ready_symbols,
            project_root=PROJECT_ROOT,
        )

    def test_metadata_complete_requires_positive_fields_and_refresh_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "config" / "instruments" / "ethusdt_swap_okx.json"
            write_json(
                path,
                instrument_payload(
                    "ETHUSDT_SWAP_OKX.GLOBAL",
                    "ETH-USDT-SWAP",
                    size=0.1,
                    pricetick=0.01,
                    min_volume=0.01,
                    needs_refresh=True,
                ),
            )

            instrument = audit_mod.inspect_instrument_file(path)

        self.assertFalse(instrument["metadata_complete"])
        self.assertFalse(instrument["can_backtest"])
        self.assertTrue(any("needs_okx_contract_metadata_refresh" in item for item in instrument["warnings"]))

    def test_metadata_complete_requires_canonical_okx_inst_id_and_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "config" / "instruments" / "ethusdt_swap_okx.json"
            payload = instrument_payload("ETHUSDT_SWAP_OKX.GLOBAL", "ETH-USDT-SWAP")
            payload.pop("okx_inst_id")
            payload.pop("product")
            write_json(path, payload)

            instrument = audit_mod.inspect_instrument_file(path)

        self.assertFalse(instrument["metadata_complete"])
        self.assertEqual(instrument["okx_inst_id"], "")
        self.assertEqual(instrument["okx_inst_id_source"], "fallback_available_but_not_canonical")
        self.assertIn("missing_canonical_field: okx_inst_id", instrument["warnings"])
        self.assertIn("missing_canonical_field: product", instrument["warnings"])

    def test_metadata_complete_true_with_canonical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "config" / "instruments" / "ethusdt_swap_okx.json"
            write_json(path, instrument_payload("ETHUSDT_SWAP_OKX.GLOBAL", "ETH-USDT-SWAP"))

            instrument = audit_mod.inspect_instrument_file(path)

        self.assertTrue(instrument["metadata_complete"])

    def test_five_required_ready_and_optional_missing_can_enter_trend_v3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            for vt_symbol, inst_id in REQUIRED_SYMBOLS:
                write_json(config_dir / f"{vt_symbol.split('.')[0].lower()}.json", instrument_payload(vt_symbol, inst_id))
            db_path = root / "database.db"
            write_history_db(
                db_path,
                [row for vt_symbol, _ in REQUIRED_SYMBOLS for row in minute_rows(vt_symbol)],
            )

            payload = self.run_temp_audit(root, database_path=db_path, start="2025-01-01", end="2025-01-01")
            readiness = payload["trend_v3_readiness"]
            by_symbol = {item["vt_symbol"]: item for item in payload["instruments"]}

        self.assertTrue(readiness["can_enter_trend_v3"])
        self.assertEqual(readiness["blocking_reasons"], [])
        self.assertEqual(readiness["ready_symbols"], [vt_symbol for vt_symbol, _ in REQUIRED_SYMBOLS])
        self.assertEqual(len(readiness["ready_symbols"]), 5)
        self.assertTrue(by_symbol["BNBUSDT_SWAP_OKX.GLOBAL"]["is_optional"])
        self.assertFalse(by_symbol["BNBUSDT_SWAP_OKX.GLOBAL"]["metadata_complete"])
        self.assertTrue(any("missing_optional_instrument_files" in item for item in readiness["optional_warnings"]))
        self.assertTrue(any("BNBUSDT_SWAP_OKX.GLOBAL" in item for item in readiness["optional_warnings"]))
        self.assertTrue(any("XRPUSDT_SWAP_OKX.GLOBAL" in item for item in readiness["optional_warnings"]))

    def test_can_enter_trend_v3_false_when_eth_required_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            ready_symbols = [item for item in REQUIRED_SYMBOLS if item[0] != "ETHUSDT_SWAP_OKX.GLOBAL"]
            for vt_symbol, inst_id in ready_symbols:
                write_json(config_dir / f"{vt_symbol.split('.')[0].lower()}.json", instrument_payload(vt_symbol, inst_id))
            db_path = root / "database.db"
            write_history_db(
                db_path,
                [
                    row
                    for vt_symbol, _ in ready_symbols
                    for row in minute_rows(vt_symbol)
                ],
            )

            payload = self.run_temp_audit(
                root,
                database_path=db_path,
                batch_targets=True,
                start="2025-01-01",
                end="2025-01-01",
            )
            readiness = payload["trend_v3_readiness"]

        self.assertFalse(readiness["can_enter_trend_v3"])
        self.assertTrue(
            any("required_ready_symbols_missing: ETHUSDT_SWAP_OKX.GLOBAL" in item for item in readiness["blocking_reasons"])
        )
        self.assertTrue(
            any("missing_required_instrument_files: ETHUSDT_SWAP_OKX.GLOBAL" in item for item in readiness["required_warnings"])
        )

    def test_required_metadata_complete_but_history_incomplete_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            for vt_symbol, inst_id in REQUIRED_SYMBOLS:
                write_json(config_dir / f"{vt_symbol.split('.')[0].lower()}.json", instrument_payload(vt_symbol, inst_id))
            db_path = root / "database.db"
            rows = []
            for vt_symbol, _ in REQUIRED_SYMBOLS:
                minutes = 720 if vt_symbol == "ETHUSDT_SWAP_OKX.GLOBAL" else 1440
                rows.extend(minute_rows(vt_symbol, minutes=minutes))
            write_history_db(
                db_path,
                rows,
            )

            payload = self.run_temp_audit(
                root,
                database_path=db_path,
                batch_targets=True,
                start="2025-01-01",
                end="2025-01-01",
            )
            readiness = payload["trend_v3_readiness"]
            by_symbol = {item["vt_symbol"]: item for item in payload["instruments"]}

        self.assertFalse(readiness["can_enter_trend_v3"])
        self.assertTrue(by_symbol["ETHUSDT_SWAP_OKX.GLOBAL"]["metadata_complete"])
        self.assertFalse(by_symbol["ETHUSDT_SWAP_OKX.GLOBAL"]["required_coverage_ready"])
        self.assertEqual(by_symbol["ETHUSDT_SWAP_OKX.GLOBAL"]["total_count"], 720)
        self.assertEqual(by_symbol["ETHUSDT_SWAP_OKX.GLOBAL"]["expected_count"], 1440)
        self.assertTrue(
            any("incomplete_required_history_coverage: ETHUSDT_SWAP_OKX.GLOBAL" in item for item in readiness["blocking_reasons"])
        )

    def test_optional_missing_metadata_and_history_do_not_block_required_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            for vt_symbol, inst_id in REQUIRED_SYMBOLS:
                write_json(config_dir / f"{vt_symbol.split('.')[0].lower()}.json", instrument_payload(vt_symbol, inst_id))
            for vt_symbol, inst_id in OPTIONAL_SYMBOLS:
                write_json(
                    config_dir / f"{vt_symbol.split('.')[0].lower()}.json",
                    instrument_payload(
                        vt_symbol,
                        inst_id,
                        size=None,
                        pricetick=None,
                        min_volume=None,
                        needs_refresh=True,
                    ),
                )
            db_path = root / "database.db"
            write_history_db(
                db_path,
                [row for vt_symbol, _ in REQUIRED_SYMBOLS for row in minute_rows(vt_symbol)],
            )

            payload = self.run_temp_audit(root, database_path=db_path, start="2025-01-01", end="2025-01-01")
            readiness = payload["trend_v3_readiness"]
            by_symbol = {item["vt_symbol"]: item for item in payload["instruments"]}

        self.assertTrue(readiness["can_enter_trend_v3"])
        self.assertEqual(readiness["blocking_reasons"], [])
        self.assertFalse(by_symbol["BNBUSDT_SWAP_OKX.GLOBAL"]["metadata_complete"])
        self.assertFalse(by_symbol["BNBUSDT_SWAP_OKX.GLOBAL"]["has_any_history"])
        self.assertTrue(any("incomplete_optional_instrument_metadata" in item for item in readiness["optional_warnings"]))
        self.assertTrue(any("missing_optional_local_sqlite_history" in item for item in readiness["optional_warnings"]))

    def test_database_path_matches_verify_default_path(self) -> None:
        self.assertEqual(audit_mod.DEFAULT_DATABASE_PATH, PROJECT_ROOT / ".vntrader" / "database.db")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = self.run_temp_audit(
                root,
                database_path=audit_mod.DEFAULT_DATABASE_PATH,
                start="2025-01-01",
                end="2025-01-01",
                required_symbols=[],
                optional_symbols=[],
                min_ready_symbols=0,
            )

        self.assertTrue(payload["database"]["matches_verify_default_database_path"])
        self.assertEqual(payload["database"]["verify_default_database_path"], str(audit_mod.DEFAULT_DATABASE_PATH))

    def test_symbol_mapping_queries_eth_symbol_and_global_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            vt_symbol = "ETHUSDT_SWAP_OKX.GLOBAL"
            write_json(config_dir / "ethusdt_swap_okx.json", instrument_payload(vt_symbol, "ETH-USDT-SWAP"))
            db_path = root / "database.db"
            write_history_db(db_path, minute_rows(vt_symbol))

            payload = self.run_temp_audit(
                root,
                database_path=db_path,
                start="2025-01-01",
                end="2025-01-01",
                required_symbols=[vt_symbol],
                optional_symbols=[],
                min_ready_symbols=1,
            )
            by_symbol = {item["vt_symbol"]: item for item in payload["instruments"]}
            eth = by_symbol[vt_symbol]

        self.assertTrue(eth["metadata_complete"])
        self.assertTrue(eth["required_coverage_ready"])
        self.assertEqual(eth["symbol"], "ETHUSDT_SWAP_OKX")
        self.assertEqual(eth["exchange"], "GLOBAL")
        self.assertEqual(eth["total_count"], 1440)
        self.assertEqual(eth["expected_count"], 1440)
        self.assertTrue(eth["can_backtest_for_window"])

    def test_generates_json_and_markdown_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(
                root / "config" / "instruments" / "btcusdt_swap_okx.json",
                instrument_payload("BTCUSDT_SWAP_OKX.GLOBAL", "BTC-USDT-SWAP"),
            )

            payload = self.run_temp_audit(root)
            json_path = Path(payload["outputs"]["json"])
            report_path = Path(payload["outputs"]["report"])
            self.assertTrue(json_path.exists())
            self.assertTrue(report_path.exists())
            reloaded = json.loads(json_path.read_text(encoding="utf-8"))
            report_text = report_path.read_text(encoding="utf-8")

        self.assertIn("configured_vt_symbols", reloaded)
        self.assertIn("can_enter_trend_v3", reloaded["trend_v3_readiness"])
        self.assertIn("Required Instruments", report_text)
        self.assertIn("Optional Instruments", report_text)
        self.assertIn("Optional Warnings", report_text)
        self.assertIn("has_any_history", report_text)
        self.assertIn("required_coverage_ready", report_text)

    def test_database_missing_warns_and_makefile_batch_targets_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_json(
                root / "config" / "instruments" / "btcusdt_swap_okx.json",
                instrument_payload("BTCUSDT_SWAP_OKX.GLOBAL", "BTC-USDT-SWAP"),
            )

            payload = self.run_temp_audit(root, batch_targets=True)
            instrument = payload["instruments"][0]

        self.assertFalse(payload["database"]["exists"])
        self.assertIn("database_not_found", payload["database"]["warning"])
        self.assertIn("database_not_found", "; ".join(instrument["warnings"]))
        self.assertTrue(payload["makefile"]["audit_multisymbol_target_exists"])
        self.assertTrue(payload["makefile"]["batch_download_target_exists"])
        self.assertTrue(payload["makefile"]["batch_verify_target_exists"])

    def test_repository_makefile_contains_required_multisymbol_targets(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        for target in (
            "refresh-okx-metadata:",
            "download-history-batch:",
            "verify-history-batch:",
            "audit-multisymbol:",
        ):
            self.assertIn(target, text)


if __name__ == "__main__":
    unittest.main()
