from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import refresh_okx_instrument_metadata as refresh_mod


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def okx_raw(inst_id: str, ct_val: str = "0.1", tick_sz: str = "0.01", min_sz: str = "0.01") -> dict[str, str]:
    return {
        "instType": "SWAP",
        "instId": inst_id,
        "ctVal": ct_val,
        "tickSz": tick_sz,
        "minSz": min_sz,
        "lotSz": min_sz,
        "state": "live",
    }


class RefreshOkxInstrumentMetadataTest(unittest.TestCase):
    def test_okx_metadata_response_maps_to_instrument_json(self) -> None:
        mapped = refresh_mod.map_okx_metadata("ETH-USDT-SWAP", okx_raw("ETH-USDT-SWAP"))

        self.assertEqual(mapped["vt_symbol"], "ETHUSDT_SWAP_OKX.GLOBAL")
        self.assertEqual(mapped["symbol"], "ETHUSDT_SWAP_OKX")
        self.assertEqual(mapped["exchange"], "GLOBAL")
        self.assertEqual(mapped["product"], "SWAP")
        self.assertEqual(mapped["size"], 0.1)
        self.assertEqual(mapped["pricetick"], 0.01)
        self.assertEqual(mapped["min_volume"], 0.01)

    def test_dry_run_does_not_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            eth_path = config_dir / "ethusdt_swap_okx.json"
            original = {
                "vt_symbol": "ETHUSDT_SWAP_OKX.GLOBAL",
                "symbol": "ETHUSDT_SWAP_OKX",
                "exchange": "GLOBAL",
                "okx_inst_id": "ETH-USDT-SWAP",
                "product": "SWAP",
                "size": None,
                "pricetick": None,
                "min_volume": None,
                "needs_okx_contract_metadata_refresh": True,
            }
            write_json(eth_path, original)
            before = eth_path.read_text(encoding="utf-8")

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["ETH-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=True,
                fetcher=lambda inst_id, timeout: okx_raw(inst_id),
            )

            self.assertEqual(eth_path.read_text(encoding="utf-8"), before)
            self.assertTrue(payload["dry_run"])
            self.assertFalse(payload["instruments"][0]["wrote_file"])
            self.assertTrue(payload["instruments"][0]["metadata_complete"])

    def test_write_updates_size_pricetick_and_min_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            sol_path = config_dir / "solusdt_swap_okx.json"
            write_json(
                sol_path,
                {
                    "vt_symbol": "SOLUSDT_SWAP_OKX.GLOBAL",
                    "symbol": "SOLUSDT_SWAP_OKX",
                    "exchange": "GLOBAL",
                    "okx_inst_id": "SOL-USDT-SWAP",
                    "product": "SWAP",
                    "size": None,
                    "pricetick": None,
                    "min_volume": None,
                    "needs_okx_contract_metadata_refresh": True,
                    "custom_user_field": "preserve-me",
                },
            )

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["SOL-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=False,
                fetcher=lambda inst_id, timeout: okx_raw(inst_id, ct_val="1", tick_sz="0.001", min_sz="0.01"),
            )
            updated = read_json(sol_path)

            self.assertEqual(updated["size"], 1.0)
            self.assertEqual(updated["pricetick"], 0.001)
            self.assertEqual(updated["min_volume"], 0.01)
            self.assertFalse(updated["needs_okx_contract_metadata_refresh"])
            self.assertEqual(updated["custom_user_field"], "preserve-me")
            self.assertTrue(payload["instruments"][0]["wrote_file"])

    def test_write_backfills_canonical_okx_inst_id_and_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            link_path = config_dir / "linkusdt_swap_okx.json"
            write_json(
                link_path,
                {
                    "vt_symbol": "LINKUSDT_SWAP_OKX.GLOBAL",
                    "symbol": "LINKUSDT_SWAP_OKX",
                    "exchange": "GLOBAL",
                    "name": "LINK-USDT-SWAP",
                    "size": 1.0,
                    "pricetick": 0.001,
                    "min_volume": 0.01,
                    "gateway_name": "OKX",
                    "history_data": True,
                },
            )

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["LINK-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=False,
                fetcher=lambda inst_id, timeout: okx_raw(inst_id, ct_val="1", tick_sz="0.001", min_sz="0.01"),
            )
            updated = read_json(link_path)

            self.assertEqual(updated["okx_inst_id"], "LINK-USDT-SWAP")
            self.assertEqual(updated["product"], "SWAP")
            self.assertEqual(updated["name"], "LINK-USDT-SWAP")
            self.assertFalse(updated["needs_okx_contract_metadata_refresh"])
            self.assertTrue(payload["instruments"][0]["metadata_complete"])

    def test_refresh_failure_does_not_write_fake_metadata(self) -> None:
        def fail_fetcher(inst_id: str, timeout: float) -> dict[str, Any]:
            raise refresh_mod.MetadataRefreshError("endpoint_unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            doge_path = config_dir / "dogeusdt_swap_okx.json"
            write_json(
                doge_path,
                {
                    "vt_symbol": "DOGEUSDT_SWAP_OKX.GLOBAL",
                    "symbol": "DOGEUSDT_SWAP_OKX",
                    "exchange": "GLOBAL",
                    "okx_inst_id": "DOGE-USDT-SWAP",
                    "product": "SWAP",
                    "size": None,
                    "pricetick": None,
                    "min_volume": None,
                    "needs_okx_contract_metadata_refresh": True,
                },
            )

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["DOGE-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=False,
                fetcher=fail_fetcher,
            )
            updated = read_json(doge_path)

            self.assertIsNone(updated["size"])
            self.assertIsNone(updated["pricetick"])
            self.assertIsNone(updated["min_volume"])
            self.assertTrue(updated["needs_okx_contract_metadata_refresh"])
            self.assertFalse(payload["instruments"][0]["metadata_complete"])
            self.assertIn("endpoint_unavailable", payload["instruments"][0]["warning"])

    def test_missing_failed_refresh_creates_placeholder(self) -> None:
        def fail_fetcher(inst_id: str, timeout: float) -> dict[str, Any]:
            raise refresh_mod.MetadataRefreshError("not_found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            bnb_path = config_dir / "bnbusdt_swap_okx.json"

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["BNB-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=False,
                fetcher=fail_fetcher,
            )
            created = read_json(bnb_path)

            self.assertEqual(created["vt_symbol"], "BNBUSDT_SWAP_OKX.GLOBAL")
            self.assertIsNone(created["size"])
            self.assertIsNone(created["pricetick"])
            self.assertIsNone(created["min_volume"])
            self.assertTrue(created["needs_okx_contract_metadata_refresh"])
            self.assertFalse(payload["instruments"][0]["metadata_complete"])

    def test_failed_refresh_preserves_existing_complete_metadata(self) -> None:
        def fail_fetcher(inst_id: str, timeout: float) -> dict[str, Any]:
            raise refresh_mod.MetadataRefreshError("temporary_network_error")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config" / "instruments"
            btc_path = config_dir / "btcusdt_swap_okx.json"
            write_json(
                btc_path,
                {
                    "vt_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                    "symbol": "BTCUSDT_SWAP_OKX",
                    "exchange": "GLOBAL",
                    "okx_inst_id": "BTC-USDT-SWAP",
                    "product": "SWAP",
                    "size": 0.01,
                    "pricetick": 0.1,
                    "min_volume": 0.01,
                },
            )

            payload = refresh_mod.refresh_metadata_for_inst_ids(
                inst_ids=["BTC-USDT-SWAP"],
                config_dir=config_dir,
                output_json=root / "reports" / "okx_metadata_refresh.json",
                dry_run=False,
                fetcher=fail_fetcher,
            )
            updated = read_json(btc_path)

            self.assertEqual(updated["size"], 0.01)
            self.assertEqual(updated["pricetick"], 0.1)
            self.assertEqual(updated["min_volume"], 0.01)
            self.assertNotEqual(updated.get("needs_okx_contract_metadata_refresh"), True)
            self.assertTrue(payload["instruments"][0]["metadata_complete"])


if __name__ == "__main__":
    unittest.main()
