from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import postmortem_trend_following_v3 as postmortem


TRADE_COLUMNS = [
    "policy_name",
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "holding_minutes",
    "volume",
    "contract_size",
    "gross_pnl",
    "fee",
    "slippage",
    "net_pnl",
    "no_cost_pnl",
    "no_cost_net_pnl",
    "r_multiple",
    "mfe",
    "mae",
    "timeframe",
    "entry_source",
    "entry_signal_time",
    "exit_signal_time",
    "entry_atr",
    "initial_risk",
    "turnover",
]


def make_trade(
    policy_name: str = "v3_1d_ema_50_200_atr5",
    symbol: str = "BTCUSDT_SWAP_OKX.GLOBAL",
    net_pnl: float = 1.0,
    no_cost_net_pnl: float | None = None,
    holding_minutes: float = 480.0,
    entry_price: float = 1000.0,
    volume: float = 1.0,
    contract_size: float = 1.0,
    exit_day: int = 2,
) -> dict[str, object]:
    no_cost = net_pnl if no_cost_net_pnl is None else no_cost_net_pnl
    return {
        "policy_name": policy_name,
        "symbol": symbol,
        "direction": "long",
        "entry_time": "2026-01-01T00:00:00+08:00",
        "entry_price": entry_price,
        "exit_time": f"2026-01-{exit_day:02d}T00:00:00+08:00",
        "exit_price": entry_price + net_pnl,
        "exit_reason": "unit_test",
        "holding_minutes": holding_minutes,
        "volume": volume,
        "contract_size": contract_size,
        "gross_pnl": no_cost,
        "fee": max(no_cost - net_pnl, 0.0),
        "slippage": 0.0,
        "net_pnl": net_pnl,
        "no_cost_pnl": no_cost,
        "no_cost_net_pnl": no_cost,
        "r_multiple": 1.0,
        "mfe": 1.0,
        "mae": 0.0,
        "timeframe": "1d",
        "entry_source": policy_name,
        "entry_signal_time": "2025-12-31T23:59:00+08:00",
        "exit_signal_time": f"2026-01-{exit_day - 1:02d}T23:59:00+08:00",
        "entry_atr": 1.0,
        "initial_risk": 1.0,
        "turnover": entry_price * volume * contract_size * 2.0,
    }


def write_split(directory: Path, split: str, trades: list[dict[str, object]], stable: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    trades_df.to_csv(directory / "trend_v3_trades.csv", index=False)
    prepared = postmortem.prepare_trades(trades_df, split)
    policy_rows = []
    for policy_name, group in prepared.groupby("policy_name", dropna=False):
        metrics = postmortem.summarize_trade_slice(group)
        policy_rows.append(
            {
                "policy_name": policy_name,
                "trade_count": metrics["trade_count"],
                "active_symbol_count": metrics["active_symbol_count"],
                "no_cost_net_pnl": metrics["no_cost_net_pnl"],
                "net_pnl": metrics["net_pnl"],
                "cost_drag": metrics["cost_drag"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown": metrics["max_drawdown"],
                "max_ddpercent": metrics["max_ddpercent"],
                "avg_holding_minutes": metrics["avg_holding_minutes"],
                "largest_symbol_pnl_share": postmortem.largest_symbol_dependency(group.groupby("symbol")["net_pnl"].sum())[0],
                "top_5pct_trade_pnl_contribution": postmortem.build_top_trade_concentration({split: group}).iloc[0]["top_5pct_share"],
                "stable_candidate": stable,
            }
        )
    leaderboard = pd.DataFrame(policy_rows)
    leaderboard.to_csv(directory / "trend_v3_policy_leaderboard.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_policy_by_symbol.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_policy_by_month.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_symbol_contribution.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_portfolio_equity_curve.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_portfolio_daily_pnl.csv", index=False)
    pd.DataFrame().to_csv(directory / "trend_v3_drawdown.csv", index=False)
    (directory / "trend_v3_summary.json").write_text(
        json.dumps({"portfolio_capital": 5000.0, "split": split}, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "trend_v3_report.md").write_text("# unit test\n", encoding="utf-8")


def write_compare(directory: Path, rows: list[dict[str, object]], stable_exists: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "trend_v3_compare_leaderboard.csv", index=False)
    (directory / "trend_v3_compare_summary.json").write_text(
        json.dumps({"stable_candidate_exists": stable_exists, "trend_following_v3_failed": not stable_exists}, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "trend_v3_compare_report.md").write_text("# compare\n", encoding="utf-8")


class PostmortemTrendFollowingV3Test(unittest.TestCase):
    def test_missing_files_warns_without_crashing_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            warnings: list[str] = []
            artifacts = postmortem.load_split_artifacts("train", Path(tmp_dir), warnings)

        self.assertIn("trend_v3_trades", artifacts)
        self.assertGreaterEqual(len(warnings), len(postmortem.REQUIRED_SPLIT_FILES))

    def test_policy_family_classification(self) -> None:
        self.assertEqual(postmortem.classify_policy_family("v3_4h_donchian_20_10_atr4"), "4h_donchian")
        self.assertEqual(postmortem.classify_policy_family("v3_1d_donchian_55_20_atr5"), "1d_donchian")
        self.assertEqual(postmortem.classify_policy_family("v3_4h_ema_50_200_atr4"), "4h_ema")
        self.assertEqual(postmortem.classify_policy_family("v3_1d_ema_50_200_atr5"), "1d_ema")
        self.assertEqual(postmortem.classify_policy_family("v3_4h_vol_compression_donchian_breakout"), "vol_compression")
        self.assertEqual(postmortem.classify_policy_family("v3_ensemble_core"), "ensemble")
        self.assertEqual(postmortem.classify_policy_family("v3_4h_donchian_55_with_risk_filters"), "risk_filtered")

    def test_symbol_contribution_calculation(self) -> None:
        trades = pd.DataFrame(
            [
                make_trade(symbol="BTCUSDT_SWAP_OKX.GLOBAL", net_pnl=10.0),
                make_trade(symbol="ETHUSDT_SWAP_OKX.GLOBAL", net_pnl=-2.0),
            ]
        )
        prepared = postmortem.prepare_trades(trades, "oos")
        result = postmortem.build_symbol_contribution_postmortem({"oos": prepared}, {"oos": 5000.0})
        btc = result[result["symbol"] == "BTCUSDT_SWAP_OKX.GLOBAL"].iloc[0]
        eth = result[result["symbol"] == "ETHUSDT_SWAP_OKX.GLOBAL"].iloc[0]

        self.assertAlmostEqual(float(btc["contribution_share"]), 1.25)
        self.assertAlmostEqual(float(btc["largest_symbol_dependency"]), 1.0)
        self.assertTrue(bool(eth["removing_symbol_improves_policy"]))
        self.assertAlmostEqual(float(eth["net_pnl_without_symbol"]), 10.0)

    def test_top_trade_concentration_calculation(self) -> None:
        trades = [make_trade(net_pnl=-0.1, exit_day=2 + index) for index in range(19)]
        trades.append(make_trade(net_pnl=10.0, exit_day=25))
        prepared = postmortem.prepare_trades(pd.DataFrame(trades), "oos")
        result = postmortem.build_top_trade_concentration({"oos": prepared})
        row = result.iloc[0]

        self.assertEqual(int(row["trade_count"]), 20)
        self.assertAlmostEqual(float(row["top_1_trade_pnl"]), 10.0)
        self.assertAlmostEqual(float(row["top_5pct_trade_pnl"]), 10.0)
        self.assertEqual(int(row["top_5pct_trade_count"]), 1)

    def test_remove_top_pnl_calculation(self) -> None:
        trades = pd.DataFrame(
            [
                make_trade(net_pnl=5.0, exit_day=2),
                make_trade(net_pnl=1.0, exit_day=3),
                make_trade(net_pnl=-4.0, exit_day=4),
            ]
        )
        prepared = postmortem.prepare_trades(trades, "oos")
        row = postmortem.build_top_trade_concentration({"oos": prepared}).iloc[0]

        self.assertAlmostEqual(float(row["total_net_pnl"]), 2.0)
        self.assertAlmostEqual(float(row["remove_top_1_pnl"]), -3.0)
        self.assertAlmostEqual(float(row["remove_top_5pct_pnl"]), -3.0)

    def test_synthetic_funding_cost_calculation(self) -> None:
        trades = pd.DataFrame([make_trade(net_pnl=1.0, holding_minutes=480.0, entry_price=1000.0, volume=1.0, contract_size=1.0)])
        prepared = postmortem.prepare_trades(trades, "oos")
        warnings: list[str] = []
        result, mode = postmortem.build_funding_sensitivity({"oos": prepared}, [1.0], "synthetic", warnings)
        row = result.iloc[0]

        self.assertEqual(mode, "synthetic")
        self.assertAlmostEqual(float(row["funding_cost"]), 0.1)
        self.assertAlmostEqual(float(row["funding_adjusted_net_pnl"]), 0.9)

    def test_funding_break_even_bps(self) -> None:
        trades = pd.DataFrame([make_trade(net_pnl=1.0, holding_minutes=480.0, entry_price=1000.0, volume=1.0, contract_size=1.0)])
        prepared = postmortem.prepare_trades(trades, "oos")
        result, _mode = postmortem.build_funding_sensitivity({"oos": prepared}, [1.0], "synthetic", [])

        self.assertAlmostEqual(float(result.iloc[0]["funding_break_even_bps"]), 10.0)

    def test_proceed_to_v3_1_decision_true_and_false(self) -> None:
        good = postmortem.decide_v3_1_recommendation(
            {
                "policy_family": "1d_ema",
                "train_no_cost_net_pnl": 2.0,
                "validation_no_cost_net_pnl": 0.5,
                "oos_no_cost_net_pnl": 1.0,
                "remove_top_1_pnl": 0.2,
                "remove_top_5pct_pnl": 0.2,
                "largest_symbol_pnl_share": 0.5,
                "top_5pct_trade_pnl_contribution": 0.5,
                "funding_adjusted_1bps": 0.8,
                "funding_adjusted_3bps": 0.4,
            },
            stable_candidate_exists=False,
        )
        bad = postmortem.decide_v3_1_recommendation(
            {
                "policy_family": "1d_ema",
                "train_no_cost_net_pnl": 2.0,
                "validation_no_cost_net_pnl": -1.0,
                "oos_no_cost_net_pnl": 1.0,
                "remove_top_1_pnl": -0.5,
                "remove_top_5pct_pnl": -0.5,
                "largest_symbol_pnl_share": 0.9,
                "top_5pct_trade_pnl_contribution": 2.0,
                "funding_adjusted_1bps": 0.2,
                "funding_adjusted_3bps": -0.2,
            },
            stable_candidate_exists=False,
        )

        self.assertTrue(good["proceed_to_v3_1"])
        self.assertFalse(bad["proceed_to_v3_1"])

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            compare_dir = root / "compare"
            output_dir = root / "postmortem"
            write_split(train_dir, "train", [make_trade(net_pnl=2.0, no_cost_net_pnl=2.1)])
            write_split(validation_dir, "validation", [make_trade(net_pnl=-1.0, no_cost_net_pnl=-0.9)])
            write_split(oos_dir, "oos", [make_trade(net_pnl=1.0, no_cost_net_pnl=1.1), make_trade(symbol="ETHUSDT_SWAP_OKX.GLOBAL", net_pnl=-0.1)])
            write_compare(
                compare_dir,
                [
                    {
                        "policy_name": "v3_1d_ema_50_200_atr5",
                        "train_no_cost_net_pnl": 2.1,
                        "validation_no_cost_net_pnl": -0.9,
                        "oos_no_cost_net_pnl": 1.1,
                        "oos_net_pnl": 0.9,
                        "stable_candidate": False,
                        "rejection_reasons": "validation_no_cost_net_pnl_not_positive",
                    }
                ],
                stable_exists=False,
            )

            postmortem.run_postmortem(train_dir, validation_dir, oos_dir, compare_dir, output_dir)

            for filename in postmortem.OUTPUT_FILES:
                self.assertTrue((output_dir / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("postmortem-trend-v3:", makefile)


if __name__ == "__main__":
    unittest.main()
