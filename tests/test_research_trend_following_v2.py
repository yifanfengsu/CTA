from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_trend_following_v2 as trend_mod


def make_policy(
    entry_window: int = 3,
    exit_window: int = 2,
    atr_mult: float = 3.0,
) -> trend_mod.PolicyRun:
    """Build a compact Donchian policy for unit tests."""

    return trend_mod.PolicyRun(
        policy_name="test_policy_atr3",
        base_policy_name="test_policy",
        timeframe="1h",
        entry_type="donchian",
        entry_window=entry_window,
        exit_window=exit_window,
        atr_mult=atr_mult,
        use_donchian_exit=True,
    )


def make_1m_bars(minutes: int = 4320, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    """Build deterministic 1m bars with an upward drift."""

    start_dt = pd.Timestamp(start)
    records = []
    previous_close = 100.0
    for index in range(minutes):
        close = 100.0 + index * 0.01
        open_price = previous_close
        high = max(open_price, close) + 0.1
        low = min(open_price, close) - 0.1
        records.append(
            {
                "datetime": (start_dt + pd.Timedelta(minutes=index)).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0 + (index % 10),
            }
        )
        previous_close = close
    return pd.DataFrame(records)


class ResearchTrendFollowingV2Test(unittest.TestCase):
    def test_donchian_entry_uses_previous_channel_not_current_bar(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=4, freq="1h", tz="Asia/Shanghai"),
                "open": [9.0, 9.0, 9.0, 9.0],
                "high": [10.0, 10.0, 10.0, 20.0],
                "low": [8.0, 8.0, 8.0, 8.0],
                "close": [9.0, 9.0, 9.0, 11.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            }
        )
        with_channels = trend_mod.add_donchian_channels(frame, [3])
        row = with_channels.iloc[-1]

        self.assertAlmostEqual(float(row["donchian_high_3_prev"]), 10.0)
        self.assertTrue(trend_mod.policy_entry_signal(row, make_policy(), "long"))
        equal_row = pd.Series({"close": 10.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0})
        self.assertFalse(trend_mod.policy_entry_signal(equal_row, make_policy(), "long"))
        self.assertFalse(trend_mod.policy_entry_signal(equal_row, make_policy(), "short"))

    def test_donchian_exit_uses_previous_channel_not_current_bar(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=4, freq="1h", tz="Asia/Shanghai"),
                "open": [9.0, 9.0, 9.0, 9.0],
                "high": [10.0, 10.0, 10.0, 20.0],
                "low": [8.0, 8.0, 8.0, 1.0],
                "close": [9.0, 9.0, 9.0, 7.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            }
        )
        row = trend_mod.add_donchian_channels(frame, [3]).iloc[-1]
        policy = make_policy(exit_window=3)

        self.assertAlmostEqual(float(row["donchian_low_3_prev"]), 8.0)
        self.assertAlmostEqual(float(row["donchian_high_3_prev"]), 10.0)
        self.assertTrue(trend_mod.check_donchian_exit(row, policy, "long"))
        self.assertTrue(trend_mod.check_donchian_exit(pd.Series({"close": 21.0, "donchian_high_3_prev": 10.0}), policy, "short"))

    def test_resample_keeps_only_closed_bars_timestamped_at_close(self) -> None:
        incomplete = make_1m_bars(minutes=151, start="2025-01-01T08:00:00+08:00")
        complete = make_1m_bars(minutes=240, start="2025-01-01T08:00:00+08:00")

        self.assertTrue(trend_mod.resample_ohlcv(incomplete, 240).empty)
        bars_4h = trend_mod.resample_ohlcv(complete, 240)

        self.assertEqual(len(bars_4h.index), 1)
        self.assertEqual(pd.Timestamp(bars_4h.iloc[0]["datetime"]), pd.Timestamp("2025-01-01T11:59:00+08:00"))

    def test_long_donchian_breakout_entry(self) -> None:
        row = pd.Series({"close": 11.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0})

        self.assertTrue(trend_mod.policy_entry_signal(row, make_policy(), "long"))
        self.assertFalse(trend_mod.policy_entry_signal(row, make_policy(), "short"))

    def test_short_donchian_breakdown_entry(self) -> None:
        row = pd.Series({"close": 7.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0})

        self.assertTrue(trend_mod.policy_entry_signal(row, make_policy(), "short"))
        self.assertFalse(trend_mod.policy_entry_signal(row, make_policy(), "long"))

    def test_atr_trailing_stop_long(self) -> None:
        hit, trail = trend_mod.check_atr_trailing_stop(
            direction="long",
            close=103.0,
            atr=2.0,
            highest_close=110.0,
            lowest_close=100.0,
            atr_mult=3.0,
        )

        self.assertTrue(hit)
        self.assertAlmostEqual(float(trail), 104.0)

        relaxed_hit, relaxed_trail = trend_mod.check_atr_trailing_stop(
            direction="long",
            close=105.0,
            atr=5.0,
            highest_close=110.0,
            lowest_close=100.0,
            atr_mult=3.0,
            current_trailing_stop=104.0,
        )
        self.assertFalse(relaxed_hit)
        self.assertAlmostEqual(float(relaxed_trail), 104.0)

    def test_atr_trailing_stop_short(self) -> None:
        hit, trail = trend_mod.check_atr_trailing_stop(
            direction="short",
            close=97.0,
            atr=2.0,
            highest_close=100.0,
            lowest_close=90.0,
            atr_mult=3.0,
        )

        self.assertTrue(hit)
        self.assertAlmostEqual(float(trail), 96.0)

        relaxed_hit, relaxed_trail = trend_mod.check_atr_trailing_stop(
            direction="short",
            close=95.0,
            atr=5.0,
            highest_close=100.0,
            lowest_close=90.0,
            atr_mult=3.0,
            current_trailing_stop=96.0,
        )
        self.assertFalse(relaxed_hit)
        self.assertAlmostEqual(float(relaxed_trail), 96.0)

    def test_donchian_exit_long_and_short(self) -> None:
        policy = make_policy(exit_window=2)
        long_row = pd.Series({"close": 7.0, "donchian_low_2_prev": 8.0})
        short_row = pd.Series({"close": 11.0, "donchian_high_2_prev": 10.0})

        self.assertTrue(trend_mod.check_donchian_exit(long_row, policy, "long"))
        self.assertTrue(trend_mod.check_donchian_exit(short_row, policy, "short"))

    def test_ema_exit_long_and_short(self) -> None:
        self.assertTrue(trend_mod.check_ema_exit(pd.Series({"close": 49.0, "ema50": 50.0}), "long"))
        self.assertFalse(trend_mod.check_ema_exit(pd.Series({"close": 51.0, "ema50": 50.0}), "long"))
        self.assertTrue(trend_mod.check_ema_exit(pd.Series({"close": 51.0, "ema50": 50.0}), "short"))
        self.assertFalse(trend_mod.check_ema_exit(pd.Series({"close": 49.0, "ema50": 50.0}), "short"))

    def test_long_and_short_pnl_direction(self) -> None:
        cases = [
            ("long", 100.0, 110.0, 10.0),
            ("long", 100.0, 90.0, -10.0),
            ("short", 100.0, 90.0, 10.0),
            ("short", 100.0, 110.0, -10.0),
        ]
        for direction, entry, exit_price, expected in cases:
            with self.subTest(direction=direction, exit_price=exit_price):
                costs = trend_mod.calculate_trade_costs(
                    entry_price=entry,
                    exit_price=exit_price,
                    volume=1.0,
                    contract_size=1.0,
                    direction=direction,
                    rate=0.0,
                    absolute_slippage=0.0,
                )
                self.assertAlmostEqual(costs["gross_pnl"], expected)
                self.assertAlmostEqual(costs["net_pnl"], expected)

    def test_cost_calculation_long_and_short_with_contract_size(self) -> None:
        costs = trend_mod.calculate_trade_costs(
            entry_price=100.0,
            exit_price=110.0,
            volume=2.0,
            contract_size=0.5,
            direction="long",
            rate=0.001,
            absolute_slippage=0.5,
        )

        self.assertAlmostEqual(costs["gross_pnl"], 10.0)
        self.assertAlmostEqual(costs["fee"], 0.21)
        self.assertAlmostEqual(costs["slippage"], 1.0)
        self.assertAlmostEqual(costs["net_pnl"], 8.79)
        self.assertAlmostEqual(costs["no_cost_net_pnl"], 10.0)

        short_costs = trend_mod.calculate_trade_costs(
            entry_price=100.0,
            exit_price=90.0,
            volume=2.0,
            contract_size=0.5,
            direction="short",
            rate=0.001,
            absolute_slippage=0.5,
        )
        self.assertAlmostEqual(short_costs["gross_pnl"], 10.0)
        self.assertAlmostEqual(short_costs["fee"], 0.19)
        self.assertAlmostEqual(short_costs["slippage"], 1.0)
        self.assertAlmostEqual(short_costs["net_pnl"], 8.81)

    def test_no_cost_has_zero_fee_and_slippage(self) -> None:
        costs = trend_mod.calculate_trade_costs(
            entry_price=100.0,
            exit_price=110.0,
            volume=1.0,
            contract_size=1.0,
            direction="long",
            rate=0.0,
            absolute_slippage=0.0,
        )

        self.assertAlmostEqual(costs["fee"], 0.0)
        self.assertAlmostEqual(costs["slippage"], 0.0)
        self.assertAlmostEqual(costs["net_pnl"], costs["no_cost_net_pnl"])

    def test_equity_curve_and_max_drawdown_include_initial_capital(self) -> None:
        trades = pd.DataFrame(
            [
                {"policy_name": "p", "entry_time": "2025-01-01T00:00:00+08:00", "exit_time": "2025-01-01T01:00:00+08:00", "net_pnl": 20.0, "no_cost_net_pnl": 20.0},
                {"policy_name": "p", "entry_time": "2025-01-01T01:00:00+08:00", "exit_time": "2025-01-01T02:00:00+08:00", "net_pnl": -30.0, "no_cost_net_pnl": -30.0},
                {"policy_name": "p", "entry_time": "2025-01-01T02:00:00+08:00", "exit_time": "2025-01-01T03:00:00+08:00", "net_pnl": 60.0, "no_cost_net_pnl": 60.0},
                {"policy_name": "p", "entry_time": "2025-01-01T03:00:00+08:00", "exit_time": "2025-01-01T04:00:00+08:00", "net_pnl": -20.0, "no_cost_net_pnl": -20.0},
            ]
        )
        max_dd, max_ddpercent = trend_mod.max_drawdown_from_trades(trades, capital=100.0)
        equity = trend_mod.build_equity_curve(trades, capital=100.0)

        self.assertAlmostEqual(max_dd, 30.0)
        self.assertAlmostEqual(max_ddpercent, 25.0)
        self.assertAlmostEqual(float(equity.iloc[0]["equity"]), 100.0)
        self.assertAlmostEqual(float(equity["drawdown"].max()), 30.0)

        first_loss = pd.DataFrame(
            [
                {"policy_name": "p", "entry_time": "2025-01-01T00:00:00+08:00", "exit_time": "2025-01-01T01:00:00+08:00", "net_pnl": -10.0, "no_cost_net_pnl": -10.0}
            ]
        )
        loss_dd, loss_ddpercent = trend_mod.max_drawdown_from_trades(first_loss, capital=100.0)
        self.assertAlmostEqual(loss_dd, 10.0)
        self.assertAlmostEqual(loss_ddpercent, 10.0)

    def test_no_cost_and_cost_aware_outputs_in_leaderboard(self) -> None:
        policy = make_policy()
        trades = pd.DataFrame(
            [
                {
                    "policy_name": policy.policy_name,
                    "base_policy_name": policy.base_policy_name,
                    "atr_mult": policy.atr_mult,
                    "timeframe": policy.timeframe,
                    "direction": "long",
                    "entry_time": "2025-01-01T00:00:00+08:00",
                    "exit_time": "2025-01-01T01:00:00+08:00",
                    "gross_pnl": 10.0,
                    "no_cost_net_pnl": 10.0,
                    "fee": 0.21,
                    "slippage": 1.0,
                    "net_pnl": 8.79,
                    "holding_minutes": 60.0,
                }
            ]
        )
        leaderboard = trend_mod.build_policy_leaderboard(trades, [policy], capital=5000.0)
        row = leaderboard.iloc[0]

        self.assertIn("no_cost_net_pnl", leaderboard.columns)
        self.assertIn("net_pnl", leaderboard.columns)
        self.assertAlmostEqual(float(row["no_cost_net_pnl"]), 10.0)
        self.assertAlmostEqual(float(row["net_pnl"]), 8.79)
        self.assertAlmostEqual(float(row["cost_drag"]), 1.21)

    def test_policy_leaderboard_counts_each_policy_independently(self) -> None:
        policy_a = make_policy()
        policy_b = trend_mod.PolicyRun(
            policy_name="other_policy_atr3",
            base_policy_name="other_policy",
            timeframe="1h",
            entry_type="donchian",
            entry_window=3,
            exit_window=2,
            atr_mult=3.0,
            use_donchian_exit=True,
        )
        trades = pd.DataFrame(
            [
                {"policy_name": policy_a.policy_name, "base_policy_name": policy_a.base_policy_name, "atr_mult": 3.0, "timeframe": "1h", "direction": "long", "entry_time": "2025-01-01T00:00:00+08:00", "exit_time": "2025-01-01T01:00:00+08:00", "gross_pnl": 1.0, "no_cost_net_pnl": 1.0, "fee": 0.0, "slippage": 0.0, "net_pnl": 1.0, "holding_minutes": 60.0},
                {"policy_name": policy_a.policy_name, "base_policy_name": policy_a.base_policy_name, "atr_mult": 3.0, "timeframe": "1h", "direction": "short", "entry_time": "2025-01-01T02:00:00+08:00", "exit_time": "2025-01-01T03:00:00+08:00", "gross_pnl": 2.0, "no_cost_net_pnl": 2.0, "fee": 0.0, "slippage": 0.0, "net_pnl": 2.0, "holding_minutes": 60.0},
                {"policy_name": policy_b.policy_name, "base_policy_name": policy_b.base_policy_name, "atr_mult": 3.0, "timeframe": "1h", "direction": "long", "entry_time": "2025-01-01T04:00:00+08:00", "exit_time": "2025-01-01T05:00:00+08:00", "gross_pnl": 5.0, "no_cost_net_pnl": 5.0, "fee": 0.0, "slippage": 0.0, "net_pnl": 5.0, "holding_minutes": 60.0},
            ]
        )
        leaderboard = trend_mod.build_policy_leaderboard(trades, [policy_a, policy_b], capital=100.0)
        rows = {row["policy_name"]: row for row in leaderboard.to_dict(orient="records")}

        self.assertEqual(rows[policy_a.policy_name]["trade_count"], 2)
        self.assertAlmostEqual(rows[policy_a.policy_name]["net_pnl"], 3.0)
        self.assertEqual(rows[policy_b.policy_name]["trade_count"], 1)
        self.assertAlmostEqual(rows[policy_b.policy_name]["net_pnl"], 5.0)

    def test_run_research_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "trend"
            history_range = trend_mod.resolve_split_range(
                "train",
                "2025-01-01T00:00:00+08:00",
                "2025-01-03T23:59:00+08:00",
                "Asia/Shanghai",
            )
            logger = logging.getLogger("test_research_trend_following_v2")
            summary = trend_mod.run_research(
                vt_symbol=trend_mod.DEFAULT_VT_SYMBOL,
                split="train",
                history_range=history_range,
                output_dir=output_dir,
                timezone_name="Asia/Shanghai",
                slippage_mode="absolute",
                slippage=0.0,
                atr_mults=[3.0],
                max_runs=2,
                data_check_strict=True,
                logger=logger,
                bars_df=make_1m_bars(),
            )

            for filename in [
                "trend_policy_summary.json",
                "trend_policy_leaderboard.csv",
                "trend_trades.csv",
                "trend_daily_pnl.csv",
                "trend_equity_curve.csv",
                "trend_policy_by_side.csv",
                "trend_policy_by_month.csv",
                "trend_report.md",
                "data_quality.json",
                "trend_research_audit.json",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)
            audit = json.loads((output_dir / "trend_research_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["split"], "train")
        self.assertEqual(summary["policy_run_count"], 2)
        for key in [
            "donchian_previous_channel_policy",
            "resample_closed_bar_policy",
            "pnl_formula",
            "trailing_stop_policy",
            "cost_policy",
            "compare_policy",
            "no_lookahead_checks",
            "sample_trades",
        ]:
            self.assertIn(key, audit)
        self.assertIn("used_incomplete_htf_bar_count", audit["no_lookahead_checks"])
        self.assertIn("entry_channel_uses_current_bar_count", audit["no_lookahead_checks"])
        for sample in audit["sample_trades"]:
            self.assertLessEqual(pd.Timestamp(sample["used_htf_bar_time"]), pd.Timestamp(sample["entry_time"]))
            self.assertIn("volume", sample)
            self.assertIn("contract_size", sample)


if __name__ == "__main__":
    unittest.main()
