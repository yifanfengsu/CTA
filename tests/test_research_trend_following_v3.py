from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_trend_following_v3 as trend_mod


SYMBOL_A = "AAAUSDT_SWAP_OKX.GLOBAL"
SYMBOL_B = "BBBUSDT_SWAP_OKX.GLOBAL"
SYMBOL_C = "CCCUSDT_SWAP_OKX.GLOBAL"


def meta(symbol: str) -> dict[str, object]:
    return {
        "vt_symbol": symbol,
        "size": 1.0,
        "pricetick": 0.1,
        "min_volume": 0.01,
        "okx_inst_id": symbol.split("_SWAP")[0] + "-SWAP",
        "product": "SWAP",
        "needs_okx_contract_metadata_refresh": False,
    }


def make_1m_bars(symbol: str, minutes: int, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    start_dt = pd.Timestamp(start)
    rows = []
    previous_close = 100.0
    for index in range(minutes):
        close = 100.0 + index * 0.01
        open_price = previous_close
        rows.append(
            {
                "vt_symbol": symbol,
                "datetime": (start_dt + pd.Timedelta(minutes=index)).isoformat(),
                "open": open_price,
                "high": max(open_price, close) + 0.2,
                "low": min(open_price, close) - 0.2,
                "close": close,
                "volume": 100.0 + index % 7,
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def make_range(minutes: int = 60) -> trend_mod.HistoryRange:
    end = pd.Timestamp("2025-01-01T00:00:00+08:00") + pd.Timedelta(minutes=minutes - 1)
    return trend_mod.resolve_split_range(
        "train",
        "2025-01-01T00:00:00+08:00",
        end.isoformat(),
        "Asia/Shanghai",
    )


def make_context(symbol: str, minutes: int = 180) -> trend_mod.SymbolContext:
    history_range = make_range(minutes)
    logger = logging.getLogger("test_research_trend_following_v3")
    return trend_mod.build_symbol_context(
        vt_symbol=symbol,
        history_range=history_range,
        timezone_name="Asia/Shanghai",
        logger=logger,
        slippage_mode="absolute",
        slippage=0.0,
        bars_from_db=False,
        bars_df=make_1m_bars(symbol, minutes),
        instrument_meta_by_symbol={symbol: meta(symbol)},
    )


class ResearchTrendFollowingV3Test(unittest.TestCase):
    def test_multi_symbol_data_filters_by_vt_symbol(self) -> None:
        history_range = make_range(60)
        symbol_a_bars = make_1m_bars(SYMBOL_A, 60)
        symbol_b_bars = make_1m_bars(SYMBOL_B, 60)
        symbol_b_bars[["open", "high", "low", "close"]] = symbol_b_bars[["open", "high", "low", "close"]] + 1000.0
        combined = pd.concat([symbol_a_bars, symbol_b_bars], ignore_index=True)
        logger = logging.getLogger("test_research_trend_following_v3")
        contexts, data_quality, warnings = trend_mod.build_symbol_contexts(
            symbols=[SYMBOL_A, SYMBOL_B],
            history_range=history_range,
            timezone_name="Asia/Shanghai",
            logger=logger,
            slippage_mode="absolute",
            slippage=0.0,
            data_check_strict=True,
            bars_from_db=False,
            bars_by_symbol={SYMBOL_A: combined, SYMBOL_B: combined},
            instrument_meta_by_symbol={SYMBOL_A: meta(SYMBOL_A), SYMBOL_B: meta(SYMBOL_B)},
        )

        self.assertEqual(set(contexts), {SYMBOL_A, SYMBOL_B})
        self.assertEqual(data_quality["symbol_coverage"][SYMBOL_A]["total_count"], 60)
        self.assertEqual(data_quality["symbol_coverage"][SYMBOL_B]["total_count"], 60)
        self.assertLess(float(contexts[SYMBOL_A].bars_1m.iloc[0]["close"]), 200.0)
        self.assertGreater(float(contexts[SYMBOL_B].bars_1m.iloc[0]["close"]), 1000.0)
        self.assertEqual(warnings, [])

    def test_missing_symbol_data_strict_mode_fails(self) -> None:
        history_range = make_range(60)
        missing_one = make_1m_bars(SYMBOL_A, 60).drop(index=[10]).reset_index(drop=True)
        logger = logging.getLogger("test_research_trend_following_v3")

        with self.assertRaises(trend_mod.TrendFollowingV3Error):
            trend_mod.build_symbol_contexts(
                symbols=[SYMBOL_A],
                history_range=history_range,
                timezone_name="Asia/Shanghai",
                logger=logger,
                slippage_mode="absolute",
                slippage=0.0,
                data_check_strict=True,
                bars_from_db=False,
                bars_by_symbol={SYMBOL_A: missing_one},
                instrument_meta_by_symbol={SYMBOL_A: meta(SYMBOL_A)},
            )

    def test_resample_4h_and_1d_uses_only_closed_bars(self) -> None:
        incomplete_4h = make_1m_bars(SYMBOL_A, 239)
        complete_4h = make_1m_bars(SYMBOL_A, 240)
        incomplete = make_1m_bars(SYMBOL_A, 1439)
        complete = make_1m_bars(SYMBOL_A, 1440)

        self.assertEqual(len(trend_mod.resample_ohlcv(incomplete_4h, 240).index), 0)
        four_hour = trend_mod.resample_ohlcv(complete_4h, 240)
        self.assertEqual(pd.Timestamp(four_hour.iloc[0]["datetime"]), pd.Timestamp("2025-01-01T03:59:00+08:00"))
        self.assertEqual(len(trend_mod.resample_ohlcv(incomplete, 1440).index), 0)
        daily = trend_mod.resample_ohlcv(complete, 1440)

        self.assertEqual(len(daily.index), 1)
        self.assertEqual(pd.Timestamp(daily.iloc[0]["datetime"]), pd.Timestamp("2025-01-01T23:59:00+08:00"))

    def test_donchian_previous_channel_entry(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=4, freq="4h", tz="Asia/Shanghai"),
                "open": [9.0, 9.0, 9.0, 9.0],
                "high": [10.0, 10.0, 10.0, 20.0],
                "low": [8.0, 8.0, 8.0, 8.0],
                "close": [9.0, 9.0, 9.0, 11.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            }
        )
        row = trend_mod.add_donchian_metrics(frame, [3]).iloc[-1]
        policy = trend_mod.PolicyRun("p", "4h", "donchian", 3, 2, 4.0, use_donchian_exit=True)

        self.assertAlmostEqual(float(row["donchian_high_3_prev"]), 10.0)
        self.assertTrue(trend_mod.policy_entry_signal(row, policy, "long"))
        self.assertFalse(trend_mod.policy_entry_signal(pd.Series({"close": 10.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0}), policy, "long"))
        self.assertTrue(trend_mod.policy_entry_signal(pd.Series({"close": 7.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0}), policy, "short"))
        self.assertFalse(trend_mod.policy_entry_signal(pd.Series({"close": 8.0, "donchian_high_3_prev": 10.0, "donchian_low_3_prev": 8.0}), policy, "short"))

    def test_donchian_exit_uses_previous_channel(self) -> None:
        policy = trend_mod.PolicyRun("p", "4h", "donchian", 3, 2, 4.0, use_donchian_exit=True)
        self.assertTrue(trend_mod.check_donchian_exit(pd.Series({"close": 7.0, "donchian_low_2_prev": 8.0}), policy.exit_window, "long"))
        self.assertFalse(trend_mod.check_donchian_exit(pd.Series({"close": 8.0, "donchian_low_2_prev": 8.0}), policy.exit_window, "long"))
        self.assertTrue(trend_mod.check_donchian_exit(pd.Series({"close": 11.0, "donchian_high_2_prev": 10.0}), policy.exit_window, "short"))
        self.assertFalse(trend_mod.check_donchian_exit(pd.Series({"close": 10.0, "donchian_high_2_prev": 10.0}), policy.exit_window, "short"))

    def test_ema_trend_entry(self) -> None:
        policy = trend_mod.PolicyRun("ema", "4h", "ema", None, None, 4.0, use_ema_exit=True)

        self.assertTrue(trend_mod.policy_entry_signal(pd.Series({"close": 105.0, "ema50": 100.0, "ema200": 95.0}), policy, "long"))
        self.assertTrue(trend_mod.policy_entry_signal(pd.Series({"close": 90.0, "ema50": 95.0, "ema200": 100.0}), policy, "short"))
        self.assertFalse(trend_mod.policy_entry_signal(pd.Series({"close": 94.0, "ema50": 95.0, "ema200": 100.0}), policy, "long"))
        self.assertTrue(trend_mod.check_ema_exit(pd.Series({"close": 99.0, "ema50": 100.0}), "long"))
        self.assertFalse(trend_mod.check_ema_exit(pd.Series({"close": 101.0, "ema50": 100.0}), "long"))
        self.assertTrue(trend_mod.check_ema_exit(pd.Series({"close": 101.0, "ema50": 100.0}), "short"))
        self.assertFalse(trend_mod.check_ema_exit(pd.Series({"close": 99.0, "ema50": 100.0}), "short"))

    def test_atr_trailing_long_and_short(self) -> None:
        long_hit, long_trail = trend_mod.check_atr_trailing_stop("long", 103.0, 2.0, 110.0, 100.0, 3.0)
        short_hit, short_trail = trend_mod.check_atr_trailing_stop("short", 97.0, 2.0, 100.0, 90.0, 3.0)

        self.assertTrue(long_hit)
        self.assertAlmostEqual(float(long_trail), 104.0)
        self.assertTrue(short_hit)
        self.assertAlmostEqual(float(short_trail), 96.0)
        relaxed_long_hit, relaxed_long_trail = trend_mod.check_atr_trailing_stop("long", 105.0, 5.0, 110.0, 100.0, 3.0, current_trailing_stop=104.0)
        relaxed_short_hit, relaxed_short_trail = trend_mod.check_atr_trailing_stop("short", 95.0, 5.0, 100.0, 90.0, 3.0, current_trailing_stop=96.0)
        self.assertFalse(relaxed_long_hit)
        self.assertAlmostEqual(float(relaxed_long_trail), 104.0)
        self.assertFalse(relaxed_short_hit)
        self.assertAlmostEqual(float(relaxed_short_trail), 96.0)

    def test_same_symbol_only_one_position_and_max_positions(self) -> None:
        history_range = make_range(180)
        contexts = {symbol: make_context(symbol, 180) for symbol in [SYMBOL_A, SYMBOL_B, SYMBOL_C]}
        policy = trend_mod.PolicyRun("test_policy", "4h", "donchian", 1, 1, 4.0, use_donchian_exit=True)
        event_time = pd.Timestamp("2025-01-01T00:10:00+08:00")
        events = pd.DataFrame(
            [
                {"event_time": event_time, "policy_name": policy.policy_name, "symbol": SYMBOL_A, "source_policy_name": "src1", "timeframe": "4h", "atr_mult": 4.0, "exit_window": 1, "use_donchian_exit": True, "use_ema_exit": False, "entry_long": True, "entry_short": False, "strength_long": 3.0, "strength_short": 0.0, "close": 101.0, "atr14": 1.0},
                {"event_time": event_time, "policy_name": policy.policy_name, "symbol": SYMBOL_A, "source_policy_name": "src2", "timeframe": "4h", "atr_mult": 4.0, "exit_window": 1, "use_donchian_exit": True, "use_ema_exit": False, "entry_long": True, "entry_short": False, "strength_long": 2.0, "strength_short": 0.0, "close": 101.0, "atr14": 1.0},
                {"event_time": event_time, "policy_name": policy.policy_name, "symbol": SYMBOL_B, "source_policy_name": "src1", "timeframe": "4h", "atr_mult": 4.0, "exit_window": 1, "use_donchian_exit": True, "use_ema_exit": False, "entry_long": True, "entry_short": False, "strength_long": 4.0, "strength_short": 0.0, "close": 101.0, "atr14": 1.0},
                {"event_time": event_time, "policy_name": policy.policy_name, "symbol": SYMBOL_C, "source_policy_name": "src1", "timeframe": "4h", "atr_mult": 4.0, "exit_window": 1, "use_donchian_exit": True, "use_ema_exit": False, "entry_long": True, "entry_short": False, "strength_long": 1.0, "strength_short": 0.0, "close": 101.0, "atr14": 1.0},
            ]
        )

        with patch.object(trend_mod, "build_policy_events", return_value=events):
            bundle = trend_mod.simulate_portfolio_policy_with_audit(
                policy,
                contexts,
                history_range,
                portfolio_capital=1000.0,
                position_sizing="fixed_contract",
                fixed_size=1.0,
                rate=0.0,
                max_symbol_weight=1.0,
                max_portfolio_positions=2,
            )
        trades = bundle.trades

        self.assertEqual(len(trades.index), 2)
        self.assertEqual(set(trades["symbol"]), {SYMBOL_A, SYMBOL_B})
        self.assertEqual(trend_mod.compute_max_concurrent_positions(trades), 2)
        self.assertIn("max_portfolio_positions", set(bundle.rejected_signals["reason"]))
        self.assertIn("duplicate_signal_merged", set(bundle.rejected_signals["reason"]))

    def test_max_symbol_weight_rejects_oversized_fixed_contract(self) -> None:
        history_range = make_range(180)
        contexts = {SYMBOL_A: make_context(SYMBOL_A, 180)}
        policy = trend_mod.PolicyRun("test_policy", "4h", "donchian", 1, 1, 4.0, use_donchian_exit=True)
        events = pd.DataFrame(
            [
                {"event_time": pd.Timestamp("2025-01-01T00:10:00+08:00"), "policy_name": policy.policy_name, "symbol": SYMBOL_A, "source_policy_name": "src", "timeframe": "4h", "atr_mult": 4.0, "exit_window": 1, "use_donchian_exit": True, "use_ema_exit": False, "entry_long": True, "entry_short": False, "strength_long": 3.0, "strength_short": 0.0, "close": 101.0, "atr14": 1.0},
            ]
        )

        with patch.object(trend_mod, "build_policy_events", return_value=events):
            bundle = trend_mod.simulate_portfolio_policy_with_audit(
                policy,
                contexts,
                history_range,
                portfolio_capital=100.0,
                position_sizing="fixed_contract",
                fixed_size=10.0,
                rate=0.0,
                max_symbol_weight=0.01,
                max_portfolio_positions=1,
            )

        self.assertTrue(bundle.trades.empty)
        self.assertEqual(bundle.rejected_signals.iloc[0]["reason"], "max_symbol_weight")

    def test_symbol_contribution_and_concentration_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"policy_name": "p", "symbol": SYMBOL_A, "direction": "long", "entry_time": "2025-01-01T00:00:00+08:00", "exit_time": "2025-01-01T01:00:00+08:00", "net_pnl": 70.0, "no_cost_net_pnl": 70.0, "gross_pnl": 70.0, "fee": 0.0, "slippage": 0.0, "holding_minutes": 60.0, "turnover": 100.0},
                {"policy_name": "p", "symbol": SYMBOL_B, "direction": "long", "entry_time": "2025-01-01T01:00:00+08:00", "exit_time": "2025-01-01T02:00:00+08:00", "net_pnl": 30.0, "no_cost_net_pnl": 30.0, "gross_pnl": 30.0, "fee": 0.0, "slippage": 0.0, "holding_minutes": 60.0, "turnover": 100.0},
            ]
        )
        contribution = trend_mod.build_symbol_contribution(trades, capital=1000.0)

        self.assertAlmostEqual(float(trend_mod.calculate_largest_symbol_pnl_share(trades)), 0.7)
        self.assertAlmostEqual(float(contribution[contribution["symbol"] == SYMBOL_A].iloc[0]["pnl_share"]), 0.7)

    def test_top_5pct_trade_pnl_contribution(self) -> None:
        trades = pd.DataFrame({"net_pnl": [80.0] + [1.0] * 20})

        self.assertAlmostEqual(float(trend_mod.calculate_top_5pct_trade_pnl_contribution(trades)), 0.81)

    def test_cost_aware_and_no_cost_outputs(self) -> None:
        costs = trend_mod.calculate_trade_costs(
            entry_price=100.0,
            exit_price=110.0,
            volume=1.0,
            contract_size=1.0,
            direction="long",
            rate=0.001,
            absolute_slippage=0.5,
        )

        self.assertEqual(costs["no_cost_pnl"], 10.0)
        self.assertEqual(costs["no_cost_net_pnl"], 10.0)
        self.assertAlmostEqual(costs["net_pnl"], 8.79)
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
        self.assertGreaterEqual(short_costs["fee"], 0.0)
        self.assertGreaterEqual(short_costs["slippage"], 0.0)
        self.assertEqual(trend_mod.resolve_absolute_slippage_from_meta({"pricetick": 0.1}, SYMBOL_A, "ticks", 2), 0.2)

    def test_portfolio_equity_daily_pnl_and_drawdown(self) -> None:
        trades = pd.DataFrame(
            [
                {"policy_name": "p", "symbol": SYMBOL_A, "direction": "long", "entry_time": "2025-01-01T00:00:00+08:00", "exit_time": "2025-01-01T01:00:00+08:00", "net_pnl": 20.0, "no_cost_net_pnl": 20.0, "gross_pnl": 20.0, "fee": 0.0, "slippage": 0.0, "holding_minutes": 60.0, "turnover": 100.0},
                {"policy_name": "p", "symbol": SYMBOL_B, "direction": "short", "entry_time": "2025-01-01T00:30:00+08:00", "exit_time": "2025-01-01T02:00:00+08:00", "net_pnl": -30.0, "no_cost_net_pnl": -30.0, "gross_pnl": -30.0, "fee": 0.0, "slippage": 0.0, "holding_minutes": 90.0, "turnover": 100.0},
                {"policy_name": "p", "symbol": SYMBOL_A, "direction": "long", "entry_time": "2025-01-02T00:00:00+08:00", "exit_time": "2025-01-02T01:00:00+08:00", "net_pnl": 40.0, "no_cost_net_pnl": 40.0, "gross_pnl": 40.0, "fee": 0.0, "slippage": 0.0, "holding_minutes": 60.0, "turnover": 100.0},
            ]
        )
        equity = trend_mod.build_equity_curve(trades, capital=100.0)
        daily = trend_mod.build_daily_pnl(trades, capital=100.0)
        max_dd, max_ddpercent = trend_mod.max_drawdown_from_trades(trades, capital=100.0)

        self.assertAlmostEqual(float(equity["drawdown"].max()), 30.0)
        self.assertAlmostEqual(max_dd, 30.0)
        self.assertAlmostEqual(max_ddpercent, 25.0)
        self.assertEqual(trend_mod.compute_max_concurrent_positions(trades), 2)
        self.assertAlmostEqual(float(daily[daily["date"] == "2025-01-01"]["net_pnl"].iloc[0]), -10.0)

    def test_negative_total_contribution_emits_warning(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "policy_name": "p",
                    "trade_count": 2,
                    "net_pnl": -10.0,
                    "largest_symbol_pnl_share": 0.7,
                    "top_5pct_trade_pnl_contribution": -0.5,
                }
            ]
        )
        warnings = trend_mod.build_contribution_warnings(leaderboard)

        self.assertTrue(any("absolute contribution" in warning for warning in warnings))
        self.assertTrue(any("small trade sample" in warning for warning in warnings))

    def test_run_research_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "trend_v3"
            history_range = make_range(1440)
            summary = trend_mod.run_research(
                symbols=[SYMBOL_A, SYMBOL_B],
                split="train",
                history_range=history_range,
                output_dir=output_dir,
                timezone_name="Asia/Shanghai",
                capital=1000.0,
                slippage_mode="absolute",
                slippage=0.0,
                rate=0.0,
                max_runs=1,
                data_check_strict=True,
                logger=logging.getLogger("test_research_trend_following_v3"),
                bars_from_db=False,
                bars_by_symbol={SYMBOL_A: make_1m_bars(SYMBOL_A, 1440), SYMBOL_B: make_1m_bars(SYMBOL_B, 1440)},
                instrument_meta_by_symbol={SYMBOL_A: meta(SYMBOL_A), SYMBOL_B: meta(SYMBOL_B)},
            )

            for filename in [
                "trend_v3_summary.json",
                "trend_v3_policy_leaderboard.csv",
                "trend_v3_portfolio_equity_curve.csv",
                "trend_v3_portfolio_daily_pnl.csv",
                "trend_v3_trades.csv",
                "trend_v3_policy_by_symbol.csv",
                "trend_v3_policy_by_month.csv",
                "trend_v3_symbol_contribution.csv",
                "trend_v3_drawdown.csv",
                "trend_v3_report.md",
                "trend_v3_research_audit.json",
                "data_quality.json",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)
            data_quality = json.loads((output_dir / "data_quality.json").read_text(encoding="utf-8"))
            audit = json.loads((output_dir / "trend_v3_research_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["ready_symbols"], [SYMBOL_A, SYMBOL_B])
        self.assertTrue(data_quality["all_required_symbols_ready"])
        self.assertIn("no_lookahead_checks", audit)
        self.assertIn("sample_trades", audit)


if __name__ == "__main__":
    unittest.main()
