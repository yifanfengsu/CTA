from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_vsvcb_v1 as vsvcb
from history_time_utils import parse_history_range


SYMBOL = "BTCUSDT_SWAP_OKX.GLOBAL"


def make_config(**overrides: object) -> vsvcb.VsvcbConfig:
    values = {
        "bb_length": 2,
        "bb_std": 2.0,
        "bb_width_lookback": 2,
        "squeeze_quantile": 0.2,
        "breakout_window": 2,
        "volume_ma_window": 2,
        "volume_ratio": 1.5,
        "hold_bars": 2,
        "event_horizons": (3, 5, 10, 20),
        "fixed_notional": 1000.0,
        "fee_bps_per_side": 5.0,
        "slippage_bps_per_side": 5.0,
    }
    values.update(overrides)
    return vsvcb.VsvcbConfig(**values)


def make_frame(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    freq: str = "15min",
) -> pd.DataFrame:
    times = pd.date_range("2025-01-01T00:14:00+08:00", periods=len(closes), freq=freq)
    open_times = times - pd.Timedelta(minutes=14)
    previous = closes[0]
    rows = []
    for index, close in enumerate(closes):
        open_price = previous
        high = highs[index] if highs is not None else max(open_price, close)
        low = lows[index] if lows is not None else min(open_price, close)
        volume = volumes[index] if volumes is not None else 100.0
        rows.append(
            {
                "open_time": open_times[index],
                "datetime": times[index],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def make_1m_bars(minutes: int = 150) -> pd.DataFrame:
    times = pd.date_range("2025-01-01T00:00:00+08:00", periods=minutes, freq="min")
    rows = []
    price = 100.0
    for index, timestamp in enumerate(times):
        if 45 <= index < 60:
            price = 101.0 + (index - 45) * 0.02
        elif index >= 60:
            price += 0.01
        volume = 100.0
        if 45 <= index < 60:
            volume = 400.0
        rows.append(
            {
                "datetime": timestamp,
                "open": price,
                "high": price + 0.05,
                "low": price - 0.05,
                "close": price,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def write_funding_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["inst_id", "funding_time", "funding_time_utc", "funding_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "inst_id": "BTC-USDT-SWAP",
                "funding_time": "1735660800000",
                "funding_time_utc": "2024-12-31T16:00:00+00:00",
                "funding_rate": "0.001",
            }
        )


class ResearchVsvcbV1Test(unittest.TestCase):
    def test_bb_width_calculation(self) -> None:
        frame = make_frame([100.0, 102.0, 104.0])
        result = vsvcb.add_vsvcb_indicators(frame, make_config())
        mean = (100.0 + 102.0) / 2.0
        std = ((100.0 - mean) ** 2 + (102.0 - mean) ** 2) ** 0.5 / (2.0**0.5)
        expected = (4.0 * std) / mean

        self.assertAlmostEqual(float(result.iloc[1]["bb_width_raw"]), expected)

    def test_squeeze_uses_previous_bar_not_current_breakout_bar(self) -> None:
        frame = make_frame([100.0, 100.0, 100.0, 110.0, 120.0])
        result = vsvcb.add_vsvcb_indicators(frame, make_config())
        expected = (
            result["bb_width_raw"].shift(1)
            <= result["bb_width_raw"].rolling(2, min_periods=2).quantile(0.2).shift(1)
        ).fillna(False)

        self.assertEqual(bool(result.iloc[3]["squeeze"]), bool(expected.iloc[3]))
        self.assertNotEqual(float(result.iloc[3]["bb_width_raw"]), float(result.iloc[3]["bb_width"]))

    def test_volume_ma_uses_shift_one(self) -> None:
        frame = make_frame([100.0, 100.0, 100.0, 101.0], volumes=[100.0, 100.0, 100.0, 300.0])
        result = vsvcb.add_vsvcb_indicators(frame, make_config())

        self.assertAlmostEqual(float(result.iloc[3]["volume_ma_prev"]), 100.0)
        self.assertAlmostEqual(float(result.iloc[3]["volume_ratio"]), 3.0)

    def test_breakout_boundary_uses_previous_high_low(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 101.0],
            highs=[100.0, 100.0, 100.0, 101.0],
            lows=[99.0, 99.0, 99.0, 100.0],
        )
        result = vsvcb.add_vsvcb_indicators(frame, make_config())

        self.assertAlmostEqual(float(result.iloc[3]["upper_boundary"]), 100.0)
        self.assertTrue(bool(result.iloc[3]["long_breakout"]))

    def test_close_equal_boundary_does_not_trigger(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 100.0],
            highs=[100.0, 100.0, 100.0, 100.0],
            lows=[99.0, 99.0, 99.0, 99.0],
        )
        result = vsvcb.add_vsvcb_indicators(frame, make_config())

        self.assertFalse(bool(result.iloc[3]["long_breakout"]))

    def test_long_breakout_event_generation(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            highs=[100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            lows=[99.0, 99.0, 99.0, 100.0, 101.0, 102.0, 103.0],
            volumes=[100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0],
        )
        events, _warnings = vsvcb.generate_events_for_frame(frame, SYMBOL, "15m", make_config())

        self.assertIn("long", set(events["direction"]))

    def test_short_breakout_event_generation(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 98.0, 97.0, 96.0, 95.0],
            highs=[101.0, 101.0, 101.0, 99.0, 98.0, 97.0, 96.0],
            lows=[99.0, 99.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            volumes=[100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0],
        )
        events, _warnings = vsvcb.generate_events_for_frame(frame, SYMBOL, "15m", make_config())

        self.assertIn("short", set(events["direction"]))

    def test_same_bar_long_short_anomaly_is_not_generated(self) -> None:
        frame = make_frame(
            [7.0, 7.0, 7.0, 7.0, 8.0],
            highs=[5.0, 5.0, 5.0, 5.0, 5.0],
            lows=[10.0, 10.0, 10.0, 10.0, 10.0],
            volumes=[100.0, 100.0, 100.0, 300.0, 100.0],
        )
        events, warnings = vsvcb.generate_events_for_frame(frame, SYMBOL, "15m", make_config())

        self.assertTrue(events.empty)
        self.assertTrue(any("same-bar long/short" in item for item in warnings))

    def test_abcd_e_grouping_and_reverse_direction(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            highs=[100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            lows=[99.0, 99.0, 99.0, 100.0, 101.0, 102.0, 103.0],
            volumes=[100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0],
        )
        events, _warnings = vsvcb.generate_events_for_frame(frame, SYMBOL, "15m", make_config())
        by_group = {row["group"]: row["direction"] for _, row in events.iterrows()}

        self.assertTrue({"A", "B", "C", "D", "E"}.issubset(set(events["group"])))
        self.assertEqual(by_group["D"], "long")
        self.assertEqual(by_group["E"], "short")

    def test_entry_and_exit_use_next_open_and_t_plus_hold_plus_one_open(self) -> None:
        frame = make_frame(
            [100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            highs=[100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            lows=[99.0, 99.0, 99.0, 100.0, 101.0, 102.0, 103.0],
            volumes=[100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0],
        )
        config = make_config(hold_bars=2)
        events, _warnings = vsvcb.generate_events_for_frame(frame, SYMBOL, "15m", config)
        events = vsvcb.assign_event_ids(events[events["group"] == "D"])
        splits = vsvcb.build_time_splits(
            parse_history_range("2025-01-01T00:00:00+08:00", "2025-01-01T02:00:00+08:00", timedelta(minutes=1), "Asia/Shanghai")
        )
        trades, _warnings = vsvcb.simulate_fixed_hold_trades(
            events,
            {(SYMBOL, "15m"): vsvcb.add_vsvcb_indicators(frame, config)},
            config,
            {},
            splits,
        )
        trade = trades.iloc[0]
        event_pos = int(events.iloc[0]["bar_index"])

        self.assertEqual(pd.Timestamp(trade["entry_time"]), pd.Timestamp(frame.iloc[event_pos + 1]["open_time"]))
        self.assertEqual(pd.Timestamp(trade["exit_time"]), pd.Timestamp(frame.iloc[event_pos + 3]["open_time"]))

    def test_single_symbol_single_group_position_conflict_filter(self) -> None:
        config = make_config(hold_bars=2)
        frame = make_frame([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        events = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "timestamp": frame.iloc[1]["datetime"],
                    "symbol": SYMBOL,
                    "timeframe": "15m",
                    "group": "D",
                    "direction": "long",
                    "bar_index": 1,
                    "inst_id": "BTC-USDT-SWAP",
                    "reversal_flag_3": False,
                    "reversal_flag_5": False,
                    "reversal_flag_10": False,
                },
                {
                    "event_id": "e2",
                    "timestamp": frame.iloc[2]["datetime"],
                    "symbol": SYMBOL,
                    "timeframe": "15m",
                    "group": "D",
                    "direction": "long",
                    "bar_index": 2,
                    "inst_id": "BTC-USDT-SWAP",
                    "reversal_flag_3": False,
                    "reversal_flag_5": False,
                    "reversal_flag_10": False,
                },
            ]
        )
        splits = vsvcb.build_time_splits(
            parse_history_range("2025-01-01T00:00:00+08:00", "2025-01-01T02:00:00+08:00", timedelta(minutes=1), "Asia/Shanghai")
        )
        trades, warnings = vsvcb.simulate_fixed_hold_trades(events, {(SYMBOL, "15m"): frame}, config, {}, splits)

        self.assertEqual(len(trades.index), 1)
        self.assertTrue(any("single_position_filter" in item for item in warnings))

    def test_funding_timestamp_alignment_is_inclusive(self) -> None:
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(
                    ["2024-12-31T16:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T08:00:00Z"],
                    utc=True,
                ),
                "funding_rate": [0.001, -0.001, 0.002],
            }
        )
        pnl, count = vsvcb.funding_pnl_for_interval(
            funding,
            pd.Timestamp("2025-01-01T00:00:00+08:00"),
            pd.Timestamp("2025-01-01T08:00:00+08:00"),
            "long",
            1000.0,
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(pnl, 0.0)

    def test_split_60_20_20_by_time(self) -> None:
        history_range = parse_history_range("2025-01-01", "2025-01-10", timedelta(minutes=1), "Asia/Shanghai")
        splits = vsvcb.build_time_splits(history_range)
        labels = [
            vsvcb.assign_split_for_time(timestamp, splits)
            for timestamp in pd.date_range("2025-01-01T00:00:00+08:00", periods=10, freq="D")
        ]

        self.assertEqual(labels.count("train"), 6)
        self.assertEqual(labels.count("validation"), 2)
        self.assertEqual(labels.count("oos"), 2)

    def test_concentration_calculation(self) -> None:
        trades = pd.DataFrame(
            [
                {"symbol": "A", "funding_adjusted_pnl": 4.0},
                {"symbol": "B", "funding_adjusted_pnl": 3.0},
                {"symbol": "C", "funding_adjusted_pnl": 3.0},
            ]
        )
        pnl_share, symbol = vsvcb.largest_symbol_pnl_share(trades, "funding_adjusted_pnl")
        count_share, _count_symbol = vsvcb.largest_symbol_trade_share(trades)
        top_share, top_pnl, top_count = vsvcb.top_trade_contribution(trades, "funding_adjusted_pnl")

        self.assertEqual(symbol, "A")
        self.assertAlmostEqual(float(pnl_share), 0.4)
        self.assertAlmostEqual(float(count_share), 1.0 / 3.0)
        self.assertAlmostEqual(float(top_share), 0.4)
        self.assertEqual(top_pnl, 4.0)
        self.assertEqual(top_count, 1)

    def test_summary_json_gates(self) -> None:
        rows = []
        symbols = list(vsvcb.DEFAULT_SYMBOLS)
        for split, count in [("train", 30), ("validation", 10), ("oos", 10)]:
            for index in range(count):
                symbol = symbols[index % len(symbols)]
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": "15m",
                        "group": "D",
                        "direction": "long",
                        "split": split,
                        "no_cost_pnl": 5.0,
                        "cost_aware_pnl": 3.0,
                        "funding_adjusted_pnl": 3.0,
                    }
                )
        for group in ["A", "B", "C", "E"]:
            for index in range(3):
                rows.append(
                    {
                        "symbol": symbols[index],
                        "timeframe": "15m",
                        "group": group,
                        "direction": "short",
                        "split": "oos",
                        "no_cost_pnl": -1.0,
                        "cost_aware_pnl": -2.0,
                        "funding_adjusted_pnl": -2.0,
                    }
                )
        gates = vsvcb.evaluate_phase1_gates(pd.DataFrame(rows), symbols, True, make_config())

        self.assertTrue(gates["train_pass"])
        self.assertTrue(gates["validation_pass"])
        self.assertTrue(gates["oos_pass"])
        self.assertTrue(gates["cost_aware_pass"])
        self.assertTrue(gates["funding_adjusted_pass"])
        self.assertTrue(gates["trade_count_pass"])
        self.assertTrue(gates["concentration_pass"])
        self.assertTrue(gates["reverse_test_pass"])
        self.assertTrue(gates["ablation_pass"])
        self.assertTrue(gates["continue_to_phase2"])

    def test_run_research_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            funding_dir = root / "funding"
            output_dir = root / "reports"
            write_funding_csv(funding_dir / "BTC-USDT-SWAP_funding_2025-01-01_2025-01-01.csv")
            history_range = parse_history_range(
                "2025-01-01T00:00:00+08:00",
                "2025-01-01T02:29:00+08:00",
                timedelta(minutes=1),
                "Asia/Shanghai",
            )
            summary = vsvcb.run_research(
                symbols=[SYMBOL],
                timeframes=["15m"],
                history_range=history_range,
                output_dir=output_dir,
                funding_dir=funding_dir,
                database_path=root / "missing.db",
                config=make_config(),
                data_check_strict=True,
                bars_by_symbol={SYMBOL: make_1m_bars(150)},
            )
            required = [
                "events.csv",
                "trades.csv",
                "summary.json",
                "summary.md",
                "event_group_summary.csv",
                "trade_group_summary.csv",
                "by_symbol.csv",
                "by_timeframe.csv",
                "by_split.csv",
                "concentration.csv",
                "reverse_test.csv",
                "funding_summary.csv",
                "data_quality.json",
                "postmortem_draft.md",
            ]

            for filename in required:
                self.assertTrue((output_dir / filename).exists(), filename)
            loaded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(loaded["strategy_development_allowed"])
            self.assertFalse(loaded["demo_live_allowed"])
            self.assertEqual(summary["status"], "research_only")

    def test_makefile_target_exists(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-vsvcb-v1:", text)
        self.assertIn("scripts/research_vsvcb_v1.py", text)


if __name__ == "__main__":
    unittest.main()
