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

import research_csrb_v1 as csrb
from history_time_utils import parse_history_range


SYMBOL = "BTCUSDT_SWAP_OKX.GLOBAL"


def make_config(**overrides: object) -> csrb.CsrbConfig:
    values = {
        "fixed_notional": 1000.0,
        "fee_bps_per_side": 5.0,
        "slippage_bps_per_side": 5.0,
        "buffer_atr": 0.0,
        "range_min_bars": 2,
        "hold_bars": 2,
        "event_horizons": (4, 8, 16, 32),
        "atr_window": 2,
        "random_seed": 17,
    }
    values.update(overrides)
    return csrb.CsrbConfig(**values)


def make_closed_frame() -> pd.DataFrame:
    open_times = pd.date_range("2025-01-01T00:00:00+00:00", periods=96, freq="15min")
    rows = []
    for index, open_time in enumerate(open_times):
        close_time = open_time + pd.Timedelta(minutes=14)
        open_price = 100.0
        close = 100.0
        high = 101.0
        low = 99.0
        if index == 32:
            close = 106.0
            high = 106.0
            low = 99.0
        if index == 52:
            close = 110.0
            high = 110.0
            low = 99.0
        rows.append(
            {
                "open_time": open_time,
                "datetime": close_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


def make_1m_bars() -> pd.DataFrame:
    times = pd.date_range("2025-01-01T00:00:00+00:00", periods=24 * 60, freq="min")
    rows = []
    for index, timestamp in enumerate(times):
        price = 100.0
        high = 101.0
        low = 99.0
        if index == 8 * 60 + 14:
            price = 106.0
            high = 106.0
        if index == 13 * 60 + 14:
            price = 110.0
            high = 110.0
        rows.append(
            {
                "datetime": timestamp,
                "open": 100.0,
                "high": high,
                "low": low,
                "close": price,
                "volume": 100.0,
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
        for millis, timestamp, rate in [
            ("1735689600000", "2025-01-01T00:00:00+00:00", "0.001"),
            ("1735718400000", "2025-01-01T08:00:00+00:00", "-0.001"),
            ("1735747200000", "2025-01-01T16:00:00+00:00", "0.001"),
        ]:
            writer.writerow(
                {
                    "inst_id": "BTC-USDT-SWAP",
                    "funding_time": millis,
                    "funding_time_utc": timestamp,
                    "funding_rate": rate,
                }
            )


class ResearchCsrbV1Test(unittest.TestCase):
    def test_session_range_calculation_correct(self) -> None:
        frame = make_closed_frame()
        events, _warnings = csrb.generate_session_breakouts_for_frame(frame, SYMBOL, "15m", make_config())
        b_event = events[events["group"] == "B"].iloc[0]

        self.assertEqual(b_event["session_type"], "asia_to_europe")
        self.assertAlmostEqual(float(b_event["range_high"]), 101.0)
        self.assertAlmostEqual(float(b_event["range_low"]), 99.0)
        self.assertEqual(int(b_event["range_bar_count"]), 32)

    def test_breakout_window_only_in_specified_time(self) -> None:
        frame = make_closed_frame()
        events, _warnings = csrb.generate_session_breakouts_for_frame(frame, SYMBOL, "15m", make_config())
        b_events = events[events["group"] == "B"]

        self.assertTrue((b_events["timestamp"].map(csrb.minute_of_day) >= 8 * 60).all())
        self.assertTrue((b_events["timestamp"].map(csrb.minute_of_day) <= 11 * 60 + 59).all())

    def test_range_does_not_use_breakout_window_data(self) -> None:
        frame = make_closed_frame()
        events, _warnings = csrb.generate_session_breakouts_for_frame(frame, SYMBOL, "15m", make_config())
        b_event = events[events["group"] == "B"].iloc[0]

        self.assertLess(float(b_event["range_high"]), float(b_event["close"]))
        self.assertEqual(pd.Timestamp(b_event["range_end"]).hour, 7)

    def test_atr_buffer_uses_t_minus_one(self) -> None:
        frame = make_closed_frame()
        frame.loc[32, "high"] = 200.0
        frame.loc[32, "low"] = 50.0
        frame.loc[32, "close"] = 104.0
        config = make_config(buffer_atr=1.0)
        events, _warnings = csrb.generate_session_breakouts_for_frame(frame, SYMBOL, "15m", config)
        b_event = events[events["group"] == "B"].iloc[0]
        indicated = csrb.add_csrb_indicators(frame, config)

        self.assertAlmostEqual(float(b_event["atr_prev"]), float(indicated.iloc[32]["atr_prev"]))
        self.assertAlmostEqual(float(b_event["breakout_boundary"]), 101.0 + float(indicated.iloc[32]["atr_prev"]))
        self.assertLess(float(indicated.iloc[32]["atr_prev"]), float(indicated.iloc[32]["atr"]))

    def test_entry_uses_next_open(self) -> None:
        frame = make_closed_frame()
        events, _warnings = csrb.generate_events_for_frame(frame, SYMBOL, "15m", make_config())
        events = csrb.assign_event_ids(events[events["group"] == "B"].head(1))
        splits = csrb.build_time_splits(
            parse_history_range("2025-01-01", "2025-01-01", timedelta(minutes=1), "UTC"),
            "UTC",
        )
        trades, _warnings = csrb.simulate_fixed_hold_trades(
            events,
            {(SYMBOL, "15m"): csrb.add_csrb_indicators(frame, make_config())},
            make_config(),
            {},
            splits,
        )
        event_pos = int(events.iloc[0]["bar_index"])

        self.assertEqual(pd.Timestamp(trades.iloc[0]["entry_time"]), pd.Timestamp(frame.iloc[event_pos + 1]["open_time"]))

    def test_exit_uses_open_t_plus_hold_plus_one(self) -> None:
        frame = make_closed_frame()
        config = make_config(hold_bars=3)
        events, _warnings = csrb.generate_events_for_frame(frame, SYMBOL, "15m", config)
        events = csrb.assign_event_ids(events[events["group"] == "B"].head(1))
        splits = csrb.build_time_splits(
            parse_history_range("2025-01-01", "2025-01-01", timedelta(minutes=1), "UTC"),
            "UTC",
        )
        trades, _warnings = csrb.simulate_fixed_hold_trades(
            events,
            {(SYMBOL, "15m"): csrb.add_csrb_indicators(frame, config)},
            config,
            {},
            splits,
        )
        event_pos = int(events.iloc[0]["bar_index"])

        self.assertEqual(pd.Timestamp(trades.iloc[0]["exit_time"]), pd.Timestamp(frame.iloc[event_pos + 4]["open_time"]))

    def test_asia_to_europe_event_generation(self) -> None:
        events, _warnings = csrb.generate_events_for_frame(make_closed_frame(), SYMBOL, "15m", make_config())

        self.assertTrue(((events["group"] == "B") & (events["session_type"] == "asia_to_europe")).any())

    def test_europe_to_us_event_generation(self) -> None:
        events, _warnings = csrb.generate_events_for_frame(make_closed_frame(), SYMBOL, "15m", make_config())

        self.assertTrue(((events["group"] == "C") & (events["session_type"] == "europe_to_us")).any())

    def test_session_agnostic_baseline_generation(self) -> None:
        events, _warnings = csrb.generate_events_for_frame(make_closed_frame(), SYMBOL, "15m", make_config())

        self.assertIn("A", set(events["group"]))
        self.assertIn("ordinary_rolling", set(events["session_type"]))

    def test_random_time_control_generation(self) -> None:
        events, _warnings = csrb.generate_events_for_frame(make_closed_frame(), SYMBOL, "15m", make_config())
        core = events[events["group"].isin(["B", "C"])]
        random = events[events["group"] == "D"]

        self.assertEqual(len(random.index), len(core.index))
        self.assertEqual(random["session_type"].unique().tolist(), ["random_time_control"])

    def test_reverse_test_direction_correct(self) -> None:
        events, _warnings = csrb.generate_events_for_frame(make_closed_frame(), SYMBOL, "15m", make_config())
        b_event = events[events["group"] == "B"].iloc[0]
        reverse = events[(events["group"] == "E") & (events["source_session_type"] == "asia_to_europe")].iloc[0]

        self.assertEqual(b_event["direction"], "long")
        self.assertEqual(reverse["direction"], "short")

    def test_single_symbol_single_group_position_filter(self) -> None:
        frame = make_closed_frame()
        config = make_config(hold_bars=4)
        events = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "timestamp": frame.iloc[32]["datetime"],
                    "symbol": SYMBOL,
                    "inst_id": "BTC-USDT-SWAP",
                    "timeframe": "15m",
                    "group": "A",
                    "session_type": "ordinary_rolling",
                    "source_session_type": "ordinary_rolling",
                    "session_date": "2025-01-01",
                    "direction": "long",
                    "bar_index": 32,
                },
                {
                    "event_id": "e2",
                    "timestamp": frame.iloc[33]["datetime"],
                    "symbol": SYMBOL,
                    "inst_id": "BTC-USDT-SWAP",
                    "timeframe": "15m",
                    "group": "A",
                    "session_type": "ordinary_rolling",
                    "source_session_type": "ordinary_rolling",
                    "session_date": "2025-01-01",
                    "direction": "long",
                    "bar_index": 33,
                },
            ]
        )
        splits = csrb.build_time_splits(
            parse_history_range("2025-01-01", "2025-01-01", timedelta(minutes=1), "UTC"),
            "UTC",
        )
        trades, warnings = csrb.simulate_fixed_hold_trades(
            events,
            {(SYMBOL, "15m"): frame},
            config,
            {},
            splits,
        )

        self.assertEqual(len(trades.index), 1)
        self.assertTrue(any("single_position_filter" in item for item in warnings))

    def test_funding_alignment_inclusive(self) -> None:
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T08:00:00Z", "2025-01-01T16:00:00Z"],
                    utc=True,
                ),
                "funding_rate": [0.001, -0.001, 0.002],
            }
        )
        pnl, count = csrb.funding_pnl_for_interval(
            funding,
            pd.Timestamp("2025-01-01T00:00:00+00:00"),
            pd.Timestamp("2025-01-01T08:00:00+00:00"),
            "long",
            1000.0,
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(pnl, 0.0)

    def test_split_60_20_20_correct(self) -> None:
        history_range = parse_history_range("2025-01-01", "2025-01-10", timedelta(minutes=1), "UTC")
        splits = csrb.build_time_splits(history_range, "UTC")
        labels = [
            csrb.assign_split_for_time(timestamp, splits)
            for timestamp in pd.date_range("2025-01-01T00:00:00+00:00", periods=10, freq="D")
        ]

        self.assertEqual(labels.count("train"), 6)
        self.assertEqual(labels.count("validation"), 2)
        self.assertEqual(labels.count("oos"), 2)

    def test_concentration_calculation_correct(self) -> None:
        trades = pd.DataFrame(
            [
                {"symbol": "A", "funding_adjusted_pnl": 4.0},
                {"symbol": "B", "funding_adjusted_pnl": 3.0},
                {"symbol": "C", "funding_adjusted_pnl": 3.0},
            ]
        )
        pnl_share, symbol = csrb.largest_symbol_pnl_share(trades, "funding_adjusted_pnl")
        count_share, _count_symbol = csrb.largest_symbol_trade_share(trades)
        top_share, top_pnl, top_count = csrb.top_trade_contribution(trades, "funding_adjusted_pnl")

        self.assertEqual(symbol, "A")
        self.assertAlmostEqual(float(pnl_share), 0.4)
        self.assertAlmostEqual(float(count_share), 1.0 / 3.0)
        self.assertAlmostEqual(float(top_share), 0.4)
        self.assertEqual(top_pnl, 4.0)
        self.assertEqual(top_count, 1)

    def test_summary_json_gates_correct(self) -> None:
        rows = []
        symbols = list(csrb.DEFAULT_SYMBOLS)
        for split, count in [("train", 30), ("validation", 10), ("oos", 10)]:
            for index in range(count):
                symbol = symbols[index % len(symbols)]
                rows.append(
                    {
                        "symbol": symbol,
                        "inst_id": csrb.symbol_to_inst_id(symbol),
                        "timeframe": "15m",
                        "group": "B" if index % 2 == 0 else "C",
                        "session_type": "asia_to_europe" if index % 2 == 0 else "europe_to_us",
                        "source_session_type": "asia_to_europe" if index % 2 == 0 else "europe_to_us",
                        "direction": "long",
                        "split": split,
                        "no_cost_pnl": 5.0,
                        "cost_aware_pnl": 3.0,
                        "funding_adjusted_pnl": 3.0,
                    }
                )
        for group in ["A", "D", "E"]:
            for index in range(3):
                symbol = symbols[index]
                rows.append(
                    {
                        "symbol": symbol,
                        "inst_id": csrb.symbol_to_inst_id(symbol),
                        "timeframe": "15m",
                        "group": group,
                        "session_type": "ordinary_rolling" if group == "A" else group,
                        "source_session_type": "asia_to_europe",
                        "direction": "short",
                        "split": "oos",
                        "no_cost_pnl": -1.0,
                        "cost_aware_pnl": -2.0,
                        "funding_adjusted_pnl": -2.0,
                    }
                )
        gates = csrb.evaluate_phase1_gates(pd.DataFrame(rows), symbols, True, make_config())

        self.assertTrue(gates["train_pass"])
        self.assertTrue(gates["validation_pass"])
        self.assertTrue(gates["oos_pass"])
        self.assertTrue(gates["cost_aware_pass"])
        self.assertTrue(gates["funding_adjusted_pass"])
        self.assertTrue(gates["trade_count_pass"])
        self.assertTrue(gates["concentration_pass"])
        self.assertTrue(gates["reverse_test_pass"])
        self.assertTrue(gates["session_vs_baseline_pass"])
        self.assertTrue(gates["continue_to_phase2"])

    def test_run_research_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            funding_dir = root / "funding"
            output_dir = root / "reports"
            write_funding_csv(funding_dir / "BTC-USDT-SWAP_funding_2025-01-01_2025-01-01.csv")
            history_range = parse_history_range("2025-01-01", "2025-01-01", timedelta(minutes=1), "UTC")
            summary = csrb.run_research(
                symbols=[SYMBOL],
                timeframes=["15m"],
                data_range=history_range,
                session_timezone="UTC",
                report_timezone="UTC",
                output_dir=output_dir,
                funding_dir=funding_dir,
                database_path=root / "missing.db",
                config=make_config(),
                data_check_strict=True,
                bars_by_symbol={SYMBOL: make_1m_bars()},
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
                "session_summary.csv",
                "concentration.csv",
                "reverse_test.csv",
                "random_time_control.csv",
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

        self.assertIn("research-csrb-v1:", text)
        self.assertIn("scripts/research_csrb_v1.py", text)


if __name__ == "__main__":
    unittest.main()
