"""MR-5m Mean Reversion strategy for OKX perpetual swaps on vnpy.

5-minute Donchian Channel fade strategy. Fades breakouts: price breaks
above 24-bar high → SHORT, price breaks below 24-bar low → LONG.

Exit priority: midline take-profit > ATR stop > max_hold time.
ATR regime filter: skip entries when 5-min ATR is below p30 threshold.

v2.0 — 5min MR with ATR regime filter.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from vnpy.trader.constant import Direction, Interval, Offset
from vnpy_ctastrategy import (
    ArrayManager,
    BarData,
    BarGenerator,
    CtaTemplate,
    OrderData,
    StopOrder,
    TickData,
    TradeData,
)


class Mr5mStrategy(CtaTemplate):
    """5-min Mean Reversion: fade 24-bar breakout, midline exit, ATR stop."""

    author = "yiast"

    # --- Parameters ---
    notional_per_trade: float = 500.0  # USD notional per trade
    lookback: int = 24                 # Donchian lookback (bars)
    atr_period: int = 14               # ATR period
    atr_stop: float = 1.0              # ATR stop multiplier
    max_hold: int = 48                 # max hold bars (~4h in 5min)
    init_days: int = 7                 # warmup days
    price_offset: int = 2              # ticks for marketable orders

    # ATR regime filter — skip entries when ATR is below threshold
    atr_filter_on: bool = True         # enable ATR regime filter
    atr_filter_threshold: float = 0.0  # symbol-specific, set in on_init

    # --- State ---
    entry_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    hold_bars: int = 0
    atr_value: float = 0.0
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    midline_value: float = 0.0

    parameters = [
        "notional_per_trade", "lookback", "atr_period", "atr_stop",
        "max_hold", "init_days", "price_offset",
        "atr_filter_on", "atr_filter_threshold",
    ]
    variables = [
        "entry_price", "highest_since_entry", "lowest_since_entry",
        "hold_bars", "atr_value", "donchian_high", "donchian_low",
        "midline_value",
    ]

    # --- p30 thresholds from 3yr backtest (2023-2026) ---
    _ATR_THRESHOLDS = {
        "BTC": 81.5, "ETH": 4.64, "SOL": 0.245,
        "LINK": 0.0212, "DOGE": 0.0002,
    }

    # FIX: problem 1 — symbol-specific ctVal fallback; keyed by vt_symbol
    # Prevents silent 10^5x sizing error when get_size() returns None/0
    # (e.g. DOGE ctVal=1000 vs BTC ctVal=0.01)
    _CONTRACT_SPECS = {
        "BTCUSDT_SWAP.OKX":  {"ctVal": 0.01},
        "ETHUSDT_SWAP.OKX":  {"ctVal": 0.1},
        "SOLUSDT_SWAP.OKX":  {"ctVal": 1.0},
        "LINKUSDT_SWAP.OKX": {"ctVal": 1.0},
        "DOGEUSDT_SWAP.OKX": {"ctVal": 1000.0},
        # Also support short keys in case vt_symbol format varies
        "BTC-USDT-SWAP":  {"ctVal": 0.01},
        "ETH-USDT-SWAP":  {"ctVal": 0.1},
        "SOL-USDT-SWAP":  {"ctVal": 1.0},
        "LINK-USDT-SWAP": {"ctVal": 1.0},
        "DOGE-USDT-SWAP": {"ctVal": 1000.0},
    }

    def __init__(self, cta_engine: Any, strategy_name: str, vt_symbol: str, setting: dict) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 1-min → 5-min bar aggregation
        self.bg = BarGenerator(self.on_bar, window=5, on_window_bar=self.on_5min_bar, interval=Interval.MINUTE)

        self.am = ArrayManager(size=max(20, self.lookback + self.atr_period + 5))
        self.active_orders: set[str] = set()

        self._init_done: bool = False
        self._bar_count: int = 0
        self._5m_count: int = 0
        self._filtered_count: int = 0

        # Order management
        self._last_order_time: float = 0.0
        self._order_timeout: int = 300  # 5 min
        self._current_1m_bar: BarData | None = None

        # Auto-detect ATR threshold if not set
        if self.atr_filter_threshold <= 0:
            symbol_root = vt_symbol.split("_")[0].removesuffix("USDT").upper()
            self.atr_filter_threshold = self._ATR_THRESHOLDS.get(symbol_root, 0.0)

    def on_init(self) -> None:
        self.write_log("MR-5m 策略初始化")
        # FIX: problem 1 — validate contract multiplier before any trades
        self._validate_multiplier()
        self.load_bar(days=self.init_days, interval=Interval.MINUTE, callback=self.on_bar, use_database=True)
        self._init_done = True
        print(f"[{self.strategy_name}] INIT DONE | 1m bars={self._bar_count} 5m bars={self._5m_count} "
              f"ATR threshold={self.atr_filter_threshold}", flush=True)
        self.put_event()

    def on_start(self) -> None:
        self.write_log("MR-5m 策略启动")
        self.put_event()

    def on_stop(self) -> None:
        self.write_log(f"MR-5m 策略停止 | filtered={self._filtered_count}")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """1-min bar callback — feeds into BarGenerator for 5-min aggregation."""
        self._current_1m_bar = bar
        self._bar_count += 1
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData) -> None:
        """Strategy logic: runs once per completed 5-min bar."""
        self._5m_count += 1
        self.am.update_bar(bar)

        if not self.am.inited:
            return

        if not self.trading:
            if self._init_done:
                print(f"[{self.strategy_name}] WARNING: trading=False", flush=True)
            return

        # Order timeout
        if self.active_orders:
            elapsed = time.time() - self._last_order_time
            if elapsed > self._order_timeout:
                if self._init_done:
                    print(f"[{self.strategy_name}] ORDER TIMEOUT {elapsed:.0f}s", flush=True)
                if self.trading:
                    self.cancel_all()
                else:
                    self.active_orders.clear()
                    self._last_order_time = 0.0
            return

        if self.active_orders:
            return

        # --- Indicators ---
        self.atr_value = self._compute_atr()

        # FIX: problem 6 — use [-(lookback+1):-1] for exactly `lookback` bars excluding
        # current. ArrayManager uses a fixed-size numpy ring buffer; when it has
        # not yet wrapped around fully, negative indices can land on zero-initialized
        # slots BEFORE the valid window. The zero filter (highs[highs > 0]) is the
        # safety net — it makes the logic correct regardless of buffer fill state.
        if self.am.inited:
            highs = self.am.high[-(self.lookback + 1):-1]   # numpy slice, lookback bars
            lows  = self.am.low[-(self.lookback + 1):-1]
            # Filter zeros: ensures Donchian isn't pulled to 0 by init slots
            # that haven't been overwritten yet
            highs_nz = highs[highs > 0]
            lows_nz  = lows[lows > 0]
            if len(highs_nz) > 0 and len(lows_nz) > 0:
                self.donchian_high = float(highs_nz.max())
                self.donchian_low  = float(lows_nz.min())
            else:
                self.donchian_high = 0.0
                self.donchian_low  = 0.0
        else:
            self.donchian_high = 0.0
            self.donchian_low  = 0.0

        close = bar.close_price

        # --- Position management ---
        if self.pos != 0:
            self.hold_bars += 1
            self._update_extremes(bar)

            if not self.active_orders:
                # Midline take-profit
                if self.donchian_high > 0 and self.donchian_low > 0:
                    self.midline_value = (self.donchian_high + self.donchian_low) / 2
                    if ((self.pos > 0 and close >= self.midline_value) or
                        (self.pos < 0 and close <= self.midline_value)):
                        self._exit_position("midline")
                        return

                # ATR stop
                stop_dist = self.atr_stop * self.atr_value
                if self.pos > 0:
                    if bar.low_price <= self.entry_price - stop_dist:
                        self._exit_position("stop")
                        return
                else:
                    if bar.high_price >= self.entry_price + stop_dist:
                        self._exit_position("stop")
                        return

                # Max hold
                if self.hold_bars >= self.max_hold:
                    self._exit_position("max_hold")
            return

        # --- Entry signals ---
        if close <= 0 or self.atr_value <= 0:
            return

        # ATR regime filter
        if self.atr_filter_on and self.atr_filter_threshold > 0:
            if self.atr_value < self.atr_filter_threshold:
                self._filtered_count += 1
                if self._init_done and self._filtered_count % 100 == 0:
                    print(f"[{self.strategy_name}] filtered={self._filtered_count}", flush=True)
                return

        long_breakout = close > self.donchian_high and self.donchian_high > 0
        short_breakout = close < self.donchian_low and self.donchian_low > 0

        # Debug log
        if self._init_done and self._5m_count % 288 == 0:  # once per day (288 × 5min = 24h)
            self.midline_value = (self.donchian_high + self.donchian_low) / 2 if self.donchian_high > 0 else 0.0
            print(f"[{self.strategy_name}] 5m #{self._5m_count} | {bar.datetime} "
                  f"c={close:.4f} ATR={self.atr_value:.4f} "
                  f"DH={self.donchian_high:.4f} DL={self.donchian_low:.4f} "
                  f"MID={self.midline_value:.4f} pos={self.pos}", flush=True)

        if long_breakout:
            self._enter_short()
        elif short_breakout:
            self._enter_long()

    # --- Order helpers ---

    def _enter_long(self) -> None:
        bar = self._current_1m_bar
        if bar is None:
            return
        price = self._buy_price(bar.close_price)
        size = self._calc_size(price)
        if size <= 0:
            return
        ids = self.buy(price, size)
        if ids:
            self.active_orders.update(ids)
            self._last_order_time = time.time()
            print(f"[{self.strategy_name}] LONG | price={price:.4f} size={size}", flush=True)

    def _enter_short(self) -> None:
        bar = self._current_1m_bar
        if bar is None:
            return
        price = self._sell_price(bar.close_price)
        size = self._calc_size(price)
        if size <= 0:
            return
        ids = self.short(price, size)
        if ids:
            self.active_orders.update(ids)
            self._last_order_time = time.time()
            print(f"[{self.strategy_name}] SHORT | price={price:.4f} size={size}", flush=True)

    def _exit_position(self, reason: str) -> None:
        bar = self._current_1m_bar
        if bar is None:
            return
        if self.pos > 0:
            price = self._sell_price(bar.close_price)
            ids = self.sell(price, abs(self.pos))
        else:
            price = self._buy_price(bar.close_price)
            ids = self.cover(price, abs(self.pos))
        if ids:
            self.active_orders.update(ids)
            self._last_order_time = time.time()
            print(f"[{self.strategy_name}] EXIT {reason} | price={price:.4f} pos={self.pos}", flush=True)

    def _calc_size(self, price: float) -> int:
        if price <= 0 or self.notional_per_trade <= 0:
            return 0
        multiplier = self.get_size()
        # FIX: problem 1 — fallback to per-symbol CONTRACT_SPECS ctVal instead of
        # hardcoded 0.01 (BTC-only). DOGE ctVal=1000; using 0.01 gives 10^5x error.
        if multiplier is None or multiplier <= 0:
            spec = self._CONTRACT_SPECS.get(self.vt_symbol, {})
            multiplier = spec.get("ctVal", None)
            if multiplier is None or multiplier <= 0:
                print(f"[{self.strategy_name}] ERROR: no valid multiplier for {self.vt_symbol}, "
                      f"skipping order", flush=True)
                return 0  # safer to skip than use wrong multiplier
        contract_value = price * multiplier
        if contract_value <= 0:
            return 1
        size = round(self.notional_per_trade / contract_value)
        return max(1, min(size, 1000))

    def _buy_price(self, base: float) -> float:
        tick = self.get_pricetick()
        if tick is None or tick <= 0:
            tick = 0.01
        return base + tick * self.price_offset

    def _sell_price(self, base: float) -> float:
        tick = self.get_pricetick()
        if tick is None or tick <= 0:
            tick = 0.01
        return base - tick * self.price_offset

    def _compute_atr(self) -> float:
        # FIX: problem 4 — use vnpy built-in ATR which calls talib.ATR (Wilder smoothed,
        # alpha=1/N). Original SMA-of-TR can differ by 10-20%, causing filter/stop mismatch
        # vs backtest. self.am.atr() matches the talib.ATR used in backtest.
        if not self.am.inited:
            return 0.0
        return self.am.atr(self.atr_period)

    def _validate_multiplier(self) -> None:
        """FIX: problem 1 — validate contract multiplier at init time.
        Logs a clear warning if get_size() is unavailable so issue is visible immediately.
        """
        m = self.get_size()
        sym = self.vt_symbol
        spec = self._CONTRACT_SPECS.get(sym, {})
        expected = spec.get("ctVal", None)
        if m is None or m <= 0:
            if expected is not None:
                print(f"[{self.strategy_name}] WARNING: get_size()={m} for {sym}, "
                      f"will use CONTRACT_SPECS ctVal={expected} as fallback", flush=True)
            else:
                print(f"[{self.strategy_name}] ERROR: get_size()={m} for {sym} and "
                      f"no CONTRACT_SPECS fallback — orders will be skipped!", flush=True)
        elif expected is not None and abs(m - expected) / (expected + 1e-12) > 0.01:
            print(f"[{self.strategy_name}] WARNING: get_size()={m} differs from "
                  f"CONTRACT_SPECS ctVal={expected} for {sym}", flush=True)
        else:
            print(f"[{self.strategy_name}] OK: multiplier={m} for {sym}", flush=True)

    def _update_extremes(self, bar: BarData) -> None:
        if self.pos > 0 and bar.high_price > self.highest_since_entry:
            self.highest_since_entry = bar.high_price
        elif self.pos < 0 and bar.low_price < self.lowest_since_entry:
            self.lowest_since_entry = bar.low_price

    # --- Callbacks ---

    def on_order(self, order: OrderData) -> None:
        if order.is_active():
            self.active_orders.add(order.vt_orderid)
        else:
            self.active_orders.discard(order.vt_orderid)
        if self._init_done:
            print(f"[{self.strategy_name}] ORDER | {order.status} {order.direction} "
                  f"price={order.price} vol={order.volume} id={order.vt_orderid[:12]}", flush=True)

    def on_trade(self, trade: TradeData) -> None:
        # FIX: 问题4 — handle partial close properly
        if trade.offset == Offset.OPEN:
            self.entry_price = trade.price
            self.highest_since_entry = trade.price
            self.lowest_since_entry = trade.price
            self.hold_bars = 0
        elif trade.offset == Offset.CLOSE:
            if self.pos == 0:
                # Fully closed: reset all state
                self.entry_price = 0.0
                self.highest_since_entry = 0.0
                self.lowest_since_entry = 0.0
                self.hold_bars = 0
            # Partial close (pos != 0): keep entry_price and hold_bars as-is
            # (vnpy framework already updated self.pos)
        if self._init_done:
            print(
                f"[{self.strategy_name}] TRADE | {trade.direction} {trade.offset} "
                f"price={trade.price} vol={trade.volume} pos={self.pos}",
                flush=True,
            )
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass
