"""MR-v1 Mean Reversion strategy for OKX perpetual swaps on vnpy.

Fades 4h breakouts: long breakout -> short, short breakout -> long.
Exits with ATR stop or max hold. Research-only for demo trading.

Aggregates 1m bars into 4h bars manually (no BarGenerator window limit).
"""

from __future__ import annotations

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


class MrV1Strategy(CtaTemplate):
    """Mean reversion: fade 4h breakout, ATR stop exit."""

    author = "yiast"

    # --- Parameters ---
    notional_per_trade: float = 500.0  # USD notional per trade
    lookback: int = 8                # breakout lookback bars (4h)
    atr_window: int = 14             # ATR period
    atr_stop: float = 1.0            # ATR stop multiplier
    max_hold: int = 60               # max hold bars (4h)
    init_days: int = 60              # warmup days
    price_offset: int = 2            # ticks for marketable orders

    # --- State ---
    entry_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    hold_bars: int = 0
    atr_value: float = 0.0
    donchian_high: float = 0.0
    donchian_low: float = 0.0

    parameters = [
        "notional_per_trade", "lookback", "atr_window", "atr_stop",
        "max_hold", "init_days", "price_offset",
    ]
    variables = [
        "entry_price", "highest_since_entry", "lowest_since_entry",
        "hold_bars", "atr_value", "donchian_high", "donchian_low",
    ]

    def __init__(self, cta_engine: Any, strategy_name: str, vt_symbol: str, setting: dict) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # Tick → 1m bar → 4h bar aggregation
        self.bg_tick = BarGenerator(self.on_bar)

        # Manual 4h bar aggregation from 1m bars
        self._4h_bars: list[BarData] = []        # collected 4h bars (full history)
        self._4h_current: BarData | None = None   # currently building 4h bar
        self._4h_minute_count: int = 0            # minutes collected in current 4h
        self._4h_last_hour: int = -1              # slot tracking

        self.am_4h = ArrayManager(size=max(200, self.lookback + 50))
        self.active_orders: set[str] = set()

    def on_init(self) -> None:
        self.write_log("MR-v1 策略初始化")
        self.load_bar(days=self.init_days, interval=Interval.MINUTE, callback=self.on_bar, use_database=True)
        self.write_log(f"历史K线预热完成, 4h bars={len(self._4h_bars)}")
        self.put_event()

    def on_start(self) -> None:
        self.write_log("MR-v1 策略启动")
        self.put_event()

    def on_stop(self) -> None:
        self.write_log("MR-v1 策略停止")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        self.bg_tick.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """Aggregate 1m bars into 4h bars, then run strategy logic."""
        self._aggregate_4h(bar)

    def _aggregate_4h(self, bar: BarData) -> None:
        """Build 4h closed bars from 1m bars."""
        dt = bar.datetime
        # 4h slot: hour // 4
        slot = dt.hour // 4

        if self._4h_current is None or slot != self._4h_last_hour:
            # Push previous completed 4h bar
            if self._4h_current is not None and self._4h_minute_count >= 180:  # at least 75% complete
                self._4h_bars.append(self._4h_current)
                self._on_4h_bar(self._4h_current)

            # Start new 4h bar
            self._4h_current = BarData(
                symbol=bar.symbol, exchange=bar.exchange,
                datetime=dt.replace(minute=0, second=0, microsecond=0),
                gateway_name=bar.gateway_name,
                open_price=bar.open_price, high_price=bar.high_price,
                low_price=bar.low_price, close_price=bar.close_price,
                volume=bar.volume, turnover=bar.turnover,
                open_interest=bar.open_interest,
            )
            self._4h_minute_count = 1
            self._4h_last_hour = slot
        else:
            # Update current 4h bar
            self._4h_current.high_price = max(self._4h_current.high_price, bar.high_price)
            self._4h_current.low_price = min(self._4h_current.low_price, bar.low_price)
            self._4h_current.close_price = bar.close_price
            self._4h_current.volume += bar.volume
            self._4h_current.turnover += bar.turnover
            self._4h_current.open_interest = bar.open_interest
            self._4h_minute_count += 1

    def _on_4h_bar(self, bar: BarData) -> None:
        """Strategy logic: runs once per completed 4h bar."""
        self.am_4h.update_bar(bar)
        if not self.am_4h.inited:
            return

        # Indicators
        self.atr_value = self._compute_atr()
        if len(self.am_4h.high) > self.lookback:
            self.donchian_high = max(self.am_4h.high[-self.lookback:-1])
            self.donchian_low = min(self.am_4h.low[-self.lookback:-1])
        else:
            self.donchian_high = 0
            self.donchian_low = 0

        close = bar.close_price

        # Position management (always run, even with pending orders)
        if self.pos != 0:
            self.hold_bars += 1
            self._update_extremes(bar)

            stop_price = self._stop_price()
            hit = (self.pos > 0 and bar.low_price <= stop_price) or (self.pos < 0 and bar.high_price >= stop_price)
            if hit:
                self._exit_position(bar, "stop")
            elif self.hold_bars >= self.max_hold:
                self._exit_position(bar, "max_hold")
            return

        # Entry signals (only when no pending orders and flat)
        if self.active_orders:
            return

        # Entry signals
        if close <= 0 or self.atr_value <= 0:
            return

        long_breakout = close > self.donchian_high and self.donchian_high > 0
        short_breakout = close < self.donchian_low and self.donchian_low > 0

        if long_breakout:
            self._enter_short(bar)
        elif short_breakout:
            self._enter_long(bar)

    # --- Order helpers ---

    def _enter_long(self, bar: BarData) -> None:
        price = self._buy_price(bar.close_price)
        size = self._calc_size(price)
        if size <= 0:
            return
        ids = self.buy(price, size)
        if ids:
            self.active_orders.update(ids)

    def _enter_short(self, bar: BarData) -> None:
        price = self._sell_price(bar.close_price)
        size = self._calc_size(price)
        if size <= 0:
            return
        ids = self.short(price, size)
        if ids:
            self.active_orders.update(ids)

    def _calc_size(self, price: float) -> int:
        """Calculate contract count from notional target and contract value."""
        if price <= 0 or self.notional_per_trade <= 0:
            return 0
        # Contract multiplier from exchange (e.g. 0.01 for BTC perpetual)
        multiplier = self.get_size()
        contract_value = price * multiplier
        if contract_value <= 0:
            return 1
        size = round(self.notional_per_trade / contract_value)
        return max(1, min(size, 1000))  # min 1, max 1000 contracts

    def _exit_position(self, bar: BarData, reason: str) -> None:
        if self.pos > 0:
            price = self._sell_price(bar.close_price)
            ids = self.sell(price, abs(self.pos))
        else:
            price = self._buy_price(bar.close_price)
            ids = self.cover(price, abs(self.pos))
        if ids:
            self.active_orders.update(ids)

    def _stop_price(self) -> float:
        if self.pos > 0:
            return self.entry_price - self.atr_stop * self.atr_value
        return self.entry_price + self.atr_stop * self.atr_value

    def _buy_price(self, base: float) -> float:
        return base + self.get_pricetick() * self.price_offset

    def _sell_price(self, base: float) -> float:
        return base - self.get_pricetick() * self.price_offset

    def _update_extremes(self, bar: BarData) -> None:
        if self.pos > 0 and bar.high_price > self.highest_since_entry:
            self.highest_since_entry = bar.high_price
        elif self.pos < 0 and bar.low_price < self.lowest_since_entry:
            self.lowest_since_entry = bar.low_price

    def _compute_atr(self) -> float:
        if len(self.am_4h.close) < self.atr_window + 1:
            return 0.0
        tr_sum = 0.0
        for i in range(-self.atr_window, 0):
            tr = max(
                self.am_4h.high[i] - self.am_4h.low[i],
                abs(self.am_4h.high[i] - self.am_4h.close[i - 1]),
                abs(self.am_4h.low[i] - self.am_4h.close[i - 1]),
            )
            tr_sum += tr
        return tr_sum / self.atr_window

    # --- Callbacks ---

    def on_order(self, order: OrderData) -> None:
        if order.is_active():
            self.active_orders.add(order.vt_orderid)
        else:
            self.active_orders.discard(order.vt_orderid)

    def on_trade(self, trade: TradeData) -> None:
        if trade.offset == Offset.OPEN:
            self.entry_price = trade.price
            self.highest_since_entry = trade.price
            self.lowest_since_entry = trade.price
            self.hold_bars = 0
        elif self.pos == 0:
            self.entry_price = 0.0
            self.highest_since_entry = 0.0
            self.lowest_since_entry = 0.0
            self.hold_bars = 0
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass
