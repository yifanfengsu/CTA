"""Adaptive multi-horizon CTA v1 for BTCUSDT perpetual on OKX."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

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


SUPPORTED_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"
INSTRUMENT_FILE: Path = (
    Path(__file__).resolve().parents[1] / "config" / "instruments" / "btcusdt_swap_okx.json"
)


class OkxAdaptiveMhfStrategy(CtaTemplate):
    """1m execution + 5m regime filter CTA for BTC perpetual on OKX."""

    author: str = "Codex"

    capital_per_strategy: float = 5_000.0
    risk_per_trade: float = 0.001
    fixed_size: float = 0.0
    max_leverage: float = 1.0
    max_notional_ratio: float = 1.0
    max_volume: float = 0.0
    min_stop_pct: float = 0.0015
    max_trades_per_day: int = 20
    daily_loss_limit_pct: float = 0.02
    disable_after_daily_loss: bool = True
    disable_after_bankrupt_guard: bool = True
    min_atr_pct_for_entry: float = 0.0
    max_atr_pct_for_entry: float = 0.0
    require_regime_persistence_bars: int = 0
    min_breakout_atr: float = 0.0
    min_bars_between_entries: int = 0
    fast_window: int = 12
    slow_window: int = 48
    breakout_window: int = 20
    atr_window: int = 14
    rsi_window: int = 14
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    vol_floor: float = 0.0010
    vol_ceiling: float = 0.02
    stop_atr: float = 1.2
    trail_atr: float = 1.8
    take_profit_atr: float = 2.4
    max_hold_bars: int = 30
    cooldown_bars: int = 3
    payup_ticks: int = 2
    init_days: int = 10
    contract_size: float = 0.0
    min_volume: float = 0.0
    pricetick: float = 0.0

    regime: str = "neutral"
    entry_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    hold_bars: int = 0
    cooldown_left: int = 0
    last_signal_ts: str = ""
    atr_1m_value: float = 0.0
    atr_5m_value: float = 0.0
    rsi_1m_value: float = 0.0
    fast_ema_5m: float = 0.0
    slow_ema_5m: float = 0.0
    breakout_high: float = 0.0
    breakout_low: float = 0.0
    vol_ratio_5m: float = 0.0
    daily_trade_count: int = 0
    daily_realized_pnl: float = 0.0
    estimated_realized_pnl_total: float = 0.0
    estimated_equity: float = 5_000.0
    daily_loss_limit_triggered: bool = False
    bankrupt_guard_triggered: bool = False
    current_trading_date: str = ""
    regime_persistence_bars: int = 0
    bar_index_1m: int = 0
    last_entry_bar_index: int | None = None

    parameters: list[str] = [
        "capital_per_strategy",
        "risk_per_trade",
        "fixed_size",
        "max_leverage",
        "max_notional_ratio",
        "max_volume",
        "min_stop_pct",
        "max_trades_per_day",
        "daily_loss_limit_pct",
        "disable_after_daily_loss",
        "disable_after_bankrupt_guard",
        "min_atr_pct_for_entry",
        "max_atr_pct_for_entry",
        "require_regime_persistence_bars",
        "min_breakout_atr",
        "min_bars_between_entries",
        "fast_window",
        "slow_window",
        "breakout_window",
        "atr_window",
        "rsi_window",
        "rsi_long",
        "rsi_short",
        "vol_floor",
        "vol_ceiling",
        "stop_atr",
        "trail_atr",
        "take_profit_atr",
        "max_hold_bars",
        "cooldown_bars",
        "payup_ticks",
        "init_days",
        "contract_size",
        "min_volume",
        "pricetick",
    ]

    variables: list[str] = [
        "regime",
        "entry_price",
        "highest_since_entry",
        "lowest_since_entry",
        "hold_bars",
        "cooldown_left",
        "last_signal_ts",
        "atr_1m_value",
        "atr_5m_value",
        "rsi_1m_value",
        "fast_ema_5m",
        "slow_ema_5m",
        "breakout_high",
        "breakout_low",
        "vol_ratio_5m",
        "daily_trade_count",
        "daily_realized_pnl",
        "estimated_realized_pnl_total",
        "estimated_equity",
        "daily_loss_limit_triggered",
        "bankrupt_guard_triggered",
        "current_trading_date",
        "regime_persistence_bars",
        "bar_index_1m",
        "last_entry_bar_index",
    ]

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ) -> None:
        """Initialize runtime state and indicator containers."""

        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg_1m: BarGenerator = BarGenerator(self.on_bar)
        self.bg_5m: BarGenerator = BarGenerator(
            self.on_bar,
            window=5,
            on_window_bar=self.on_5m_bar,
            interval=Interval.MINUTE,
        )
        self.am_1m: ArrayManager = ArrayManager(size=max(200, self.breakout_window + 50))
        self.am_5m: ArrayManager = ArrayManager(size=max(200, self.slow_window + 50))

        self.active_orderids: set[str] = set()
        self.instrument_cache: dict[str, Any] = {}
        self.instrument_cache_loaded: bool = False
        self.symbol_supported: bool = self.vt_symbol == SUPPORTED_VT_SYMBOL
        self.last_bar_dt: datetime | None = None
        self.last_5m_bar_dt: datetime | None = None
        self.contract_specs_logged: bool = False
        self.estimated_equity = float(self.capital_per_strategy)
        self.entry_filter_log_counts: dict[str, int] = {}

    def on_init(self) -> None:
        """Initialize strategy and warm up indicators from local database."""

        self.write_log("策略初始化")

        if not self.symbol_supported:
            self.write_log(
                f"当前版本仅支持 {SUPPORTED_VT_SYMBOL}，收到 {self.vt_symbol}，策略将保持空转"
            )

        try:
            self.refresh_contract_specifications(log_changes=True)
            self.reset_trade_state()
            self.load_bar(
                days=self.init_days,
                interval=Interval.MINUTE,
                callback=self.on_bar,
                use_database=True,
            )
            self.write_log("历史K线预热完成")
        except Exception as exc:
            self.write_log(f"历史K线预热失败: {exc!r}")

        self.put_event()

    def on_start(self) -> None:
        """Handle strategy start callback."""

        try:
            self.refresh_contract_specifications(log_changes=True)
            self.write_log("策略启动")
            self.put_event()
        except Exception as exc:
            self.write_log(f"on_start异常: {exc!r}")

    def on_stop(self) -> None:
        """Handle strategy stop callback."""

        try:
            self.write_log("策略停止")
            self.cancel_pending_orders()
            self.put_event()
        except Exception as exc:
            self.write_log(f"on_stop异常: {exc!r}")

    def on_tick(self, tick: TickData) -> None:
        """Aggregate live ticks into 1m bars."""

        try:
            if not tick.last_price:
                return

            self.bg_1m.update_tick(tick)
        except Exception as exc:
            self.write_log(f"on_tick异常: {exc!r}")

    def on_bar(self, bar: BarData) -> None:
        """Run 1m execution logic and feed the 5m regime filter."""

        try:
            self.last_bar_dt = bar.datetime
            self.bar_index_1m += 1
            self.sync_daily_risk_state(bar.datetime.date())
            self.refresh_contract_specifications(log_changes=False)
            self.update_bankrupt_guard()

            self.am_1m.update_bar(bar)
            self.bg_5m.update_bar(bar)

            if self.cooldown_left > 0:
                self.cooldown_left -= 1

            if not self.am_1m.inited:
                self.put_event()
                return

            self.atr_1m_value = self.calculate_atr(self.am_1m, self.atr_window)
            self.rsi_1m_value = self.calculate_rsi(self.am_1m, self.rsi_window)
            self.breakout_high, self.breakout_low = self.calculate_donchian(
                self.am_1m,
                self.breakout_window,
                include_current=False,
            )

            if not self.trading or not self.symbol_supported:
                self.put_event()
                return

            if self.active_orderids:
                self.write_log(
                    f"检测到未完成委托 {sorted(self.active_orderids)}，本bar先撤单以避免重复下单"
                )
                self.cancel_pending_orders()
                self.put_event()
                return

            if self.pos != 0:
                self.manage_position(bar)
                self.put_event()
                return

            signal: int = self.generate_entry_signal(bar)
            if signal == 0:
                self.put_event()
                return

            volume: float = self.compute_order_volume(bar.close_price)
            if volume <= 0:
                self.put_event()
                return

            self.last_signal_ts = bar.datetime.isoformat()
            if signal > 0:
                order_price: float = self.get_marketable_price(bar.close_price, is_buy=True)
                vt_orderids: list[str] = self.buy(order_price, volume)
                action: str = "开多"
            else:
                order_price = self.get_marketable_price(bar.close_price, is_buy=False)
                vt_orderids = self.short(order_price, volume)
                action = "开空"

            if vt_orderids:
                self.active_orderids.update(vt_orderids)
                self.write_log(
                    f"{action}委托已发送 price={order_price:.8f}, volume={volume:.8f}, "
                    f"regime={self.regime}, hh={self.breakout_high:.8f}, ll={self.breakout_low:.8f}, "
                    f"rsi1={self.rsi_1m_value:.2f}, atr1={self.atr_1m_value:.8f}"
                )
            else:
                self.write_log(
                    f"{action}信号触发但委托发送失败 price={order_price:.8f}, volume={volume:.8f}"
                )

            self.put_event()
        except Exception as exc:
            self.write_log(f"on_bar异常: {exc!r}")

    def on_5m_bar(self, bar: BarData) -> None:
        """Update 5m regime state from aggregated 1m bars."""

        try:
            self.last_5m_bar_dt = bar.datetime
            self.am_5m.update_bar(bar)

            if not self.am_5m.inited:
                return

            self.update_regime()
            self.put_event()
        except Exception as exc:
            self.write_log(f"on_5m_bar异常: {exc!r}")

    def on_order(self, order: OrderData) -> None:
        """Track active orders and log order state changes."""

        try:
            if order.is_active():
                self.active_orderids.add(order.vt_orderid)
            else:
                self.active_orderids.discard(order.vt_orderid)

            self.write_log(
                f"委托更新 vt_orderid={order.vt_orderid}, status={order.status.value}, "
                f"direction={order.direction.value if order.direction else ''}, "
                f"offset={order.offset.value}, price={order.price:.8f}, "
                f"volume={order.volume:.8f}, traded={order.traded:.8f}"
            )
            self.put_event()
        except Exception as exc:
            self.write_log(f"on_order异常: {exc!r}")

    def on_trade(self, trade: TradeData) -> None:
        """Maintain position state after fills."""

        try:
            self.last_signal_ts = trade.datetime.isoformat() if trade.datetime else ""
            if trade.datetime:
                self.sync_daily_risk_state(trade.datetime.date())
            self.write_log(
                f"成交回报 vt_tradeid={trade.vt_tradeid}, direction={trade.direction.value if trade.direction else ''}, "
                f"offset={trade.offset.value}, price={trade.price:.8f}, volume={trade.volume:.8f}, pos={self.pos:.8f}"
            )

            if trade.offset == Offset.OPEN:
                self.register_open_trade(trade)
            else:
                self.register_close_trade(trade)

            if self.pos == 0:
                self.write_log("持仓归零，进入冷却期")
                self.reset_trade_state()
                self.cooldown_left = self.cooldown_bars
                self.write_log(f"cooldown_left 已重置为 {self.cooldown_left}")
                self.put_event()
                return

            if trade.offset == Offset.OPEN:
                self.update_entry_state_from_trade(trade)
            else:
                self.write_log(
                    f"平仓成交后剩余持仓 pos={self.pos:.8f}，保留原开仓状态并继续管理"
                )

            self.put_event()
        except Exception as exc:
            self.write_log(f"on_trade异常: {exc!r}")

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """Log stop-order state changes."""

        try:
            self.write_log(
                f"停止单更新 stop_orderid={stop_order.stop_orderid}, status={stop_order.status.value}, "
                f"price={stop_order.price:.8f}, volume={stop_order.volume:.8f}"
            )
            self.put_event()
        except Exception as exc:
            self.write_log(f"on_stop_order异常: {exc!r}")

    def reset_trade_state(self) -> None:
        """Reset trade lifecycle state after flat or during initialization."""

        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 0.0
        self.hold_bars = 0
        self.write_log("重置交易状态 entry/highest/lowest/hold_bars -> 0")

    def log_entry_filter(self, reason: str, message: str) -> None:
        """Log entry filter decisions without flooding the log output."""

        count = self.entry_filter_log_counts.get(reason, 0) + 1
        self.entry_filter_log_counts[reason] = count
        if count <= 3 or count % 100 == 0:
            self.write_log(f"{message} [count={count}]")

    def sync_daily_risk_state(self, trading_date: date) -> None:
        """Reset daily counters when the bar/trade date changes."""

        trading_date_str = trading_date.isoformat()
        if self.current_trading_date == trading_date_str:
            return

        previous_date = self.current_trading_date
        self.current_trading_date = trading_date_str
        self.daily_trade_count = 0
        self.daily_realized_pnl = 0.0
        self.daily_loss_limit_triggered = False
        self.write_log(
            f"重置日内风控状态 date={self.current_trading_date}, previous_date={previous_date or 'N/A'}"
        )

    def update_regime(self) -> None:
        """Update 5m long/short/neutral regime state."""

        if not self.am_5m.inited:
            self.regime = "neutral"
            return

        self.fast_ema_5m = self.calculate_ema(self.am_5m, self.fast_window)
        self.slow_ema_5m = self.calculate_ema(self.am_5m, self.slow_window)
        self.atr_5m_value = self.calculate_atr(self.am_5m, self.atr_window)

        close_5m: float = float(self.am_5m.close[-1])
        if close_5m <= 0:
            self.vol_ratio_5m = 0.0
        else:
            self.vol_ratio_5m = self.atr_5m_value / close_5m

        previous_regime: str = self.regime
        if (
            self.fast_ema_5m > self.slow_ema_5m
            and self.vol_floor <= self.vol_ratio_5m <= self.vol_ceiling
        ):
            self.regime = "long"
        elif (
            self.fast_ema_5m < self.slow_ema_5m
            and self.vol_floor <= self.vol_ratio_5m <= self.vol_ceiling
        ):
            self.regime = "short"
        else:
            self.regime = "neutral"

        if self.regime == "neutral":
            self.regime_persistence_bars = 0
        elif previous_regime == self.regime:
            self.regime_persistence_bars += 1
        else:
            self.regime_persistence_bars = 1

        if previous_regime != self.regime:
            self.write_log(
                f"5m regime切换 {previous_regime} -> {self.regime}, "
                f"fast={self.fast_ema_5m:.8f}, slow={self.slow_ema_5m:.8f}, "
                f"atr5={self.atr_5m_value:.8f}, vol_ratio={self.vol_ratio_5m:.6f}, "
                f"persistence={self.regime_persistence_bars}"
            )

    def generate_entry_signal(self, bar: BarData) -> int:
        """Generate 1m breakout entry signal with 5m regime filter."""

        if not self.am_5m.inited:
            return 0

        if self.breakout_high <= 0 or self.breakout_low <= 0:
            return 0

        if self.disable_after_bankrupt_guard and self.bankrupt_guard_triggered:
            self.write_log("放弃信号: bankrupt_guard_triggered=True")
            return 0

        if self.disable_after_daily_loss and self.daily_loss_limit_triggered:
            self.write_log(
                f"放弃信号: daily_loss_limit_triggered=True, daily_realized_pnl={self.daily_realized_pnl:.8f}"
            )
            return 0

        if self.max_trades_per_day > 0 and self.daily_trade_count >= self.max_trades_per_day:
            self.write_log(
                f"放弃信号: daily_trade_count={self.daily_trade_count} 已达到 "
                f"max_trades_per_day={self.max_trades_per_day}"
            )
            return 0

        if (
            self.min_bars_between_entries > 0
            and self.last_entry_bar_index is not None
            and self.bar_index_1m - self.last_entry_bar_index < self.min_bars_between_entries
        ):
            self.log_entry_filter(
                "min_bars_between_entries",
                "放弃信号: 与上次开仓间隔不足 "
                f"bars_since_last_entry={self.bar_index_1m - self.last_entry_bar_index}, "
                f"min_bars_between_entries={self.min_bars_between_entries}",
            )
            return 0

        atr_pct = self.get_entry_atr_pct(bar.close_price)
        if self.min_atr_pct_for_entry > 0 and atr_pct < self.min_atr_pct_for_entry:
            self.log_entry_filter(
                "min_atr_pct_for_entry",
                f"放弃信号: atr_pct={atr_pct:.6f} < min_atr_pct_for_entry={self.min_atr_pct_for_entry:.6f}",
            )
            return 0

        if self.max_atr_pct_for_entry > 0 and atr_pct > self.max_atr_pct_for_entry:
            self.log_entry_filter(
                "max_atr_pct_for_entry",
                f"放弃信号: atr_pct={atr_pct:.6f} > max_atr_pct_for_entry={self.max_atr_pct_for_entry:.6f}",
            )
            return 0

        long_breakout: bool = bar.close_price > self.breakout_high
        short_breakout: bool = bar.close_price < self.breakout_low
        if not long_breakout and not short_breakout:
            return 0

        if self.cooldown_left > 0:
            self.write_log(
                f"放弃信号: cooldown_left={self.cooldown_left}, close={bar.close_price:.8f}, "
                f"hh={self.breakout_high:.8f}, ll={self.breakout_low:.8f}"
            )
            return 0

        if long_breakout:
            if self.regime != "long":
                self.write_log(
                    f"放弃多头信号: close={bar.close_price:.8f} 已突破 hh={self.breakout_high:.8f}，"
                    f"但 regime={self.regime}"
                )
                return 0
            if (
                self.require_regime_persistence_bars > 0
                and self.regime_persistence_bars < self.require_regime_persistence_bars
            ):
                self.log_entry_filter(
                    "require_regime_persistence_bars_long",
                    "放弃多头信号: regime persistence 不足 "
                    f"{self.regime_persistence_bars} < {self.require_regime_persistence_bars}",
                )
                return 0
            if self.rsi_1m_value < self.rsi_long:
                self.write_log(
                    f"放弃多头信号: rsi1={self.rsi_1m_value:.2f} < rsi_long={self.rsi_long:.2f}"
                )
                return 0
            if self.min_breakout_atr > 0:
                breakout_distance = bar.close_price - self.breakout_high
                required_distance = max(self.atr_1m_value, 0.0) * self.min_breakout_atr
                if breakout_distance < required_distance:
                    self.log_entry_filter(
                        "min_breakout_atr_long",
                        "放弃多头信号: breakout_distance 不足 "
                        f"{breakout_distance:.8f} < {required_distance:.8f}",
                    )
                    return 0

            self.write_log(
                f"生成多头信号 close={bar.close_price:.8f}, hh={self.breakout_high:.8f}, "
                f"rsi1={self.rsi_1m_value:.2f}, regime={self.regime}"
            )
            return 1

        if short_breakout:
            if self.regime != "short":
                self.write_log(
                    f"放弃空头信号: close={bar.close_price:.8f} 已跌破 ll={self.breakout_low:.8f}，"
                    f"但 regime={self.regime}"
                )
                return 0
            if (
                self.require_regime_persistence_bars > 0
                and self.regime_persistence_bars < self.require_regime_persistence_bars
            ):
                self.log_entry_filter(
                    "require_regime_persistence_bars_short",
                    "放弃空头信号: regime persistence 不足 "
                    f"{self.regime_persistence_bars} < {self.require_regime_persistence_bars}",
                )
                return 0
            if self.rsi_1m_value > self.rsi_short:
                self.write_log(
                    f"放弃空头信号: rsi1={self.rsi_1m_value:.2f} > rsi_short={self.rsi_short:.2f}"
                )
                return 0
            if self.min_breakout_atr > 0:
                breakout_distance = self.breakout_low - bar.close_price
                required_distance = max(self.atr_1m_value, 0.0) * self.min_breakout_atr
                if breakout_distance < required_distance:
                    self.log_entry_filter(
                        "min_breakout_atr_short",
                        "放弃空头信号: breakout_distance 不足 "
                        f"{breakout_distance:.8f} < {required_distance:.8f}",
                    )
                    return 0

            self.write_log(
                f"生成空头信号 close={bar.close_price:.8f}, ll={self.breakout_low:.8f}, "
                f"rsi1={self.rsi_1m_value:.2f}, regime={self.regime}"
            )
            return -1

        return 0

    def manage_position(self, bar: BarData) -> None:
        """Manage exits for existing long/short positions."""

        if self.pos == 0:
            return

        self.hold_bars += 1
        if self.highest_since_entry <= 0:
            self.highest_since_entry = bar.high_price
        else:
            self.highest_since_entry = max(self.highest_since_entry, bar.high_price)

        if self.lowest_since_entry <= 0:
            self.lowest_since_entry = bar.low_price
        else:
            self.lowest_since_entry = min(self.lowest_since_entry, bar.low_price)

        atr1: float = max(self.atr_1m_value, self.get_effective_pricetick())
        stop_distance: float = self.compute_stop_distance(bar.close_price)
        exit_reason: str = ""
        order_price: float = 0.0
        volume: float = abs(self.pos)

        if self.pos > 0:
            initial_stop: float = self.entry_price - stop_distance
            trailing_stop: float = self.highest_since_entry - self.trail_atr * atr1
            protective_stop: float = max(initial_stop, trailing_stop)
            take_profit: float = self.entry_price + self.take_profit_atr * atr1

            if self.regime == "short":
                exit_reason = "regime_flip_short"
            elif self.hold_bars >= self.max_hold_bars:
                exit_reason = "time_stop"
            elif bar.close_price <= protective_stop:
                exit_reason = "long_stop_loss"
            elif bar.close_price >= take_profit:
                exit_reason = "long_take_profit"

            if exit_reason:
                order_price = self.get_marketable_price(bar.close_price, is_buy=False)
                vt_orderids: list[str] = self.sell(order_price, volume)
                if vt_orderids:
                    self.active_orderids.update(vt_orderids)
                    self.write_log(
                        f"触发平多 {exit_reason}, price={order_price:.8f}, volume={volume:.8f}, "
                        f"entry={self.entry_price:.8f}, init_stop={initial_stop:.8f}, "
                        f"trail_stop={trailing_stop:.8f}, take_profit={take_profit:.8f}, "
                        f"hold_bars={self.hold_bars}"
                    )
                else:
                    self.write_log(f"平多委托发送失败 reason={exit_reason}, volume={volume:.8f}")

        elif self.pos < 0:
            initial_stop = self.entry_price + stop_distance
            trailing_stop = self.lowest_since_entry + self.trail_atr * atr1
            protective_stop = min(initial_stop, trailing_stop)
            take_profit = self.entry_price - self.take_profit_atr * atr1

            if self.regime == "long":
                exit_reason = "regime_flip_long"
            elif self.hold_bars >= self.max_hold_bars:
                exit_reason = "time_stop"
            elif bar.close_price >= protective_stop:
                exit_reason = "short_stop_loss"
            elif bar.close_price <= take_profit:
                exit_reason = "short_take_profit"

            if exit_reason:
                order_price = self.get_marketable_price(bar.close_price, is_buy=True)
                vt_orderids = self.cover(order_price, volume)
                if vt_orderids:
                    self.active_orderids.update(vt_orderids)
                    self.write_log(
                        f"触发平空 {exit_reason}, price={order_price:.8f}, volume={volume:.8f}, "
                        f"entry={self.entry_price:.8f}, init_stop={initial_stop:.8f}, "
                        f"trail_stop={trailing_stop:.8f}, take_profit={take_profit:.8f}, "
                        f"hold_bars={self.hold_bars}"
                    )
                else:
                    self.write_log(f"平空委托发送失败 reason={exit_reason}, volume={volume:.8f}")

    def compute_order_volume(self, close_price: float) -> float:
        """Compute strategy order size using fixed-size or ATR risk sizing."""

        effective_min_volume: float = self.get_effective_min_volume()
        effective_contract_size: float = self.get_effective_contract_size()

        if effective_min_volume <= 0:
            self.write_log("放弃信号: min_volume 未就绪")
            return 0.0

        if close_price <= 0:
            self.write_log(f"放弃信号: close_price 无效 {close_price:.8f}")
            return 0.0

        if self.fixed_size > 0:
            return self.apply_volume_hard_caps(self.fixed_size, close_price, "fixed_size")

        if effective_contract_size <= 0:
            self.write_log("放弃信号: contract_size 未就绪，无法进行风控仓位计算")
            return 0.0

        risk_cash: float = self.capital_per_strategy * self.risk_per_trade
        stop_distance: float = self.compute_stop_distance(close_price)
        if risk_cash <= 0 or stop_distance <= 0:
            self.write_log(
                f"放弃信号: risk_cash={risk_cash:.8f}, stop_distance={stop_distance:.8f}"
            )
            return 0.0

        raw_volume: float = risk_cash / (stop_distance * effective_contract_size)
        return self.apply_volume_hard_caps(raw_volume, close_price, "risk_sizing")

    def compute_stop_distance(self, close_price: float) -> float:
        """Compute a conservative stop distance floor from ATR, percentage, and tick size."""

        atr_component = max(float(self.atr_1m_value), 0.0) * max(float(self.stop_atr), 0.0)
        pct_component = max(float(close_price), 0.0) * max(float(self.min_stop_pct), 0.0)
        tick_component = self.get_effective_pricetick() * 3.0
        stop_distance = max(atr_component, pct_component, tick_component)

        if stop_distance <= 0:
            self.write_log(
                f"止损距离无效 atr_component={atr_component:.8f}, "
                f"pct_component={pct_component:.8f}, tick_component={tick_component:.8f}"
            )
        return stop_distance

    def get_entry_atr_pct(self, close_price: float) -> float:
        """Return 1m ATR as a fraction of current close price."""

        if close_price <= 0:
            return 0.0
        return max(float(self.atr_1m_value), 0.0) / float(close_price)

    def apply_volume_hard_caps(
        self,
        requested_volume: float,
        close_price: float,
        source: str,
    ) -> float:
        """Apply min-volume rounding plus max leverage/notional/absolute volume caps."""

        effective_min_volume: float = self.get_effective_min_volume()
        effective_contract_size: float = self.get_effective_contract_size()
        capital_limit: float = max(float(self.capital_per_strategy), 0.0)

        if requested_volume <= 0 or not isfinite(requested_volume):
            self.write_log(
                f"放弃信号: requested_volume 无效 source={source}, requested_volume={requested_volume}"
            )
            return 0.0

        if close_price <= 0 or effective_contract_size <= 0 or capital_limit <= 0:
            self.write_log(
                f"放弃信号: 无法应用仓位上限 source={source}, close_price={close_price:.8f}, "
                f"contract_size={effective_contract_size:.8f}, capital={capital_limit:.8f}"
            )
            return 0.0

        notional_per_contract: float = close_price * effective_contract_size
        if notional_per_contract <= 0 or not isfinite(notional_per_contract):
            self.write_log(
                f"放弃信号: notional_per_contract 无效 source={source}, "
                f"notional_per_contract={notional_per_contract}"
            )
            return 0.0

        cap_candidates: list[float] = []
        cap_details: list[str] = []

        if self.max_leverage > 0:
            max_volume_by_leverage = capital_limit * self.max_leverage / notional_per_contract
            cap_candidates.append(max_volume_by_leverage)
            cap_details.append(f"max_volume_by_leverage={max_volume_by_leverage:.8f}")

        if self.max_notional_ratio > 0:
            max_volume_by_notional = capital_limit * self.max_notional_ratio / notional_per_contract
            cap_candidates.append(max_volume_by_notional)
            cap_details.append(f"max_volume_by_notional_ratio={max_volume_by_notional:.8f}")

        if self.max_volume > 0:
            cap_candidates.append(float(self.max_volume))
            cap_details.append(f"max_volume={float(self.max_volume):.8f}")

        clipped_volume = float(requested_volume)
        if cap_candidates:
            positive_caps = [cap for cap in cap_candidates if cap > 0 and isfinite(cap)]
            if not positive_caps:
                self.write_log(
                    f"放弃信号: 风控上限无有效正值 source={source}, "
                    f"requested_volume={requested_volume:.8f}, {', '.join(cap_details)}"
                )
                return 0.0

            max_allowed_volume = min(positive_caps)
            if requested_volume > max_allowed_volume:
                self.write_log(
                    f"仓位被风控裁剪 source={source}, requested_volume={requested_volume:.8f}, "
                    f"clipped_volume={max_allowed_volume:.8f}, notional_per_contract={notional_per_contract:.8f}, "
                    f"{', '.join(cap_details)}"
                )
            clipped_volume = min(requested_volume, max_allowed_volume)

        rounded_volume: float = self.round_volume(clipped_volume)
        if rounded_volume < effective_min_volume:
            self.write_log(
                f"放弃信号: 最终下单量低于 min_volume source={source}, "
                f"requested_volume={requested_volume:.8f}, clipped_volume={clipped_volume:.8f}, "
                f"rounded_volume={rounded_volume:.8f}, min_volume={effective_min_volume:.8f}"
            )
            return 0.0

        return rounded_volume

    def register_open_trade(self, trade: TradeData) -> None:
        """Track daily open-trade count after an entry fill."""

        self.daily_trade_count += 1
        self.last_entry_bar_index = self.bar_index_1m
        self.write_log(
            f"记录开仓成交 daily_trade_count={self.daily_trade_count}, "
            f"max_trades_per_day={self.max_trades_per_day}, trade_volume={trade.volume:.8f}, "
            f"last_entry_bar_index={self.last_entry_bar_index}"
        )

    def register_close_trade(self, trade: TradeData) -> None:
        """Update realized pnl and protective disable flags after an exit fill."""

        realized_pnl = self.estimate_realized_trade_pnl(trade)
        self.daily_realized_pnl += realized_pnl
        self.estimated_realized_pnl_total += realized_pnl
        self.estimated_equity = self.capital_per_strategy + self.estimated_realized_pnl_total

        self.write_log(
            f"记录平仓成交 realized_pnl={realized_pnl:.8f}, daily_realized_pnl={self.daily_realized_pnl:.8f}, "
            f"estimated_realized_pnl_total={self.estimated_realized_pnl_total:.8f}, "
            f"estimated_equity={self.estimated_equity:.8f}"
        )

        self.update_daily_loss_guard()
        self.update_bankrupt_guard()

    def estimate_realized_trade_pnl(self, trade: TradeData) -> float:
        """Estimate realized pnl for one exit fill using strategy entry_price."""

        contract_size = self.get_effective_contract_size()
        if contract_size <= 0 or self.entry_price <= 0 or trade.volume <= 0:
            return 0.0

        if trade.direction == Direction.SHORT:
            return (trade.price - self.entry_price) * trade.volume * contract_size
        if trade.direction == Direction.LONG:
            return (self.entry_price - trade.price) * trade.volume * contract_size
        return 0.0

    def update_daily_loss_guard(self) -> None:
        """Disable new entries for the day after crossing the daily realized loss limit."""

        if not self.disable_after_daily_loss:
            return

        if self.daily_loss_limit_pct <= 0 or self.capital_per_strategy <= 0:
            return

        daily_loss_limit_cash = self.capital_per_strategy * self.daily_loss_limit_pct
        if self.daily_realized_pnl <= -daily_loss_limit_cash and not self.daily_loss_limit_triggered:
            self.daily_loss_limit_triggered = True
            self.write_log(
                f"触发日内亏损停机 daily_realized_pnl={self.daily_realized_pnl:.8f}, "
                f"daily_loss_limit_cash={daily_loss_limit_cash:.8f}"
            )

    def update_bankrupt_guard(self) -> None:
        """Disable new entries permanently when internal equity estimate becomes abnormal."""

        if not self.disable_after_bankrupt_guard:
            return

        if self.bankrupt_guard_triggered:
            return

        if not isfinite(self.estimated_equity) or self.estimated_equity <= 0:
            self.bankrupt_guard_triggered = True
            self.write_log(
                f"触发权益异常停机 estimated_equity={self.estimated_equity}, "
                f"estimated_realized_pnl_total={self.estimated_realized_pnl_total:.8f}"
            )

    def round_volume(self, volume: float) -> float:
        """Round volume down to the nearest valid min-volume step."""

        step: float = self.get_effective_min_volume()
        if volume <= 0 or step <= 0:
            return 0.0

        volume_decimal = Decimal(str(volume))
        step_decimal = Decimal(str(step))
        units_decimal = (volume_decimal / step_decimal).to_integral_value(rounding=ROUND_FLOOR)
        rounded: Decimal = units_decimal * step_decimal
        return float(rounded)

    def cancel_pending_orders(self) -> None:
        """Cancel all active strategy orders to avoid duplicate execution."""

        if not self.active_orderids:
            return

        self.write_log(f"撤销挂单数量: {len(self.active_orderids)}")
        self.cancel_all()

    def update_entry_state_from_trade(self, trade: TradeData) -> None:
        """Update entry price and excursion state for a newly opened position."""

        if self.pos > 0:
            previous_pos: float = max(self.pos - trade.volume, 0.0)
            if previous_pos > 0 and self.entry_price > 0:
                self.entry_price = (
                    self.entry_price * previous_pos + trade.price * trade.volume
                ) / self.pos
            else:
                self.entry_price = trade.price

            self.highest_since_entry = max(self.highest_since_entry, trade.price)
            self.lowest_since_entry = trade.price if self.lowest_since_entry <= 0 else min(
                self.lowest_since_entry,
                trade.price,
            )
            self.hold_bars = 0
            self.write_log(
                f"开多成交后更新状态 entry_price={self.entry_price:.8f}, pos={self.pos:.8f}"
            )
            return

        if self.pos < 0:
            previous_abs_pos: float = max(abs(self.pos + trade.volume), 0.0)
            current_abs_pos: float = abs(self.pos)
            if previous_abs_pos > 0 and self.entry_price > 0 and current_abs_pos > 0:
                self.entry_price = (
                    self.entry_price * previous_abs_pos + trade.price * trade.volume
                ) / current_abs_pos
            else:
                self.entry_price = trade.price

            self.lowest_since_entry = (
                trade.price if self.lowest_since_entry <= 0 else min(self.lowest_since_entry, trade.price)
            )
            self.highest_since_entry = max(self.highest_since_entry, trade.price)
            self.hold_bars = 0
            self.write_log(
                f"开空成交后更新状态 entry_price={self.entry_price:.8f}, pos={self.pos:.8f}"
            )

    def refresh_contract_specifications(self, log_changes: bool = False) -> None:
        """Hydrate contract_size, min_volume and pricetick from live/backtest/local metadata."""

        changes: list[str] = []
        instrument_data: dict[str, Any] = self.load_instrument_cache()

        if self.contract_size <= 0:
            engine_size: float = self.safe_get_engine_size()
            cached_size: float = self.safe_float(instrument_data.get("size"))
            resolved_size: float = engine_size if engine_size > 0 else cached_size
            if resolved_size > 0:
                self.contract_size = resolved_size
                changes.append(f"contract_size={self.contract_size}")

        if self.pricetick <= 0:
            engine_pricetick: float = self.safe_get_engine_pricetick()
            cached_pricetick: float = self.safe_float(instrument_data.get("pricetick"))
            resolved_pricetick: float = engine_pricetick if engine_pricetick > 0 else cached_pricetick
            if resolved_pricetick > 0:
                self.pricetick = resolved_pricetick
                changes.append(f"pricetick={self.pricetick}")

        if self.min_volume <= 0:
            live_contract_min_volume: float = self.safe_get_live_contract_min_volume()
            cached_min_volume: float = self.safe_float(instrument_data.get("min_volume"))
            resolved_min_volume: float = 0.0
            if live_contract_min_volume > 0:
                resolved_min_volume = live_contract_min_volume
            elif cached_min_volume > 0:
                resolved_min_volume = cached_min_volume
            else:
                resolved_min_volume = 1.0

            self.min_volume = resolved_min_volume
            changes.append(f"min_volume={self.min_volume}")

        if changes and (log_changes or not self.contract_specs_logged):
            self.write_log("合约规格就绪: " + ", ".join(changes))
            self.contract_specs_logged = True

    def load_instrument_cache(self) -> dict[str, Any]:
        """Load cached instrument metadata from local config only once."""

        if self.instrument_cache_loaded:
            return self.instrument_cache

        self.instrument_cache_loaded = True
        if not INSTRUMENT_FILE.exists():
            return self.instrument_cache

        try:
            self.instrument_cache = json.loads(INSTRUMENT_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            self.write_log(f"读取本地合约元数据失败: {exc!r}")
            self.instrument_cache = {}
        return self.instrument_cache

    def safe_get_engine_size(self) -> float:
        """Get contract size from CTA engine when available."""

        try:
            return float(self.get_size())
        except Exception:
            return 0.0

    def safe_get_engine_pricetick(self) -> float:
        """Get pricetick from CTA engine when available."""

        try:
            return float(self.get_pricetick())
        except Exception:
            return 0.0

    def safe_get_live_contract_min_volume(self) -> float:
        """Get min_volume from live MainEngine contract cache when available."""

        main_engine: Any | None = getattr(self.cta_engine, "main_engine", None)
        if main_engine is None:
            return 0.0

        try:
            contract: Any | None = main_engine.get_contract(self.vt_symbol)
        except Exception:
            return 0.0

        if contract is None:
            return 0.0

        return self.safe_float(getattr(contract, "min_volume", 0.0))

    def calculate_ema(self, am: ArrayManager, window: int) -> float:
        """Calculate EMA via ArrayManager or numpy fallback."""

        if hasattr(am, "ema"):
            try:
                value: float = float(am.ema(window))
                if not np.isnan(value):
                    return value
            except Exception:
                pass
        return self.fallback_ema(am.close, window)

    def calculate_rsi(self, am: ArrayManager, window: int) -> float:
        """Calculate RSI via ArrayManager or numpy fallback."""

        if hasattr(am, "rsi"):
            try:
                value: float = float(am.rsi(window))
                if not np.isnan(value):
                    return value
            except Exception:
                pass
        return self.fallback_rsi(am.close, window)

    def calculate_atr(self, am: ArrayManager, window: int) -> float:
        """Calculate ATR via ArrayManager or numpy fallback."""

        if hasattr(am, "atr"):
            try:
                value: float = float(am.atr(window))
                if not np.isnan(value):
                    return value
            except Exception:
                pass
        return self.fallback_atr(am.high, am.low, am.close, window)

    def calculate_donchian(
        self,
        am: ArrayManager,
        window: int,
        include_current: bool = False,
    ) -> tuple[float, float]:
        """Calculate Donchian high/low with numpy fallback."""

        highs: np.ndarray = am.high
        lows: np.ndarray = am.low

        if include_current:
            high_slice = highs[-window:]
            low_slice = lows[-window:]
        else:
            high_slice = highs[-(window + 1):-1]
            low_slice = lows[-(window + 1):-1]

        if high_slice.size == 0 or low_slice.size == 0:
            return 0.0, 0.0

        return float(np.max(high_slice)), float(np.min(low_slice))

    def fallback_ema(self, series: np.ndarray, window: int) -> float:
        """Numpy EMA fallback implementation."""

        data: np.ndarray = np.asarray(series, dtype=float)
        if data.size == 0:
            return 0.0

        alpha: float = 2.0 / (window + 1.0)
        ema_value: float = float(data[0])
        for price in data[1:]:
            ema_value = alpha * float(price) + (1.0 - alpha) * ema_value
        return ema_value

    def fallback_rsi(self, series: np.ndarray, window: int) -> float:
        """Numpy RSI fallback implementation using Wilder smoothing."""

        closes: np.ndarray = np.asarray(series, dtype=float)
        if closes.size <= window:
            return 50.0

        deltas: np.ndarray = np.diff(closes)
        gains: np.ndarray = np.clip(deltas, 0.0, None)
        losses: np.ndarray = np.clip(-deltas, 0.0, None)

        avg_gain: float = float(np.mean(gains[:window]))
        avg_loss: float = float(np.mean(losses[:window]))

        for index in range(window, gains.size):
            avg_gain = (avg_gain * (window - 1) + float(gains[index])) / window
            avg_loss = (avg_loss * (window - 1) + float(losses[index])) / window

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs: float = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def fallback_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int,
    ) -> float:
        """Numpy ATR fallback implementation using Wilder smoothing."""

        high_array: np.ndarray = np.asarray(highs, dtype=float)
        low_array: np.ndarray = np.asarray(lows, dtype=float)
        close_array: np.ndarray = np.asarray(closes, dtype=float)
        if high_array.size <= 1 or low_array.size <= 1 or close_array.size <= 1:
            return 0.0

        prev_close: np.ndarray = np.roll(close_array, 1)
        prev_close[0] = close_array[0]
        true_range: np.ndarray = np.maximum.reduce(
            [
                high_array - low_array,
                np.abs(high_array - prev_close),
                np.abs(low_array - prev_close),
            ]
        )

        if true_range.size <= window:
            return float(np.mean(true_range))

        atr_value: float = float(np.mean(true_range[:window]))
        for index in range(window, true_range.size):
            atr_value = ((window - 1) * atr_value + float(true_range[index])) / window
        return atr_value

    def get_effective_contract_size(self) -> float:
        """Return the current usable contract size."""

        return max(float(self.contract_size), 0.0)

    def get_effective_min_volume(self) -> float:
        """Return the current usable minimum volume."""

        return max(float(self.min_volume), 0.0)

    def get_effective_pricetick(self) -> float:
        """Return the current usable pricetick."""

        return max(float(self.pricetick), 0.0)

    def get_marketable_price(self, close_price: float, is_buy: bool) -> float:
        """Create a marketable limit price using payup ticks."""

        tick_size: float = self.get_effective_pricetick()
        if tick_size <= 0:
            return close_price

        adjusted_price: float = close_price + self.payup_ticks * tick_size if is_buy else close_price - self.payup_ticks * tick_size
        return self.round_price(adjusted_price, tick_size=tick_size, round_up=is_buy)

    def round_price(self, price: float, tick_size: float, round_up: bool) -> float:
        """Round price to tick size with directional control."""

        if tick_size <= 0:
            return price

        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick_size))
        ratio: Decimal = price_decimal / tick_decimal
        rounding_mode = ROUND_CEILING if round_up else ROUND_FLOOR
        steps: Decimal = ratio.to_integral_value(rounding=rounding_mode)
        return float(steps * tick_decimal)

    def safe_float(self, value: Any) -> float:
        """Convert arbitrary input into float safely."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
