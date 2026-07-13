# VSVCB-v1 Phase 1 Research

## 1. Hypothesis
VSVCB-v1 tests whether ordinary breakouts become more persistent when they occur after a low-volatility Bollinger Band Width squeeze and with breakout-bar volume expansion. It is research-only and uses next-bar-open fixed-hold benchmarks.

## 2. Data Range
- start=2023-01-01T00:00:00+08:00
- end_exclusive=2026-04-01T00:00:00+08:00
- timezone=Asia/Shanghai
- funding_data_complete=true

## 3. Symbols And Timeframes
- symbols=['BTCUSDT_SWAP_OKX.GLOBAL', 'ETHUSDT_SWAP_OKX.GLOBAL', 'SOLUSDT_SWAP_OKX.GLOBAL']
- timeframes=['15m', '30m', '1h']

## 4. Train / Validation / OOS Dates
- train=2023-01-01T00:00:00+08:00 to 2024-12-12T14:24:00+08:00
- validation=2024-12-12T14:24:00+08:00 to 2025-08-06T19:12:00+08:00
- oos=2025-08-06T19:12:00+08:00 to 2026-04-01T00:00:00+08:00

## 5. A/B/C/D/E Event Counts
- A=31649
- B=7904
- C=19213
- D=4928
- E=4928

## 6. Event Group Summary
| group | timeframe | event_count | mean_future_return_10 | reversal_rate_10 |
|---|---|---|---|---|
| A | 15m | 19455 | -0.0060 | 0.5936 |
| A | 1h | 3823 | -0.0090 | 0.5726 |
| A | 30m | 8371 | -0.0081 | 0.5813 |
| B | 15m | 4980 | -0.0032 | 0.5922 |
| B | 1h | 866 | -0.0085 | 0.5520 |
| B | 30m | 2058 | -0.0054 | 0.5850 |
| C | 15m | 11469 | -0.0079 | 0.6055 |
| C | 1h | 2470 | -0.0073 | 0.5494 |
| C | 30m | 5274 | -0.0092 | 0.5783 |
| D | 15m | 3005 | -0.0049 | 0.6130 |
| D | 1h | 596 | -0.0072 | 0.5319 |
| D | 30m | 1327 | -0.0075 | 0.5938 |
| E | 15m | 3005 | 0.0049 | 0.3857 |
| E | 1h | 596 | 0.0072 | 0.4664 |
| E | 30m | 1327 | 0.0075 | 0.4062 |

## 7. D Group 15m Trade Result
| group | timeframe | split | trade_count | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl | max_drawdown_pct |
|---|---|---|---|---|---|---|---|
| D | 15m | oos | 508 | -2496.6260 | -3512.6260 | -3511.6987 | 3.4755 |
| D | 15m | train | 1651 | -4030.6580 | -7332.6580 | -7319.9984 | 7.3641 |
| D | 15m | validation | 387 | -6083.4356 | -6857.4356 | -6857.0697 | 6.9155 |

## 8. Required Answers
1. D group 是否优于 A/B/C 中至少两个？true
2. 反向测试是否明显更差？false [{'timeframe': '15m', 'split': 'oos', 'd_trade_count': 508, 'e_trade_count': 508, 'd_no_cost_pnl': -2496.626039435, 'e_no_cost_pnl': 2496.626039435, 'd_cost_aware_pnl': -3512.626039435, 'e_cost_aware_pnl': 1480.626039435, 'd_funding_adjusted_pnl': -3511.6987233856, 'e_funding_adjusted_pnl': 1479.6987233856, 'reverse_weaker': False}]
3. no-cost 是否通过？train=false, validation=false, oos=false
4. cost-aware 是否通过？false
5. funding-adjusted 是否通过？false
6. 收益是否集中在单一 symbol？concentration_pass=false [{'group': 'D', 'timeframe': '15m', 'split': 'oos', 'trade_count': 508, 'total_funding_adjusted_pnl': -3511.6987233856, 'largest_symbol_pnl_share': 0.4350910587, 'largest_symbol_pnl_symbol': 'ETHUSDT_SWAP_OKX.GLOBAL', 'largest_symbol_trade_count_share': 0.4133858268, 'largest_symbol_trade_count_symbol': 'BTCUSDT_SWAP_OKX.GLOBAL', 'top_5pct_trade_pnl_contribution': -0.2605158554, 'top_5pct_trade_pnl': 914.8531968926, 'top_5pct_trade_count': 26, 'max_drawdown': 3537.5417752112, 'max_drawdown_pct': 3.4755241942}]
7. 收益是否集中在 top trades？top_5pct_trade_pnl_contribution=-0.2605
8. 是否允许进入 Phase 2？false
9. 是否允许修改正式策略？false
10. 是否允许 demo/live？false

## 9. Final Decision
- final_decision=postmortem
- continue_to_phase2=false
- strategy_development_allowed=false
- demo_live_allowed=false
